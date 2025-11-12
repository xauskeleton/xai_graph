import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import roi_align
import torch_geometric as pyg
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, global_mean_pool, GCNConv 


class FasterRCNNFeatureExtractor(nn.Module):
    """Trích xuất đặc trưng từ ảnh bằng Faster R-CNN với spatial_scale đúng"""
    
    def __init__(self, num_objects=36, feature_dim=512):
        super().__init__()
        self.num_objects = num_objects
        self.feature_dim = feature_dim
        
        # Load Faster R-CNN pre-trained
        frcnn = fasterrcnn_resnet50_fpn(weights='DEFAULT')
        self.backbone = frcnn.backbone
        self.rpn = frcnn.rpn
        
        # Transform để chuẩn hóa input cho torchvision models
        self.transform = frcnn.transform
        
        # Giảm chiều feature (256 channels * 7 * 7 = 12544)
        self.fc = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, feature_dim)
        )
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.rpn.parameters():
            param.requires_grad = False
            
    def forward(self, images_list):
        """
        Args:
            images: [B, 3, H, W] - tensor đã normalize
        Returns:
            features: [B, num_objects, feature_dim]
            boxes: [B, num_objects, 4] - normalized to [0, 1]
        """
        batch_size = len(images_list)
        original_image_sizes = [img.shape[-2:] for img in images_list]
        
        with torch.no_grad():
            # Transform images (resize, normalize)
            images_transformed, _ = self.transform(
                [images_list[i] for i in range(batch_size)],
                None
            )
            
            # Extract features từ backbone
            features = self.backbone(images_transformed.tensors)
            
            # Tạo proposals
            proposals, _ = self.rpn(images_transformed, features)
        
        # Lấy feature map ở level '0' (P2 - stride 4)
        feature_map = features['0']  # [B, 256, H_feat, W_feat]
        
        # Tính spatial_scale: tỷ lệ giữa feature map và image gốc
        _, _, h_feat, w_feat = feature_map.shape
        _, _, h_img, w_img = images_transformed.tensors.shape
        spatial_scale = h_feat / h_img
        
        all_features = []
        all_boxes = []
        
        for i in range(batch_size):
            # Lấy top-k proposals và đảm bảo có đủ
            boxes = proposals[i][:self.num_objects]
            
            # Padding nếu không đủ proposals
            if len(boxes) < self.num_objects:
                padding = torch.zeros(
                    self.num_objects - len(boxes), 4,
                    device=boxes.device
                )
                boxes = torch.cat([boxes, padding], dim=0)
            
            # ROI Align với spatial_scale đúng
            rois = roi_align(
                feature_map[i:i+1],
                [boxes],
                output_size=(7, 7),
                spatial_scale=spatial_scale,
                aligned=True  # Tăng độ chính xác
            )
            
            # Flatten và project xuống feature_dim
            rois = rois.view(rois.size(0), -1)  # [num_objects, 256*7*7]
            feats = self.fc(rois)  # [num_objects, feature_dim]
            
            # Normalize boxes về [0, 1]
            h, w = original_image_sizes[i]
            normalized_boxes = boxes.clone()
            normalized_boxes[:, [0, 2]] /= w
            normalized_boxes[:, [1, 3]] /= h
            
            all_features.append(feats)
            all_boxes.append(normalized_boxes)
        
        features = torch.stack(all_features)  # [B, num_objects, feature_dim]
        boxes = torch.stack(all_boxes)  # [B, num_objects, 4]
        
        return features, boxes


class GraphConstructor(nn.Module):
    """Tạo graph data từ object features và boxes - VECTORIZED"""
    
    def __init__(self, feature_dim, spatial_dim=8):
        super().__init__()
        self.feature_dim = feature_dim
        self.spatial_dim = spatial_dim
        
        # Project spatial features
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim)
        )
    
    def compute_spatial_features_vectorized(self, boxes):
        """
        Tính spatial features cho tất cả cặp nodes - VECTORIZED
        Args:
            boxes: [B, N, 4] (x1, y1, x2, y2) normalized
        Returns:
            spatial_feats: [B, N, N, spatial_dim]
        """
        B, N, _ = boxes.shape
        
        # Tính centers và sizes
        centers = (boxes[..., :2] + boxes[..., 2:]) / 2  # [B, N, 2]
        sizes = boxes[..., 2:] - boxes[..., :2]  # [B, N, 2]
        
        # Expand để tính pairwise
        centers_i = centers.unsqueeze(2)  # [B, N, 1, 2]
        centers_j = centers.unsqueeze(1)  # [B, 1, N, 2]
        sizes_i = sizes.unsqueeze(2)  # [B, N, 1, 2]
        sizes_j = sizes.unsqueeze(1)  # [B, 1, N, 2]
        
        # Relative position
        rel_pos = centers_j - centers_i  # [B, N, N, 2]
        
        # Distance
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)  # [B, N, N, 1]
        
        # Angle
        angle = torch.atan2(rel_pos[..., 1:2], rel_pos[..., 0:1])  # [B, N, N, 1]
        
        # Relative size
        rel_size = sizes_j / (sizes_i + 1e-6)  # [B, N, N, 2]
        
        # IoU approximation (overlap)
        x1_max = torch.maximum(boxes.unsqueeze(2)[..., 0], boxes.unsqueeze(1)[..., 0])
        y1_max = torch.maximum(boxes.unsqueeze(2)[..., 1], boxes.unsqueeze(1)[..., 1])
        x2_min = torch.minimum(boxes.unsqueeze(2)[..., 2], boxes.unsqueeze(1)[..., 2])
        y2_min = torch.minimum(boxes.unsqueeze(2)[..., 3], boxes.unsqueeze(1)[..., 3])
        
        intersection = torch.clamp(x2_min - x1_max, min=0) * torch.clamp(y2_min - y1_max, min=0)
        area_i = (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])
        area_j = area_i.clone()
        union = area_i.unsqueeze(2) + area_j.unsqueeze(1) - intersection
        iou = (intersection / (union + 1e-6)).unsqueeze(-1)  # [B, N, N, 1]
        
        # Concatenate all features
        spatial_feats = torch.cat([
            rel_pos,      # 2
            dist,         # 1
            angle,        # 1
            rel_size,     # 2
            iou          # 1
        ], dim=-1)  # [B, N, N, 8]
        
        return spatial_feats
    
    def forward(self, node_features, boxes, k_neighbors=5):
        """
        Args:
            node_features: [B, N, feature_dim]
            boxes: [B, N, 4]
            k_neighbors: số neighbors cho mỗi node (k-NN graph)
        Returns:
            pyg_data_list: list of PyG Data objects
        """
        B, N, _ = node_features.shape
        
        # Compute spatial features (vectorized)
        spatial_feats = self.compute_spatial_features_vectorized(boxes)  # [B, N, N, 8]
        
        # Encode spatial features
        spatial_encoded = self.spatial_encoder(
            spatial_feats.view(B * N * N, -1)
        ).view(B, N, N, self.feature_dim)
        
        # Create PyG Data objects
        data_list = []
        for b in range(B):
            # Tính edge weights từ spatial features (dùng distance)
            distances = torch.norm(
                spatial_feats[b, :, :, :2], dim=-1
            )  # [N, N]
            
            # Create k-NN graph (fully connected nếu k >= N)
            if k_neighbors >= N - 1:
                # Fully connected graph
                edge_index = torch.combinations(
                    torch.arange(N, device=node_features.device), 
                    r=2, 
                    with_replacement=False
                ).t()
                # Add reverse edges
                edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            else:
                # k-NN graph
                _, topk_indices = torch.topk(
                    -distances, k=k_neighbors + 1, dim=1
                )  # +1 vì bao gồm chính nó
                topk_indices = topk_indices[:, 1:]  # Bỏ self-loop
                
                # Create edge_index
                src = torch.arange(N, device=node_features.device).unsqueeze(1).expand(-1, k_neighbors)
                edge_index = torch.stack([src.flatten(), topk_indices.flatten()], dim=0)
            
            # Get edge attributes
            src_idx, dst_idx = edge_index
            edge_attr = spatial_encoded[b, src_idx, dst_idx]  # [E, feature_dim]
            
            # Create PyG Data
            data = Data(
                x=node_features[b],  # [N, feature_dim]
                edge_index=edge_index,  # [2, E]
                edge_attr=edge_attr,  # [E, feature_dim]
                num_nodes=N
            )
            data_list.append(data)
        
        return data_list




class MotifGNN(nn.Module):
    """Graph Neural Network sử dụng PyG layers"""
    
    def __init__(self, feature_dim, hidden_dim, num_layers=3, heads=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.node_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # GAT layers with edge features
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    dropout=0.1,
                    edge_dim=feature_dim,  # Edge features
                    concat=True
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Global pooling
        self.global_pool = global_mean_pool
        
    def forward(self, data_list):
        """
        Args:
            data_list: list of PyG Data objects
        Returns:
            graph_features: [B, hidden_dim]
        """
        # Batch graphs
        batch = Batch.from_data_list(data_list)
        
        x = batch.x  # [total_nodes, feature_dim]
        edge_index = batch.edge_index  # [2, total_edges]
        edge_attr = batch.edge_attr  # [total_edges, feature_dim]
        batch_idx = batch.batch  # [total_nodes]
        
        # Encode nodes
        x = self.node_encoder(x)  # [total_nodes, hidden_dim]
        
        # GNN layers
        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index, edge_attr)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x = x + x_new  # Residual connection
        
        # Global pooling
        graph_features = self.global_pool(x, batch_idx)  # [B, hidden_dim]
        
        return graph_features


class LSTMDecoder(nn.Module):
    """LSTM decoder KHÔNG CÓ attention (Đơn giản)"""
    
    def __init__(self, embed_dim, hidden_dim, vocab_size, num_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout_emb = nn.Dropout(0.3)
        
        # --- THAY ĐỔI 1: XÓA `+ hidden_dim` ---
        # Input của LSTM giờ chỉ là word embedding
        self.lstm = nn.LSTM(
            embed_dim,  # <-- BỎ `+ hidden_dim` Ở ĐÂY
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, graph_features, captions=None, max_length=20):
        batch_size = graph_features.size(0)
        device = graph_features.device
        
        h0 = graph_features.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        
        if captions is not None:
            embedded = self.embedding(captions)
            embedded = self.dropout_emb(embedded)

            lstm_out, _ = self.lstm(embedded, (h0, c0)) # <-- DÙNG `embedded`
            
            lstm_out = self.dropout(lstm_out)
            outputs = self.fc(lstm_out)
        else:
            outputs = []
            input_token = torch.ones(batch_size, dtype=torch.long, device=device) # <START>
            hidden = (h0, c0)
            
            for _ in range(max_length):
                embedded = self.embedding(input_token).unsqueeze(1) # [B, 1, E]
                embedded = self.dropout_emb(embedded)
                
                lstm_out, hidden = self.lstm(embedded, hidden) # <-- DÙNG `embedded`
                
                logits = self.fc(lstm_out.squeeze(1))
                outputs.append(logits)
                input_token = logits.argmax(dim=1)
                
            outputs = torch.stack(outputs, dim=1)
            
        return outputs


class ImageCaptioningModel(nn.Module):
    """Model hoàn chỉnh: FRCNN + PyG GNN + LSTM"""
    
    def __init__(
        self, 
        vocab_size, 
        embed_dim=256, 
        hidden_dim=512, 
        num_objects=36,
        gnn_layers=3,
        gnn_heads=4,
        k_neighbors=5
    ):
        super().__init__()
        
        self.num_objects = num_objects
        self.k_neighbors = k_neighbors
        
        # Feature extractor
        self.feature_extractor = FasterRCNNFeatureExtractor(
            num_objects=num_objects,
            feature_dim=hidden_dim
        )
        
        # Graph constructor
        self.graph_constructor = GraphConstructor(
            feature_dim=hidden_dim,
            spatial_dim=8
        )
        
        # GNN
        self.gnn = MotifGNN(
            feature_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=gnn_layers,
            heads=gnn_heads
        )
        
        # Decoder
        self.decoder = LSTMDecoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            num_layers=2
        )
        
    def forward(self, images, captions=None):
        """
        Args:
            images: [B, 3, H, W]
            captions: [B, seq_len] (optional)
        Returns:
            outputs: [B, seq_len, vocab_size]
        """
        # Extract features
        features, boxes = self.feature_extractor(images)
        
        # Construct graphs
        data_list = self.graph_constructor(features, boxes, self.k_neighbors)
        
        # GNN forward
        graph_features = self.gnn(data_list)
        
        # Decode
        outputs = self.decoder(graph_features, captions)
        
        return outputs
    
    def generate_caption(self, images, max_length=20, beam_size=1):
        """
        Generate captions (inference mode)
        Args:
            images: [B, 3, H, W]
            max_length: max sequence length
            beam_size: beam search size (1 = greedy)
        Returns:
            captions: [B, max_length]
        """
        self.eval()
        with torch.no_grad():
            features, boxes = self.feature_extractor(images)
            data_list = self.graph_constructor(features, boxes, self.k_neighbors)
            graph_features = self.gnn(data_list)
            
            if beam_size == 1:
                outputs = self.decoder(graph_features, None, max_length)
                captions = outputs.argmax(dim=-1)
            else:
                # Implement beam search nếu cần
                raise NotImplementedError("Beam search chưa implement")
            
            return captions

