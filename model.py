import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import roi_align
import torch_geometric as pyg
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, global_mean_pool


class FasterRCNNFeatureExtractor(nn.Module):
    """Trích xuất đặc trưng từ Faster R-CNN"""

    def __init__(self, num_objects=36, feature_dim=512):
        super().__init__()
        self.num_objects = num_objects
        self.feature_dim = feature_dim

        frcnn = fasterrcnn_resnet50_fpn(weights='DEFAULT')
        self.backbone = frcnn.backbone
        self.rpn = frcnn.rpn
        self.transform = frcnn.transform

        self.fc = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, feature_dim)
        )

        # Đóng băng tham số
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.rpn.parameters():
            param.requires_grad = False

        self.backbone.eval()
        self.rpn.eval()

    def train(self, mode: bool = True):
        """Giữ backbone và rpn luôn ở eval mode"""
        self.fc.train(mode)
        self.backbone.eval()
        self.rpn.eval()
        return self

    def forward(self, images_list):
        batch_size = len(images_list)
        original_image_sizes = [img.shape[-2:] for img in images_list]

        with torch.no_grad():
            images_transformed, _ = self.transform(images_list, None)
            features = self.backbone(images_transformed.tensors)
            proposals, _ = self.rpn(images_transformed, features)

        feature_map = features['0']
        _, _, h_feat, w_feat = feature_map.shape
        _, _, h_img_trans, w_img_trans = images_transformed.tensors.shape
        spatial_scale = h_feat / h_img_trans

        all_features = []
        all_boxes = []

        for i in range(batch_size):
            boxes = proposals[i][:self.num_objects]

            if len(boxes) < self.num_objects:
                padding = torch.zeros(
                    self.num_objects - len(boxes), 4,
                    device=boxes.device
                )
                boxes = torch.cat([boxes, padding], dim=0)

            current_feature_map = feature_map[i:i + 1]

            rois = roi_align(
                current_feature_map,
                [boxes],
                output_size=(7, 7),
                spatial_scale=spatial_scale,
                aligned=True
            )

            rois = rois.view(rois.size(0), -1)
            feats = self.fc(rois)

            normalized_boxes = boxes.clone()
            normalized_boxes[:, [0, 2]] /= w_img_trans
            normalized_boxes[:, [1, 3]] /= h_img_trans
            normalized_boxes = normalized_boxes.clamp(0, 1)

            all_features.append(feats)
            all_boxes.append(normalized_boxes)

        features = torch.stack(all_features)
        boxes = torch.stack(all_boxes)

        return features, boxes


class GraphConstructor(nn.Module):
    """Tạo graph data từ object features và boxes"""

    def __init__(self, feature_dim, spatial_dim=8):
        super().__init__()
        self.feature_dim = feature_dim
        self.spatial_dim = spatial_dim

        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim)
        )

    def compute_spatial_features_vectorized(self, boxes):
        B, N, _ = boxes.shape

        centers = (boxes[..., :2] + boxes[..., 2:]) / 2
        sizes = boxes[..., 2:] - boxes[..., :2]

        centers_i = centers.unsqueeze(2)
        centers_j = centers.unsqueeze(1)
        sizes_i = sizes.unsqueeze(2)
        sizes_j = sizes.unsqueeze(1)

        rel_pos = centers_j - centers_i
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        angle = torch.atan2(rel_pos[..., 1:2], rel_pos[..., 0:1])
        rel_size = sizes_j / (sizes_i + 1e-6)

        x1_max = torch.maximum(boxes.unsqueeze(2)[..., 0], boxes.unsqueeze(1)[..., 0])
        y1_max = torch.maximum(boxes.unsqueeze(2)[..., 1], boxes.unsqueeze(1)[..., 1])
        x2_min = torch.minimum(boxes.unsqueeze(2)[..., 2], boxes.unsqueeze(1)[..., 2])
        y2_min = torch.minimum(boxes.unsqueeze(2)[..., 3], boxes.unsqueeze(1)[..., 3])

        intersection = torch.clamp(x2_min - x1_max, min=0) * torch.clamp(y2_min - y1_max, min=0)
        area_i = (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])
        area_j = area_i.clone()
        union = area_i.unsqueeze(2) + area_j.unsqueeze(1) - intersection
        iou = (intersection / (union + 1e-6)).unsqueeze(-1)

        aspect_i = sizes_i[..., 0] / (sizes_i[..., 1] + 1e-6)
        aspect_j = sizes_j[..., 0] / (sizes_j[..., 1] + 1e-6)
        aspect_diff = (aspect_j / (aspect_i + 1e-6)).unsqueeze(-1)

        spatial_feats = torch.cat([
            rel_pos, dist, angle, rel_size, iou, aspect_diff
        ], dim=-1)

        return spatial_feats

    def forward(self, node_features, boxes, k_neighbors=5):
        B, N, _ = node_features.shape

        spatial_feats = self.compute_spatial_features_vectorized(boxes)
        spatial_encoded = self.spatial_encoder(
            spatial_feats.view(B * N * N, -1)
        ).view(B, N, N, self.feature_dim)

        data_list = []
        for b in range(B):
            distances = torch.norm(spatial_feats[b, :, :, :2], dim=-1)

            if k_neighbors >= N - 1:
                edge_index = torch.combinations(
                    torch.arange(N, device=node_features.device),
                    r=2,
                    with_replacement=False
                ).t()
                edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            else:
                _, topk_indices = torch.topk(-distances, k=k_neighbors + 1, dim=1)
                topk_indices = topk_indices[:, 1:]

                src = torch.arange(N, device=node_features.device).unsqueeze(1).expand(-1, k_neighbors)
                edge_index = torch.stack([src.flatten(), topk_indices.flatten()], dim=0)

            src_idx, dst_idx = edge_index
            edge_attr = spatial_encoded[b, src_idx, dst_idx]

            data = Data(
                x=node_features[b],
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=N
            )
            data_list.append(data)

        return data_list


class MotifGNN(nn.Module):
    """Graph Neural Network với GAT layers"""

    def __init__(self, feature_dim, hidden_dim, num_layers=3, heads=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.node_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15)
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    dropout=0.15,
                    edge_dim=feature_dim,
                    concat=True
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.global_pool = global_mean_pool

    def forward(self, data_list):
        batch = Batch.from_data_list(data_list)

        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        batch_idx = batch.batch

        x = self.node_encoder(x)

        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index, edge_attr)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x = x + x_new

        graph_features = self.global_pool(x, batch_idx)

        return graph_features, x, batch_idx  # Trả về cả node features để dùng attention


class AttentionLSTMDecoder(nn.Module):
    """LSTM decoder với Attention Mechanism"""

    def __init__(self, embed_dim, hidden_dim, vocab_size, num_layers=2, dropout=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout_emb = nn.Dropout(dropout)

        # LSTM nhận embedding + context vector
        self.lstm = nn.LSTM(
            embed_dim + hidden_dim,  # Input = word embedding + attended context
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism
        self.attention = nn.Linear(hidden_dim, hidden_dim)
        self.context_attention = nn.Linear(hidden_dim, hidden_dim)
        self.attention_combine = nn.Linear(hidden_dim, 1)

        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def compute_attention(self, hidden_state, node_features, batch_idx):
        """
        Tính attention weights trên các object nodes
        Args:
            hidden_state: [B, hidden_dim] - LSTM hidden state
            node_features: [total_nodes, hidden_dim] - All node features
            batch_idx: [total_nodes] - Batch index cho mỗi node
        Returns:
            context: [B, hidden_dim] - Weighted sum of node features
        """
        batch_size = hidden_state.size(0)
        device = hidden_state.device

        # Project hidden state và node features
        hidden_proj = self.attention(hidden_state)  # [B, hidden_dim]
        node_proj = self.context_attention(node_features)  # [total_nodes, hidden_dim]

        # Compute attention scores cho từng batch
        context_list = []
        for b in range(batch_size):
            # Lấy nodes thuộc batch b
            mask = (batch_idx == b)
            nodes_b = node_proj[mask]  # [N_b, hidden_dim]

            # Attention score
            scores = torch.matmul(nodes_b, hidden_proj[b].unsqueeze(-1))  # [N_b, 1]
            weights = F.softmax(scores, dim=0)  # [N_b, 1]

            # Weighted sum
            context_b = (nodes_b * weights).sum(dim=0)  # [hidden_dim]
            context_list.append(context_b)

        context = torch.stack(context_list, dim=0)  # [B, hidden_dim]
        return context

    def forward(self, graph_features, node_features, batch_idx, captions=None, max_length=20):
        """
        Args:
            graph_features: [B, hidden_dim] - Global graph representation
            node_features: [total_nodes, hidden_dim] - Node features cho attention
            batch_idx: [total_nodes] - Batch index
            captions: [B, seq_len] - Ground truth captions (training)
            max_length: int - Max sequence length (inference)
        """
        batch_size = graph_features.size(0)
        device = graph_features.device

        # Initialize hidden state
        h0 = graph_features.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)

        if captions is not None:
            # Teacher forcing mode
            seq_len = captions.size(1)
            outputs = []

            hidden = (h0, c0)

            for t in range(seq_len):
                # Get current word embedding
                word_embed = self.embedding(captions[:, t])  # [B, embed_dim]
                word_embed = self.dropout_emb(word_embed)

                # Compute attention context
                current_hidden = hidden[0][-1]  # [B, hidden_dim] - Last layer hidden
                context = self.compute_attention(current_hidden, node_features, batch_idx)

                # Concatenate word embedding with context
                lstm_input = torch.cat([word_embed, context], dim=-1)  # [B, embed_dim + hidden_dim]
                lstm_input = lstm_input.unsqueeze(1)  # [B, 1, embed_dim + hidden_dim]

                # LSTM step
                lstm_out, hidden = self.lstm(lstm_input, hidden)  # lstm_out: [B, 1, hidden_dim]

                # Project to vocabulary
                lstm_out = self.dropout(lstm_out.squeeze(1))  # [B, hidden_dim]
                output = self.fc(lstm_out)  # [B, vocab_size]

                outputs.append(output)

            outputs = torch.stack(outputs, dim=1)  # [B, seq_len, vocab_size]

        else:
            # Inference mode (greedy decoding)
            outputs = []
            input_token = torch.ones(batch_size, dtype=torch.long, device=device)  # <START>
            hidden = (h0, c0)

            for t in range(max_length):
                # Get word embedding
                word_embed = self.embedding(input_token)  # [B, embed_dim]
                word_embed = self.dropout_emb(word_embed)

                # Compute attention context
                current_hidden = hidden[0][-1]
                context = self.compute_attention(current_hidden, node_features, batch_idx)

                # Concatenate
                lstm_input = torch.cat([word_embed, context], dim=-1)
                lstm_input = lstm_input.unsqueeze(1)

                # LSTM step
                lstm_out, hidden = self.lstm(lstm_input, hidden)

                # Project
                lstm_out = self.dropout(lstm_out.squeeze(1))
                logits = self.fc(lstm_out)

                outputs.append(logits)

                # Next input token
                input_token = logits.argmax(dim=1)

            outputs = torch.stack(outputs, dim=1)  # [B, max_length, vocab_size]

        return outputs


class ImageCaptioningModel(nn.Module):
    """Model hoàn chỉnh: FRCNN + GNN + Attention LSTM"""

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

        # Decoder với attention
        self.decoder = AttentionLSTMDecoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            num_layers=2,
            dropout=0.3
        )

    def forward(self, images, captions=None):
        """
        Args:
            images: list of [3, H, W] tensors
            captions: [B, seq_len] (optional)
        Returns:
            outputs: [B, seq_len, vocab_size]
        """
        # Extract features
        features, boxes = self.feature_extractor(images)

        # Construct graphs
        data_list = self.graph_constructor(features, boxes, self.k_neighbors)

        # GNN forward
        graph_features, node_features, batch_idx = self.gnn(data_list)

        # Decode với attention
        outputs = self.decoder(graph_features, node_features, batch_idx, captions)

        return outputs

    def generate_caption(self, images, max_length=20):
        """
        Generate captions (inference mode)
        Args:
            images: list of [3, H, W] tensors
            max_length: max sequence length
        Returns:
            captions: [B, max_length]
        """
        self.eval()
        with torch.no_grad():
            features, boxes = self.feature_extractor(images)
            data_list = self.graph_constructor(features, boxes, self.k_neighbors)
            graph_features, node_features, batch_idx = self.gnn(data_list)

            outputs = self.decoder(graph_features, node_features, batch_idx, None, max_length)
            captions = outputs.argmax(dim=-1)

            return captions