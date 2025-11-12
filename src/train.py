# Tên file: src/train.py
# (ĐÃ SỬA LỖI `AttributeError: 'list' object has no attribute 'to'`)

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
import os
from tqdm import tqdm # Thư viện để xem tiến độ (pip install tqdm)

# Import từ các file của bạn
from data_loader import get_loader
from model import ImageCaptioningModel
from utils import save_checkpoint, load_checkpoint, get_model_summary

# --- 1. CẤU HÌNH (CONFIG) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {DEVICE}")

# Đường dẫn (giống trong data_loader.py)
TRAIN_JSON_PATH = 'data/train_data.json' 
TRAIN_IMAGE_DIR = 'data/train/'
TEST_JSON_PATH = 'data/test_data.json'
TEST_IMAGE_DIR = 'data/test/' 

# Hyperparameters
VOCAB_SIZE = 30522 # Sẽ được cập nhật từ tokenizer
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_OBJECTS = 36
GNN_LAYERS = 3
GNN_HEADS = 4
K_NEIGHBORS = 10
NUM_EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 2e-5

# Checkpoint
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "best_model.pth.tar")
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# --- 2. HÀM CHẠY (MAIN) ---
def main():
    
    # --- Tải Tokenizer ---
    print("--- 🤖 Đang tải Tokenizer (PhoBERT) ---")
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    VOCAB_SIZE = tokenizer.vocab_size # Lấy vocab size chính xác
    PAD_TOKEN_ID = tokenizer.pad_token_id

    # --- Tải DataLoaders ---
    print("\n--- 🚀 Đang khởi tạo DataLoaders ---")
    train_loader, _ = get_loader(
        TRAIN_JSON_PATH, TRAIN_IMAGE_DIR, tokenizer, 
        BATCH_SIZE, shuffle=True, num_workers=4
    )
    test_loader, _ = get_loader(
        TEST_JSON_PATH, TEST_IMAGE_DIR, tokenizer, 
        BATCH_SIZE, shuffle=False, num_workers=4
    )
    print("✅ Tải DataLoaders thành công!")

    # --- Khởi tạo Model, Loss, Optimizer ---
    print("\n--- 🛠️ Đang khởi tạo Model ---")
    model = ImageCaptioningModel(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_objects=NUM_OBJECTS,
        gnn_layers=GNN_LAYERS,
        gnn_heads=GNN_HEADS,
        k_neighbors=K_NEIGHBORS
    ).to(DEVICE)
    
    get_model_summary(model) # In tóm tắt model
    
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=LEARNING_RATE,
        weight_decay=1e-5
    )
    
    best_test_loss = float('inf')


    # --- 3. VÒNG LẶP HUẤN LUYỆN ---
    print(f"\n--- 🔥 BẮT ĐẦU HUẤN LUYỆN VỚI {NUM_EPOCHS} EPOCHS ---")
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch [{epoch+1}/{NUM_EPOCHS}] ---")
        
        # --- Giai đoạn TRAIN ---
        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, leave=True)
        # Sửa `images` thành `images_list` để rõ ràng
        for i, (images_list, token_batch) in enumerate(loop): 
            
            # --- SỬA DÒNG NÀY ---
            # Chuyển từng ảnh trong list sang DEVICE
            images = [img.to(DEVICE) for img in images_list]
            # --- KẾT THÚC SỬA ---
            
            input_ids = token_batch['input_ids'].to(DEVICE)
            
            captions_input = input_ids[:, :-1]
            captions_target = input_ids[:, 1:]
            
            # 1. Forward pass
            outputs = model(images, captions_input) # `images` giờ là 1 list
            
            # 2. Tính Loss
            loss = criterion(
                outputs.reshape(-1, VOCAB_SIZE),
                captions_target.reshape(-1)
            )
            
            # 3. Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
            loop.set_postfix(train_loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)

        # --- Giai đoạn EVALUATE (Test) ---
        model.eval()
        test_loss = 0.0
        
        with torch.no_grad():
            # Sửa `images` thành `images_list`
            for images_list, token_batch in test_loader:
                
                # --- SỬA DÒNG NÀY ---
                images = [img.to(DEVICE) for img in images_list]
                # --- KẾT THÚC SỬA ---
                
                input_ids = token_batch['input_ids'].to(DEVICE)
                
                captions_input = input_ids[:, :-1]
                captions_target = input_ids[:, 1:]
                
                outputs = model(images, captions_input) # `images` là 1 list
                loss = criterion(
                    outputs.reshape(-1, VOCAB_SIZE),
                    captions_target.reshape(-1)
                )
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        
        # --- 4. IN KẾT QUẢ & LƯU MODEL ---
        print(f"--- ⭐️ KẾT THÚC EPOCH {epoch+1} ---")
        print(f"   Loss Huấn luyện (Train Loss): {avg_train_loss:.4f}")
        print(f"   Loss Kiểm tra (Test Loss):  {avg_test_loss:.4f}")
        
        if avg_test_loss < best_test_loss:
            print(f"   (Test Loss giảm từ {best_test_loss:.4f} xuống {avg_test_loss:.4f}. Đang lưu model...)")
            best_test_loss = avg_test_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_test_loss
            }
            save_checkpoint(checkpoint, CHECKPOINT_FILE)
        else:
            print(f"   (Test Loss không cải thiện so với {best_test_loss:.4f})")

    print("\n--- ✅ Huấn luyện hoàn tất! ---")

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n❌ LỖI KHỞI TẠO: KHÔNG TÌM THẤY FILE.")
        print(f"   {e}")
        print("   Hãy chắc chắn rằng đường dẫn trong file train.py là ĐÚNG.")
    except Exception as e:
        print(f"\n❌ LỖI CHƯƠNG TRÌNH: {e}")