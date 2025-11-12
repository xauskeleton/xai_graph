# Tên file: src/evaluate.py

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tqdm import tqdm
import os

# Import từ các file của bạn
from data_loader import get_loader
from model import ImageCaptioningModel
from utils import load_checkpoint # Chỉ cần load

# --- CẤU HÌNH (CONFIG) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {DEVICE}")

# Đường dẫn
TEST_JSON_PATH = 'data/test_data.json'
TEST_IMAGE_DIR = 'data/test/' 

# Hyperparameters (PHẢI GIỐNG HỆT file train.py)
VOCAB_SIZE = 30522 
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_OBJECTS = 36
GNN_LAYERS = 3
GNN_HEADS = 4
K_NEIGHBORS = 10
BATCH_SIZE = 16

# Checkpoint
CHECKPOINT_FILE = "checkpoints/best_model.pth.tar"

# --- HÀM ĐÁNH GIÁ ---
def evaluate():
    
    print("--- 🤖 Đang tải Tokenizer (PhoBERT) ---")
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    VOCAB_SIZE = tokenizer.vocab_size
    PAD_TOKEN_ID = tokenizer.pad_token_id

    print("\n--- 🧪 Đang khởi tạo DataLoader cho TẬP KIỂM TRA (Test) ---")
    test_loader, test_dataset = get_loader(
        TEST_JSON_PATH, TEST_IMAGE_DIR, tokenizer, 
        BATCH_SIZE, shuffle=False, num_workers=4
    )
    print(f"✅ Tải thành công! {len(test_dataset)} mẫu kiểm tra.")

    # --- Khởi tạo Model, Loss ---
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
    
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
    
    # --- TẢI CHECKPOINT ---
    if not os.path.exists(CHECKPOINT_FILE):
        print(f"❌ LỖI: Không tìm thấy checkpoint tại '{CHECKPOINT_FILE}'")
        print("Vui lòng chạy train.py trước!")
        return
        
    print(f"--- 💾 Đang tải checkpoint từ {CHECKPOINT_FILE} ---")
    # Chúng ta không cần load optimizer vì không train
    load_checkpoint(model, CHECKPOINT_FILE, device=DEVICE)

    # --- Bắt đầu vòng lặp Đánh giá ---
    print("\n--- 📊 Bắt đầu Đánh giá Loss trên tập Test ---")
    model.eval() # Rất quan trọng!
    
    total_test_loss = 0.0
    
    with torch.no_grad(): # Tắt gradient
        loop = tqdm(test_loader, leave=True)
        for images, token_batch in loop:
            images = images.to(DEVICE)
            input_ids = token_batch['input_ids'].to(DEVICE)
            
            captions_input = input_ids[:, :-1]
            captions_target = input_ids[:, 1:]
            
            outputs = model(images, captions_input)
            loss = criterion(
                outputs.reshape(-1, VOCAB_SIZE),
                captions_target.reshape(-1)
            )
            total_test_loss += loss.item()
            loop.set_postfix(test_loss=loss.item())

    avg_test_loss = total_test_loss / len(test_loader)
    
    print("\n" + "=" * 50)
    print("--- KẾT QUẢ ĐÁNH GIÁ HOÀN TẤT ---")
    print(f"  Average Test Loss: {avg_test_loss:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    evaluate()