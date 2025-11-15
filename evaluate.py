# Tên file: src/evaluate.py
# (Phiên bản chạy trực tiếp, không cần terminal)

import torch
import torch.nn as nn
from transformers import AutoTokenizer
import os
import json
from tqdm import tqdm
# import argparse # <-- KHÔNG CẦN NỮA

# Import từ các file của bạn
from data_loader import get_loader
from model import ImageCaptioningModel
from utils import load_checkpoint, clean_sentence, get_model_summary

# Import thư viện metrics
import aac_metrics
os.environ['JAVA_TOOL_OPTIONS'] = '-Xmx4096M'

# --- 1. CẤU HÌNH (PHẢI GIỐNG HỆT train.py) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {DEVICE}")

# Đường dẫn (Chỉ cần tập TEST)
TEST_JSON_PATH = 'data/test_data.json' 
TEST_IMAGE_DIR = 'data/test/' 

# Hyperparameters (SAO CHÉP TỪ FILE train.py CỦA BẠN)
VOCAB_SIZE = 30522 
EMBED_DIM = 256
HIDDEN_DIM = 1024       
NUM_OBJECTS = 48        
GNN_LAYERS = 4          
GNN_HEADS = 4
K_NEIGHBORS = 15   
BATCH_SIZE = 2        

# --- 2. HÀM CHẠY (MAIN) ---
def evaluate():
    
    # --- 💥 THAY ĐỔI Ở ĐÂY 💥 ---
    # Tự điền đường dẫn đến checkpoint của bạn
    CHECKPOINT_FILE = "checkpoints/best_model.pth.tar" # <--- SỬA ĐƯỜNG DẪN NÀY
    
    # --- KẾT THÚC THAY ĐỔI ---

    if not os.path.exists(CHECKPOINT_FILE):
        print(f"❌ LỖI: Không tìm thấy checkpoint tại: {CHECKPOINT_FILE}")
        return

    # --- 1. Tải dependencies cho Java (SPICE/METEOR) ---
    try:
        print("\n--- 📦 Đang kiểm tra/tải dependencies cho aac-metrics (Java)... ---")
        aac_metrics.download_java_dependencies(force=False)
        print("✅ Dependencies OK!")
    except Exception as e:
        print(f"⚠️ LỖI khi tải dependencies cho Java: {e}")
        print("   -> Metric SPICE và METEOR có thể sẽ không hoạt động.")
        print("   -> (Nếu ở trên server, hãy thử chạy: sudo apt install default-jdk)")

    # --- 2. Tải Tokenizer ---
    print("--- 🤖 Đang tải Tokenizer (PhoBERT) ---")
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    VOCAB_SIZE = tokenizer.vocab_size 

    # --- 3. Tải DataLoaders (Chỉ cần TEST) ---
    print("\n--- 🚀 Đang khởi tạo Test DataLoader ---")
    test_loader, _ = get_loader(
        TEST_JSON_PATH, TEST_IMAGE_DIR, tokenizer, 
        BATCH_SIZE, shuffle=False, num_workers=4
    )
    print("✅ Tải Test Loader thành công!")

    # --- 4. Khởi tạo Model (VỚI CẤU HÌNH ĐÚNG) ---
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
    
    # --- 5. Tải Checkpoint ---
    print(f"\n--- 💾 Đang tải checkpoint từ: {CHECKPOINT_FILE} ---")
    try:
        model = load_checkpoint(model, CHECKPOINT_FILE, device=DEVICE)
        print("✅ Tải checkpoint thành công!")
    except Exception as e:
        print(f"❌ LỖI KHI TẢI CHECKPOINT: {e}")
        print("   Lỗi này thường xảy ra khi cấu hình model (evaluate.py) không khớp với model đã lưu (train.py).")
        return

    # --- 6. Chạy Đánh giá ---
    model.eval() # Bật chế độ EVAL
    all_predictions = []
    all_references = []

    print("\n--- 🧐 Đang chạy Đánh giá trên toàn bộ tập Test ---")
    loop_test = tqdm(test_loader, leave=True, desc="Testing")

    with torch.no_grad(): # Không cần tính gradient
        for images_list, token_batch in loop_test:
            images = [img.to(DEVICE) for img in images_list]
            input_ids = token_batch['input_ids'].to(DEVICE)

            generated_ids = model.generate_caption(images, max_length=50)

            pred_sentences_batch = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=False
            )
            ref_sentences_batch = tokenizer.batch_decode(
                input_ids,
                skip_special_tokens=False
            )

            for pred_str, ref_str in zip(pred_sentences_batch, ref_sentences_batch):
                all_predictions.append(clean_sentence(pred_str))
                all_references.append([clean_sentence(ref_str)])

    print("✅ Đã chạy xong! Bắt đầu tính điểm...")

    # --- 7. TÍNH TOÁN VÀ IN ĐIỂM SỐ ---
    print("\n--- 📊 Đang tính điểm (BLEU, METEOR, ROUGE, CIDEr-D, SPICE)... ---")
    
    try:
        all_scores = aac_metrics.evaluate(all_predictions, all_references)
        
        # --- 8. IN KẾT QUẢ ---
        print("\n" + "="*50)
        print(f"KẾT QUẢ ĐÁNH GIÁ CHO CHECKPOINT: {CHECKPOINT_FILE}")
        print("="*50)
        for metric, score in all_scores.items():
            print(f"   {metric:<10}: {score:.2f}")
        print("="*50)

    except Exception as e:
        print(f"\n❌ LỖI KHI TÍNH TOÁN METRICS (aac-metrics): {e}")
        print("   Lỗi này thường xảy ra nếu Java (JDK) chưa được cài đặt đúng cách.")
        print("   Hãy thử chạy: sudo apt install default-jdk")


if __name__ == "__main__":
    try:
        evaluate()
    except Exception as e:
        print(f"\n❌ LỖI CHƯA XỬ LÝ: {e}")