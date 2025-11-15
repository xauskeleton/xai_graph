# Tên file: inference_with_utils.py
# (Phiên bản chuẩn, sử dụng model.py và utils.py)

import torch
import os
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoTokenizer

# --- 1. IMPORT TỪ FILE CỦA BẠN ---
try:
    # Import class model
    from model import ImageCaptioningModel
    
    # Import các hàm tiện ích
    from utils import load_checkpoint, clean_sentence
    
except ImportError as e:
    print(f"❌ LỖI: Không tìm thấy file 'model.py' hoặc 'utils.py'.")
    print(f"   Chi tiết: {e}")
    print("   Hãy chắc chắn file này nằm cùng thư mục với 2 file đó.")
    input("Nhấn Enter để thoát...")
    exit()

# --- 2. CÀI ĐẶT ĐƯỜNG DẪN ---
# ⬇️ THAY ĐỔI ĐƯỜNG DẪN ẢNH VÀ CHECKPOINT TẠI ĐÂY ⬇️

IMAGE_PATH_TO_TEST = "view.jpg"  # <--- (1) SỬA ĐƯỜNG DẪN ẢNH NÀY
CHECKPOINT_PATH_TO_LOAD = "checkpoints/best_model.pth.tar" # <--- (2) SỬA NẾU CẦN

# --- ------------------------------------- ---

# Cấu hình chung
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_CAPTION_LENGTH = 50 


# --- 3. HÀM HỖ TRỢ ---

def preprocess_image(image_path):
    """ Tải và tiền xử lý ảnh đầu vào. """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Lỗi khi mở ảnh: {image_path}")
        print(f"   Chi tiết: {e}")
        return None
    
    # Chỉ cần ToTensor, model sẽ tự xử lý phần còn lại
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_tensor = transform(image)
    return image_tensor


# --- 4. HÀM MAIN ĐỂ CHẠY INFERENCE ---
def main_inference():
    """
    Hàm chính: Tải model, xử lý ảnh và sinh caption.
    """
    
    # --- A. Kiểm tra file tồn tại ---
    if not os.path.exists(IMAGE_PATH_TO_TEST):
        print(f"❌ LỖI: Không tìm thấy ảnh tại: {IMAGE_PATH_TO_TEST}")
        print("   Vui lòng cập nhật biến 'IMAGE_PATH_TO_TEST'.")
        return
    if not os.path.exists(CHECKPOINT_PATH_TO_LOAD):
        print(f"❌ LỖI: Không tìm thấy checkpoint tại: {CHECKPOINT_PATH_TO_LOAD}")
        print("   Vui lòng cập nhật biến 'CHECKPOINT_PATH_TO_LOAD'.")
        return

    print(f"--- 🚀 Bắt đầu Script Inference ---")
    print(f"Sử dụng thiết bị: {DEVICE}")

    # --- B. Tải Checkpoint (Chỉ lấy config) ---
    print(f"--- ⌛ Đang đọc config từ checkpoint: {CHECKPOINT_PATH_TO_LOAD} ---")
    try:
        checkpoint = torch.load(CHECKPOINT_PATH_TO_LOAD, map_location=DEVICE)
    except Exception as e:
        print(f"❌ LỖI: Không thể tải file checkpoint. File có thể bị hỏng.")
        print(f"   Chi tiết: {e}")
        return

    # --- C. LẤY CẤU HÌNH TỪ CHECKPOINT ---
    try:
        model_config = checkpoint['config']
        print("--- ⚙️ Đã tải cấu hình model từ checkpoint ---")
    except KeyError:
        print("❌ LỖI: Checkpoint này không chứa 'config'.")
        print("   File checkpoint này có thể đã cũ.")
        print("   Vui lòng huấn luyện lại model với file train.py mới nhất.")
        return

    # --- D. Tải Tokenizer ---
    print(f"--- 🤖 Đang tải Tokenizer ({model_config['tokenizer']}) ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_config['tokenizer'])
        model_config['vocab_size'] = tokenizer.vocab_size # Cập nhật vocab_size
    except Exception as e:
        print(f"❌ LỖI: Không thể tải Tokenizer '{model_config['tokenizer']}'.")
        print("   Vui lòng kiểm tra kết nối internet.")
        return

    # --- E. Khởi tạo Model ---
    print("\n--- 🛠️ Đang khởi tạo Model ---")
    model = ImageCaptioningModel(
        vocab_size=model_config['vocab_size'],
        embed_dim=model_config['embed_dim'],
        hidden_dim=model_config['hidden_dim'],
        num_objects=model_config['num_objects'],
        gnn_layers=model_config['gnn_layers'],
        gnn_heads=model_config['gnn_heads'],
        k_neighbors=model_config['k_neighbors']
    ).to(DEVICE)
    
    # --- F. Tải trọng số (SỬ DỤNG HÀM TỪ UTILS.PY) ---
    print("--- 💾 Đang tải trọng số (state_dict) ---")
    try:
        # Sử dụng hàm load_checkpoint từ utils.py
        # Giả định hàm load_checkpoint của bạn chỉ load state_dict
        # và trả về model đã load.
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # QUAN TRỌNG: Chuyển sang chế độ dự đoán
        print("✅ Tải model thành công!")
    except Exception as e:
        print("❌ LỖI: Không thể tải state_dict. Cấu hình model có thể không khớp.")
        print(f"   Chi tiết: {e}")
        return

    # --- G. Tải và xử lý ảnh ---
    print(f"\n--- 🏞️ Đang xử lý ảnh: {IMAGE_PATH_TO_TEST} ---")
    image_tensor = preprocess_image(IMAGE_PATH_TO_TEST)
    if image_tensor is None:
        return
    
    image_batch_list = [image_tensor.to(DEVICE)]
    
    # --- H. Sinh Caption ---
    print("--- ✍️ Đang sinh caption... ---")
    
    generated_ids = None
    with torch.no_grad():
        generated_ids = model.generate_caption(
            image_batch_list, 
            max_length=MAX_CAPTION_LENGTH
        )
    
    caption_ids = generated_ids[0] 
    
    # --- I. Decode và In kết quả ---
    caption_str_raw = tokenizer.decode(
        caption_ids, 
        skip_special_tokens=False
    )
    
    # SỬ DỤNG HÀM TỪ UTILS.PY
    final_caption = clean_sentence(caption_str_raw)
    
    print("\n" + "="*50)
    print(f"Ảnh đầu vào: {IMAGE_PATH_TO_TEST}")
    print(f"✍️ Caption dự đoán:")
    print(f"   {final_caption}")
    print("="*50)


# --- 5. CHẠY TRỰC TIẾP ---
if __name__ == "__main__":
    try:
        main_inference()
    except Exception as e:
        print(f"\n❌ LỖI CHƯƠNG TRÌNH KHÔNG MONG MUỐN: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n--- ✅ Đã chạy xong! ---")
    try:
        input("Nhấn Enter để thoát...")
    except EOFError:
        pass