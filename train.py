# Tên file: src/train.py
# (PHIÊN BẢN NÂNG CẤP: Thêm Early Stopping, Lưu Config đầy đủ và Training History)

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
import os
import json
from datetime import datetime
from tqdm import tqdm

# Import từ các file của bạn
from data_loader import get_loader
from model import ImageCaptioningModel
from utils import save_checkpoint, load_checkpoint, get_model_summary, calculate_bleu_scores, clean_sentence

# --- 1. CẤU HÌNH (CONFIG) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {DEVICE}")

# Đường dẫn
TRAIN_JSON_PATH = 'data/train_data.json'
TRAIN_IMAGE_DIR = 'data/train/'
TEST_JSON_PATH = 'data/test_data.json'
TEST_IMAGE_DIR = 'data/test/'

# Hyperparameters
VOCAB_SIZE = 30522
EMBED_DIM = 256
HIDDEN_DIM = 1024
NUM_OBJECTS = 48
GNN_LAYERS = 4
GNN_HEADS = 4
K_NEIGHBORS = 15
NUM_EPOCHS = 30
BATCH_SIZE =64
LEARNING_RATE = 5e-5

# 💥 Early Stopping Configuration
EARLY_STOPPING_PATIENCE = 5  # Dừng sau 5 epochs không cải thiện
MIN_DELTA = 1e-4  # Ngưỡng cải thiện tối thiểu

# Checkpoint
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"  # 💥 THÊM MỚI: Thư mục lưu logs
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "best_model.pth.tar")

# Tạo thư mục nếu chưa có
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
if not os.path.exists(LOG_DIR):  # 💥 THÊM MỚI
    os.makedirs(LOG_DIR)


# --- 2. HÀM CHẠY (MAIN) ---
def main():
    # --- Tải Tokenizer ---
    print("--- 🤖 Đang tải Tokenizer (PhoBERT) ---")
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    VOCAB_SIZE = tokenizer.vocab_size  # Lấy vocab size chính xác
    PAD_TOKEN_ID = tokenizer.pad_token_id

    # 💥 THÊM MỚI: Tạo dict cấu hình đầy đủ với timestamp
    model_config = {
        'vocab_size': VOCAB_SIZE,
        'embed_dim': EMBED_DIM,
        'hidden_dim': HIDDEN_DIM,
        'num_objects': NUM_OBJECTS,
        'gnn_layers': GNN_LAYERS,
        'gnn_heads': GNN_HEADS,
        'k_neighbors': K_NEIGHBORS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'num_epochs': NUM_EPOCHS,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'min_delta': MIN_DELTA,
        'tokenizer': 'vinai/phobert-base',
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'device': str(DEVICE)
    }

    print(f"--- ⚙️ Đang huấn luyện với cấu hình: ---")
    for key, value in model_config.items():
        print(f"   {key}: {value}")

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
        vocab_size=model_config['vocab_size'],
        embed_dim=model_config['embed_dim'],
        hidden_dim=model_config['hidden_dim'],
        num_objects=model_config['num_objects'],
        gnn_layers=model_config['gnn_layers'],
        gnn_heads=model_config['gnn_heads'],
        k_neighbors=model_config['k_neighbors']
    ).to(DEVICE)

    get_model_summary(model)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-5
    )

    best_test_loss = float('inf')
    best_bleu_4 = 0.0  # 💥 THÊM: Track best BLEU-4

    # 💥 THÊM MỚI: Early Stopping variables
    epochs_no_improve = 0
    best_epoch = 0

    # 💥 THÊM MỚI: Training history để lưu vào log
    training_history = {
        'config': model_config,
        'epochs': [],
        'best_epoch': 0,
        'best_test_loss': float('inf'),
        'best_bleu_4': 0.0
    }

    # --- 3. VÒNG LẶP HUẤN LUYỆN ---
    print(f"\n--- 🔥 BẮT ĐẦU HUẤN LUYỆN VỚI {NUM_EPOCHS} EPOCHS ---")

    for epoch in range(NUM_EPOCHS):
        print(f"\n{'=' * 80}")
        print(f"--- Epoch [{epoch + 1}/{NUM_EPOCHS}] ---")
        print(f"{'=' * 80}")

        # --- Giai đoạn TRAIN ---
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, leave=True)
        for i, (images_list, token_batch) in enumerate(loop):
            images = [img.to(DEVICE) for img in images_list]
            input_ids = token_batch['input_ids'].to(DEVICE)
            captions_input = input_ids[:, :-1]
            captions_target = input_ids[:, 1:]

            outputs = model(images, captions_input)
            loss = criterion(outputs.reshape(-1, VOCAB_SIZE), captions_target.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_description(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
            loop.set_postfix(train_loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # --- Giai đoạn EVALUATE (Test) ---
        model.eval()
        test_loss = 0.0

        all_predictions = []
        all_references = []

        print("\n--- 🧐 Đang chạy Đánh giá (Evaluate) trên tập Test ---")

        loop_test = tqdm(test_loader, leave=True, desc="Testing")

        with torch.no_grad():
            for images_list, token_batch in loop_test:
                images = [img.to(DEVICE) for img in images_list]
                input_ids = token_batch['input_ids'].to(DEVICE)

                captions_input = input_ids[:, :-1]
                captions_target = input_ids[:, 1:]

                # Tính test loss
                outputs_logits = model(images, captions_input)
                loss = criterion(
                    outputs_logits.reshape(-1, VOCAB_SIZE),
                    captions_target.reshape(-1)
                )
                test_loss += loss.item()

                # Sinh caption để tính BLEU
                generated_ids = model.generate_caption(images, max_length=50)

                # Decode
                pred_sentences_batch = tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=False
                )
                ref_sentences_batch = tokenizer.batch_decode(
                    input_ids,
                    skip_special_tokens=False
                )

                # Dọn dẹp và lưu
                for pred_str, ref_str in zip(pred_sentences_batch, ref_sentences_batch):
                    cleaned_pred = clean_sentence(pred_str)
                    cleaned_ref = clean_sentence(ref_str)
                    all_predictions.append(cleaned_pred)
                    all_references.append([cleaned_ref])

        avg_test_loss = test_loss / len(test_loader)

        # Tính BLEU scores
        bleu_scores = calculate_bleu_scores(all_predictions, all_references)

        # --- IN KẾT QUẢ ---
        print(f"\n{'=' * 80}")
        print(f"--- ⭐️ KẾT THÚC EPOCH {epoch + 1} ---")
        print(f"{'=' * 80}")
        print(f"   📈 Loss Huấn luyện (Train Loss): {avg_train_loss:.4f}")
        print(f"   📉 Loss Kiểm tra (Test Loss):    {avg_test_loss:.4f}")
        print("   --- 📊 ĐIỂM BLEU (trên tập Test) ---")
        print(f"       BLEU-1: {bleu_scores['BLEU-1']:.2f}")
        print(f"       BLEU-2: {bleu_scores['BLEU-2']:.2f}")
        print(f"       BLEU-3: {bleu_scores['BLEU-3']:.2f}")
        print(f"       BLEU-4: {bleu_scores['BLEU-4']:.2f}  ⭐ (Chỉ số quan trọng)")

        # 💥 THÊM: Lưu epoch info vào history
        epoch_info = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss,
            'bleu_1': bleu_scores['BLEU-1'],
            'bleu_2': bleu_scores['BLEU-2'],
            'bleu_3': bleu_scores['BLEU-3'],
            'bleu_4': bleu_scores['BLEU-4']
        }
        training_history['epochs'].append(epoch_info)

        # --- LOGIC EARLY STOPPING ---
        # Kiểm tra xem loss có cải thiện không
        if avg_test_loss < (best_test_loss - MIN_DELTA):
            # Loss cải thiện đáng kể
            print(f"\n   ✅ Test Loss cải thiện: {best_test_loss:.4f} → {avg_test_loss:.4f}")
            print(f"   💾 Đang lưu model tốt nhất...")

            best_test_loss = avg_test_loss
            best_bleu_4 = bleu_scores['BLEU-4']
            best_epoch = epoch + 1
            epochs_no_improve = 0  # Reset counter

            # 💥 THÊM: Lưu checkpoint với config đầy đủ
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss,
                'bleu_scores': bleu_scores,
                'config': model_config,  # 💥 Lưu config
                'training_history': training_history  # 💥 Lưu history
            }
            save_checkpoint(checkpoint, CHECKPOINT_FILE)

            # 💥 Update training history
            training_history['best_epoch'] = best_epoch
            training_history['best_test_loss'] = best_test_loss
            training_history['best_bleu_4'] = best_bleu_4

        else:
            # Loss không cải thiện
            epochs_no_improve += 1
            print(f"\n   ⚠️ Test Loss không cải thiện (Best: {best_test_loss:.4f})")
            print(f"   📊 Early Stopping Counter: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")

            # 💥 KIỂM TRA EARLY STOPPING
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\n{'=' * 80}")
                print(f"🛑 EARLY STOPPING được kích hoạt tại epoch {epoch + 1}!")
                print(f"{'=' * 80}")
                print(f"   📌 Best Epoch: {best_epoch}")
                print(f"   📉 Best Test Loss: {best_test_loss:.4f}")
                print(f"   📊 Best BLEU-4: {best_bleu_4:.2f}")
                print(f"   ⏱️ Đã không cải thiện sau {EARLY_STOPPING_PATIENCE} epochs")
                print(f"{'=' * 80}")
                break  # 💥 DỪNG TRAINING

        # 💥 THÊM: Lưu training log mỗi 5 epochs
        if (epoch + 1) % 5 == 0:
            log_file = os.path.join(LOG_DIR, f"training_log_epoch_{epoch + 1}.json")
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(training_history, f, indent=4, ensure_ascii=False)
            print(f"   📝 Đã lưu training log: {log_file}")

    # --- KẾT THÚC TRAINING ---
    print(f"\n{'=' * 80}")
    print("--- ✅ Huấn luyện hoàn tất! ---")
    print(f"{'=' * 80}")
    print(f"   📊 Tổng số epochs đã chạy: {len(training_history['epochs'])}")
    print(f"   🏆 Best Epoch: {training_history['best_epoch']}")
    print(f"   📉 Best Test Loss: {training_history['best_test_loss']:.4f}")
    print(f"   📊 Best BLEU-4: {training_history['best_bleu_4']:.2f}")

    # 💥 THÊM: Lưu final training log
    final_log_file = os.path.join(LOG_DIR, "training_log_final.json")
    with open(final_log_file, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=4, ensure_ascii=False)
    print(f"\n   📁 Files đã lưu:")
    print(f"      ✅ Model checkpoint: {CHECKPOINT_FILE}")
    print(f"      ✅ Training log: {final_log_file}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n❌ LỖI KHỞI TẠO: KHÔNG TÌM THẤY FILE.")
        print(f"   {e}")
        print("   Hãy chắc chắn rằng đường dẫn trong file train.py là ĐÚNG.")
    except KeyboardInterrupt:
        print("\n\n⚠️ Training bị gián đoạn bởi người dùng!")
    except Exception as e:
        print(f"\n❌ LỖI TRONG QUÁ TRÌNH CHẠY: {e}")
        import traceback

        traceback.print_exc()