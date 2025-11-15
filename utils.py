import torch
from nltk.translate.bleu_score import corpus_bleu
import re

def count_parameters(model, trainable_only=True):
    """Đếm số parameters"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def load_checkpoint(model, checkpoint_path, device='cuda'):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def save_checkpoint(state, path):
    """
    Lưu checkpoint (state dictionary) vào một file.
    Args:
        state (dict): Dictionary chứa mọi thứ bạn muốn lưu.
        path (str): Đường dẫn file để lưu.
    """
    print("=> Đang lưu checkpoint...")
    torch.save(state, path)
    print(f"=> Đã lưu checkpoint tại {path}")


def get_model_summary(model):
    """In tóm tắt model"""
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("=" * 60)
    
    # Breakdown by component
    print("\nComponent breakdown:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name}: {params:,} ({trainable:,} trainable)")
    print("=" * 60)


# (Bạn có thể lấy PAD_TOKEN_ID từ tokenizer, nhưng dùng string dễ hơn)
PAD_TOKEN_STR = '<pad>'
BOS_TOKEN_STR = '<s>'
EOS_TOKEN_STR = '</s>'


def clean_sentence(sentence_str):
    """
    Hàm helper để dọn dẹp câu sau khi decode từ tokenizer
    """
    # Xóa padding
    sentence_str = sentence_str.replace(PAD_TOKEN_STR, '')

    # Xóa token bắt đầu câu
    sentence_str = sentence_str.replace(BOS_TOKEN_STR, '')

    # Xóa token kết thúc câu
    sentence_str = sentence_str.replace(EOS_TOKEN_STR, '')

    # Xóa khoảng trắng thừa ở đầu/cuối
    return sentence_str.strip()


def calculate_bleu_scores(predictions, references):
    """
    Tính điểm BLEU-1 đến BLEU-4 sử dụng NLTK.

    Args:
        predictions (list): List các câu dự đoán (list[str])
        references (list): List CỦA CÁC list câu tham chiếu (list[list[str]])
                           Mỗi ảnh có thể có 1 hoặc nhiều câu tham chiếu.
                           Trong trường hợp của bạn, nó sẽ là list[list[str]] với 1 câu.

    Returns:
        dict: Một dict chứa điểm BLEU-1, BLEU-2, BLEU-3, BLEU-4.
    """

    # --- CHUẨN BỊ DỮ LIỆU CHO NLTK ---
    # 1. Tách các câu thành list các từ (token)
    #    Ví dụ: "con mèo" -> ["con", "mèo"]
    pred_tokens = [sentence.split() for sentence in predictions]

    # 2. Định dạng lại references
    #    Ví dụ: [["con mèo"]] -> [[["con", "mèo"]]]
    ref_tokens = [[sentence.split() for sentence in ref_list] for ref_list in references]

    print(f"\n--- 📈 Đang tính điểm BLEU cho {len(pred_tokens)} mẫu ---")

    # Tính điểm
    bleu_1 = corpus_bleu(ref_tokens, pred_tokens, weights=(1.0, 0, 0, 0))
    bleu_2 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.33, 0.33, 0.33, 0))
    bleu_4 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25))

    scores = {
        "BLEU-1": bleu_1 * 100,
        "BLEU-2": bleu_2 * 100,
        "BLEU-3": bleu_3 * 100,
        "BLEU-4": bleu_4 * 100  # Đây là chỉ số thường được báo cáo nhất
    }

    return scores