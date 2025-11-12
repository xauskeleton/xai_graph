import torch

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

