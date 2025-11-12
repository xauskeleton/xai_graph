if __name__ == "__main__":
    print("Testing Image Captioning Model with PyG...")
    
    # Config
    vocab_size = 10000
    batch_size = 2
    
    # Initialize model
    model = ImageCaptioningModel(
        vocab_size=vocab_size,
        embed_dim=256,
        hidden_dim=512,
        num_objects=36,
        gnn_layers=3,
        gnn_heads=4,
        k_neighbors=10
    )
    
    # Model summary
    get_model_summary(model)
    
    # Test data
    images = torch.randn(batch_size, 3, 224, 224)
    captions = torch.randint(0, vocab_size, (batch_size, 20))
    
    # Training mode
    print("\n" + "=" * 60)
    print("TRAINING MODE TEST")
    print("=" * 60)
    model.train()
    outputs = model(images, captions)
    print(f"Input shape: {images.shape}")
    print(f"Caption shape: {captions.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Output range: [{outputs.min():.2f}, {outputs.max():.2f}]")
    
    # Inference mode
    print("\n" + "=" * 60)
    print("INFERENCE MODE TEST")
    print("=" * 60)
    model.eval()
    generated = model.generate_caption(images, max_length=15)
    print(f"Generated shape: {generated.shape}")
    print(f"Sample caption: {generated[0].tolist()}")
    
    print("\n✅ All tests passed!")