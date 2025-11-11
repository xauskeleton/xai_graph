import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
# --- THÊM CÁC THƯ VIỆN CẦN THIẾT ---
from transformers import AutoTokenizer

class KTVICDataset(Dataset):
    def __init__(self, json_file, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        self.caption_key = "caption" 

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.image_id_to_filename = {}
        for img_info in data['images']:
            self.image_id_to_filename[img_info['id']] = img_info['filename']

        self.annotations = data['annotations']

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        ann = self.annotations[index]
        caption = ann[self.caption_key]
        image_id = ann['image_id']
        filename = self.image_id_to_filename[image_id]
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, caption

class CollateFn:

    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):

        #Tách riêng ảnh và caption
        images = [item[0] for item in batch]
        captions = [item[1] for item in batch]

        images_batch = torch.stack(images, dim=0)

        tokenized_batch = self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return images_batch, tokenized_batch

def get_loader(json_file, image_dir, tokenizer, batch_size=32, shuffle=True, num_workers=4):
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = KTVICDataset(
        json_file=json_file,
        image_dir=image_dir,
        transform=transform,
    )

    # --- SỬA Ở ĐÂY ---
    # Khởi tạo CollateFn với tokenizer
    collate_fn = CollateFn(tokenizer)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn  # <-- Dùng collate_fn mới
    )

    return data_loader, dataset


if __name__ == '__main__':
    
    TRAIN_JSON_PATH = 'data/train_data.json' 
    TRAIN_IMAGE_DIR = 'data/train/'
    TEST_JSON_PATH = 'data/test_data.json'
    TEST_IMAGE_DIR = 'data/test/' 

    BATCH_SIZE = 4
    
    print("--- 🤖 Đang tải Tokenizer (PhoBERT) ---")
    try:
        # Tải tokenizer PhoBERT
        # (Cần cài đặt: pip install transformers)
        tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
        print(f"✅ Tải tokenizer thành công! Vocab size: {tokenizer.vocab_size}")

        print("\n--- 🚀 Đang khởi tạo DataLoader cho TẬP HUẤN LUYỆN (Train) ---")
        train_loader, train_dataset = get_loader(
            json_file=TRAIN_JSON_PATH,
            image_dir=TRAIN_IMAGE_DIR,
            tokenizer=tokenizer, # <-- Truyền tokenizer vào
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        print(f"✅ Tải thành công! Tổng số mẫu huấn luyện: {len(train_dataset)}")
        
        # Lấy thử 1 batch train
        # Đầu ra thứ 2 bây giờ là 1 dictionary
        train_images, train_tokens = next(iter(train_loader))
        
        print(f"   -> Kích thước batch ảnh train: {train_images.shape}")
        print(f"   -> Dữ liệu text (dictionary keys): {train_tokens.keys()}")
        print(f"   -> Kích thước input_ids: {train_tokens['input_ids'].shape}")
        print(f"   -> Kích thước attention_mask: {train_tokens['attention_mask'].shape}")
        
        # Giải mã (decode) caption đầu tiên để xem
        first_caption_ids = train_tokens['input_ids'][0]
        first_caption_text = tokenizer.decode(first_caption_ids, skip_special_tokens=False)
        print(f"   -> Caption 0 (dạng số): {first_caption_ids}")
        print(f"   -> Caption 0 (dạng chữ): {first_caption_text}")


        print("\n--- 🧪 Đang khởi tạo DataLoader cho TẬP KIỂM TRA (Test) ---")
        test_loader, test_dataset = get_loader(
            json_file=TEST_JSON_PATH,
            image_dir=TEST_IMAGE_DIR,
            tokenizer=tokenizer, # <-- Truyền tokenizer vào
            batch_size=BATCH_SIZE,
            shuffle=False
        )
        print(f"✅ Tải thành công! Tổng số mẫu kiểm tra: {len(test_dataset)}")

        # Lấy thử 1 batch test
        test_images, test_tokens = next(iter(test_loader))
        print(f"   -> Kích thước batch ảnh test: {test_images.shape}")
        print(f"   -> Kích thước input_ids (test): {test_tokens['input_ids'].shape}")

    except ImportError:
        print("\n❌ LỖI: Không tìm thấy thư viện 'transformers'.")
        print("   Vui lòng cài đặt bằng lệnh: pip install transformers")
    except Exception as e:
        print(f"\n❌ LỖI: {e}")
        print(f"   Hãy kiểm tra lại các đường dẫn trong file data_loader.py")