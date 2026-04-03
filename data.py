"""
data.py – Quản lý Dataset, Transforms và DataLoader.
"""

import glob
import os
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

import config

def load_data(base_dir):
    """Quét thư mục và lấy đường dẫn ảnh + nhãn."""
    paths, labels = [], []
    for idx, cls in enumerate(config.CLASSES):
        class_dir = os.path.join(base_dir, cls)
        if not os.path.exists(class_dir):
            print(f"   Không tìm thấy thư mục {class_dir}")
            continue
            
        files = []
        # Hỗ trợ nhiều định dạng ảnh
        for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']:
            files.extend(glob.glob(os.path.join(class_dir, ext)))
        
        valid_count = 0
        for f in files:
            try:
                # Kiểm tra ảnh có hợp lệ và ở hệ màu RGB không
                with Image.open(f) as img:
                    if img.mode == 'RGB':
                        paths.append(f)
                        labels.append(idx)
                        valid_count += 1
            except:
                continue
        print(f'   ✓ {cls}: {valid_count} images')
    
    if len(paths) == 0:
        raise ValueError(f"Không tìm thấy ảnh nào trong {base_dir}. Vui lòng kiểm tra lại DATA_DIR trong config.py")
        
    return paths, labels

def get_transforms():
    """Định nghĩa Augmentation dựa trên config.AUG_MODE."""
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])

    if config.AUG_MODE == 'balanced':
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(config.MEAN, config.STD),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ])
    elif config.AUG_MODE == 'imbalanced':
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(config.MEAN, config.STD),
        ])
    else: # 'none'
        train_tf = val_tf
        
    return train_tf, val_tf

class CustomDataset(Dataset):
    """Dataset trả về (ảnh, nhãn, đường_dẫn_file)."""
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(label, dtype=torch.long), path

class InferenceDataset(Dataset):
    """Dataset dùng cho ảnh mới không có nhãn."""
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, path

def build_dataloaders(all_paths, all_labels, train_tf, val_tf, seed):
    """Chia dữ liệu 70/20/10 và tạo DataLoader."""
    # Split Train / (Val + Test)
    tr_p, tmp_p, tr_l, tmp_l = train_test_split(
        all_paths, all_labels, 
        test_size=0.3, 
        stratify=all_labels, 
        random_state=seed
    )
    
    # Split Val / Test (từ 30% còn lại chia 2/3 và 1/3 -> 20% và 10%)
    v_p, te_p, v_l, te_l = train_test_split(
        tmp_p, tmp_l, 
        test_size=1/3, 
        stratify=tmp_l, 
        random_state=seed
    )

    kw = dict(
        batch_size=config.HPARAMS['batch_size'],
        num_workers=config.HPARAMS['num_workers'],
        pin_memory=True
    )

    loader_train = DataLoader(CustomDataset(tr_p, tr_l, train_tf), shuffle=True, **kw)
    loader_val   = DataLoader(CustomDataset(v_p, v_l, val_tf),   shuffle=False, **kw)
    loader_test  = DataLoader(CustomDataset(te_p, te_l, val_tf), shuffle=False, **kw)

    print(f'Data Split: Train={len(tr_p)} | Val={len(v_p)} | Test={len(te_p)}')
    return loader_train, loader_val, loader_test