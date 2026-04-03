"""
config.py – Tất cả cấu hình, hyperparameters, đường dẫn.
Chỉnh sửa file này trước khi chạy train.py.
"""

import os

# ─── Tên project & W&B ──────────────────────────────────────────────────────
PROJECT_NAME  = 'testpart3_densenet121'
WANDB_API_KEY = 'a4794ed6f8c305b6367db182a46673ebbd057fb6'
WANDB_MODE    = 'online'   # 'online' | 'offline' | 'disabled'

# ─── Dữ liệu ────────────────────────────────────────────────────────────────
DATA_DIR = r'D:/TUNGG/data_Chile_HoaKi_ThoNhiKy/Multi_class_Classification_TNK - Copy'   # ← sửa đường dẫn tới thư mục data

CLASSES = [
    'Acute Otitis Media',
    'Chronic Suppurative Otitis Media',
    'Tympanoskleros',
    'Normal',
]
NUM_CLS     = len(CLASSES)
IDX_TO_CLS  = {i: c for i, c in enumerate(CLASSES)}

# ─── Model ──────────────────────────────────────────────────────────────────
# ResNet   : 'resnet50', 'resnet101'
# Efficient: 'efficientnet_b0' → 'efficientnet_b7'
# ViT      : 'vit_base_patch16_224'
# ConvNeXt : 'convnext_tiny', 'convnext_base'
# Swin     : 'swin_tiny_patch4_window7_224'
# DenseNet : 'densenet121', 'densenet201'
MODEL_NAME     = 'resnet50'
MODEL_PRETRAIN = True

# ─── Augmentation ───────────────────────────────────────────────────────────
AUG_MODE = 'balanced'   # 'balanced' | 'imbalanced' | 'none'

# ─── Hyperparameters ────────────────────────────────────────────────────────
HPARAMS = dict(
    seeds        = [1111],
    epochs       = 20,
    batch_size   = 16,
    lr           = 1e-4,
    weight_decay = 1e-4,
    patience     = 15,
    num_workers  = 2,
    use_amp      = True,
    print_freq   = 10,
)

# ─── Normalisation (ImageNet) ───────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ─── Output directories ─────────────────────────────────────────────────────
BASE_EXP_DIR = f'experiments/{PROJECT_NAME}'

DIRS = {
    'models':  os.path.join(BASE_EXP_DIR, 'Models'),
    'errors':  os.path.join(BASE_EXP_DIR, 'Errors'),
    'reports': os.path.join(BASE_EXP_DIR, 'Reports'),
    'plots':   os.path.join(BASE_EXP_DIR, 'Plots'),
}
INF_CONFIG = {
    'input':      'D:/TUNGG/data_Chile_HoaKi_ThoNhiKy/ThoNhiKy/abnormal/aom', # Thư mục chứa ảnh mới HOẶC 1 file ảnh lẻ
    'output_csv': os.path.join(BASE_EXP_DIR, 'Predictions_Summary.csv'),
    'batch_size': 1, # Dự đoán từng ảnh 
}