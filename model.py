"""
model.py – Khởi tạo model từ thư viện timm.
"""

import timm
import torch
import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_model():
    """Tạo model dựa trên config.MODEL_NAME."""
    print(f'   Building model: {config.MODEL_NAME}...')
    
    try:
        model = timm.create_model(
            config.MODEL_NAME,
            pretrained=config.MODEL_PRETRAIN,
            num_classes=config.NUM_CLS
        )
    except Exception as e:
        print(f" Lỗi: Không thể tạo model '{config.MODEL_NAME}'. Kiểm tra lại tên model trong config.py hoặc thư viện timm.")
        raise e

    # Đưa model lên GPU/CPU
    model = model.to(device)
    
    # Tính số lượng tham số
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'   Model ready | Params: {n_params:.2f}M | Device: {device}')
    
    return model

if __name__ == "__main__":
    # Test nhanh model
    m = create_model()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = m(dummy_input)
    print(f"   Test output shape: {output.shape}")