"""
predict.py – File dự đoán tổng quát.
1. Tự động xử lý Ensemble (nhiều seeds) hoặc dự đoán đơn (1 seed).
2. Tự động đặt tên file kết quả theo ngày giờ (không đè file cũ).
3. Tách cột chuẩn cho Excel và chống lỗi Permission Error.
"""

import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime  

import config
import data
import model as model_builder


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_inference(image_path, model_list, mdl, transform):
    """Hàm lõi: Dự đoán 1 ảnh bằng danh sách N mô hình."""
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"  Lỗi khi mở ảnh {image_path}: {e}")
        return None
    
    all_probs = []
    
    with torch.no_grad():
        for ckpt in model_list:
            # Nạp trọng số của từng seed vào cùng 1 khung xương model
            checkpoint = torch.load(ckpt, map_location=device)
            mdl.load_state_dict(checkpoint['model_state_dict'])
            mdl.eval()
            
            # Lấy xác suất sau Softmax
            logits = mdl(img_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy() 
            all_probs.append(probs)
    
    # Tính trung bình cộng xác suất của N seeds
    avg_probs = np.mean(all_probs, axis=0)[0]
    pred_idx = np.argmax(avg_probs)
    
    return {
        'File': os.path.basename(image_path),
        'Prediction': config.CLASSES[pred_idx],
        'Confidence (%)': round(avg_probs[pred_idx] * 100, 2)
    }

def main():

    input_path = config.INF_CONFIG['input']
    if os.path.isfile(input_path):
        img_list = [input_path]
    else:
        img_list = [os.path.join(input_path, f) for f in os.listdir(input_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not img_list:
        print(f"  Không tìm thấy ảnh nào tại: {input_path}")
        return

    # Lấy danh sách tất cả các Seeds (.pth) hiện có trong thư mục Models
    model_dir = config.DIRS['models']
    ckpt_files = glob.glob(os.path.join(model_dir, "*.pth"))
    
    if not ckpt_files:
        print(f"  Không tìm thấy model (.pth) nào trong {model_dir}")
        return
    
    print(f"  Khởi động Inference với {len(ckpt_files)} Seeds...")
    print(f"  Đang xử lý {len(img_list)} ảnh...")

   # Khởi tạo Model và Transform một lần duy nhất
    mdl = model_builder.create_model()
    _, val_tf = data.get_transforms()

    #  Chạy vòng lặp dự đoán
    final_results = []
    for path in img_list:
        res = run_inference(path, ckpt_files, mdl, val_tf)
        if res:
            final_results.append(res)
            print(f"   {res['File']}: {res['Prediction']} ({res['Confidence (%)']}%)")

    
    # Tạo tên file dạng: Predictions_20240331_214530.csv
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"Predictions_{now}.csv"
    
    # Lưu vào thư mục Reports hoặc thư mục gốc của Project (BASE_EXP_DIR)
    final_output_path = os.path.join(config.BASE_EXP_DIR, output_filename)

    
    df = pd.DataFrame(final_results)
    
    
    try:
        df.to_csv(final_output_path, index=False, encoding='utf-8-sig')
        print("-" * 30)
        print(f"  THÀNH CÔNG!")
        print(f"  File kết quả: {output_filename}")
        print(f"  Đường dẫn: {final_output_path}")
    except PermissionError:
        print("\n  LỖI: Không thể ghi file. Có thể bạn đang mở một file kết quả cũ trong Excel.")
        print("  Hãy đóng Excel lại và chạy lại lệnh này.")

if __name__ == "__main__":
    main()