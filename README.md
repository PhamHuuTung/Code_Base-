classification
config.py          ← Tất cả cấu hình (model, data, hyperparams, W&B, data dự doán)
data.py            ← Dataset, augmentation, DataLoader
model.py           ← Khởi tạo model với timm
engine.py          ← Training loop, predict, checkpoint
metrics.py         ← Tính metrics, vẽ biểu đồ, lưu ảnh lỗi
wandb_utils.py     ← Logging lên Weights & Biases
train.py           ← File chính để chạy
predict.py         ← Dự đoán file ảnh ngoài

1. config
   tên folder ra cùng tên với tên project trên wandb
   các hyperparams: seed, epochs, batch_size, learning_rate,..
   địa chỉ folder chứa ảnh inference predict
   tên và số lượng loại bệnh phải trùng tên và thứ tự folder ảnh ví dụ:
       Multi_class_Classification_TNK_Copy
        Acute Otitis Media
        Chronic Suppurative Otitis Media
        Tympanoskleros
        Normal
   tên model 

   
