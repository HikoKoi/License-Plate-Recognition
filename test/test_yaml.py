import torch
from ultralytics import YOLO

print("=== KIỂM TRA TORCH ===")
print("Phiên bản torch:", torch.__version__)
print("Có GPU CUDA không?:", torch.cuda.is_available())

print("\n=== KIỂM TRA TẢI MODEL YOLO CƠ BẢN ===")
model = YOLO("models/yolov8n.pt")
print("Model YOLO đã load xong.")

print("\n=== KIỂM TRA YAML DATASET ===")

try:
    model.train(
        data="./dataset/license_plate.yaml", 
        epochs=1,                           
        imgsz=640,
        batch=2,
        device=0 if torch.cuda.is_available() else "cpu"
    )
    print("YAML và dữ liệu OK (train 1 epoch test thành công).")
except Exception as e:
    print("Có lỗi khi dùng YAML/dataset:")
    print(e)
