import torch
from ultralytics import YOLO

MODEL_NAME = "models/yolov8n.pt"

def main():
    model = YOLO(MODEL_NAME)

    model.train(
        data="./dataset/license_plate.yaml",
        epochs=50,           # epochs: số vòng train (50–100 là ổn với dataset nhỏ)
        imgsz=640,           # imgsz: resize ảnh về kích thước vuông 640x640 (chuẩn YOLO)
        batch=2,             # batch: số ảnh xử lý trong 1 lần (tuỳ GPU)
        device=0 if torch.cuda.is_available() else "cpu",
        workers=4,           # số luồng load data
        project="models",
        name="license_plate_yolov8",   # folder output: models/license_plate_yolov8/
        exist_ok=True        # ghi đè nếu folder đã tồn tại
    )


if __name__ == "__main__":
    main()
