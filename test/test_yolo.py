from pathlib import Path
import cv2
from ultralytics import YOLO

# Thư mục chứa ảnh test
TEST_IMG_DIR = Path("./dataset/yolo/images/test")

# Thư mục xuất ảnh sau khi detect
OUTPUT_DIR = Path("./dataset/outputs/detect")

# Trọng số YOLO đã train xong (best.pt)
WEIGHTS = "models/license_plate_yolov8/weights/best.pt"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def draw_box(image, box, color=(0, 255, 0), thickness=2):

    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image


def process_image(model, img_path: Path):

    # 1) Đọc ảnh bằng OpenCV
    img = cv2.imread(str(img_path))

    if img is None:
        print(f"Không đọc được ảnh: {img_path}")
        return None

    # 2) YOLO infer (trả về list kết quả)
    results = model(img)[0]   # results[0] = kết quả cho ảnh này

    # 3) Lặp qua từng bbox
    for box in results.boxes:
        # box.xyxy[0] = tensor (x1,y1,x2,y2)
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Vẽ bbox lên ảnh
        img = draw_box(img, (x1, y1, x2, y2))

    return img


def main():
    model = YOLO(WEIGHTS)

    ensure_dir(OUTPUT_DIR)

    # Lấy tất cả ảnh test
    img_files = list(TEST_IMG_DIR.glob("*.*"))

    print(f"Tổng ảnh test tìm thấy: {len(img_files)}")

    for img_path in img_files:
        print(f"Xử lý: {img_path.name}")

        img_out = process_image(model, img_path)

        if img_out is None:
            continue

        # Lưu ảnh đã vẽ bbox
        out_path = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(out_path), img_out)

    print(f"Ảnh đầu ra nằm tại: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
