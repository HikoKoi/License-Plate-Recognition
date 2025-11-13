from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from unsloth import FastVisionModel
from transformers import TextStreamer


# ================================
# 1. ĐƯỜNG DẪN
# ================================
TEST_IMG_DIR = Path("./dataset/yolo/images/test")
OUTPUT_DIR = Path("./dataset/outputs/ocr")

YOLO_WEIGHTS = "models/license_plate_yolov8/weights/best.pt"    # YOLO detect
UNSLOTH_OCR_DIR = "models/unsloth_ocr"                            # model đã finetune OCR


# ================================
# 2. HÀM TIỆN ÍCH
# ================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def draw_box_text(img, box, text):
    """
    Vẽ bbox + text lên ảnh.
    box: (x1, y1, x2, y2)
    text: chuỗi OCR
    """
    x1, y1, x2, y2 = map(int, box)

    # Vẽ bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Vẽ nền text
    cv2.rectangle(img, (x1, y1 - 30), (x1 + 200, y1), (0, 255, 0), -1)

    # Vẽ text
    cv2.putText(
        img, text, (x1 + 5, y1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 0, 0), 2
    )
    return img


# ================================
# 3. Load OCR Model (Unsloth)
# ================================
def load_ocr_model():
    """
    Load Unsloth model đã merge từ folder models/unsloth_ocr
    """
    print("🔤 Loading OCR model...")
    model, tokenizer = FastVisionModel.from_pretrained(
        UNSLOTH_OCR_DIR,
        load_in_4bit=True,
        device_map="auto"
    )

    FastVisionModel.for_inference(model)
    return model, tokenizer


# ================================
# 4. OCR Một ảnh crop
# ================================
def ocr_plate(model, tokenizer, crop_img_np):
    """
    Input:
        crop_img_np: numpy array (BGR)
    Output:
        text OCR (string)
    """
    # convert sang RGB
    crop_rgb = cv2.cvtColor(crop_img_np, cv2.COLOR_BGR2RGB)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": crop_rgb},
                {"type": "text", "text": "Extract license plate text."}
            ],
        }
    ]

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    inputs = tokenizer(
        crop_rgb,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    streamer = TextStreamer(tokenizer, skip_prompt=True)
    out = model.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=64,
        temperature=0.2,
    )

    # Convert output tokens -> text
    text_out = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    return text_out


# ================================
# 5. Xử lý 1 ảnh
# ================================
def process_image(yolo, ocr_model, tokenizer, img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"❌ Không đọc được ảnh: {img_path}")
        return None

    # YOLO detect
    results = yolo(img)[0]

    # Lặp từng bbox
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Crop vùng biển số
        crop = img[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0:
            continue

        # OCR vùng crop
        text = ocr_plate(ocr_model, tokenizer, crop)

        # Vẽ text + bbox lên ảnh đầy đủ
        img = draw_box_text(img, (x1, y1, x2, y2), text)

    return img


# ================================
# 6. MAIN
# ================================
def main():
    print("🚗 Loading YOLO detector...")
    yolo = YOLO(YOLO_WEIGHTS)

    ocr_model, tokenizer = load_ocr_model()

    ensure_dir(OUTPUT_DIR)

    img_files = list(TEST_IMG_DIR.glob("*.*"))
    print(f"Tìm thấy {len(img_files)} ảnh test.")

    for img_path in img_files:
        print(f"➡ Xử lý: {img_path.name}")

        out_img = process_image(yolo, ocr_model, tokenizer, img_path)
        if out_img is None:
            continue

        out_path = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(out_path), out_img)

    print(f"🎉 Xong! Ảnh output lưu tại: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
