### License-Plate-Recognition | YOLOv8 + vLLM Qwen2-VL Fine-tuning ###

Một hệ thống nhận diện và trích xuất biển số xe gồm 3 phần:
1. **Huấn luyện mô hình YOLOv8** để phát hiện vị trí biển số xe (ô tô, xe máy).
2. **Huấn luyện (Fine-tune) mô hình vLLM (image_to_text)** dựa trên `unsloth/Qwen2-VL-2B-Instruct-bnb-4bit` để trích xuất thông tin biển số xe.
3. **Ứng dụng Streamlit** tích hợp mô hình YOLOv8 và mô hình vLLM đã fine-tune để trích xuất nội dung biển số theo thời gian thực hoặc từ ảnh tải lên.


## 🧠 Kiến trúc hệ thống

```
├── dataset/yolo/                    # Datasets phục vụ cho việc training và test mô hình
│ ├── images                         # Thư mục chưa hình ảnh biển số
│ └── labels                         # Thư mục gán nhãn cho hình ảnh
│                         
├── models                           # Chứa các mô hình được sử dụng
│                           
├── src/
│ ├── finetune_OCR.py                # Chương trình finetune mô hình vLLM
│ └── train_yolo.py                  # Chương trình huấn luyện mô hình YOLO
│                  
├── main.py                          # Chương trình chính
```

## 🚀 Tính năng

- **Tính năng chính**: Nhận diện và trích xuất biển số từ hỉnh ảnh hoặc video được cung cấp.
- **Giao diện Web thân thiện**: Xây dựng bằng Streamlit, cho phép người dùng tương tác dễ dàng.
- **Hỗ trợ đa dạng định dạng**: Có thể tải lên hình ảnh, video hoặc nhận diện Real-time qua Webcam/Camera.
- **Tính năng hỗ trợ**: Hiển thị FPS, vẽ bounding box và biển số lên video.

## ⚙️ Cài đặt và Chạy dự án

### STEP-00:

Clone the repository

``` bash
git clone https://github.com/HikoKoi/License-Plate-Recognition.git
```
## STEP-01: Tạo môi trường ảo

``` bash
python -m venv venv
```

``` bash
source venv/Scripts/activate
```
## STEP-02: Tải các thư viện cần thiết requirements.txt

``` bash
pip install -r requirements.txt
```
## STEP-03: Thêm data của bạn để training cho mô hình nhận diện:

Có thể sử dụng label-studio.

Datasets tham khảo:
- [Bộ ảnh biển số xe máy – GreenParking](https://github.com/thigiacmaytinh/DataThiGiacMayTinh/blob/main/GreenParking.zip)  
  Gồm nhiều góc chụp, điều kiện ánh sáng khác nhau, phù hợp cho nhận diện biển số xe máy.
- [Bộ ảnh biển số ô tô](https://drive.google.com/file/d/1U5ebTzW2c_sVVTCSX1QH-ZJFpLijMdUv/view)  
  Bao gồm đầy đủ các loại biển xe ô tô: biển dài, biển vuông, và biển vàng.
 
Thêm vào các thư mục **images** và **label**

## STEP-04: Chạy các chương trình traning/finetune:

``` bash
python src/train_yolo.py
```
``` bash
python src/finetune_OCR.py
```

## STEP-04: Chạy chương trình chính:

``` bash
streamlit run main.py
```
Link:

``` bash
http://localhost:8501
```

## Công nghệ sử dụng:

- Python
- Streamlit
- OpenCV
- Unsloth
- YOLOv8
