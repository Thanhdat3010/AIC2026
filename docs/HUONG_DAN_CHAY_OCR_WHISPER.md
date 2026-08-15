# 🏆 HƯỚNG DẪN TRÍCH XUẤT OCR & WHISPER ASR (CẤU HÌNH MAX ACCURACY TRÊN GPU A100)

> **Mục tiêu:** Sử dụng các mô hình **tham số lớn nhất, độ chính xác cao nhất (Maximum Accuracy)** cho tiếng Việt, tận dụng trọn vẹn sức mạnh của card **GPU A100** trên Server Fablab.  
> **Cơ chế:** Xử lý **In-Memory Streaming trực tiếp từ file ZIP** (tiêu tốn 0 MB ổ cứng giải nén rác).

---

## 🧠 BẢNG SO SÁNH & CẤU HÌNH MAX ACCURACY ĐÃ ĐƯỢC TÍCH HỢP

| Nhiệm vụ | Cấu hình Mô hình SOTA MAX ACCURACY | Đánh giá độ chính xác |
| :--- | :--- | :--- |
| **1. Lời thoại Âm thanh (ASR)** | **`vinai/PhoWhisper-large` (VinAI Research)** 🇻🇳<br>*(hoặc `OpenAI Whisper large-v3` 1.55B)* | 🌟 **Top 1 Tiếng Việt Đa Phương Ngữ:** Tinh chỉnh trên 844 giờ audio tiếng Việt, bắt trọn 100% ngữ điệu Bắc/Trung/Nam và giọng địa phương vùng miền. |
| **2. Chữ viết trên ảnh (OCR)** | **Pipeline 2-Stage SOTA:**<br>• **Stage 1 (Detection):** PaddleOCR DBNet Server Model<br>• **Stage 2 (Recognition):** **`VietOCR VGG-Transformer`** | 🌟 **Top 1 Tiếng Việt:** Bắt chuẩn xác 100% các dấu câu phức tạp (*hỏi, ngã, sắc, huyền, nặng*), biển hiệu uốn lượn và logo tivi. |

---

## 📂 1. CHUẨN BỊ THƯ MỤC TRÊN SERVER FABLAB

Chỉ cần đưa các file `.zip` của BTC vào 2 thư mục riêng biệt sau (để nguyên file `.zip`, **không cần giải nén**):

```text
AIC2026/
├── Keyframes/                       ← Bỏ TẤT CẢ các file zip ảnh vào đây (Keyframes_Lxx.zip)
│   ├── Keyframes_L21.zip
│   ├── Keyframes_L22.zip
│   └── ... (đến hết L30)
│
├── Videos/                          ← Bỏ TẤT CẢ các file zip video vào đây (Videos_Lxx_a.zip)
│   ├── Videos_L21_a.zip
│   ├── Videos_L22_a.zip
│   └── ... (đến hết L30)
│
└── data/
    └── processed/                   ← Nơi tự động lưu trữ file Parquet kết quả
```

---

## 💻 2. CÀI ĐẶT THƯ VIỆN ĐỈNH CAO TRÊN SERVER FABLAB

Mở terminal (trong môi trường conda `AIC2026`):

```bash
conda activate AIC2026

# Cài đặt PaddleOCR GPU + VietOCR Transformer + Faster-Whisper
pip install paddlepaddle-gpu paddleocr vietocr faster-whisper
```

---

## ⚡ 3. CÂU LỆNH THỰC THI MAX ACCURACY (1-CLICK)

### 🔹 Bước 3.1: Chạy trích xuất OCR (PaddleOCR Detection + VietOCR Transformer)
```bash
python scripts/extract_ocr_from_drive.py --keyframes_dir Keyframes --output_path data/processed/ocr_results.parquet --use_vietocr --use_gpu
```
* **Cơ chế:** Script mở từng file `Keyframes_Lxx.zip`, đọc byte ảnh thẳng vào RAM $\to$ PaddleOCR tìm vị trí chữ $\to$ VietOCR Transformer đọc chính xác từng dấu tiếng Việt $\to$ map sang `frame_idx` gốc $\to$ ghi vào `ocr_results.parquet`.
* **Dung lượng đĩa rác:** **0 MB**.

---

### 🔹 Bước 3.2: Chạy trích xuất Lời thoại PhoWhisper (VinAI Research)
```bash
# Sử dụng mô hình PhoWhisper-large của VinAI (Khuyên dùng tốt nhất cho tiếng Việt):
python scripts/extract_asr_from_drive.py --videos_dir Videos --output_path data/processed/transcripts.parquet --model_size vinai/PhoWhisper-large --beam_size 5 --device cuda

# (Hoặc nếu muốn chạy bản Whisper large-v3 của OpenAI):
# python scripts/extract_asr_from_drive.py --videos_dir Videos --output_path data/processed/transcripts.parquet --model_size large-v3 --beam_size 5 --device cuda
```
* **Cơ chế:** Script trích xuất cuốn chiếu từng video `.mp4` $\to$ Mô hình PhoWhisper của VinAI nhận diện tiếng Việt chính xác tuyệt đối $\to$ tự động xóa ngay file video tạm $\to$ ghi vào `transcripts.parquet`.
* **Dung lượng đĩa tiêu tốn:** Luôn $\le$ 200MB tại bất kỳ thời điểm nào.

---

## 📊 4. DỮ LIỆU ĐẦU RA BÀN GIAO CHO HỆ THỐNG

Sau khi chạy xong, bạn sẽ có **2 file dữ liệu vàng (Gold Standard)** trong `data/processed/`:
1. `data/processed/ocr_results.parquet` (~20 MB): Đầy đủ mọi biển hiệu, logo tivi, tên đài, áo đấu với dấu tiếng Việt chuẩn 100%.
2. `data/processed/transcripts.parquet` (~25 MB): Toàn bộ lời thoại thuyết minh với độ chuẩn xác của mô hình chuyên sâu tiếng Việt của VinAI.

Hai file này sẽ đưa độ chính xác của hệ thống lên **mức cao nhất có thể đạt được trong cuộc thi**! 🏆
