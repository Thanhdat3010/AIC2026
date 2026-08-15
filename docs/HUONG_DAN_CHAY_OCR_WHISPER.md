# 🏆 HƯỚNG DẪN TRÍCH XUẤT OCR & WHISPER (TẢI TRỰC TIẾP TỪ MÁY CHỦ BTC HOẶC DRIVE)

> **Cơ chế đỉnh cao:** Script tự động **Tải trực tiếp qua Link BTC (`aic-data.ledo.io.vn`) $\to$ Xử lý trong RAM bằng GPU A100 $\to$ Tự động xóa ngay file zip trên server $\to$ Tải tiếp file sau**.  
> **Cơ chế chịu lỗi (Fault-Tolerant):** Nếu bị rớt mạng hoặc đứt quãng giữa chừng, script **tự động Resume** hoặc cho phép bạn **chọn chạy từ gói bị hỏng trở đi** mà không phải chạy lại từ đầu!

---

## 🧠 CÁC MÔ HÌNH MAX ACCURACY ĐƯỢC SỬ DỤNG

1. **OCR (Chữ trên ảnh):** **PaddleOCR Detection + `VietOCR VGG-Transformer`** (Nhận diện chuẩn xác 100% dấu tiếng Việt).
2. **ASR (Lời thoại video):** **`vinai/PhoWhisper-large` (VinAI Research)** (Tinh chỉnh chuyên sâu trên 844 giờ audio tiếng Việt).

---

## 💻 1. CÀI ĐẶT THƯ VIỆN TRÊN SERVER FABLAB

```bash
conda activate AIC2026
pip install paddleocr==2.8.1 requests vietocr faster-whisper gdown
```

---

## ⚡ 2. CÂU LỆNH CHẠY TOÀN BỘ (TỰ ĐỘNG TỪ ĐẦU ĐẾN CUỐI)

### 🔹 Chạy OCR:
```bash
python scripts/extract_ocr_from_drive.py --urls_file config/drive_keyframes_urls.txt --output_path data/processed/ocr_results.parquet --use_vietocr --use_gpu
```

### 🔹 Chạy PhoWhisper:
```bash
python scripts/extract_asr_from_drive.py --urls_file config/drive_videos_urls.txt --output_path data/processed/transcripts.parquet --model_size vinai/PhoWhisper-large --beam_size 5 --device cuda
```

---

## 🔄 3. CƠ CHẾ TIẾP TỤC CHẠY KHI BỊ HỎNG / GIÁN ĐOẠN (RESUME)

### 🌟 Cách 1: Tự Động Resume Thông Minh (Không cần làm gì cả)
Nếu đang chạy mà bị rớt mạng hoặc ngắt giữa chừng, bạn chỉ cần **chạy lại y nguyên lệnh cũ**:
* Script sẽ tự động nạp file parquet cũ.
* Tự động **bỏ qua 100% các video đã làm xong** và chỉ xử lý tiếp những video còn thiếu!

---

### 🌟 Cách 2: Chọn chạy từ một gói cụ thể bằng `--start_from`
Ví dụ: Đang chạy đến gói `Keyframes_L25.zip` bị ngắt, bạn muốn chạy tiếp từ gói này:

```bash
# OCR chạy tiếp từ Keyframes_L25.zip:
python scripts/extract_ocr_from_drive.py --start_from Keyframes_L25.zip --use_vietocr --use_gpu

# PhoWhisper chạy tiếp từ Videos_L25_a.zip:
python scripts/extract_asr_from_drive.py --start_from Videos_L25_a.zip --model_size vinai/PhoWhisper-large --device cuda
```

---

### 🌟 Cách 3: Chọn chạy từ số thứ tự gói bằng `--start_index`
Ví dụ: Có 14 gói, bạn muốn bắt đầu chạy từ gói số 5 (tức từ gói 5 đến 14):

```bash
# OCR chạy từ gói 5:
python scripts/extract_ocr_from_drive.py --start_index 5 --use_vietocr --use_gpu

# PhoWhisper chạy từ gói 5:
python scripts/extract_asr_from_drive.py --start_index 5 --model_size vinai/PhoWhisper-large --device cuda
```

---

### 🌟 Cách 4: Chạy riêng lẻ duy nhất 1 link cụ thể bằng `--url`
```bash
python scripts/extract_ocr_from_drive.py --url "https://aic-data.ledo.io.vn/Keyframes_L26_a.zip" --use_vietocr --use_gpu
```
