# 🏆 HƯỚNG DẪN TRÍCH XUẤT OCR & WHISPER (TẢI TRỰC TIẾP TỪ MÁY CHỦ BTC HOẶC DRIVE)

> **Cơ chế đỉnh cao:** Script tự động **Tải trực tiếp qua Link BTC (`aic-data.ledo.io.vn`) hoặc Google Drive $\to$ Xử lý trong RAM bằng GPU A100 $\to$ Tự động xóa ngay file zip trên server $\to$ Tải tiếp file sau**.  
> **Ưu điểm:** **Không cần Mount**, tốc độ tải mạng cực đại 1Gbps, ổ cứng server luôn $\le$ 3GB.

---

## 🧠 CÁC MÔ HÌNH MAX ACCURACY ĐƯỢC SỬ DỤNG

1. **OCR (Chữ trên ảnh):** **PaddleOCR Detection + `VietOCR VGG-Transformer`** (Nhận diện chuẩn xác 100% dấu tiếng Việt).
2. **ASR (Lời thoại video):** **`vinai/PhoWhisper-large` (VinAI Research)** (Tinh chỉnh chuyên sâu trên 844 giờ audio tiếng Việt đa phương ngữ Bắc/Trung/Nam).

---

## 📋 1. DANH SÁCH LINK ĐÃ ĐƯỢC CẤU HÌNH SẴN 100%

Hệ thống đã nạp sẵn toàn bộ 14 link tải trực tiếp từ máy chủ BTC (`aic-data.ledo.io.vn`) vào 2 file cấu hình:

* 👉 **`config/drive_keyframes_urls.txt`** (Chứa đủ 14 link `Keyframes_L21.zip` $\to$ `Keyframes_L30.zip`)
* 👉 **`config/drive_videos_urls.txt`** (Chứa đủ 14 link `Videos_L21_a.zip` $\to$ `Videos_L30_a.zip`)

---

## 💻 2. CÀI ĐẶT THƯ VIỆN TRÊN SERVER FABLAB

Mở terminal trên Server Fablab:

```bash
cd AIC2026
git pull origin master

conda activate AIC2026
pip install requests paddlepaddle-gpu paddleocr vietocr faster-whisper gdown
```

---

## ⚡ 3. CÂU LỆNH CHẠY TỰ ĐỘNG CUỐN CHIẾU (1-CLICK)

### 🔹 Bước 3.1: Chạy OCR (Tự động kéo từ máy chủ BTC $\to$ VietOCR đọc chữ $\to$ Tự xóa ZIP)
```bash
python scripts/extract_ocr_from_drive.py \
    --urls_file config/drive_keyframes_urls.txt \
    --output_path data/processed/ocr_results.parquet \
    --use_vietocr \
    --use_gpu
```

---

### 🔹 Bước 3.2: Chạy PhoWhisper (Tự động kéo từ máy chủ BTC $\to$ PhoWhisper nghe $\to$ Tự xóa ZIP)
```bash
python scripts/extract_asr_from_drive.py \
    --urls_file config/drive_videos_urls.txt \
    --output_path data/processed/transcripts.parquet \
    --model_size vinai/PhoWhisper-large \
    --beam_size 5 \
    --device cuda
```

---

### 💡 MẸO TEST THỬ 1 LINK TRỰC TIẾP QUA DÒNG LỆNH:
```bash
# Test OCR 1 file zip từ link BTC:
python scripts/extract_ocr_from_drive.py --url "https://aic-data.ledo.io.vn/Keyframes_L21.zip" --use_vietocr --use_gpu

# Test PhoWhisper 1 file zip từ link BTC:
python scripts/extract_asr_from_drive.py --url "https://aic-data.ledo.io.vn/Videos_L21_a.zip" --model_size vinai/PhoWhisper-large --device cuda
```
