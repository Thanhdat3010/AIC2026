# 🏆 HƯỚNG DẪN TRÍCH XUẤT OCR & WHISPER (TẢI TRỰC TIẾP TỪ LINK GOOGLE DRIVE)

> **Cơ chế đỉnh cao:** Script tự động **Tải trực tiếp qua Link Google Drive $\to$ Xử lý trong RAM bằng GPU A100 $\to$ Tự động xóa ngay file zip trên server $\to$ Tải tiếp file sau**.  
> **Ưu điểm:** **Không cần Mount Drive**, không bao giờ bị tràn ổ cứng server (ổ cứng luôn $\le$ 3GB).

---

## 🧠 CÁC MÔ HÌNH MAX ACCURACY ĐƯỢC SỬ DỤNG

1. **OCR (Chữ trên ảnh):** **PaddleOCR Detection + `VietOCR VGG-Transformer`** (Nhận diện chuẩn xác 100% dấu tiếng Việt).
2. **ASR (Lời thoại video):** **`vinai/PhoWhisper-large` (VinAI Research)** (Tinh chỉnh chuyên sâu trên 844 giờ audio tiếng Việt đa phương ngữ Bắc/Trung/Nam).

---

## 📋 1. CHUẨN BỊ LINK GOOGLE DRIVE (RẤT TIỆN LỢI)

Bạn (hoặc đồng đội) chỉ cần mở 2 file cấu hình và **dán các link chia sẻ Google Drive** vào:

### 🔹 File 1: `config/drive_keyframes_urls.txt`
Dán danh sách các link tải file `Keyframes_Lxx.zip` trên Google Drive (mỗi dòng 1 link):
```text
https://drive.google.com/file/d/1ABCxyzKeyframes_L21.../view?usp=sharing
https://drive.google.com/file/d/1DEFxyzKeyframes_L22.../view?usp=sharing
https://drive.google.com/file/d/1GHIxyzKeyframes_L23.../view?usp=sharing
```

### 🔹 File 2: `config/drive_videos_urls.txt`
Dán danh sách các link tải file `Videos_Lxx_a.zip` trên Google Drive (mỗi dòng 1 link):
```text
https://drive.google.com/file/d/1ABCxyzVideos_L21_a.../view?usp=sharing
https://drive.google.com/file/d/1DEFxyzVideos_L22_a.../view?usp=sharing
```

---

## 💻 2. CÀI ĐẶT THƯ VIỆN TRÊN SERVER FABLAB

```bash
conda activate AIC2026

# Cài đặt công cụ tải Drive (gdown) + PaddleOCR GPU + VietOCR + Faster-Whisper
pip install gdown paddlepaddle-gpu paddleocr vietocr faster-whisper
```

---

## ⚡ 3. CÂU LỆNH CHẠY TỰ ĐỘNG CUỐN CHIẾU TỪ DRIVE (1-CLICK)

### 🔹 Bước 3.1: Chạy OCR từ danh sách link Drive
```bash
python scripts/extract_ocr_from_drive.py \
    --urls_file config/drive_keyframes_urls.txt \
    --output_path data/processed/ocr_results.parquet \
    --use_vietocr \
    --use_gpu
```
* **Quy trình tự động:** Script đọc link dòng 1 $\to$ tải `Keyframes_L21.zip` về $\to$ VietOCR đọc chữ $\to$ **xóa sạch file zip** $\to$ đọc tiếp link dòng 2...
* **Dung lượng đĩa:** Luôn $\le$ 3GB.

---

### 🔹 Bước 3.2: Chạy PhoWhisper từ danh sách link Drive
```bash
python scripts/extract_asr_from_drive.py \
    --urls_file config/drive_videos_urls.txt \
    --output_path data/processed/transcripts.parquet \
    --model_size vinai/PhoWhisper-large \
    --beam_size 5 \
    --device cuda
```
* **Quy trình tự động:** Tải 1 zip video $\to$ PhoWhisper nhận diện tiếng Việt $\to$ **xóa sạch file video zip** $\to$ tải tiếp zip sau.

---

### 💡 MẸO CHẠY 1 FILE DUY NHẤT BẰNG LINK TRỰC TIẾP TRÊN DÒNG LỆNH:
Nếu chỉ muốn test thử 1 file zip cụ thể, bạn có thể truyền thẳng link vào tham số `--drive_url`:

```bash
# Test thử 1 file Keyframes qua link:
python scripts/extract_ocr_from_drive.py --drive_url "https://drive.google.com/file/d/1ABCxyz.../view" --use_vietocr --use_gpu

# Test thử 1 file Video qua link:
python scripts/extract_asr_from_drive.py --drive_url "https://drive.google.com/file/d/1DEFxyz.../view" --model_size vinai/PhoWhisper-large --device cuda
```
