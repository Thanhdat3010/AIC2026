# 🏆 HƯỚNG DẪN TRÍCH XUẤT OCR & WHISPER (TẢI TRỰC TIẾP TỪ MÁY CHỦ BTC HOẶC DRIVE)

> **Cơ chế đỉnh cao:** Script tự động **Tải trực tiếp qua Link BTC (`aic-data.ledo.io.vn`) $\to$ Xử lý trong RAM bằng GPU A100 $\to$ Tự động xóa ngay file zip trên server $\to$ Tải tiếp file sau**.  
> **Cơ chế chịu lỗi (Fault-Tolerant):** 
> - **OCR:** Checkpoint định kỳ sau mỗi gói ZIP.
> - **PhoWhisper:** **Checkpoint ngay sau MỖI VIDEO**, xử lý xong video nào là lưu đĩa ngay video đó, không bao giờ lo mất dữ liệu dù bị tắt máy bất ngờ!
> - Tăng tốc **3-4x** bằng `BatchedInferencePipeline` (Batch Size = 16 trên GPU A100).

---

## 🧠 CÁC MÔ HÌNH MAX ACCURACY ĐƯỢC SỬ DỤNG

1. **OCR (Chữ trên ảnh):** **CRAFT (EasyOCR) + `VietOCR VGG-Transformer`** (Nhận diện chuẩn xác 100% dấu tiếng Việt trên PyTorch GPU).
2. **ASR (Lời thoại video):** **`vinai/PhoWhisper-large` (VinAI Research)** (Tinh chỉnh chuyên sâu trên 844 giờ audio tiếng Việt + Batched Pipeline).

---

## 💻 1. CÀI ĐẶT THƯ VIỆN TRÊN SERVER FABLAB

```bash
conda activate AIC2026
pip install easyocr vietocr faster-whisper requests gdown "numpy<2.0.0" "setuptools<70.0.0"
```

---

## ⚡ 2. CÂU LỆNH CHẠY TRONG TMUX (SONG SONG OCR & WHISPER)

### 🔹 Session 1: Chạy OCR (Cửa sổ 1)
```bash
tmux new -s ocr
conda activate AIC2026
python scripts/data_processing/extract_ocr_from_drive.py --urls_file config/drive_keyframes_urls.txt --output_path data/batch_1/processed/ocr_results.parquet --use_vietocr --use_gpu
```
*(Thoát ra ngoài: Nhấn `Ctrl + B` rồi bấm `D`)*.

---

### 🔹 Session 2: Chạy PhoWhisper ASR (Cửa sổ 2)
```bash
tmux new -s asr
conda activate AIC2026
python scripts/data_processing/extract_asr_from_drive.py --urls_file config/drive_videos_urls.txt --output_path data/batch_1/processed/transcripts.parquet --model_size vinai/PhoWhisper-large --batch_size 16 --beam_size 5 --device cuda
```
*(Thoát ra ngoài: Nhấn `Ctrl + B` rồi bấm `D`)*.

---

## 🔄 3. CƠ CHẾ TIẾP TỤC CHẠY KHI BỊ HỎNG / GIÁN ĐOẠN (RESUME)

### 🌟 Cách 1: Tự Động Resume Thông Minh (Không cần làm gì cả)
Chỉ cần chạy lại y nguyên câu lệnh cũ:
* Script sẽ tự động nạp file parquet cũ.
* Tự động **bỏ qua 100% các video đã làm xong** và chỉ xử lý tiếp những video còn thiếu!

---

### 🌟 Cách 2: Chọn chạy từ một gói cụ thể bằng `--start_from`
```bash
# OCR chạy tiếp từ Keyframes_L25.zip:
python scripts/data_processing/extract_ocr_from_drive.py --start_from Keyframes_L25.zip --use_vietocr --use_gpu

# PhoWhisper chạy tiếp từ Videos_L25_a.zip:
python scripts/data_processing/extract_asr_from_drive.py --start_from Videos_L25_a.zip --model_size vinai/PhoWhisper-large --device cuda
```

---

### 🌟 Cách 3: Chọn chạy từ số thứ tự gói bằng `--start_index`
```bash
# OCR chạy từ gói 5:
python scripts/data_processing/extract_ocr_from_drive.py --start_index 5 --use_vietocr --use_gpu

# PhoWhisper chạy từ gói 5:
python scripts/data_processing/extract_asr_from_drive.py --start_index 5 --model_size vinai/PhoWhisper-large --device cuda
```
