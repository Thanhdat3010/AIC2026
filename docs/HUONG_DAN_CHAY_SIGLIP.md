# 🚀 HƯỚNG DẪN TRÍCH XUẤT SIGLIP TRÊN SERVER BẰNG TMUX

---

## 📥 1. CẬP NHẬT CODE MỚI NHẤT TRÊN SERVER
```bash
git pull origin main
```

---

## ⚡ 2. CÂU LỆNH CHẠY BẰNG TMUX TRÊN GPU A100

### 🔹 Bước 1: Tạo phiên làm việc Tmux
```bash
tmux new -s siglip
```

### 🔹 Bước 2: Kích hoạt Conda và Chạy Trích Xuất (Lệnh 1 dòng)
```bash
conda activate AIC2026
python scripts/data_processing/extract_visual_features.py --urls_file config/drive_keyframes_urls.txt --model_name google/siglip-so400m-patch14-384 --output_path data/batch_1/processed/siglip_features.npy --batch_size 128 --device cuda
```

### 🔹 Bước 3: Thoát ra ngoài an toàn (để script tự chạy ngầm)
* Nhấn tổ hợp phím: `Ctrl + B` sau đó nhấn phím `D`.

---

## 🔍 3. CÁC LỆNH QUẢN LÝ TMUX HỮU ÍCH

### 🔹 Quay lại xem tiến trình đang chạy:
```bash
tmux attach -t siglip
```

### 🔹 Nếu bị cúp điện / gián đoạn mạng (Tự động Resume chạy tiếp):
```bash
tmux attach -t siglip || tmux new -s siglip
python scripts/data_processing/extract_visual_features.py --urls_file config/drive_keyframes_urls.txt --model_name google/siglip-so400m-patch14-384 --output_path data/batch_1/processed/siglip_features.npy --batch_size 128 --device cuda
```

---

## 📦 4. ĐÓNG GÓI FAISS INDEX (KHI CHẠY XONG FILE .NPY)
```bash
python scripts/data_processing/build_index.py --features data/batch_1/processed/siglip_features.npy --out indexes/batch_1/siglip-so400m.faiss
```
*(Quá trình đóng gói FAISS chỉ mất khoảng 3 đến 5 giây).*
