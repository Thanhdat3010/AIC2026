# 🚀 HƯỚNG DẪN MÔI TRƯỜNG CONDA, CHẠY CONFIG & STREAMLIT (AIC 2026)

---

## 🐍 1. CONTEXT MÔI TRƯỜNG CONDA

* **Tên môi trường Conda:** `AIC2026`
* **Đường dẫn Python thực thi:** `C:\Users\Lenovo\miniconda3\envs\AIC2026\python.exe`
* **Lệnh kích hoạt môi trường:**
  ```powershell
  conda activate AIC2026
  ```
* **Chuyển về thư mục dự án:**
  ```powershell
  cd d:\HCMUS\AIC2026
  ```

---

## ⚙️ 2. CONTEXT CÁCH CHẠY CÁC CẤU HÌNH (CONFIGS)

### 2.1. Lệnh Sinh Gói Nộp Bài (Submission Generator):
* **Cấu hình SOTA Quán Quân (Khuyên dùng):** **`Config 25`** (Tier-3 WRRF - Macro Score `0.5532`, KIS `0.6500`, Video Recall@100 `97.9%`).
* **Cú pháp lệnh chạy:**
  ```powershell
  conda activate AIC2026
  python scripts/submission/run_submission.py --input query/THUNGHIEM-bo-de-thi --output_dir output/thunghiem --config 25
  ```
* **Các tham số:**
  * `--input`: Đường dẫn thư mục chứa các file đề thi `.txt` (ví dụ: `query/THUNGHIEM-bo-de-thi` hoặc thư mục đề thi mới của BTC).
  * `--output_dir`: Đường dẫn thư mục xuất kết quả bên trong `output/` (ví dụ: `output/thunghiem`, `output/chinhthuc`).
  * `--config`: Mã cấu hình cần chạy (`25` là SOTA tốt nhất, hoặc `22`, `24`, `26`).
  * `--top_k`: Số lượng dòng dự đoán cho mỗi câu (Mặc định: `100`).

* **Kết quả đầu ra:**
  * File CSV từng câu: `output/<tên_thư_mục>/submission/<query_id>.csv` (Mỗi file đúng 100 dòng).
  * File zip nộp bài: `output/<tên_thư_mục>/submission.zip`.

### 2.2. Lệnh Đo Lường & Đánh Giá Kiểm Chuẩn (Ablation Benchmark):
```powershell
conda activate AIC2026
python scripts/evaluation/evaluate_ablation.py --config 25
```

---

## 🖥️ 3. CÁCH CHẠY GIAO DIỆN STREAMLIT CONSOLE

### 3.1. Lệnh khởi động Streamlit:
```powershell
conda activate AIC2026
python -m streamlit run app/streamlit_app.py --server.port 8501
```

### 3.2. Truy cập ứng dụng:
* **Địa chỉ duyệt web:** **`http://localhost:8501`**

### 3.3. Các chức năng chính trên Web Console:
1. **Tab 1 - Báo Cáo Thí Nghiệm & Leaderboard:** Xem lại bảng so sánh điểm các config (22 $\rightarrow$ 26) trên 47 câu test.
2. **Tab 2 - Duyệt & Chỉnh Sửa Kết Quả Nộp Bài (Submission Console):**
   * Chọn gói kết quả trong `output/` (ví dụ: `thunghiem`).
   * Soi Top 10 hình ảnh ứng viên của từng câu.
   * Đổi ngôi **Rank 1 bằng 1-Click**, sửa text đáp án QA, kéo chỉnh chuỗi thời gian TRAKE.
   * Nút **`⚡ Chạy Lại Riêng Câu Này Trên GPU (SOTA Engine)`**: Chạy lại riêng 1 câu KIS/QA trên GPU với độ chính xác cao nhất.
   * Hệ thống **tự động đồng bộ file CSV và cập nhật `submission.zip` tức thì** khi bạn chỉnh sửa.
3. **Tab 3 - Tìm Kiếm Trực Tiếp (Live Search):** Nhập text tùy ý để truy vấn nhanh frame video trên GPU.
