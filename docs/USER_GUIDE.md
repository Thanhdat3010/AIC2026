# 🖥️ Hướng Dẫn Sử Dụng Web Championship Console (User Guide)

Tài liệu này hướng dẫn chi tiết cách vận hành và sử dụng giao diện **Streamlit Championship Console** phục vụ công tác kiểm duyệt, tìm kiếm trực tiếp và xuất gói nộp bài cho cuộc thi **AI Challenge 2026**.

---

## 1. KHỞI ĐỘNG GIAO DIỆN

Chạy lệnh sau trên terminal:
```bash
python -m streamlit run app/streamlit_app.py --server.port 8501
```
Mở trình duyệt truy cập: `http://localhost:8501`.

---

## 2. HƯỚNG DẪN CHI TIẾT 3 CHẾ ĐỘ HOẠT ĐỘNG (3 TABS)

### 📊 Tab 1: Báo Cáo Thí Nghiệm & Ablation Leaderboard
* **Chức năng**:
  - Tự động nạp dữ liệu đo lường thực tế từ `data/benchmark/ablation_study_summary.json`.
  - Hiển thị bảng tổng sắp so sánh đối đầu 8 cấu hình thuật toán (`B1..B3`, `M1..M5`).
  - Phân tích chi tiết 47 câu kiểm thử: tỉ lệ trúng Rank 1, trúng Top 5, trúng Top 20 và các trường hợp Near-Miss.

---

### 📂 Tab 2: Duyệt & Chỉnh Sửa Kết Quả Nộp Bài (Submission Console)
* **Chức năng chính**:
  1. **Duyệt 25 Câu Sơ Tuyển 1**: Chọn câu truy vấn từ danh sách 25 câu hỏi của Ban Tổ Chức.
  2. **Trình Phát Video On-Demand (Video Player)**: Xem video trực tiếp tại đúng khung hình dự đoán để kiểm tra ngữ cảnh trước và sau khoảnh khắc.
  3. **Hiệu Chỉnh Rank 1 & Khung Hình Thời Gian Thực**:
     - Điều chỉnh vi sai $\pm 50$ khung hình bằng thanh trượt.
     - 1-Click Promote bất kỳ khung hình nào lên vị trí **Rank #1**.
  4. **Hiệu Chỉnh Đáp Án QA**:
     - Nhập hoặc sửa câu trả lời cho câu hỏi QA và áp dụng đồng loạt cho toàn bộ 100 dòng.
  5. **Hiệu Chỉnh Chuỗi Sự Kiện TRAKE**:
     - Xem và chỉnh sửa các mốc thời gian $E_1, E_2, E_3$ đảm bảo luôn thỏa mãn điều kiện $E_1 < E_2 < E_3$.
  6. **Đóng Gói & Tải submission.zip Chuẩn BTC**:
     - Nhấn nút kiểm tra tính hợp lệ toàn diện 100% của 25 file CSV.
     - Tải trực tiếp file `submission.zip` đã được đóng gói chuẩn format.

---

### 🔍 Tab 3: Tìm Kiếm Trực Tiếp (Live Multimodal Search Engine)
* **Chức năng**:
  1. **Nhập Truy Vấn Tiếng Việt**: Nhập câu mô tả tự do từ người dùng hoặc đề thi mới.
  2. **Chọn Chế Độ Task**: Auto (LLM Refiner), KIS (Khoảnh khắc), QA (Hỏi - Đáp), TRAKE (Chuỗi hành động).
  3. **Kết Quả Trực Quan**:
     - Hiển thị đáp án VLM cho câu hỏi QA.
     - Hiển thị JSON phân tích ngữ nghĩa và trọng số từ `LLMQueryRefiner`.
     - Grid View hiển thị ảnh keyframe, video ID, frame index và score.

---

## 3. QUY TRÌNH KIỂM DUYỆT & ĐÓNG GÓI NỘP BÀI SƠ TUYỂN

### 📋 Quy Chuẩn Định Dạng File CSV (Không Header, Tối đa 100 Dòng, UTF-8)
| Loại Truy Vấn | Cấu Trúc Dòng | Ví Dụ Chuẩn |
| :--- | :--- | :--- |
| **Textual KIS** | `<video_id>, <frame_idx>` | `L00_V000, 1234` |
| **Question Answering (QA)** | `<video_id>, <frame_idx>, "<answer>"` | `L01_V028, 3450, "5"` hoặc `L02_V011, 1200, "Năm người"` |
| **TRAKE ($N$ Events)** | `<video_id>, <frame_1>, <frame_2>, ..., <frame_N>` | `L10_V001, 1200, 1850, 2100` *(3 events)* |

> [!IMPORTANT]
> **Quy Tắc Đóng Gói `submission.zip`**:
> * File nén `.zip` BẮT BUỘC phải chứa thư mục `submission/` bên trong:
>   ```
>   submission.zip
>   └── submission/
>       ├── query-1-kis.csv
>       ├── query-2-qa.csv
>       ├── query-3-trake.csv
>       └── ...
>   ```
> * Tên video KHÔNG có đuôi `.mp4`.
> * Số lượng cột Frame ID trong file TRAKE phải khớp CHÍNH XÁC với số events $N$ trong đề bài.
> * Câu trả lời QA tối đa 100 ký tự. Luôn đặt dấu ngoặc kép bao quanh answer để an toàn tuyệt đối khi answer chứa dấu phẩy.

---

## 4. CHECKLIST 10 ĐIỂM KIỂM TRA TRƯỚC KHI NỘP BÀI

- [x] File có đuôi `.csv` thuần túy (không phải `.xlsx` hay `.xls`).
- [x] Không có dòng Header ở đầu file.
- [x] Mở file bằng Notepad thấy dữ liệu text phân cách bằng dấu phẩy.
- [x] Tên file khớp với tên truy vấn của BTC (`query-1-kis.csv`, `query-2-qa.csv`, ...).
- [x] Định dạng đúng theo từng loại truy vấn (KIS / QA / TRAKE).
- [x] Answer QA không quá 100 ký tự và được escape bằng dấu ngoặc kép.
- [x] TRAKE có đúng số lượng frame theo số events $N$ yêu cầu và $f(E_1) < f(E_2) < ... < f(E_N)$.
- [x] Tên video không có đuôi `.mp4`.
- [x] Đã tạo thư mục `submission/` và đặt tất cả file CSV vào đó trước khi nén thành `.zip`.
- [x] Đã kiểm tra thông báo màu xanh `Toàn bộ file CSV đều HỢP LỆ 100% chuẩn quy chế BTC` trên Tab 2 Console.

