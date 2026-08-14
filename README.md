# 🎬 AIC 2026 - Multimodal Video Retrieval System (Known-Item Search Engine)

Hệ thống tìm kiếm video đa phương thức toàn diện cho cuộc thi **AI Challenge (AIC) TP. Hồ Chí Minh 2026**, được thiết kế để giải quyết 3 nhiệm vụ chính:
1. **Task 1 - Textual Known Item Search (KIS):** Định vị chính xác khung hình video dựa trên miêu tả văn bản tự nhiên.
2. **Task 2 - Visual Question Answering (QA):** Tìm kiếm phân cảnh liên quan và trả lời câu hỏi trực quan.
3. **Task 3 - Temporal Retrieval & Alignment (TRAKE):** Truy xuất video và căn chỉnh chuỗi sự kiện tuần tự theo thời gian.

---

## 📑 Mục Lục
- [1. Cấu Trúc Cây Thư Mục & Thiết Lập Dữ Liệu](#1-cấu-trúc-cây-thư-mục--thiết-lập-dữ-liệu)
- [2. Cài Đặt Môi Trường (Installation)](#2-cài-đặt-môi-trường-installation)
- [3. Quy Trình Khởi Chạy Nhanh (Quick Start)](#3-quy-trình-khởi-chạy-nhanh-quick-start)
- [4. Hướng Dẫn Sử Dụng Chi Tiết](#4-hướng-dẫn-sử-dụng-chi-tiết)
  - [A. Giao diện Interactive UI (Streamlit)](#a-giao-diện-interactive-ui-streamlit)
  - [B. Chạy đơn lẻ từ Terminal (CLI Search)](#b-chạy-đơn-lẻ-từ-terminal-cli-search)
  - [C. Chạy Batch toàn bộ đề thi & Xuất file nộp bài](#c-chạy-batch-toàn-bộ-đề-thi--xuất-file-nộp-bài)
- [5. Kiến Trúc Pipeline & Giải Thuật](#5-kiến-trúc-pipeline--giải-thuật)
- [6. Chuẩn Định Dạng File Nộp Bài (Submission Format)](#6-chuẩn-định-dạng-file-nộp-bài-submission-format)

---

## 1. Cấu Trúc Cây Thư Mục & Thiết Lập Dữ Liệu

Khi bạn tải (clone) mã nguồn này về, cấu trúc thư mục chuẩn được tổ chức như sau:

```text
AIC2026/
├── app/
│   └── streamlit_app.py         # Giao diện Web tương tác tìm kiếm trực quan
├── config/
│   └── config.yaml              # File cấu hình trung tâm (đường dẫn, model, trọng số)
├── docs/                        # Tài liệu thể lệ và cấu trúc dữ liệu của BTC
├── raw/                         # [QUAN TRỌNG] Nơi đặt các file nén dữ liệu từ BTC
│   ├── clip-features-32-aic25-b1.zip
│   ├── map-keyframes-aic25-b1.zip
│   ├── media-info-aic25-b1.zip
│   └── objects-aic25-b1.zip
├── data/
│   └── processed/               # Nơi lưu Parquet & NumPy sau khi tiền xử lý (tự sinh)
│       ├── frames.parquet       # Bảng ánh xạ 177.321 keyframes sang frame_idx gốc
│       ├── videos.parquet       # Bảng metadata 873 video YouTube
│       └── clip_features.npy    # Ma trận vector CLIP (177321, 512)
├── indexes/
│   └── clip.faiss               # Vector Database FAISS IndexFlatIP (tự sinh)
├── query/                       # Chứa các file câu hỏi truy vấn của BTC
│   └── query-p1-groupA/         # Tập câu hỏi mẫu (*-kis.txt, *-qa.txt, *-trake.txt)
├── outputs/
│   └── submission/              # Nơi tự động xuất các file CSV nộp bài (*.csv)
├── scripts/                     # Các script thực thi dòng lệnh (pre-processing, indexing, search)
├── src/                         # Mã nguồn module hóa cốt lõi (query, retrieval, reranking, submission)
├── tasks/                       # Quản lý runner cho 3 task riêng biệt: KIS, QA, TRAKE
├── tests/                       # Bộ kiểm thử tự động (Pytest test suite)
├── requirements.txt             # Danh sách thư viện Python
└── README.md
```

> **Lưu ý về dữ liệu:** Do các thư mục `raw/`, `data/`, `indexes/`, `Keyframes/`, `Videos/` có dung lượng lớn nên đã được cấu hình trong `.gitignore` để không đưa lên GitHub. Bạn chỉ cần tải dữ liệu BTC bỏ vào `raw/` và chạy 2 lệnh bên dưới để tự động tạo lại toàn bộ dữ liệu.

---

## 2. Cài Đặt Môi Trường (Installation)

Yêu cầu hệ thống: **Python 3.11** và **Conda** (Windows / Linux / macOS).

```bash
# 1. Tạo môi trường Conda mới
conda create -n AIC2026 python=3.11 -y
conda activate AIC2026

# 2. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

---

## 3. Quy Trình Khởi Chạy Nhanh (Quick Start)

Chỉ cần 3 bước để dựng hoàn chỉnh toàn bộ hệ thống từ đầu:

### Bước 1: Đặt các file ZIP của BTC vào thư mục `raw/`
Đảm bảo thư mục `raw/` chứa đủ 4 file nén cơ bản:
* `clip-features-32-aic25-b1.zip`
* `map-keyframes-aic25-b1.zip`
* `media-info-aic25-b1.zip`
* `objects-aic25-b1.zip`

### Bước 2: Tiền xử lý dữ liệu sang Apache Parquet & NumPy
```bash
python scripts/preprocess_all.py
```
*(Chỉ mất khoảng 10 giây để đọc, trích xuất và tối ưu hóa 177.321 khung hình sang Parquet).*

### Bước 3: Xây dựng Vector Database Index bằng FAISS
```bash
python scripts/build_index.py
```
*(Tạo file `indexes/clip.faiss` với 177.321 vector L2-normalized 512 chiều).*

---

## 4. Hướng Dẫn Sử Dụng Chi Tiết

### A. Giao diện Interactive UI (Streamlit)
Khởi động giao diện web đồ họa trực quan trên trình duyệt:
```bash
streamlit run app/streamlit_app.py
```
👉 Truy cập: `http://localhost:8501` để nhập câu hỏi tiếng Việt, tùy chỉnh Top-K, số frame/video và xem kết quả xếp hạng.

---

### B. Chạy đơn lẻ từ Terminal (CLI Search)
Tìm kiếm trực tiếp từ dòng lệnh cho một câu miêu tả:
```bash
python scripts/run_kis.py --query "một con chó màu vàng đang chạy trên cỏ"
```

---

### C. Chạy Batch toàn bộ đề thi & Xuất file nộp bài
Hệ thống hỗ trợ chạy hàng loạt toàn bộ file câu hỏi và tự động xuất các file CSV nộp bài vào `outputs/submission/`:

```bash
# Chạy riêng Task 1 (Textual KIS)
python scripts/run_task.py --task kis --query_dir query/query-p1-groupA

# Chạy riêng Task 2 (Visual QA)
python scripts/run_task.py --task qa --query_dir query/query-p1-groupA

# Chạy riêng Task 3 (TRAKE)
python scripts/run_task.py --task trake --query_dir query/query-p1-groupA

# Hoặc chạy toàn bộ cả 3 Task cùng lúc
python scripts/run_task.py --task all --query_dir query/query-p1-groupA
```

---

## 5. Kiến Trúc Pipeline & Giải Thuật

Hệ thống được thiết kế theo kiến trúc **Multimodal Hybrid Retrieval**:

1. **Query Decomposition:** Tách câu truy vấn phức tạp thành nhiều vế (cues) về chủ thể, hành động, bối cảnh.
2. **Bi-Lingual Embedding Fusion:** Kết hợp mô hình dịch thuật `Helsinki-NLP/opus-mt-vi-en` và `CLIP ViT-B/32` để mã hóa câu hỏi thành vector đặc trưng không gian đa ngôn ngữ.
3. **Dense Vector Retrieval:** Tìm kiếm siêu tốc qua thuật toán `faiss.IndexFlatIP` trên 177.321 vector khung hình.
4. **Cue Coverage & Video Aggregation:** Đánh giá mức độ bao phủ các vế của câu hỏi trên từng video và chọn lọc $N=2$ khung hình tiêu biểu nhất cho mỗi video.
5. **Metadata Reranking:** So khớp từ khóa với tiêu đề và mô tả YouTube qua mô hình TF-IDF.
6. **Lookup Frame ID:** Tra cứu tức thì từ bảng `frames.parquet` để đổi `keyframe_id` sang số thứ tự `frame_idx` gốc trong video.

---

## 6. Chuẩn Định Dạng File Nộp Bài (Submission Format)

Tất cả các file kết quả xuất ra trong thư mục `outputs/submission/` đều tuân thủ nghiêm ngặt 100% quy định của BTC:

| Task | Tên file xuất ra | Định dạng từng dòng (Không header, tối đa 100 dòng) | Ví dụ dòng nộp bài |
| :--- | :--- | :--- | :--- |
| **1. KIS** | `query-<id>-kis.csv` | `<video_id>,<frame_idx>` | `L21_V001,261` |
| **2. QA** | `query-<id>-qa.csv` | `<video_id>,<frame_idx>,<answer>` | `L05_V005,888,màu xanh` |
| **3. TRAKE** | `query-<id>-trake.csv` | `<video_id>,<frame_1>,<frame_2>,...,<frame_n>` | `L10_V010,101,156,203,251` |

---

## 🧪 Kiểm Thử Tự Động (Unit Tests)
Chạy bộ test suite để đảm bảo toàn bộ hệ thống hoạt động ổn định:
```bash
pytest tests/ -v
```

---
*Chúc đội thi đạt kết quả xuất sắc nhất tại AI Challenge 2026! 🚀*
