# 🏆 AIC 2026 - SOTA Multimodal Video Retrieval System (HCMUS)

> **Dự án thi đấu AI Challenge (AIC 2026) - Đội tuyển Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM (HCMUS)**  
> **Kiến trúc Chủ lực:** 🚀 **Full 3-Layer Hierarchical Architecture + Task-Specialized Agents + Multi-Signal Blind Spot Gate (GPU Accelerated)**  
> **Điểm chuẩn kỷ lục:** 🥇 **`70.91%` BTC Final Score | `100.0%` Video Recall@10**

---

## 📌 1. TỔNG QUAN KIẾN TRÚC HỆ THỐNG (FULL 3-LAYER SOTA ARCHITECTURE)

Hệ thống được thiết kế theo mô hình **3 Tầng Phân Cấp (3-Layer Hierarchical)** kết hợp cùng các Agent chuyên biệt theo từng dạng bài toán của Ban Tổ Chức (KIS, QA, TRAKE):

```
                        ┌───────────────────────────────────────────────┐
                        │        CÂU TRUY VẤN TIẾNG VIỆT (QUERY)         │
                        └───────────────────────┬───────────────────────┘
                                                │
                                                ▼
                        ┌───────────────────────────────────────────────┐
                        │   🧠 GEMINI 2.5 FLASH LITE MULTIMODAL ROUTER  │
                        │    • Phân loại bài toán (KIS / QA / TRAKE)     │
                        │    • Dịch & Sinh 3 chiều Prompt tiếng Anh     │
                        │    • Nhận diện tín hiệu OCR / ASR / Hành động │
                        └───────────────────────┬───────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 🚀 TẦNG 1: COARSE-GRAINED MULTIMODAL RETRIEVAL (177,321 KEYFRAMES)                          │
│   • Visual Backbone: Google SigLIP-2 SO400M (1152d, Tensor Cores FP16 trên GPU RTX 3050)    │
│   • Shared Singleton Encoder: Tiết kiệm 800MB VRAM, xử lý ma trận song song                 │
│   • Lọc nhanh Top 100 Video ứng viên tiềm năng                                              │
└───────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ ⏳ TẦNG 2: INTRA-VIDEO TEMPORAL RERANKER & MULTI-CUE FUSION                                 │
│   • Gaussian Smoothing Kernel: Làm mượt đường cong phân phối thời gian dọc video            │
│   • Multi-Cue Fusion: Kết hợp trọng số Động lực học Thị giác + OCR + ASR Whisper             │
│   • Task-Specialized Agents:                                                                │
│       - 🎯 KIS Agent   : Xác định đỉnh cao trào (Visual Peak Detection)                      │
│       - ❓ QA Agent    : Gemini 2.5 Flash Lite VLM trả lời thực thể và câu hỏi chi tiết     │
│       - ⏱️ TRAKE Agent : Thuật toán Quy hoạch động Đơn điệu (Monotonic DP Alignment)       │
└───────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 🔬 TẦNG 3: DENSE VIDEO REFINER & MULTI-SIGNAL BLIND SPOT GATE                                │
│   • Multi-Signal Gate: Kiểm tra 4 tín hiệu (Động từ hành động, Khoảng mù Gap >= 75 frames,  │
│                        Gemini Semantic ASR Grounding, Đồ thị Context Plateau Dip)           │
│   • OpenCV CUDA Seek: Quét vi sai từng frame xung quanh khoảng mù trên GPU RTX 3050          │
│   • Tái định vị chính xác khung hình lọt khe đáp án đúng BTC                                │
└───────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                                │
                                                ▼
                        ┌───────────────────────────────────────────────┐
                        │   🏆 STREAMLIT REVIEW CONSOLE (3 CHẾ ĐỘ)      │
                        │    • Soi ma trận ảnh Top 10 trực quan         │
                        │    • Đổi ngôi 1-Click Promote to Rank #1      │
                        │    • Micro-Slider vi chỉnh ±50 khung hình     │
                        │    • Đóng gói AIC2026_submission.zip chuẩn BTC│
                        └───────────────────────────────────────────────┘
```

---

## 🏆 2. BẢNG TỔNG SẮP KỶ LỤC SOTA (ABLATION BENCHMARK RESULTS)

Đánh giá thực nghiệm trực tiếp trên bộ 11 Test Cases chuẩn của Ban Tổ Chức (`data/benchmark/ground_truth.json`):

| Chỉ số Đánh giá | Cấu hình 11 (Baseline) | Cấu hình 15 | 🚀 CẤU HÌNH 16 (HIỆN TẠI) | Mức độ Tăng trưởng |
| :--- | :---: | :---: | :---: | :---: |
| 🎯 **KIS Score** | `0.6571` | `0.6857` | **`0.7143`** | **+8.7%** 🚀 |
| ❓ **QA Score** | `0.8000` | `0.7000` | **`0.8000`** | **+14.3%** 🚀 |
| ⏱️ **TRAKE Score** | `0.2400` | `0.4400` | **`0.6000`** | **+150.0%** 🚀🚀 |
| 🏆 **BTC FINAL SCORE** | **`0.6073` (60.73%)** | **`0.6436` (64.36%)** | 🥇 **`0.7091` (70.91%)** | **+16.8% KỶ LỤC CAO NHẤT** 🎉 |
| 🔍 **Video Recall@5** | 81.8% | 81.8% | **`90.9%`** | **+11.1%** |
| 🔍 **Video Recall@10** | 90.9% | 90.9% | 🎯 **`100.0%` (11/11 Video)** | **HOÀN HẢO 100%** |
| 🔍 **Video Recall@20** | 90.9% | 90.9% | 🎯 **`100.0%` (11/11 Video)** | **HOÀN HẢO 100%** |

---

## 📂 3. CẤU TRÚC THƯ MỤC DỰ ÁN (PROJECT STRUCTURE)

```text
d:\HCMUS\AIC2026\
├── app/
│   └── streamlit_app.py          # Giao diện Review Console 3 Chế độ
├── data/
│   ├── batch_1/processed/        # Dữ liệu 177,321 keyframe embeddings & features
│   └── benchmark/
│       ├── ground_truth.json     # 11 câu kiểm chuẩn chính thức BTC
│       └── latest_ablation_results.json # Kết quả đo lường benchmark động
├── docs/
│   ├── ABLATION_STUDY_RESULTS.md # Báo cáo chi tiết từng lần chạy thực nghiệm
│   └── ABLATION_LEADERBOARD.md   # Bảng xếp hạng các cấu hình
├── output/
│   └── batch_1/                  # 24 file CSV kết quả thi đấu chuẩn BTC
├── query/
│   └── batch_1/query-p1-groupA/  # 24 câu hỏi đề thi chính thức Batch 1
├── raw/
│   └── batch_1/                  # Video Zips, Keyframe Zips, Audio, OCR gốc
├── scripts/
│   ├── data_processing/          # Xử lý dữ liệu (extract_features, build_index, v.v.)
│   ├── evaluation/               # Đánh giá và kiểm chuẩn AI (evaluate_ablation, audit_gt)
│   ├── submission/               # Tạo và kiểm tra file nộp bài chuẩn BTC
│   └── verification/             # Kiểm tra tính toàn vẹn dữ liệu và môi trường
├── src/
│   ├── evaluation/btc_metric.py  # Công thức tính điểm BTC chính thống
│   ├── query/
│   │   ├── gemini_router.py      # Bộ định tuyến AI & Phân rã truy vấn
│   │   └── text_encoder.py       # Bộ mã hóa SigLIP-2 Text Encoder
│   ├── reranking/
│   │   ├── blind_spot_gate.py    # Cổng phát hiện vùng mù đa tín hiệu
│   │   ├── dense_video_refiner.py# Kính lúp vi sai GPU CUDA RTX 3050
│   │   └── intra_video_reranker.py # Intra-video temporal smoother
│   ├── retrieval/
│   │   ├── keyframe_loader.py    # Nạp ảnh Keyframe siêu tốc từ Zip
│   │   └── task_specialized_engine.py # Tầng điều phối truy xuất đa tác vụ
│   ├── submission/
│   │   └── submission_validator.py # Bộ kiểm chuẩn và đóng gói ZIP nộp bài
│   └── tasks/
│       ├── qa_agent.py           # Agent trả lời câu hỏi thị giác (QA)
│       └── trake_agent.py        # Agent quy hoạch động chuỗi sự kiện (TRAKE)
├── .env                          # Chứa Gemini API Key Pool
└── requirements.txt              # Danh sách thư viện phụ thuộc
```

---

## ⚡ 4. HƯỚNG DẪN CÀI ĐẶT & KÍCH HOẠT GPU (SETUP)

### 1️⃣ Khởi tạo môi trường Conda:
```bash
conda create -n AIC2026 python=3.11 -y
conda activate AIC2026
```

### 2️⃣ Cài đặt PyTorch hỗ trợ GPU CUDA 12.4 (NVIDIA RTX 3050):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 3️⃣ Thiết lập Gemini API Keys:
Tạo hoặc cập nhật file `.env` tại thư mục gốc:
```env
GEMINI_API_KEYS="AIzaSy...,AIzaSy...,AIzaSy..."
```

---

## 🎮 5. HƯỚNG DẪN VẬN HÀNH DỰ ÁN

### 🧪 A. Chạy Thử Nghiệm & Đo Điểm Chuẩn (Ablation Benchmark):
Để đánh giá độ chính xác trên 11 test cases chuẩn BTC:
```bash
python scripts/evaluation/evaluate_ablation.py --config 16
```
> Kết quả đo lường và ma trận điểm số sẽ tự động lưu vào `data/benchmark/latest_ablation_results.json` và `docs/ABLATION_STUDY_RESULTS.md`.

---

### 🚀 B. Chạy Toàn Bộ 24 Câu Đề Thi BTC (Batch Run):
Để chạy mô hình AI tự động trên toàn bộ 24 câu hỏi của đợt thi:
```bash
python scripts/submission/generate_official_batch1.py
```
> Toàn bộ 24 file kết quả CSV sẽ được lưu tự động tại `output/batch_1/*.csv`.

---

### 🔍 C. Kiểm Chuẩn Định Dạng File Nộp Bài BTC:
```bash
python scripts/submission/validate_submission.py
```

---

### 🖥️ D. Mở Giao Diện Tương Tác Streamlit Review Console:
```bash
streamlit run app/streamlit_app.py
```
Truy cập trình duyệt tại: 👉 **`http://localhost:8501`**

#### 🎛️ 3 Chế độ hoạt động trên giao diện:
1. **📊 Chế độ 1: Benchmark & Ground Truth (11 Câu Đã Eval):**
   - Đọc kết quả động từ lần chạy mới nhất.
   - Cho phép chọn từng câu (như `test-kis-08`), soi Top 10 ảnh thực tế, bấm **⭐ Đưa lên #1** và kéo thanh trượt vi sai để thấy điểm số nhảy lên `1.0000`!
2. **📂 Chế độ 2: Đề Thi Chính Thức BTC (24 Câu Batch 1):**
   - Xem toàn bộ 24 câu đề thi kèm ma trận Top 10 ảnh dự đoán.
   - Thao tác đổi ngôi Rank 1, vi chỉnh khung hình và bấm **1-Click Tải Gói Nộp Bài (`AIC2026_submission.zip`)**.
3. **🔍 Chế độ 3: Tìm Kiếm Trực Tiếp (Live Search):**
   - Nhập truy vấn đa phương thức tùy ý để tìm kiếm thời gian thực.

---

## 👥 THÀNH VIÊN ĐỘI THI
- **Đội tuyển:** HCMUS AIC 2026  
- **Cơ quan:** Trường Đại học Khoa học Tự nhiên, Đại học Quốc gia TP. Hồ Chí Minh
