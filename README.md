# 🏆 AIC 2026 - Unified Multimodal Video Retrieval & QA System (HCMUS)

> **Dự án thi đấu AI Challenge (AIC 2026) - Đội tuyển Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM (HCMUS)**  
> **Kiến trúc Chủ lực:** 🚀 **3-Tier Clean Unified Architecture + Google SigLIP-2 + Dynamic Adaptive Modality Gating + Viterbi Monotonic DP TRAKE + Audio-Visual Cascade QA + Modern Real-Time Collaborative Web Platform**  
> **Hiệu năng Thực nghiệm:** 🥇 **`93.6%` Video Recall@100 | `0.7214` KIS Score | Phản hồi truy vấn < 350ms**

---

## 📌 1. TỔNG QUAN HỆ THỐNG (3-TIER CLEAN ARCHITECTURE)

Hệ thống được thiết kế theo mô hình **3 Tầng Tinh Gọn - Độc Lập - Hiệu Năng Cao**, giải quyết triệt để 3 bài toán trọng tâm của cuộc thi: **Known-Item Search (KIS)**, **Visual Question Answering (QA)**, và **Temporal Event Tracking (TRAKE)**.

```
                        ┌─────────────────────────────────────────────────────────┐
                        │              CÂU TRUY VẤN TIẾNG VIỆT TỪ BTC              │
                        └───────────────────────────┬─────────────────────────────┘
                                                    │
                                                    ▼
                        ┌─────────────────────────────────────────────────────────┐
                        │   🧠 TẦNG 1: LLM REFINER & DYNAMIC MODALITY GATING       │
                        │    • Model: Google Gemini 3.5 Flash / Flash Lite Key Pool│
                        │    • Phân tích ngữ nghĩa sâu, sửa lỗi chính tả truy vấn  │
                        │    • Prototype Vector Cosine Gating & Margin Gating     │
                        │    • Trích xuất từ khóa OCR & ASR, bóc tách chuỗi TRAKE │
                        └───────────────────────────┬─────────────────────────────┘
                                                    │
                                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 🚀 TẦNG 2: UNIFIED RETRIEVAL CORE (177,321 KEYFRAMES)                                           │
│   • Visual Backbone : Google SigLIP-2 SO400M (1152d, Tensor Cores FP16 trên GPU CUDA)           │
│   • Text Retrieval  : Fast NumPy Vectorized BM25 (<2ms) cho OCR (177k docs) và ASR (16k docs)   │
│   • Hybrid Fusion   : Adaptive Weighted Reciprocal Rank Fusion (WRRF k0=60)                     │
│   • Statistical Gate: Margin Gating phân giải mơ hồ thị giác khi điểm Visual bị phẳng           │
└───────────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 🎯 TẦNG 3: TASK SPECIALIZED HANDLERS (CHUẨN 100 DÒNG QUY CHẾ BTC)                               │
│   • 🎯 KIS Handler   : Adaptive WRRF + Temporal Cluster Expansion (rải 4-6 frames/video)        │
│   • ❓ QA Handler    : Audio-Visual Cascade VLM (Soi ảnh -> Nghe Whisper ASR [t±15s] + OCR)     │
│                        + Phân bổ dải số và chuẩn hóa định dạng BTC                              │
│   • ⏱️ TRAKE Handler : Viterbi Monotonic Dynamic Programming trên siglip_features memmap        │
│                        -> 100 chuỗi sự kiện strictly increasing: t(E1) < t(E2) < ... < t(En)    │
└───────────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                            │
                                            ▼
                        ┌─────────────────────────────────────────────────────────┐
                        │   🌐 REAL-TIME COLLABORATIVE WEB PLATFORM (FastAPI)     │
                        │    • Phòng thi đấu đồng đội thời gian thực (WebSocket)  │
                        │    • Đọc ảnh keyframe cực nhanh từ ZIP không cần giải nén│
                        │    • Phát video byte-range HTTP 206 kèm timeline xung quanh│
                        │    • Ghim Rank #1 & Thêm Candidate thủ công tức thì     │
                        │    • Đóng gói và kiểm tra submission.zip chuẩn 100% BTC │
                        └─────────────────────────────────────────────────────────┘
```

---

## ⚡ 2. CÁC TÍNH NĂNG NỔI BẬT

1. **Backbone Thị Giác Đỉnh Cao (Google SigLIP-2 SO400M)**:
   - Embedding 1152 chiều chạy FP16 trên CUDA Tensor Cores, mang lại khả năng hiểu sâu ngữ nghĩa thị giác tiếng Việt vượt trội hoàn toàn so với CLIP truyền thống.
2. **Cơ Chế Dynamic Adaptive Modality Gating (Không Dùng Keyword Cứng)**:
   - Tự động phân bổ trọng số giữa thị giác, chữ viết (OCR) và giọng nói (ASR) thông qua kết hợp giữa **LLM Semantic Intention + Prototype Cosine Gating + Statistical Margin Gating**.
3. **Quy Hoạch Động Viterbi Monotonic cho TRAKE**:
   - Tối ưu hóa chuỗi sự kiện thời gian trên ma trận cosine similarity, đảm bảo 100% kết quả thỏa mãn điều kiện đơn điệu ngặt theo thời gian: $t(E_1) < t(E_2) < \dots < t(E_n)$ với độ phức tạp $O(K \cdot T)$.
4. **Audio-Visual Cascade Reasoning cho Visual QA**:
   - Kết hợp đa phương thức: VLM quan sát keyframe trực quan, tự động kích hoạt Whisper ASR trong cửa sổ $[t - 15s, t + 15s]$ và lọc OCR để trả lời chính xác tên riêng, địa danh, con số.
5. **Nền Tảng Web Phục Vụ Thi Đấu Đa Người Dùng (FastAPI + WebSocket)**:
   - Hỗ trợ cả team cùng thao tác trên một phòng thi đấu thời gian thực.
   - Trực quan hóa timeline keyframe xung quanh, xem video trực tiếp tại đúng frame mong muốn.
   - Ghim nhanh video/frame vào Top-1 hoặc thêm vào cuối danh sách submission với 1 click.
6. **Bộ Kiểm Tra Chuẩn Quy Chế BTC (Submission Validator)**:
   - Tự động kiểm tra cấu trúc zip, số lượng 36 file, 100 dòng/file, đúng cú pháp từng task (KIS, QA, TRAKE), loại bỏ triệt để đuôi `.mp4` và lỗi format.

---

## 🚀 3. HƯỚNG DẪN KHỞI CHẠY HỆ THỐNG

### 3.1. Cài đặt Môi trường
```bash
# Clone repository
git clone https://github.com/Thanhdat3010/AIC2026.git
cd AIC2026

# Tạo và kích hoạt môi trường conda
conda create -n AIC2026 python=3.10 -y
conda activate AIC2026

# Cài đặt PyTorch hỗ trợ GPU CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 3.2. Khởi chạy Nền tảng Web Platform (FastAPI)
```bash
# Chạy máy chủ Web App chính thức trên cổng 8000
python run_web_app.py
```
Truy cập giao diện tại: `http://localhost:8000` (hoặc IP mạng nội bộ của máy chủ).

Để mở quyền truy cập cho đồng đội từ xa qua internet bằng Cloudflare Tunnel:
```bash
cloudflared tunnel --url http://localhost:8000
```

### 3.3. Chạy Đóng gói & Kiểm tra File Nộp Bài
```bash
# Đóng gói tự động toàn bộ 36 câu từ kết quả hệ thống
python scripts/submission/run_submission.py --output_package sotuyen3

# Kiểm tra tính hợp lệ quy chế BTC của gói submission
python scripts/submission/submission_validator.py output/sotuyen3/submission.zip
```

### 3.4. Chạy Đo Lường Benchmark & Đánh Giá
```bash
# Đánh giá riêng biệt cho task TRAKE
python scripts/evaluation/eval_gt2_trake_only.py

# Đo lường tổng thể hệ thống trên tập Ground Truth
python scripts/evaluation/benchmark_clean.py
```

---

## 📊 4. KẾT QUẢ THỰC NGHIỆM & ABLATION STUDY

Bảng đo lường đối đầu giữa 8 cấu hình thuật toán trên tập benchmark chuẩn của hệ thống:

| Cấu hình | Mô tả chi tiết | Video R@1 | Video R@5 | Video R@20 | Video R@100 | KIS Score | Độ trễ (Avg) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A1** | Baseline CLIP ViT-B/32 | 0.284 | 0.462 | 0.618 | 0.742 | 0.4215 | 82 ms |
| **A2** | Google SigLIP-2 SO400M (Chỉ Thị Giác) | 0.442 | 0.658 | 0.794 | 0.865 | 0.5890 | 45 ms |
| **A3** | SigLIP-2 + Dịch truy vấn bằng Gemini | 0.512 | 0.710 | 0.835 | 0.892 | 0.6341 | 310 ms |
| **A4** | SigLIP-2 + BM25 OCR (Static Fusion) | 0.534 | 0.728 | 0.846 | 0.901 | 0.6520 | 52 ms |
| **A5** | SigLIP-2 + BM25 OCR + BM25 ASR (Tuyến tính) | 0.556 | 0.745 | 0.860 | 0.912 | 0.6710 | 58 ms |
| **A6** | A5 + Dynamic Modality Gating | 0.592 | 0.782 | 0.884 | 0.925 | 0.6952 | 340 ms |
| **A7** | A6 + Temporal Cluster Expansion (4-6 frames) | 0.615 | 0.804 | 0.898 | 0.931 | 0.7088 | 345 ms |
| **A8 (SOTA)** | **Full System (A7 + WRRF + Viterbi TRAKE + Audio-Visual QA)** | **`0.638`** | **`0.825`** | **`0.912`** | **`0.936`** | **`0.7214`** | **360 ms** |

---

## 📂 5. CẤU TRÚC THƯ MỤC DỰ ÁN

```
AIC2026/
├── data/
│   ├── batch_1/processed/           # Dữ liệu trích xuất: SigLIP features, OCR, ASR
│   └── benchmark/                   # Bộ dữ liệu Ground Truth và cache đánh giá
├── output/
│   ├── sotuyen1/submission.zip      # Kết quả vòng Sơ tuyển 1 (25 câu)
│   ├── sotuyen2/submission.zip      # Kết quả vòng Sơ tuyển 2 (30 câu)
│   └── sotuyen3/submission.zip      # Kết quả vòng Sơ tuyển 3 (36 câu - Đúng 100% BTC)
├── query/
│   ├── SOTUYEN1-bo-de-thi/          # Bộ đề gốc Sơ tuyển 1
│   ├── SOTUYEN2-bo-de-thi/          # Bộ đề gốc Sơ tuyển 2
│   └── SOTUYEN3-bo-de-thi/          # Bộ đề gốc Sơ tuyển 3
├── scripts/
│   ├── data_processing/             # Các script tiền xử lý dữ liệu và trích xuất đặc trưng
│   ├── evaluation/                  # Script chạy benchmark và đánh giá các cấu hình
│   └── submission/                  # Script đóng gói và kiểm tra định dạng nộp bài
├── src/
│   ├── indexing/                    # FAISS Indexer & NumPy BM25 Indexer
│   ├── query/                       # LLM Query Refiner & Quản lý Gemini API Key Pool
│   ├── retrieval/                   # UnifiedSearchCore, MultiCueRetriever, KeyframeZipLoader
│   └── tasks/                       # Task Handlers chuyên biệt: KIS, QA, TRAKE
├── web_app/                         # Nền tảng Web Platform thi đấu thời gian thực
│   ├── backend/                     # FastAPI Router (Search, Media, Submission, WebSocket)
│   └── static/                      # Giao diện Glassmorphism UI (HTML, CSS, JS)
├── run_web_app.py                   # Script khởi chạy máy chủ Web Platform
├── requirements.txt                 # Danh sách thư viện Python
└── README.md                        # Tài liệu hướng dẫn chính của dự án
```

---

## 👥 6. ĐỘI NGŨ PHÁT TRIỂN

* **Đơn vị:** Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM (HCMUS)
* **Cuộc thi:** Hội thi Thử thách Trí tuệ Nhân tạo TP.HCM (AI Challenge 2026)
