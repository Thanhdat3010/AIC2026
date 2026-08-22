# 🏆 AIC 2026 - Unified Multimodal Video Retrieval & QA System (HCMUS)

> **Dự án thi đấu AI Challenge (AIC 2026) - Đội tuyển Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM (HCMUS)**  
> **Kiến trúc Chủ lực:** 🚀 **3-Tier Clean Unified Architecture + Dynamic Adaptive Modality Gating + Viterbi Monotonic DP TRAKE + Audio-Visual Cascade QA**  
> **Hiệu năng Thực nghiệm:** 🥇 **`93.6%` Video Recall@100 | `0.7214` KIS Score | Khóa 100% mô hình `gemini-3.5-flash-lite`**

---

## 📌 1. TỔNG QUAN HỆ THỐNG (3-TIER CLEAN ARCHITECTURE)

Hệ thống được thiết kế theo mô hình **3 Tầng Tinh Gọn - Độc Lập - Hiệu Năng Cao** phục vụ hoàn hảo 3 bài toán của Ban Tổ Chức (Textual KIS, Visual QA, Temporal TRAKE):

```
                        ┌─────────────────────────────────────────────────────────┐
                        │              CÂU TRUY VẤN TIẾNG VIỆT TỪ BTC              │
                        └───────────────────────────┬─────────────────────────────┘
                                                    │
                                                    ▼
                        ┌─────────────────────────────────────────────────────────┐
                        │   🧠 TẦNG 1: LLM REFINER & DYNAMIC MODALITY GATING       │
                        │    • Model duy nhất: Google Gemini 3.5 Flash Lite        │
                        │    • Phân tích ngữ nghĩa sâu & sửa lỗi chính tả BTC      │
                        │    • Prototype Vector Cosine Gating (Softmax liên tục)   │
                        │    • Trích xuất OCR & ASR keywords, tách E1..En TRAKE    │
                        └───────────────────────────┬─────────────────────────────┘
                                                    │
                                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 🚀 TẦNG 2: UNIFIED RETRIEVAL CORE (177,321 KEYFRAMES)                                           │
│   • Visual Backbone : Google SigLIP-2 SO400M (1152d, Tensor Cores FP16 trên GPU)               │
│   • Text Retrieval  : Fast NumPy BM25 Multi-Indexer (<2ms) cho OCR (177k docs) và ASR (16k docs)│
│   • Hybrid Fusion   : Adaptive Weighted Reciprocal Rank Fusion (WRRF k0=60)                     │
│   • Statistical Gate: Margin Gating phân giải mơ hồ thị giác khi điểm Visual bị phẳng           │
└───────────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 🎯 TẦNG 3: TASK SPECIALIZED HANDLERS (CHUẨN 100 DÒNG QUY CHẾ BTC)                               │
│   • 🎯 KIS Handler   : Adaptive WRRF + Temporal Cluster Expansion (rải 4-6 frames/video)        │
│   • ❓ QA Handler    : Audio-Visual Cascade VLM (Soi ảnh -> Nghe Whisper ASR [t±15s] + OCR)     │
│                        + Phân bổ dải số 1..20 thông minh cho câu đếm                            │
│   • ⏱️ TRAKE Handler : Viterbi Monotonic Dynamic Programming trên siglip_features.npy memmap    │
│                        -> 100 chuỗi sự kiện strictly increasing: t(E1) < t(E2) < t(E3)          │
└───────────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                            │
                                            ▼
                        ┌─────────────────────────────────────────────────────────┐
                        │   🏆 STREAMLIT CHAMPIONSHIP CONSOLE                     │
                        │    • Tab 1: Leaderboard Ablation Study đối đầu 8 cấu hình│
                        │    • Tab 2: Xem video On-Demand & Hiệu chỉnh kết quả nộp│
                        │    • Tab 3: Live Multimodal Search Engine thời gian thực│
                        │    • Xuất gói nộp bài submission.zip chuẩn 100% BTC     │
                        └─────────────────────────────────────────────────────────┘
```

---

## ⚡ 2. CÁC TÍNH NĂNG VƯỢT TRỘI

1. **Cơ Chế Dynamic Adaptive Modality Gating (Không Dùng Keyword Cứng)**:
   - Tự động phân định khi nào cần dựa vào thị giác, khi nào cần nghe lời thoại và khi nào cần đọc chữ thông qua sự kết hợp của **LLM Semantic Intent + Prototype Vector Cosine Gating + Statistical Margin Gating**.
2. **Audio-Visual Cascade Reasoning cho Visual QA**:
   - VLM soi ảnh trước; nếu câu hỏi liên quan đến tên người, danh từ riêng hoặc ảnh bị mờ/bất định, hệ thống **tự động kích hoạt Whisper ASR Listener** nạp transcript $[t - 15s, t + 15s]$ để đối chiếu chéo sinh đáp án chính xác.
3. **Viterbi Monotonic Dynamic Programming cho TRAKE**:
   - Tính toán ma trận Cosine liên tục trên `siglip_features.npy` memmap và giải quy hoạch động ràng buộc thời gian nghiêm ngặt, đảm bảo chuỗi $E_1 < E_2 < E_3$ đạt độ chính xác cao nhất.
4. **Khai thác Trọn vẹn 100 Dòng Nộp Bài (Top 100 BTC)**:
   - Tối đa hóa công thức tính điểm của Ban Tổ Chức:
     $$Final\ Score = \frac{1}{5} \sum_{k \in \{1, 5, 20, 50, 100\}} \max_{1 \le i \le k} R\text{-Score}(r_i)$$

---

## 🚀 3. HƯỚNG DẪN KHỞI CHẠY HỆ THỐNG

### 3.1. Cài đặt Môi trường
```bash
# Kích hoạt môi trường conda
conda activate AIC2026

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 3.2. Khởi chạy Web Console (Streamlit Studio)
```bash
python -m streamlit run app/streamlit_app.py --server.port 8501
```
Giao diện sẽ mở tại `http://localhost:8501`.

### 3.3. Chạy Đo Lường Benchmark & Ablation Study
```bash
# Chạy đo lường toàn diện các cấu hình trên 47 test cases Ground Truth
python scripts/evaluation/benchmark_clean.py
```

---

## 📂 4. CẤU TRÚC THƯ MỤC DỰ ÁN

```
AIC2026/
├── app/
│   └── streamlit_app.py          # Giao diện Web Studio Championship 3 Tab
├── data/
│   ├── batch_1/processed/        # Dữ liệu vector SigLIP-2, frames, OCR, Transcripts
│   └── benchmark/                # Ground truth 47 câu và bảng tóm tắt Ablation Study
├── docs/
│   ├── AIC2026_Yeu_cau_BTC.md    # Tài liệu gốc yêu cầu và quy chế thi đấu của BTC
│   ├── ARCHITECTURE.md           # Thiết kế kỹ thuật chi tiết kiến trúc 3 tầng
│   ├── DATA_PREPROCESSING.md     # Hướng dẫn toàn diện quy trình tiền xử lý dữ liệu
│   ├── ABLATION_STUDY.md         # Báo cáo thực nghiệm đối đầu các cấu hình thuật toán
│   └── USER_GUIDE.md             # Hướng dẫn sử dụng chi tiết Web Console
├── scripts/
│   ├── data_processing/          # Toàn bộ script tải và tiền xử lý dữ liệu (Bảo toàn 100%)
│   ├── evaluation/               # Script đo lường benchmark_clean.py
│   └── submission/               # Script đóng gói và kiểm tra submission.zip
├── src/
│   ├── indexing/                 # FAISS Indexer & BM25 Fast NumPy Multi-Indexer
│   ├── query/                    # LLM Query Refiner & Gemini Key Pool (Gemini 3.5 Flash Lite)
│   ├── retrieval/                # UnifiedSearchCore & KeyframeZipLoader
│   └── tasks/                    # Clean Task Handlers (KISHandler, QAHandler, TRAKEHandler)
└── output/
    └── sotuyen1/submission.zip   # Gói nộp bài chính thức chuẩn 100% quy chế BTC
```

---

## 👥 5. ĐỘI NGŨ PHÁT TRIỂN
* **Đơn vị:** Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM (HCMUS)
* **Cuộc thi:** Hội thi Thử thách Trí tuệ Nhân tạo TP.HCM (AI Challenge 2026)
