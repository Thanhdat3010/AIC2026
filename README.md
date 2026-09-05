# 🏆 V-GATE: Video Gated Adaptive Temporal Engine (AIC 2026)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FAISS](https://img.shields.io/badge/FAISS-GPU%20FlatIP-green.svg)](https://github.com/facebookresearch/faiss)
[![FastAPI](https://img.shields.io/badge/FastAPI-Collaborative%20Web-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Official Repository for AI Challenge HCMC (AIC) 2026**  
> **Faculty of Information Technology, University of Science, VNU-HCM (HCMUS)**  
> **Core Architecture:** 🚀 **3-Tier Decoupled Architecture + Google SigLIP-2 + Dynamic Modality Gating + CoDE MQ-DPF + Viterbi Monotonic Dynamic Programming + Audio-Visual Cascade VLM**  
> **Benchmark Performance:** 🥇 **Macro Score `0.7250` (+42.0\% vs Baseline) | KIS Score `0.8636` | TRAKE Score `0.7333` | Video Recall@1 `71.9%` | Video Recall@20 `93.8%`**

---

## 📌 1. TỔNG QUAN KIẾN TRÚC V-GATE (3-TIER CLEAN ARCHITECTURE)

Hệ thống **V-GATE (Video Gated Adaptive Temporal Engine)** được thiết kế theo mô hình 3 tầng phân ly độc lập, giải quyết triệt để 3 bài toán trọng tâm của cuộc thi: **Known-Item Search (KIS)**, **Visual Question Answering (QA)**, và **Temporal Event Tracking (TRAKE)** trên kho lưu trữ truyền hình 1,478 video (>324 giờ, 177,321 keyframes).

```
                        ┌─────────────────────────────────────────────────────────┐
                        │              CÂU TRUY VẤN TIẾNG VIỆT TỪ BTC              │
                        └───────────────────────────┬─────────────────────────────┘
                                                    │
                                                    ▼
                        ┌─────────────────────────────────────────────────────────┐
                        │   🧠 TẦNG 1: QUERY REFINER & DYNAMIC MODALITY GATING     │
                        │    • LLM Dual-Lingual Semantic Projection: 0.7Vi + 0.3En│
                        │    • Softmax Intent Gating: [alpha_vis, alpha_ocr, asr] │
                        │    • Epsilon-Threshold Bypass: Tiết kiệm 17.4% độ trễ   │
                        └───────────────────────────┬─────────────────────────────┘
                                                    │
                                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 🚀 TẦNG 2: HIGH-THROUGHPUT UNIFIED RETRIEVAL CORE (177,321 KEYFRAMES)                           │
│   • Visual Backbone : Google SigLIP-2 SO400M (1152d, Tensor Cores FP16 trên GPU FAISS FlatIP)  │
│   • Text Retrieval  : Fast Vectorized BM25 cho OCR (177k docs) và Whisper ASR (16k docs)        │
│   • Multi-Cue Fusion: Weighted Reciprocal Rank Fusion (WRRF với hệ số kappa_0 = 60)             │
└───────────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 🎯 TẦNG 3: TASK-SPECIALIZED SOLVERS (CHUẨN 100 DÒNG QUY CHẾ BTC)                                │
│   • 🎯 KIS Solver   : CoDE Multi-Query Dual-Perspective Fusion (S_global + S_core)              │
│                       + Temporal Cluster Expansion (rải 4-6 frames/video ứng viên)              │
│   • ❓ QA Solver    : Audio-Visual Cascade VLM (Keyframe Inspection + Whisper ASR [t±15s])     │
│                       + Chuẩn hóa định dạng và phân tích suy luận số học/thực thể               │
│   • ⏱️ TRAKE Solver : Viterbi Monotonic Dynamic Programming trên ma trận độ tương đồng cosine    │
│                       -> 100 chuỗi sự kiện strictly increasing: t(E1) < t(E2) < ... < t(En)     │
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

## ⚡ 2. CỐT LÕI CÔNG NGHỆ ĐỘT PHÁ

1. **Chiếu Song Ngữ Tối Ưu (Dual-Lingual Semantic Projection)**:
   - Kết hợp biểu diễn đa ngôn ngữ: $\mathbf{e}_Q = 0.7 \cdot \text{Enc}_{\text{text}}(Q_{\text{vi}}) + 0.3 \cdot \text{Enc}_{\text{text}}(Q_{\text{en}})$.
   - Khắc phục triệt để hiện tượng thưa thớt từ vựng (lexical sparsity) của các mô hình foundation thị giác khi truy vấn bằng tiếng Việt, nâng KIS Score từ **0.7364** lên **0.8455** (+14.8%).
2. **Cơ Chế Dynamic Modality Gating (Không Dùng Heuristic Cứng)**:
   - Tự động suy biến vector chủ đích $\mathbf{z} \in \mathbb{R}^3$ thông qua Softmax Intent Gating với nhiệt độ $\tau = 0.8$.
   - Khi $\alpha_{\text{ocr}} < 0.05$, hệ thống tự động bỏ qua lượt quét index OCR, giúp giảm độ trễ trung bình từ **242.4 ms** xuống **200.3 ms** (**tiết kiệm 17.4% thời gian**).
3. **Quy Hoạch Động Đơn Điệu Viterbi (Monotonic Dynamic Programming) cho TRAKE**:
   - Giải quyết trật tự chuỗi sự kiện trên ma trận $D(i, j) = C(i, j) + \max_{1 \le k < j} D(i-1, k)$.
   - Áp đặt điều kiện thời gian tăng ngặt $t(E_1) < t(E_2) < \dots < t(E_n)$ trong thời gian tối ưu $O(M \cdot T)$ nhờ prefix maximum caching, nâng điểm TRAKE từ **0.2000** lên **0.7333** (+266%).
4. **CoDE Multi-Query Dual-Perspective Fusion (MQ-DPF)**:
   - Tách truy vấn thành góc nhìn toàn cảnh ($S_{\text{global}}$) và hành động trọng tâm ($S_{\text{core}}$), kết hợp với Temporal Cluster Expansion giúp phân bổ dày đặc các khung hình đúng xung quanh keyframe mục tiêu.
5. **Audio-Visual Cascade VLM Solver cho QA**:
   - Tự động trích xuất âm thanh trong cửa sổ $[t - 15\text{s}, t + 15\text{s}]$ xung quanh keyframe visual, kết hợp Google Gemini Flash Lite để suy luận chính xác các thực thể, biển số, thời gian và sự kiện.

---

## 📊 3. KẾT QUẢ THỰC NGHIỆM & ABLATION STUDY

Đánh giá toàn diện trên bộ benchmark chuẩn 32 câu hỏi chính thức (22 KIS, 7 QA, 3 TRAKE):

| Cấu Hình Thực Nghiệm | Bi-Proj | TNCA | MM | Dyn-G | Clust | Vit-DP | Aud-QA | KIS | QA | TRAKE | Macro Score | Video R@1 | Độ Trễ (Avg) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Visual Baseline (SigLIP-2)** | | | | | | | | 0.7364 | 0.0000 | 0.0444 | **0.5104** | 53.1% | 120.5 ms |
| *+ Dual-Lingual Projection* | $\checkmark$ | | | | | | | 0.8455 | 0.0000 | 0.2000 | **0.6000** | 62.5% | 117.9 ms |
| *+ Temporal Context (TNCA)* | $\checkmark$ | $\checkmark$ | | | | | | 0.8364 | 0.0000 | 0.2000 | **0.5938** | 62.5% | 131.4 ms |
| *+ Static Multimodal Fusion* | $\checkmark$ | $\checkmark$ | $\checkmark$ | | | | | 0.8364 | 0.0000 | 0.2000 | **0.5938** | 68.8% | 242.4 ms |
| *+ Dynamic Modality Gating* | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | | | | 0.8364 | 0.0000 | 0.2000 | **0.5938** | 68.8% | 200.3 ms |
| *+ Temporal Cluster Expansion* | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | | | 0.8091 | 0.0000 | 0.6444 | **0.6167** | 62.5% | 457.5 ms |
| **V-GATE Full Architecture (Ours)** | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | **0.8636** | **0.2857** | **0.7333** | **`0.7250`** | **`71.9%`** | 1143.3 ms |
| *w/o Dynamic Modality Gating* | $\checkmark$ | $\checkmark$ | $\checkmark$ | -- | $\checkmark$ | $\checkmark$ | $\checkmark$ | 0.8636 | 0.2857 | 0.7333 | **0.7250** | 71.9% | 451.5 ms |
| *w/o Monotonic Viterbi DP* | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | -- | $\checkmark$ | 0.8636 | 0.2857 | 0.7111 | **0.7229** | 71.9% | 327.3 ms |
| *w/o Audio-Visual Cascade QA* | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | -- | 0.8636 | 0.2857 | 0.7333 | **0.7250** | 71.9% | 956.4 ms |

> 📌 **Ghi chú**: Kết quả chi tiết và bảng LaTeX phục vụ xuất bản bài báo được lưu tại [`ABLATION_STUDY.md`](ABLATION_STUDY.md) và [`data/benchmark/ground_truth_2_ablation_summary.json`](data/benchmark/ground_truth_2_ablation_summary.json).

---

## 🚀 4. HƯỚNG DẪN CÀI ĐẶT & TÁI LẬP (REPRODUCIBILITY)

### 4.1. Khởi tạo Môi trường
```bash
# Clone repository
git clone https://github.com/Thanhdat3010/AIC2026.git
cd AIC2026

# Tạo và kích hoạt môi trường conda
conda create -n AIC2026 python=3.10 -y
conda activate AIC2026

# Cài đặt PyTorch với CUDA hỗ trợ
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Cài đặt toàn bộ thư viện phụ thuộc
pip install -r requirements.txt
```

### 4.2. Chạy Tái Lập Toàn Bộ Ablation Study (10 Cấu Hình)
```bash
# Chạy đánh giá tự động và xuất báo cáo JSON/Markdown
python scripts/evaluation/run_sota_ablation_gt2.py
```

### 4.3. Khởi Chạy Nền Tảng Thi Đấu Trực Tuyến (FastAPI Web Platform)
```bash
# Khởi chạy server FastAPI tại cổng 8000
python run_web_app.py
```
Truy cập giao diện thi đấu tại: `http://localhost:8000`.

### 4.4. Đóng Gói & Kiểm Tra File Nộp Bài Chuẩn BTC
```bash
# Đóng gói tự động 36 file CSV
python scripts/submission/run_submission.py --output_package sotuyen3

# Kiểm tra tính toàn vẹn 100 dòng theo quy chế BTC
python scripts/submission/submission_validator.py output/sotuyen3/submission.zip
```

---

## 📂 5. CẤU TRÚC THƯ MỤC DỰ ÁN

```
AIC2026/
├── ABLATION_STUDY.md                # Báo cáo chi tiết 10 cấu hình thực nghiệm
├── paper/
│   └── AIC_paper/                   # Toàn bộ bản thảo LaTeX & BibTeX cho SOICT 2026
│       ├── main.tex                 # Bản thảo hoàn chỉnh của bài báo V-GATE
│       ├── mybibliography.bib       # 20 tài liệu tham khảo đã kiểm chứng 100% DOI
│       └── llncs.cls                # Class template chuẩn Springer LNCS
├── data/
│   └── benchmark/                   # Ground Truth 2 & file kết quả ablation summary
├── scripts/
│   ├── evaluation/
│   │   └── run_sota_ablation_gt2.py # Script tái lập thực nghiệm chuẩn
│   └── submission/
│       ├── run_submission.py        # Đóng gói submission chuẩn 36 câu
│       └── submission_validator.py  # Bộ validator kiểm tra quy chế BTC
├── src/
│   ├── indexing/                    # FAISS Indexer & NumPy BM25 Indexer
│   ├── query/                       # LLM Query Refiner & Gemini Key Pool
│   ├── retrieval/                   # UnifiedSearchCore, MultiCueRetriever
│   └── tasks/                       # KISHandler, QAHandler, TRAKEAlignmentAgent
├── web_app/                         # Giao diện Web Platform thi đấu thời gian thực
├── run_web_app.py                   # Script chạy hệ thống web app
└── requirements.txt                 # Danh sách thư viện cần thiết
```

---

## 📝 6. TRÍCH DẪN (CITATION)

Nếu bạn sử dụng mã nguồn hoặc kết quả nghiên cứu của hệ thống **V-GATE**, vui lòng trích dẫn bài báo:

```bibtex
@inproceedings{nguyen2026vgate,
  author    = {Hung D. Nguyen and Dat T. Truong and Tam K. Pham and Tung Le},
  title     = {{V-GATE}: Dynamic Modality Gating and Monotonic Temporal Alignment for Multi-Task Video Retrieval in {AI Challenge HCMC} 2026},
  booktitle = {Proceedings of the 15th International Symposium on Information and Communication Technology (SOICT 2026)},
  series    = {Communications in Computer and Information Science},
  publisher = {Springer},
  year      = {2026}
}
```

---

## 👥 7. ĐỘI NGŨ PHÁT TRIỂN & BẢN QUYỀN

* **Đơn vị:** Khoa Công nghệ Thông tin, Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM (HCMUS).
* **Bản quyền:** Phát hành dưới giấy phép [MIT License](LICENSE).
