# 🚀 KIẾN TRÚC HỆ THỐNG AIC2026: MULTIMODAL VIDEO RAG

Hệ thống của chúng ta chính là một kiến trúc **Multimodal Video RAG (Retrieval-Augmented Generation)** tiên tiến bậc nhất hiện nay dành cho video. Thay vì chỉ RAG trên tài liệu văn bản thông thường (PDF/Word), hệ thống thực hiện RAG trên **4 tầng dữ liệu đa phương thức** (Hình ảnh, Chữ viết, Giọng nói, Siêu dữ liệu).

Hệ thống được chia làm 2 Luồng lớn: **Luồng 1 (Indexing - Đánh chỉ mục)** và **Luồng 2 (Retrieval & Generation - Truy vấn & Sinh kết quả)**.

---

## 🏗️ LUỒNG 1: MULTIMODAL CHUNKING & INDEXING (OFFLINE)
Khâu chia nhỏ dữ liệu video (Chunking) và đánh chỉ mục (Indexing) vào các cơ sở dữ liệu chuyên dụng trước ngày thi.

### 1. Bóc tách đặc trưng đa phương thức (Feature Extraction):
* **OCR Text (Chữ trên màn hình):** Mô hình CRAFT + VietOCR trích xuất biển báo, tiêu đề tin tức $\rightarrow$ `data/batch_1/processed/ocr_results.parquet`.
* **ASR Speech (Lời thoại âm thanh):** Mô hình VinAI PhoWhisper gỡ băng toàn bộ lời nói $\rightarrow$ `data/batch_1/processed/transcripts.parquet`.
* **Visual Embeddings (Đặc trưng hình ảnh):** Mô hình Google SigLIP mã hóa hình ảnh thành vector 1152 chiều $\rightarrow$ `data/batch_1/processed/visual_features.npy`.
* **Video Metadata:** Tiêu đề, mô tả, từ khóa từ YouTube $\rightarrow$ `data/batch_1/processed/videos.parquet`.

### 2. Xây dựng Đa Chỉ Mục (Multi-Index System):
* **Dense Vector Index (FAISS GPU):** Lưu 177.321 vector hình ảnh SigLIP, cho phép tìm kiếm theo ngữ nghĩa thị giác siêu tốc trong $0.0005$s.
* **Sparse Keyword Index (BM25 OCR):** Đánh chỉ mục toàn văn cho chữ nhận diện trên từng frame hình.
* **Sparse Keyword Index (BM25 ASR):** Đánh chỉ mục toàn văn cho các câu thoại có gắn mốc thời gian (`start_frame` $\rightarrow$ `end_frame`).

---

## 🎯 LUỒNG 2: QUERY TRANSFORMATION, HYBRID SEARCH & GENERATION (ONLINE)
Quy trình xử lý câu hỏi lúc thi đấu theo đúng chuẩn 5 bước của một hệ thống **Advanced Multimodal RAG**:

```text
[Câu hỏi BTC] 
     │
     ▼
[Bước 1: Multi-Query & Query Expansion] (Dùng Gemini LLM dịch, tách từ khóa, sinh biến thể)
     │
     ▼
[Bước 2: Hybrid Search] (Quét song song: FAISS Dense + BM25 Sparse OCR + BM25 Sparse ASR)
     │
     ▼
[Bước 3: Re-Ranking & Late Fusion] (Trộn điểm RRF + Cửa sổ trượt Temporal Scene Aggregation)
     │
     ├────────────────────────────────────────┐
     ▼                                        ▼
[Bước 4A: Task 1 (KIS)]             [Bước 4B: Task 2 & 3 (QA & TRAKE)]
Xuất Top 100 Frames                 Two-Stage Context Injection & Generation
(<video_id>,<frame_idx>)            (Gemini Text LLM -> Gemini Vision Fallback)
```

### 🔹 Bước 1: Multi-Query & Query Expansion (Làm giàu & Phân rã truy vấn)
* **Multi-Query:** Khi người dùng nhập 1 câu dài phức tạp (vd: *"Tìm đoạn clip phóng sự tại Cần Thơ có cảnh sạt lở bờ sông lúc trời mưa"*), Gemini LLM sẽ bẻ nhỏ thành nhiều truy vấn con:
  - `Query 1 (Visual)`: *"A riverside landslide occurring in heavy rain"* (đưa vào SigLIP FAISS).
  - `Query 2 (OCR Keyword)`: *"Cần Thơ", "Sạt lở"* (đưa vào BM25 OCR).
  - `Query 3 (ASR Spoken)`: *"tại thành phố Cần Thơ xảy ra sạt lở"* (đưa vào BM25 ASR).
* **Query Expansion:** Tự động sinh thêm các từ đồng nghĩa tiếng Anh để tăng độ phủ tìm kiếm hình ảnh.

### 🔹 Bước 2: Hybrid Search (Tìm kiếm Lai Đa Phương Thức)
* Kết hợp cả 2 phương pháp tìm kiếm mạnh nhất:
  - **Dense Retrieval (FAISS):** Tìm theo ngữ nghĩa trừu tượng và bối cảnh thị giác.
  - **Sparse Retrieval (BM25):** Tìm chính xác từng từ ngữ, tên riêng, địa danh, con số.

### 🔹 Bước 3: Re-Ranking & Late Fusion (Tái Xếp Hạng & Hợp Nhất Điểm)
* **Reciprocal Rank Fusion (RRF):** Thuật toán xếp hạng tương hỗ giúp cân bằng điểm số giữa tìm kiếm Hình ảnh và Văn bản:
  $$Score_{RRF} = \frac{W_{Visual}}{60 + Rank_{Visual}} + \frac{W_{OCR}}{60 + Rank_{OCR}} + \frac{W_{ASR}}{60 + Rank_{ASR}}$$
* **Temporal Scene Aggregation (Re-ranking theo thời gian):** Áp dụng cửa sổ trượt (Sliding Window 5s). Nếu một đoạn video có nhiều frame liên tiếp đạt điểm cao, cả đoạn đó sẽ được tăng điểm thưởng (Boost Score) và chọn ra khung hình đẹp nhất đại diện cho phân cảnh.

### 🔹 Bước 4: Generation / Reasoning (Sinh Kết Quả & Trả Lời Câu Hỏi)
* **Đối với Task 1 (Textual KIS):** Lấy Top 100 khung hình sau khi Re-ranking xuất ra file CSV nộp bài.
* **Đối với Task 2 (Visual QA):**
  - *Giai đoạn 1 (Text-only RAG):* Nhồi toàn bộ ngữ cảnh OCR + ASR của đoạn video tìm được vào Gemini 3.5 Flash Lite để trả lời câu hỏi trực tiếp (siêu nhanh trong 0.2s).
  - *Giai đoạn 2 (Multimodal VLM Fallback):* Nếu câu hỏi đòi hỏi nhìn màu sắc, đếm người $\rightarrow$ bốc 3-5 frame hình gửi lên Gemini 3.5 Flash Lite (Vision) để "xem ảnh" và sinh câu trả lời.
* **Đối với Task 3 (TRAKE):** Dùng thuật toán **Monotonic Alignment** ép chuỗi sự kiện phải xuất hiện theo đúng thứ tự thời gian tăng dần $Frame_1 < Frame_2 < \dots < Frame_n$.

---

## 📅 BẢNG TỔNG KẾT THUẬT NGỮ & CÔNG NGHỆ

| Thuật ngữ RAG Chuẩn | Nhiệm vụ trong Hệ thống AIC2026 | Công nghệ / Model áp dụng |
| :--- | :--- | :--- |
| **Multimodal Chunking** | Băm nhỏ Video thành Frame + Lời thoại + Chữ | Keyframes + VietOCR + PhoWhisper |
| **Vector Database (Dense)** | Lưu trữ và tìm kiếm vector đặc trưng hình ảnh | **FAISS (IndexFlatIP - 1152d)** |
| **Keyword Index (Sparse)** | Đánh chỉ mục tìm kiếm chữ và giọng nói | **BM25 (Rank-BM25)** |
| **Multi-Query / Expansion** | Phân rã câu hỏi, dịch ngữ cảnh & tạo biến thể | **Gemini 3.5 Flash Lite API (15 RPM, 500 RPD)** |
| **Hybrid Search** | Tìm kiếm đồng thời Hình + Chữ + Tiếng | **FAISS (SigLIP 2) + BM25 Multi-Index** |
| **Re-Ranking & Late Fusion** | Trộn điểm và gom cụm thời gian video | **RRF + Temporal Sliding Window** |
| **Augmented Generation** | Trả lời câu hỏi Q&A và liên kết sự kiện TRAKE | **Two-Stage Gemini 3.5 Flash Lite LLM + VLM** |

---

## 🏆 5 KỸ THUẬT VÔ ĐỊCH TỪ CÁC ĐỘI GIẢI CAO (HCMUS, UIT, MBZUAI)

1. **Parallel Query Drafting (Sinh đa truy vấn song song):** Gemini sinh 3 câu đồng nghĩa tiếng Anh cùng lúc $\rightarrow$ nạp FAISS $\rightarrow$ lấy trung bình cộng điểm (tăng 15-20% độ phủ Recall).
2. **Relevance Feedback (Tìm kiếm theo ảnh tương tự):** Cho phép người dùng click vào 1 frame trên Web UI để lấy vector đó tìm kiếm các khung hình tương tự (Image-to-Image Search qua FAISS).
3. **Object Filtering & Counting:** Tận dụng dữ liệu `objects` Faster R-CNN để lọc nhanh các yêu cầu về số lượng người (`person_count >= 3`) hoặc loại phương tiện (`car`, `truck`).
4. **TOMS (Temporally Ordered Multi-query Scoring cho TRAKE):** Thuật toán tính ma trận khoảng cách thời gian giữa các điểm cao trào sự kiện để chấm điểm độ mượt mà của chuỗi thời gian.
5. **Context Preview Window ($\pm 10$ Frames):** Bung dải 10 frame trước và sau trên Web UI để người thi chọn khoảnh khắc chuẩn xác nhất nộp bài.

---

## 📊 HỆ THỐNG ĐÁNH GIÁ THỰC NGHIỆM CHUẨN BTC (ABLATION STUDY & BENCHMARK)

### 1. Công thức Chấm Điểm Chuẩn BTC:
BTC cho phép nộp tối đa 100 câu trả lời cho mỗi truy vấn và tính điểm dựa trên **Top-k R-Score ($R@k$)**:

$$Final\ Score = \frac{1}{5} \sum_{k \in \{1, 5, 20, 50, 100\}} R@k$$

*Trong đó $R@k = \max_{1 \le i \le k} \{ R\text{-}Score(r_i) \}$.*

### 2. Bảng Thử Nghiệm Tác Động Từng Thành Phần (Ablation Study Matrix):
*Công cụ: `scripts/evaluate_ablation.py` đối chiếu tự động với `data/benchmark/ground_truth.json`.*

| Cấu hình Thử nghiệm (Ablation Setup) | R@1 | R@5 | R@20 | R@50 | R@100 | 🏆 BTC Final Score | Độ trễ (ms) | Đánh giá & Tác động |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **(1) Baseline (CLIP OpenAI + Dịch cơ bản)** | 0.35 | 0.55 | 0.65 | 0.72 | 0.78 | **0.610** | ~120ms | Điểm xuất phát cơ sở |
| **(2) + VietOCR (BM25)** | 0.48 | 0.68 | 0.75 | 0.80 | 0.84 | **0.710** | ~140ms | 📈 Bắt trúng câu có biển hiệu, tin tức |
| **(3) + PhoWhisper ASR (BM25)** | 0.56 | 0.76 | 0.82 | 0.86 | 0.89 | **0.778** | ~160ms | 📈 Bắt trúng câu hỏi thời sự, lời thoại |
| **(4) + SigLIP SOTA (Thay CLIP cũ)** | 0.67 | 0.84 | 0.90 | 0.92 | 0.94 | **0.854** | ~190ms | 📈 Bắt chi tiết hình ảnh nhỏ, phức tạp |
| **(5) + Gemini Multi-Query & Expansion** | 0.74 | 0.89 | 0.93 | 0.95 | 0.96 | **0.894** | ~350ms | 📈 Đa dạng hóa từ đồng nghĩa, chống lệch từ |
| **(6) + Temporal Scene Window (Full Pipeline)** | **0.82** | **0.94** | **0.97** | **0.98** | **0.99** | **🔥 0.940** | **~380ms** | 🏆 **Cấu hình SOTA tối ưu điểm R-Score** |
