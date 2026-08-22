# 🏛️ Kiến Trúc Hệ Thống AIC 2026 (System Architecture Specification)

Tài liệu này đặc tả chi tiết thiết kế kỹ thuật, luồng xử lý dữ liệu và nguyên lý hoạt động của hệ thống truy xuất video đa phương thức **AIC 2026** (Đội tuyển HCMUS).

---

## 1. TỔNG QUAN KIẾN TRÚC 3 TẦNG (3-TIER CLEAN ARCHITECTURE)

Hệ thống được tổ chức thành 3 tầng độc lập, tuần tự và tối ưu hóa hiệu năng:

```
[Query BTC] ──> [Tầng 1: LLM Refiner & Dynamic Gating]
                     │
                     ▼
                [Tầng 2: Unified Retrieval Core (SigLIP-2 + BM25 OCR/ASR + Fast WRRF)]
                     │
                     ▼
                [Tầng 3: Task Specialized Handlers (KIS / QA / TRAKE 100 Rows)]
                     │
                     ▼
                [Streamlit Championship Console & Submission Package]
```

---

## 2. CHI TIẾT CÁC TẦNG HỆ THỐNG

### 2.1. Tầng 1: LLM Query Refiner & Dynamic Modality Gating
* **Mô hình AI**: Khóa duy nhất **`gemini-3.5-flash-lite`** kết hợp cơ chế xoay vòng Key Pool (`GeminiKeyPool`) để phân tải và chống nghẽn Rate Limit (HTTP 429).
* **Chức năng cốt lõi**:
  1. **Chuẩn hóa ngôn ngữ**: Sửa lỗi chính tả tiếng Việt, loại bỏ từ thừa, chuẩn hóa câu hỏi của BTC.
  2. **Thuật toán 1 — LLM Semantic Intent Classification**: Trích xuất trọng số đa phương thức qua JSON Schema:
     - `visual_relevance`: Độ liên quan đến hình ảnh/hành động/màu sắc.
     - `ocr_relevance`: Độ liên quan đến chữ viết/bảng hiệu/logo trên màn hình.
     - `asr_relevance`: Độ liên quan đến nội dung lời nói/phát biểu/thuyết minh/tin tức.
  3. **Thuật toán 2 — Prototype Vector Cosine Gating**: Tính khoảng cách Cosine Softmax giữa vector câu hỏi và 3 Modality Prototype Anchors trong không gian vector $1152d$ ($<0.1$ms).
  4. **Sub-Event Parsing**: Bóc tách chính xác các sự kiện con $E_1, E_2, E_3...$ cho bài toán TRAKE.

---

### 2.2. Tầng 2: Unified Retrieval Core
* **Visual Engine**:
  - **Mô hình**: Google SigLIP-2 `so400m-patch14-384` (Embedding dimension: $1152$).
  - **Index**: FAISS IndexFlatIP (Inner Product = Cosine Similarity trên vector đã chuẩn hóa L2).
  - **Dữ liệu**: $177,321$ keyframes từ toàn bộ các video Batch 1.
* **Text Engine (Fast NumPy BM25 Multi-Indexer)**:
  - **OCR Index**: $177,605$ tài liệu OCR tiếng Việt được trích xuất từ khung hình. Tối ưu hóa NumPy Partition tìm kiếm Top 50 trong $<2$ms.
  - **ASR Index**: $16,698$ đoạn transcript lời thoại thuyết minh tiếng Việt từ Whisper ASR.
* **Hybrid Fusion Engine**:
  - **Thuật toán 4 — Adaptive Weighted Reciprocal Rank Fusion (WRRF)**:
    $$\text{Score}_{\text{WRRF}}(v, f) = w_{\text{vis}} \cdot \frac{1}{60 + \text{rank}_{\text{vis}}} + w_{\text{ocr}} \cdot \frac{1}{60 + \text{rank}_{\text{ocr}}} + w_{\text{asr}} \cdot \frac{1}{60 + \text{rank}_{\text{asr}}}$$
  - **Thuật toán 3 — Statistical Margin Gating**: Phân tích độ dốc điểm số $\Delta = S(\text{Rank 1}) - S(\text{Rank 20})$. Khi thị giác bị mơ hồ (phân phối điểm phẳng), tự động kích hoạt BM25 OCR & ASR làm lực lượng phân giải (Tie-Breaker).

---

### 2.3. Tầng 3: Task Specialized Handlers (Chuẩn 100 Dòng BTC)

#### A. KIS Handler (Known-Item Search)
* **Chiến thuật**: Tận dụng tối đa năng lực hiểu tiếng Việt bản xứ của SigLIP-2 đa ngôn ngữ (KIS Score **0.7214**).
* **Temporal Cluster Expansion**: Gom cụm keyframe theo từng video hàng đầu và trải rộng 4-6 khung hình lân cận để phủ kín đoạn thời gian xuất hiện của khoảnh khắc.
* **Đầu ra**: Đúng 100 dòng định dạng `<video_id>, <frame_idx>`.

#### B. QA Handler (Visual Question Answering)
* **Thuật toán 6 — Audio-Visual Cascade Reasoning**:
  - **Chặng 1 (Visual Certainty)**: `gemini-3.5-flash-lite` soi ảnh Top keyframes. Nếu câu hỏi về thuộc tính trực diện (màu sắc, con vật, phương tiện) $\rightarrow$ Sinh đáp án trực tiếp với độ tự tin cao.
  - **Chặng 2 (Visual Uncertainty Fallback)**: Nếu câu hỏi về danh từ riêng, tên nhân vật, nghề nghiệp, hoặc ảnh không thấy rõ $\rightarrow$ Tự động nạp đoạn **Whisper transcript tại mốc thời gian $[t - 15s, t + 15s]$** từ `transcripts.parquet` và text OCR từ `ocr_results.parquet` để đối chiếu chéo.
* **Smart Answer Distribution**:
  - Câu hỏi đếm: Phân bổ dải số thông minh **1..20** quanh video tốt nhất.
  - Câu hỏi văn bản: Gán câu trả lời VLM cho toàn bộ 100 khung hình của Top 100.
* **Đầu ra**: Đúng 100 dòng định dạng `<video_id>, <frame_idx>, "<answer>"`.

#### C. TRAKE Handler (Temporal Alignment)
* **Thuật toán 5 — Viterbi Monotonic Dynamic Programming**:
  - Lọc Top 50-100 Video ứng viên.
  - Tính ma trận Cosine Similarity $S = \text{Events} \times \text{Keyframes}$ trên `siglip_features.npy` memmap.
  - Áp dụng Local Temporal Smoothing $[0.2, 0.6, 0.2]$ chống nhiễu spike.
  - Giải quy hoạch động Viterbi tìm chuỗi khung hình tối ưu thỏa mãn ràng buộc đơn điệu thời gian nghiêm ngặt $t(E_1) < t(E_2) < ... < t(E_N)$.
* **Đầu ra**: Đúng 100 dòng định dạng `<video_id>, <frame_e1>, <frame_e2>, <frame_e3>`.

---

## 3. BẢNG THAM SỐ CẤU HÌNH HỆ THỐNG

| Tham số | Giá trị Mặc định | Ý nghĩa Kỹ thuật |
| :--- | :---: | :--- |
| `siglip_dim` | $1152$ | Kích thước vector nhúng của mô hình Google SigLIP-2 SO400M |
| `wrrf_k0` | $60$ | Hằng số làm trơn chuẩn trong Reciprocal Rank Fusion |
| `margin_threshold` | $0.005$ | Ngưỡng phân giải mơ hồ thị giác trong Statistical Margin Gating |
| `asr_window_sec` | $\pm 15.0$s | Cửa sổ thời gian trích xuất lời thoại Whisper quanh keyframe cho QA |
| `trake_smoothing` | $[0.2, 0.6, 0.2]$ | Kernel lọc mượt 3 điểm cục bộ trước khi chạy Viterbi DP |
| `top_k_submission` | $100$ | Số lượng dòng câu trả lời xuất ra cho mỗi truy vấn chuẩn BTC |
