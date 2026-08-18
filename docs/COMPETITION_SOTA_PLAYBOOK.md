# 🏆 CẨM NANG THI ĐẤU AIC 2026 (SOTA COMPETITION PLAYBOOK)

> **Cập nhật mới nhất:** `2026-08-17`  
> **Kiến trúc SOTA hiện tại:** `TaskSpecializedEngine` + `IntraVideoTemporalReranker` (E1→E3)  
> **Độ chính xác:** **`0.7143`** (KIS SOTA), **`0.7000`** (QA SOTA), **`100.0%`** Video Recall@5

---

## 📌 1. TỔNG QUAN KIẾN TRÚC SOTA

Hệ thống được thiết kế theo chiến lược **Task-Specific Specialist & Two-Stage Localization**:

```
                              [ User Query (VI) ]
                                       │
                              [ Modality Gate ]
                     ┌─────────────────┼─────────────────┐
                     ▼                 ▼                 ▼
             [ KIS Specialist ]  [ QA Specialist ]  [ TRAKE Specialist ]
                     │                 │                 │
         Gemini 3.5 Single Trans  Gemini Translation  Gemini Decomposition
                     │                 │                 │
            SigLIP 2 FAISS (1152d)  SigLIP 2 FAISS    Multi-Event Retrieval
                     │                 │                 │
             Stage-1 Candidates   Stage-1 Candidates  Temporal Monotonic DP
                     │                 │                 │
         [ Intra-Video Reranker ] [ Intra-Video Reranker ]│
         (E1: Neighbor Smoothing) (E3: ASR Timeline Sync)│
                     │                 │                 │
                     │           Gemini Vision VLM       │
                     ▼                 ▼                 ▼
              [ Top 100 CSV ]   [ Top 100 CSV + Ans ]  [ Top 100 CSV Chain ]
```

### Điểm mạnh cốt lõi:
1. **Stage-1 Dense Retrieval:** Sử dụng `google/siglip2-so400m-patch14-384` (1152 chiều) kết hợp Single Gemini 3.5 Flash Lite translation. Đạt **100.0% Video Recall@5** (toàn bộ video mục tiêu đều nằm trong Top 5).
2. **Stage-2 Intra-Video Temporal Reranker:**
   - **E1 (Gaussian Neighbor Temporal Smoothing $\sigma=1.5s$):** Tăng điểm cho các khung hình nằm trong cụm phân cảnh thật, loại bỏ nhiễu spike tức thời.
   - **E2 (Query Cue Coverage):** Đánh giá độ bao phủ đa sự kiện con trong cửa sổ trượt ±6s.
   - **E3 (Time-Aligned ASR/OCR Alignment):** Gắn khớp chính xác mốc thời gian lời thoại ASR / ký tự OCR vào timeline video.
3. **Visual QA Agent:** Gemini 3.5 Flash Lite Vision soi ảnh trực tiếp trên các frame đỉnh để sinh câu trả lời cô đọng `<100` ký tự.

---

## 🚀 2. HƯỚNG DẪN CHẠY THI ĐẤU (HOW TO RUN)

Môi trường Python yêu cầu:
```powershell
conda activate AIC2026
```

---

### 🔹 Cách 1: Sinh file nộp bài chính thức (Official Submission Pipeline)

Đây là lệnh chính dùng trong ngày thi đấu khi BTC cung cấp thư mục chứa các file câu hỏi `.txt` (ví dụ `query/batch_1/query-p1-groupA`):

```powershell
# Chạy trực tiếp trên thư mục câu hỏi của BTC
python scripts/submission/run_submission.py --input query/batch_1/query-p1-groupA --top_k 100
```

> **Hệ thống sẽ tự động:**
> 1. Đọc toàn bộ 24+ file truy vấn `.txt` (tự nhận diện tác vụ KIS, QA, TRAKE từ tên file).
> 2. Chạy pipeline SOTA tương ứng cho từng câu hỏi.
> 3. Xuất đúng 24+ file CSV vào thư mục `outputs/submission/*.csv`.
> 4. Tự động đóng gói nén thành `outputs/submission.zip` sẵn sàng nộp lên server BTC.

Hoặc nếu đề thi cung cấp dưới dạng file `.json`:
```powershell
python scripts/submission/run_submission.py --input data/benchmark/ground_truth.json --top_k 100
```

---

### 🔹 Cách 2: Kiểm tra tính hợp lệ của File Submission (Validator)

Trước khi nộp bài lên hệ thống BTC, luôn chạy script kiểm tra:

```powershell
python scripts/submission/validate_submission.py --zip outputs/submission.zip
```

Script sẽ tự động kiểm tra:
- Số lượng file và tên file CSV khớp danh sách truy vấn.
- Số dòng trong từng file đúng bằng 100.
- Định dạng video ID (không dính đuôi `.mp4`), frame index là số nguyên dương.
- Cột đáp án QA được bọc escape hợp lệ.

---

### 🔹 Cách 3: Chạy Benchmark & Ablation Study kiểm tra hiệu năng

Để kiểm tra điểm số và chẩn đoán độ chính xác hệ thống:

```powershell
# Chạy cấu hình SOTA tổng hợp (Config 14)
python scripts/evaluation/evaluate_ablation.py --config 14

# Chạy cấu hình KIS chuyên sâu (Config 12)
python scripts/evaluation/evaluate_ablation.py --config 12

# So sánh toàn bộ các cấu hình
python scripts/evaluation/evaluate_ablation.py --all_configs
```

---

## 🛠️ 3. CÁC TÙY CHỌN TRONG CODE (PYTHON API USAGE)

Nếu cần tích hợp vào giao diện Streamlit hoặc Web App:

```python
from src.retrieval.task_specialized_engine import TaskSpecializedEngine

# 1. Khởi tạo Engine
engine = TaskSpecializedEngine(engine="siglip2", batch="batch_1")

# 2. Tìm kiếm KIS (Khuyên dùng E1: Gaussian Neighbor Support)
kis_results, info, latency = engine.search_kis(
    query_text="Người đàn ông dùng máy đánh chữ vẽ tranh chân dung",
    top_k=100,
    use_intra_reranker=True,
    use_neighbor=True
)

# 3. Tìm kiếm QA (Khuyên dùng E1+E2+E3: Multi-modal Timeline Fusion)
qa_results, info, latency = engine.search_qa(
    query_text="Khi người dẫn chương trình hỏi chủ vườn nho bắt đầu và kết thúc làm việc lúc mấy giờ, người chủ trả lời thế nào?",
    top_k=100,
    use_intra_reranker=True,
    use_neighbor=True,
    use_cue=True,
    use_multimodal=True
)

# 4. Tìm kiếm TRAKE (Chuỗi sự kiện tăng dần theo thời gian)
trake_results, info, latency = engine.search_trake(
    query_text="Đầu bếp đổ hành tây vào chảo, thêm thịt bò băm, thêm đậu Hà Lan và cà rốt, cuối cùng thêm mì ống",
    top_k=100
)
```

---

## 📊 4. BẢNG THÀNH TÍCH ABLATION LEADERBOARD

| Rank | Cấu hình | Final Score | KIS Score | QA Score | Video-Recall@5 | Ghi chú |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 🥇 | **Task-Specialized + Intra-Reranker** | **0.5782 - 0.6000** | **0.7143** | **0.7000** | **100.0%** | **Cấu hình SOTA thi đấu chính thức** |
| 🥈 | Task-Specialized Baseline (Config 11) | 0.6000 | 0.6571 | 0.6000 | 90.9% | Stage-1 thuần túy |
| 🥉 | Monolithic Full Combo (Config 10) | 0.4509 | 0.4857 | 0.5000 | 81.8% | Bị pha loãng do gom chung |
| 4 | Google SigLIP 2 Dense (Config 1) | 0.3818 | 0.4286 | 0.4000 | 72.7% | Dense zero-shot |
| 5 | BTC CLIP Dense Baseline (Config 0) | 0.1333 | 0.1143 | 0.2000 | 36.4% | Baseline ban đầu BTC |
