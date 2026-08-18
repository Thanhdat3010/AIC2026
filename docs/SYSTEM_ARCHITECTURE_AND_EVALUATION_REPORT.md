# 🏛️ BÁO CÁO TOÀN DIỆN VỀ KIẾN TRÚC HỆ THỐNG, DỮ LIỆU ĐA PHƯƠNG THỨC & MA TRẬN ĐÁNH GIÁ 20 TEST CASES

> **Dự án:** HCMUS AI Challenge 2026 (AIC 2026)  
> **Phiên bản:** Kiến trúc SOTA 3 Tầng Chuyên Biệt Hóa Tác Vụ (Task-Specialized Architecture)  
> **Cập nhật:** Ngày 18/08/2026  
> **Tác giả:** Đội ngũ AI HCMUS  

---

## 📦 1. TỔNG QUAN DỮ LIỆU ĐA PHƯƠNG THỨC HIỆN CÓ (DATA ASSETS)

Toàn bộ dữ liệu của Đợt 1 (Batch 1 - 324 giờ video, 300+ video từ `L21` đến `L30`) đã được tiền xử lý và lưu trữ trong thư mục `data/batch_1/processed/`:

```
data/batch_1/processed/
├── frames.parquet            (3.5 MB)   : 177,321 Keyframes mapping (video_id, frame_idx, pts_time, global_id)
├── siglip_features.npy       (408.5 MB) : Ma trận 177,321 x 1152 chiều (Google SigLIP 2 SO400M FP16 mmap)
├── clip_features.npy         (181.5 MB) : Ma trận 177,321 x 512 chiều (OpenAI CLIP ViT-B/32 FP32)
├── ocr_results.parquet       (5.0 MB)   : Văn bản chữ đọc được trên 177,321 Keyframes (PaddleOCR / EasyOCR)
├── transcripts.parquet       (4.0 MB)   : Lời thoại âm thanh nhận diện bởi Whisper ASR (kèm start_time, end_time)
├── object_summary.parquet    (2.1 MB)   : Danh sách vật thể nhận diện bởi YOLO (Object detections per frame)
├── videos.parquet            (358 KB)   : Metadata tổng thể của 300+ video
└── video_zip_map.json        (64 KB)    : Bản đồ vị trí lưu trữ file MP4 trong các gói Videos_*.zip
```

### Chi tiết các tệp dữ liệu:
1. **Khung hình Keyframes (`frames.parquet`):**
   - Tổng cộng **177,321 Keyframes** trích xuất từ hơn 300 video.
   - Mỗi dòng chứa: `global_id` (0 đến 177,320), `video_id`, `frame_idx`, `pts_time` (giây).
2. **Đặc trưng thị giác SigLIP 2 (`siglip_features.npy`):**
   - Trích xuất từ mô hình `google/siglip2-so400m-patch14-384`.
   - Lưu trữ dạng `float16` trên ổ cứng và nạp bằng cơ chế `np.memmap` (đọc ngẫu nhiên tức thì `<0.01ms`).
3. **Văn bản thị giác OCR (`ocr_results.parquet`):**
   - Chứa các chuỗi ký tự nhận diện được trên màn hình: Biển hiệu, bảng trao giải, phụ đề, tên riêng, tỷ số thể thao.
4. **Lời thoại âm thanh Whisper ASR (`transcripts.parquet`):**
   - Chứa các đoạn hội thoại có lời nói trong video kèm mốc thời gian bắt đầu và kết thúc chính xác đến mili-giây.
5. **Video gốc MP4 (`raw/batch_1/Videos/*.zip`):**
   - Các gói zip nén chứa video MP4 gốc (25 fps), được quản lý bởi `VideoZipManager` với bộ nhớ đệm cache 50 file trong `scratch/video_cache/`.

---

## ⚙️ 2. KIẾN TRÚC PIPELINE THỰC THI HIỆN TẠI (3-LAYER EXECUTION FLOW)

```
                       [ Câu Truy Vấn Tiếng Việt ]
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │ TẦNG 0: Gemini Router (Task Classification & Context)   │
       │ - Phân loại Task: KIS, QA, hay TRAKE                    │
       │ - Dịch sang tiếng Anh thị giác & trích xuất thực thể    │
       └────────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │ TẦNG 1: Coarse Retrieval (FAISS GPU Indexing)           │
       │ - Vector hóa câu hỏi bằng Text Encoder SigLIP-2 (1152d) │
       │ - Quét ma trận 177,321 Keyframes bằng Cosine Sim CUDA   │
       │ - Xuất ra Top 100 Candidates tiềm năng (<50ms)          │
       └────────────────────────────┬────────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
     [ CHUYÊN GIA KIS ]    [ CHUYÊN GIA QA ]    [ CHUYÊN GIA TRAKE ]
              │                     │                     │
              ▼                     ▼                     ▼
       ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
       │  TẦNG 2:     │      │  TẦNG 2:     │      │  TRAKE DP:   │
       │  Intra-Video │      │  Multi-Cue   │      │  Monotonic   │
       │  Temporal    │      │  ASR/OCR     │      │  Sequence DP │
       │  Gaussian    │      │  Timeline    │      │  E1 < E2...  │
       │  Smoothing   │      │  Fusion      │      │  En          │
       └──────┬───────┘      └──────┬───────┘      └──────┬───────┘
              │                     │                     │
              │              ┌──────┴──────┐              │
              │              │ Gemini VLM  │              │
              │              │ Answering   │              │
              │              │ (<100 chars)│              │
              │              └──────┬──────┘              │
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │ TẦNG 3: Multi-Signal Gated Dense Video Refinement       │
       │ - Blind Spot Gate: Phát hiện khoảng mù Keyframe (>75f)  │
       │ - OpenCV Vi Sai: Nhảy cóc vào MP4 đọc 15 frames         │
       │ - Mini-batch FP16 SigLIP-2: Tìm đỉnh thời gian chuẩn xác│
       └────────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
                    [ Xuất 24 File CSV Chuẩn BTC ]
```

---

## 📊 3. BẢNG SỐ LIỆU ĐO LƯỜNG TOÀN DIỆN TRÊN 20 TEST CASES

Dưới đây là dữ liệu thực nghiệm mới nhất được đo lường tự động trên toàn bộ 20 câu hỏi Ground Truth (`data/benchmark/ground_truth.json`):

| STT | Query ID | Task | Video Mục Tiêu | Video Rank | Frame Rank | R@1 | R@5 | R@20 | R@50 | Điểm BTC | Độ Trễ |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `test-kis-01` | KIS | `L28_V009 [15866-15977]` | **#1** | **#5** | 0.00 | 1.00 | 1.00 | 1.00 | **0.8000** | 6.2s |
| 2 | `test-qa-02` | QA | `L27_V002 [910-959]` | **#3** | **#29** | 0.00 | 0.00 | 0.00 | 1.00 | **0.4000** | 45.6s |
| 3 | `test-trake-03` | TRAKE | `L26_V008 (5 ev)` | **#1** | **#1** | 0.40 | 0.40 | 0.40 | 0.40 | **0.4000** | 4.5s |
| 4 | `test-kis-04` | KIS | `L26_V355 [4662-4727]` | **#1** | **#1** | 1.00 | 1.00 | 1.00 | 1.00 | **1.0000** | 1.7s |
| 5 | `test-kis-05` | KIS | `L27_V012 [11686-11732]` | **#2** | **#2** | 0.00 | 1.00 | 1.00 | 1.00 | **0.8000** | 1.9s |
| 6 | `test-kis-06` | KIS | `L27_V013 [10112-10140]` | **#2** | **#4** | 0.00 | 1.00 | 1.00 | 1.00 | **0.8000** | 2.0s |
| 7 | `test-qa-07` | QA | `L29_V013 [21438-22339]` | **#1** | **#2** | 0.00 | 1.00 | 1.00 | 1.00 | **0.8000** | 12.2s |
| 8 | `test-kis-08` | KIS | `L22_V003 [17762-17787]` | **#2** | **MISS** | 0.00 | 0.00 | 0.00 | 0.00 | **0.0000** | 1.9s |
| 9 | `test-trake-09` | TRAKE | `L22_V006 (5 ev)` | **#1** | **#1** | 0.40 | 0.40 | 0.40 | 0.40 | **0.4000** | 6.4s |
| 10 | `test-kis-10` | KIS | `L22_V001 [14158-15797]` | **#1** | **#2** | 0.00 | 1.00 | 1.00 | 1.00 | **0.8000** | 2.1s |
| 11 | `test-kis-11` | KIS | `L23_V017 [347-410]` | **#11** | **#22** | 0.00 | 0.00 | 0.00 | 1.00 | **0.4000** | 1.4s |
| 12 | `test-trake-12` | TRAKE | `L23_V018 (4 ev)` | **#3** | **#3** | 0.00 | 0.50 | 0.50 | 0.50 | **0.4000** | 3.3s |
| 13 | `test-kis-13` | KIS | `L23_V023 [9707-9788]` | **#1** | **#1** | 1.00 | 1.00 | 1.00 | 1.00 | **1.0000** | 1.8s |
| 14 | `test-qa-14` | QA | `L24_V020 [2684-3164]` | **#2** | **#9** | 0.00 | 0.00 | 1.00 | 1.00 | **0.6000** | 21.4s |
| 15 | `test-kis-15` | KIS | `L24_V041 [745-769]` | **#18** | **#46** | 0.00 | 0.00 | 0.00 | 1.00 | **0.4000** | 2.1s |
| 16 | `test-qa-16` | QA | `L24_V026 [13072-13181]` | **#4** | **MISS** | 0.00 | 0.00 | 0.00 | 0.00 | **0.0000** | 131s |
| 17 | `test-qa-17` | QA | `L25_V067 [2019-2039]` | **#9** | **MISS** | 0.00 | 0.00 | 0.00 | 0.00 | **0.0000** | 22.3s |
| 18 | `test-trake-18` | TRAKE | `L25_V041 (4 ev)` | **#50** | **#50** | 0.00 | 0.00 | 0.00 | 0.75 | **0.3000** | 5.0s |
| 19 | `test-kis-19` | KIS | `L25_V063 [2316-2342]` | **#14** | **MISS** | 0.00 | 0.00 | 0.00 | 0.00 | **0.0000** | 2.0s |
| 20 | `test-qa-20` | QA | `L25_V044 [2730-2748]` | **#21** | **#21** | 0.00 | 0.00 | 0.00 | 1.00 | **0.4000** | 53.2s |

---

### 📈 Thống kê tổng hợp:
- 🎯 **Điểm KIS trung bình (10 câu):** **`0.6000` (60.00%)**
- ❓ **Điểm QA trung bình (6 câu):** **`0.3667` (36.67%)**
- ⏱️ **Điểm TRAKE trung bình (4 câu):** **`0.3750` (37.50%)**
- 🏆 **Điểm BTC Final Score toàn diện:** **`0.4850` (48.50%)**
- 🔍 **Độ phủ Video Retrieval (Stage-1 Retriever):**
  - **Video Recall@5 :** **`70.0%`** (14/20 video đúng nằm trong Top 5)
  - **Video Recall@20:** **`90.0%`** (18/20 video đúng nằm trong Top 20)
  - **Video Recall@50:** **`100.0%`** (Toàn bộ 20/20 video đúng đều nằm trong Top 50)

---

## 🔬 4. PHÂN TÍCH CHUYÊN SÂU CÁC VẤN ĐỀ HIỆN HỮU (BOTTLENECK DIAGNOSIS)

Qua việc đối chiếu từng câu hỏi đạt điểm thấp với dữ liệu gốc, chúng ta xác định được 4 nhóm vấn đề cốt lõi:

### 🔴 Vấn đề 1: Trượt Video Rank ở các câu có Văn bản/Tên riêng (OCR & Named Entity Gap)
- **Các ca điển hình:**
  - `test-kis-19` (*"TRAO TẶNG QUỸ HỌC BỔNG TÀI NĂNG TRẺ"*) -> Video `L25_V063` bị xếp ở **Rank #14**.
  - `test-qa-17` (*"EHUD BARAK"*) -> Video `L25_V067` bị xếp ở **Rank #9**.
- **Nguyên nhân cốt lõi:**
  - Mô hình Dense Embedding (SigLIP 2) mã hóa bức ảnh dựa trên **ngữ cảnh tổng thể** (người đàn ông, sân khấu, hội trường). Nó không thể "đọc" được chính xác từng ký tự chữ in hoa trên tấm biển nhỏ.
  - Khi không có cơ chế kết hợp từ khóa văn bản (Sparse Text Search / BM25) vào danh sách ứng viên Top 10 của Tầng 1, video chứa đúng văn bản bị các video có hình ảnh tương tự nhưng khác nội dung chữ chen lên trước.

### 🔴 Vấn đề 2: Lệch từ khóa nhận diện của mô hình VLM (Fine-grained Visual Ambiguity)
- **Ca điển hình:** `test-qa-16` (*"loài hoa màu vàng nào đang được đặt tại đây?"*).
  - Ground Truth: `"hoa cúc"` (Video `L24_V026`).
  - AI tìm kiếm: Đã tìm trúng Video `L24_V026` và Frame `13072` (Rank #7 và #15).
  - Kết quả VLM: Gemini nhìn vào bức ảnh hoa màu vàng trong góc đệm nhỏ và sinh câu trả lời: **`"Hoa mai vàng"`**.
- **Nguyên nhân:**
  - Gemini quan sát toàn bộ bức ảnh toàn cảnh (Full-frame) mà không được hướng dẫn zoom/crop vào vùng hoa ở góc đệm, dẫn đến việc phán đoán nhầm giữa hoa mai và hoa cúc.

### 🔴 Vấnned 3: Khoảng mù thời gian giữa các Keyframes (Temporal Blind Spot)
- **Ca điển hình:** `test-kis-08` (Nữ sinh cầm điện thoại màu tím).
  - Target: `L22_V003` (Khoảng chuẩn BTC: `[17762 - 17787]`).
  - Thực tế Keyframe: BTC chỉ trích xuất Keyframe tại `17694` và `17787` (cách nhau **93 frames = 3.7 giây**).
  - AI tìm kiếm: Đoán đúng Video `L22_V003` ở **Rank #2**, nhưng khung hình Keyframe `17694` chưa diễn ra hành động cầm điện thoại sát mặt $\rightarrow$ Bị 0 điểm Frame.
- **Bản chất:** Bắt buộc phải có Tầng 3 (Quét vi sai các khung hình ẩn bên trong file MP4) để bắt được Frame `17780`.

### 🔴 Vấn đề 4: Chuỗi sự kiện trải dài qua nhiều bối cảnh khác nhau (Multi-Scene DP Tracking)
- **Ca điển hình:** `test-trake-18` (Sân bóng rổ ngoài trời $\rightarrow$ Đàn Piano $\rightarrow$ Đàn Guitar $\rightarrow$ Dàn trống).
  - Target: `L25_V041` (Video clip giới thiệu trường học).
  - AI tìm kiếm: Bắt trúng 3/4 sự kiện nhưng video bị xếp ở **Rank #50**.
  - **Nguyên nhân:** Sự kiện 1 (sân bóng rổ) có hình ảnh thị giác hoàn toàn khác với các sự kiện âm nhạc trong nhà (piano, guitar, trống). Thuật toán cộng dồn điểm số hiện tại bị thiên lệch về các video chỉ tập trung vào một bối cảnh đồng nhất.

---

## 🎯 5. HƯỚNG NGHIÊN CỨU & KỸ THUẬT TIỀM NĂNG

Từ 4 vấn đề hiện hữu trên, các hướng kỹ thuật mang tính tổng quát (Generalizable Solutions) có thể xem xét:

1. **Hybrid Retrieval ở Tầng 1 (Dense SigLIP-2 + Sparse BM25 Fusion):**
   - Tự động phát hiện thực thể chữ viết (OCR Intent) để kết hợp điểm số BM25 trên `ocr_results.parquet` và `transcripts.parquet`, giúp đưa video chứa đúng chữ/lời thoại lên Top 5.
2. **Crop-and-Zoom VLM Inspection cho Visual QA:**
   - Khi câu hỏi nhắm vào chi tiết nhỏ (như loài hoa ở góc đệm), kết hợp thông tin tọa độ bounding box hoặc crop vùng quan tâm trước khi gửi cho Gemini VLM.
3. **Chuẩn hóa điểm số đa sự kiện (Normalized Multi-Event Score Aggregation) cho TRAKE:**
   - Áp dụng chuẩn hóa min-max hoặc softmax cho từng sự kiện con trước khi chạy Quy hoạch động Monotonic DP, tránh việc 1 sự kiện có điểm số quá cao lấn át các sự kiện khác.
4. **Khai thác Tầng 3 Vi Sai On-Demand trên Streamlit:**
   - Tận dụng giao diện Kính Lúp Vi Sai để người dùng kiểm duyệt nhanh các video nằm trong Top 5 có khoảng mù lớn mà không làm nóng máy.
