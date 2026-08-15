# 📖 HƯỚNG DẪN XÂY DỰNG TẬP TEST BENCHMARK CÓ GROUND TRUTH (AIC 2026)

> **Dành cho:** Thành viên nhóm phụ trách tạo dữ liệu kiểm thử (Evaluation / Benchmark).  
> **Mục tiêu:** Tạo ra một bộ câu hỏi kiểm thử có sẵn **Đáp án chuẩn (Ground Truth)** để đo lường chỉ số chính xác ($R@1, R@5, R@20, R@50, R@100$ và **Final Score**) của hệ thống mỗi khi nâng cấp mô hình.

---

## 🎯 1. TẠI SAO PHẢI TỰ LÀM TẬP TEST?

* Ban Tổ Chức (BTC) chỉ phát file câu hỏi đề thi (`query-xxx-kis.txt`) mà **KHÔNG phát file đáp án**.
* Nếu không có tập test có sẵn đáp án, nhóm sẽ **không biết** mô hình mới (như SigLIP, RAG, OCR...) làm tăng hay giảm điểm số so với baseline.
* Mục tiêu của bạn: **Tạo khoảng 20 – 50 câu test mẫu có đáp án chuẩn xác 100%**.

---

## 📐 2. QUY CÁCH CHUẨN ĐÁP ÁN THEO QUY ĐỊNH BTC

| Task | Đề bài cần viết | Đáp án cần ghi lại (Ground Truth) |
| :--- | :--- | :--- |
| **Task 1: KIS** | 1 đoạn văn miêu tả cảnh visual | `video_id`, `start_frame`, `end_frame` (Khoảng khung hình diễn ra cảnh đó) |
| **Task 2: QA** | 1 đoạn ngữ cảnh + 1 câu hỏi | `video_id`, `frame_idx`, `answer` (Câu trả lời ngắn) |
| **Task 3: TRAKE** | Chuỗi sự kiện $E_1, E_2, \dots, E_n$ | `video_id`, danh sách các khoảng khung hình $[s_1, e_1], [s_2, e_2]...$ |

---

## 🛠️ 3. QUY TRÌNH 4 BƯỚC ĐỂ TẠO 1 CÂU TEST (DÀNH CHO BẠN CỦA BẠN)

### 📌 Bước 1: Chọn một Video và Phân cảnh bất kỳ
1. Mở một video bất kỳ trong thư mục `Videos/` (hoặc mở folder ảnh `Keyframes/Lxx_Vxxx/`).
2. Chọn ra 1 phân cảnh rõ ràng, ấn tượng (Ví dụ: Một người mặc áo đỏ đang tưới hoa bên hiên nhà).

---

### 📌 Bước 2: Xác định Tọa độ Đáp án (`video_id` & `frame_idx`)
* **Tên video:** Xem tên file video (Ví dụ: `L21_V001`).
* **Khung hình diễn ra sự kiện:**
  * **Cách A (Nếu xem từ ảnh Keyframe):**
    * Nhìn thấy bức ảnh `Keyframes/L21_V001/003.jpg` có cảnh đó.
    * Mở file `raw/map-keyframes/L21_V001.csv`, tìm dòng `n = 3` $\to$ Cột `frame_idx` ghi là **`261`**.
    * Lấy khoảng ước lượng lúc cảnh diễn ra (ví dụ: từ frame `240` đến frame `280`).
  * **Cách B (Nếu xem video `.mp4`):**
    * Nhìn số giây lúc cảnh diễn ra (Ví dụ: từ giây `8.0s` đến `9.5s`).
    * Tính frame: $\text{Frame} = \text{Số giây} \times \text{FPS}$ (Ví dụ: $8.0 \times 30 = 240$, $9.5 \times 30 = 285$).

---

### 📌 Bước 3: Đặt câu hỏi / Viết câu miêu tả (Query)
Hãy đóng vai Ban Giám Khảo và viết câu miêu tả bằng tiếng Việt tự nhiên theo 1 trong các dạng bài:

#### 🔹 Dạng 1: KIS (Tìm kiếm chi tiết thị giác)
* *Câu query:* `"Đoạn clip quay cận cảnh một người phụ nữ mặc áo màu đỏ đang dùng bình tưới hoa bên hiên nhà gỗ."`
* *Đáp án:* `video_id: "L21_V001"`, `start_frame: 240`, `end_frame: 280`.

#### 🔹 Dạng 2: QA (Hỏi đáp thông tin)
* *Câu query:* `"Đoạn video quay cảnh bữa tiệc sinh nhật. Hỏi chiếc bánh kem có bao nhiêu cây nến?"`
* *Đáp án:* `video_id: "L22_V015"`, `frame_idx: 1450`, `answer: "5"` (hoặc `"năm cây nến"`).

#### 🔹 Dạng 3: TRAKE (Chuỗi hành động tuần tự)
* *Câu query:*  
  `"E1: Khoảnh khắc người đàn ông bắt đầu mở cửa xe."`  
  `"E2: Khoảnh khắc người đàn ông bước hẳn một chân xuống đường."`  
  `"E3: Khoảnh khắc cửa xe được đóng lại."`
* *Đáp án:* `video_id: "L25_V008"`, các khoảng tương ứng $[100, 120], [150, 170], [210, 230]$.

---

### 📌 Bước 4: Nhập vào file `data/benchmark/ground_truth.json`
Lưu toàn bộ câu hỏi và đáp án vào file cấu trúc JSON chuẩn dưới đây.

---

## 💾 4. CẤU TRÚC FILE DỮ LIỆU ĐÁP ÁN CHUẨN (`ground_truth.json`)

Tạo file: **`data/benchmark/ground_truth.json`** với mẫu sau:

```json
{
  "benchmark_version": "1.0",
  "created_by": "Team AIC 2026",
  "total_queries": 3,
  "test_cases": [
    {
      "query_id": "test-kis-01",
      "task_type": "kis",
      "query_text": "Tìm cảnh người phụ nữ mặc áo đỏ đang tưới hoa bên hiên nhà gỗ.",
      "ground_truth": {
        "video_id": "L21_V001",
        "start_frame": 240,
        "end_frame": 280
      }
    },
    {
      "query_id": "test-qa-01",
      "task_type": "qa",
      "query_text": "Trong đoạn video trao quà từ thiện của CLB FANA tại Khánh Hòa, hỏi xã này có tên là gì?",
      "ground_truth": {
        "video_id": "L26_V112",
        "frame_idx": 4520,
        "answer": "Khánh Bình"
      }
    },
    {
      "query_id": "test-trake-01",
      "task_type": "trake",
      "query_text": "E1: Khoảnh khắc đầu tiên bột được bỏ vào tô.\nE2: Khoảnh khắc miếng măng tây tiếp xúc với dầu.\nE3: Khoảnh khắc vớt măng tây ra dĩa.",
      "ground_truth": {
        "video_id": "L24_V043",
        "events": [
          {"event_id": "E1", "start_frame": 2300, "end_frame": 2360},
          {"event_id": "E2", "start_frame": 7900, "end_frame": 7960},
          {"event_id": "E3", "start_frame": 8500, "end_frame": 8560}
        ]
      }
    }
  ]
}
```

---

## 🌟 5. BÍ QUYẾT TẠO TẬP TEST CHẤT LƯỢNG CAO (GHI ĐIỂM)

Để tập test phản ánh đúng 100% độ khó của đề thi AIC thật, bạn nên phân bổ đa dạng các nhóm câu hỏi:

1. **Nhóm nhận diện màu sắc & trang phục (30%):** *"Người mặc áo xanh lá, đội mũ bảo hiểm vàng"*.
2. **Nhóm nhận diện chữ viết / Biển hiệu (OCR) (25%):** *"Xe tải có in dòng chữ Vinamilk"*, *"Biển hiệu tiệm cắt tóc có số điện thoại..."*.
3. **Nhóm hành động & chuỗi thời gian (25%):** *"Người đàn ông cúi xuống nhặt chìa khóa rồi lên xe phóng đi"*.
4. **Nhóm bối cảnh & phong cảnh (20%):** *"Cảnh flycam quay toàn cảnh cánh đồng lúa chín vàng lúc hoàng hôn"*.

---

## ⚡ 6. CÁCH CHẠY ĐÁNH GIÁ ĐIỂM SỐ TỰ ĐỘNG

Sau khi bạn của bạn đã điền xong file `ground_truth.json`, bất kỳ ai trong nhóm chỉ cần chạy 1 lệnh:

```bash
python scripts/evaluate_benchmark.py
```

Hệ thống sẽ tự động chạy qua toàn bộ tập test và in ra bảng tổng kết chuẩn theo công thức BTC:
```text
============================================================
           AIC 2026 BENCHMARK EVALUATION REPORT
============================================================
Total Test Queries: 25
------------------------------------------------------------
Recall@1  (R@1)  : 48.0%  (12/25 queries đúng ngay Top 1)
Recall@5  (R@5)  : 72.0%  (18/25 queries đúng trong Top 5)
Recall@20 (R@20) : 88.0%  (22/25 queries đúng trong Top 20)
Recall@100(R@100): 96.0%  (24/25 queries đúng trong Top 100)
------------------------------------------------------------
🏆 FINAL SCORE (BTC Formula): 0.768 / 1.000
============================================================
```

> [!TIP]
> Hãy lưu giữ file `ground_truth.json` cẩn thận. Mỗi khi nhóm thử nghiệm một Model mới (như SigLIP, RAG, OCR), chỉ cần chạy lại lệnh trên để so sánh xem Final Score tăng từ `0.768` lên bao nhiêu!
