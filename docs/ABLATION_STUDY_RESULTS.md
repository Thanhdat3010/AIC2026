# 📊 BÁO CÁO ĐO LƯỜNG & CHẨN ĐOÁN BOTTLENECK ABLATION STUDY (AIC 2026)

> **Thời gian cập nhật:** `2026-08-18 09:55:54`  
> **Tập dữ liệu kiểm chuẩn:** `data/benchmark/ground_truth.json` (11 Test Cases chuẩn BTC)  
> **Chẩn đoán:** So sánh trực tiếp **Stage-1 Video Recall (Retriever)** vs **Frame Recall (BTC Official Score)**

---

## 🏆 1. MA TRẬN CHẨN ĐOÁN HIỆU SUẤT VÀ NÚT THẮT (BOTTLENECK DIAGNOSIS MATRIX)

| # | Cấu hình Thử nghiệm | 🏆 BTC Final Score | V-R@1 | V-R@5 | V-R@10 | V-R@20 | Latency | Đánh giá Nút thắt |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **16** | Cấu hình 16 (Full 3-Layer Master): Config 15 + Layer 3 Gated Dense Video Refinement (OpenCV Vi Sai) | **0.7091** | 72.7% | 90.9% | 100.0% | 100.0% | 90596ms | 🔥 Cần Reranker kéo V-R@5 lên #1 |

---

## 🔍 2. CHI TIẾT TỪNG QUERY: SO SÁNH VIDEO RANK VS FRAME RANK

### 🧪 Cấu hình 16: Cấu hình 16 (Full 3-Layer Master): Config 15 + Layer 3 Gated Dense Video Refinement (OpenCV Vi Sai)

- **BTC Final Score:** `0.7091` | **Video Recall@5:** `90.9%` | **Video Recall@20:** `100.0%`

| Query ID | Task | Target Video | Video Rank | Frame Rank | R@1 | R@5 | R@20 | R@50 | Final Score | Latency |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: |
| test-kis-01 | KIS | L28_V009 [15866-15977] | **#1** | **#5** | 0.00 | 1.00 | 1.00 | 1.00 | 0.8000 | 149154ms |
| test-qa-02 | QA | L27_V002 [910-959] | **#1** | **#10** | 0.00 | 0.00 | 1.00 | 1.00 | 0.6000 | 11011ms |
| test-trake-03 | TRAKE | L26_V008 (5 ev) | **#1** | **#1** | 0.40 | 0.40 | 0.40 | 0.40 | 0.4000 | 8012ms |
| test-kis-04 | KIS | L26_V355 [4662-4727] | **#1** | **#1** | 1.00 | 1.00 | 1.00 | 1.00 | 1.0000 | 1991ms |
| test-kis-05 | KIS | L27_V012 [11686-11732] | **#3** | **#3** | 0.00 | 1.00 | 1.00 | 1.00 | 0.8000 | 115076ms |
| test-kis-06 | KIS | L27_V013 [10112-10140] | **#1** | **#2** | 0.00 | 1.00 | 1.00 | 1.00 | 0.8000 | 117381ms |
| test-qa-07 | QA | L29_V013 [21438-22339] | **#1** | **#1** | 1.00 | 1.00 | 1.00 | 1.00 | 1.0000 | 184929ms |
| test-kis-08 | KIS | L22_V003 [17762-17787] | **#2** | **MISS** | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 | 141166ms |
| test-trake-09 | TRAKE | L22_V006 (5 ev) | **#1** | **#1** | 0.80 | 0.80 | 0.80 | 0.80 | 0.8000 | 5979ms |
| test-kis-10 | KIS | L22_V001 [14158-15797] | **#1** | **#1** | 1.00 | 1.00 | 1.00 | 1.00 | 1.0000 | 142778ms |
| test-kis-11 | KIS | L23_V017 [347-410] | **#9** | **#18** | 0.00 | 0.00 | 1.00 | 1.00 | 0.6000 | 119076ms |

