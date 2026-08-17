# 📊 BÁO CÁO ĐO LƯỜNG & CHẨN ĐOÁN BOTTLENECK ABLATION STUDY (AIC 2026)

> **Thời gian cập nhật:** `2026-08-17 17:44:51`  
> **Tập dữ liệu kiểm chuẩn:** `data/benchmark/ground_truth.json` (11 Test Cases chuẩn BTC)  
> **Chẩn đoán:** So sánh trực tiếp **Stage-1 Video Recall (Retriever)** vs **Frame Recall (BTC Official Score)**

---

## 🏆 1. MA TRẬN CHẨN ĐOÁN HIỆU SUẤT VÀ NÚT THẮT (BOTTLENECK DIAGNOSIS MATRIX)

| # | Cấu hình Thử nghiệm | 🏆 BTC Final Score | V-R@1 | V-R@5 | V-R@10 | V-R@20 | V-R@50 | Latency | Đánh giá Nút thắt |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | Baseline 0: BTC CLIP (512d) + Dịch từ điển thô | **0.2800** | 27.3% | 45.5% | 54.5% | 81.8% | 90.9% | 59ms | Cần Reranker đưa từ Top 5 lên #1 |
| **1** | Baseline 1: Google SigLIP 2 (1152d) + Dịch từ điển thô | **0.5600** | 45.5% | 72.7% | 72.7% | 81.8% | 90.9% | 329ms | Cần Reranker đưa từ Top 5 lên #1 |
| **1b** | Cấu hình 1b: SigLIP 2 + Single Gemini 3.5 Flash Lite Translation | **0.5236** | 54.5% | 81.8% | 90.9% | 90.9% | 90.9% | 2087ms | Cần Reranker đưa từ Top 5 lên #1 |
| **10** | Cấu hình 10: Full Combo Monolithic Pipeline (SigLIP 2 + MultiPrompt + Gating + QA + SoftFilter) | **0.5200** | 63.6% | 90.9% | 90.9% | 100.0% | 100.0% | 4236ms | Cần Reranker đưa từ Top 5 lên #1 |
| **11** | Cấu hình 11: 🚀 TASK-SPECIALIZED SOTA ARCHITECTURE (Chuyên biệt hóa KIS, QA, TRAKE + Gating thông minh) | **0.6000** | 81.8% | 81.8% | 90.9% | 90.9% | 100.0% | 4019ms | Retriever tốt |

---

## 🔍 2. CHI TIẾT TỪNG QUERY: SO SÁNH VIDEO RANK VS FRAME RANK

### 🧪 Cấu hình 0: Baseline 0: BTC CLIP (512d) + Dịch từ điển thô

- **BTC Final Score:** `0.2800` | **Video Recall@5:** `45.5%` | **Video Recall@20:** `81.8%`

| Query ID | Task | Target Video | Video Rank | Frame Rank | R@1 | R@5 | R@20 | R@50 | Final Score | Latency |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| test-kis-01 | KIS | L28_V009 [15866-15977] | **#1** | **#1** | 1.00 | 1.00 | 1.00 | 1.00 | 1.0000 | 88ms |
| test-qa-02 | QA | L27_V002 [910-959] | **#11** | **#14** | 0.00 | 0.00 | 1.00 | 1.00 | 0.6000 | 47ms |
| test-trake-03 | TRAKE | L26_V008 (5 ev) | **#1** | **#1** | 0.20 | 0.20 | 0.20 | 0.20 | 0.2000 | 52ms |
| test-kis-04 | KIS | L26_V355 [4662-4727] | **#5** | **MISS** | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 | 50ms |
| test-kis-05 | KIS | L27_V012 [11686-11732] | **#11** | **#86** | 0.00 | 0.00 | 0.00 | 0.00 | 0.2000 | 48ms |
| test-kis-06 | KIS | L27_V013 [10112-10140] | **#15** | **#45** | 0.00 | 0.00 | 0.00 | 1.00 | 0.4000 | 95ms |
| test-qa-07 | QA | L29_V013 [21438-22339] | **#1** | **MISS** | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 | 50ms |
| test-kis-08 | KIS | L22_V003 [17762-17787] | **MISS** | **MISS** | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 | 49ms |
| test-trake-09 | TRAKE | L22_V006 (5 ev) | **#42** | **#50** | 0.00 | 0.00 | 0.00 | 0.20 | 0.0800 | 62ms |
| test-kis-10 | KIS | L22_V001 [14158-15797] | **#6** | **#6** | 0.00 | 0.00 | 1.00 | 1.00 | 0.6000 | 56ms |
| test-kis-11 | KIS | L23_V017 [347-410] | **#2** | **MISS** | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 | 53ms |


### 🧪 Cấu hình 1: Baseline 1: Google SigLIP 2 (1152d) + Dịch từ điển thô

- **BTC Final Score:** `0.5600` | **Video Recall@5:** `72.7%` | **Video Recall@20:** `81.8%`

| Query ID | Task | Target Video | Video Rank | Frame Rank | R@1 | R@5 | R@20 | R@50 | Final Score | Latency |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| test-kis-01 | KIS | L28_V009 [15866-15977] | **#1** | **#2** | 0.00 | 1.00 | 1.00 | 1.00 | 0.8000 | 724ms |
| test-qa-02 | QA | L27_V002 [910-959] | **#2** | **#5** | 0.00 | 1.00 | 1.00 | 1.00 | 0.8000 | 282ms |
| test-trake-03 | TRAKE | L26_V008 (5 ev) | **#1** | **#2** | 0.00 | 0.20 | 0.20 | 0.20 | 0.1600 | 282ms |
| test-kis-04 | KIS | L26_V355 [4662-4727] | **#1** | **#1** | 1.00 | 1.00 | 1.00 | 1.00 | 1.0000 | 303ms |
| test-kis-05 | KIS | L27_V012 [11686-11732] | **#2** | **#2** | 0.00 | 1.00 | 1.00 | 1.00 | 0.8000 | 280ms |
| test-kis-06 | KIS | L27_V013 [10112-10140] | **#3** | **#4** | 0.00 | 1.00 | 1.00 | 1.00 | 0.8000 | 327ms |
| test-qa-07 | QA | L29_V013 [21438-22339] | **#1** | **#18** | 0.00 | 0.00 | 1.00 | 1.00 | 0.6000 | 282ms |
| test-kis-08 | KIS | L22_V003 [17762-17787] | **#26** | **MISS** | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 | 287ms |
| test-trake-09 | TRAKE | L22_V006 (5 ev) | **MISS** | **MISS** | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 | 283ms |
| test-kis-10 | KIS | L22_V001 [14158-15797] | **#1** | **#2** | 0.00 | 1.00 | 1.00 | 1.00 | 0.8000 | 278ms |
| test-kis-11 | KIS | L23_V017 [347-410] | **#11** | **#27** | 0.00 | 0.00 | 0.00 | 1.00 | 0.4000 | 294ms |


### 🧪 Cấu hình 1b: Cấu hình 1b: SigLIP 2 + Single Gemini 3.5 Flash Lite Translation

- **BTC Final Score:** `0.5236` | **Video Recall@5:** `81.8%` | **Video Recall@20:** `90.9%`

| Query ID | Task | Target Video | Video Rank | Frame Rank | R@1 | R@5 | R@20 | R@50 | Final Score | Latency |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| test-kis-01 | KIS | L28_V009 [15866-15977] | **#1** | **#6** | 0.00 | 0.00 | 1.00 | 1.00 | 0.6000 | 2175ms |
| test-qa-02 | QA | L27_V002 [910-959] | **#3** | **#12** | 0.00 | 0.00 | 1.00 | 1.00 | 0.6000 | 2173ms |
| test-trake-03 | TRAKE | L26_V008 (5 ev) | **#2** | **#4** | 0.00 | 0.20 | 0.20 | 0.20 | 0.1600 | 1986ms |
| test-kis-04 | KIS | L26_V355 [4662-4727] | **#1** | **#1** | 1.00 | 1.00 | 1.00 | 1.00 | 1.0000 | 1881ms |
| test-kis-05 | KIS | L27_V012 [11686-11732] | **#1** | **#1** | 1.00 | 1.00 | 1.00 | 1.00 | 1.0000 | 1839ms |
| test-kis-06 | KIS | L27_V013 [10112-10140] | **#1** | **#4** | 0.00 | 1.00 | 1.00 | 1.00 | 0.8000 | 1877ms |
| test-qa-07 | QA | L29_V013 [21438-22339] | **#1** | **#23** | 0.00 | 0.00 | 0.00 | 1.00 | 0.4000 | 2533ms |
| test-kis-08 | KIS | L22_V003 [17762-17787] | **#3** | **MISS** | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 | 2020ms |
| test-trake-09 | TRAKE | L22_V006 (5 ev) | **#71** | **MISS** | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 | 2240ms |
| test-kis-10 | KIS | L22_V001 [14158-15797] | **#1** | **#2** | 0.00 | 1.00 | 1.00 | 1.00 | 0.8000 | 2220ms |
| test-kis-11 | KIS | L23_V017 [347-410] | **#10** | **#26** | 0.00 | 0.00 | 0.00 | 1.00 | 0.4000 | 2007ms |


### 🧪 Cấu hình 10: Cấu hình 10: Full Combo Monolithic Pipeline (SigLIP 2 + MultiPrompt + Gating + QA + SoftFilter)

- **BTC Final Score:** `0.5200` | **Video Recall@5:** `90.9%` | **Video Recall@20:** `100.0%`

| Query ID | Task | Target Video | Video Rank | Frame Rank | R@1 | R@5 | R@20 | R@50 | Final Score | Latency |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| test-kis-01 | KIS | L28_V009 [15866-15977] | **#1** | **#6** | 0.00 | 0.00 | 1.00 | 1.00 | 0.6000 | 2126ms |
| test-qa-02 | QA | L27_V002 [910-959] | **#1** | **#6** | 0.00 | 0.00 | 1.00 | 1.00 | 0.6000 | 11064ms |
| test-trake-03 | TRAKE | L26_V008 (5 ev) | **#1** | **#2** | 0.00 | 0.20 | 0.20 | 0.20 | 0.1600 | 2418ms |
| test-kis-04 | KIS | L26_V355 [4662-4727] | **#1** | **#1** | 1.00 | 1.00 | 1.00 | 1.00 | 1.0000 | 2194ms |
| test-kis-05 | KIS | L27_V012 [11686-11732] | **#1** | **#1** | 1.00 | 1.00 | 1.00 | 1.00 | 1.0000 | 2285ms |
| test-kis-06 | KIS | L27_V013 [10112-10140] | **#2** | **#8** | 0.00 | 0.00 | 1.00 | 1.00 | 0.6000 | 2218ms |
| test-qa-07 | QA | L29_V013 [21438-22339] | **#1** | **#49** | 0.00 | 0.00 | 0.00 | 1.00 | 0.4000 | 14635ms |
| test-kis-08 | KIS | L22_V003 [17762-17787] | **#3** | **MISS** | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 | 2289ms |
| test-trake-09 | TRAKE | L22_V006 (5 ev) | **#3** | **#5** | 0.00 | 0.20 | 0.20 | 0.20 | 0.1600 | 2531ms |
| test-kis-10 | KIS | L22_V001 [14158-15797] | **#1** | **#2** | 0.00 | 1.00 | 1.00 | 1.00 | 0.8000 | 2652ms |
| test-kis-11 | KIS | L23_V017 [347-410] | **#15** | **#48** | 0.00 | 0.00 | 0.00 | 1.00 | 0.4000 | 2182ms |


### 🧪 Cấu hình 11: Cấu hình 11: 🚀 TASK-SPECIALIZED SOTA ARCHITECTURE (Chuyên biệt hóa KIS, QA, TRAKE + Gating thông minh)

- **BTC Final Score:** `0.6000` | **Video Recall@5:** `81.8%` | **Video Recall@20:** `90.9%`

| Query ID | Task | Target Video | Video Rank | Frame Rank | R@1 | R@5 | R@20 | R@50 | Final Score | Latency |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| test-kis-01 | KIS | L28_V009 [15866-15977] | **#1** | **#6** | 0.00 | 0.00 | 1.00 | 1.00 | 0.6000 | 1969ms |
| test-qa-02 | QA | L27_V002 [910-959] | **#1** | **#6** | 0.00 | 0.00 | 1.00 | 1.00 | 0.6000 | 11050ms |
| test-trake-03 | TRAKE | L26_V008 (5 ev) | **#1** | **MISS** | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 | 3264ms |
| test-kis-04 | KIS | L26_V355 [4662-4727] | **#1** | **#1** | 1.00 | 1.00 | 1.00 | 1.00 | 1.0000 | 1690ms |
| test-kis-05 | KIS | L27_V012 [11686-11732] | **#1** | **#1** | 1.00 | 1.00 | 1.00 | 1.00 | 1.0000 | 1695ms |
| test-kis-06 | KIS | L27_V013 [10112-10140] | **#1** | **#4** | 0.00 | 1.00 | 1.00 | 1.00 | 0.8000 | 2027ms |
| test-qa-07 | QA | L29_V013 [21438-22339] | **#1** | **#10** | 0.00 | 0.00 | 1.00 | 1.00 | 0.6000 | 12953ms |
| test-kis-08 | KIS | L22_V003 [17762-17787] | **#23** | **MISS** | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 | 2401ms |
| test-trake-09 | TRAKE | L22_V006 (5 ev) | **#1** | **#1** | 0.80 | 0.80 | 0.80 | 0.80 | 0.8000 | 3646ms |
| test-kis-10 | KIS | L22_V001 [14158-15797] | **#1** | **#2** | 0.00 | 1.00 | 1.00 | 1.00 | 0.8000 | 1797ms |
| test-kis-11 | KIS | L23_V017 [347-410] | **#10** | **#25** | 0.00 | 0.00 | 0.00 | 1.00 | 0.4000 | 1720ms |

