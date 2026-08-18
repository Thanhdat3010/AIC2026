# 📊 BÁO CÁO ĐO LƯỜNG & CHẨN ĐOÁN BOTTLENECK ABLATION STUDY (AIC 2026)

> **Thời gian cập nhật:** `2026-08-18 19:30:26`  
> **Tập dữ liệu kiểm chuẩn:** `data/benchmark/ground_truth.json` (20 Test Cases chuẩn BTC)  
> **Chẩn đoán:** So sánh trực tiếp **Stage-1 Video Recall (Retriever)** vs **Frame Recall (BTC Official Score)**

---

## 🏆 1. MA TRẬN CHẨN ĐOÁN HIỆU SUẤT VÀ NÚT THẮT (BOTTLENECK DIAGNOSIS MATRIX)

| # | Cấu hình Thử nghiệm | 🏆 BTC Final Score | V-MRR | V-R@1 | V-R@5 | V-R@20 | Latency | Đánh giá Nút thắt |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **18** | Cấu hình 18 (Baseline Toàn Diện): Config 17 + Optimized DP (TRAKE) + Tri-modal QA (Adaptive Gating) | **0.4790** | 0.3936 | 25.0% | 55.0% | 80.0% | 8943ms | Thử nghiệm |
| **19** | Cấu hình 19 (Ablation 1): Config 18 + Multi-Query FAISS Union (TRAKE Top-50) | **0.4890** | 0.3939 | 25.0% | 55.0% | 80.0% | 6093ms | Thử nghiệm |
| **20** | Cấu hình 20 (Ablation 2): Config 19 + Temporal NMS Event Coverage & Soft-Min (TRAKE) | **0.5050** | 0.5244 | 40.0% | 70.0% | 80.0% | 11584ms | Thử nghiệm |
| **21** | Cấu hình 21 (Ablation 3): Config 20 + Row-Normalized Monotonic DP (TRAKE) | **0.5325** | 0.5650 | 45.0% | 65.0% | 90.0% | 6283ms | Thử nghiệm |
| **22** | Cấu hình 22 (SOTA Master V4): Config 21 + EERCF TIB (QA Top-50 Normalized) + Diverse 3-Frame VLM | **0.4725** | 0.5084 | 40.0% | 65.0% | 90.0% | 6765ms | Thử nghiệm |

---

## 🔍 2. CHI TIẾT TỪNG QUERY: SO SÁNH VIDEO RANK VS FRAME RANK

### 🧪 Cấu hình 18: Cấu hình 18 (Baseline Toàn Diện): Config 17 + Optimized DP (TRAKE) + Tri-modal QA (Adaptive Gating)

- **BTC Final Score:** `0.4790` | **Video MRR:** `0.3936` | **Video Recall@5:** `55.0%`

| Query ID | Task | Target Video | Video Rank | Pos Rank | Perf Rank | R@1 | R@5 | R@20 | R@50 | R@100 | Final Score | Latency | Error Type |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| test-kis-01 | KIS | L28_V009 [15866-15977] | **#1** | #4 | #4 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.8000** | 6054ms | `PERFECT_IN_TOP5` |
| test-qa-02 | QA | L27_V002 [910-959] | **#2** | #10 | #10 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | **0.6000** | 10326ms | `PARTIAL_HIT` |
| test-trake-03 | TRAKE | L26_V008 (5 ev) | **#13** | MISS | MISS | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.0000** | 3160ms | `FRAME_MISS` |
| test-kis-04 | KIS | L26_V355 [4662-4727] | **#1** | #1 | #1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.0000** | 1662ms | `PERFECT_RANK_1` |
| test-kis-05 | KIS | L27_V012 [11686-11732] | **#8** | #35 | #35 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | **0.4000** | 1667ms | `PARTIAL_HIT` |
| test-kis-06 | KIS | L27_V013 [10112-10140] | **#2** | #7 | #7 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | **0.6000** | 1866ms | `PARTIAL_HIT` |
| test-qa-07 | QA | L29_V013 [21438-22339] | **#1** | #1 | #1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.0000** | 11414ms | `PERFECT_RANK_1` |
| test-kis-08 | KIS | L22_V003 [17762-17787] | **#51** | MISS | MISS | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.0000** | 1787ms | `VIDEO_LATE (>Top20)` |
| test-trake-09 | TRAKE | L22_V006 (5 ev) | **#2** | #2 | MISS | 0.00 | 0.60 | 0.60 | 0.60 | 0.60 | **0.4800** | 3442ms | `PARTIAL_HIT` |
| test-kis-10 | KIS | L22_V001 [14158-15797] | **#1** | #1 | #1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.0000** | 2312ms | `PERFECT_RANK_1` |
| test-kis-11 | KIS | L23_V017 [347-410] | **#11** | #25 | #25 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | **0.4000** | 1647ms | `PARTIAL_HIT` |
| test-trake-12 | TRAKE | L23_V018 (4 ev) | **#13** | #13 | MISS | 0.00 | 0.00 | 0.50 | 0.50 | 0.50 | **0.3000** | 3441ms | `PARTIAL_HIT` |
| test-kis-13 | KIS | L23_V023 [9707-9788] | **#1** | #1 | #1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.0000** | 1554ms | `PERFECT_RANK_1` |
| test-qa-14 | QA | L24_V020 [2684-3164] | **#3** | #8 | #8 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | **0.6000** | 56324ms | `PARTIAL_HIT` |
| test-kis-15 | KIS | L24_V041 [745-769] | **#26** | #90 | #90 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | **0.2000** | 1910ms | `VIDEO_LATE (>Top20)` |
| test-qa-16 | QA | L24_V026 [13072-13181] | **#4** | MISS | MISS | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.0000** | 10752ms | `TEMPORAL_NEAR_MISS (<=25f)` |
| test-qa-17 | QA | L25_V067 [2019-2039] | **MISS** | MISS | MISS | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.0000** | 11804ms | `VIDEO_NOT_IN_TOP100` |
| test-trake-18 | TRAKE | L25_V041 (4 ev) | **#4** | #4 | MISS | 0.00 | 0.75 | 0.75 | 0.75 | 0.75 | **0.6000** | 3514ms | `PARTIAL_HIT` |
| test-kis-19 | KIS | L25_V063 [2316-2342] | **MISS** | MISS | MISS | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.0000** | 2648ms | `VIDEO_NOT_IN_TOP100` |
| test-qa-20 | QA | L25_V044 [2730-2748] | **#9** | #19 | #19 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | **0.6000** | 41579ms | `PARTIAL_HIT` |


### 🧪 Cấu hình 19: Cấu hình 19 (Ablation 1): Config 18 + Multi-Query FAISS Union (TRAKE Top-50)

- **BTC Final Score:** `0.4890` | **Video MRR:** `0.3939` | **Video Recall@5:** `55.0%`

| Query ID | Task | Target Video | Video Rank | Pos Rank | Perf Rank | R@1 | R@5 | R@20 | R@50 | R@100 | Final Score | Latency | Error Type |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| test-kis-01 | KIS | L28_V009 [15866-15977] | **#1** | #4 | #4 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.8000** | 1937ms | `PERFECT_IN_TOP5` |
| test-qa-02 | QA | L27_V002 [910-959] | **#2** | #6 | #6 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | **0.6000** | 11279ms | `PARTIAL_HIT` |
| test-trake-03 | TRAKE | L26_V008 (5 ev) | **#11** | MISS | MISS | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.0000** | 2681ms | `FRAME_MISS` |
| test-kis-04 | KIS | L26_V355 [4662-4727] | **#1** | #1 | #1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.0000** | 1619ms | `PERFECT_RANK_1` |
| test-kis-05 | KIS | L27_V012 [11686-11732] | **#7** | #9 | #9 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | **0.6000** | 1690ms | `PARTIAL_HIT` |
| test-kis-06 | KIS | L27_V013 [10112-10140] | **#3** | #18 | #18 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | **0.6000** | 1747ms | `PARTIAL_HIT` |
| test-qa-07 | QA | L29_V013 [21438-22339] | **#1** | #1 | #1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.0000** | 10648ms | `PERFECT_RANK_1` |
| test-kis-08 | KIS | L22_V003 [17762-17787] | **MISS** | MISS | MISS | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.0000** | 1792ms | `VIDEO_NOT_IN_TOP100` |
| test-trake-09 | TRAKE | L22_V006 (5 ev) | **#2** | #2 | MISS | 0.00 | 0.60 | 0.60 | 0.60 | 0.60 | **0.4800** | 3929ms | `PARTIAL_HIT` |
| test-kis-10 | KIS | L22_V001 [14158-15797] | **#1** | #2 | #2 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.8000** | 2151ms | `PERFECT_IN_TOP5` |
| test-kis-11 | KIS | L23_V017 [347-410] | **#11** | #25 | #25 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | **0.4000** | 1720ms | `PARTIAL_HIT` |
| test-trake-12 | TRAKE | L23_V018 (4 ev) | **#14** | #14 | MISS | 0.00 | 0.00 | 0.50 | 0.50 | 0.50 | **0.3000** | 3283ms | `PARTIAL_HIT` |
| test-kis-13 | KIS | L23_V023 [9707-9788] | **#1** | #1 | #1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.0000** | 1467ms | `PERFECT_RANK_1` |
| test-qa-14 | QA | L24_V020 [2684-3164] | **#2** | #8 | #8 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | **0.6000** | 39006ms | `PARTIAL_HIT` |
| test-kis-15 | KIS | L24_V041 [745-769] | **#21** | #48 | #48 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | **0.4000** | 1581ms | `VIDEO_LATE (>Top20)` |
| test-qa-16 | QA | L24_V026 [13072-13181] | **#4** | MISS | MISS | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.0000** | 10170ms | `TEMPORAL_NEAR_MISS (<=25f)` |
| test-qa-17 | QA | L25_V067 [2019-2039] | **MISS** | MISS | MISS | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.0000** | 10028ms | `VIDEO_NOT_IN_TOP100` |
| test-trake-18 | TRAKE | L25_V041 (4 ev) | **#4** | #4 | MISS | 0.00 | 0.75 | 0.75 | 0.75 | 0.75 | **0.6000** | 3532ms | `PARTIAL_HIT` |
| test-kis-19 | KIS | L25_V063 [2316-2342] | **MISS** | MISS | MISS | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.0000** | 3078ms | `VIDEO_NOT_IN_TOP100` |
| test-qa-20 | QA | L25_V044 [2730-2748] | **#10** | #15 | #15 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | **0.6000** | 8512ms | `PARTIAL_HIT` |


### 🧪 Cấu hình 20: Cấu hình 20 (Ablation 2): Config 19 + Temporal NMS Event Coverage & Soft-Min (TRAKE)

- **BTC Final Score:** `0.5050` | **Video MRR:** `0.5244` | **Video Recall@5:** `70.0%`

| Query ID | Task | Target Video | Video Rank | Pos Rank | Perf Rank | R@1 | R@5 | R@20 | R@50 | R@100 | Final Score | Latency | Error Type |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| test-kis-01 | KIS | L28_V009 [15866-15977] | **#1** | #4 | #4 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.8000** | 1799ms | `PERFECT_IN_TOP5` |
| test-qa-02 | QA | L27_V002 [910-959] | **#3** | #15 | #15 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | **0.6000** | 10633ms | `PARTIAL_HIT` |
| test-trake-03 | TRAKE | L26_V008 (5 ev) | **#1** | MISS | MISS | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.0000** | 2622ms | `FRAME_MISS` |
| test-kis-04 | KIS | L26_V355 [4662-4727] | **#1** | #1 | #1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.0000** | 1602ms | `PERFECT_RANK_1` |
| test-kis-05 | KIS | L27_V012 [11686-11732] | **#1** | #1 | #1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.0000** | 1644ms | `PERFECT_RANK_1` |
| test-kis-06 | KIS | L27_V013 [10112-10140] | **#2** | #6 | #6 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | **0.6000** | 2211ms | `PARTIAL_HIT` |
| test-qa-07 | QA | L29_V013 [21438-22339] | **#1** | #1 | #1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.0000** | 34527ms | `PERFECT_RANK_1` |
| test-kis-08 | KIS | L22_V003 [17762-17787] | **#2** | MISS | MISS | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.0000** | 1760ms | `FRAME_MISS` |
| test-trake-09 | TRAKE | L22_V006 (5 ev) | **#1** | #1 | MISS | 0.80 | 0.80 | 0.80 | 0.80 | 0.80 | **0.8000** | 3649ms | `PARTIAL_HIT` |
| test-kis-10 | KIS | L22_V001 [14158-15797] | **#1** | #2 | #2 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.8000** | 2348ms | `PERFECT_IN_TOP5` |
| test-kis-11 | KIS | L23_V017 [347-410] | **#11** | #25 | #25 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | **0.4000** | 1413ms | `PARTIAL_HIT` |
| test-trake-12 | TRAKE | L23_V018 (4 ev) | **#3** | #3 | MISS | 0.00 | 0.50 | 0.50 | 0.50 | 0.50 | **0.4000** | 3423ms | `PARTIAL_HIT` |
| test-kis-13 | KIS | L23_V023 [9707-9788] | **#1** | #1 | #1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.0000** | 1901ms | `PERFECT_RANK_1` |
| test-qa-14 | QA | L24_V020 [2684-3164] | **#3** | #8 | #8 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | **0.6000** | 11590ms | `PARTIAL_HIT` |
| test-kis-15 | KIS | L24_V041 [745-769] | **#25** | #73 | #73 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | **0.2000** | 1712ms | `VIDEO_LATE (>Top20)` |
| test-qa-16 | QA | L24_V026 [13072-13181] | **#4** | MISS | MISS | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.0000** | 68674ms | `TEMPORAL_NEAR_MISS (<=25f)` |
| test-qa-17 | QA | L25_V067 [2019-2039] | **MISS** | MISS | MISS | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.0000** | 10962ms | `VIDEO_NOT_IN_TOP100` |
| test-trake-18 | TRAKE | L25_V041 (4 ev) | **#22** | #22 | MISS | 0.00 | 0.00 | 0.00 | 0.75 | 0.75 | **0.3000** | 3507ms | `VIDEO_LATE (>Top20)` |
| test-kis-19 | KIS | L25_V063 [2316-2342] | **MISS** | MISS | MISS | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.0000** | 2681ms | `VIDEO_NOT_IN_TOP100` |
| test-qa-20 | QA | L25_V044 [2730-2748] | **#16** | #20 | #20 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | **0.6000** | 63014ms | `PARTIAL_HIT` |


### 🧪 Cấu hình 21: Cấu hình 21 (Ablation 3): Config 20 + Row-Normalized Monotonic DP (TRAKE)

- **BTC Final Score:** `0.5325` | **Video MRR:** `0.5650` | **Video Recall@5:** `65.0%`

| Query ID | Task | Target Video | Video Rank | Pos Rank | Perf Rank | R@1 | R@5 | R@20 | R@50 | R@100 | Final Score | Latency | Error Type |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| test-kis-01 | KIS | L28_V009 [15866-15977] | **#1** | #5 | #5 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.8000** | 2123ms | `PERFECT_IN_TOP5` |
| test-qa-02 | QA | L27_V002 [910-959] | **#2** | #11 | #11 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | **0.6000** | 11414ms | `PARTIAL_HIT` |
| test-trake-03 | TRAKE | L26_V008 (5 ev) | **#1** | MISS | MISS | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.0000** | 2588ms | `FRAME_MISS` |
| test-kis-04 | KIS | L26_V355 [4662-4727] | **#1** | #1 | #1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.0000** | 1626ms | `PERFECT_RANK_1` |
| test-kis-05 | KIS | L27_V012 [11686-11732] | **#1** | #1 | #1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.0000** | 1638ms | `PERFECT_RANK_1` |
| test-kis-06 | KIS | L27_V013 [10112-10140] | **#2** | #9 | #9 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | **0.6000** | 1763ms | `PARTIAL_HIT` |
| test-qa-07 | QA | L29_V013 [21438-22339] | **#1** | #1 | #1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.0000** | 11812ms | `PERFECT_RANK_1` |
| test-kis-08 | KIS | L22_V003 [17762-17787] | **#2** | MISS | MISS | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.0000** | 1771ms | `FRAME_MISS` |
| test-trake-09 | TRAKE | L22_V006 (5 ev) | **#1** | #1 | MISS | 0.60 | 0.60 | 0.60 | 0.60 | 0.60 | **0.6000** | 3759ms | `PARTIAL_HIT` |
| test-kis-10 | KIS | L22_V001 [14158-15797] | **#1** | #2 | #2 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.8000** | 2178ms | `PERFECT_IN_TOP5` |
| test-kis-11 | KIS | L23_V017 [347-410] | **#11** | #25 | #25 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | **0.4000** | 1418ms | `PARTIAL_HIT` |
| test-trake-12 | TRAKE | L23_V018 (4 ev) | **#3** | #3 | MISS | 0.00 | 0.50 | 0.50 | 0.50 | 0.50 | **0.4000** | 3223ms | `PARTIAL_HIT` |
| test-kis-13 | KIS | L23_V023 [9707-9788] | **#1** | #1 | #1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.0000** | 1652ms | `PERFECT_RANK_1` |
| test-qa-14 | QA | L24_V020 [2684-3164] | **#1** | #1 | #1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.0000** | 12959ms | `PERFECT_RANK_1` |
| test-kis-15 | KIS | L24_V041 [745-769] | **#15** | #36 | #36 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | **0.4000** | 1638ms | `PARTIAL_HIT` |
| test-qa-16 | QA | L24_V026 [13072-13181] | **#6** | MISS | MISS | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.0000** | 9366ms | `TEMPORAL_NEAR_MISS (<=25f)` |
| test-qa-17 | QA | L25_V067 [2019-2039] | **MISS** | MISS | MISS | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.0000** | 10076ms | `VIDEO_NOT_IN_TOP100` |
| test-trake-18 | TRAKE | L25_V041 (4 ev) | **#14** | #14 | MISS | 0.00 | 0.00 | 0.75 | 0.75 | 0.75 | **0.4500** | 3552ms | `PARTIAL_HIT` |
| test-kis-19 | KIS | L25_V063 [2316-2342] | **MISS** | MISS | MISS | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.0000** | 2844ms | `VIDEO_NOT_IN_TOP100` |
| test-qa-20 | QA | L25_V044 [2730-2748] | **#14** | #18 | #18 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | **0.6000** | 38263ms | `PARTIAL_HIT` |


### 🧪 Cấu hình 22: Cấu hình 22 (SOTA Master V4): Config 21 + EERCF TIB (QA Top-50 Normalized) + Diverse 3-Frame VLM

- **BTC Final Score:** `0.4725` | **Video MRR:** `0.5084` | **Video Recall@5:** `65.0%`

| Query ID | Task | Target Video | Video Rank | Pos Rank | Perf Rank | R@1 | R@5 | R@20 | R@50 | R@100 | Final Score | Latency | Error Type |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| test-kis-01 | KIS | L28_V009 [15866-15977] | **#1** | #4 | #4 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.8000** | 1903ms | `PERFECT_IN_TOP5` |
| test-qa-02 | QA | L27_V002 [910-959] | **#2** | #8 | #8 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | **0.6000** | 10829ms | `PARTIAL_HIT` |
| test-trake-03 | TRAKE | L26_V008 (5 ev) | **#1** | MISS | MISS | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.0000** | 2843ms | `FRAME_MISS` |
| test-kis-04 | KIS | L26_V355 [4662-4727] | **#1** | #1 | #1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.0000** | 1538ms | `PERFECT_RANK_1` |
| test-kis-05 | KIS | L27_V012 [11686-11732] | **#12** | #28 | #28 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | **0.4000** | 1696ms | `PARTIAL_HIT` |
| test-kis-06 | KIS | L27_V013 [10112-10140] | **#3** | #24 | #24 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | **0.4000** | 1830ms | `PARTIAL_HIT` |
| test-qa-07 | QA | L29_V013 [21438-22339] | **#1** | #1 | #1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.0000** | 11333ms | `PERFECT_RANK_1` |
| test-kis-08 | KIS | L22_V003 [17762-17787] | **#2** | MISS | MISS | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.0000** | 1888ms | `FRAME_MISS` |
| test-trake-09 | TRAKE | L22_V006 (5 ev) | **#1** | #1 | MISS | 0.60 | 0.60 | 0.60 | 0.60 | 0.60 | **0.6000** | 3982ms | `PARTIAL_HIT` |
| test-kis-10 | KIS | L22_V001 [14158-15797] | **#1** | #2 | #2 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.8000** | 2279ms | `PERFECT_IN_TOP5` |
| test-kis-11 | KIS | L23_V017 [347-410] | **#16** | #45 | #45 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | **0.4000** | 1636ms | `PARTIAL_HIT` |
| test-trake-12 | TRAKE | L23_V018 (4 ev) | **#4** | #4 | MISS | 0.00 | 0.50 | 0.50 | 0.50 | 0.50 | **0.4000** | 3269ms | `PARTIAL_HIT` |
| test-kis-13 | KIS | L23_V023 [9707-9788] | **#1** | #1 | #1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.0000** | 1548ms | `PERFECT_RANK_1` |
| test-qa-14 | QA | L24_V020 [2684-3164] | **#1** | #14 | #14 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | **0.6000** | 26255ms | `PARTIAL_HIT` |
| test-kis-15 | KIS | L24_V041 [745-769] | **#15** | #36 | #36 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | **0.4000** | 1713ms | `PARTIAL_HIT` |
| test-qa-16 | QA | L24_V026 [13072-13181] | **#4** | MISS | MISS | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.0000** | 10044ms | `TEMPORAL_NEAR_MISS (<=25f)` |
| test-qa-17 | QA | L25_V067 [2019-2039] | **MISS** | MISS | MISS | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.0000** | 11185ms | `VIDEO_NOT_IN_TOP100` |
| test-trake-18 | TRAKE | L25_V041 (4 ev) | **#17** | #17 | MISS | 0.00 | 0.00 | 0.75 | 0.75 | 0.75 | **0.4500** | 7173ms | `PARTIAL_HIT` |
| test-kis-19 | KIS | L25_V063 [2316-2342] | **MISS** | MISS | MISS | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.0000** | 3750ms | `VIDEO_NOT_IN_TOP100` |
| test-qa-20 | QA | L25_V044 [2730-2748] | **#16** | #20 | #20 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | **0.6000** | 28605ms | `PARTIAL_HIT` |

