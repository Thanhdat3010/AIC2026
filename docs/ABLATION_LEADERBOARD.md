# 🏆 BẢNG TỔNG SẮP CÁC CẤU HÌNH ĐẠT ĐIỂM CAO NHẤT (AIC 2026 LEADERBOARD)

> **Cập nhật lần cuối:** `2026-08-18 19:30:26` (Run ID: `20260818_193026`)

| Rank | Cấu hình | Final Score | Video MRR | Video-R@1 | Video-R@5 | Video-R@20 | Latency | Kết luận Chiến thuật |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 | **Cấu hình 21 (Ablation 3): Config 20 + Row-Normalized Monotonic DP (TRAKE)** | **0.5325** | 0.5650 | 45.0% | 65.0% | 90.0% | 6283ms | 🔥 KHUYÊN DÙNG THI ĐẤU |
| 🥈 | **Cấu hình 20 (Ablation 2): Config 19 + Temporal NMS Event Coverage & Soft-Min (TRAKE)** | **0.5050** | 0.5244 | 40.0% | 70.0% | 80.0% | 11584ms | Thử nghiệm |
| 🥉 | **Cấu hình 19 (Ablation 1): Config 18 + Multi-Query FAISS Union (TRAKE Top-50)** | **0.4890** | 0.3939 | 25.0% | 55.0% | 80.0% | 6093ms | Thử nghiệm |
| #4 | **Cấu hình 18 (Baseline Toàn Diện): Config 17 + Optimized DP (TRAKE) + Tri-modal QA (Adaptive Gating)** | **0.4790** | 0.3936 | 25.0% | 55.0% | 80.0% | 8943ms | Thử nghiệm |
| #5 | **Cấu hình 22 (SOTA Master V4): Config 21 + EERCF TIB (QA Top-50 Normalized) + Diverse 3-Frame VLM** | **0.4725** | 0.5084 | 40.0% | 65.0% | 90.0% | 6765ms | Thử nghiệm |