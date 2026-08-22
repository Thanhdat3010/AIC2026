# 🏆 BÁO CÁO TỔNG KẾT THỰC NGHIỆM ABLATION STUDY VÀ HỆ THỐNG SOTA AIC 2026

Tài liệu này tổng kết toàn bộ kết quả thực nghiệm **Ma trận Ablation Study Lũy tiến & Đối đầu Độc lập (A0 $\rightarrow$ A7)** và các nâng cấp cốt lõi về **DIEM (CVPR 2024) / ROCLING 2025 Query Decomposition**, **Pure Vector-Driven Top 100**, **TNCA Temporal Context**, và **D3TW (CVPR 2019) Monotonic DP** trên tập kiểm thử chuẩn 47 câu Ground Truth của cuộc thi AI Challenge 2026.

---

## 📊 1. BẢNG TỔNG SẮP THỰC NGHIỆM ĐỐI ĐẦU ABLATION STUDY

| Rank | Cấu hình | Mô tả Thành phần Kỹ thuật (Paper) | KIS Score | QA Score | TRAKE Score | 🏆 Macro BTC | Video-R@1 | Video-R@5 | Video-R@20 | Video-R@100 | Latency | Đánh giá & Đóng góp |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 | **`A7` (Final)** | **🏆 Grand Master SOTA Engine** | **0.7286** | **0.4286** | **0.3800** | **0.6021** | **53.2%** | **74.5%** | **83.0%** | **95.7%** | **5.5s** | 🏆 **QUÁN QUÂN - VƯỢT NGƯỠNG 0.60+ MACRO** |
| 🥈 | **`A6_1`** | **+ DIEM Query Decomposition (T1)** | 0.7214 | **0.4286** | 0.3800 | 0.5979 | **55.3%** | 74.5% | 83.0% | 93.6% | 5.9s | QA Score tăng vọt gấp 2.5 lần |
| 🥉 | **`A6_2`** | **+ TVR/WACV Temporal Grounding (T2)** | 0.7286 | 0.3000 | 0.3800 | 0.5638 | 53.2% | 70.2% | 83.0% | 93.6% | 5.6s | Khóa mốc giây ASR nội video |
| #4 | **`A6`** | **+ Unified TNCA + Multimodal Cascade** | 0.7357 | 0.1714 | 0.3800 | 0.5298 | 55.3% | 68.1% | 80.9% | 85.1% | 6.6s | Mốc SOTA tiền đề |
| #5 | **`A5`** | **+ Joint Coverage Viterbi DP (T4)** | 0.7214 | 0.1714 | 0.3800 | 0.5213 | 51.1% | 70.2% | **85.1%** | 89.4% | 7.7s | Nâng 3x điểm số TRAKE |
| #6 | **`A4`** | **+ Unified Multimodal QA (Gemini 3.5 Lite)** | **0.7500** | 0.1857 | 0.1220 | 0.5151 | 51.1% | 66.0% | 80.9% | 87.2% | 5.9s | KIS Score đỉnh cao nhất (0.7500) |
| #7 | **`A2`** | **+ Temporal Neighbor Context (TNCA $[t\pm30s]$)** | 0.7429 | 0.0000 | 0.1220 | 0.4555 | 46.8% | 66.0% | 80.9% | 93.6% | 1.9s | Triệt tiêu False Positives |
| #8 | **`A1`** | **+ Dual Text Embedding ($0.7\text{vi} + 0.3\text{en}$)** | 0.7286 | 0.0000 | 0.1220 | 0.4470 | 46.8% | 66.0% | 78.7% | 93.6% | 1.8s | Vượt qua Baseline SigLIP B1 |
| #9 | **`B1`** | **Pure Visual SigLIP-2 (1152d)** | 0.7214 | 0.0000 | 0.0820 | 0.4385 | 51.1% | 74.5% | 83.0% | 91.5% | **0.13s** | Mốc nền tảng thị giác |
| #10 | **`A0`** | **Baseline SigLIP-2 (Query Tiếng Việt)** | 0.6429 | 0.0000 | 0.1220 | 0.3960 | 42.6% | 63.8% | 72.3% | 91.5% | **0.12s** | Mốc kiểm chuẩn gốc |

---

## 🔍 2. CÁC NÂNG CẤP KIẾN TRÚC ĐỘT PHÁ CỦA A7

### 1. DIEM (CVPR 2024) / ROCLING 2025 Query Decomposition & Rewriting
* Tách tự động câu hỏi QA thành **Mệnh đề Bối cảnh Thị giác Tinh khiết (`visual_scene_vi`, `visual_scene_en`)** và **Câu hỏi Trực diện (`qa_direct_question`)**.
* Loại bỏ hiện tượng pha loãng Self-Attention trong Transformer Text Encoder của SigLIP-2, đưa điểm số **QA Score tăng vọt từ $0.1714 \rightarrow \mathbf{0.4286}$ (gấp 2.5 lần)**!

### 2. Triết Lý Xếp Hạng Thuần Vector (Pure Vector-Driven Top 100)
* Xóa bỏ hoàn toàn việc chia slot nhân tạo (`k_vids = 10`, `frames_per_vid = 10`).
* Để không gian vector và hàm điểm liên tục của 177,321 keyframes tự quyết định thứ hạng Top 100 tự nhiên kết hợp MMR temporal deduplication.
* **Hệ quả**: **Video Recall@100 bứt phá lập kỷ lục mới $\mathbf{95.7\%}$**!

### 3. D3TW (CVPR 2019) / DTW (CVPR 2021) Segmental Monotonic DP Cho TRAKE
* Quy hoạch động Viterbi đảm bảo nghiệm xuất ra nghiêm ngặt $100\%$ tính đơn điệu thời gian $f(E_1) < f(E_2) < ... < f(E_N)$.
* Hàm điểm lồi chuẩn hóa $S_{\text{video}} = \alpha S_{\text{independent}} + (1-\alpha) S_{\text{monotonic}}$ nâng TRAKE Score từ $0.0820 \rightarrow \mathbf{0.3800}$.

---

## 🚀 3. HƯỚNG DẪN NỘP BÀI CHÍNH THỨC

Để sinh file zip nộp bài chính thức chuẩn 100% quy chế BTC cho toàn bộ đề thi:

```powershell
# Chạy dự đoán với Cấu hình Quán quân A7 (Grand Master SOTA)
C:\Users\Lenovo\miniconda3\envs\AIC2026\python.exe scripts/submission/run_submission.py --input query/SOTUYEN1-bo-de-thi --output_dir output/sotuyen1 --config A7
```

File nộp bài `output/sotuyen1/submission.zip` sẽ tự động được kiểm tra tính hợp lệ và đóng gói hoàn chỉnh sẵn sàng nộp lên cổng thi của BTC!
