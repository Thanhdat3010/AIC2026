# BẢNG SỐ LIỆU THỰC NGHIỆM ABLATION STUDY TRÊN GROUND TRUTH 2 (32 QUERIES)

## 1. Bảng Tổng Hợp Kết Quả Thực Nghiệm (Markdown Format)

| Configuration | Description / Module | KIS (22 Qs) | QA (7 Qs) | TRAKE (3 Qs) | **Macro Score** | **Δ vs Baseline** | Video R@1 | Video R@20 | Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| M0: Raw SigLIP-2 Baseline | Visual-only Zero-shot (Tiếng Việt) | 0.7364 | 0.0000 | 0.0444 | 0.5104 | — | 53.1% | 75.0% | 121ms |
| M1: + Dual-Lingual Refinement | Bilingual Embedding (0.7 Vi + 0.3 En) | 0.8455 | 0.0000 | 0.2000 | 0.6000 | +0.0896 | 62.5% | 90.6% | 118ms |
| M2: + TNCA Window | Temporal Neighbor Support [t-30s, t+30s] | 0.8364 | 0.0000 | 0.2000 | 0.5938 | +0.0833 | 62.5% | 90.6% | 131ms |
| M3: + Multimodal Fusion (Static) | OCR & Whisper ASR (fixed weights) | 0.8364 | 0.0000 | 0.2000 | 0.5938 | +0.0833 | 68.8% | 93.8% | 242ms |
| M4: + Dynamic Modality Gating | Continuous Intent Softmax Gating | 0.8364 | 0.0000 | 0.2000 | 0.5938 | +0.0833 | 68.8% | 93.8% | 200ms |
| M5: + Temporal Cluster Expansion | Proximity Keyframe Density (4-6 frames) | 0.8091 | 0.0000 | 0.6444 | 0.6167 | +0.1062 | 62.5% | 93.8% | 457ms |
| **M6: Full Proposed SOTA** | CoDE MQ-DPF + AV-VLM QA + Viterbi DP | 0.8636 | 0.2857 | 0.7333 | **0.7250** | +0.2146 | 71.9% | 93.8% | 1143ms |
|   - w/o Dynamic Modality Gating | Replaced by static fusion weights | 0.8636 | 0.2857 | 0.7333 | 0.7250 | +0.2146 | 71.9% | 93.8% | 451ms |
|   - w/o Monotonic Viterbi DP | Greedy unconstrained event alignment | 0.8636 | 0.2857 | 0.7111 | 0.7229 | +0.2125 | 71.9% | 93.8% | 327ms |
|   - w/o Audio-Visual Cascade QA | Visual-only VLM reasoning (no ASR) | 0.8636 | 0.2857 | 0.7333 | 0.7250 | +0.2146 | 71.9% | 93.8% | 956ms |

---

## 2. Mã Nguồn Bảng LaTeX Chuẩn IEEE / ACM Conference

```latex
\begin{table*}[t]
\centering
\small
\caption{Ablation Study across 32 challenge video queries from the HCMUS AI Challenge Benchmark (22 KIS, 7 QA, 3 TRAKE).}
\label{tab:ablation_study}
\begin{tabular}{lcccccc}
\toprule
\textbf{Pipeline Configuration} & \textbf{KIS Score} & \textbf{QA Score} & \textbf{TRAKE Score} & \textbf{Macro Score} & \textbf{VR@1 (\%)} & \textbf{Latency (ms)} \\
\midrule
\textit{Incremental Additive Pipeline:} \\
M0: Raw SigLIP-2 Baseline & 0.7364 & 0.0000 & 0.0444 & 0.5104 & 53.1 & 121 \\
M1: +  Dual-Lingual Refinement & 0.8455 & 0.0000 & 0.2000 & 0.6000 & 62.5 & 118 \\
M2: +  TNCA Window & 0.8364 & 0.0000 & 0.2000 & 0.5938 & 62.5 & 131 \\
M3: +  Multimodal Fusion (Static) & 0.8364 & 0.0000 & 0.2000 & 0.5938 & 68.8 & 242 \\
M4: +  Dynamic Modality Gating & 0.8364 & 0.0000 & 0.2000 & 0.5938 & 68.8 & 200 \\
M5: +  Temporal Cluster Expansion & 0.8091 & 0.0000 & 0.6444 & 0.6167 & 62.5 & 457 \\
\textbf{Full Proposed SOTA (Ours)} & \textbf{0.8636} & \textbf{0.2857} & \textbf{0.7333} & \textbf{0.7250} & \textbf{71.9} & 1143 \\
\midrule
\textit{Leave-One-Out Ablation (Subtractive from SOTA):} \\
\quad w/o Dynamic Modality Gating & 0.8636 & 0.2857 & 0.7333 & 0.7250 & 71.9 & 451 \\
\quad w/o Monotonic Viterbi DP & 0.8636 & 0.2857 & 0.7111 & 0.7229 & 71.9 & 327 \\
\quad w/o Audio-Visual Cascade QA & 0.8636 & 0.2857 & 0.7333 & 0.7250 & 71.9 & 956 \\
\bottomrule
\end{tabular}
\end{table*}
```
