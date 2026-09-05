import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

# Force UTF-8 on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.indexing.faiss_indexer import load_faiss_index
from src.query.text_encoder import UnifiedTextEncoder

class TRAKEAlignmentAgent:
    """
    TRAKE Sequential Temporal Alignment Agent (SOTA Monotonic Sequence Dynamic Programming):
    - Phân rã chuỗi mô tả thành n sự kiện con: E_1 -> E_2 -> ... -> E_n.
    - Tính toán Ma trận tương đồng liên tục (Continuous Vectorized Cosine Similarity) trên toàn bộ keyframes của video.
    - Sử dụng thuật toán Quy hoạch động (Viterbi Monotonic Sequence DP) để tìm chuỗi khung hình tối ưu
      thỏa mãn điều kiện thời gian tăng dần nghiêm ngặt: t(E_1) < t(E_2) < ... < t(E_n).
    """
    def __init__(self, engine: str = "siglip2", batch: str = "batch_1", text_encoder=None):
        if text_encoder is not None:
            self.text_encoder = text_encoder
        else:
            self.text_encoder = UnifiedTextEncoder(engine=engine)
            
        self.faiss_index, self.df_frames = load_faiss_index(engine=engine, batch=batch)
        self.df_frames["row_idx"] = np.arange(len(self.df_frames))
        
        # Nạp Memory-mapped SigLIP features (đọc ngẫu nhiên tức thì <0.01ms)
        siglip_path = BASE_DIR / "data" / batch / "processed" / "siglip_features.npy"
        if siglip_path.exists():
            n_total = len(self.df_frames)
            self.siglip_features = np.memmap(siglip_path, dtype=np.float16, mode="r", shape=(n_total, 1152))
        else:
            self.siglip_features = None

        # Tạo sẵn mapping video -> keyframe data đã sắp xếp theo pts_time
        self.video_frames = {}
        for v_id, grp in self.df_frames.groupby("video_id"):
            s_grp = grp.sort_values("pts_time")
            self.video_frames[v_id] = {
                "row_indices": s_grp["row_idx"].to_numpy(),
                "frame_indices": s_grp["frame_idx"].to_numpy(),
                "pts_times": s_grp["pts_time"].to_numpy()
            }

    def _extract_temporal_nms_peaks(self, scores: np.ndarray, pts_times: np.ndarray, k: int = 3, nms_radius_sec: float = 2.0) -> list[float]:
        """Trích xuất k đỉnh độc lập về mặt thời gian qua Temporal NMS."""
        if len(scores) == 0:
            return [0.0] * k
        
        sc_copy = scores.copy()
        peaks = []
        for _ in range(k):
            max_idx = int(np.argmax(sc_copy))
            max_val = float(sc_copy[max_idx])
            if max_val <= -1e8:
                break
            peaks.append(max_val)
            # Triệt tiêu các frame trong bán kính nms_radius_sec
            t_peak = pts_times[max_idx]
            suppress_mask = np.abs(pts_times - t_peak) <= nms_radius_sec
            sc_copy[suppress_mask] = -1e9
            
        while len(peaks) < k:
            peaks.append(peaks[-1] if peaks else 0.0)
        return peaks[:k]

    def _solve_monotonic_dp(
        self,
        sim_matrix: np.ndarray,
        pts_times: np.ndarray,
        max_gap_sec: float = 90.0,
        use_adaptive_gap: bool = False
    ) -> list[int]:
        """
        Row-Normalized Monotonic Dynamic Programming Solver (CVPR 2021 D3TW / Moment-DETR NeurIPS 2021):
        - Sử dụng Local Support 3-point [0.2, 0.6, 0.2] để chống spike mồ côi.
        - Ràng buộc thứ tự nghiêm ngặt: t(E_1) < t(E_2) < ... < t(E_N).
        - Khi use_adaptive_gap=True: Co giãn max_gap_sec = max(300.0, duration * 0.4) cho video dài.
        """
        N, M = sim_matrix.shape
        if M < N:
            return list(range(min(N, M))) + [M - 1] * max(0, N - M)

        if use_adaptive_gap:
            duration = float(pts_times[-1] - pts_times[0]) if len(pts_times) > 1 else 300.0
            eff_max_gap = max(300.0, duration * 0.4)
        else:
            eff_max_gap = max_gap_sec

        # 1. Local Temporal Support 3-point
        S_smooth = np.zeros_like(sim_matrix)
        for i in range(N):
            row = sim_matrix[i]
            left = np.pad(row[:-1], (1, 0), mode='edge')
            right = np.pad(row[1:], (0, 1), mode='edge')
            S_smooth[i] = 0.2 * left + 0.6 * row + 0.2 * right

        # 2. Row Normalization (Min-Max per row)
        S_norm = np.zeros_like(S_smooth)
        for i in range(N):
            r_min = np.min(S_smooth[i])
            r_max = np.max(S_smooth[i])
            if r_max - r_min > 1e-6:
                S_norm[i] = (S_smooth[i] - r_min) / (r_max - r_min)
            else:
                S_norm[i] = S_smooth[i]

        dp = np.full((N, M), -1e9, dtype=np.float32)
        parent = np.full((N, M), -1, dtype=np.int32)

        # Khởi tạo cho Event 0
        dp[0, :] = S_norm[0, :]

        # Quy hoạch động cho Event 1 đến N-1
        for i in range(1, N):
            for j in range(i, M):
                t_j = pts_times[j]
                prev_indices = np.arange(j)
                t_k = pts_times[prev_indices]
                dt = t_j - t_k
                
                # Ràng buộc thời gian tăng dần nghiêm ngặt (dt > 0)
                valid_mask = (dt > 0) & (dp[i-1, :j] > -1e8)
                if not np.any(valid_mask):
                    continue

                valid_k = prev_indices[valid_mask]
                valid_scores = dp[i-1, valid_k]
                
                # Phạt nếu khoảng cách giữa 2 sự kiện vượt quá eff_max_gap
                excess_dt = np.maximum(0.0, dt[valid_mask] - eff_max_gap)
                penalties = 0.005 * excess_dt + 0.0005 * dt[valid_mask]
                
                total_candidates = valid_scores - penalties
                best_idx = int(np.argmax(total_candidates))
                best_k = int(valid_k[best_idx])
                
                dp[i, j] = S_norm[i, j] + total_candidates[best_idx]
                parent[i, j] = best_k

        # Backtracking
        best_end_j = int(np.argmax(dp[N-1, :]))
        if dp[N-1, best_end_j] <= -1e8:
            chosen_j = [int(np.argmax(S_norm[i])) for i in range(N)]
        else:
            chosen_j = [0] * N
            curr = best_end_j
            for i in range(N - 1, -1, -1):
                chosen_j[i] = curr
                curr = parent[i, curr]
                if curr == -1 and i > 0:
                    curr = max(0, chosen_j[i] - 1)

        self.last_dp_debug = {
            "S_smooth": S_smooth,
            "S_norm": S_norm,
            "dp": dp,
            "parent": parent,
            "pts_times": pts_times,
            "chosen_j": chosen_j
        }
        
        return chosen_j

    def _solve_segmental_dp(
        self,
        sim_matrix: np.ndarray,
        pts_times: np.ndarray,
        max_gap_sec: float = 90.0,
        max_seg_len_frames: int = 5,
        lambda_d: float = 0.02,
        lambda_g: float = 0.005
    ) -> list[int]:
        """
        Segmental Dynamic Programming Solver cho TRAKE (NeurIPS 2021 Moment-DETR & D3TW):
        - Gán sự kiện vào một phân đoạn [s, e] và chọn Exact Peak Frame trong phân đoạn.
        - Ràng buộc chặt chẽ max_gap_sec = 90s.
        """
        N, M = sim_matrix.shape
        if M < N:
            return list(range(min(N, M))) + [M - 1] * max(0, N - M)

        S_smooth = np.zeros_like(sim_matrix)
        for i in range(N):
            row = sim_matrix[i]
            left = np.pad(row[:-1], (1, 0), mode='edge')
            right = np.pad(row[1:], (0, 1), mode='edge')
            S_smooth[i] = 0.2 * left + 0.6 * row + 0.2 * right

        S_norm = np.zeros_like(S_smooth)
        for i in range(N):
            r_min = np.min(S_smooth[i])
            r_max = np.max(S_smooth[i])
            if r_max - r_min > 1e-6:
                S_norm[i] = (S_smooth[i] - r_min) / (r_max - r_min)
            else:
                S_norm[i] = S_smooth[i]

        dp = np.full((N, M), -1e9, dtype=np.float32)
        parent_e = np.full((N, M), -1, dtype=np.int32)
        seg_start = np.full((N, M), -1, dtype=np.int32)

        # Event 0
        for e in range(M):
            max_s = max(0, e - max_seg_len_frames + 1)
            for s in range(max_s, e + 1):
                dur = e - s
                score = np.mean(S_norm[0, s:e+1]) - lambda_d * dur
                if score > dp[0, e]:
                    dp[0, e] = score
                    seg_start[0, e] = s

        # Event 1 to N-1
        for i in range(1, N):
            best_prev = np.full(M, -1e9, dtype=np.float32)
            best_prev_idx = np.full(M, -1, dtype=np.int32)
            
            for s in range(1, M):
                prev_indices = np.arange(s)
                valid_mask = dp[i-1, :s] > -1e8
                if not np.any(valid_mask):
                    continue
                valid_prev_e = prev_indices[valid_mask]
                valid_scores = dp[i-1, valid_prev_e]
                
                dt = pts_times[s] - pts_times[valid_prev_e]
                excess_dt = np.maximum(0.0, dt - max_gap_sec)
                penalties = lambda_g * excess_dt + 0.0005 * dt
                
                total_candidates = valid_scores - penalties
                best_idx = int(np.argmax(total_candidates))
                best_prev[s] = total_candidates[best_idx]
                best_prev_idx[s] = valid_prev_e[best_idx]
            
            for e in range(1, M):
                max_s = max(1, e - max_seg_len_frames + 1)
                for s in range(max_s, e + 1):
                    if best_prev[s] <= -1e8:
                        continue
                    dur = e - s
                    seg_score_val = np.mean(S_norm[i, s:e+1]) - lambda_d * dur
                    total_score = best_prev[s] + seg_score_val
                    if total_score > dp[i, e]:
                        dp[i, e] = total_score
                        parent_e[i, e] = best_prev_idx[s]
                        seg_start[i, e] = s

        # Backtracking
        best_end_e = int(np.argmax(dp[N-1, :]))
        if dp[N-1, best_end_e] <= -1e8:
            chosen_j = [int(np.argmax(S_norm[i])) for i in range(N)]
        else:
            chosen_j = [0] * N
            curr_e = best_end_e
            for i in range(N - 1, -1, -1):
                s = seg_start[i, curr_e]
                if s == -1: s = curr_e
                # SOTA Pinpoint: Chọn đỉnh nhọn có S_norm cao nhất trong phân đoạn [s, curr_e]
                peak_offset = int(np.argmax(S_norm[i, s:curr_e+1]))
                chosen_j[i] = s + peak_offset
                curr_e = parent_e[i, curr_e]
                if curr_e == -1 and i > 0:
                    curr_e = max(0, s - 1)

        self.last_dp_debug = {
            "S_norm": S_norm,
            "dp": dp,
            "chosen_j": chosen_j
        }
        
        return chosen_j

    def align_events(
        self,
        raw_query: str,
        events: list[str],
        top_k: int = 100,
        use_multi_query: bool = True,
        use_event_coverage: bool = True,
        use_row_norm_dp: bool = True,
        use_segmental_dp: bool = False,
        use_adaptive_gap: bool = False,
        use_viterbi_dp: bool = True
    ) -> list[dict]:
        """
        Tìm kiếm và căn chỉnh chuỗi sự kiện bằng Multi-Query Retrieval & Calibrated Event Coverage.
        """
        if not events:
            return []

        import re
        clean_prefix_regex = r"^(?:(?:[eE]|sự kiện|event|bước|cảnh|scene|giai đoạn)\s*\d+[\s:.-]*|\d+[\.\)]\s*|(?:đầu tiên|tiếp theo|sau đó|kế đến|kế tiếp|cuối cùng|lần lượt|rồi)\s*[:,\s.-]*)+"
        cleaned_events = []
        for ev in events:
            if isinstance(ev, str):
                c = re.sub(clean_prefix_regex, "", ev.strip(), flags=re.IGNORECASE).strip()
                cleaned_events.append(c if len(c) > 3 else ev.strip())
            else:
                cleaned_events.append(str(ev))
        events = cleaned_events

        n_events = len(events)
        
        # 1. Mã hóa sự kiện & Global Query
        event_vecs = np.array([self.text_encoder.encode_text(ev)[0] for ev in events], dtype=np.float32)
        q_global = self.text_encoder.encode_text(raw_query)[0].astype(np.float32)

        # 2. Stage-1 Candidate Retrieval (Multi-Query vs Single-Query)
        fetch_k = max(500, top_k * 5)
        if use_multi_query:
            all_queries = np.vstack([event_vecs, q_global.reshape(1, -1)])
            scores, indices = self.faiss_index.search(all_queries, fetch_k)
            candidate_videos = set()
            for row_indices in indices:
                for idx in row_indices:
                    v_id = self.df_frames.iloc[idx]["video_id"]
                    if v_id in self.video_frames:
                        candidate_videos.add(v_id)
        else:
            scores, indices = self.faiss_index.search(q_global.reshape(1, -1), fetch_k * 2)
            candidate_videos = set()
            for idx in indices[0]:
                v_id = self.df_frames.iloc[idx]["video_id"]
                if v_id in self.video_frames:
                    candidate_videos.add(v_id)

        # Đảm bảo tập ứng viên có đủ tối thiểu top_k video để xuất đủ số dòng theo quy chế BTC
        if len(candidate_videos) < top_k:
            scores_extra, indices_extra = self.faiss_index.search(q_global.reshape(1, -1), min(len(self.df_frames), 30000))
            for idx in indices_extra[0]:
                v_id = self.df_frames.iloc[idx]["video_id"]
                if v_id in self.video_frames:
                    candidate_videos.add(v_id)
                if len(candidate_videos) >= top_k:
                    break
            if len(candidate_videos) < top_k:
                for v_id in self.video_frames.keys():
                    candidate_videos.add(v_id)
                    if len(candidate_videos) >= top_k:
                        break

        if self.siglip_features is None or not candidate_videos:
            return []

        # 3. Stage-2: Temporal NMS Peak & Event Coverage Reranking
        candidate_data = []
        for v_id in candidate_videos:
            v_info = self.video_frames[v_id]
            row_indices = v_info["row_indices"]
            pts_times = v_info["pts_times"]
            v_feats = np.asarray(self.siglip_features[row_indices], dtype=np.float32)
            
            # Ma trận tương đồng (N_events, M_keyframes)
            sim_matrix = np.dot(event_vecs, v_feats.T)
            
            # Tính Top-3 temporal NMS peak cho mỗi event
            event_peaks = []
            for j in range(n_events):
                peaks_j = self._extract_temporal_nms_peaks(sim_matrix[j], pts_times, k=3, nms_radius_sec=2.0)
                event_peaks.append(np.mean(peaks_j))
            
            event_peaks = np.array(event_peaks, dtype=np.float32)
            
            # Điểm toàn cục
            sim_global = np.dot(v_feats, q_global)
            s_global_top = float(np.mean(np.sort(sim_global)[-3:]))
            
            candidate_data.append({
                "video_id": v_id,
                "event_peaks": event_peaks,
                "s_global": s_global_top,
                "sim_matrix": sim_matrix,
                "pts_times": pts_times,
                "frame_indices": v_info["frame_indices"]
            })

        if use_event_coverage and candidate_data:
            # Chuẩn hóa Event-wise Calibration giữa các candidate videos
            all_peaks = np.array([cd["event_peaks"] for cd in candidate_data])  # (K_cands, N_events)
            min_p = np.min(all_peaks, axis=0, keepdims=True)
            max_p = np.max(all_peaks, axis=0, keepdims=True)
            denom = np.where(max_p - min_p > 1e-6, max_p - min_p, 1.0)
            norm_peaks = (all_peaks - min_p) / denom

            # Tính Video Score tổng hợp
            for idx, cd in enumerate(candidate_data):
                p_norm = norm_peaks[idx]
                mean_q = float(np.mean(p_norm))
                soft_min_q = float(-0.2 * np.log(np.sum(np.exp(-5.0 * p_norm)) + 1e-6))
                s_g = cd["s_global"]
                cd["v_score"] = 0.40 * mean_q + 0.30 * soft_min_q + 0.30 * s_g
        else:
            for cd in candidate_data:
                cd["v_score"] = cd["s_global"]

        # Sắp xếp và chọn Top candidate videos tốt nhất đưa vào DP (mặc định top_k = 100)
        candidate_data.sort(key=lambda x: x["v_score"], reverse=True)
        top_cands = candidate_data[:top_k]

        # 4. Stage-3: Row-Normalized Monotonic DP Alignment
        final_predictions = []
        for v_idx, cd in enumerate(top_cands):
            v_id = cd["video_id"]
            sim_matrix = cd["sim_matrix"]
            pts_times = cd["pts_times"]
            f_indices = cd["frame_indices"]
            
            if not use_viterbi_dp:
                chosen_kf_indices = [int(np.argmax(sim_matrix[i])) for i in range(n_events)]
            elif use_segmental_dp:
                chosen_kf_indices = self._solve_segmental_dp(sim_matrix, pts_times)
            else:
                chosen_kf_indices = self._solve_monotonic_dp(sim_matrix, pts_times, use_adaptive_gap=use_adaptive_gap)
            chosen_frames = [int(f_indices[j]) for j in chosen_kf_indices]
            
            # Điểm tương đồng thực tế của chuỗi được chọn
            raw_dp_scores = [sim_matrix[i, chosen_kf_indices[i]] for i in range(n_events)]
            dp_score = float(np.mean(raw_dp_scores))
            
            # Kết hợp điểm Rerank và điểm DP
            v_rank_score = (1.0 / (v_idx + 1)) * 3.0 + dp_score + cd["v_score"]
            
            final_predictions.append({
                "video_id": v_id,
                "frame_idx": chosen_frames[0] if chosen_frames else 0,
                "event_frames": chosen_frames,
                "score": v_rank_score,
                "sim_matrix": sim_matrix,
                "pts_times": pts_times
            })

        final_predictions.sort(key=lambda x: x["score"], reverse=True)
        for r, p in enumerate(final_predictions, 1):
            p["rank"] = r

        return final_predictions[:top_k]

if __name__ == "__main__":
    agent = TRAKEAlignmentAgent("siglip2")
    events_sample = [
        "Chef pours diced onions into a pan",
        "Adds minced beef and sautés",
        "Adds green peas into the pan",
        "Adds diced carrots into the pan",
        "Pours boiled pasta into the pan"
    ]
    res = agent.align_events("Nấu mì Ý", events_sample, top_k=5)
    print("🏆 Kết quả Monotonic Dynamic Programming Vectorized:")
    for r in res:
        print(f"Rank #{r['rank']} | Video: {r['video_id']} | Score: {r['score']:.4f} | Event Frames: {r['event_frames']}")
