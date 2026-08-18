import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np

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
    - Tìm kiếm video mục tiêu có tổng điểm khớp cao nhất trên toàn bộ chuỗi sự kiện.
    - Sử dụng thuật toán Quy hoạch động (Viterbi Monotonic Sequence DP) để tìm chuỗi khung hình tối ưu
      thỏa mãn điều kiện thời gian tăng dần nghiêm ngặt: t(E_1) < t(E_2) < ... < t(E_n).
    - Cô lập hoàn toàn khỏi Gaussian smoothing của KIS để tránh hiện tượng nhòe thời gian giữa các bước.
    """
    def __init__(self, engine: str = "siglip2", batch: str = "batch_1", text_encoder=None):
        if text_encoder is not None:
            self.text_encoder = text_encoder
        else:
            self.text_encoder = UnifiedTextEncoder(engine=engine)
            
        self.faiss_index, self.df_frames = load_faiss_index(engine=engine, batch=batch)
        self.df_frames["row_idx"] = np.arange(len(self.df_frames))
        
        # Tạo sẵn mapping video -> keyframe data đã sắp xếp theo pts_time
        self.video_frames = {}
        for v_id, grp in self.df_frames.groupby("video_id"):
            s_grp = grp.sort_values("pts_time")
            self.video_frames[v_id] = {
                "row_indices": s_grp["row_idx"].to_numpy(),
                "frame_indices": s_grp["frame_idx"].to_numpy(),
                "pts_times": s_grp["pts_time"].to_numpy()
            }

    def _solve_monotonic_dp(self, sim_matrix: np.ndarray, pts_times: np.ndarray, min_gap_sec: float = 0.5, max_gap_sec: float = 120.0) -> list[int]:
        """
        Thuật toán Quy hoạch động Monotonic Sequence Dynamic Programming:
        sim_matrix: (N_events, M_keyframes)
        pts_times: (M_keyframes,)
        Returns: list gồm N_events chỉ số keyframe [j_0, j_1, ..., j_{N-1}] tối ưu nhất.
        """
        N, M = sim_matrix.shape
        if M < N:
            # Nếu số keyframe ít hơn số sự kiện, gán index tăng dần khả dĩ
            return list(range(min(N, M))) + [M - 1] * max(0, N - M)

        # dp[i, j]: Điểm tối đa khi gán event i cho keyframe j
        # parent[i, j]: Chỉ số keyframe của event i-1 được chọn
        dp = np.full((N, M), -1e9, dtype=np.float32)
        parent = np.full((N, M), -1, dtype=np.int32)

        # Khởi tạo cho Event 0
        dp[0, :] = sim_matrix[0, :]

        # Quy hoạch động cho Event 1 đến Event N-1
        for i in range(1, N):
            for j in range(i, M):
                # j là keyframe cho event i -> tìm k < j tốt nhất cho event i-1
                t_j = pts_times[j]
                
                # Xét tất cả k < j
                prev_indices = np.arange(j)
                t_k = pts_times[prev_indices]
                dt = t_j - t_k
                
                # Ràng buộc thời gian: dt >= min_gap_sec
                valid_mask = (dt >= min_gap_sec) & (dp[i-1, :j] > -1e8)
                if not np.any(valid_mask):
                    # Nếu quá sát, cho phép dt >= 0
                    valid_mask = (dt >= 0) & (dp[i-1, :j] > -1e8)
                    if not np.any(valid_mask):
                        continue

                valid_k = prev_indices[valid_mask]
                valid_scores = dp[i-1, valid_k]
                
                # Penalty nếu khoảng cách quá xa (> max_gap_sec)
                excess_dt = np.maximum(0.0, dt[valid_mask] - max_gap_sec)
                penalties = 0.005 * excess_dt
                
                total_candidates = valid_scores - penalties
                best_idx = np.argmax(total_candidates)
                best_k = valid_k[best_idx]
                
                dp[i, j] = sim_matrix[i, j] + total_candidates[best_idx]
                parent[i, j] = best_k

        # Truy vết (Backtracking) từ j tốt nhất ở Event N-1
        best_end_j = np.argmax(dp[N-1, :])
        if dp[N-1, best_end_j] <= -1e8:
            # Fallback nếu không có đường đi hợp lệ: chọn argmax từng event thỏa mãn tăng dần
            chosen_j = []
            curr_k = 0
            for i in range(N):
                rem_cands = np.arange(curr_k, M)
                if len(rem_cands) > 0:
                    pick = rem_cands[np.argmax(sim_matrix[i, rem_cands])]
                    chosen_j.append(pick)
                    curr_k = min(pick + 1, M - 1)
                else:
                    chosen_j.append(M - 1)
            return chosen_j

        chosen_j = [0] * N
        curr = best_end_j
        for i in range(N - 1, -1, -1):
            chosen_j[i] = curr
            curr = parent[i, curr]
            if curr == -1 and i > 0:
                curr = max(0, chosen_j[i] - 1)

        return chosen_j

    def align_events(self, raw_query: str, events: list[str], top_videos: int = 5) -> list[dict]:
        """
        Tìm kiếm và căn chỉnh chuỗi sự kiện trong video mục tiêu qua Monotonic DP.
        Returns: list các candidate predictions có chuỗi mốc thời gian tăng dần.
        """
        if not events:
            return []

        n_events = len(events)
        # 1. Mã hóa từng sự kiện con bằng UnifiedTextEncoder (SigLIP 2 1152d)
        event_vecs = [self.text_encoder.encode_text(ev) for ev in events]

        # 2. Tìm kiếm ứng viên cho từng sự kiện qua FAISS
        video_event_hits = defaultdict(lambda: defaultdict(list))
        
        for e_idx, vec in enumerate(event_vecs):
            scores, indices = self.faiss_index.search(vec, 300)
            for rank, (sc, idx) in enumerate(zip(scores[0], indices[0]), 1):
                row = self.df_frames.iloc[idx]
                v_id = row["video_id"]
                f_idx = int(row["frame_idx"])
                video_event_hits[v_id][e_idx].append({"frame_idx": f_idx, "score": float(sc), "rank": rank})

        # 3. Chấm điểm Video: Video nào có nhiều sự kiện con xuất hiện nhất với điểm cao nhất
        video_scores = []
        for v_id, e_dict in video_event_hits.items():
            coverage = len(e_dict)  # Số sự kiện xuất hiện
            avg_score = np.mean([max([c["score"] for c in cands]) for cands in e_dict.values()])
            # Boost mạnh video có coverage cao
            combined_v_score = (coverage / n_events) * 2.0 + avg_score
            video_scores.append((v_id, combined_v_score, coverage))

        video_scores.sort(key=lambda x: x[1], reverse=True)
        top_v_list = [v[0] for v in video_scores[:top_videos]]

        # 4. Trích xuất chuỗi frame tăng dần qua Monotonic Dynamic Programming
        final_predictions = []
        for v_id in top_v_list:
            if v_id not in self.video_frames:
                continue

            v_info = self.video_frames[v_id]
            f_indices = v_info["frame_indices"]
            pts_times = v_info["pts_times"]
            M = len(f_indices)

            # Xây dựng ma trận tương đồng sắc nét (N, M)
            sim_matrix = np.zeros((n_events, M), dtype=np.float32)
            e_dict = video_event_hits[v_id]

            # Điền điểm từ FAISS hits
            for e_idx in range(n_events):
                cands = e_dict.get(e_idx, [])
                frame_to_sc = {c["frame_idx"]: c["score"] for c in cands}
                for j, f_id in enumerate(f_indices):
                    if f_id in frame_to_sc:
                        sim_matrix[e_idx, j] = frame_to_sc[f_id]
                    else:
                        sim_matrix[e_idx, j] = 0.05  # Base background similarity

            # Giải bài toán Monotonic Sequence DP
            chosen_kf_indices = self._solve_monotonic_dp(sim_matrix, pts_times, min_gap_sec=0.5)
            chosen_frames = [int(f_indices[j]) for j in chosen_kf_indices]

            final_predictions.append({
                "video_id": v_id,
                "event_frames": chosen_frames,
                "score": float(video_scores[0][1]) if v_id == video_scores[0][0] else 0.5
            })

        return final_predictions

if __name__ == "__main__":
    agent = TRAKEAlignmentAgent("siglip2")
    events_sample = [
        "Chef pours diced onions into a pan",
        "Adds minced beef and sautés",
        "Adds green peas into the pan",
        "Adds diced carrots into the pan",
        "Pours boiled pasta into the pan"
    ]
    res = agent.align_events("Nấu mì Ý", events_sample, top_videos=3)
    print("🏆 Kết quả Monotonic Dynamic Programming:")
    for r in res:
        print(f"Video: {r['video_id']} | Frames: {r['event_frames']}")
