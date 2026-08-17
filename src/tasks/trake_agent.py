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
    TRAKE Sequential Temporal Alignment Agent (Chuyên trách bài toán Chuỗi Hành Động Theo Thời Gian):
    - Phân rã chuỗi mô tả thành n sự kiện con: E_1 -> E_2 -> ... -> E_n.
    - Tìm kiếm video mục tiêu có tổng điểm khớp cao nhất trên toàn bộ chuỗi sự kiện.
    - Trong video mục tiêu, tìm các khung hình đại diện cho từng E_j và căn chỉnh theo thứ tự tăng dần:
      t(E_1) <= t(E_2) <= ... <= t(E_n).
    """
    def __init__(self, engine: str = "siglip2", batch: str = "batch_1"):
        self.text_encoder = UnifiedTextEncoder(engine=engine)
        self.faiss_index, self.df_frames = load_faiss_index(engine=engine, batch=batch)

    def align_events(self, raw_query: str, events: list[str], top_videos: int = 5) -> list[dict]:
        """
        Tìm kiếm và căn chỉnh chuỗi sự kiện trong video mục tiêu.
        Returns: list các candidate predictions có chuỗi mốc thời gian tăng dần.
        """
        if not events:
            return []

        n_events = len(events)
        # 1. Mã hóa từng sự kiện con
        event_vecs = [self.text_encoder.encode_text(ev) for ev in events]

        # 2. Tìm kiếm ứng viên cho từng sự kiện
        video_event_hits = defaultdict(lambda: defaultdict(list))
        
        for e_idx, vec in enumerate(event_vecs):
            scores, indices = self.faiss_index.search(vec, 200)
            for rank, (sc, idx) in enumerate(zip(scores[0], indices[0]), 1):
                row = self.df_frames.iloc[idx]
                v_id = row["video_id"]
                f_idx = int(row["frame_idx"])
                video_event_hits[v_id][e_idx].append({"frame_idx": f_idx, "score": float(sc), "rank": rank})

        # 3. Chấm điểm video: Video nào có nhiều sự kiện con xuất hiện nhất với điểm cao nhất
        video_scores = []
        for v_id, e_dict in video_event_hits.items():
            coverage = len(e_dict) # Số sự kiện có mặt trong video này
            avg_score = np.mean([max([c["score"] for c in cands]) for cands in e_dict.values()])
            combined_v_score = (coverage / n_events) * 1.5 + avg_score
            video_scores.append((v_id, combined_v_score, coverage))

        video_scores.sort(key=lambda x: x[1], reverse=True)
        top_v_list = [v[0] for v in video_scores[:top_videos]]

        # 4. Trích xuất chuỗi frame tăng dần cho từng video trong Top V
        final_predictions = []
        for v_id in top_v_list:
            e_dict = video_event_hits[v_id]
            df_v = self.df_frames[self.df_frames["video_id"] == v_id].sort_values("frame_idx")
            all_v_frames = df_v["frame_idx"].tolist()

            # Dynamic Programming / Greedy Monotonic Alignment
            chosen_frames = []
            last_frame = -1
            for e_idx in range(n_events):
                cands = e_dict.get(e_idx, [])
                valid_cands = [c for c in cands if c["frame_idx"] >= last_frame]
                if valid_cands:
                    best_cand = max(valid_cands, key=lambda x: x["score"])
                    chosen_frames.append(best_cand["frame_idx"])
                    last_frame = best_cand["frame_idx"]
                else:
                    # Nếu không có frame sau last_frame, lấy frame kế tiếp trong video
                    subsequent = [f for f in all_v_frames if f > last_frame]
                    if subsequent:
                        chosen_frames.append(subsequent[0])
                        last_frame = subsequent[0]
                    elif all_v_frames:
                        chosen_frames.append(all_v_frames[-1])
                    else:
                        chosen_frames.append(0)

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
    print("🏆 Kết quả Alignment Agent:")
    for r in res:
        print(f"Video: {r['video_id']} | Frames: {r['event_frames']}")
