import numpy as np
from collections import defaultdict

class TemporalSceneSmoother:
    """
    Bộ làm mịn theo dòng thời gian video (Temporal Gaussian Context Smoothing):
    - Tích lũy điểm cho các khung hình thuộc cùng phân cảnh lân cận (+-3 giây).
    - Bảo toàn toàn bộ các khung hình trong phân cảnh (không dùng hard-suppression làm mất frame đáp án).
    """
    def __init__(self, fps: float = 25.0, window_sec: float = 3.0, sigma_sec: float = 1.5):
        self.fps = fps
        self.window_frames = int(fps * window_sec)
        self.sigma_frames = fps * sigma_sec

    def smooth_and_rerank(self, candidates: list[dict], top_k: int = 100, alpha: float = 0.3) -> list[dict]:
        """
        Cộng hưởng điểm số theo phân cảnh lân cận và xếp hạng lại Top K.
        """
        if not candidates:
            return []

        # 1. Gom nhóm candidates theo từng video_id
        video_groups = defaultdict(list)
        for cand in candidates:
            video_groups[cand["video_id"]].append(cand)

        smoothed_candidates = []

        # 2. Áp dụng Gaussian Smoothing theo từng video
        for video_id, group in video_groups.items():
            frames = np.array([c["frame_idx"] for c in group])
            scores = np.array([c["score"] for c in group])

            new_scores = []
            for i, f in enumerate(frames):
                diffs = np.abs(frames - f)
                in_window = diffs <= self.window_frames
                weights = np.exp(- (diffs[in_window] ** 2) / (2 * (self.sigma_frames ** 2)))
                smoothed_score = scores[i] + alpha * np.sum(scores[in_window] * weights)
                new_scores.append(smoothed_score)

            for i, cand in enumerate(group):
                c_copy = cand.copy()
                c_copy["raw_score"] = cand["score"]
                c_copy["score"] = float(new_scores[i])
                smoothed_candidates.append(c_copy)

        # 3. Sắp xếp lại danh sách theo điểm số đã làm mịn
        smoothed_candidates.sort(key=lambda x: x["score"], reverse=True)

        # 4. Cập nhật lại thứ hạng rank
        final_top = smoothed_candidates[:top_k]
        for rank, cand in enumerate(final_top, 1):
            cand["rank"] = rank

        return final_top

if __name__ == "__main__":
    smoother = TemporalSceneSmoother()
    sample_cands = [
        {"rank": 1, "video_id": "L28_V009", "frame_idx": 16228, "score": 0.0131},
        {"rank": 2, "video_id": "L28_V009", "frame_idx": 16230, "score": 0.0129},
        {"rank": 3, "video_id": "L28_V009", "frame_idx": 16662, "score": 0.0127},
    ]
    res = smoother.smooth_and_rerank(sample_cands, top_k=3)
    print("Kết quả sau Temporal Smoothing:")
    for r in res:
        print(f"Rank #{r['rank']} | Video: {r['video_id']} | Frame: {r['frame_idx']} | Score: {r['score']:.5f}")
