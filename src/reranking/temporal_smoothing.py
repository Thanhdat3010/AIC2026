import numpy as np
from collections import defaultdict

class TemporalSceneSmoother:
    """
    Bộ làm mịn và gom cụm không gian - thời gian (Temporal Smoothing & Scene NMS):
    1. Gaussian Temporal Smoothing: Tích lũy điểm cho các khung hình lân cận trong cùng phân cảnh (+-3s).
    2. Scene Non-Maximum Suppression (NMS): Loại bỏ các frame trùng lặp liên tiếp trong cùng 1 phân cảnh 
       để đa dạng hóa Top 100 nộp bài, giúp tăng mạnh xác suất phủ trúng đáp án ở các mốc R@5, R@20, R@50, R@100.
    """
    def __init__(self, fps: float = 25.0, window_sec: float = 3.0, sigma_sec: float = 1.5, nms_window_sec: float = 2.0):
        self.fps = fps
        self.window_frames = int(fps * window_sec)
        self.sigma_frames = fps * sigma_sec
        self.nms_window_frames = int(fps * nms_window_sec)

    def smooth_and_rerank(self, candidates: list[dict], top_k: int = 100, alpha: float = 0.4) -> list[dict]:
        """
        Làm mịn điểm theo dòng thời gian video và áp dụng NMS.
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

        # 3. Sắp xếp sơ bộ theo điểm đã làm mịn
        smoothed_candidates.sort(key=lambda x: x["score"], reverse=True)

        # 4. Áp dụng Non-Maximum Suppression (Scene NMS) để chọn đại diện ưu tú nhất từng phân cảnh
        selected = []
        suppressed_keys = defaultdict(list)

        for cand in smoothed_candidates:
            v_id = cand["video_id"]
            f_idx = cand["frame_idx"]
            
            # Kiểm tra xem frame này có quá gần một frame đã được chọn trước đó trong cùng video không
            is_suppressed = False
            for prev_f in suppressed_keys[v_id]:
                if abs(f_idx - prev_f) <= self.nms_window_frames:
                    is_suppressed = True
                    break

            if not is_suppressed:
                selected.append(cand)
                suppressed_keys[v_id].append(f_idx)
                if len(selected) >= top_k:
                    break

        # Nếu danh sách sau NMS chưa đủ top_k, chèn thêm các frame còn lại
        if len(selected) < top_k:
            selected_set = {(c["video_id"], c["frame_idx"]) for c in selected}
            for cand in smoothed_candidates:
                key = (cand["video_id"], cand["frame_idx"])
                if key not in selected_set:
                    selected.append(cand)
                    selected_set.add(key)
                    if len(selected) >= top_k:
                        break

        # Cập nhật lại thứ hạng rank từ 1 đến len(selected)
        for rank, cand in enumerate(selected, 1):
            cand["rank"] = rank

        return selected[:top_k]

if __name__ == "__main__":
    smoother = TemporalSceneSmoother()
    sample_cands = [
        {"rank": 1, "video_id": "L28_V009", "frame_idx": 16228, "score": 0.0131},
        {"rank": 2, "video_id": "L28_V009", "frame_idx": 16230, "score": 0.0129},
        {"rank": 3, "video_id": "L28_V009", "frame_idx": 16662, "score": 0.0127},
        {"rank": 4, "video_id": "L28_V009", "frame_idx": 15866, "score": 0.0125},
    ]
    res = smoother.smooth_and_rerank(sample_cands, top_k=3)
    print("Kết quả sau Temporal Smoothing & NMS:")
    for r in res:
        print(f"Rank #{r['rank']} | Video: {r['video_id']} | Frame: {r['frame_idx']} | Score: {r['score']:.5f}")
