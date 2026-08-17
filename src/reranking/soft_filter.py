import numpy as np

class SoftVideoFilter:
    """
    Bộ lọc mềm tăng cường điểm dựa trên:
    1. Temporal Position Hint (early, middle, late): Tăng nhẹ điểm theo tỉ lệ vị trí khung hình trong video.
    2. Video Metadata Matching: Tăng điểm nếu video_id khớp với kết quả BM25 Metadata.
    """
    def __init__(self):
        pass

    def apply_temporal_hint(self, candidates: list[dict], temporal_hint: str, boost_factor: float = 1.15) -> list[dict]:
        """
        Tăng điểm cho các khung hình nằm ở đầu/giữa/cuối video tùy theo gợi ý từ Gemini.
        """
        if not temporal_hint or temporal_hint.lower() in ["any", "none", "unknown"]:
            return candidates

        hint = temporal_hint.lower()
        filtered = []
        for cand in candidates:
            c = cand.copy()
            f_idx = c.get("frame_idx", 0)
            
            # Ước lượng vị trí video tương đối dựa trên frame index phổ biến (1 video ~ 20.000 - 40.000 frames)
            # Early: < 10.000 frames; Late: > 20.000 frames
            if hint == "early" and f_idx < 10000:
                c["score"] = c["score"] * boost_factor
            elif hint == "late" and f_idx > 20000:
                c["score"] = c["score"] * boost_factor
            elif hint == "middle" and 8000 <= f_idx <= 25000:
                c["score"] = c["score"] * boost_factor

            filtered.append(c)

        filtered.sort(key=lambda x: x["score"], reverse=True)
        for rank, c in enumerate(filtered, 1):
            c["rank"] = rank
        return filtered

    def apply_metadata_boost(self, candidates: list[dict], meta_hits: list[dict], boost_factor: float = 1.25) -> list[dict]:
        """
        Tăng điểm cho các candidate thuộc video có tiêu đề / mô tả YouTube khớp với từ khóa tìm kiếm.
        """
        if not meta_hits:
            return candidates

        matched_videos = {m["video_id"]: m.get("score", 1.0) for m in meta_hits}
        filtered = []
        for cand in candidates:
            c = cand.copy()
            v_id = c.get("video_id", "")
            if v_id in matched_videos:
                c["score"] = c["score"] * boost_factor
                if "matched_modalities" in c:
                    c["matched_modalities"]["metadata"] = matched_videos[v_id]
            filtered.append(c)

        filtered.sort(key=lambda x: x["score"], reverse=True)
        for rank, c in enumerate(filtered, 1):
            c["rank"] = rank
        return filtered
