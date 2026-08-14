from typing import List, Dict, Any
from collections import defaultdict

class VideoAggregator:
    """
    Aggregates frame-level results into video-level rankings.
    Ensures that each video returns its top N most representative keyframes.
    """
    def __init__(self, max_frames_per_video: int = 2):
        self.max_frames = max_frames_per_video
        
    def aggregate(self, frame_results: List[Dict[str, Any]], top_k_videos: int = 50) -> List[Dict[str, Any]]:
        # Group frames by video_id
        video_groups = defaultdict(list)
        for frame in frame_results:
            video_groups[frame["video_id"]].append(frame)
            
        # Calculate video-level scores
        video_scores = []
        for vid, frames in video_groups.items():
            # Sort frames in this video by score descending
            frames.sort(key=lambda x: x["score"], reverse=True)
            
            # Keep top N frames per video
            top_frames = frames[:self.max_frames]
            
            # Video score can be max score, or sum of top frames
            max_score = top_frames[0]["score"]
            cue_coverage = max(f.get("cue_coverage", 1.0) for f in top_frames)
            
            video_scores.append({
                "video_id": vid,
                "video_score": max_score,
                "cue_coverage": cue_coverage,
                "top_frames": top_frames
            })
            
        # Sort videos by cue coverage then video score
        video_scores.sort(key=lambda x: (x["cue_coverage"], x["video_score"]), reverse=True)
        
        # Keep top K videos
        video_scores = video_scores[:top_k_videos]
        
        # Flatten back to frame list but maintain rank order
        final_frames = []
        for v_group in video_scores:
            final_frames.extend(v_group["top_frames"])
            
        return final_frames
