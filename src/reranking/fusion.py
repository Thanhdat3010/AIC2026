from typing import List, Dict, Any
from src.config import settings
from src.reranking.metadata_reranker import MetadataReranker

class FusionEngine:
    def __init__(self, metadata_reranker: MetadataReranker = None):
        self.metadata_reranker = metadata_reranker
        self.weights = settings.reranking.weights
        
    def rerank(self, query: str, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for frame in frames:
            # Base CLIP score
            clip_score = frame.get("score", 0.0)
            
            # Cue coverage multiplier
            coverage = frame.get("cue_coverage", 1.0)
            coverage_bonus = coverage * self.weights.cue_coverage_multiplier
            
            # Metadata score
            meta_score = 0.0
            if self.metadata_reranker:
                meta_score = self.metadata_reranker.score(query, frame["video_id"])
                
            # Compute final weighted score
            # Object & Temporal reranking can be added here later (M6 expansion)
            final_score = (
                (clip_score * self.weights.clip_score) +
                (meta_score * self.weights.metadata_score) +
                coverage_bonus
            )
            
            frame["final_score"] = final_score
            frame["meta_score"] = meta_score
            
        # Sort by final score
        frames.sort(key=lambda x: x["final_score"], reverse=True)
        return frames
