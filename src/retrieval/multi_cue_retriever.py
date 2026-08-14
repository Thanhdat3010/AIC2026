from typing import List, Dict, Any
import numpy as np

from src.query.query_decomposer import QueryDecomposer
from src.query.text_encoder import TextEncoderInterface
from src.retrieval.faiss_retriever import FAISSRetriever

class MultiCueRetriever:
    """
    Orchestrates the decomposition of a query, encodes each cue,
    retrieves frames, and performs temporal intersection or score aggregation.
    """
    def __init__(self, decomposer: QueryDecomposer, encoder: TextEncoderInterface, retriever: FAISSRetriever):
        self.decomposer = decomposer
        self.encoder = encoder
        self.retriever = retriever
        
    def search(self, query: str, top_k_per_cue: int = 500) -> List[Dict[str, Any]]:
        cues = self.decomposer.decompose(query)
        if not cues:
            return []
            
        print(f"Query decomposed into {len(cues)} cues: {cues}")
        
        cue_results = []
        for cue in cues:
            vec = self.encoder.encode(cue)
            frames = self.retriever.retrieve(vec, top_k=top_k_per_cue)
            cue_results.append(frames)
            
        # If there's only 1 cue, just return the results directly
        if len(cues) == 1:
            # Add cue_coverage for consistency
            for frame in cue_results[0]:
                frame["cue_coverage"] = 1.0
            return cue_results[0]
            
        # Multi-cue aggregation (Late Fusion at Frame/Video level)
        # We assign a score based on how many cues a video matches, and max score per cue.
        # But here, we just pool all frames and compute cue coverage.
        
        # Track max score per global_id and how many cues hit this video
        global_id_scores = {}
        video_cue_hits = {}
        
        for cue_idx, frames in enumerate(cue_results):
            for frame in frames:
                gid = frame["global_id"]
                vid = frame["video_id"]
                
                # Update global_id max score
                if gid not in global_id_scores:
                    global_id_scores[gid] = frame
                else:
                    global_id_scores[gid]["score"] = max(global_id_scores[gid]["score"], frame["score"])
                    
                # Update video cue hits
                if vid not in video_cue_hits:
                    video_cue_hits[vid] = set()
                video_cue_hits[vid].add(cue_idx)
                
        # Inject cue coverage into results
        aggregated_frames = []
        total_cues = len(cues)
        
        for gid, frame in global_id_scores.items():
            vid = frame["video_id"]
            frame["cue_coverage"] = len(video_cue_hits.get(vid, set())) / total_cues
            aggregated_frames.append(frame)
            
        # Sort by cue coverage first, then max score
        aggregated_frames.sort(key=lambda x: (x["cue_coverage"], x["score"]), reverse=True)
        
        return aggregated_frames
