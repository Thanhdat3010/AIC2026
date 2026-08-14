import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from src.indexing.faiss_indexer import FAISSIndexer

class FAISSRetriever:
    """
    Wraps the FAISS index and the frames metadata to return 
    human-readable frame results for vector queries.
    """
    def __init__(self, index_path: Path, frames_meta_path: Path):
        print(f"Loading FAISS Index from {index_path}...")
        self.indexer = FAISSIndexer(index_path)
        
        print(f"Loading Frames Metadata from {frames_meta_path}...")
        # Load parquet. We need global_id, video_id, frame_idx, pts_time, position_ratio
        self.frames_df = pd.read_parquet(frames_meta_path)
        # Ensure it's indexed by global_id for fast O(1) lookup
        self.frames_df = self.frames_df.set_index("global_id")
        
    def retrieve(self, query_vector: np.ndarray, top_k: int = 100) -> List[Dict[str, Any]]:
        distances, indices = self.indexer.search(query_vector, top_k=top_k)
        
        results = []
        # distances and indices are shape (1, top_k)
        for i in range(len(indices[0])):
            global_id = indices[0][i]
            score = float(distances[0][i])
            
            if global_id == -1:
                continue # FAISS returns -1 if not enough results
                
            # Lookup metadata
            try:
                row = self.frames_df.loc[global_id]
                results.append({
                    "global_id": int(global_id),
                    "video_id": row["video_id"],
                    "frame_idx": int(row["frame_idx"]),
                    "pts_time": float(row["pts_time"]),
                    "position_ratio": float(row.get("position_ratio", 0.0)),
                    "score": score
                })
            except KeyError:
                print(f"[WARNING] Global ID {global_id} found in index but missing in metadata.")
                
        return results
