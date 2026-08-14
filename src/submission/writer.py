import csv
from pathlib import Path
from typing import List, Dict, Any

class SubmissionWriter:
    """
    Writes KIS results to the format required by BTC:
    video_id,frame_idx (no header)
    """
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def write(self, query_id: str, results: List[Dict[str, Any]], top_k: int = 100) -> Path:
        output_path = self.output_dir / f"query-{query_id}-kis.csv"
        
        # Take the top K results (or fewer if we don't have K)
        top_results = results[:top_k]
        
        with open(output_path, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            for frame in top_results:
                writer.writerow([frame['video_id'], frame['frame_idx']])
                
        print(f"Written {len(top_results)} results to {output_path}")
        return output_path
