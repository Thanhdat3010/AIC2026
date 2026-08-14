import json
import zipfile
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.config import settings

def process_videos(raw_dir: Path, out_dir: Path):
    media_zip_path = raw_dir / "media-info-aic25-b1.zip"
    
    videos_data = []
    
    with zipfile.ZipFile(media_zip_path, 'r') as z:
        json_files = sorted([f for f in z.namelist() if f.endswith('.json')])
        
        for f in tqdm(json_files, desc="Processing Video Metadata"):
            video_id = Path(f).stem
            with z.open(f) as jf:
                data = json.load(jf)
                
            # Handle potentially missing fields
            author = data.get("author", "")
            title = data.get("title", "")
            description = data.get("description", "") or ""
            keywords = data.get("keywords", []) or []
            length = data.get("length", 0)
            publish_date = data.get("publish_date", "")
            
            # Create a combined search text for baseline metadata retrieval
            search_text = f"{title} {description} {' '.join(keywords)}".lower()
            
            videos_data.append({
                "video_id": video_id,
                "author": author,
                "title": title,
                "description": description,
                "keywords": keywords,
                "length": length,
                "publish_date": publish_date,
                "search_text": search_text
            })
            
    df = pd.DataFrame(videos_data)
    
    # Save to Parquet
    out_path = out_dir / "videos_raw.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} video records to {out_path}")
    return df
