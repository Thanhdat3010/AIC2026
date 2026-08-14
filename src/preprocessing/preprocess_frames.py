import zipfile
import io
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def process_frames(raw_dir: Path, out_dir: Path, videos_df: pd.DataFrame):
    map_zip_path = raw_dir / "map-keyframes-aic25-b1.zip"
    
    frames_data = []
    video_ranges = []
    
    global_id_counter = 0
    
    # Create a lookup for video lengths
    video_lengths = dict(zip(videos_df['video_id'], videos_df['length']))
    
    with zipfile.ZipFile(map_zip_path, 'r') as z:
        csv_files = sorted([f for f in z.namelist() if f.endswith('.csv')])
        
        for f in tqdm(csv_files, desc="Processing Keyframe Mappings"):
            video_id = Path(f).stem
            length = video_lengths.get(video_id, 1.0)
            if length == 0:
                length = 1.0 # prevent division by zero
                
            with z.open(f) as cf:
                df = pd.read_csv(io.BytesIO(cf.read()))
            
            num_keyframes = len(df)
            if num_keyframes == 0:
                continue
                
            first_global_id = global_id_counter
            
            # Map columns
            # df has: n, pts_time, fps, frame_idx
            for i, row in df.iterrows():
                pts_time = float(row['pts_time'])
                frames_data.append({
                    "global_id": global_id_counter,
                    "video_id": video_id,
                    "keyframe_index": int(row['n']),
                    "pts_time": pts_time,
                    "fps": float(row['fps']),
                    "frame_idx": int(row['frame_idx']),
                    "clip_row": i,
                    "position_ratio": min(1.0, pts_time / length)
                })
                global_id_counter += 1
                
            last_global_id = global_id_counter - 1
            
            video_ranges.append({
                "video_id": video_id,
                "first_global_id": first_global_id,
                "last_global_id": last_global_id,
                "num_keyframes": num_keyframes
            })
            
    frames_df = pd.DataFrame(frames_data)
    ranges_df = pd.DataFrame(video_ranges)
    
    # Save Parquet
    frames_path = out_dir / "frames.parquet"
    ranges_path = out_dir / "video_ranges.parquet"
    
    frames_df.to_parquet(frames_path, index=False)
    ranges_df.to_parquet(ranges_path, index=False)
    
    # Merge video ranges into videos_df and overwrite
    final_videos_df = pd.merge(videos_df, ranges_df, on="video_id", how="left")
    final_videos_path = out_dir / "videos.parquet"
    final_videos_df.to_parquet(final_videos_path, index=False)
    
    print(f"Saved {len(frames_df)} frames to {frames_path}")
    print(f"Saved {len(ranges_df)} video ranges to {ranges_path}")
    print(f"Updated videos.parquet with ranges")
    
    return frames_df
