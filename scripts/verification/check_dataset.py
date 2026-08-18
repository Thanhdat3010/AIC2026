import zipfile
import json
import io
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# Setup paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "raw"

# Archive filenames
CLIP_ZIP = RAW_DIR / "clip-features-32-aic25-b1.zip"
MAP_ZIP = RAW_DIR / "map-keyframes-aic25-b1.zip"
MEDIA_ZIP = RAW_DIR / "media-info-aic25-b1.zip"
OBJECT_ZIP = RAW_DIR / "objects-aic25-b1.zip"

def get_video_id(filename: str) -> str:
    """Extract Lxx_Vyyy from a filename."""
    return Path(filename).stem

def audit_dataset():
    print(f"Auditing raw datasets in {RAW_DIR}...")
    
    # Check if files exist
    for f in [CLIP_ZIP, MAP_ZIP, MEDIA_ZIP, OBJECT_ZIP]:
        if not f.exists():
            print(f"[ERROR] Missing required file: {f}")
            sys.exit(1)
            
    # 1. Read file lists from ZIPs
    print("Listing archive entries...")
    with zipfile.ZipFile(CLIP_ZIP, 'r') as z:
        clip_files = [f for f in z.namelist() if f.endswith('.npy')]
    
    with zipfile.ZipFile(MAP_ZIP, 'r') as z:
        map_files = [f for f in z.namelist() if f.endswith('.csv')]
        
    with zipfile.ZipFile(MEDIA_ZIP, 'r') as z:
        media_files = [f for f in z.namelist() if f.endswith('.json')]
        
    clip_videos = set(get_video_id(f) for f in clip_files)
    map_videos = set(get_video_id(f) for f in map_files)
    media_videos = set(get_video_id(f) for f in media_files)
    
    print(f"Found {len(clip_videos)} videos in CLIP.")
    print(f"Found {len(map_videos)} videos in MAP.")
    print(f"Found {len(media_videos)} videos in MEDIA.")
    
    if not (clip_videos == map_videos == media_videos):
        print("[ERROR] Video ID sets do not match across archives!")
        sys.exit(1)
        
    if len(clip_videos) != 873:
        print(f"[WARNING] Expected 873 videos, found {len(clip_videos)}")
        
    # 2. Check row matching and sum total keyframes
    print("Checking row-by-row consistency between CLIP features and Map Keyframes...")
    total_keyframes = 0
    total_map_rows = 0
    
    # Sort files by video id to ensure deterministic checking
    clip_files = sorted(clip_files)
    map_files = sorted(map_files)
    
    clip_z = zipfile.ZipFile(CLIP_ZIP, 'r')
    map_z = zipfile.ZipFile(MAP_ZIP, 'r')
    
    for c_file, m_file in tqdm(zip(clip_files, map_files), total=len(clip_files), desc="Auditing Videos"):
        v_id1 = get_video_id(c_file)
        v_id2 = get_video_id(m_file)
        assert v_id1 == v_id2, "Mismatch in sorted file iteration."
        
        # Read Numpy shape without loading full data if possible, but they are small so we can load
        with clip_z.open(c_file) as f:
            # We can use np.load with io.BytesIO to read npy from zip
            arr = np.load(io.BytesIO(f.read()))
            clip_rows = arr.shape[0]
            assert arr.shape[1] == 512, f"Expected 512 dimensions in {c_file}"
            
        with map_z.open(m_file) as f:
            df = pd.read_csv(io.BytesIO(f.read()))
            map_rows = len(df)
            
        if clip_rows != map_rows:
            print(f"\n[ERROR] Row mismatch for {v_id1}: CLIP={clip_rows}, MAP={map_rows}")
            sys.exit(1)
            
        total_keyframes += clip_rows
        total_map_rows += map_rows

    clip_z.close()
    map_z.close()
    
    print(f"\nTotal Keyframes (CLIP): {total_keyframes}")
    print(f"Total Keyframes (MAP) : {total_map_rows}")
    
    if total_keyframes != 177321:
        print(f"[WARNING] Expected 177321 keyframes, found {total_keyframes}")
        
    print("\n[SUCCESS] Dataset audit passed! No missing files, row counts match perfectly.")

if __name__ == "__main__":
    audit_dataset()
