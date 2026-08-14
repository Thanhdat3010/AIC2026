import argparse
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.preprocessing.preprocess_videos import process_videos
from src.preprocessing.preprocess_frames import process_frames
from src.preprocessing.preprocess_clip import process_clip
from src.preprocessing.preprocess_objects import process_objects
from src.config import settings

def main():
    parser = argparse.ArgumentParser(description="Preprocess AIC 2026 BTC dataset")
    parser.add_argument("--raw_dir", type=str, default=str(settings.directories.raw))
    parser.add_argument("--out_dir", type=str, default=str(settings.directories.processed))
    
    args = parser.parse_args()
    
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    
    if not raw_dir.exists():
        print(f"[ERROR] Raw directory {raw_dir} does not exist.")
        sys.exit(1)
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== Step 1: Preprocessing Video Metadata ===")
    videos_df = process_videos(raw_dir, out_dir)
    
    print("\n=== Step 2: Preprocessing Keyframe Mappings ===")
    frames_df = process_frames(raw_dir, out_dir, videos_df)
    
    print("\n=== Step 3: Preprocessing CLIP Features ===")
    process_clip(raw_dir, out_dir, settings.data.expected_keyframes)
    
    print("\n=== Step 4: Preprocessing Object JSONs ===")
    process_objects(raw_dir, out_dir, frames_df)
    
    print("\n[SUCCESS] Preprocessing completed successfully!")

if __name__ == "__main__":
    main()
