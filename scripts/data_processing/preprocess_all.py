import argparse
from pathlib import Path
import sys

# Force UTF-8 on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.preprocessing.preprocess_videos import process_videos
from src.preprocessing.preprocess_frames import process_frames
from src.preprocessing.preprocess_clip import process_clip
from src.preprocessing.preprocess_objects import process_objects
from src.config import settings

def main():
    parser = argparse.ArgumentParser(description="Preprocess AIC 2026 BTC dataset")
    parser.add_argument("--batch", type=str, default="batch_1",
                        help="Tên batch cần tiền xử lý ('batch_1', 'batch_2')")
    parser.add_argument("--raw_dir", type=str, default=None,
                        help="Thư mục chứa raw metadata ZIP")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Thư mục lưu output processed parquet")
    
    args = parser.parse_args()
    
    if args.raw_dir:
        raw_dir = Path(args.raw_dir)
    else:
        raw_dir = PROJECT_ROOT / "raw" / args.batch
        
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = PROJECT_ROOT / "data" / args.batch / "processed"
    
    if not raw_dir.exists():
        print(f"[ERROR] Thư mục raw {raw_dir} không tồn tại.")
        sys.exit(1)
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"🚀 BẮT ĐẦU TIỀN XỬ LÝ TOÀN BỘ DỮ LIỆU {args.batch.upper()}")
    print(f"   • Raw Directory      : {raw_dir}")
    print(f"   • Processed Directory: {out_dir}")
    print("=" * 80)
    
    print("\n=== Step 1: Preprocessing Video Metadata ===")
    videos_df = process_videos(raw_dir, out_dir)
    
    print("\n=== Step 2: Preprocessing Keyframe Mappings ===")
    frames_df = process_frames(raw_dir, out_dir, videos_df)
    
    print("\n=== Step 3: Preprocessing CLIP Features ===")
    process_clip(raw_dir, out_dir)
    
    print("\n=== Step 4: Preprocessing Object JSONs ===")
    process_objects(raw_dir, out_dir, frames_df)
    
    print("\n" + "=" * 80)
    print(f"🎉 [SUCCESS] ĐÃ HOÀN TẤT TIỀN XỬ LÝ TOÀN BỘ DỮ LIỆU {args.batch.upper()} VÀO {out_dir}!")
    print("=" * 80)

if __name__ == "__main__":
    main()

