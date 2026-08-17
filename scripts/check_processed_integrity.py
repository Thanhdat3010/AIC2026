import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"d:\HCMUS\AIC2026\data\batch_1\processed")

print("=" * 70)
print("[CHECK] BAT DAU KIEM TRA DO DAY DU VA TOAN VEN CUA DU LIEU BATCH 1")
print("=" * 70)

# 1. Doc frames.parquet & videos.parquet
frames_df = pd.read_parquet(DATA_DIR / "frames.parquet")
videos_df = pd.read_parquet(DATA_DIR / "videos.parquet") if (DATA_DIR / "videos.parquet").exists() else None

all_video_ids = sorted(frames_df["video_id"].unique().tolist())
total_videos = len(all_video_ids)
total_frames = len(frames_df)

print(f"[CHUAN] Tong so Video chuan: {total_videos:,}")
print(f"[CHUAN] Tong so Keyframes chuan: {total_frames:,}")
print(f"[CHUAN] Danh sach Video range: tu {all_video_ids[0]} den {all_video_ids[-1]}")
print("-" * 70)

# 2. Kiem tra ocr_results.parquet
ocr_path = DATA_DIR / "ocr_results.parquet"
if ocr_path.exists():
    ocr_df = pd.read_parquet(ocr_path)
    ocr_videos = sorted(ocr_df["video_id"].unique().tolist())
    ocr_unique_frames = len(ocr_df[["video_id", "frame_idx"]].drop_duplicates())
    
    missing_ocr_videos = set(all_video_ids) - set(ocr_videos)
    
    print(f"[OCR RESULTS]:")
    print(f"   • Kich thuoc file: {ocr_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"   • Tong so dong text (Bounding Boxes): {len(ocr_df):,}")
    print(f"   • So video co ket qua OCR: {len(ocr_videos)} / {total_videos} ({len(ocr_videos)/total_videos*100:.1f}%)")
    print(f"   • So keyframes co phat hien chu: {ocr_unique_frames:,}")
    print(f"   • Cac cot du lieu: {list(ocr_df.columns)}")
    print(f"   • Kiem tra NaN: {ocr_df.isna().sum().to_dict()}")
    print(f"   • Video dau - cuoi OCR: {ocr_videos[0]} -> {ocr_videos[-1]}")
    
    if len(missing_ocr_videos) == 0:
        print(f"   👉 [STATUS] DO BAO PHU OCR: 100% HOAN TAT ({total_videos} videos)!")
    else:
        print(f"   👉 [STATUS] Con thieu {len(missing_ocr_videos)} videos: {sorted(list(missing_ocr_videos))[:10]}...")
else:
    print("[ERROR] Khong tim thay ocr_results.parquet!")

print("-" * 70)

# 3. Kiem tra transcripts.parquet
asr_path = DATA_DIR / "transcripts.parquet"
if asr_path.exists():
    asr_df = pd.read_parquet(asr_path)
    asr_videos = sorted(asr_df["video_id"].unique().tolist())
    total_segments = len(asr_df)
    
    missing_asr_videos = set(all_video_ids) - set(asr_videos)
    
    print(f"[ASR TRANSCRIPTS]:")
    print(f"   • Kich thuoc file: {asr_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"   • Tong so cau thoai (Segments): {total_segments:,}")
    print(f"   • So video co loi thoai: {len(asr_videos)} / {total_videos} ({len(asr_videos)/total_videos*100:.1f}%)")
    print(f"   • Cac cot du lieu: {list(asr_df.columns)}")
    print(f"   • Kiem tra NaN: {asr_df.isna().sum().to_dict()}")
    print(f"   • Video dau - cuoi ASR: {asr_videos[0]} -> {asr_videos[-1]}")
    
    if len(missing_asr_videos) == 0:
        print(f"   👉 [STATUS] DO BAO PHU ASR: 100% HOAN TAT ({total_videos} videos)!")
    else:
        print(f"   👉 [STATUS] So video khong co segment thoai: {len(missing_asr_videos)} videos: {sorted(list(missing_asr_videos))[:10]}...")
else:
    print("[ERROR] Khong tim thay transcripts.parquet!")

print("=" * 70)
