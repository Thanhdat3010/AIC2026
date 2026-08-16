import pandas as pd
from pathlib import Path

p = Path("data/processed/ocr_results.parquet")
if not p.exists():
    print("FILE_NOT_FOUND")
else:
    df = pd.read_parquet(p)
    print("=" * 60)
    print(f"📊 BÁO CÁO PHÂN TÍCH CHẤT LƯỢNG DATA OCR")
    print("=" * 60)
    print(f"- Tổng số khung hình có chữ: {len(df):,}")
    print(f"- Số lượng Video IDs: {df['video_id'].nunique():,}")
    print(f"- Cột dữ liệu: {list(df.columns)}")
    print(f"- Độ tin cậy trung bình: {df['confidence'].mean():.3f}")
    print(f"- Số khung hình mapping frame_idx hợp lệ (>0): {(df['frame_idx'] > 0).sum():,} / {len(df):,}")
    print("\n" + "=" * 60)
    print("📋 10 MẪU ĐẦU TIÊN:")
    print("=" * 60)
    print(df[['video_id', 'keyframe_index', 'frame_idx', 'ocr_text', 'confidence']].head(10).to_string())
    
    print("\n" + "=" * 60)
    print("🔍 15 MẪU CHỮ TIẾNG VIỆT NGẪU NHIÊN:")
    print("=" * 60)
    long_samples = df[df['ocr_text'].str.len() > 8]
    samples = long_samples.sample(min(15, len(long_samples)), random_state=42)
    for _, row in samples.iterrows():
        print(f"[{row['video_id']} | Frame {row['frame_idx']} | Conf: {row['confidence']}]: {row['ocr_text']}")
