import sys
import pandas as pd
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import settings

p = settings.directories.processed / "transcripts.parquet"
if not p.exists():
    print(f"FILE_NOT_FOUND: Chưa tìm thấy file transcripts.parquet tại {p}")
else:
    df = pd.read_parquet(p)
    print("=" * 60)
    print("🎙️ BÁO CÁO PHÂN TÍCH CHẤT LƯỢNG LỜI THOẠI (TRANSCRIPTS)")
    print("=" * 60)
    print(f"- Tổng số câu thoại đã trích xuất: {len(df):,}")
    print(f"- Số lượng Video IDs: {df['video_id'].nunique():,}")
    print(f"- Danh sách Videos: {list(df['video_id'].unique())}")
    print(f"- Cột dữ liệu: {list(df.columns)}")
    print("\n" + "=" * 60)
    print("📋 10 MẪU LỜI THOẠI ĐẦU TIÊN TỪ VINAI PHOWHISPER:")
    print("=" * 60)
    for idx, row in df.head(10).iterrows():
        print(f"[{row['video_id']} | {row['start_time']}s -> {row['end_time']}s | Frames {row['start_frame']} -> {row['end_frame']}]:\n  👉 \"{row['transcript']}\"\n")
