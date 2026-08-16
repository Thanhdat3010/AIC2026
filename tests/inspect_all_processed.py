import sys
import io
import pandas as pd
from pathlib import Path

# Force UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 70)
print("🔍 BÁO CÁO TOÀN DIỆN DỮ LIỆU ĐÃ TRÍCH XUẤT (OCR & TRANSCRIPTS)")
print("=" * 70)

ocr_path = Path("data/processed/ocr_results.parquet")
if ocr_path.exists():
    ocr_df = pd.read_parquet(ocr_path)
    print(f"\n🖼️ [1. DỮ LIỆU OCR (CRAFT + VietOCR)]")
    print(f"  • Tổng số khung hình có chữ : {len(ocr_df):,} bản ghi")
    print(f"  • Số lượng Videos đã quét   : {ocr_df['video_id'].nunique():,} video")
    print(f"  • Danh sách Video IDs       : {sorted(ocr_df['video_id'].unique())[:15]} ...")
    print(f"  • Độ tin cậy trung bình     : {ocr_df['confidence'].mean():.2f}")
    print("\n  👉 8 MẪU OCR TIÊU BIỂU:")
    samples = ocr_df[ocr_df['ocr_text'].str.len() > 10].head(8)
    for _, r in samples.iterrows():
        print(f"    - [{r['video_id']} | KF #{r['keyframe_index']} | Frame #{r['frame_idx']}]: \"{r['ocr_text']}\"")
else:
    print("\n❌ Không tìm thấy data/processed/ocr_results.parquet")

asr_path = Path("data/processed/transcripts.parquet")
if asr_path.exists():
    asr_df = pd.read_parquet(asr_path)
    print(f"\n🎙️ [2. DỮ LIỆU TRANSCRIPTS (VinAI PhoWhisper-large)]")
    print(f"  • Tổng số câu thoại đã nghe : {len(asr_df):,} câu")
    print(f"  • Số lượng Videos đã ASR    : {asr_df['video_id'].nunique():,} video")
    print(f"  • Danh sách Video IDs       : {sorted(asr_df['video_id'].unique())[:15]} ...")
    print("\n  👉 8 MẪU CÂU THOẠI TIÊU BIỂU:")
    for _, r in asr_df.head(8).iterrows():
        print(f"    - [{r['video_id']} | {r['start_time']:.1f}s -> {r['end_time']:.1f}s | Frame {r['start_frame']}->{r['end_frame']}]:")
        print(f"      \"{r['transcript']}\"")
else:
    print("\n❌ Không tìm thấy data/processed/transcripts.parquet")

print("\n" + "=" * 70)
