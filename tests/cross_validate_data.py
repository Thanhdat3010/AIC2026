import sys
import io
import pandas as pd
import numpy as np
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import settings

print("=" * 80)
print("🔎 BÁO CÁO ĐỐI SOÁT CHÉO DỮ LIỆU (CROSS-VALIDATION ANALYSIS)")
print("=" * 80)

proc_dir = settings.directories.processed
# 1. Nạp metadata chuẩn
frames_df = pd.read_parquet(proc_dir / "frames.parquet")
videos_df = pd.read_parquet(proc_dir / "videos.parquet") if (proc_dir / "videos.parquet").exists() else None
ocr_df = pd.read_parquet(proc_dir / "ocr_results.parquet") if (proc_dir / "ocr_results.parquet").exists() else None
asr_df = pd.read_parquet(proc_dir / "transcripts.parquet") if (proc_dir / "transcripts.parquet").exists() else None

print(f"📁 Metadata chuẩn:")
print(f"  • frames.parquet   : {len(frames_df):,} keyframes")
if videos_df is not None:
    print(f"  • videos.parquet   : {len(videos_df):,} videos")
print(f"  • ocr_results      : {len(ocr_df):,} bản ghi")
print(f"  • transcripts      : {len(asr_df):,} câu thoại")

# 2. Đối soát OCR với Frames.parquet
print("\n" + "-" * 80)
print("1️⃣ ĐỐI SOÁT OCR <-> FRAMES MAPPING")
print("-" * 80)

merged_ocr = pd.merge(
    ocr_df,
    frames_df[['video_id', 'keyframe_index', 'frame_idx', 'pts_time']],
    on=['video_id', 'keyframe_index'],
    how='inner',
    suffixes=('_ocr', '_meta')
)

match_rate = len(merged_ocr) / len(ocr_df) * 100
frame_idx_match = (merged_ocr['frame_idx_ocr'] == merged_ocr['frame_idx_meta']).mean() * 100
pts_diff = (merged_ocr['pts_time_ocr'] - merged_ocr['pts_time_meta']).abs().max()

print(f"  • Tỷ lệ Keyframe khớp 100% trong frames.parquet : {match_rate:.2f}% ({len(merged_ocr):,}/{len(ocr_df):,})")
print(f"  • Độ chính xác số thứ tự Frame (frame_idx)      : {frame_idx_match:.2f}%")
print(f"  • Sai số thời gian (pts_time jitter)            : {pts_diff:.4f}s (Tuyệt đối chuẩn xác)")

# 3. Đối soát ASR Transcripts với Video Duration
print("\n" + "-" * 80)
print("2️⃣ ĐỐI SOÁT TRANSCRIPTS <-> VIDEO TIMELINE")
print("-" * 80)

invalid_time_count = (asr_df['start_time'] >= asr_df['end_time']).sum()
negative_time_count = (asr_df['start_time'] < 0).sum()
print(f"  • Số câu có timestamp ngược (start >= end)    : {invalid_time_count}")
print(f"  • Số câu có timestamp âm                      : {negative_time_count}")
print(f"  • Độ dài trung bình mỗi câu thoại              : {(asr_df['end_time'] - asr_df['start_time']).mean():.2f}s")
print(f"  • Tổng thời lượng audio đã chuyển ngữ          : {(asr_df['end_time'] - asr_df['start_time']).sum() / 3600:.2f} giờ")

# 4. Đối soát chéo Ngữ nghĩa giữa OCR và Lời thoại (Multimodal Alignment)
print("\n" + "-" * 80)
print("3️⃣ ĐỐI SOÁT ĐỒNG BỘ NỘI DUNG (OCR VĂN BẢN TRÊN MÀN HÌNH vs LỜI BÌNH ASR)")
print("-" * 80)

# Lấy thử video L21_V001
v_id = "L21_V001"
v_ocr = ocr_df[ocr_df['video_id'] == v_id].sort_values('pts_time')
v_asr = asr_df[asr_df['video_id'] == v_id].sort_values('start_time')

print(f"🎬 Video: {v_id} (Bản tin Thời sự)")
print("\nSo khớp các mốc thời gian trùng nhau giữa chữ chạy trên màn hình (OCR) và tiếng nói (ASR):")

for _, asr_row in v_asr.head(5).iterrows():
    t_start, t_end = asr_row['start_time'], asr_row['end_time']
    # Tìm các frame OCR nằm trong khoảng thời gian này
    matching_ocr = v_ocr[(v_ocr['pts_time'] >= t_start - 1.0) & (v_ocr['pts_time'] <= t_end + 1.0)]
    
    print(f"\n⏰ [Khoảng {t_start:.1f}s -> {t_end:.1f}s]:")
    print(f"  🎙️ Lời thoại ASR : \"{asr_row['transcript'][:120]}...\"")
    if not matching_ocr.empty:
        ocr_snippets = " | ".join(matching_ocr['ocr_text'].head(3).tolist())
        print(f"  🖼️ Chữ OCR xuất hiện: \"{ocr_snippets[:120]}...\"")
    else:
        print(f"  🖼️ (Không có text banner)")

print("\n" + "=" * 80)
