import os
import sys
import io
import time
import json
import zipfile
import requests
from pathlib import Path
import pandas as pd

# Force UTF-8 on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

def build_batch_2_frames_metadata():
    print("=" * 80, flush=True)
    print("🚀 BẮT ĐẦU TẠO FRAMES.PARQUET & METADATA CHO BATCH 2 TỪ MAP-KEYFRAMES-B2")
    print("=" * 80, flush=True)

    out_proc = BASE_DIR / "data" / "batch_2" / "processed"
    out_proc.mkdir(parents=True, exist_ok=True)

    url_map = "https://aic-data.ledo.io.vn/map-keyframes-b2.zip"
    print(f"📥 Đang tải {url_map}...", flush=True)
    resp = requests.get(url_map, timeout=60)
    resp.raise_for_status()

    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    csv_files = [n for n in zf.namelist() if n.lower().endswith('.csv')]
    print(f"✅ Đã tìm thấy {len(csv_files)} file CSV mapping trong zip.", flush=True)

    all_frames = []
    video_records = []
    video_ranges = []
    video_zip_map = {}

    current_global_id = 0

    for csv_path in sorted(csv_files):
        video_id = Path(csv_path).stem  # e.g. K18_V008
        k_prefix = video_id.split("_")[0]  # e.g. K18
        zip_name = f"Keyframes_{k_prefix}.zip"
        video_zip_map[video_id] = f"Keyframes/{zip_name}"

        content = zf.read(csv_path).decode('utf-8')
        df_vid = pd.read_csv(io.StringIO(content))

        # Schema: n, pts_time, fps, frame_idx
        # Rename 'n' -> 'keyframe_index'
        if 'n' in df_vid.columns:
            df_vid = df_vid.rename(columns={'n': 'keyframe_index'})

        df_vid['video_id'] = video_id
        start_idx = current_global_id
        end_idx = current_global_id + len(df_vid) - 1

        df_vid['id'] = range(start_idx, end_idx + 1)
        current_global_id += len(df_vid)

        fps = float(df_vid['fps'].iloc[0]) if len(df_vid) > 0 and 'fps' in df_vid.columns else 25.0
        duration = float(df_vid['pts_time'].max()) if len(df_vid) > 0 and 'pts_time' in df_vid.columns else 0.0

        all_frames.append(df_vid)
        video_ranges.append({
            'video_id': video_id,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'count': len(df_vid)
        })
        video_records.append({
            'video_id': video_id,
            'fps': fps,
            'duration': duration,
            'num_keyframes': len(df_vid),
            'zip_name': zip_name
        })

    df_all_frames = pd.concat(all_frames, ignore_index=True)
    # Order columns: id, video_id, keyframe_index, frame_idx, pts_time, fps
    col_order = ['id', 'video_id', 'keyframe_index', 'frame_idx', 'pts_time', 'fps']
    df_all_frames = df_all_frames[[c for c in col_order if c in df_all_frames.columns]]

    # 1. Save frames.parquet
    frames_file = out_proc / "frames.parquet"
    df_all_frames.to_parquet(frames_file, index=False)
    print(f"💾 [1/4] Đã lưu {len(df_all_frames):,} keyframes vào: {frames_file}")

    # 2. Save video_ranges.parquet
    ranges_file = out_proc / "video_ranges.parquet"
    pd.DataFrame(video_ranges).to_parquet(ranges_file, index=False)
    print(f"💾 [2/4] Đã lưu phạm vi {len(video_ranges)} video vào: {ranges_file}")

    # 3. Save videos.parquet
    vids_file = out_proc / "videos.parquet"
    pd.DataFrame(video_records).to_parquet(vids_file, index=False)
    print(f"💾 [3/4] Đã lưu metadata {len(video_records)} video vào: {vids_file}")

    # 4. Save video_zip_map.json
    zip_map_file = out_proc / "video_zip_map.json"
    with open(zip_map_file, "w", encoding="utf-8") as f:
        json.dump(video_zip_map, f, ensure_ascii=False, indent=2)
    print(f"💾 [4/4] Đã lưu bảng ánh xạ video_zip_map.json ({len(video_zip_map)} videos).")

    print("\n" + "=" * 80)
    print("🎉 HOÀN TẤT TẠO TOÀN BỘ METADATA FRAMES CHO BATCH 2!")
    print(f"   • Tổng số video   : {len(video_records):,}")
    print(f"   • Tổng số keyframe: {len(df_all_frames):,}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    build_batch_2_frames_metadata()
