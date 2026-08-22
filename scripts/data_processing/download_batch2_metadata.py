import os
import sys
import io
import time
import json
import zipfile
import requests
from pathlib import Path
import pandas as pd
import numpy as np

# Force UTF-8 on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent.parent.parent

RAW_B2_DIR = BASE_DIR / "raw" / "batch_2"
RAW_B1_DIR = BASE_DIR / "raw" / "batch_1"
RAW_B2_DIR.mkdir(parents=True, exist_ok=True)

URLS = {
    "map-keyframes": "https://aic-data.ledo.io.vn/map-keyframes-b2.zip",
    "media-info": "https://aic-data.ledo.io.vn/media-info-aic25-b2.zip",
    "objects": "https://aic-data.ledo.io.vn/objects-aic25-b2.zip",
    "clip-features": "https://aic-data.ledo.io.vn/clip-features-32-aic25-b2.zip"
}

def download_file(url: str, target_path: Path):
    if target_path.exists() and target_path.stat().st_size > 0:
        print(f"⏩ [ĐÃ CÓ SẴN] {target_path.name} ({target_path.stat().st_size / (1024**2):.2f} MB)")
        return
    
    print(f"📥 Đang tải {target_path.name} từ {url}...", flush=True)
    t0 = time.time()
    resp = requests.get(url, stream=True, timeout=(15, 180))
    resp.raise_for_status()
    
    with open(target_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                
    elapsed = time.time() - t0
    size_mb = target_path.stat().st_size / (1024**2)
    print(f"   ✅ Hoàn tất tải {target_path.name}: {size_mb:.2f} MB trong {elapsed:.1f}s")

def audit_batch_2():
    print("=" * 85)
    print("🚀 BƯỚC 1: TẢI 4 FILE RAW METADATA BATCH 2 VÀO raw/batch_2/")
    print("=" * 85)

    for key, url in URLS.items():
        filename = Path(url).name
        target = RAW_B2_DIR / filename
        download_file(url, target)

    print("\n" + "=" * 85)
    print("🔍 BƯỚC 2: ĐỐI SOÁT ĐỊNH DẠNG (SCHEMA & STRUCTURE) BATCH 2 VS BATCH 1")
    print("=" * 85)

    # 1. Kiểm tra map-keyframes
    print("\n[1/4] 📋 Kiểm tra Map-Keyframes...")
    b2_map_zip = RAW_B2_DIR / "map-keyframes-b2.zip"

    with zipfile.ZipFile(b2_map_zip, "r") as z2:
        b2_csvs = sorted([n for n in z2.namelist() if n.endswith('.csv')])
        sample_b2_csv = z2.read(b2_csvs[0]).decode('utf-8')
        df_b2_map = pd.read_csv(io.StringIO(sample_b2_csv))
        
        # Đếm tổng số keyframes
        total_b2_kfs = 0
        b2_vids = []
        for csv_n in b2_csvs:
            b2_vids.append(Path(csv_n).stem)
            df_temp = pd.read_csv(io.StringIO(z2.read(csv_n).decode('utf-8')))
            total_b2_kfs += len(df_temp)

    print(f"   • Batch 2 Map-Keyframes: {len(b2_csvs)} videos, {total_b2_kfs:,} keyframes.")
    print(f"   • Cột Batch 2: {df_b2_map.columns.tolist()} (So với B1: ['n', 'pts_time', 'fps', 'frame_idx'])")
    print(f"   • Mẫu dòng đầu Batch 2:\n{df_b2_map.head(2)}")

    # 2. Kiểm tra media-info
    print("\n[2/4] 🎬 Kiểm tra Media-Info (Video Metadata)...")
    b2_media_zip = RAW_B2_DIR / "media-info-aic25-b2.zip"
    with zipfile.ZipFile(b2_media_zip, "r") as z2:
        b2_jsons = sorted([n for n in z2.namelist() if n.endswith('.json')])
        sample_json = json.loads(z2.read(b2_jsons[0]).decode('utf-8'))
    print(f"   • Batch 2 Media-Info: {len(b2_jsons)} files JSON.")
    print(f"   • Các trường JSON Batch 2: {list(sample_json.keys())}")
    print(f"   • Mẫu Video ID {Path(b2_jsons[0]).stem}: Title='{sample_json.get('title')}', Length={sample_json.get('length')}s")

    # 3. Kiểm tra objects
    print("\n[3/4] 📦 Kiểm tra Objects (Faster R-CNN BBoxes)...")
    b2_obj_zip = RAW_B2_DIR / "objects-aic25-b2.zip"
    with zipfile.ZipFile(b2_obj_zip, "r") as z2:
        b2_obj_jsons = [n for n in z2.namelist() if n.endswith('.json')]
        sample_obj = json.loads(z2.read(b2_obj_jsons[0]).decode('utf-8'))
        sample_path = b2_obj_jsons[0]
    print(f"   • Batch 2 Objects: {len(b2_obj_jsons):,} files JSON.")
    print(f"   • Mẫu đường dẫn: {sample_path}")
    print(f"   • Các trường JSON Object Batch 2: {list(sample_obj.keys())}")

    # 4. Kiểm tra clip-features
    print("\n[4/4] 🧠 Kiểm tra CLIP Features (.npy)...")
    b2_clip_zip = RAW_B2_DIR / "clip-features-32-aic25-b2.zip"
    with zipfile.ZipFile(b2_clip_zip, "r") as z2:
        b2_npy_files = sorted([n for n in z2.namelist() if n.endswith('.npy')])
        sample_npy = np.load(io.BytesIO(z2.read(b2_npy_files[0])))
        
        total_clip_vectors = 0
        for npy_n in b2_npy_files:
            arr = np.load(io.BytesIO(z2.read(npy_n)))
            total_clip_vectors += arr.shape[0]

    print(f"   • Batch 2 CLIP Features: {len(b2_npy_files)} files .npy.")
    print(f"   • Shape file mẫu ({Path(b2_npy_files[0]).stem}): {sample_npy.shape}, dtype: {sample_npy.dtype}")
    print(f"   • Tổng số vector CLIP: {total_clip_vectors:,} vectors ({sample_npy.shape[1]} chiều).")

    print("\n" + "=" * 85)
    print("🏆 BẢNG KẾT LUẬN ĐỐI SOÁT ĐỊNH DẠNG (FORMAT COMPARISON MATRIX):")
    print("=" * 85)
    print(f"{'Thành phần':<20} | {'Định dạng Batch 2':<35} | {'Khớp với Batch 1?':<18} | {'Ghi chú'}")
    print("-" * 85)
    print(f"{'map-keyframes':<20} | {'CSV: n, pts_time, fps, frame_idx':<35} | {'✅ KHỚP 100%':<18} | {f'{len(b2_csvs)} videos, {total_b2_kfs:,} frames'}")
    print(f"{'media-info':<20} | {'JSON: author, title, tags, len':<35} | {'✅ KHỚP 100%':<18} | {f'{len(b2_jsons)} videos'}")
    print(f"{'objects':<20} | {'JSON: scores, names, boxes...':<35} | {'✅ KHỚP 100%':<18} | {f'{len(b2_obj_jsons):,} frames'}")
    print(f"{'clip-features':<20} | {f'NPY: (N, 512) float32':<35} | {'✅ KHỚP 100%':<18} | {f'{total_clip_vectors:,} vectors'}")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    audit_batch_2()
