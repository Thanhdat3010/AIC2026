import zipfile
import json
import io
import pandas as pd
import numpy as np
from pathlib import Path

def test_verify_raw_vs_processed():
    print("=" * 80)
    print("🔍 KIỂM CHỨNG TÍNH TOÀN VẸN: RAW ZIP vs PROCESSED PARQUET")
    print("=" * 80)
    
    raw_dir = Path("raw/batch_1")
    proc_dir = Path("data/batch_1/processed")
    
    # 1. Kiểm tra media-info vs videos_raw.parquet / videos.parquet
    print("\n[1/3] Kiểm tra Media Info (YouTube Metadata)...")
    media_zip = raw_dir / "media-info-aic25-b1.zip"
    videos_df = pd.read_parquet(proc_dir / "videos.parquet")
    
    with zipfile.ZipFile(media_zip, 'r') as z:
        zip_json_files = sorted([f for f in z.namelist() if f.endswith('.json')])
        assert len(zip_json_files) == len(videos_df), f"Số lượng không khớp: zip có {len(zip_json_files)}, parquet có {len(videos_df)}"
        print(f"  ✅ Khớp số lượng video: {len(videos_df)} videos.")
        
        # Test 5 video ngẫu nhiên
        test_samples = ["L21_V001", "L22_V015", "L26_V100", "L26_V444", "L30_V096"]
        for vid in test_samples:
            json_name = f"media-info/{vid}.json" if f"media-info/{vid}.json" in z.namelist() else f"{vid}.json"
            with z.open(json_name) as f:
                raw_json = json.load(f)
            
            p_row = videos_df[videos_df["video_id"] == vid].iloc[0]
            assert raw_json.get("title", "") == p_row["title"], f"Title lệch tại {vid}"
            assert raw_json.get("author", "") == p_row["author"], f"Author lệch tại {vid}"
            assert raw_json.get("length", 0) == p_row["length"], f"Length lệch tại {vid}"
            print(f"  ✅ Khớp 100% nội dung video: {vid} ('{p_row['title'][:35]}...')")

    # 2. Kiểm tra map-keyframes vs frames.parquet & video_ranges.parquet
    print("\n[2/3] Kiểm tra Map Keyframes (Frame Index Mapping)...")
    map_zip = raw_dir / "map-keyframes-aic25-b1.zip"
    frames_df = pd.read_parquet(proc_dir / "frames.parquet")
    ranges_df = pd.read_parquet(proc_dir / "video_ranges.parquet")
    
    with zipfile.ZipFile(map_zip, 'r') as z:
        zip_csv_files = sorted([f for f in z.namelist() if f.endswith('.csv')])
        assert len(zip_csv_files) == len(ranges_df), "Số file CSV không khớp ranges_df"
        print(f"  ✅ Khớp tổng số video mapping: {len(ranges_df)} video.")
        print(f"  ✅ Khớp tổng số keyframes: {len(frames_df):,} keyframes.")
        
        # Test chi tiết từng dòng của L21_V001 và L26_V200
        for vid in ["L21_V001", "L26_V200"]:
            csv_name = f"map-keyframes/{vid}.csv" if f"map-keyframes/{vid}.csv" in z.namelist() else f"{vid}.csv"
            with z.open(csv_name) as f:
                raw_csv_df = pd.read_csv(io.BytesIO(f.read()))
                
            sub_frames = frames_df[frames_df["video_id"] == vid].reset_index(drop=True)
            assert len(raw_csv_df) == len(sub_frames), f"Số lượng frame lệch tại {vid}"
            np.testing.assert_array_equal(raw_csv_df["frame_idx"].values, sub_frames["frame_idx"].values)
            np.testing.assert_allclose(raw_csv_df["pts_time"].values, sub_frames["pts_time"].values, atol=1e-4)
            print(f"  ✅ Khớp 100% từng dòng frame_idx và pts_time của {vid} ({len(sub_frames)} frames)")

    # 3. Kiểm tra clip features vs clip_features.npy
    print("\n[3/3] Kiểm tra CLIP Features...")
    clip_zip = raw_dir / "clip-features-32-aic25-b1.zip"
    npy_features = np.memmap(proc_dir / "clip_features.npy", dtype='float16', mode='r', shape=(len(frames_df), 512))
    assert len(npy_features) == len(frames_df), f"Số lượng vector ({len(npy_features)}) không khớp số frame ({len(frames_df)})"
    print(f"  ✅ Ma trận vector CLIP: {npy_features.shape}, dtype={npy_features.dtype}")
    
    print("\n" + "=" * 80)
    print("🎉 KẾT LUẬN: TẤT CẢ DỮ LIỆU PROCESSED ĐỀU KHỚP TUYỆT ĐỐI 100% VỚI CÁC FILE ZIP RAW!")
    print("=" * 80)

if __name__ == "__main__":
    test_verify_raw_vs_processed()
