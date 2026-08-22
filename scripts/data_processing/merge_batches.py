import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import faiss

# Force UTF-8 on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.indexing.bm25_indexer import BM25MultiIndexer

def merge_batches(batch_list: list[str], output_batch_name: str = "merged"):
    print("=" * 85, flush=True)
    print(f"🚀 BẮT ĐẦU QUY TRÌNH HỢP NHẤT DỮ LIỆU ĐA PHƯƠNG THỨC (MERGE BATCHES)")
    print(f"📦 Danh sách Batch đầu vào : {batch_list}")
    print(f"📁 Thư mục xuất dữ liệu gộp: data/{output_batch_name}/processed/")
    print("=" * 85, flush=True)

    out_processed = BASE_DIR / "data" / output_batch_name / "processed"
    out_indexes = BASE_DIR / "indexes" / output_batch_name
    out_processed.mkdir(parents=True, exist_ok=True)
    out_indexes.mkdir(parents=True, exist_ok=True)

    # 1. Hợp nhất frames.parquet và zip_map
    print("\n[1/5] 📋 Đang hợp nhất frames.parquet & video_zip_map.json...")
    merged_frames = []
    merged_zip_map = {}
    current_global_id = 0

    for b in batch_list:
        b_proc = BASE_DIR / "data" / b / "processed"
        frames_p = b_proc / "frames.parquet"
        zip_map_p = b_proc / "video_zip_map.json"

        if not frames_p.exists():
            print(f"❌ CẢNH BÁO: Không tìm thấy {frames_p} của {b}!")
            continue

        df_f = pd.read_parquet(frames_p).copy()
        # Đánh lại ID toàn cục liên tục
        df_f["id"] = range(current_global_id, current_global_id + len(df_f))
        current_global_id += len(df_f)
        merged_frames.append(df_f)

        if zip_map_p.exists():
            with open(zip_map_p, "r", encoding="utf-8") as f:
                zmap = json.load(f)
            merged_zip_map.update(zmap)

    if not merged_frames:
        print("❌ LỖI: Không có dữ liệu frames nào để gộp!")
        return

    df_merged_frames = pd.concat(merged_frames, ignore_index=True)
    df_merged_frames.to_parquet(out_processed / "frames.parquet", index=False)
    with open(out_processed / "video_zip_map.json", "w", encoding="utf-8") as f:
        json.dump(merged_zip_map, f, ensure_ascii=False, indent=2)

    total_frames = len(df_merged_frames)
    print(f"   ✅ Đã gộp thành công {total_frames:,} keyframes ({len(merged_zip_map)} videos).")

    # 2. Hợp nhất SigLIP-2 Features (.npy) & Xây lại FAISS Index
    print("\n[2/5] 🧠 Đang hợp nhất ma trận vector SigLIP-2 & Re-indexing FAISS...")
    siglip_arrays = []
    for b in batch_list:
        b_proc = BASE_DIR / "data" / b / "processed"
        siglip_p = b_proc / "siglip_features.npy"
        if siglip_p.exists():
            arr = np.load(siglip_p, mmap_mode="r")
            siglip_arrays.append(arr)
            print(f"   + Nạp {b} SigLIP: {arr.shape}")
        else:
            print(f"⚠️ {b} chưa có siglip_features.npy!")

    if siglip_arrays:
        t0 = time.time()
        merged_siglip = np.concatenate(siglip_arrays, axis=0)
        np.save(out_processed / "siglip_features.npy", merged_siglip)
        print(f"   💾 Đã lưu ma trận gộp: {merged_siglip.shape} vào siglip_features.npy ({time.time() - t0:.2f}s)")

        # Re-build FAISS index
        t0 = time.time()
        dim = merged_siglip.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(merged_siglip.astype(np.float32))
        faiss_path = out_indexes / "siglip2.faiss"
        faiss.write_index(index, str(faiss_path))
        print(f"   🎉 Đã xây FAISS Index gộp: {index.ntotal:,} vectors tại {faiss_path.name} ({time.time() - t0:.2f}s)")

    # 3. Hợp nhất OCR Results (.parquet)
    print("\n[3/5] 📝 Đang hợp nhất OCR Results...")
    ocr_dfs = []
    for b in batch_list:
        b_proc = BASE_DIR / "data" / b / "processed"
        ocr_p = b_proc / "ocr_results.parquet"
        if ocr_p.exists():
            df_ocr = pd.read_parquet(ocr_p)
            ocr_dfs.append(df_ocr)
            print(f"   + Nạp {b} OCR: {len(df_ocr):,} records")

    if ocr_dfs:
        merged_ocr = pd.concat(ocr_dfs, ignore_index=True)
        merged_ocr.to_parquet(out_processed / "ocr_results.parquet", index=False)
        print(f"   💾 Đã lưu {len(merged_ocr):,} OCR records.")

    # 4. Hợp nhất Transcripts ASR (.parquet)
    print("\n[4/5] 🎙️ Đang hợp nhất Whisper Transcripts...")
    asr_dfs = []
    for b in batch_list:
        b_proc = BASE_DIR / "data" / b / "processed"
        asr_p = b_proc / "transcripts.parquet"
        if asr_p.exists():
            df_asr = pd.read_parquet(asr_p)
            asr_dfs.append(df_asr)
            print(f"   + Nạp {b} ASR: {len(df_asr):,} lines")

    if asr_dfs:
        merged_asr = pd.concat(asr_dfs, ignore_index=True)
        merged_asr.to_parquet(out_processed / "transcripts.parquet", index=False)
        print(f"   💾 Đã lưu {len(merged_asr):,} ASR lines.")

    # 5. Khởi tạo & Lưu Cache BM25 Indexer cho thư mục gộp
    print("\n[5/5] ⚡ Đang tiền xử lý & lưu Cache BM25 OCR/ASR cho Merged Dataset...")
    t0 = time.time()
    indexer = BM25MultiIndexer(batch=output_batch_name)
    indexer._load_or_build_ocr_index()
    indexer._load_or_build_asr_index()
    print(f"   🎉 BM25 Indexer đã sẵn sàng ({time.time() - t0:.2f}s)")

    print("\n" + "=" * 85)
    print(f"🏆 HOÀN TẤT HỢP NHẤT TOÀN BỘ DỮ LIỆU VÀO: data/{output_batch_name}/processed/")
    print(f"   • Dữ liệu lẻ của {batch_list} được GIỮ NGUYÊN 100% không suy suyển!")
    print(f"   • Hệ thống đã sẵn sàng tìm kiếm trên toàn bộ kho dữ liệu hợp nhất!")
    print("=" * 85 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Script hợp nhất đa phương thức các Batch cho AIC 2026")
    parser.add_argument("--batches", nargs="+", default=["batch_1"], help="Danh sách các batch cần gộp (ví dụ: batch_1 batch_2)")
    parser.add_argument("--output", type=str, default="merged", help="Tên thư mục batch gộp (mặc định 'merged')")
    args = parser.parse_args()

    merge_batches(batch_list=args.batches, output_batch_name=args.output)

if __name__ == "__main__":
    main()
