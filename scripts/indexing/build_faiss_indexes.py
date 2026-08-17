import os
import sys
import io
import time
from pathlib import Path
import numpy as np
import pandas as pd
import faiss

# Force UTF-8 on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

def build_faiss_index(features_path: Path, output_faiss_path: Path, dim: int, num_frames: int):
    """
    Xây dựng FAISS IndexFlatIP (Cosine Similarity) từ file vector memmap float16 / npy.
    """
    print("=" * 70, flush=True)
    print(f"📦 BẮT ĐẦU XÂY DỰNG FAISS INDEX: {output_faiss_path.name} ({dim} chiều)", flush=True)
    print("=" * 70, flush=True)
    
    if not features_path.exists():
        print(f"❌ LỖI: Không tìm thấy file vector tại {features_path}", flush=True)
        return False

    t0 = time.time()
    # Nạp ma trận bằng memmap float16
    print(f"[*] Đang nạp {num_frames:,} vectors từ: {features_path.name}...", flush=True)
    try:
        mat = np.memmap(features_path, dtype=np.float16, mode='r', shape=(num_frames, dim))
    except Exception as e:
        print(f"⚠️ Lỗi memmap, thử load npy thường: {e}", flush=True)
        mat = np.load(features_path, allow_pickle=True)

    # Chuyển sang float32 và chuẩn hóa L2 cho FAISS IndexFlatIP
    print(f"[*] Đang chuẩn hóa L2 norm và thêm vào FAISS IndexFlatIP...", flush=True)
    index = faiss.IndexFlatIP(dim)

    # Thêm theo batch để tránh tràn RAM
    batch_size = 30000
    for start_idx in range(0, num_frames, batch_size):
        end_idx = min(start_idx + batch_size, num_frames)
        chunk = mat[start_idx:end_idx].astype(np.float32)
        # Chuẩn hóa L2
        faiss.normalize_L2(chunk)
        index.add(chunk)
        print(f"   + Đã nạp {end_idx:,}/{num_frames:,} vectors ({(end_idx/num_frames)*100:.1f}%)", flush=True)

    # Lưu chỉ mục ra đĩa
    output_faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_faiss_path))
    elapsed = time.time() - t0

    file_size_mb = output_faiss_path.stat().st_size / (1024**2)
    print(f"🎉 [HOÀN TẤT] Đã lưu FAISS Index vào: {output_faiss_path}")
    print(f"   - Tổng số vector: {index.ntotal:,}")
    print(f"   - Kích thước file: {file_size_mb:.2f} MB")
    print(f"   - Thời gian xử lý: {elapsed:.2f} giây\n", flush=True)
    return True

def load_faiss_index(engine: str = "siglip2", batch: str = "batch_1", base_dir: Path = None):
    """
    Helper nạp nhanh FAISS Index và bảng metadata frames.parquet vào bộ nhớ.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent.parent

    processed_dir = base_dir / "data" / batch / "processed"
    indexes_dir = base_dir / "indexes" / batch
    frames_path = processed_dir / "frames.parquet"

    if engine == "siglip2":
        faiss_path = indexes_dir / "siglip2.faiss"
        dim = 1152
    elif engine == "clip":
        faiss_path = indexes_dir / "clip_btc.faiss"
        dim = 512
    else:
        raise ValueError(f"Unknown engine: {engine}. Use 'siglip2' or 'clip'.")

    if not faiss_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {faiss_path}. Please run build_faiss_indexes.py first.")

    index = faiss.read_index(str(faiss_path))
    df_frames = pd.read_parquet(frames_path)
    return index, df_frames

def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    processed_dir = base_dir / "data" / "batch_1" / "processed"
    indexes_dir = base_dir / "indexes" / "batch_1"
    indexes_dir.mkdir(parents=True, exist_ok=True)

    frames_file = processed_dir / "frames.parquet"
    if not frames_file.exists():
        print(f"❌ Không tìm thấy {frames_file}", flush=True)
        return

    df_frames = pd.read_parquet(frames_file)
    n_frames = len(df_frames)
    print(f"📋 Tìm thấy {n_frames:,} frames trong frames.parquet\n", flush=True)

    # 1. Build Baseline 0: BTC CLIP Index (512 dims)
    clip_file = processed_dir / "clip_features.npy"
    clip_faiss = indexes_dir / "clip_btc.faiss"
    build_faiss_index(clip_file, clip_faiss, dim=512, num_frames=n_frames)

    # 2. Build Baseline 1 SOTA: Google SigLIP 2 Index (1152 dims)
    siglip_file = processed_dir / "siglip_features.npy"
    siglip_faiss = indexes_dir / "siglip2.faiss"
    build_faiss_index(siglip_file, siglip_faiss, dim=1152, num_frames=n_frames)

    print("=" * 70, flush=True)
    print("🏆 TẤT CẢ 2 FILE FAISS INDEX ĐÃ ĐƯỢC XÂY DỰNG XONG HOÀN HẢO!")
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
