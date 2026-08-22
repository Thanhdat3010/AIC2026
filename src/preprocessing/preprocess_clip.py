import zipfile
import io
import numpy as np
from pathlib import Path
from tqdm import tqdm

def process_clip(raw_dir: Path, out_dir: Path, expected_keyframes: int = None):
    clip_zips = list(raw_dir.glob("clip-features*.zip"))
    if not clip_zips:
        raise FileNotFoundError(f"Không tìm thấy clip-features*.zip trong {raw_dir}")
    clip_zip_path = clip_zips[0]
    
    out_path = out_dir / "clip_features.npy"
    
    arrays = []
    total_rows = 0
    dim = 512
    
    with zipfile.ZipFile(clip_zip_path, 'r') as z:
        npy_files = sorted([f for f in z.namelist() if f.endswith('.npy')])
        
        for f in tqdm(npy_files, desc="Processing CLIP Features"):
            with z.open(f) as nf:
                arr = np.load(io.BytesIO(nf.read()))
                if arr.dtype != np.float16:
                    arr = arr.astype(np.float16)
                arrays.append(arr)
                total_rows += arr.shape[0]
                dim = arr.shape[1]
                
    if arrays:
        merged = np.concatenate(arrays, axis=0)
        np.save(out_path, merged)
        print(f"Saved concatenated CLIP features of shape {merged.shape} to {out_path}")

