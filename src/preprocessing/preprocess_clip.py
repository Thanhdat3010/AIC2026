import zipfile
import io
import numpy as np
from pathlib import Path
from tqdm import tqdm

def process_clip(raw_dir: Path, out_dir: Path, expected_keyframes: int):
    clip_zip_path = raw_dir / "clip-features-32-aic25-b1.zip"
    
    out_path = out_dir / "clip_features.npy"
    
    # Initialize a memory-mapped array for 177321 x 512 in float16
    fp = np.memmap(out_path, dtype='float16', mode='w+', shape=(expected_keyframes, 512))
    
    current_idx = 0
    
    with zipfile.ZipFile(clip_zip_path, 'r') as z:
        npy_files = sorted([f for f in z.namelist() if f.endswith('.npy')])
        
        for f in tqdm(npy_files, desc="Processing CLIP Features"):
            with z.open(f) as nf:
                # Load the individual numpy array from bytes
                arr = np.load(io.BytesIO(nf.read()))
                
                rows = arr.shape[0]
                # Copy into the memory-mapped file
                fp[current_idx:current_idx + rows, :] = arr
                current_idx += rows
                
    fp.flush()
    print(f"Saved concatenated CLIP features of shape ({current_idx}, 512) to {out_path}")
