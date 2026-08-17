import os
import io
import sys
import zipfile
from pathlib import Path
from PIL import Image
import pandas as pd

# Force UTF-8 on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class KeyframeZipLoader:
    """
    Bộ nạp Keyframe siêu tốc trực tiếp từ file Zip (Zero Disk Extract Overhead):
    - Tự động định tuyến video_id -> file zip tương ứng.
    - Đọc ảnh JPEG vào bộ nhớ và trả về PIL Image trong vòng <5ms.
    """
    def __init__(self, keyframes_dir: Path = None, processed_dir: Path = None):
        if keyframes_dir is None:
            keyframes_dir = BASE_DIR / "raw" / "batch_1" / "Keyframes"
        if processed_dir is None:
            processed_dir = BASE_DIR / "data" / "batch_1" / "processed"

        self.keyframes_dir = keyframes_dir
        self.processed_dir = processed_dir

        # Đọc bảng tra cứu frames.parquet (Vectorized dictionary lookup)
        self.df_frames = pd.read_parquet(self.processed_dir / "frames.parquet")
        self.lookup = dict(zip(zip(self.df_frames["video_id"], self.df_frames["frame_idx"].astype(int)), self.df_frames["keyframe_index"].astype(int)))

        # Cache các zipfile handle mở sẵn
        self.zip_handles = {}
        self._init_zip_handles()

    def _init_zip_handles(self):
        if not self.keyframes_dir.exists():
            return
        for zpath in self.keyframes_dir.glob("Keyframes_*.zip"):
            try:
                self.zip_handles[zpath.name] = zipfile.ZipFile(zpath, 'r')
            except Exception as e:
                print(f"⚠️ Không thể mở zip {zpath.name}: {e}", flush=True)

    def get_image(self, video_id: str, frame_idx: int) -> Image.Image:
        """
        Nạp ảnh PIL Image cho một video_id và frame_idx cụ thể.
        """
        key = (video_id, int(frame_idx))
        if key in self.lookup:
            k_idx = self.lookup[key]
        else:
            # Tìm frame gần nhất của video đó
            df_v = self.df_frames[self.df_frames["video_id"] == video_id]
            if df_v.empty:
                return None
            diffs = (df_v["frame_idx"] - frame_idx).abs()
            nearest_row = df_v.loc[diffs.idxmin()]
            k_idx = int(nearest_row["keyframe_index"])

        # Xác định tên zip
        prefix = video_id.split("_")[0] # ví dụ L27
        candidate_zips = [k for k in self.zip_handles.keys() if f"Keyframes_{prefix}" in k]
        if not candidate_zips:
            return None

        # Thử các định dạng tên file: 001.jpg, 0001.jpg, 1.jpg
        name_formats = [
            f"keyframes/{video_id}/{k_idx:03d}.jpg",
            f"keyframes/{video_id}/{k_idx:04d}.jpg",
            f"keyframes/{video_id}/{k_idx}.jpg",
            f"{video_id}/{k_idx:03d}.jpg",
            f"{video_id}/{k_idx:04d}.jpg",
            f"{video_id}/{k_idx}.jpg",
        ]

        for zname in candidate_zips:
            zh = self.zip_handles[zname]
            for target_name in name_formats:
                try:
                    img_bytes = zh.read(target_name)
                    return Image.open(io.BytesIO(img_bytes)).convert("RGB")
                except KeyError:
                    continue

        return None

if __name__ == "__main__":
    loader = KeyframeZipLoader()
    print("Testing loader on L27_V002 frame 920...")
    img = loader.get_image("L27_V002", 920)
    if img:
        print(f"✅ Loaded Image thành công! Kích thước: {img.size}")
    else:
        print("❌ Không tìm thấy ảnh!")
