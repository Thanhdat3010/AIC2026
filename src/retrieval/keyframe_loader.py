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
        elif isinstance(keyframes_dir, str) and not ("/" in keyframes_dir or "\\" in keyframes_dir):
            batch_name = keyframes_dir
            keyframes_dir = BASE_DIR / "raw" / batch_name / "Keyframes"
            if processed_dir is None:
                processed_dir = BASE_DIR / "data" / batch_name / "processed"

        self.keyframes_dir = Path(keyframes_dir)
        self.processed_dir = Path(processed_dir) if processed_dir is not None else (BASE_DIR / "data" / "batch_1" / "processed")

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

    def get_image_bytes(self, video_id: str, frame_idx: int) -> bytes:
        """
        Đọc trực tiếp raw JPEG bytes từ file zip không qua bước giải mã PIL (sub-millisecond streaming).
        """
        key = (video_id, int(frame_idx))
        if key in self.lookup:
            k_idx = self.lookup[key]
        else:
            df_v = self.df_frames[self.df_frames["video_id"] == video_id]
            if df_v.empty:
                return None
            diffs = (df_v["frame_idx"] - frame_idx).abs()
            nearest_row = df_v.loc[diffs.idxmin()]
            k_idx = int(nearest_row["keyframe_index"])

        prefix = video_id.split("_")[0]
        candidate_zips = [k for k in self.zip_handles.keys() if f"Keyframes_{prefix}" in k]
        if not candidate_zips:
            return None

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
                    return zh.read(target_name)
                except KeyError:
                    continue
        return None

    def load_frame(self, video_id: str, frame_idx: int) -> Image.Image:
        """Alias cho get_image."""
        return self.get_image(video_id, frame_idx)

    def get_keyframe_image(self, video_id: str, frame_idx: int) -> Image.Image:
        """Alias cho get_image."""
        return self.get_image(video_id, frame_idx)

    def get_surrounding_keyframes(self, video_id: str, frame_idx: int, count: int = 5) -> list:
        """
        Lấy danh sách các keyframes lân cận (trước và sau) để soi dải phim ngữ cảnh.
        """
        df_v = self.df_frames[self.df_frames["video_id"] == video_id].sort_values("frame_idx")
        if df_v.empty:
            return []

        frame_list = df_v["frame_idx"].tolist()
        import bisect
        pos = bisect.bisect_left(frame_list, int(frame_idx))
        pos = min(pos, len(frame_list) - 1)

        half = count // 2
        start_idx = max(0, pos - half)
        end_idx = min(len(frame_list), start_idx + count)
        if end_idx - start_idx < count:
            start_idx = max(0, end_idx - count)

        sub_frames = frame_list[start_idx:end_idx]
        results = []
        for f in sub_frames:
            img = self.get_image(video_id, f)
            results.append({
                "frame_idx": f,
                "is_current": (f == frame_list[pos]),
                "image": img
            })
        return results

    def get_all_video_keyframes(self, video_id: str) -> list[int]:
        """Trả về danh sách toàn bộ các frame_idx đã trích xuất của video đó theo thứ tự tăng dần."""
        df_v = self.df_frames[self.df_frames["video_id"] == video_id].sort_values("frame_idx")
        if df_v.empty:
            return []
        return [int(x) for x in df_v["frame_idx"].tolist()]

    def get_pts_time(self, video_id: str, frame_idx: int) -> float:
        """Trả về mốc thời gian giây (pts_time) của frame đó."""
        df_v = self.df_frames[(self.df_frames["video_id"] == video_id) & (self.df_frames["frame_idx"] == frame_idx)]
        if not df_v.empty and "pts_time" in df_v.columns:
            return float(df_v.iloc[0]["pts_time"])
        # Nếu frame nằm giữa 2 keyframes, tính chuẩn theo FPS thực của video
        df_vid = self.df_frames[self.df_frames["video_id"] == video_id]
        if not df_vid.empty and "fps" in df_vid.columns:
            fps = float(df_vid.iloc[0]["fps"])
            if fps > 0:
                return float(frame_idx) / fps
        return float(frame_idx) / 25.0

    def get_exact_frame_from_time(self, video_id: str, time_sec: float) -> int:
        """Tính toán chính xác frame_idx từ mốc thời gian giây dựa trên FPS thực của video."""
        df_v = self.df_frames[self.df_frames["video_id"] == video_id]
        if not df_v.empty and "fps" in df_v.columns:
            fps = float(df_v.iloc[0]["fps"])
            if fps > 0:
                return int(round(time_sec * fps))
        return int(round(time_sec * 25.0))

    def get_nearest_frame_from_time(self, video_id: str, time_sec: float) -> int:
        """Tra cứu chính xác frame_idx từ mốc thời gian giây dựa trên frames.parquet (chuẩn 100% BTC)."""
        df_v = self.df_frames[self.df_frames["video_id"] == video_id]
        if df_v.empty:
            return int(time_sec * 25.0)
        diffs = (df_v["pts_time"] - time_sec).abs()
        nearest_row = df_v.loc[diffs.idxmin()]
        return int(nearest_row["frame_idx"])

    def get_dense_video_frame(self, video_id: str, frame_idx: int) -> Image.Image:
        """
        Trích xuất frame video chính xác từng frame từ file MP4 gốc qua OpenCV.
        """
        try:
            import cv2
            from src.retrieval.video_player_manager import VideoPlayerManager
            zm = VideoPlayerManager()
            v_path = zm.get_video_path(video_id)
            if not v_path or not v_path.exists():
                return None
            cap = cv2.VideoCapture(str(v_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            cap.release()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(frame_rgb)
        except Exception:
            pass
        return None

if __name__ == "__main__":
    loader = KeyframeZipLoader()
    print("Testing loader on L27_V002 frame 920...")
    img = loader.get_image("L27_V002", 920)
    if img:
        print(f"✅ Loaded Image thành công! Kích thước: {img.size}")
    else:
        print("❌ Không tìm thấy ảnh!")
