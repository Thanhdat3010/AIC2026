import os
import sys
import json
import zipfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_ZIP_DIR = BASE_DIR / "raw" / "batch_1" / "Videos"
DEFAULT_CACHE_DIR = BASE_DIR / "scratch" / "video_cache"
ZIP_INDEX_FILE = BASE_DIR / "data" / "batch_1" / "processed" / "video_zip_index.json"

class VideoPlayerManager:
    """
    Quản lý trích xuất On-Demand các file video MP4 từ các gói zip (Videos_*.zip)
    vào bộ đệm scratch/video_cache/ để phát video trực tiếp trên Streamlit.
    """
    def __init__(self, zip_dir: Path = DEFAULT_ZIP_DIR, cache_dir: Path = DEFAULT_CACHE_DIR):
        self.zip_dir = Path(zip_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.zip_index: Dict[str, Tuple[str, str]] = {}
        self._build_or_load_index()

    def _build_or_load_index(self):
        # 1. Thử nạp từ cache index nếu có
        if ZIP_INDEX_FILE.exists():
            try:
                with open(ZIP_INDEX_FILE, "r", encoding="utf-8") as f:
                    self.zip_index = json.load(f)
                if self.zip_index:
                    return
            except Exception:
                pass

        # 2. Quét nhanh tất cả các file zip trong zip_dir
        if self.zip_dir.exists():
            zip_files = sorted(list(self.zip_dir.glob("*.zip")))
            for zf in zip_files:
                try:
                    with zipfile.ZipFile(zf, "r") as z:
                        for name in z.namelist():
                            if name.lower().endswith(".mp4"):
                                vid = Path(name).stem
                                # Lưu (đường dẫn zip, tên file bên trong)
                                self.zip_index[vid] = (str(zf), name)
                except Exception as e:
                    print(f"⚠️ Lỗi quét zip {zf.name}: {e}", flush=True)

            if self.zip_index:
                try:
                    ZIP_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
                    with open(ZIP_INDEX_FILE, "w", encoding="utf-8") as f:
                        json.dump(self.zip_index, f, ensure_ascii=False)
                except Exception:
                    pass

    def get_video_path(self, video_id: str) -> Optional[Path]:
        """
        Trả về đường dẫn file MP4 sẵn sàng phát:
        - Nếu đã có trong cache -> trả về ngay lập tức (0ms).
        - Nếu chưa có -> trích xuất on-demand từ file zip trong 0.2 - 0.4 giây.
        """
        clean_vid = video_id.strip()
        cached_file = self.cache_dir / f"{clean_vid}.mp4"
        if cached_file.exists() and cached_file.stat().st_size > 1024:
            return cached_file

        # Kiểm tra trong index zip
        if clean_vid not in self.zip_index:
            # Thử tìm không phân biệt chữ hoa thường
            for k, v in self.zip_index.items():
                if k.lower() == clean_vid.lower():
                    clean_vid = k
                    break

        if clean_vid not in self.zip_index:
            return None

        zip_path_str, internal_name = self.zip_index[clean_vid]
        zip_path = Path(zip_path_str)
        if not zip_path.exists():
            return None

        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                with z.open(internal_name) as source, open(cached_file, "wb") as target:
                    shutil.copyfileobj(source, target)
            return cached_file
        except Exception as e:
            print(f"⚠️ Lỗi giải nén on-demand video {clean_vid}: {e}", flush=True)
            if cached_file.exists():
                cached_file.unlink(missing_ok=True)
            return None

    def get_optimized_clip(self, video_id: str, target_sec: float, clip_window: float = 60.0) -> Tuple[Optional[Path], float, float]:
        """
        Cắt siêu tốc đoạn clip ngắn (60 giây) bao quanh mốc target_sec với chuẩn +faststart
        để trình duyệt phát tức thì không bị giật lag/buffering dù video gốc dài hàng tiếng (200MB+).
        Trả về: (đường_dẫn_clip, thời_điểm_bắt_đầu_clip, độ_dài_clip)
        """
        clean_vid = video_id.strip()
        full_mp4 = self.get_video_path(clean_vid)
        if not full_mp4 or not full_mp4.exists():
            return None, 0.0, 0.0

        clip_start = max(0.0, float(target_sec) - 10.0)
        clip_name = f"{clean_vid}_clip_{int(clip_start)}_{int(clip_window)}.mp4"
        clip_path = self.cache_dir / clip_name

        if clip_path.exists() and clip_path.stat().st_size > 1024:
            return clip_path, clip_start, clip_window

        try:
            import imageio_ffmpeg
            import subprocess
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_exe, "-y",
                "-ss", str(clip_start),
                "-i", str(full_mp4),
                "-t", str(clip_window),
                "-c", "copy",
                "-movflags", "+faststart",
                str(clip_path)
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            if clip_path.exists() and clip_path.stat().st_size > 1024:
                return clip_path, clip_start, clip_window
        except Exception as e:
            print(f"⚠️ Không thể tạo micro-clip qua ffmpeg: {e}", flush=True)

        # Fallback về file gốc nếu không cắt được
        return full_mp4, 0.0, 0.0
