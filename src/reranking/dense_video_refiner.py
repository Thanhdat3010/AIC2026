import os
import cv2
import json
import zipfile
import shutil
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from typing import List, Dict, Tuple, Optional, Any

BASE_DIR = Path(__file__).resolve().parent.parent.parent
VIDEO_ZIP_DIR = BASE_DIR / "raw" / "batch_1" / "Videos"
VIDEO_CACHE_DIR = BASE_DIR / "scratch" / "video_cache"
ZIP_MAP_CACHE = BASE_DIR / "data" / "batch_1" / "processed" / "video_zip_map.json"

class VideoZipManager:
    """Quản lý và trích xuất on-demand các file MP4 từ các gói zip Videos_*.zip."""
    def __init__(self, video_zip_dir: Path = VIDEO_ZIP_DIR, cache_dir: Path = VIDEO_CACHE_DIR, max_cached_files: int = 5):
        self.video_zip_dir = Path(video_zip_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cached_files = max_cached_files
        self.video_map: Dict[str, Tuple[str, str]] = {}
        self._load_or_build_index()

    def _load_or_build_index(self):
        if ZIP_MAP_CACHE.exists():
            try:
                with open(ZIP_MAP_CACHE, "r", encoding="utf-8") as f:
                    self.video_map = json.load(f)
                return
            except Exception:
                pass

        # Build index from all video zip files
        if self.video_zip_dir.exists():
            zip_files = sorted(list(self.video_zip_dir.glob("*.zip")))
            for zf in zip_files:
                try:
                    with zipfile.ZipFile(zf, "r") as z:
                        for name in z.namelist():
                            if name.endswith(".mp4"):
                                vid = Path(name).stem
                                self.video_map[vid] = (zf.name, name)
                except Exception as e:
                    print(f"⚠️ Lỗi đọc zip {zf.name}: {e}")

            ZIP_MAP_CACHE.parent.mkdir(parents=True, exist_ok=True)
            with open(ZIP_MAP_CACHE, "w", encoding="utf-8") as f:
                json.dump(self.video_map, f, ensure_ascii=False, indent=2)

    def has_video(self, video_id: str) -> bool:
        return video_id in self.video_map

    def get_video_path(self, video_id: str) -> Optional[Path]:
        """Trích xuất file mp4 duy nhất vào thư mục tạm nếu chưa có."""
        if not self.has_video(video_id):
            return None

        cached_mp4 = self.cache_dir / f"{video_id}.mp4"
        if cached_mp4.exists() and cached_mp4.stat().st_size > 0:
            return cached_mp4

        # Dọn dẹp cache nếu vượt quá giới hạn
        self._cleanup_cache()

        zip_name, internal_path = self.video_map[video_id]
        zip_path = self.video_zip_dir / zip_name
        if not zip_path.exists():
            return None

        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                with z.open(internal_path) as src, open(cached_mp4, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            return cached_mp4
        except Exception as e:
            print(f"❌ Lỗi trích xuất video {video_id} từ {zip_name}: {e}")
            if cached_mp4.exists():
                cached_mp4.unlink(missing_ok=True)
            return None

    def _cleanup_cache(self):
        try:
            cached_files = sorted(list(self.cache_dir.glob("*.mp4")), key=lambda p: p.stat().st_mtime)
            while len(cached_files) >= self.max_cached_files:
                oldest = cached_files.pop(0)
                oldest.unlink(missing_ok=True)
        except Exception:
            pass


class DenseVideoRefiner:
    """
    Layer 3: Gated Dense Video Refinement (Kính lúp vi sai bằng OpenCV).
    Trích xuất ~50-100 frame dày đặc quanh candidate frame và chấm điểm bằng SigLIP-2.
    """
    def __init__(self, engine: str = "siglip2", device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.engine = engine
        self.zip_manager = VideoZipManager()
        self.model = None
        self.processor = None
        self.dim = 1152 if "siglip2" in engine else 768

        # Tối ưu hóa đa luồng CPU nếu chạy trên CPU
        if self.device == "cpu":
            try:
                num_cores = os.cpu_count() or 8
                torch.set_num_threads(min(num_cores, 8))
            except Exception:
                pass

    def _init_vision_model(self):
        """Khởi tạo mô hình Vision Encoder (Lazy loading khi thực sự cần)."""
        if self.model is not None:
            return

        model_name = "google/siglip2-so400m-patch14-384"
        print(f"[*] Khởi tạo Dense Vision Encoder [{model_name}] (Device: {self.device})...", flush=True)

        try:
            from transformers import AutoProcessor, AutoModel
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device).eval()
        except Exception as e:
            print(f"⚠️ Lỗi nạp SigLIP 2 Vision: {e}")

    def extract_dense_frames(
        self,
        video_path: Path,
        approx_frame_idx: int,
        window_seconds: float = 2.5,
        step: int = 4
    ) -> List[Tuple[int, Image.Image]]:
        """
        Dùng OpenCV nhảy cóc (Seek) đến khoảng thời gian [f - window, f + window]
        và đọc ra danh sách (frame_idx, PIL Image) đã được resize 384x384 siêu tốc.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 25.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        window_frames = int(window_seconds * fps)

        start_frame = max(0, approx_frame_idx - window_frames)
        end_frame = min(total_frames - 1, approx_frame_idx + window_frames)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame

        dense_frames = []
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if (current_frame - start_frame) % step == 0:
                # Resize ngay trong OpenCV C++ sang 384x384 để tăng tốc xử lý
                resized = cv2.resize(frame, (384, 384), interpolation=cv2.INTER_AREA)
                rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                dense_frames.append((current_frame, pil_img))

            current_frame += 1

        cap.release()
        return dense_frames

    def refine_candidate(
        self,
        video_id: str,
        approx_frame_idx: int,
        query_vec: np.ndarray,
        window_seconds: float = 2.0,
        step: int = 1
    ) -> Dict[str, Any]:
        """
        Tìm frame đỉnh cao nhất xung quanh approx_frame_idx.
        """
        if not self.zip_manager.has_video(video_id):
            return {
                "refined": False,
                "frame_idx": approx_frame_idx,
                "score": 0.0,
                "reason": "Video not found in zip"
            }

        mp4_path = self.zip_manager.get_video_path(video_id)
        if not mp4_path:
            return {
                "refined": False,
                "frame_idx": approx_frame_idx,
                "score": 0.0,
                "reason": "Failed to extract mp4"
            }

        dense_frames = self.extract_dense_frames(
            video_path=mp4_path,
            approx_frame_idx=approx_frame_idx,
            window_seconds=window_seconds,
            step=step
        )

        if not dense_frames:
            return {
                "refined": False,
                "frame_idx": approx_frame_idx,
                "score": 0.0,
                "reason": "No frames extracted"
            }

        self._init_vision_model()
        if self.model is None or self.processor is None:
            return {
                "refined": False,
                "frame_idx": approx_frame_idx,
                "score": 0.0,
                "reason": "Vision model unavailable"
            }

        # Trích xuất vector cho toàn bộ dense frames theo batch
        images_list = [img for _, img in dense_frames]
        frame_indices = [idx for idx, _ in dense_frames]

        inputs = self.processor(images=images_list, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(self.device == "cuda")):
                if hasattr(self.model, "get_image_features"):
                    out = self.model.get_image_features(**inputs)
                    if isinstance(out, torch.Tensor):
                        feats = out
                    elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                        feats = out.pooler_output
                    else:
                        feats = out[0]
                else:
                    out = self.model(**inputs)
                    if isinstance(out, torch.Tensor):
                        feats = out
                    elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                        feats = out.pooler_output
                    else:
                        feats = out.last_hidden_state[:, 0]

                feats = feats / feats.norm(dim=-1, keepdim=True)
                feats_np = feats.cpu().float().numpy()

        # Chuẩn hóa query_vec
        q_vec = query_vec.reshape(1, -1)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        # Tính Cosine Similarity với toàn bộ dense frames
        sims = (feats_np @ q_vec.T).flatten()

        best_idx_pos = int(np.argmax(sims))
        best_frame_idx = frame_indices[best_idx_pos]
        best_score = float(sims[best_idx_pos])

        return {
            "refined": True,
            "frame_idx": best_frame_idx,
            "score": best_score,
            "total_scanned_frames": len(dense_frames),
            "sim_distribution": {
                "min": float(np.min(sims)),
                "max": float(np.max(sims)),
                "mean": float(np.mean(sims))
            }
        }
