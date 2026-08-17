import os
import sys
import argparse
import io
import time
import zipfile
import tempfile
import re
import json
import requests
from pathlib import Path
import pandas as pd
import numpy as np

# Polyfill for PIL._util in Python 3.12 / Pillow 11 compatibility with torchvision
try:
    import PIL._util
    if not hasattr(PIL._util, "is_directory"):
        PIL._util.is_directory = lambda f: isinstance(f, (str, bytes, os.PathLike)) and os.path.isdir(f)
    if not hasattr(PIL._util, "is_path"):
        PIL._util.is_path = lambda f: isinstance(f, (str, bytes, os.PathLike))
except Exception:
    pass

import torch
from PIL import Image
from tqdm import tqdm

# Force UTF-8 stdout on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import settings

def download_file(url_or_id, target_path, pkg_idx, total_pkgs, max_retries=5):
    """Tải file zip keyframes kèm thanh tiến trình tốc độ cao và TỰ ĐỘNG THỬ LẠI KHI ĐỨT MẠNG"""
    filename = Path(url_or_id).name if "http" in url_or_id else f"Package_{pkg_idx}.zip"
    
    if "ledo.io.vn" in url_or_id or (url_or_id.startswith("http") and "drive.google.com" not in url_or_id):
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(url_or_id, stream=True, timeout=(15, 180))
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024 * 1024
                
                with open(target_path, 'wb') as file, tqdm(
                    desc=f"📥 [Tải Keyframes {pkg_idx}/{total_pkgs}] {filename} (Lần {attempt})",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    leave=False
                ) as bar:
                    for data in response.iter_content(block_size):
                        if data:
                            size = file.write(data)
                            bar.update(size)
                            
                # Kiểm tra tính toàn vẹn sơ bộ của file zip
                if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
                    try:
                        with zipfile.ZipFile(target_path, 'r') as test_z:
                            _ = test_z.namelist()
                        return # Hoàn tất tải file hợp lệ
                    except Exception:
                        print(f"\n⚠️ File keyframes zip {filename} tải về chưa hoàn tất. Đang thử lại lần {attempt + 1}/{max_retries}...")
                        if os.path.exists(target_path):
                            os.remove(target_path)
            except Exception as e:
                print(f"\n⚠️ Mạng gián đoạn khi tải {filename} ({e}). Đang thử lại lần {attempt + 1}/{max_retries} sau 3 giây...")
                time.sleep(3)
                if os.path.exists(target_path):
                    try:
                        os.remove(target_path)
                    except Exception:
                        pass
        raise RuntimeError(f"Không thể tải file {filename} sau {max_retries} lần thử lại!")
    else:
        try:
            import gdown
        except ImportError:
            print("[ERROR] Chưa cài gdown! pip install gdown")
            sys.exit(1)
        if "drive.google.com" in url_or_id:
            gdown.download(url=url_or_id, output=str(target_path), quiet=False, fuzzy=True)
        else:
            gdown.download(id=url_or_id, output=str(target_path), quiet=False)

class VisualFeatureExtractor:
    """
    Trích xuất đặc trưng hình ảnh SOTA (SigLIP / CLIP) trên GPU A100:
    - Xử lý Batching siêu tốc với PyTorch
    - Tự động chuẩn hóa L2 (L2 Normalize)
    - Xuất float16 tiết kiệm bộ nhớ
    """
    def __init__(self, model_name="google/siglip2-so400m-patch14-384", device="cuda"):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model_name = model_name
        
        print(f"=== Khởi tạo Mô hình Thị giác SOTA SigLIP 2: {model_name} trên {self.device} ===")
        try:
            from transformers import AutoImageProcessor, AutoModel
            self.processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device).eval()
        except Exception as e_proc:
            print(f"[INFO] AutoImageProcessor info: {e_proc}, thử AutoProcessor...")
            from transformers import AutoProcessor, AutoModel
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device).eval()
        
        # Xác định số chiều vector (embedding dim)
        if hasattr(self.model.config, "vision_config") and hasattr(self.model.config.vision_config, "hidden_size"):
            self.dim = self.model.config.vision_config.hidden_size
        elif hasattr(self.model.config, "projection_dim"):
            self.dim = self.model.config.projection_dim
        elif hasattr(self.model.config, "text_config") and hasattr(self.model.config.text_config, "hidden_size"):
            self.dim = self.model.config.text_config.hidden_size
        else:
            self.dim = 1152 if "so400m" in model_name else 768
            
        print(f"✅ Mô hình đã sẵn sàng! Embedding dimension: {self.dim}")

    def extract_batch(self, pil_images_list):
        if not pil_images_list:
            return np.empty((0, self.dim), dtype=np.float16)
            
        inputs = self.processor(images=pil_images_list, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(self.device == "cuda")):
                if hasattr(self.model, "get_image_features"):
                    feats = self.model.get_image_features(**inputs)
                elif hasattr(self.model, "vision_model"):
                    outputs = self.model.vision_model(**inputs)
                    feats = outputs.pooler_output if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None else outputs.last_hidden_state[:, 0]
                else:
                    outputs = self.model(**inputs)
                    feats = outputs.pooler_output if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None else outputs.last_hidden_state[:, 0]
                    
                # L2 Normalization
                feats = feats / feats.norm(dim=-1, keepdim=True)
            
        return feats.cpu().to(torch.float16).numpy()

def process_zip_archive(zpath, extractor, frames_lookup, memmap_array, checkpoint_file, pkg_idx, total_pkgs, batch_size=64):
    """
    Xử lý 1 file ZIP:
    - Mở zip đọc ảnh trực tiếp trong RAM (không giải nén ra đĩa)
    - Trích xuất theo từng lô batch_size
    - Ghi trực tiếp vào ma trận memmap
    - Cập nhật checkpoint danh sách video đã làm
    """
    processed_videos = set()
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                processed_videos = set(json.load(f))
        except Exception:
            pass

    zip_name = Path(zpath).name
    
    with zipfile.ZipFile(zpath, "r") as zf:
        img_names = [n for n in zf.namelist() if n.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        # Gom nhóm ảnh theo từng video
        video_groups = {}
        for img_name in img_names:
            match_vid = re.search(r'(L\d+_V\d+)', img_name)
            if match_vid:
                vid = match_vid.group(1)
                video_groups.setdefault(vid, []).append(img_name)

        with tqdm(img_names, desc=f"🚀 [Trích xuất {pkg_idx}/{total_pkgs}] {zip_name}", unit="frame", leave=False) as pbar:
            for video_id, v_img_names in video_groups.items():
                if video_id in processed_videos:
                    pbar.update(len(v_img_names))
                    pbar.set_postfix({"video": video_id, "status": "đã_xong (skip)"})
                    continue

                for i in range(0, len(v_img_names), batch_size):
                    batch_names = v_img_names[i:i + batch_size]
                    batch_pil_images = []
                    batch_global_ids = []

                    for img_name in batch_names:
                        filename = Path(img_name).name
                        match_idx = re.search(r'(\d+)', Path(filename).stem)
                        if not match_idx:
                            continue
                        kf_idx = int(match_idx.group(1))

                        if (video_id, kf_idx) not in frames_lookup:
                            continue

                        global_id = frames_lookup[(video_id, kf_idx)]
                        
                        try:
                            img_bytes = zf.read(img_name)
                            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                            batch_pil_images.append(pil_img)
                            batch_global_ids.append(global_id)
                        except Exception:
                            continue

                    if batch_pil_images:
                        feats = extractor.extract_batch(batch_pil_images)
                        memmap_array[batch_global_ids, :] = feats

                    pbar.update(len(batch_names))

                # Ghi checkpoint video
                processed_videos.add(video_id)
                with open(checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump(list(processed_videos), f)
                    
                pbar.set_postfix({"video": video_id, "status": "xong"})

    memmap_array.flush()

def main():
    parser = argparse.ArgumentParser(description="Extract SOTA Visual Features (SigLIP/CLIP) -> Streaming Zip & Auto-Delete -> Memmap")
    parser.add_argument("--url", type=str, default=None,
                        help="Link tải 1 file ZIP cụ thể")
    parser.add_argument("--urls_file", type=str, default="config/drive_keyframes_urls.txt",
                        help="File danh sách link ZIP keyframes")
    parser.add_argument("--start_index", type=int, default=1,
                        help="Bắt đầu chạy từ gói số N (1-indexed, ví dụ: --start_index 5)")
    parser.add_argument("--start_from", type=str, default=None,
                        help="Bắt đầu chạy từ file cụ thể (ví dụ: --start_from Keyframes_L25.zip)")
    parser.add_argument("--keyframes_dir", type=str, default="raw/batch_1/Keyframes",
                        help="Thư mục keyframes ZIP local")
    parser.add_argument("--model_name", type=str, default="google/siglip2-so400m-patch14-384",
                        help="Mô hình thị giác HuggingFace SOTA (Mặc định: SigLIP 2)")
    parser.add_argument("--output_path", type=str, default="data/batch_1/processed/siglip_features.npy",
                        help="Nơi lưu ma trận vector hoàn chỉnh")
    parser.add_argument("--frames_path", type=str, default=str(settings.directories.processed / "frames.parquet"),
                        help="Đường dẫn tới file frames.parquet")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Kích thước batch trích xuất trên GPU (Mặc định: 32)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Thiết bị chạy (cuda)")
    args = parser.parse_args()

    # 1. Nạp mapping từ frames.parquet
    frames_path = Path(args.frames_path)
    if not frames_path.exists():
        print(f"[ERROR] Không tìm thấy {frames_path}. Hãy chắc chắn file frames.parquet tồn tại!")
        sys.exit(1)
        
    frames_df = pd.read_parquet(frames_path)
    total_frames = len(frames_df)
    frames_lookup = dict(zip(zip(frames_df["video_id"], frames_df["keyframe_index"]), frames_df["global_id"]))
    print(f"✅ Đã nạp mapping {total_frames:,} keyframes từ: {frames_path}")

    # 2. Khởi tạo mô hình
    extractor = VisualFeatureExtractor(model_name=args.model_name, device=args.device)

    # 3. Khởi tạo file bộ nhớ đệm Memmap
    out_file = Path(args.output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_file = out_file.parent / f"{out_file.stem}_checkpoint.json"

    # Mở hoặc tạo mới file memmap
    mode = "r+" if out_file.exists() else "w+"
    memmap_array = np.memmap(out_file, dtype='float16', mode=mode, shape=(total_frames, extractor.dim))
    print(f"📦 File vector đích: {out_file} [Shape: ({total_frames}, {extractor.dim}), dtype=float16]")

    # 4. Nạp danh sách nguồn
    all_targets = []
    if args.url:
        all_targets.append(args.url)
    elif args.urls_file and Path(args.urls_file).exists():
        with open(args.urls_file, "r", encoding="utf-8") as f:
            all_targets = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        print(f"📋 Đã nạp {len(all_targets)} link từ {args.urls_file}")
    else:
        kf_dir = Path(args.keyframes_dir)
        if kf_dir.exists():
            all_targets = sorted(list(kf_dir.glob("*.zip")) + list(kf_dir.glob("*/*.zip")))
            print(f"📋 Đã tìm thấy {len(all_targets)} file ZIP local trong {kf_dir}")

    if not all_targets:
        print("[ERROR] Không tìm thấy link tải hoặc file ZIP nào để xử lý!")
        sys.exit(1)

    # Xử lý lọc start_index hoặc start_from
    if args.start_from:
        found_idx = None
        for idx, t in enumerate(all_targets):
            if args.start_from in str(t):
                found_idx = idx
                break
        if found_idx is not None:
            all_targets = all_targets[found_idx:]
            print(f"⏩ Bắt đầu chạy từ file: {args.start_from} (còn lại {len(all_targets)} gói)")
        else:
            print(f"[WARNING] Không tìm thấy '{args.start_from}' trong danh sách, chạy từ đầu.")
    elif args.start_index > 1:
        skip_n = args.start_index - 1
        all_targets = all_targets[skip_n:]
        print(f"⏩ Bắt đầu chạy từ gói số {args.start_index} (còn lại {len(all_targets)} gói)")

    total_pkgs = len(all_targets)
    start_time = time.time()

    with tqdm(enumerate(all_targets, start=1), total=total_pkgs, desc="📦 [TỔNG] Keyframe Packages", unit="gói") as main_bar:
        for pkg_idx, target in main_bar:
            pkg_name = Path(target).name if "http" in target else str(target)
            main_bar.set_postfix({"gói_hiện_tại": pkg_name})

            is_remote = isinstance(target, str) and ("http" in target or "drive.google.com" in target or (len(target) > 20 and not Path(target).exists()))

            if is_remote:
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip:
                    temp_zip_path = tmp_zip.name

                try:
                    # Tải về
                    download_file(target, temp_zip_path, pkg_idx, total_pkgs)
                    # Trích xuất và ghi thẳng vào memmap
                    process_zip_archive(temp_zip_path, extractor, frames_lookup, memmap_array, checkpoint_file, pkg_idx, total_pkgs, batch_size=args.batch_size)
                finally:
                    # XÓA NGAY LẬP TỨC FILE ZIP TẠM ĐỂ TIẾT KIỆM BỘ NHỚ ĐĨA
                    if os.path.exists(temp_zip_path):
                        os.remove(temp_zip_path)
            else:
                process_zip_archive(target, extractor, frames_lookup, memmap_array, checkpoint_file, pkg_idx, total_pkgs, batch_size=args.batch_size)

    memmap_array.flush()
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print(f"🎉 HOÀN TẤT TRÍCH XUẤT ĐẶC TRƯNG THỊ GIÁC SOTA: {out_file}")
    print(f"⏱️ Tổng thời gian: {elapsed:.0f}s ({elapsed/3600:.2f} giờ)")
    print("="*70)

if __name__ == "__main__":
    main()
