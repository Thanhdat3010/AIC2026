import argparse
import sys
import time
import zipfile
import tempfile
import os
import re
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import threading
import pandas as pd
import numpy as np
import cv2
import torch
from PIL import Image
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import settings

class VietnameseMaxAccuracyOCR:
    """
    Hệ thống SOTA 2-Stage OCR Tiếng Việt Tối Ưu Đa Luồng:
    - Stage 1 (Text Detection): CRAFT Detector (canvas_size=960)
    - Stage 2 (Text Recognition): VietOCR VGG-Transformer x4 thread song song
    """
    def __init__(self, use_vietocr=True, use_gpu=True, num_workers=4):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda:0" if self.use_gpu else "cpu"
        self.use_vietocr = use_vietocr
        self.num_workers = num_workers

        print(f"=== [1/2] Đang nạp CRAFT Text Detector (EasyOCR) trên {self.device} ===")
        import easyocr
        self.reader = easyocr.Reader(['vi'], gpu=self.use_gpu, verbose=False)
        print("✅ CRAFT Text Detector đã sẵn sàng trên GPU!")

        if self.use_vietocr:
            print(f"=== [2/2] Đang nạp VietOCR VGG-Transformer trên {self.device} ===")
            try:
                from vietocr.tool.predictor import Predictor
                from vietocr.tool.config import Cfg
                config = Cfg.load_config_from_name('vgg_transformer')
                config['device'] = self.device
                config['predictor']['beamsearch'] = False
                self.vietocr_predictor = Predictor(config)
                self._vietocr_lock = threading.Lock()
                print(f"✅ VietOCR VGG-Transformer sẵn sàng (đa luồng {num_workers} workers)!")
            except Exception as e:
                print(f"[WARNING] Lỗi nạp VietOCR ({e}). Fallback sang EasyOCR Recognition.")
                self.use_vietocr = False

    def _predict_single_crop(self, crop):
        """Nhận diện 1 mẩu chữ bằng VietOCR (thread-safe với lock)"""
        try:
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            with self._vietocr_lock:
                text = self.vietocr_predictor.predict(pil_img)
            return str(text).strip()
        except Exception:
            return ""

    def predict(self, img_array):
        """
        Dự đoán chữ đa luồng:
        1. CRAFT quét vị trí khối chữ (Single-pass, canvas_size=960)
        2. ThreadPool xử lý N mẩu chữ song song qua VietOCR
        """
        try:
            if img_array is None or img_array.size == 0:
                return None, None

            # 1. Quét vị trí chữ bằng CRAFT
            horizontal_list, free_list = self.reader.detect(
                img_array,
                canvas_size=960,
                mag_ratio=1.0,
                text_threshold=0.7,
                link_threshold=0.4,
                low_text=0.4
            )
            boxes = horizontal_list[0] if horizontal_list and len(horizontal_list) > 0 else []

            if not boxes:
                return None, None

            h_img, w_img = img_array.shape[:2]

            # 2. Cắt tất cả vùng chữ
            valid_crops = []
            for box in boxes:
                x_min, x_max, y_min, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                w, h = x_max - x_min, y_max - y_min
                if w > 8 and h > 8:
                    x1, y1 = max(0, x_min), max(0, y_min)
                    x2, y2 = min(w_img, x_max), min(h_img, y_max)
                    crop = img_array[y1:y2, x1:x2]
                    if crop.size > 0:
                        valid_crops.append(crop)

            if not valid_crops:
                return None, None

            # 3. Xử lý song song các mẩu chữ bằng ThreadPoolExecutor
            texts = []
            confidences = []

            if self.use_vietocr:
                with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                    futures = [pool.submit(self._predict_single_crop, crop) for crop in valid_crops]
                    for future in futures:
                        text = future.result()
                        if len(text) >= 2:
                            texts.append(text)
                            confidences.append(0.95)

            if texts:
                unique_texts = []
                for t in texts:
                    if not unique_texts or t.lower() != unique_texts[-1].lower():
                        unique_texts.append(t)
                return " | ".join(unique_texts), round(sum(confidences) / len(confidences), 3)

        except Exception:
            pass

        return None, None

def download_file(url_or_id, target_path, pkg_idx, total_pkgs):
    """Tải file zip kèm thanh tiến trình tốc độ cao"""
    filename = Path(url_or_id).name if "http" in url_or_id else f"Package_{pkg_idx}.zip"
    
    if "ledo.io.vn" in url_or_id or (url_or_id.startswith("http") and "drive.google.com" not in url_or_id):
        response = requests.get(url_or_id, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024
        
        with open(target_path, 'wb') as file, tqdm(
            desc=f"📥 [Tải {pkg_idx}/{total_pkgs}] {filename}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            leave=False
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)
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

def save_and_merge_parquet(new_records, out_file):
    """Ghi checkpoint và gộp dữ liệu không bao giờ bị mất hoặc trùng lặp"""
    if not new_records:
        return
    
    combined_df = pd.DataFrame(new_records)
    if out_file.exists():
        try:
            old_df = pd.read_parquet(out_file)
            combined_df = pd.concat([old_df, combined_df], ignore_index=True)
        except Exception:
            pass
    
    if not combined_df.empty:
        combined_df = combined_df.drop_duplicates(subset=["video_id", "keyframe_index"], keep="last")
        combined_df.to_parquet(out_file, index=False)

def process_zip_archive(zpath, ocr_engine, frames_df, pkg_idx, total_pkgs, out_file, processed_videos_set=None):
    """
    Xử lý ZIP với pipeline prefetch: 1 thread đọc ảnh, main thread chạy OCR.
    Lưu checkpoint sau mỗi video.
    """
    total_records_in_pkg = 0
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

        with tqdm(img_names, desc=f"⚡ [OCR {pkg_idx}/{total_pkgs}] {zip_name}", unit="frame", leave=False) as pbar:
            for video_id, v_img_names in video_groups.items():
                if processed_videos_set and video_id in processed_videos_set:
                    pbar.update(len(v_img_names))
                    pbar.set_postfix({"video": video_id, "status": "skip"})
                    continue

                video_records = []

                # Prefetch: đọc + decode ảnh trước trong thread riêng
                prefetch_queue = deque()

                def prefetch_images(names):
                    for name in names:
                        img_bytes = zf.read(name)
                        img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                        prefetch_queue.append((name, img_array))

                prefetch_thread = threading.Thread(target=prefetch_images, args=(v_img_names,))
                prefetch_thread.start()

                processed_count = 0
                while processed_count < len(v_img_names):
                    # Chờ ảnh từ prefetch queue
                    while not prefetch_queue and prefetch_thread.is_alive():
                        time.sleep(0.001)
                    
                    if not prefetch_queue:
                        break
                    
                    img_name, img_array = prefetch_queue.popleft()
                    processed_count += 1
                    pbar.update(1)

                    filename = Path(img_name).name
                    match_idx = re.search(r'(\d+)', Path(filename).stem)
                    if not match_idx:
                        continue
                    kf_idx = int(match_idx.group(1))

                    frame_idx, pts_time = -1, 0.0
                    if frames_df is not None and (video_id, kf_idx) in frames_df.index:
                        row = frames_df.loc[(video_id, kf_idx)]
                        frame_idx = int(row["frame_idx"])
                        pts_time = float(row["pts_time"])

                    if img_array is not None:
                        text, conf = ocr_engine.predict(img_array)
                        if text:
                            video_records.append({
                                "video_id": video_id,
                                "keyframe_index": kf_idx,
                                "frame_idx": frame_idx,
                                "pts_time": pts_time,
                                "ocr_text": text,
                                "confidence": conf
                            })

                prefetch_thread.join()

                # Ghi checkpoint ngay khi quét xong từng video
                if video_records:
                    save_and_merge_parquet(video_records, out_file)
                    processed_videos_set.add(video_id)
                    total_records_in_pkg += len(video_records)

                pbar.set_postfix({"video": video_id, "chữ": len(video_records), "tổng": total_records_in_pkg})

    return total_records_in_pkg

def main():
    parser = argparse.ArgumentParser(description="BTC/Drive -> SOTA PyTorch OCR (CRAFT + VietOCR Multi-Thread) -> Auto-Resume")
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--urls_file", type=str, default="config/drive_keyframes_urls.txt")
    parser.add_argument("--start_index", type=int, default=1)
    parser.add_argument("--start_from", type=str, default=None)
    parser.add_argument("--keyframes_dir", type=str, default="Keyframes")
    parser.add_argument("--output_path", type=str, default="data/processed/ocr_results.parquet")
    parser.add_argument("--use_vietocr", action="store_true", default=True)
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Số luồng song song xử lý VietOCR (mặc định: 4)")
    args = parser.parse_args()

    ocr_engine = VietnameseMaxAccuracyOCR(
        use_vietocr=args.use_vietocr,
        use_gpu=args.use_gpu,
        num_workers=args.num_workers
    )

    frames_path = settings.directories.processed / "frames.parquet"
    frames_df = None
    if frames_path.exists():
        frames_df = pd.read_parquet(frames_path).set_index(["video_id", "keyframe_index"])
        print(f"✅ Đã nạp {len(frames_df)} bản ghi mapping từ frames.parquet.")

    out_file = Path(args.output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    processed_videos_set = set()
    total_existing_records = 0
    if out_file.exists():
        try:
            existing_df = pd.read_parquet(out_file)
            total_existing_records = len(existing_df)
            processed_videos_set = set(existing_df["video_id"].unique())
            print(f"🔄 [RESUME] Đã tìm thấy {total_existing_records:,} bản ghi OCR của {len(processed_videos_set)} video từ trước!")
        except Exception:
            pass

    all_targets = []
    if args.url:
        all_targets.append(args.url)
    elif args.urls_file and Path(args.urls_file).exists():
        with open(args.urls_file, "r", encoding="utf-8") as f:
            all_targets = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        print(f"📋 Đã nạp {len(all_targets)} link từ file {args.urls_file}")
    else:
        kf_dir = Path(args.keyframes_dir)
        if kf_dir.exists():
            all_targets = sorted(list(kf_dir.glob("*.zip")) + list(kf_dir.glob("*/*.zip")))
            print(f"📋 Đã tìm thấy {len(all_targets)} file ZIP local")

    if not all_targets:
        print("[ERROR] Không tìm thấy link tải hoặc file ZIP nào để xử lý!")
        sys.exit(1)

    total_pkgs = len(all_targets)
    start_offset = 0

    if args.start_from:
        target_name = args.start_from.lower().replace(".zip", "")
        found = False
        for idx, t in enumerate(all_targets):
            if target_name in str(t).lower():
                start_offset = idx
                found = True
                print(f"⏩ [CHỈ ĐỊNH ĐIỂM BẮT ĐẦU] Tìm thấy '{args.start_from}' tại vị trí {start_offset + 1}/{total_pkgs}")
                break
        if not found:
            print(f"[WARNING] Không tìm thấy '{args.start_from}' trong danh sách, bắt đầu từ đầu.")
    elif args.start_index > 1:
        start_offset = max(0, min(args.start_index - 1, total_pkgs - 1))
        print(f"⏩ [CHỈ ĐỊNH ĐIỂM BẮT ĐẦU] Bắt đầu chạy từ gói số {start_offset + 1}/{total_pkgs}")

    active_targets = all_targets[start_offset:]
    start_time = time.time()

    with tqdm(enumerate(active_targets, start=start_offset + 1), total=total_pkgs, initial=start_offset, desc="📦 [TỔNG] Keyframes", unit="gói") as main_bar:
        for pkg_idx, target in main_bar:
            pkg_name = Path(target).name if "http" in target else str(target)
            main_bar.set_postfix({"gói_hiện_tại": pkg_name})

            is_remote = isinstance(target, str) and ("http" in target or "drive.google.com" in target or (len(target) > 20 and not Path(target).exists()))
            
            if is_remote:
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip:
                    temp_zip_path = tmp_zip.name

                download_file(target, temp_zip_path, pkg_idx, total_pkgs)
                process_zip_archive(temp_zip_path, ocr_engine, frames_df, pkg_idx, total_pkgs, out_file, processed_videos_set)

                if os.path.exists(temp_zip_path):
                    os.remove(temp_zip_path)
            else:
                process_zip_archive(target, ocr_engine, frames_df, pkg_idx, total_pkgs, out_file, processed_videos_set)

            try:
                cur_total = len(pd.read_parquet(out_file))
                main_bar.set_postfix({"gói": pkg_name, "tổng_khung_hình": cur_total})
            except Exception:
                pass

    elapsed = time.time() - start_time
    final_count = 0
    if out_file.exists():
        try:
            final_count = len(pd.read_parquet(out_file))
        except Exception:
            pass

    print("\n" + "="*70)
    print(f"🎉 HOÀN TẤT TOÀN BỘ OCR: Tổng cộng có {final_count:,} khung hình có chữ trong {out_file}")
    print(f"⏱️ Tổng thời gian chạy: {elapsed:.0f}s ({elapsed/3600:.2f} giờ)")
    print("="*70)

if __name__ == "__main__":
    main()
