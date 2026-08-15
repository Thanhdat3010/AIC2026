import argparse
import sys
import time
import zipfile
import tempfile
import os
import re
import requests
from pathlib import Path

# ==============================================================================
# CẤU HÌNH TỐI ƯU HÓA VRAM GPU A100
# ==============================================================================
os.environ['FLAGS_allocator_strategy'] = 'auto_growth'
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.10'
os.environ['FLAGS_eager_delete_tensor_gb'] = '0.0'
os.environ['FLAGS_fast_eager_deletion_mode'] = 'True'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import pandas as pd
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import settings

class VietnameseMaxAccuracyOCR:
    """
    Hệ thống OCR 2-Stage Đạt Độ Chính Xác Tối Đa Cho Tiếng Việt:
    - Stage 1 (Text Detection): PaddleOCR DBNet
    - Stage 2 (Text Recognition): VietOCR VGG-Transformer (hoặc PaddleOCR Rec nếu chưa cài VietOCR)
    """
    def __init__(self, use_vietocr=True, use_gpu=True):
        self.use_vietocr = use_vietocr
        self.use_gpu = use_gpu

        print("=== [1/2] Đang nạp mô hình Text Detection (PaddleOCR DBNet) ===")
        from paddleocr import PaddleOCR
        try:
            self.detector = PaddleOCR(
                use_angle_cls=False,
                lang='vi',
                use_gpu=use_gpu,
                show_log=False
            )
            print("✅ PaddleOCR DBNet đã sẵn sàng trên GPU.")
        except Exception as e:
            print(f"[ERROR] Không thể khởi tạo PaddleOCR: {e}")
            raise e

        if self.use_vietocr:
            print("=== [2/2] Đang nạp mô hình Text Recognition (VietOCR VGG-Transformer) ===")
            try:
                from vietocr.tool.predictor import Predictor
                from vietocr.tool.config import Cfg
                config = Cfg.load_config_from_name('vgg_transformer')
                config['device'] = 'cuda:0' if use_gpu else 'cpu'
                config['predictor']['beamsearch'] = False  # Beamsearch False để nhanh và nhẹ
                self.vietocr_predictor = Predictor(config)
                print("✅ VietOCR VGG-Transformer đã sẵn sàng (MAX ACCURACY cho tiếng Việt).")
            except Exception as e:
                print(f"[WARNING] Chưa cài hoặc lỗi VietOCR ({e}). Fallback về PaddleOCR Recognition.")
                self.use_vietocr = False

    def predict(self, img_array):
        try:
            if img_array is None or img_array.size == 0:
                return None, None

            result = self.detector.ocr(img_array, cls=False)
            if not result or result[0] is None or len(result[0]) == 0:
                return None, None

            texts = []
            confidences = []

            for line in result[0]:
                if not line or len(line) < 2:
                    continue
                box = line[0]
                rec_res = line[1]
                paddle_text = str(rec_res[0]).strip() if isinstance(rec_res, (list, tuple)) else ""
                paddle_score = float(rec_res[1]) if isinstance(rec_res, (list, tuple)) and len(rec_res) > 1 else 0.0

                if self.use_vietocr:
                    try:
                        pts = np.array(box, dtype=np.int32)
                        rect = cv2.boundingRect(pts)
                        x, y, w, h = rect
                        if w > 8 and h > 8:
                            h_img, w_img = img_array.shape[:2]
                            x1, y1 = max(0, x), max(0, y)
                            x2, y2 = min(w_img, x + w), min(h_img, y + h)
                            crop = img_array[y1:y2, x1:x2]
                            if crop.size > 0:
                                pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                                vietocr_text, prob = self.vietocr_predictor.predict(pil_img, return_prob=True)
                                vietocr_text = vietocr_text.strip()
                                if len(vietocr_text) >= 2 and prob >= 0.4:
                                    texts.append(vietocr_text)
                                    confidences.append(prob)
                                    continue
                    except Exception:
                        pass
                
                # Fallback PaddleOCR text nếu không dùng VietOCR hoặc crop lỗi
                if len(paddle_text) >= 2 and paddle_score >= 0.4:
                    texts.append(paddle_text)
                    confidences.append(paddle_score)

            if texts:
                return " | ".join(texts), round(sum(confidences) / len(confidences), 3)
        except Exception as e:
            # Debug nếu cần
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

def process_zip_archive(zpath, ocr_engine, frames_df, pkg_idx, total_pkgs, processed_videos_set=None):
    """Đọc trực tiếp byte ảnh từ ZIP vào RAM và chạy OCR kèm thanh tiến trình (Bỏ qua video đã làm)"""
    records = []
    zip_name = Path(zpath).name
    
    with zipfile.ZipFile(zpath, "r") as zf:
        img_names = [n for n in zf.namelist() if n.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        with tqdm(img_names, desc=f"⚡ [OCR {pkg_idx}/{total_pkgs}] {zip_name}", unit="frame", leave=False) as pbar:
            for img_name in pbar:
                match_vid = re.search(r'(L\d+_V\d+)', img_name)
                if not match_vid:
                    continue
                video_id = match_vid.group(1)

                # Bỏ qua nếu video này đã được xử lý xong từ trước
                if processed_videos_set and video_id in processed_videos_set:
                    continue

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

                img_bytes = zf.read(img_name)
                img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                
                if img_array is not None:
                    text, conf = ocr_engine.predict(img_array)
                    if text:
                        records.append({
                            "video_id": video_id,
                            "keyframe_index": kf_idx,
                            "frame_idx": frame_idx,
                            "pts_time": pts_time,
                            "ocr_text": text,
                            "confidence": conf
                        })
                        pbar.set_postfix({"chữ_tìm_thấy": len(records), "video": video_id})
    return records

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
        print(f"\n💾 [CHECKPOINT] Đã ghi thành công {len(combined_df):,} khung hình có chữ vào {out_file}")

def main():
    parser = argparse.ArgumentParser(description="BTC/Drive -> In-Memory SOTA OCR -> Auto-Delete ZIP & Auto-Resume")
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--urls_file", type=str, default="config/drive_keyframes_urls.txt")
    parser.add_argument("--start_index", type=int, default=1)
    parser.add_argument("--start_from", type=str, default=None)
    parser.add_argument("--keyframes_dir", type=str, default="Keyframes")
    parser.add_argument("--output_path", type=str, default="data/processed/ocr_results.parquet")
    parser.add_argument("--use_vietocr", action="store_true", default=True)
    parser.add_argument("--use_gpu", action="store_true", default=True)
    args = parser.parse_args()

    ocr_engine = VietnameseMaxAccuracyOCR(use_vietocr=args.use_vietocr, use_gpu=args.use_gpu)

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
            
            new_records = []
            if is_remote:
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip:
                    temp_zip_path = tmp_zip.name

                download_file(target, temp_zip_path, pkg_idx, total_pkgs)
                new_records = process_zip_archive(temp_zip_path, ocr_engine, frames_df, pkg_idx, total_pkgs, processed_videos_set)

                if os.path.exists(temp_zip_path):
                    os.remove(temp_zip_path)
            else:
                new_records = process_zip_archive(target, ocr_engine, frames_df, pkg_idx, total_pkgs, processed_videos_set)

            if new_records:
                save_and_merge_parquet(new_records, out_file)
                for r in new_records:
                    processed_videos_set.add(r["video_id"])
                
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
    print(f"🎉 HOÀN TẤT OCR: Tổng cộng có {final_count:,} khung hình có chữ trong {out_file}")
    print(f"⏱️ Tổng thời gian chạy đợt này: {elapsed:.0f}s")
    print("="*70)

if __name__ == "__main__":
    main()
