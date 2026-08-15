import argparse
import sys
import time
import zipfile
import tempfile
import os
import re
import requests
from pathlib import Path
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
    - Stage 1 (Text Detection): PaddleOCR DBNet (paddleocr==2.8.1)
    - Stage 2 (Text Recognition): VietOCR VGG-Transformer
    """
    def __init__(self, use_vietocr=True, use_gpu=True):
        self.use_vietocr = use_vietocr
        self.use_gpu = use_gpu

        print("=== [1/2] Đang nạp mô hình Text Detection (PaddleOCR DBNet) ===")
        from paddleocr import PaddleOCR
        # paddleocr==2.8.1 classic API (ổn định, đã test kỹ trên GPU A100)
        self.detector = PaddleOCR(
            use_angle_cls=True,
            lang='vi',
            use_gpu=use_gpu,
            show_log=False
        )
        print("✅ PaddleOCR DBNet đã sẵn sàng.")

        if self.use_vietocr:
            print("=== [2/2] Đang nạp mô hình Text Recognition (VietOCR VGG-Transformer) ===")
            try:
                from vietocr.tool.predictor import Predictor
                from vietocr.tool.config import Cfg
                config = Cfg.load_config_from_name('vgg_transformer')
                config['device'] = 'cuda:0' if use_gpu else 'cpu'
                config['predictor']['beamsearch'] = True
                self.vietocr_predictor = Predictor(config)
                print("✅ VietOCR VGG-Transformer đã sẵn sàng (MAX ACCURACY cho tiếng Việt).")
            except ImportError:
                print("[WARNING] Chưa cài VietOCR. Fallback về PaddleOCR Recognition. (pip install vietocr)")
                self.use_vietocr = False

    def predict(self, img_array):
        try:
            result = self.detector.ocr(img_array, cls=True)
            if not result or not result[0]:
                return None, None

            texts = []
            confidences = []

            for line in result[0]:
                box = line[0]
                paddle_text = line[1][0].strip()
                paddle_score = float(line[1][1])

                if self.use_vietocr:
                    pts = np.array(box, dtype=np.int32)
                    rect = cv2.boundingRect(pts)
                    x, y, w, h = rect
                    if w > 5 and h > 5:
                        crop = img_array[y:y+h, x:x+w]
                        if crop.size > 0:
                            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                            vietocr_text, prob = self.vietocr_predictor.predict(pil_img, return_prob=True)
                            if len(vietocr_text.strip()) >= 2 and prob >= 0.5:
                                texts.append(vietocr_text.strip())
                                confidences.append(prob)
                                continue
                
                if len(paddle_text) >= 2 and paddle_score >= 0.5:
                    texts.append(paddle_text)
                    confidences.append(paddle_score)

            if texts:
                return " | ".join(texts), round(sum(confidences) / len(confidences), 3)
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

def process_zip_archive(zpath, ocr_engine, frames_df, pkg_idx, total_pkgs):
    """Đọc trực tiếp byte ảnh từ ZIP vào RAM và chạy OCR kèm thanh tiến trình"""
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
                        pbar.set_postfix({"found": len(records), "vid": video_id})
    return records

def main():
    parser = argparse.ArgumentParser(description="BTC/Drive -> In-Memory SOTA OCR -> Auto-Delete ZIP")
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--urls_file", type=str, default="config/drive_keyframes_urls.txt")
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

    all_ocr_records = []
    out_file = Path(args.output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if out_file.exists():
        try:
            existing_df = pd.read_parquet(out_file)
            all_ocr_records = existing_df.to_dict("records")
            print(f"🔄 Resume: đã nạp {len(all_ocr_records)} bản ghi OCR hiện có.")
        except Exception:
            pass

    targets = []
    if args.url:
        targets.append(args.url)
    elif args.urls_file and Path(args.urls_file).exists():
        with open(args.urls_file, "r", encoding="utf-8") as f:
            targets = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        print(f"📋 Đã nạp {len(targets)} link từ file {args.urls_file}")
    else:
        kf_dir = Path(args.keyframes_dir)
        if kf_dir.exists():
            targets = sorted(list(kf_dir.glob("*.zip")) + list(kf_dir.glob("*/*.zip")))
            print(f"📋 Đã tìm thấy {len(targets)} file ZIP local")

    if not targets:
        print("[ERROR] Không tìm thấy link tải hoặc file ZIP nào!")
        sys.exit(1)

    start_time = time.time()
    total_pkgs = len(targets)

    with tqdm(enumerate(targets, start=1), total=total_pkgs, desc="📦 [TỔNG] Keyframes", unit="gói") as main_bar:
        for pkg_idx, target in main_bar:
            is_remote = isinstance(target, str) and ("http" in target or "drive.google.com" in target or (len(target) > 20 and not Path(target).exists()))
            
            if is_remote:
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip:
                    temp_zip_path = tmp_zip.name

                download_file(target, temp_zip_path, pkg_idx, total_pkgs)
                records = process_zip_archive(temp_zip_path, ocr_engine, frames_df, pkg_idx, total_pkgs)
                all_ocr_records.extend(records)

                if os.path.exists(temp_zip_path):
                    os.remove(temp_zip_path)
            else:
                records = process_zip_archive(target, ocr_engine, frames_df, pkg_idx, total_pkgs)
                all_ocr_records.extend(records)

            if all_ocr_records:
                pd.DataFrame(all_ocr_records).to_parquet(out_file, index=False)
                main_bar.set_postfix({"tổng_chữ": len(all_ocr_records)})

    elapsed = time.time() - start_time
    print("\n" + "="*65)
    print(f"🎉 HOÀN TẤT OCR: {len(all_ocr_records):,} khung hình có chữ")
    print(f"💾 Lưu tại: {out_file} ({elapsed:.0f}s)")
    print("="*65)

if __name__ == "__main__":
    main()
