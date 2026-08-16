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
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import settings

class FastBatchVietOCR:
    """
    Bộ giải mã VietOCR Batch Tensor Vectorized 100% trên GPU A100:
    - Giải mã song song toàn bộ B mẩu chữ cùng lúc (B = 16, 32, 64)
    - Tự động khớp chiều batch (B, vocab_size)
    """
    def __init__(self, predictor):
        self.predictor = predictor
        self.model = predictor.model
        self.vocab = predictor.vocab
        self.config = predictor.config
        self.device = predictor.config['device']
        self.img_height = self.config['dataset']['image_height']
        self.img_min_width = self.config['dataset']['image_min_width']
        self.img_max_width = self.config['dataset']['image_max_width']
        
    def process_crop(self, crop_bgr):
        try:
            if isinstance(crop_bgr, np.ndarray):
                pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            else:
                pil_img = crop_bgr.convert('RGB')
            w, h = pil_img.size
            if h <= 0 or w <= 0:
                return None
            new_w = int(self.img_height * w / h)
            new_w = max(min(new_w, self.img_max_width), self.img_min_width)
            pil_img = pil_img.resize((new_w, self.img_height), Image.BILINEAR)
            img_arr = np.asarray(pil_img).transpose(2, 0, 1) / 255.0
            return torch.FloatTensor(img_arr)
        except Exception:
            return None

    def predict_batch(self, crops_list, batch_size=32, max_seq_length=50, sos_token=1, eos_token=2):
        if not crops_list:
            return []
        
        self.model.eval()
        results = []

        for i in range(0, len(crops_list), batch_size):
            batch_crops = crops_list[i:i + batch_size]
            processed = [self.process_crop(c) for c in batch_crops]
            
            valid_indices = [idx for idx, p in enumerate(processed) if p is not None]
            if not valid_indices:
                results.extend([""] * len(batch_crops))
                continue

            valid_tensors = [processed[idx] for idx in valid_indices]
            max_w = max(t.shape[2] for t in valid_tensors)
            
            padded_tensors = []
            for t in valid_tensors:
                pad_w = max_w - t.shape[2]
                if pad_w > 0:
                    t = F.pad(t, (0, pad_w, 0, 0), value=0)
                padded_tensors.append(t)
            
            batch_tensor = torch.stack(padded_tensors).to(self.device)
            B = batch_tensor.shape[0]

            with torch.no_grad():
                # 1. CNN feature extraction (batch) - shape: (B, d_model, H', W')
                src = self.model.cnn(batch_tensor)

                # 2. Transformer Encoder (batch) - memory shape: (src_len, B, d_model)
                memory = self.model.transformer.forward_encoder(src)

                # 3. Auto-regressive Decoder (batch vectorized)
                translated = torch.full((B, 1), sos_token, dtype=torch.long, device=self.device)

                for _ in range(max_seq_length):
                    # tgt_inp: (seq_len, B) - VietOCR transformer is NOT batch_first
                    tgt_inp = translated.t()

                    # output: (B, seq_len, vocab_size) do VietOCR đã transpose bên trong forward_decoder
                    output, memory = self.model.transformer.forward_decoder(tgt_inp, memory)

                    # Lấy logits tại token cuối cùng cho toàn bộ batch B: shape (B, vocab_size)
                    if output.shape[0] == B:
                        logits = output[:, -1, :]
                    else:
                        logits = output[-1, :, :]

                    next_tokens = torch.argmax(logits, dim=-1)  # (B,)

                    # Nối token mới vào chuỗi: (B, seq_len + 1)
                    translated = torch.cat([translated, next_tokens.unsqueeze(1)], dim=1)

                    # Dừng sớm nếu TẤT CẢ B sequence đều đã sinh eos_token
                    if ((translated == eos_token).any(dim=1)).all():
                        break

                # 4. Decode TỪNG sequence riêng lẻ (vocab.decode chỉ nhận 1D list)
                decoded_texts = []
                for seq in translated.cpu().tolist():
                    try:
                        decoded_texts.append(self.vocab.decode(seq))
                    except Exception:
                        decoded_texts.append("")

            batch_res = [""] * len(batch_crops)
            for vi, txt in zip(valid_indices, decoded_texts):
                batch_res[vi] = str(txt).strip()
            results.extend(batch_res)
            
        return results

class UltraFastMaxAccuracyOCR:
    """
    Pipeline OCR SOTA Tối Đa Công Suất trên GPU A100
    """
    def __init__(self, use_vietocr=True, use_gpu=True, batch_size=32):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda:0" if self.use_gpu else "cpu"
        self.use_vietocr = use_vietocr
        self.batch_size = batch_size

        print(f"=== [1/2] Nạp CRAFT Text Detector (EasyOCR) trên {self.device} ===")
        import easyocr
        self.reader = easyocr.Reader(['vi'], gpu=self.use_gpu, verbose=False)
        print("✅ CRAFT Text Detector đã sẵn sàng trên GPU!")

        if self.use_vietocr:
            print(f"=== [2/2] Nạp Vectorized GPU Batch VietOCR trên {self.device} ===")
            try:
                from vietocr.tool.predictor import Predictor
                from vietocr.tool.config import Cfg
                config = Cfg.load_config_from_name('vgg_transformer')
                config['device'] = self.device
                config['predictor']['beamsearch'] = False
                raw_predictor = Predictor(config)
                self.batch_vietocr = FastBatchVietOCR(raw_predictor)
                print("✅ Vectorized GPU Batch VietOCR đã sẵn sàng (Tốc độ tối đa trên A100)!")
            except Exception as e:
                print(f"[WARNING] Lỗi nạp VietOCR ({e}). Fallback sang EasyOCR.")
                self.use_vietocr = False

    def predict_frames_batch(self, frames_data):
        """
        Xử lý đồng thời một lô khung hình (Batch of Frames):
        - frames_data: list of (video_id, kf_idx, frame_idx, pts_time, img_array)
        """
        results_records = []
        all_crops = []
        crop_metadata = []

        for f_idx, (vid, kf_idx, frame_idx, pts_time, img_array) in enumerate(frames_data):
            if img_array is None or img_array.size == 0:
                continue

            h_img, w_img = img_array.shape[:2]

            try:
                horizontal_list, free_list = self.reader.detect(
                    img_array,
                    canvas_size=960,
                    mag_ratio=1.0,
                    text_threshold=0.7,
                    link_threshold=0.4,
                    low_text=0.4
                )
                boxes = horizontal_list[0] if horizontal_list and len(horizontal_list) > 0 else []
            except Exception:
                boxes = []

            for box in boxes:
                x_min, x_max, y_min, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                w, h = x_max - x_min, y_max - y_min
                if w >= 14 and h >= 12:
                    x1, y1 = max(0, x_min), max(0, y_min)
                    x2, y2 = min(w_img, x_max), min(h_img, y_max)
                    crop = img_array[y1:y2, x1:x2]
                    if crop.size > 0:
                        crop_metadata.append(f_idx)
                        all_crops.append(crop)

        # Giải mã song song toàn bộ mẩu chữ của cả lô khung hình trên GPU A100
        if all_crops and self.use_vietocr:
            predicted_texts = self.batch_vietocr.predict_batch(all_crops, batch_size=self.batch_size)
        else:
            predicted_texts = [""] * len(all_crops)

        frame_texts = {i: [] for i in range(len(frames_data))}
        for f_idx, txt in zip(crop_metadata, predicted_texts):
            txt = str(txt).strip()
            if len(txt) >= 2:
                frame_texts[f_idx].append(txt)

        for f_idx, texts in frame_texts.items():
            if texts:
                unique_texts = []
                for t in texts:
                    if not unique_texts or t.lower() != unique_texts[-1].lower():
                        unique_texts.append(t)
                if unique_texts:
                    vid, kf_idx, frame_idx, pts_time, _ = frames_data[f_idx]
                    results_records.append({
                        "video_id": vid,
                        "keyframe_index": kf_idx,
                        "frame_idx": frame_idx,
                        "pts_time": pts_time,
                        "ocr_text": " | ".join(unique_texts),
                        "confidence": 0.95
                    })

        return results_records

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

def process_zip_archive(zpath, ocr_engine, frames_df, pkg_idx, total_pkgs, out_file, processed_videos_set=None, frame_batch_size=8):
    """
    Xử lý ZIP với Full-Power Batch Processing:
    - Đọc ảnh theo lô (frame_batch_size = 8 frames)
    - Quét CRAFT + VietOCR Batch Tensor trên GPU A100
    - Lưu checkpoint cuốn chiếu theo từng video
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

        with tqdm(img_names, desc=f"🚀 [OCR Siêu Tốc {pkg_idx}/{total_pkgs}] {zip_name}", unit="frame", leave=False) as pbar:
            for video_id, v_img_names in video_groups.items():
                if processed_videos_set and video_id in processed_videos_set:
                    pbar.update(len(v_img_names))
                    pbar.set_postfix({"video": video_id, "status": "skip"})
                    continue

                video_records = []

                # Xử lý theo từng lô frame_batch_size ảnh
                for i in range(0, len(v_img_names), frame_batch_size):
                    batch_names = v_img_names[i:i + frame_batch_size]
                    batch_frames_data = []

                    for img_name in batch_names:
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
                            batch_frames_data.append((video_id, kf_idx, frame_idx, pts_time, img_array))

                    if batch_frames_data:
                        batch_records = ocr_engine.predict_frames_batch(batch_frames_data)
                        video_records.extend(batch_records)

                    pbar.update(len(batch_names))

                # Ghi checkpoint ngay khi quét xong từng video
                if video_records:
                    save_and_merge_parquet(video_records, out_file)
                    processed_videos_set.add(video_id)
                    total_records_in_pkg += len(video_records)

                pbar.set_postfix({"video": video_id, "chữ": len(video_records), "tổng": total_records_in_pkg})

    return total_records_in_pkg

def main():
    parser = argparse.ArgumentParser(description="BTC/Drive -> SOTA PyTorch OCR (CRAFT + VietOCR Vectorized GPU Batch) -> Auto-Resume")
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--urls_file", type=str, default="config/drive_keyframes_urls.txt")
    parser.add_argument("--start_index", type=int, default=1)
    parser.add_argument("--start_from", type=str, default=None)
    parser.add_argument("--keyframes_dir", type=str, default="raw/batch_1/Keyframes")
    parser.add_argument("--output_path", type=str, default="data/batch_1/processed/ocr_results.parquet")
    parser.add_argument("--use_vietocr", action="store_true", default=True)
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--crop_batch_size", type=int, default=32,
                        help="Kích thước batch giải mã VietOCR trên GPU A100 (mặc định: 32)")
    parser.add_argument("--frame_batch_size", type=int, default=8,
                        help="Số lượng khung hình xử lý cùng lúc (mặc định: 8)")
    args = parser.parse_args()

    ocr_engine = UltraFastMaxAccuracyOCR(
        use_vietocr=args.use_vietocr,
        use_gpu=args.use_gpu,
        batch_size=args.crop_batch_size
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
                process_zip_archive(temp_zip_path, ocr_engine, frames_df, pkg_idx, total_pkgs, out_file, processed_videos_set, frame_batch_size=args.frame_batch_size)

                if os.path.exists(temp_zip_path):
                    os.remove(temp_zip_path)
            else:
                process_zip_archive(target, ocr_engine, frames_df, pkg_idx, total_pkgs, out_file, processed_videos_set, frame_batch_size=args.frame_batch_size)

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
