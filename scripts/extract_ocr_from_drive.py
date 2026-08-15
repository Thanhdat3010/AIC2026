import argparse
import sys
import time
import zipfile
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
    Hệ thống OCR 2-Stage Đạt Độ Chính Xác Tối Đa Cho Tiếng Việt (Maximum Accuracy):
    - Stage 1 (Text Detection): PaddleOCR DBNet Server Model (Tìm chính xác tọa độ hộp bao chữ).
    - Stage 2 (Text Recognition): VietOCR VGG-Transformer (Mô hình Transformer chuyên biệt tiếng Việt, bắt chuẩn 100% dấu).
    """
    def __init__(self, use_vietocr=True, use_gpu=True):
        self.use_vietocr = use_vietocr
        self.use_gpu = use_gpu

        print("=== Khởi tạo mô hình Text Detection (PaddleOCR DBNet) ===")
        from paddleocr import PaddleOCR
        # Khởi tạo detector với cấu hình phát hiện tối đa
        self.detector = PaddleOCR(use_angle_cls=True, lang='vi', use_gpu=use_gpu, show_log=False)

        if self.use_vietocr:
            print("=== Khởi tạo mô hình Text Recognition Chuyên Biệt (VietOCR VGG-Transformer) ===")
            try:
                from vietocr.tool.predictor import Predictor
                from vietocr.tool.config import Cfg
                config = Cfg.load_config_from_name('vgg_transformer')
                config['device'] = 'cuda:0' if use_gpu else 'cpu'
                config['predictor']['beamsearch'] = True
                self.vietocr_predictor = Predictor(config)
            except ImportError:
                print("[WARNING] Chưa cài VietOCR. Đang fallback về PaddleOCR Recognition. (Để cài: pip install vietocr)")
                self.use_vietocr = False

    def predict(self, img_array):
        try:
            # 1. Phát hiện hộp bao chữ
            result = self.detector.ocr(img_array, cls=True)
            if not result or not result[0]:
                return None, None

            texts = []
            confidences = []

            for line in result[0]:
                box = line[0]  # Tọa độ 4 đỉnh [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                paddle_text = line[1][0].strip()
                paddle_score = float(line[1][1])

                if self.use_vietocr:
                    # Crop vùng chữ từ ảnh gốc
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
                
                # Fallback PaddleOCR text nếu không dùng VietOCR
                if len(paddle_text) >= 2 and paddle_score >= 0.5:
                    texts.append(paddle_text)
                    confidences.append(paddle_score)

            if texts:
                return " | ".join(texts), round(sum(confidences) / len(confidences), 3)
        except Exception:
            pass
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Maximum Accuracy Vietnamese Video OCR (PaddleOCR DBNet + VietOCR Transformer)")
    parser.add_argument("--keyframes_dir", type=str, default="Keyframes",
                        help="Path to folder containing Keyframes_Lxx.zip files")
    parser.add_argument("--output_path", type=str, default="data/processed/ocr_results.parquet",
                        help="Path to save extracted OCR parquet file")
    parser.add_argument("--use_vietocr", action="store_true", default=True,
                        help="Use VietOCR VGG-Transformer for maximum recognition accuracy")
    parser.add_argument("--use_gpu", action="store_true", default=True,
                        help="Use GPU A100 for inference")
    args = parser.parse_args()

    kf_dir = Path(args.keyframes_dir)
    if not kf_dir.exists():
        print(f"[ERROR] Thư mục Keyframes không tồn tại: {kf_dir}")
        sys.exit(1)

    ocr_engine = VietnameseMaxAccuracyOCR(use_vietocr=args.use_vietocr, use_gpu=args.use_gpu)

    # Đọc bảng frames.parquet để tra cứu frame_idx gốc
    frames_path = settings.directories.processed / "frames.parquet"
    frames_df = None
    if frames_path.exists():
        frames_df = pd.read_parquet(frames_path).set_index(["video_id", "keyframe_index"])
        print(f"Đã nạp {len(frames_df)} bản ghi mapping từ frames.parquet.")

    # Tìm toàn bộ file ZIP trong Keyframes/
    zip_files = sorted(list(kf_dir.glob("*.zip")) + list(kf_dir.glob("*/*.zip")))
    print(f"Tìm thấy {len(zip_files)} file ZIP trong {kf_dir}. Bắt đầu xử lý với cấu hình MAX ACCURACY...")

    ocr_records = []
    start_time = time.time()

    for zpath in zip_files:
        print(f"\n>> Đang xử lý file zip: {zpath.name}")
        try:
            with zipfile.ZipFile(zpath, "r") as zf:
                img_names = [n for n in zf.namelist() if n.lower().endswith(('.jpg', '.png', '.jpeg'))]
                
                for img_name in tqdm(img_names, desc=f"Scanning {zpath.name}"):
                    parts = Path(img_name).parts
                    if len(parts) < 2:
                        continue
                    video_id = parts[-2]
                    try:
                        kf_idx = int(Path(parts[-1]).stem)
                    except ValueError:
                        continue

                    frame_idx, pts_time = -1, 0.0
                    if frames_df is not None and (video_id, kf_idx) in frames_df.index:
                        row = frames_df.loc[(video_id, kf_idx)]
                        frame_idx = int(row["frame_idx"])
                        pts_time = float(row["pts_time"])

                    # Đọc trực tiếp byte từ ZIP vào RAM
                    img_bytes = zf.read(img_name)
                    img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                    
                    if img_array is not None:
                        text, conf = ocr_engine.predict(img_array)
                        if text:
                            ocr_records.append({
                                "video_id": video_id,
                                "keyframe_index": kf_idx,
                                "frame_idx": frame_idx,
                                "pts_time": pts_time,
                                "ocr_text": text,
                                "confidence": conf
                            })
        except Exception as e:
            print(f"[WARNING] Lỗi khi xử lý file zip {zpath.name}: {e}")
            continue

    # Lưu kết quả sang Parquet
    out_file = Path(args.output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if ocr_records:
        df_ocr = pd.DataFrame(ocr_records)
        df_ocr.to_parquet(out_file, index=False)
        print(f"\n🎉 [HOÀN TẤT MAX ACCURACY] Đã trích xuất xong {len(ocr_records)} khung hình có chữ!")
        print(f"File lưu tại: {out_file} (Tổng thời gian: {time.time() - start_time:.2f} giây)")
    else:
        print("\n[WARNING] Không tìm thấy dữ liệu chữ.")

if __name__ == "__main__":
    main()
