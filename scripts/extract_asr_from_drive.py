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
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import settings

def download_file(url_or_id, target_path):
    """Tải 1 file zip video tốc độ cao trực tiếp từ máy chủ BTC (ledo.io.vn) hoặc Google Drive"""
    print(f"\n📥 Đang tải: {url_or_id}")
    
    if "ledo.io.vn" in url_or_id or (url_or_id.startswith("http") and "drive.google.com" not in url_or_id):
        response = requests.get(url_or_id, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1MB chunk
        
        with open(target_path, 'wb') as file, tqdm(
            desc=Path(url_or_id).name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)
    else:
        try:
            import gdown
        except ImportError:
            print("[ERROR] Chưa cài gdown! Hãy chạy: pip install gdown")
            sys.exit(1)
        if "drive.google.com" in url_or_id:
            gdown.download(url=url_or_id, output=str(target_path), quiet=False, fuzzy=True)
        else:
            gdown.download(id=url_or_id, output=str(target_path), quiet=False)

def process_video_zip(zpath, model, beam_size):
    """Trích xuất cuốn chiếu từng video trong file zip và nhận diện tiếng Việt bằng PhoWhisper/Whisper"""
    records = []
    with zipfile.ZipFile(zpath, "r") as zf:
        mp4_names = [n for n in zf.namelist() if n.lower().endswith('.mp4')]
        
        for mp4_name in tqdm(mp4_names, desc=f"Transcribing {Path(zpath).name}"):
            # Bắt chính xác video_id dạng Lxx_Vxxx (bất chấp thư mục lồng nhau bên trong zip)
            match_vid = re.search(r'(L\d+_V\d+)', mp4_name)
            if match_vid:
                video_id = match_vid.group(1)
            else:
                video_id = Path(mp4_name).stem
            
            # Giải nén tạm 1 video
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(zf.read(mp4_name))

            try:
                segments, info = model.transcribe(
                    tmp_path,
                    language="vi",
                    beam_size=beam_size,
                    best_of=beam_size,
                    temperature=[0.0, 0.2, 0.4],
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                    condition_on_previous_text=True
                )
                
                for seg in segments:
                    text = seg.text.strip()
                    if len(text) > 2:
                        fps = 30.0
                        start_frame = int(seg.start * fps)
                        end_frame = int(seg.end * fps)
                        
                        records.append({
                            "video_id": video_id,
                            "start_time": round(seg.start, 2),
                            "end_time": round(seg.end, 2),
                            "start_frame": start_frame,
                            "end_frame": end_frame,
                            "transcript": text
                        })
            except Exception as e:
                print(f"[WARNING] Lỗi xử lý video {video_id}: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    return records

def main():
    parser = argparse.ArgumentParser(description="Download Video ZIP from BTC/Drive Link -> VinAI PhoWhisper ASR -> Auto-Delete")
    parser.add_argument("--url", type=str, default=None,
                        help="Single direct link of Videos_Lxx_a.zip (e.g. from ledo.io.vn or Drive)")
    parser.add_argument("--urls_file", type=str, default="config/drive_videos_urls.txt",
                        help="Path to .txt file containing list of links (default: config/drive_videos_urls.txt)")
    parser.add_argument("--videos_dir", type=str, default="Videos",
                        help="Local folder containing Videos_Lxx_a.zip (if already downloaded)")
    parser.add_argument("--output_path", type=str, default="data/processed/transcripts.parquet",
                        help="Path to save extracted ASR transcripts parquet file")
    parser.add_argument("--model_size", type=str, default="vinai/PhoWhisper-large",
                        help="Model choice: 'vinai/PhoWhisper-large' (VinAI) OR 'large-v3' (OpenAI)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run Whisper (cuda on GPU A100)")
    parser.add_argument("--beam_size", type=int, default=5,
                        help="Beam search size (5 or 10 for best quality)")
    parser.add_argument("--compute_type", type=str, default="float16",
                        help="Compute type: float16 on GPU A100")
    args = parser.parse_args()

    print(f"=== Khởi tạo Mô Hình ASR Tiếng Việt: {args.model_size} trên GPU {args.device} ===")
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("[ERROR] faster-whisper chưa được cài đặt! Hãy chạy: pip install faster-whisper")
        sys.exit(1)

    model = WhisperModel(args.model_size, device=args.device, compute_type=args.compute_type)

    all_transcript_records = []
    out_file = Path(args.output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if out_file.exists():
        try:
            existing_df = pd.read_parquet(out_file)
            all_transcript_records = existing_df.to_dict("records")
            print(f"Đã nạp {len(all_transcript_records)} phân đoạn lời thoại hiện có từ file trước đó.")
        except Exception:
            pass

    targets = []
    if args.url:
        targets.append(args.url)
    elif args.urls_file and Path(args.urls_file).exists():
        with open(args.urls_file, "r", encoding="utf-8") as f:
            targets = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        print(f"Đã nạp {len(targets)} link từ file {args.urls_file}")
    else:
        vid_dir = Path(args.videos_dir)
        if vid_dir.exists():
            targets = sorted(list(vid_dir.glob("*.zip")) + list(vid_dir.glob("*/*.zip")))
            print(f"Đã tìm thấy {len(targets)} file ZIP trong thư mục local {vid_dir}")

    if not targets:
        print("[ERROR] Không tìm thấy link tải hoặc file ZIP nào để xử lý!")
        sys.exit(1)

    start_time = time.time()

    for idx, target in enumerate(targets, start=1):
        print(f"\n========================================================")
        print(f"🎥 Đang xử lý gói video {idx}/{len(targets)}: {target}")
        print(f"========================================================")

        is_remote_link = isinstance(target, str) and ("http" in target or "drive.google.com" in target or len(target) > 20 and not Path(target).exists())

        if is_remote_link:
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip:
                temp_zip_path = tmp_zip.name

            download_file(target, temp_zip_path)
            
            records = process_video_zip(temp_zip_path, model, args.beam_size)
            all_transcript_records.extend(records)

            if os.path.exists(temp_zip_path):
                os.remove(temp_zip_path)
                print(f"🗑️ Đã xóa sạch file video zip tạm thời trên server để bảo vệ dung lượng đĩa!")
        else:
            records = process_video_zip(target, model, args.beam_size)
            all_transcript_records.extend(records)

        # Lưu checkpoint
        if all_transcript_records:
            pd.DataFrame(all_transcript_records).to_parquet(out_file, index=False)
            print(f"💾 Đã lưu checkpoint: {len(all_transcript_records)} phân đoạn lời thoại vào {out_file}")

    print(f"\n🎉 [HOÀN TẤT TOÀN BỘ] Tổng cộng trích xuất được {len(all_transcript_records)} phân đoạn lời thoại!")
    print(f"File lưu tại: {out_file} (Tổng thời gian: {time.time() - start_time:.2f}s)")

if __name__ == "__main__":
    main()
