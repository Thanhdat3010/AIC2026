import argparse
import sys
import time
import zipfile
import tempfile
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import settings

def main():
    parser = argparse.ArgumentParser(description="Maximum Accuracy Vietnamese ASR (VinAI PhoWhisper & OpenAI Whisper large-v3)")
    parser.add_argument("--videos_dir", type=str, default="Videos",
                        help="Path to folder containing Videos_Lxx_a.zip files OR .mp4 files")
    parser.add_argument("--output_path", type=str, default="data/processed/transcripts.parquet",
                        help="Path to save extracted ASR transcripts parquet file")
    parser.add_argument("--model_size", type=str, default="vinai/PhoWhisper-large",
                        help="Model choice: 'vinai/PhoWhisper-large' (VinAI SOTA for Vietnamese accents) OR 'large-v3' (OpenAI)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run Whisper (cuda on GPU A100)")
    parser.add_argument("--beam_size", type=int, default=5,
                        help="Beam search size (5 or 10 for best quality)")
    parser.add_argument("--compute_type", type=str, default="float16",
                        help="Compute type: float16 on GPU A100")
    args = parser.parse_args()

    vid_dir = Path(args.videos_dir)
    if not vid_dir.exists():
        print(f"[ERROR] Thư mục Videos không tồn tại: {vid_dir}")
        sys.exit(1)

    print(f"=== Khởi tạo Mô Hình ASR Tiếng Việt: {args.model_size} trên GPU {args.device} ===")
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("[ERROR] faster-whisper chưa được cài đặt! Hãy chạy: pip install faster-whisper")
        sys.exit(1)

    # Khởi tạo mô hình (Tự động nạp PhoWhisper của VinAI hoặc Whisper large-v3)
    model = WhisperModel(args.model_size, device=args.device, compute_type=args.compute_type)

    zip_files = sorted(list(vid_dir.glob("*.zip")) + list(vid_dir.glob("*/*.zip")))
    raw_mp4s = sorted(list(vid_dir.glob("*.mp4")) + list(vid_dir.glob("*/*.mp4")))

    transcript_records = []
    start_time = time.time()

    if zip_files:
        print(f"Tìm thấy {len(zip_files)} file ZIP video. Bắt đầu trích xuất với {args.model_size} (Beam Size {args.beam_size})...")
        for zpath in zip_files:
            print(f"\n>> Đang xử lý file zip video: {zpath.name}")
            with zipfile.ZipFile(zpath, "r") as zf:
                mp4_names = [n for n in zf.namelist() if n.lower().endswith('.mp4')]
                
                for mp4_name in tqdm(mp4_names, desc=f"Transcribing {zpath.name}"):
                    video_id = Path(mp4_name).stem
                    
                    # Trích xuất tạm 1 video
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                        tmp_path = tmp_file.name
                        tmp_file.write(zf.read(mp4_name))

                    try:
                        segments, info = model.transcribe(
                            tmp_path,
                            language="vi",
                            beam_size=args.beam_size,
                            best_of=args.beam_size,
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
                                
                                transcript_records.append({
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
    else:
        print(f"Quét {len(raw_mp4s)} file video .mp4...")
        for vpath in tqdm(raw_mp4s, desc="Transcribing Videos"):
            video_id = vpath.stem
            try:
                segments, info = model.transcribe(
                    str(vpath),
                    language="vi",
                    beam_size=args.beam_size,
                    best_of=args.beam_size,
                    temperature=[0.0, 0.2, 0.4],
                    vad_filter=True,
                    condition_on_previous_text=True
                )
                for seg in segments:
                    text = seg.text.strip()
                    if len(text) > 2:
                        fps = 30.0
                        transcript_records.append({
                            "video_id": video_id,
                            "start_time": round(seg.start, 2),
                            "end_time": round(seg.end, 2),
                            "start_frame": int(seg.start * fps),
                            "end_frame": int(seg.end * fps),
                            "transcript": text
                        })
            except Exception as e:
                continue

    # Lưu kết quả sang Parquet
    out_file = Path(args.output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if transcript_records:
        df_trans = pd.DataFrame(transcript_records)
        df_trans.to_parquet(out_file, index=False)
        print(f"\n🎉 [HOÀN TẤT MAX ACCURACY] Đã trích xuất xong {len(transcript_records)} phân đoạn lời thoại!")
        print(f"File lưu tại: {out_file} (Tổng thời gian: {time.time() - start_time:.2f} giây)")
    else:
        print("\n[WARNING] Không trích xuất được lời thoại nào.")

if __name__ == "__main__":
    main()
