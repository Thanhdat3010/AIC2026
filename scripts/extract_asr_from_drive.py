import argparse
import sys
import time
import zipfile
import tempfile
import os
import re
import glob
import ctypes
import requests
from pathlib import Path

# ==============================================================================
# TỰ ĐỘNG NẠP THƯ VIỆN NVIDIA CUDA / CUBLAS CHO CTRANSLATE2 TRÊN GPU
# ==============================================================================
def preload_nvidia_cuda_libs():
    """Tự động tìm và nạp các file .so của nvidia-cublas, nvidia-cudnn vào tiến trình"""
    for p in sys.path:
        if "site-packages" in p:
            nvidia_dir = os.path.join(p, "nvidia")
            if os.path.exists(nvidia_dir):
                for lib_so in glob.glob(os.path.join(nvidia_dir, "*", "lib", "*.so*")):
                    try:
                        ctypes.CDLL(lib_so)
                    except Exception:
                        pass
                # Thêm vào LD_LIBRARY_PATH
                for lib_dir in glob.glob(os.path.join(nvidia_dir, "*", "lib")):
                    cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
                    if lib_dir not in cur_ld:
                        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{cur_ld}"

preload_nvidia_cuda_libs()

import pandas as pd
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import settings

def get_actual_model_id(model_size):
    """
    Ánh xạ tên mô hình sang bản CTranslate2 chuẩn của faster-whisper:
    - Nếu chọn VinAI PhoWhisper-large -> Dùng kiendt/PhoWhisper-large-ct2 (Bản CTranslate2 gốc của VinAI)
    """
    if "vinai" in model_size.lower() or "phowhisper" in model_size.lower():
        return "kiendt/PhoWhisper-large-ct2"
    return model_size

def download_file(url_or_id, target_path, pkg_idx, total_pkgs, max_retries=5):
    """Tải file video zip từ server BTC kèm thanh tiến trình tốc độ cao và TỰ ĐỘNG THỬ LẠI KHI ĐỨT MẠNG"""
    filename = Path(url_or_id).name if "http" in url_or_id else f"Video_Package_{pkg_idx}.zip"
    
    if "ledo.io.vn" in url_or_id or (url_or_id.startswith("http") and "drive.google.com" not in url_or_id):
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(url_or_id, stream=True, timeout=(15, 180))
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024 * 1024
                
                with open(target_path, 'wb') as file, tqdm(
                    desc=f"📥 [Tải Video {pkg_idx}/{total_pkgs}] {filename} (Lần {attempt})",
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
                        print(f"\n⚠️ File video zip {filename} tải về chưa hoàn tất. Đang thử lại lần {attempt + 1}/{max_retries}...")
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

def save_and_merge_parquet(new_records, out_file):
    """Ghi checkpoint ngay lập tức sau mỗi video và khử trùng lặp dữ liệu"""
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
        # Khử trùng lặp theo (video_id, start_frame, end_frame)
        combined_df = combined_df.drop_duplicates(subset=["video_id", "start_frame", "end_frame"], keep="last")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_parquet(out_file, index=False)

def process_video_zip(zpath, batched_pipeline, beam_size, batch_size, video_fps_map, pkg_idx, total_pkgs, out_file, processed_videos_set):
    """
    Trích xuất ASR theo cơ chế Checkpoint Từng Video:
    - Xong video nào ghi đĩa ngay video đó (không sợ mất mát)
    - Dùng BatchedInferencePipeline để tăng tốc 3-4x trên GPU A100
    """
    total_records_in_pkg = 0
    zip_name = Path(zpath).name
    
    with zipfile.ZipFile(zpath, "r") as zf:
        mp4_names = [n for n in zf.namelist() if n.lower().endswith('.mp4')]
        
        with tqdm(mp4_names, desc=f"🎙️ [ASR {pkg_idx}/{total_pkgs}] {zip_name}", unit="video", leave=False) as pbar:
            for mp4_name in pbar:
                match_vid = re.search(r'(L\d+_V\d+)', mp4_name)
                if match_vid:
                    video_id = match_vid.group(1)
                else:
                    video_id = Path(mp4_name).stem

                # 1. Bỏ qua nếu video này đã có trong dữ liệu trước đó
                if processed_videos_set and video_id in processed_videos_set:
                    pbar.set_postfix({"video": video_id, "status": "đã_xong (skip)"})
                    continue
                
                fps = video_fps_map.get(video_id, 30.0)

                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                    tmp_file.write(zf.read(mp4_name))

                video_records = []
                try:
                    # 2. Xử lý Batch Inference song song trên GPU
                    try:
                        segments, info = batched_pipeline.transcribe(
                            tmp_path,
                            language="vi",
                            batch_size=batch_size,
                            beam_size=beam_size,
                            vad_filter=True,
                            vad_parameters=dict(min_silence_duration_ms=500),
                            condition_on_previous_text=True
                        )
                    except Exception as e_batch:
                        if "out of memory" in str(e_batch).lower() or "cudaerror" in str(e_batch).lower():
                            print(f"\n[INFO] Video {video_id} dài ngốn VRAM -> Tự động Fallback sang Sequential Inference an toàn...")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            segments, info = batched_pipeline.model.transcribe(
                                tmp_path,
                                language="vi",
                                beam_size=beam_size,
                                vad_filter=True,
                                vad_parameters=dict(min_silence_duration_ms=500),
                                condition_on_previous_text=True
                            )
                        else:
                            raise e_batch
                    
                    for seg in segments:
                        text = seg.text.strip()
                        if len(text) > 2:
                            start_frame = int(round(seg.start * fps))
                            end_frame = int(round(seg.end * fps))
                            
                            video_records.append({
                                "video_id": video_id,
                                "start_time": round(seg.start, 2),
                                "end_time": round(seg.end, 2),
                                "fps": fps,
                                "start_frame": start_frame,
                                "end_frame": end_frame,
                                "transcript": text
                            })
                    
                    # 3. GHI CHECKPOINT NGAY LẬP TỨC CHO VIDEO VỪA XONG
                    if video_records:
                        save_and_merge_parquet(video_records, out_file)
                        processed_videos_set.add(video_id)
                        total_records_in_pkg += len(video_records)
                    
                    pbar.set_postfix({"video": video_id, "câu_mới": len(video_records), "tổng_pkg": total_records_in_pkg})
                    
                except Exception as e:
                    print(f"\n[WARNING] Lỗi xử lý video {video_id}: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                        
    return total_records_in_pkg

def main():
    parser = argparse.ArgumentParser(description="Download Video ZIP -> VinAI PhoWhisper ASR (Batched + Per-Video Checkpoint) -> Auto-Resume")
    parser.add_argument("--url", type=str, default=None,
                        help="Chạy 1 link cụ thể (VD: https://aic-data.ledo.io.vn/Videos_L25_a.zip)")
    parser.add_argument("--urls_file", type=str, default="config/drive_videos_urls.txt",
                        help="File danh sách link (mặc định: config/drive_videos_urls.txt)")
    parser.add_argument("--start_index", type=int, default=1,
                        help="Bắt đầu chạy từ gói số N (1-indexed, ví dụ: --start_index 5)")
    parser.add_argument("--start_from", type=str, default=None,
                        help="Bắt đầu chạy từ file cụ thể (ví dụ: --start_from Videos_L25_a.zip)")
    parser.add_argument("--videos_dir", type=str, default="raw/batch_1/Videos",
                        help="Thư mục video ZIP local")
    parser.add_argument("--output_path", type=str, default="data/batch_1/processed/transcripts.parquet",
                        help="Nơi lưu file kết quả lời thoại parquet")
    parser.add_argument("--model_size", type=str, default="vinai/PhoWhisper-large",
                        help="Mô hình: 'vinai/PhoWhisper-large' hoặc 'large-v3'")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Thiết bị chạy (cuda trên A100)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Kích thước batch inference trên GPU A100 (mặc định: 16)")
    parser.add_argument("--beam_size", type=int, default=5,
                        help="Beam search size (5)")
    parser.add_argument("--compute_type", type=str, default="float16",
                        help="Compute type: float16 (nhanh & nhẹ VRAM)")
    args = parser.parse_args()

    # Nạp mapping FPS thực tế
    frames_path = settings.directories.processed / "frames.parquet"
    video_fps_map = {}
    if frames_path.exists():
        frames_df = pd.read_parquet(frames_path, columns=["video_id", "fps"]).drop_duplicates(subset=["video_id"])
        video_fps_map = dict(zip(frames_df["video_id"], frames_df["fps"]))
        print(f"✅ Đã nạp mapping FPS thực tế của {len(video_fps_map)} video từ frames.parquet.")

    actual_model_id = get_actual_model_id(args.model_size)

    print(f"=== [1/2] Khởi tạo Mô Hình VinAI PhoWhisper: {actual_model_id} trên GPU {args.device} ({args.compute_type}) ===")
    try:
        from faster_whisper import WhisperModel, BatchedInferencePipeline
    except ImportError:
        print("[ERROR] faster-whisper chưa được cài đặt! Hãy chạy: pip install faster-whisper")
        sys.exit(1)

    model = WhisperModel(actual_model_id, device=args.device, compute_type=args.compute_type)
    
    print(f"=== [2/2] Bật BatchedInferencePipeline (Batch Size = {args.batch_size}) để tăng tốc 3-4x ===")
    batched_pipeline = BatchedInferencePipeline(model=model)
    print("✅ Hệ thống VinAI PhoWhisper Batch Pipeline đã sẵn sàng trên GPU A100!")

    out_file = Path(args.output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Đọc danh sách video đã hoàn thành trước đó (Auto-Resume)
    processed_videos_set = set()
    total_existing_records = 0
    if out_file.exists():
        try:
            existing_df = pd.read_parquet(out_file)
            total_existing_records = len(existing_df)
            processed_videos_set = set(existing_df["video_id"].unique())
            print(f"🔄 [RESUME] Đã tìm thấy {total_existing_records:,} phân đoạn lời thoại của {len(processed_videos_set)} video từ trước!")
        except Exception:
            pass

    # Nạp danh sách link video
    all_targets = []
    if args.url:
        all_targets.append(args.url)
    elif args.urls_file and Path(args.urls_file).exists():
        with open(args.urls_file, "r", encoding="utf-8") as f:
            all_targets = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        print(f"📋 Đã nạp {len(all_targets)} link từ file {args.urls_file}")
    else:
        vid_dir = Path(args.videos_dir)
        if vid_dir.exists():
            all_targets = sorted(list(vid_dir.glob("*.zip")) + list(vid_dir.glob("*/*.zip")))
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

    with tqdm(enumerate(active_targets, start=start_offset + 1), total=total_pkgs, initial=start_offset, desc="📦 [TỔNG] Videos", unit="gói") as main_bar:
        for pkg_idx, target in main_bar:
            pkg_name = Path(target).name if "http" in target else str(target)
            main_bar.set_postfix({"gói_hiện_tại": pkg_name})

            is_remote = isinstance(target, str) and ("http" in target or "drive.google.com" in target or (len(target) > 20 and not Path(target).exists()))

            if is_remote:
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip:
                    temp_zip_path = tmp_zip.name

                download_file(target, temp_zip_path, pkg_idx, total_pkgs)
                process_video_zip(temp_zip_path, batched_pipeline, args.beam_size, args.batch_size, video_fps_map, pkg_idx, total_pkgs, out_file, processed_videos_set)

                if os.path.exists(temp_zip_path):
                    os.remove(temp_zip_path)
            else:
                process_video_zip(target, batched_pipeline, args.beam_size, args.batch_size, video_fps_map, pkg_idx, total_pkgs, out_file, processed_videos_set)

            # Cập nhật tổng số dòng hiện có
            try:
                cur_total = len(pd.read_parquet(out_file))
                main_bar.set_postfix({"gói": pkg_name, "tổng_câu_thoại": cur_total})
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
    print(f"🎉 HOÀN TẤT ASR: Tổng cộng có {final_count:,} phân đoạn lời thoại trong {out_file}")
    print(f"⏱️ Tổng thời gian chạy: {elapsed:.0f}s ({elapsed/3600:.2f} giờ)")
    print("="*70)

if __name__ == "__main__":
    main()
