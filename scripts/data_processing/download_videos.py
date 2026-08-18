import os
import sys
import time
from pathlib import Path
import requests
from tqdm import tqdm

# Force UTF-8 stdout/stderr on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

def download_file(url: str, dest_path: Path, max_retries: int = 10):
    """Tải file video với thanh tiến trình tqdm, hỗ trợ resume khi bị đứt mạng."""
    filename = dest_path.name
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    
    # Kiểm tra kích thước file gốc từ server
    total_size = 0
    try:
        head_res = requests.head(url, timeout=20, allow_redirects=True)
        total_size = int(head_res.headers.get('content-length', 0))
    except Exception as e:
        print(f"[THÔNG BÁO] Không lấy được Content-Length trước: {e}", flush=True)

    # Nếu file đích đã tồn tại và đủ dung lượng
    if dest_path.exists():
        local_size = dest_path.stat().st_size
        if total_size > 0 and local_size == total_size:
            print(f"[ĐÃ CÓ SẴN 100%] {filename} ({local_size / (1024**3):.2f} GB) -> Bỏ qua!", flush=True)
            return True
        elif total_size == 0 and local_size > 500 * 1024 * 1024:
            print(f"[ĐÃ CÓ SẴN] {filename} ({local_size / (1024**3):.2f} GB) -> Bỏ qua!", flush=True)
            return True

    # Hỗ trợ tải tiếp (Resume)
    initial_bytes = 0
    headers = {}
    if temp_path.exists():
        initial_bytes = temp_path.stat().st_size
        if total_size > 0 and initial_bytes < total_size:
            headers['Range'] = f"bytes={initial_bytes}-"
            print(f"[TẢI TIẾP] {filename} từ {initial_bytes / (1024**2):.1f} MB / {total_size / (1024**3):.2f} GB...", flush=True)
        elif total_size > 0 and initial_bytes >= total_size:
            if dest_path.exists():
                dest_path.unlink()
            temp_path.rename(dest_path)
            print(f"[HOÀN TẤT] {filename}", flush=True)
            return True

    for attempt in range(1, max_retries + 1):
        try:
            mode = 'ab' if initial_bytes > 0 else 'wb'
            res = requests.get(url, headers=headers, stream=True, timeout=30)
            res.raise_for_status()

            pbar_total = total_size if total_size > 0 else None
            desc = f"Tải {filename}"
            with open(temp_path, mode) as f, tqdm(
                total=pbar_total,
                initial=initial_bytes,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=desc,
                ncols=90,
                ascii=" >="
            ) as pbar:
                for chunk in res.iter_content(chunk_size=2 * 1024 * 1024):  # 2MB chunk
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            # Hoàn tất -> đổi tên file .part thành .zip
            if dest_path.exists():
                dest_path.unlink()
            temp_path.rename(dest_path)
            print(f"\n[THÀNH CÔNG 100%] {filename} ({dest_path.stat().st_size / (1024**3):.2f} GB)", flush=True)
            return True

        except Exception as e:
            print(f"\n[CẢNH BÁO] Lỗi khi tải {filename} (Thử lần {attempt}/{max_retries}): {e}", flush=True)
            if attempt < max_retries:
                time.sleep(3)
                if temp_path.exists():
                    initial_bytes = temp_path.stat().st_size
                    headers['Range'] = f"bytes={initial_bytes}-"
            else:
                print(f"[THẤT BÀI] Không thể tải {filename} sau {max_retries} lần thử!", flush=True)
                return False

def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    urls_file = base_dir / "config" / "drive_videos_urls.txt"
    dest_dir = base_dir / "raw" / "batch_1" / "Videos"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not urls_file.exists():
        print(f"[LỖI] Không tìm thấy file {urls_file}", flush=True)
        return

    with open(urls_file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    total_pkgs = len(urls)
    print("=" * 80, flush=True)
    print(f"🚀 BẮT ĐẦU TẢI {total_pkgs} GÓI RAW VIDEOS VÀO: {dest_dir}", flush=True)
    print("=" * 80, flush=True)

    # Dọn dẹp các file rác .crdownload chưa hoàn tất nếu có
    for cr in dest_dir.glob("*.crdownload"):
        try:
            cr.unlink()
        except Exception:
            pass

    for idx, url in enumerate(urls, 1):
        filename = url.split("/")[-1]
        dest_file = dest_dir / filename
        print(f"\n[{idx}/{total_pkgs}] ⚡ Đang xử lý: {filename}...", flush=True)
        success = download_file(url, dest_file)
        if not success:
            print(f"[CẢNH BÁO] Gói {filename} tải chưa thành công, sẽ thử tiếp gói sau.", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("🎉 ĐÃ HOÀN TẤT TẢI TOÀN BỘ CÁC GÓI RAW VIDEOS BATCH 1!", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
