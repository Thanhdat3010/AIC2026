import os
import sys
import io
import time
from pathlib import Path
import requests
from tqdm import tqdm

# Force UTF-8 stdout/stderr on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

def download_file(url: str, dest_path: Path, max_retries: int = 5):
    """Tai file voi thanh tien trinh tqdm, ho tro resume neu bi dut quang."""
    filename = dest_path.name
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    
    # Kiem tra kich thuoc file goc tu server
    total_size = 0
    try:
        head_res = requests.head(url, timeout=15, allow_redirects=True)
        total_size = int(head_res.headers.get('content-length', 0))
    except Exception:
        pass

    # Neu file dich da ton tai va du dung luong
    if dest_path.exists():
        local_size = dest_path.stat().st_size
        if total_size > 0 and local_size == total_size:
            print(f"[DA CO SAN] {filename} ({local_size / (1024**3):.2f} GB) -> Bo qua!", flush=True)
            return True
        elif total_size == 0 and local_size > 500 * 1024 * 1024:
            print(f"[DA CO SAN] {filename} ({local_size / (1024**3):.2f} GB) -> Bo qua!", flush=True)
            return True

    # Ho tro tai tiep (Resume)
    initial_bytes = 0
    headers = {}
    if temp_path.exists():
        initial_bytes = temp_path.stat().st_size
        if total_size > 0 and initial_bytes < total_size:
            headers['Range'] = f"bytes={initial_bytes}-"
            print(f"[TAI TIEP] {filename} tu {initial_bytes / (1024**2):.1f} MB...", flush=True)
        elif total_size > 0 and initial_bytes >= total_size:
            temp_path.rename(dest_path)
            print(f"[HOAN TAT] {filename}", flush=True)
            return True

    for attempt in range(1, max_retries + 1):
        try:
            mode = 'ab' if initial_bytes > 0 else 'wb'
            res = requests.get(url, headers=headers, stream=True, timeout=30)
            res.raise_for_status()

            pbar_total = total_size if total_size > 0 else None
            desc = f"Tai {filename}"
            with open(temp_path, mode) as f, tqdm(
                total=pbar_total,
                initial=initial_bytes,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=desc,
                ncols=85,
                ascii=" >="
            ) as pbar:
                for chunk in res.iter_content(chunk_size=1024 * 1024): # 1MB chunk
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            # Hoan tat -> doi ten file .tmp thanh .zip
            if dest_path.exists():
                dest_path.unlink()
            temp_path.rename(dest_path)
            print(f"\n[THANH CONG 100%] {filename} ({dest_path.stat().st_size / (1024**3):.2f} GB)", flush=True)
            return True

        except Exception as e:
            print(f"\n[CANH BAO] Loi khi tai {filename} (Thu lan {attempt}/{max_retries}): {e}", flush=True)
            if attempt < max_retries:
                time.sleep(3)
                if temp_path.exists():
                    initial_bytes = temp_path.stat().st_size
                    headers['Range'] = f"bytes={initial_bytes}-"
            else:
                print(f"[THAT BAI] Khong the tai {filename} sau {max_retries} lan thu!", flush=True)
                return False

def main():
    base_dir = Path(__file__).resolve().parent.parent
    urls_file = base_dir / "config" / "drive_keyframes_urls.txt"
    dest_dir = base_dir / "raw" / "batch_1" / "Keyframes"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not urls_file.exists():
        print(f"[LOI] Khong tim thay file {urls_file}", flush=True)
        return

    with open(urls_file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    total_pkgs = len(urls)
    print("=" * 70, flush=True)
    print(f"[*] BAT DAU TAI {total_pkgs} GOI KEYFRAMES VAO: {dest_dir}", flush=True)
    print("=" * 70, flush=True)

    for idx, url in enumerate(urls, 1):
        filename = url.split("/")[-1]
        dest_file = dest_dir / filename
        print(f"\n[{idx}/{total_pkgs}] Dang xu ly: {filename}", flush=True)
        success = download_file(url, dest_file)
        if not success:
            print(f"[CANH BAO] Goi {filename} tai chua thanh cong.", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("[*] DA HOAN TAT TAI TOAN BO CAC GOI KEYFRAMES!", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
