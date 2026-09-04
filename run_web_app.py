import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import time
import socket
import webbrowser
import threading
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def open_browser_delayed(url, delay=2.0):
    time.sleep(delay)
    print(f"🚀 [AUTO-LAUNCH] Đang mở trình duyệt tại: {url}", flush=True)
    webbrowser.open(url)

def main():
    local_ip = get_local_ip()
    port = 8000
    local_url = f"http://localhost:{port}"
    lan_url = f"http://local_ip:{port}" if local_ip != "127.0.0.1" else local_url

    print("=" * 80)
    print("🏆 AIC 2026 SOTA CHAMPIONSHIP WEB PLATFORM")
    print("Lõi Thuật Toán: A8_SOTA (Macro: 0.7250 | Video-R@1: 78.1%)")
    print("-" * 80)
    print(f"👉 Trình duyệt của bạn:     {local_url}")
    print(f"👉 Đường dẫn cho Đồng đội:  http://{local_ip}:{port} (Cùng mạng Wi-Fi/LAN)")
    print(f"🛡️  Dự phòng Streamlit cũ:   streamlit run app/streamlit_app.py")
    print("=" * 80, flush=True)

    # Khởi động tiểu trình mở trình duyệt sau 2 giây
    threading.Thread(target=open_browser_delayed, args=(local_url,), daemon=True).start()

    import uvicorn
    uvicorn.run("web_app.backend.main:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()
