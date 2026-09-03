from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = PROJECT_ROOT / "web_app" / "static"
OUTPUT_DIR = PROJECT_ROOT / "output"
QUERY_DIR = PROJECT_ROOT / "query"
DATA_DIR = PROJECT_ROOT / "data"

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
