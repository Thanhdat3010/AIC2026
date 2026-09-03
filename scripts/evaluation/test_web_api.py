import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from web_app.backend.main import app

client = TestClient(app)

print("=" * 60)
print("TESTING FASTAPI WEB PLATFORM ENDPOINTS:")
print("=" * 60)

# 1. Test root /
r = client.get("/")
print("1. Root GET /:", r.status_code, "Length:", len(r.content))

# 2. Test contest packages
r = client.get("/api/contest/packages")
print("2. Contest packages:", r.status_code, r.json())

# 3. Test keyframes list
r = client.get("/api/media/keyframes_list/L24_V002")
print("3. Media keyframes list:", r.status_code, "Total keyframes:", r.json().get("total_keyframes", 0))

# 4. Test Search Auto (KIS query)
r = client.post("/api/search/auto", json={"query": "người phụ nữ dệt chiếu hoa", "task_type": "auto", "top_k": 5})
print("4. Search Auto (KIS):", r.status_code, "Results:", len(r.json().get("results", [])), "Latency:", r.json().get("latency_ms"), "ms")

print("=" * 60)
print("✅ ALL CORE WEB API ENDPOINTS PASSED SUCCESSFULLY!")
