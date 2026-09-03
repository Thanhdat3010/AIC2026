import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Đảm bảo đường dẫn gốc nằm trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .config import STATIC_DIR
from .routes_search import router as router_search, get_engine
from .routes_media import router as router_media
from .routes_submission import router as router_submission
from .websocket_hub import hub

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi tạo nạp model nền
    print("🚀 [AIC 2026 WEB PLATFORM] Đang khởi động máy chủ API...", flush=True)
    try:
        # Pre-warm search engine
        print("[*] Pre-warming A8 SOTA Search Engine...", flush=True)
        get_engine()
        print("✅ [AIC 2026 WEB PLATFORM] Hệ thống sẵn sàng phục vụ thi đấu!", flush=True)
    except Exception as e:
        print(f"⚠️ Cảnh báo khởi tạo: {e}", flush=True)
    yield
    print("🛑 [AIC 2026 WEB PLATFORM] Máy chủ đã dừng.", flush=True)

app = FastAPI(
    title="AIC 2026 Championship Multi-Modal Platform",
    description="Hệ thống Tìm kiếm Sự kiện Video Đa phương thức & Phòng Thi Đấu Đa Người Dùng",
    version="2.0",
    lifespan=lifespan
)

# Cấu hình CORS mở để đồng đội trên mạng LAN hoặc Cloudflare Tunnel truy cập tự do
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký các Router API
app.include_router(router_search)
app.include_router(router_media)
app.include_router(router_submission)

# WebSocket Endpoint cho Phòng thi đấu Real-Time
@app.websocket("/ws/collaborate")
async def websocket_collaborate_endpoint(websocket: WebSocket, room: str = "default"):
    await hub.connect(websocket, room_id=room)
    try:
        while True:
            data = await websocket.receive_json()
            await hub.handle_message(websocket, room_id=room, data=data)
    except WebSocketDisconnect:
        hub.disconnect(websocket, room_id=room)
        await hub.broadcast(room, {
            "type": "member_left",
            "active_members_count": len(hub.active_rooms.get(room, []))
        })
    except Exception:
        hub.disconnect(websocket, room_id=room)

# Mount thư mục tĩnh
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "AIC 2026 Web Platform Backend is running. Please add index.html in static/"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_app.backend.main:app", host="0.0.0.0", port=8000, reload=False)
