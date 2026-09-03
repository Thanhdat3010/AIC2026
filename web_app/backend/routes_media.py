import os
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse, FileResponse
from src.retrieval.keyframe_loader import KeyframeZipLoader
from src.retrieval.video_player_manager import VideoPlayerManager
from .config import PROJECT_ROOT

router = APIRouter(prefix="/api/media", tags=["media"])

# Singletons
_loader = None
_video_mgr = None

def get_loader():
    global _loader
    if _loader is None:
        _loader = KeyframeZipLoader()
    return _loader

def get_video_mgr():
    global _video_mgr
    if _video_mgr is None:
        _video_mgr = VideoPlayerManager()
    return _video_mgr

@router.get("/keyframe/{video_id}/{frame_idx}")
async def get_keyframe_image(video_id: str, frame_idx: int):
    """
    Phục vụ ảnh Keyframe JPEG trực tiếp từ file Zip với tốc độ sub-millisecond.
    Có kèm header Cache-Control để trình duyệt lưu cache cục bộ.
    """
    loader = get_loader()
    img_bytes = loader.get_image_bytes(video_id, frame_idx)
    if not img_bytes:
        # Fallback thử PIL get_image
        pil_img = loader.get_image(video_id, frame_idx)
        if pil_img:
            import io
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=85)
            img_bytes = buf.getvalue()
    
    if not img_bytes:
        raise HTTPException(status_code=404, detail=f"Keyframe not found for {video_id} @ frame {frame_idx}")
    
    return Response(
        content=img_bytes,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "public, max-age=86400", # Cache 24 giờ
            "X-Video-ID": video_id,
            "X-Frame-Idx": str(frame_idx)
        }
    )

@router.get("/keyframes_list/{video_id}")
async def get_video_keyframes_list(video_id: str):
    """
    Trả về toàn bộ danh sách frame_idx và keyframe_index của video đó để hiển thị thanh timeline.
    """
    loader = get_loader()
    df_v = loader.df_frames[loader.df_frames["video_id"] == video_id].sort_values("frame_idx")
    if df_v.empty:
        return {"video_id": video_id, "keyframes": []}
    
    records = df_v[["frame_idx", "keyframe_index"]].to_dict(orient="records")
    return {
        "video_id": video_id,
        "total_keyframes": len(records),
        "keyframes": records
    }

@router.get("/surrounding/{video_id}/{frame_idx}")
async def get_surrounding_filmstrip(video_id: str, frame_idx: int, count: int = 7):
    """
    Lấy danh sách các khung hình lân cận để render dải phim ngữ cảnh (Context Filmstrip).
    """
    loader = get_loader()
    kfs = loader.get_surrounding_keyframes(video_id, frame_idx, count=count)
    return {
        "video_id": video_id,
        "center_frame": frame_idx,
        "surrounding_frames": kfs
    }

@router.get("/video_stream/{video_id}")
async def stream_video(video_id: str, request: Request):
    """
    Phục vụ stream file video MP4 trên đĩa với hỗ trợ HTTP 206 Partial Content (Seek tua tức thì).
    """
    video_mgr = get_video_mgr()
    v_path = video_mgr.get_video_path(video_id)
    if not v_path or not v_path.exists():
        raise HTTPException(status_code=404, detail=f"MP4 Video not found for {video_id}")

    file_size = v_path.stat().st_size
    range_header = request.headers.get("range")

    if range_header:
        # Xử lý HTTP Range header
        range_val = range_header.replace("bytes=", "").strip()
        parts = range_val.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
        length = end - start + 1

        def iter_chunk():
            with open(v_path, "rb") as f:
                f.seek(start)
                bytes_left = length
                while bytes_left > 0:
                    chunk_size = min(64 * 1024, bytes_left)
                    data = f.read(chunk_size)
                    if not data:
                        break
                    bytes_left -= len(data)
                    yield data

        return StreamingResponse(
            iter_chunk(),
            status_code=206,
            media_type="video/mp4",
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(length),
                "Content-Type": "video/mp4"
            }
        )
    else:
        return FileResponse(v_path, media_type="video/mp4")
