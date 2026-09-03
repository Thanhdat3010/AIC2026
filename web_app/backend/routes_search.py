import time
import re
from typing import Optional, Literal
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

from src.retrieval.unified_search_core import UnifiedSearchCore
from src.query.llm_query_refiner import LLMQueryRefiner
from src.tasks.clean_task_handlers import KISHandler, QAHandler, TRAKEHandler

router = APIRouter(prefix="/api/search", tags=["search"])

# Shared Singleton Instances
_search_core = None
_refiner = None
_kis_handler = None
_qa_handler = None
_trake_handler = None

def get_engine():
    global _search_core, _refiner, _kis_handler, _qa_handler, _trake_handler
    if _search_core is None:
        _search_core = UnifiedSearchCore(engine="siglip2", batch="batch_1")
        _refiner = LLMQueryRefiner()
        _kis_handler = KISHandler(_search_core, _refiner)
        _qa_handler = QAHandler(_search_core, _refiner)
        _trake_handler = TRAKEHandler(_search_core, _refiner)
    return _search_core, _refiner, _kis_handler, _qa_handler, _trake_handler

class SearchRequest(BaseModel):
    query: str
    task_type: Optional[Literal["auto", "kis", "qa", "trake"]] = "auto"
    top_k: int = 100
    config_name: str = "A8_SOTA"

def detect_task_type(query_text: str) -> str:
    text_lower = query_text.lower()
    
    # 1. Nhận diện TRAKE
    trake_signals = [
        "thứ tự thời gian", "theo thứ tự", "các phân cảnh liên quan",
        "liệt kê theo", "e1:", "e2:", "sự kiện 1", "khoảnh khắc",
        "phân cảnh 1", "giai đoạn 1", "đầu tiên khi", "sau đó", "cuối cùng"
    ]
    if any(s in text_lower for s in trake_signals) and (":" in query_text or "phân cảnh" in text_lower):
        return "trake"
        
    # 2. Nhận diện Visual QA
    qa_signals = [
        "màu gì", "chữ gì", "số mấy", "bao nhiêu", "ai đang", "đeo gì",
        "cầm gì", "mặc gì", "đội gì", "có chữ", "biển số", "là gì?", "ở đâu?"
    ]
    if any(s in text_lower for s in qa_signals) or "?" in query_text:
        return "qa"
        
    return "kis"

@router.post("/auto")
async def unified_search(req: SearchRequest):
    """
    Điểm truy cập tìm kiếm duy nhất, tự động nhận diện KIS/QA/TRAKE và chạy bằng cấu hình tối thượng A8_SOTA.
    """
    q = req.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query text is empty.")

    task = req.task_type
    if task == "auto" or not task:
        task = detect_task_type(q)

    search_core, refiner, kis_handler, qa_handler, trake_handler = get_engine()

    t0 = time.perf_counter()
    if task == "qa":
        preds, info, lat = qa_handler.search(q, top_k=req.top_k, config_name=req.config_name)
    elif task == "trake":
        preds, info, lat = trake_handler.search(q, top_k=req.top_k, config_name=req.config_name)
    else:
        preds, info, lat = kis_handler.search(q, top_k=req.top_k, config_name=req.config_name)
    total_latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "status": "success",
        "task_type": task,
        "config_name": req.config_name,
        "total_results": len(preds),
        "latency_ms": round(total_latency_ms, 1),
        "info": info,
        "results": preds
    }

@router.post("/kis")
async def search_kis_explicit(req: SearchRequest):
    _, _, kis_handler, _, _ = get_engine()
    t0 = time.perf_counter()
    preds, info, lat = kis_handler.search(req.query, top_k=req.top_k, config_name=req.config_name)
    total_lat = (time.perf_counter() - t0) * 1000
    return {
        "status": "success",
        "task_type": "kis",
        "total_results": len(preds),
        "latency_ms": round(total_lat, 1),
        "info": info,
        "results": preds
    }

@router.post("/qa")
async def search_qa_explicit(req: SearchRequest):
    _, _, _, qa_handler, _ = get_engine()
    t0 = time.perf_counter()
    preds, info, lat = qa_handler.search(req.query, top_k=req.top_k, config_name=req.config_name)
    total_lat = (time.perf_counter() - t0) * 1000
    return {
        "status": "success",
        "task_type": "qa",
        "total_results": len(preds),
        "latency_ms": round(total_lat, 1),
        "info": info,
        "results": preds
    }

@router.post("/trake")
async def search_trake_explicit(req: SearchRequest):
    _, _, _, _, trake_handler = get_engine()
    t0 = time.perf_counter()
    preds, info, lat = trake_handler.search(req.query, top_k=req.top_k, config_name=req.config_name)
    total_lat = (time.perf_counter() - t0) * 1000
    return {
        "status": "success",
        "task_type": "trake",
        "total_results": len(preds),
        "latency_ms": round(total_lat, 1),
        "info": info,
        "results": preds
    }
