import json
import zipfile
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from .config import OUTPUT_DIR, QUERY_DIR, PROJECT_ROOT, DATA_DIR

router = APIRouter(prefix="/api/contest", tags=["contest"])

# Bộ nhớ đệm lịch sử hoàn tác (Undo History)
_undo_history: dict[str, list] = {}

def sync_submission_zip(csv_dir: Path, zip_dest: Path):
    """Tự động đóng gói và đồng bộ file submission.zip chuẩn 100% BTC."""
    if not csv_dir.exists():
        return
    zip_dest.parent.mkdir(parents=True, exist_ok=True)
    temp_zip = zip_dest.parent / f"{zip_dest.stem}_temp.zip"
    try:
        with zipfile.ZipFile(temp_zip, "w", zipfile.ZIP_DEFLATED) as z:
            for csv_f in sorted(list(csv_dir.glob("*.csv"))):
                z.write(csv_f, arcname=f"submission/{csv_f.name}")
        if temp_zip.exists():
            if zip_dest.exists():
                zip_dest.unlink(missing_ok=True)
            temp_zip.rename(zip_dest)
    except Exception as e:
        print(f"⚠️ Lỗi đồng bộ zip: {e}", flush=True)

@router.get("/packages")
async def list_contest_packages():
    """
    Quét danh sách các gói đề bài trong query/ và gói kết quả trong output/.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    avail_outputs = [p.name for p in OUTPUT_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not avail_outputs:
        avail_outputs = ["thunghiem", "sotuyen1"]
        for d in avail_outputs:
            (OUTPUT_DIR / d / "submission").mkdir(parents=True, exist_ok=True)

    QUERY_DIR.mkdir(parents=True, exist_ok=True)
    avail_queries = [p.name for p in QUERY_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")]

    return {
        "output_packages": avail_outputs,
        "query_packages": avail_queries,
        "default_output": "sotuyen1" if "sotuyen1" in avail_outputs else avail_outputs[0],
        "default_query": "SOTUYEN1-bo-de-thi" if "SOTUYEN1-bo-de-thi" in avail_queries else (avail_queries[0] if avail_queries else "None")
    }

@router.get("/queries")
async def list_queries(query_package: str, output_package: str):
    """
    Liệt kê danh sách các câu hỏi của gói đề, kèm trạng thái đã làm hay chưa.
    """
    q_dir = QUERY_DIR / query_package
    out_dir = OUTPUT_DIR / output_package / "submission"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_csvs = {p.stem: p for p in out_dir.glob("*.csv")}
    query_files = sorted(list(q_dir.glob("*.txt"))) if q_dir.exists() else []

    queries = []
    if query_files:
        for q_path in query_files:
            stem = q_path.stem
            task_tag = "QA" if "qa" in stem.lower() else ("TRAKE" if "trake" in stem.lower() else "KIS")
            with open(q_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            queries.append({
                "id": stem,
                "filename": q_path.name,
                "task_type": task_tag,
                "content": content,
                "is_completed": (stem in existing_csvs),
                "csv_path": str(existing_csvs[stem]) if stem in existing_csvs else None
            })
    else:
        # Fallback từ các file csv đã có sẵn
        for stem, csv_p in existing_csvs.items():
            task_tag = "QA" if "qa" in stem.lower() else ("TRAKE" if "trake" in stem.lower() else "KIS")
            queries.append({
                "id": stem,
                "filename": csv_p.name,
                "task_type": task_tag,
                "content": f"Truy vấn từ kết quả {stem}",
                "is_completed": True,
                "csv_path": str(csv_p)
            })

    total = len(queries)
    completed = sum(1 for q in queries if q["is_completed"])
    return {
        "query_package": query_package,
        "output_package": output_package,
        "total": total,
        "completed": completed,
        "queries": queries
    }

@router.get("/submission_data")
async def get_submission_data(output_package: str, query_id: str):
    """
    Lấy dữ liệu CSV hiện tại của một câu hỏi.
    """
    csv_path = OUTPUT_DIR / output_package / "submission" / f"{query_id}.csv"
    if not csv_path.exists():
        return {"query_id": query_id, "exists": False, "rows": []}

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            l = line.strip()
            if l:
                rows.append(l.split(","))

    return {"query_id": query_id, "exists": True, "rows": rows}

class SaveSubmissionRequest(BaseModel):
    output_package: str
    query_id: str
    task_type: str
    rows: list # List of predictions or raw CSV rows

@router.post("/save")
async def save_submission(req: SaveSubmissionRequest):
    """
    Lưu kết quả nộp bài cho câu hỏi và tự động đồng bộ file submission.zip.
    """
    out_dir = OUTPUT_DIR / req.output_package / "submission"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{req.query_id}.csv"
    zip_dest = OUTPUT_DIR / req.output_package / "submission.zip"

    # Lưu snapshot vào undo history
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            old_lines = f.read()
        _undo_history.setdefault(req.query_id, []).append(old_lines)
        _undo_history[req.query_id] = _undo_history[req.query_id][-10:]

    with open(csv_path, "w", encoding="utf-8") as f:
        for r in req.rows:
            if isinstance(r, dict):
                v = r.get("video_id", "")
                if req.task_type.lower() == "qa":
                    ans = r.get("answer", "")
                    ans_clean = f'"{ans}"' if ans else '""'
                    f.write(f"{v},{r.get('frame_idx', 0)},{ans_clean}\n")
                elif req.task_type.lower() == "trake" and "event_frames" in r:
                    ev_str = ",".join([str(x) for x in r["event_frames"]])
                    f.write(f"{v},{ev_str}\n")
                else:
                    f.write(f"{v},{r.get('frame_idx', 0)}\n")
            elif isinstance(r, list):
                f.write(",".join([str(x) for x in r]) + "\n")
            elif isinstance(r, str):
                f.write(r.strip() + "\n")

    sync_submission_zip(out_dir, zip_dest)

    return {
        "status": "success",
        "query_id": req.query_id,
        "saved_rows": len(req.rows),
        "zip_synced": str(zip_dest.name)
    }

class OverrideRank1Request(BaseModel):
    output_package: str
    query_id: str
    task_type: str
    video_id: str
    frame_idx: int
    qa_answer: Optional[str] = ""
    trake_frames: Optional[str] = ""

@router.post("/override_rank1")
async def override_rank1(req: OverrideRank1Request):
    """
    Ghim thủ công Video ID và Frame Index lên vị trí Rank #1 ngay lập tức.
    """
    out_dir = OUTPUT_DIR / req.output_package / "submission"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{req.query_id}.csv"
    zip_dest = OUTPUT_DIR / req.output_package / "submission.zip"

    existing_lines = []
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            existing_lines = [l.strip() for l in f if l.strip()]
        _undo_history.setdefault(req.query_id, []).append("\n".join(existing_lines))

    # Chuẩn bị dòng Rank 1
    t = req.task_type.lower()
    if t == "qa":
        ans = f'"{req.qa_answer.strip()}"' if req.qa_answer else '""'
        new_row = f"{req.video_id.strip()},{req.frame_idx},{ans}"
    elif t == "trake" and req.trake_frames:
        new_row = f"{req.video_id.strip()},{req.trake_frames.strip()}"
    else:
        new_row = f"{req.video_id.strip()},{req.frame_idx}"

    updated_lines = [new_row] + [l for l in existing_lines if l != new_row][:99]

    with open(csv_path, "w", encoding="utf-8") as f:
        for l in updated_lines:
            f.write(l + "\n")

    sync_submission_zip(out_dir, zip_dest)
    return {"status": "success", "query_id": req.query_id, "rank1_set": new_row}

@router.post("/undo")
async def undo_submission(output_package: str, query_id: str):
    """
    Hoàn tác thao tác chỉnh sửa gần nhất.
    """
    history = _undo_history.get(query_id, [])
    if not history:
        raise HTTPException(status_code=400, detail="Không có lịch sử hoàn tác cho câu hỏi này.")

    prev_state = history.pop()
    csv_path = OUTPUT_DIR / output_package / "submission" / f"{query_id}.csv"
    zip_dest = OUTPUT_DIR / output_package / "submission.zip"

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(prev_state)

    sync_submission_zip(csv_path.parent, zip_dest)
    return {"status": "success", "message": "Đã hoàn tác thành công về trạng thái trước đó."}

@router.get("/download_zip")
async def download_zip(output_package: str):
    """
    Tải file submission.zip sẵn sàng nộp cho Ban Giám Khảo.
    """
    zip_path = OUTPUT_DIR / output_package / "submission.zip"
    if not zip_path.exists():
        out_dir = OUTPUT_DIR / output_package / "submission"
        sync_submission_zip(out_dir, zip_path)

    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="Chưa có file submission.zip nào.")

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"submission_{output_package}.zip"
    )

@router.get("/leaderboard")
async def get_leaderboard_data():
    """
    Trả về dữ liệu tổng kết Ablation Study phục vụ tab Leaderboard.
    """
    ablation_path = DATA_DIR / "benchmark" / "ablation_study_summary.json"
    if not ablation_path.exists():
        return {"leaderboard": []}
    with open(ablation_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {"leaderboard": data}
