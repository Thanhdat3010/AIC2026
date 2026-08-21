import os
import sys
import io
import re
import json
import time
import shutil
import zipfile
import argparse
from pathlib import Path
from datetime import datetime

# Force UTF-8 on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.retrieval.task_specialized_engine import TaskSpecializedEngine

def clean_video_name(video_id: str) -> str:
    """Loại bỏ phần mở rộng .mp4 nếu có theo đúng quy chuẩn BTC."""
    v = video_id.strip()
    if v.lower().endswith(".mp4"):
        v = v[:-4]
    return v

def format_qa_answer_csv(answer: str) -> str:
    """
    Format Answer cho Q&A theo đúng chuẩn CSV của BTC:
    - Độ dài tối đa 100 ký tự.
    - Escape dấu ngoặc kép đôi nếu có bên trong chuỗi.
    - Bọc trong dấu ngoặc kép an toàn.
    """
    ans = answer.strip()[:100]
    ans_escaped = ans.replace('"', '""')
    return f'"{ans_escaped}"'

def parse_input_queries(input_path: Path) -> list[dict]:
    """
    Tự động đọc gói câu hỏi từ Thư mục chứa các file .txt hoặc File .json.
    """
    queries = []

    if input_path.is_dir():
        txt_files = sorted(list(input_path.glob("*.txt")))
        print(f"📂 Tìm thấy {len(txt_files)} file truy vấn .txt trong thư mục {input_path}")
        for tf in txt_files:
            stem = tf.stem
            with open(tf, "r", encoding="utf-8") as f:
                content = f.read().strip()

            lower_stem = stem.lower()
            if "trake" in lower_stem:
                ttype = "trake"
            elif "qa" in lower_stem or "q&a" in lower_stem:
                ttype = "qa"
            else:
                ttype = "kis"

            queries.append({
                "query_id": stem,
                "task_type": ttype,
                "query_text": content
            })

    elif input_path.is_file():
        if input_path.suffix.lower() == ".json":
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "test_cases" in data:
                raw_cases = data["test_cases"]
            elif isinstance(data, list):
                raw_cases = data
            else:
                raw_cases = [data]

            for idx, c in enumerate(raw_cases, 1):
                qid = c.get("query_id", f"query-{idx}")
                ttype = c.get("task_type", "kis").lower()
                if not any(ttype in qid.lower() for ttype in ["kis", "qa", "trake"]):
                    qid = f"{qid}-{ttype}"
                
                queries.append({
                    "query_id": qid,
                    "task_type": ttype,
                    "query_text": c.get("query_text", ""),
                    "ground_truth": c.get("ground_truth", {})
                })
        elif input_path.suffix.lower() == ".txt":
            stem = input_path.stem
            with open(input_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            ttype = "trake" if "trake" in stem.lower() else ("qa" if "qa" in stem.lower() else "kis")
            queries.append({
                "query_id": stem,
                "task_type": ttype,
                "query_text": content
            })

    return queries

def get_config_params(config_code: str = "22") -> dict:
    """Trả về bộ cờ cấu hình tương thích 100% với evaluate_ablation.py."""
    cfg = {
        "config_code": config_code,
        "config_name": f"Config {config_code}",
        "engine": "siglip2",
        "use_intra_reranker": True,
        "use_neighbor": True,
        "use_cue": True,
        "use_multimodal": True,
        "use_vlm_verification": True,
        "use_dense_video_refiner": False,
        "use_rrf": True,
        "use_neighbor_expansion": True,
        "use_multi_crop": True,
        "use_multi_query": True,
        "use_event_coverage": True,
        "use_row_norm_dp": True,
        "use_segmental_dp": True,
    }
    if config_code == "0":
        cfg.update({"engine": "clip", "use_intra_reranker": False, "use_rrf": False, "use_multimodal": False, "use_segmental_dp": False})
    elif config_code == "1":
        cfg.update({"engine": "siglip2", "use_intra_reranker": False, "use_rrf": False, "use_multimodal": False, "use_segmental_dp": False})
    elif config_code == "11":
        cfg.update({"use_intra_reranker": False, "use_cue": False, "use_multimodal": False, "use_rrf": False, "use_segmental_dp": False})
    elif config_code == "12":
        cfg.update({"use_intra_reranker": True, "use_neighbor": True, "use_cue": False, "use_multimodal": False, "use_rrf": False, "use_segmental_dp": False})
    elif config_code == "14":
        cfg.update({"use_intra_reranker": True, "use_neighbor": True, "use_cue": True, "use_multimodal": True, "use_rrf": False, "use_segmental_dp": False})
    elif config_code == "20":
        cfg.update({"use_multi_query": True, "use_event_coverage": True, "use_row_norm_dp": False, "use_segmental_dp": False})
    elif config_code == "21":
        cfg.update({"use_multi_query": True, "use_event_coverage": True, "use_row_norm_dp": True, "use_segmental_dp": False})
    elif config_code == "22":
        cfg.update({"config_name": "Config 22 (Ablation 4 - Segmental DP)", "use_multi_query": True, "use_event_coverage": True, "use_row_norm_dp": True, "use_segmental_dp": True})
    elif config_code == "23":
        cfg.update({"config_name": "Config 23 (Ablation 5 - Fast Linguistic Gate)", "use_multi_query": True, "use_event_coverage": True, "use_row_norm_dp": True, "use_segmental_dp": True})
    elif config_code == "24":
        cfg.update({"config_name": "Config 24 (Ablation 6 - Entity Expansion & Cleaned Dual Index)", "use_multi_query": True, "use_event_coverage": True, "use_row_norm_dp": True, "use_segmental_dp": True})
    elif config_code == "25":
        cfg.update({"config_name": "Config 25 (Ablation 7 - WRRF)", "use_multi_query": True, "use_event_coverage": True, "use_row_norm_dp": True, "use_segmental_dp": True})
    elif config_code == "26":
        cfg.update({"config_name": "Config 26 (Master SOTA - Tiered Modality Routing Master)", "use_multi_query": True, "use_event_coverage": True, "use_row_norm_dp": True, "use_segmental_dp": True})
    return cfg

def generate_submission(input_path: Path, output_root: Path = None, top_k: int = 100, config_code: str = "22"):
    if output_root is None:
        output_root = BASE_DIR / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)

    submission_dir = output_root / "submission"
    if submission_dir.exists():
        shutil.rmtree(submission_dir)
    submission_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_config_params(config_code)

    print("=" * 85, flush=True)
    print("🚀 BẮT ĐẦU CHẠY TASK-SPECIALIZED PIPELINE SINH SUBMISSION CHUẨN 100% BTC", flush=True)
    print(f"⚙️ Cấu hình thực thi: CẤU HÌNH {config_code} ({cfg['config_name']})", flush=True)
    print(f"📂 Đề bài đầu vào: {input_path}", flush=True)
    print(f"📁 Thư mục xuất CSV: {submission_dir}", flush=True)
    print("=" * 85, flush=True)

    queries = parse_input_queries(input_path)
    if not queries:
        print(f"❌ LỖI: Không tìm thấy truy vấn nào từ {input_path}", flush=True)
        return

    # Khởi tạo TaskSpecializedEngine
    engine = TaskSpecializedEngine(engine=cfg["engine"])

    total_queries = len(queries)
    summary_report = []

    for idx, q_info in enumerate(queries, 1):
        qid = q_info["query_id"]
        ttype = q_info["task_type"]
        qtext = q_info["query_text"]

        csv_filename = f"{qid}.csv"
        csv_path = submission_dir / csv_filename

        print(f"\n[{idx}/{total_queries}] ⚡ Đang xử lý: [{ttype.upper()}] {qid}...")
        print(f"    Nội dung: {qtext[:75]}...", flush=True)

        lines = []

        if ttype == "kis":
            preds, info, lat = engine.search_kis(
                query_text=qtext,
                top_k=top_k,
                use_intra_reranker=cfg["use_intra_reranker"],
                use_neighbor=cfg["use_neighbor"],
                use_cue=cfg["use_cue"],
                use_multimodal=cfg["use_multimodal"],
                use_vlm_verification=cfg["use_vlm_verification"],
                use_dense_video_refiner=cfg["use_dense_video_refiner"],
                use_rrf=cfg["use_rrf"],
                use_neighbor_expansion=cfg["use_neighbor_expansion"]
            )
            for p in preds[:top_k]:
                v_clean = clean_video_name(p["video_id"])
                f_idx = int(p["frame_idx"])
                lines.append(f"{v_clean},{f_idx}")

        elif ttype == "qa":
            preds, info, lat = engine.search_qa(
                query_text=qtext,
                top_k=top_k,
                use_intra_reranker=cfg["use_intra_reranker"],
                use_neighbor=cfg["use_neighbor"],
                use_cue=cfg["use_cue"],
                use_multimodal=cfg["use_multimodal"],
                use_rrf=cfg["use_rrf"],
                use_multi_crop=cfg["use_multi_crop"]
            )
            default_ans = info.get("generated_qa_answer", "Không xác định")
            for p in preds[:top_k]:
                v_clean = clean_video_name(p["video_id"])
                f_idx = int(p["frame_idx"])
                cand_ans = p.get("qa_answer", default_ans)
                ans_formatted = format_qa_answer_csv(cand_ans)
                lines.append(f"{v_clean},{f_idx},{ans_formatted}")

        elif ttype == "trake":
            preds, info, lat = engine.search_trake(
                query_text=qtext,
                top_k=top_k,
                use_multi_query=cfg["use_multi_query"],
                use_event_coverage=cfg["use_event_coverage"],
                use_row_norm_dp=cfg["use_row_norm_dp"],
                use_segmental_dp=cfg["use_segmental_dp"]
            )
            for p in preds[:top_k]:
                v_clean = clean_video_name(p["video_id"])
                event_frames = p.get("event_frames", [p["frame_idx"]])
                frames_str = ",".join(str(int(f)) for f in event_frames)
                lines.append(f"{v_clean},{frames_str}")

        # Ghi file CSV thuần túy (Encoding UTF-8, Không Header, Phân cách bằng dấu phẩy)
        with open(csv_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")

        print(f"    ✅ Đã ghi {len(lines)} dòng vào: {submission_dir.name}/{csv_filename}")
        summary_report.append({
            "query_id": qid,
            "task": ttype.upper(),
            "csv_file": csv_filename,
            "lines_count": len(lines),
            "sample_line": lines[0] if lines else "EMPTY"
        })

    # Đóng gói ZIP chuẩn Codabench
    zip_path = output_root / "submission.zip"
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for csv_file in submission_dir.glob("*.csv"):
            arcname = f"submission/{csv_file.name}"
            z.write(csv_file, arcname=arcname)

    print("\n" + "=" * 85)
    print("🏆 HOÀN TẤT QUÁ TRÌNH SINH KẾT QUẢ VÀ ĐÓNG GÓI SUBMISSION!")
    print(f"📦 File Zip sẵn sàng nộp lên Codabench: {zip_path}")
    print("=" * 85)

    print("\n📋 BẢNG KIỂM TRA ĐỊNH DẠNG (FORMAT VERIFICATION TABLE):")
    print("-" * 85)
    print(f"{'Query ID':<20} | {'Task':<7} | {'Số dòng':<8} | {'Mẫu dòng đầu tiên (Sample Row)'}")
    print("-" * 85)
    for s in summary_report:
        print(f"{s['query_id']:<20} | {s['task']:<7} | {s['lines_count']:<8} | {s['sample_line']}")
    print("-" * 85 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Tool sinh kết quả nộp bài chuẩn 100% BTC AIC 2026")
    parser.add_argument("--input", type=str, default="query/THUNGHIEM-bo-de-thi", help="Thư mục chứa file .txt hoặc đường dẫn file .json")
    parser.add_argument("--output_dir", type=str, default="output/thunghiem", help="Thư mục lưu submission và file zip (mặc định 'output/thunghiem')")
    parser.add_argument("--config", type=str, default="25", help="Mã cấu hình muốn chạy (ví dụ: 22, 23, 24, 25, 26). Mặc định là 25.")
    parser.add_argument("--top_k", type=int, default=100, help="Số lượng kết quả dự đoán tối đa cho mỗi query (mặc định 100)")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = BASE_DIR / in_path

    out_root = Path(args.output_dir)
    if not out_root.is_absolute():
        out_root = BASE_DIR / out_root

    generate_submission(input_path=in_path, output_root=out_root, top_k=args.top_k, config_code=args.config)

if __name__ == "__main__":
    main()
