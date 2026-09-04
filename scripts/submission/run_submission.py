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

# Force UTF-8 on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.tasks.clean_task_handlers import MasterPipelineRunner
from src.submission.submission_validator import SubmissionValidator

def clean_video_name(video_id: str) -> str:
    """Loại bỏ phần mở rộng .mp4 nếu có theo đúng quy chuẩn BTC."""
    v = video_id.strip()
    if v.lower().endswith(".mp4"):
        v = v[:-4]
    return v

def format_qa_answer_csv(answer: str) -> str:
    """
    Format Answer cho Q&A theo đúng chuẩn CSV của BTC:
    - Độ dài tối đa 100 ký tự (cắt tối đa 90 ký tự an toàn).
    - Escape dấu ngoặc kép đôi nếu có bên trong chuỗi.
    - Bọc trong dấu ngoặc kép an toàn.
    """
    ans = answer.strip()[:90]
    ans_escaped = ans.replace('"', '""')
    return f'"{ans_escaped}"'

def parse_input_queries(input_path: Path) -> list[dict]:
    """
    Tự động đọc gói câu hỏi từ Thư mục chứa các file .txt hoặc File .json.
    """
    queries = []

    if input_path.is_dir():
        # Sắp xếp tự nhiên (Natural sort: 1, 2, 3... thay vì 1, 10, 11...)
        txt_files = sorted(
            list(input_path.glob("*.txt")),
            key=lambda p: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', p.stem)]
        )
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

def generate_submission(input_path: Path, output_root: Path = None, top_k: int = 100, config_code: str = "A7"):
    if output_root is None:
        output_root = BASE_DIR / "output" / "submission"
    output_root.mkdir(parents=True, exist_ok=True)

    submission_dir = output_root / "submission"
    if submission_dir.exists():
        shutil.rmtree(submission_dir)
    submission_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85, flush=True)
    print("🚀 BẮT ĐẦU CHẠY UNIFIED SOTA PIPELINE SINH SUBMISSION CHUẨN 100% BTC", flush=True)
    print(f"⚙️ Cấu hình thực thi: CẤU HÌNH [{config_code}]", flush=True)
    print(f"📂 Đề bài đầu vào: {input_path}", flush=True)
    print(f"📁 Thư mục xuất CSV: {submission_dir}", flush=True)
    print("=" * 85, flush=True)

    queries = parse_input_queries(input_path)
    if not queries:
        print(f"❌ LỖI: Không tìm thấy truy vấn nào từ {input_path}", flush=True)
        return

    # Khởi tạo MasterPipelineRunner (Single Source of Truth)
    runner = MasterPipelineRunner(engine="siglip2", batch="batch_1")

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

        preds, info, lat = runner.run_query(
            query_text=qtext,
            task_type=ttype,
            config_name=config_code,
            top_k=top_k
        )

        lines = []
        if ttype == "kis":
            for p in preds[:top_k]:
                v_clean = clean_video_name(p["video_id"])
                f_idx = int(p["frame_idx"])
                lines.append(f"{v_clean},{f_idx}")

        elif ttype == "qa":
            default_ans = info.get("vlm_answer", preds[0].get("answer", "Không xác định") if preds else "Không xác định")
            for p in preds[:top_k]:
                v_clean = clean_video_name(p["video_id"])
                f_idx = int(p["frame_idx"])
                cand_ans = p.get("qa_answer", p.get("answer", default_ans))
                ans_formatted = format_qa_answer_csv(cand_ans)
                lines.append(f"{v_clean},{f_idx},{ans_formatted}")

        elif ttype == "trake":
            for p in preds[:top_k]:
                v_clean = clean_video_name(p["video_id"])
                event_frames = p.get("event_frames", p.get("events", [p.get("frame_idx", 0)]))
                frames_str = ",".join(str(int(f)) for f in event_frames)
                lines.append(f"{v_clean},{frames_str}")

        # Ghi file CSV thuần túy (Encoding UTF-8, Không Header, Phân cách bằng dấu phẩy)
        with open(csv_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")

        print(f"    ✅ Đã ghi {len(lines)} dòng vào: {submission_dir.name}/{csv_filename} ({lat:.0f} ms)")
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
        for csv_file in sorted(list(submission_dir.glob("*.csv"))):
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

    validator = SubmissionValidator()
    val_res = validator.validate_directory(submission_dir)
    if val_res.get("all_valid", False):
        print(f"🎉 VALIDATION PASSED: Toàn bộ {val_res['total_files']} file CSV đều HỢP LỆ 100% chuẩn quy chế BTC!\n")
    else:
        print(f"⚠️ CẢNH BÁO VALIDATION: {val_res.get('error')}\n")

def main():
    parser = argparse.ArgumentParser(description="Tool sinh kết quả nộp bài chuẩn 100% BTC AIC 2026")
    parser.add_argument("--input", type=str, default="query/THUNGHIEM-bo-de-thi", help="Thư mục chứa file .txt hoặc đường dẫn file .json")
    parser.add_argument("--output_dir", type=str, default="output/thunghiem", help="Thư mục lưu submission và file zip (mặc định 'output/thunghiem')")
    parser.add_argument("--config", type=str, default="A10_FINAL", help="Mã cấu hình muốn chạy (mặc định A10_FINAL - SOTA 0.6064).")
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

