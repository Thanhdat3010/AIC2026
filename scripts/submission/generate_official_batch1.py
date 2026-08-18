import os
import sys
import time
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.retrieval.task_specialized_engine import TaskSpecializedEngine
from src.submission.submission_validator import SubmissionValidator

def run_official_batch():
    print("=" * 80)
    print("🚀 BẮT ĐẦU CHẠY TỰ ĐỘNG FULL 24 CÂU ĐỀ THI BTC BATCH 1 (SOTA GPU PIPELINE)...")
    print("=" * 80)
    
    engine = TaskSpecializedEngine(engine="siglip2", batch="batch_1")
    validator = SubmissionValidator()
    
    query_dir = BASE_DIR / "query" / "batch_1" / "query-p1-groupA"
    output_dir = BASE_DIR / "output" / "batch_1"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    query_files = sorted(list(query_dir.glob("*.txt")))
    print(f"[*] Tìm thấy {len(query_files)} câu hỏi trong {query_dir.name}\n")
    
    for idx, q_path in enumerate(query_files, 1):
        with open(q_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            
        q_name = q_path.name
        is_qa = "qa" in q_name.lower()
        is_trake = "trake" in q_name.lower()
        
        t0 = time.time()
        if is_qa:
            preds, info, _ = engine.search_qa(content, top_k=100, use_intra_reranker=True, use_cue=True, use_multimodal=True)
            ttype = "QA"
        elif is_trake:
            preds, info, _ = engine.search_trake(content, top_k=100)
            ttype = "TRAKE"
        else:
            preds, info, _ = engine.search_kis(content, top_k=100, use_intra_reranker=True, use_dense_video_refiner=False)
            ttype = "KIS"
            
        latency = (time.time() - t0) * 1000
        
        out_csv = output_dir / f"{q_path.stem}.csv"
        with open(out_csv, "w", encoding="utf-8") as f:
            for p in preds:
                if is_qa:
                    ans = p.get("answer", info.get("generated_qa_answer", ""))
                    ans_clean = f'"{ans}"' if ans else '""'
                    f.write(f"{p['video_id']},{p['frame_idx']},{ans_clean}\n")
                elif is_trake and "event_frames" in p:
                    ev_str = ",".join([str(x) for x in p["event_frames"]])
                    f.write(f"{p['video_id']},{ev_str}\n")
                else:
                    f.write(f"{p['video_id']},{p['frame_idx']}\n")
                    
        top1_str = f"{preds[0]['video_id']}:{preds[0]['frame_idx']}" if preds else "N/A"
        print(f"[{idx:02d}/{len(query_files):02d}] {q_name:24s} ({ttype:5s}) -> Top 1: {top1_str:15s} | Saved {len(preds)} rows ({latency:.0f}ms)")
        
    print("\n" + "=" * 80)
    print("📦 KIỂM TRA ĐỊNH DẠNG NỘP BÀI:")
    summary = validator.validate_directory(output_dir)
    print(f"   • Tổng số file CSV : {summary['total_files']} / {len(query_files)}")
    print(f"   • Trạng thái hợp lệ: {'✅ 100% HỢP LỆ CHUẨN BTC' if summary['all_valid'] else '⚠️ Cần kiểm tra lại'}")
    print("=" * 80)

if __name__ == "__main__":
    run_official_batch()
