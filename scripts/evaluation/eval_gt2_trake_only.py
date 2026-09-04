import os
import sys
import time
import json
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.retrieval.unified_search_core import UnifiedSearchCore
from src.query.llm_query_refiner import LLMQueryRefiner
from src.tasks.clean_task_handlers import TRAKEHandler
from src.evaluation.btc_metric import evaluate_query_predictions

def main():
    print("=" * 80)
    print("🚀 BẮT ĐẦU KIỂM TRA ĐỘ CHÍNH XÁC TRAKE MỚI TRÊN GROUND TRUTH 2...")
    print("=" * 80)

    gt2_file = BASE_DIR / "data" / "benchmark" / "ground_truth_2.json"
    with open(gt2_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    trake_cases = [tc for tc in gt_data["test_cases"] if tc["task_type"] == "trake"]
    print(f"Loaded {len(trake_cases)} TRAKE test cases from ground_truth_2.json.")

    search_core = UnifiedSearchCore(engine="siglip2", batch="batch_1")
    refiner = LLMQueryRefiner()
    trake_handler = TRAKEHandler(search_core, refiner)

    scores = []
    for idx, tc in enumerate(trake_cases, 1):
        qid = tc["query_id"]
        qtext = tc["query_text"]
        gt = tc["ground_truth"]

        t0 = time.time()
        preds, info, lat = trake_handler.search(qtext, top_k=100, config_name="A8_SOTA")
        elapsed = (time.time() - t0) * 1000

        eval_res = evaluate_query_predictions(preds, gt, "trake")
        score = eval_res["final_score"]
        scores.append(score)

        vid_rank = f"#{eval_res['video_hit_rank']}" if eval_res.get('video_hit_rank', -1) > 0 else "MISS"
        pos_rank = f"#{eval_res['first_pos_rank']}" if eval_res.get('first_pos_rank', -1) > 0 else "MISS"
        status = "🟢" if score > 0 else "🔴"

        num_ev = info.get("num_events", len(preds[0].get("event_frames", [])))
        print(f"[{idx}/{len(trake_cases)}] {status} {qid:14s} -> Events: {num_ev} | Video: {vid_rank:6s} | Pos: {pos_rank:6s} | Score: {score:.4f} ({elapsed:.0f}ms)")
        if preds:
            print(f"     Top 1: {preds[0]['video_id']} | Frames: {preds[0].get('event_frames', [])}")

    avg_score = sum(scores) / len(scores) if scores else 0.0
    print("\n" + "=" * 80)
    print(f"📊 ĐIỂM TRUNG BÌNH TRAKE MỚI: {avg_score:.4f} (Điểm SOTA cũ trước khi sửa: 0.7333)")
    diff = avg_score - 0.7333
    if diff >= 0:
        print(f"✅ KHÔNG HỀ BỊ GIẢM ĐIỂM! Điểm số giữ vững / tăng thêm (+{diff:.4f})")
    else:
        print(f"⚠️ Biến động điểm: {diff:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
