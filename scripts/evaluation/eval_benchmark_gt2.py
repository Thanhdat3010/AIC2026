import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.retrieval.unified_search_core import UnifiedSearchCore
from src.query.llm_query_refiner import LLMQueryRefiner
from src.tasks.clean_task_handlers import KISHandler, QAHandler, TRAKEHandler
from src.evaluation.btc_metric import evaluate_query_predictions


def run_benchmark_gt2(config_id: str, search_core: UnifiedSearchCore, refiner: LLMQueryRefiner, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    print("\n" + "=" * 100, flush=True)
    print(f"🚀 BẮT ĐẦU ĐO LƯỜNG CẤU HÌNH SOTA [{config_id}] TRÊN TẬP THỬ THÁCH MỚI GROUND_TRUTH_2 ({len(test_cases)} CÂU)...", flush=True)
    print("=" * 100, flush=True)

    kis_handler = KISHandler(search_core, refiner)
    qa_handler = QAHandler(search_core, refiner)
    trake_handler = TRAKEHandler(search_core, refiner)

    query_results = []
    latencies = []

    for idx, tc in enumerate(test_cases, 1):
        qid = tc["query_id"]
        ttype = tc["task_type"]
        qtext = tc["query_text"]
        gt = tc["ground_truth"]

        t_start = time.time()

        if config_id in ["A0", "B1"]:
            # Baseline: Pure Standard SigLIP-2 Tiếng Việt gốc
            vec = search_core.encode_text(qtext)
            hits = search_core.search_visual(vec, top_k=100)
            if ttype == "trake":
                preds = []
                for rank, h in enumerate(hits[:100], 1):
                    f0 = h["frame_idx"]
                    preds.append({"video_id": h["video_id"], "events": [str(f0), str(f0+25), str(f0+50)], "event_frames": [f0, f0+25, f0+50], "rank": rank})
            else:
                preds = hits
        else:
            # SOTA Configuration (A6, A7, Grand Master)
            if ttype == "kis":
                preds, _, _ = kis_handler.search(qtext, top_k=100, config_name=config_id)
            elif ttype == "qa":
                preds, _, _ = qa_handler.search(qtext, top_k=100, config_name=config_id)
            elif ttype == "trake":
                preds, _, _ = trake_handler.search(qtext, top_k=100, config_name=config_id)
            else:
                preds, _, _ = kis_handler.search(qtext, top_k=100, config_name=config_id)

        latency = (time.time() - t_start) * 1000
        latencies.append(latency)

        # Đánh giá câu hiện tại
        eval_res = evaluate_query_predictions(preds, gt, ttype)
        eval_res["query_id"] = qid
        eval_res["task_type"] = ttype
        eval_res["latency_ms"] = latency
        eval_res["score"] = eval_res["final_score"]
        query_results.append(eval_res)

        vid_rank_str = f"#{eval_res['video_hit_rank']}" if eval_res.get('video_hit_rank', -1) > 0 else "MISS"
        pos_rank_str = f"#{eval_res['first_pos_rank']}" if eval_res.get('first_pos_rank', -1) > 0 else "MISS"
        
        status_icon = "🟢" if eval_res['score'] > 0 else "🔴"
        print(f"[{idx:02d}/{len(test_cases)}] {status_icon} {qid:12s} ({ttype.upper():5s}) -> Video: {vid_rank_str:6s} | Pos: {pos_rank_str:6s} | Score: {eval_res['score']:.4f} | {eval_res.get('error_type', '')} ({latency:.0f}ms)", flush=True)

    # Tổng hợp metrics
    df = pd.DataFrame(query_results)
    kis_df = df[df["task_type"] == "kis"]
    qa_df = df[df["task_type"] == "qa"]
    trake_df = df[df["task_type"] == "trake"]

    kis_score = float(kis_df["score"].mean()) if not kis_df.empty else 0.0
    qa_score = float(qa_df["score"].mean()) if not qa_df.empty else 0.0
    trake_score = float(trake_df["score"].mean()) if not trake_df.empty else 0.0
    macro_score = float(df["score"].mean())

    # Video recall
    v_r1 = float((df["video_hit_rank"] == 1).mean())
    v_r5 = float(((df["video_hit_rank"] > 0) & (df["video_hit_rank"] <= 5)).mean())
    v_r20 = float(((df["video_hit_rank"] > 0) & (df["video_hit_rank"] <= 20)).mean())
    v_r100 = float(((df["video_hit_rank"] > 0) & (df["video_hit_rank"] <= 100)).mean())

    summary = {
        "config_id": config_id,
        "total_queries": len(test_cases),
        "kis_score": kis_score,
        "qa_score": qa_score,
        "trake_score": trake_score,
        "macro_score": macro_score,
        "video_r1": v_r1,
        "video_r5": v_r5,
        "video_r20": v_r20,
        "video_r100": v_r100,
        "avg_latency_ms": float(np.mean(latencies)),
        "detailed_results": query_results
    }

    print("\n" + "-" * 80, flush=True)
    print(f"📊 KẾT QUẢ CẤU HÌNH [{config_id}] TRÊN GROUND_TRUTH_2:", flush=True)
    print(f"   • KIS Score (22 câu)   : {kis_score:.4f}", flush=True)
    print(f"   • QA Score (7 câu)     : {qa_score:.4f}", flush=True)
    print(f"   • TRAKE Score (3 câu)  : {trake_score:.4f}", flush=True)
    print(f"   • 🏆 MACRO BTC SCORE   : {macro_score:.4f}", flush=True)
    print(f"   • Video Recall@1/5/20/100: {v_r1:.1%} / {v_r5:.1%} / {v_r20:.1%} / {v_r100:.1%}", flush=True)
    print(f"   • Độ trễ trung bình: {summary['avg_latency_ms']:.1f} ms", flush=True)
    print("-" * 80, flush=True)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Ablation Benchmark on Ground Truth 2")
    parser.add_argument("--configs", nargs="+", default=["A0", "A7", "A8_1", "A8_2", "A8_3", "A8_4", "A8"], help="List of configs to evaluate")
    args = parser.parse_args()

    gt2_file = BASE_DIR / "data" / "benchmark" / "ground_truth_2.json"
    with open(gt2_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    test_cases = gt_data["test_cases"]

    print(f"Loaded ground_truth_2.json with {len(test_cases)} queries.")
    search_core = UnifiedSearchCore(engine="siglip2", batch="batch_1")
    refiner = LLMQueryRefiner()

    # Nạp sẵn cache cũ nếu có
    out_file = BASE_DIR / "data" / "benchmark" / "ground_truth_2_results.json"
    results = {}
    if out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            results = {}

    configs_to_run = args.configs

    for cfg in configs_to_run:
        res = run_benchmark_gt2(cfg, search_core, refiner, test_cases)
        results[cfg] = res

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # In Bảng Tổng Sắp Ablation Study Đối Đầu
    print("\n" + "=" * 115)
    print("🏆 BẢNG TỔNG SẮP ABLATION STUDY ĐỐI ĐẦU TRÊN GROUND TRUTH 2 (32 CÂU):")
    print("=" * 115)
    print(f"{'Config':<10} | {'KIS (22)':<10} | {'QA (7)':<10} | {'TRAKE (3)':<10} | {'Macro Score':<12} | {'Δ vs A7':<10} | {'Video-R@1':<10} | {'Latency':<10}")
    print("-" * 115)

    base_a7_macro = results.get("A7", {}).get("macro_score", 0.6708)

    for cfg in configs_to_run:
        if cfg in results:
            r = results[cfg]
            k_sc = r.get("kis_score", 0.0)
            q_sc = r.get("qa_score", 0.0)
            t_sc = r.get("trake_score", 0.0)
            m_sc = r.get("macro_score", 0.0)
            vr1 = r.get("video_r1", 0.0)
            lat = r.get("avg_latency_ms", 0.0)
            delta = m_sc - base_a7_macro
            delta_str = f"{delta:+.4f}" if cfg != "A7" else "0.0000"
            print(f"{cfg:<10} | {k_sc:10.4f} | {q_sc:10.4f} | {t_sc:10.4f} | {m_sc:12.4f} | {delta_str:<10} | {vr1*100:9.1f}% | {lat:8.1f}ms")

    print("=" * 115)
    print(f"✅ Đã lưu kết quả chi tiết vào: {out_file}")


if __name__ == "__main__":
    main()
