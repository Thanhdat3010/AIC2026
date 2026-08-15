import json
import argparse
import sys
import time
from pathlib import Path
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import settings
from tasks.kis import KISTaskRunner

def evaluate_kis(test_cases, kis_runner, top_k=100):
    k_thresholds = [1, 5, 20, 50, 100]
    r_at_k_hits = {k: 0 for k in k_thresholds}
    final_scores = []
    
    kis_cases = [tc for tc in test_cases if tc.get("task_type", "kis") == "kis"]
    total = len(kis_cases)
    
    if total == 0:
        print("[WARNING] No KIS test cases found in benchmark.")
        return
        
    print(f"\nEvaluating {total} KIS test cases...\n")
    
    for tc in kis_cases:
        qid = tc["query_id"]
        qtext = tc["query_text"]
        gt = tc["ground_truth"]
        gt_video = gt["video_id"]
        s = gt["start_frame"]
        e = gt["end_frame"]
        
        # Run search
        candidate_frames = kis_runner.multi_retriever.search(qtext, top_k_per_cue=settings.retrieval.top_k_per_cue)
        aggregated_frames = kis_runner.aggregator.aggregate(candidate_frames, top_k_videos=settings.retrieval.top_k_videos)
        final_frames = kis_runner.fusion.rerank(qtext, aggregated_frames)
        
        # Check ranks
        # r_i is 1 if vid matches and frame_idx in [s, e]
        ranks_match = []
        for rank, frame in enumerate(final_frames[:top_k], start=1):
            if frame["video_id"] == gt_video and (s <= frame["frame_idx"] <= e):
                ranks_match.append(rank)
                
        best_rank = ranks_match[0] if ranks_match else None
        
        # Compute R@k for k in {1, 5, 20, 50, 100}
        case_r_at_k = {}
        for k in k_thresholds:
            hit = 1.0 if (best_rank is not None and best_rank <= k) else 0.0
            case_r_at_k[k] = hit
            if hit:
                r_at_k_hits[k] += 1
                
        # Final Score for this query: average of R@k
        case_final_score = sum(case_r_at_k.values()) / len(k_thresholds)
        final_scores.append(case_final_score)
        
        status = f"✅ Hit @ Rank {best_rank}" if best_rank else "❌ Missed"
        print(f"[{qid}] {status} | Score: {case_final_score:.2f} | Query: '{qtext[:40]}...'")

    # Print Summary Report
    print("\n" + "="*60)
    print("           AIC 2026 BENCHMARK EVALUATION REPORT")
    print("="*60)
    print(f"Total Test Queries : {total}")
    print("-"*60)
    for k in k_thresholds:
        pct = (r_at_k_hits[k] / total) * 100
        print(f"Recall@{k:<3} (R@{k:<3})  : {pct:5.1f}%  ({r_at_k_hits[k]}/{total} queries)")
    print("-"*60)
    avg_final = np.mean(final_scores)
    print(f"🏆 FINAL SCORE (BTC Formula): {avg_final:.4f} / 1.0000")
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate AIC 2026 Benchmark Test Set")
    parser.add_argument("--benchmark_path", type=str, default="data/benchmark/ground_truth.json",
                        help="Path to ground truth JSON file")
    args = parser.parse_args()
    
    bench_file = Path(args.benchmark_path)
    if not bench_file.exists():
        print(f"[ERROR] Benchmark file not found at: {bench_file}")
        sys.exit(1)
        
    with open(bench_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    test_cases = data.get("test_cases", [])
    
    print("=== Initializing KIS Pipeline for Evaluation ===")
    kis_runner = KISTaskRunner()
    
    evaluate_kis(test_cases, kis_runner)

if __name__ == "__main__":
    main()
