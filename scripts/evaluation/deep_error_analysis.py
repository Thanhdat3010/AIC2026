import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Force UTF-8 on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.retrieval.task_specialized_engine import TaskSpecializedEngine

def main():
    print("Loading TaskSpecializedEngine...")
    engine = TaskSpecializedEngine()
    
    gt_path = os.path.join(BASE_DIR, "data/benchmark/ground_truth.json")
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
        
    df_frames = pd.read_parquet(os.path.join(BASE_DIR, "data/batch_1/processed/frames.parquet"))
    
    ocr_path = os.path.join(BASE_DIR, "data/batch_1/processed/ocr_results.parquet")
    df_ocr = pd.read_parquet(ocr_path) if os.path.exists(ocr_path) else None

    queries_to_analyze = [
        "test-trake-03",
        "test-qa-16",
        "test-kis-19",
        "test-qa-17"
    ]
    
    print("\nStarting Deep Error Analysis...\n" + "="*50)
    for tc in gt_data["test_cases"]:
        q_id = tc["query_id"]
        if q_id not in queries_to_analyze:
            continue
            
        print(f"\n[QUERY] {q_id}: {tc['query_text']}")
        gt = tc["ground_truth"]
        target_video = gt["video_id"]
        
        # 1. Determine GT bounds
        if tc["task_type"] == "trake":
            events_gt = gt["events"]
            gt_intervals = [(ev["start_frame"]/30.0, ev["end_frame"]/30.0) for ev in events_gt]
        else:
            gt_intervals = [(gt["start_frame"]/30.0, gt["end_frame"]/30.0)]
            
        print(f"Target Video: {target_video}")
        
        # 2. Run Engine
        candidates, meta, latency = engine.search(tc['query_text'], task_type=tc['task_type'], top_k=100)
        
        # 3. Print Parsed Info & Asserts
        if tc["task_type"] == "trake":
            parsed_events = meta.get("events", [])
            print(f"  [TRAKE DECOMPOSITION]")
            for idx, pe in enumerate(parsed_events):
                print(f"    Parsed Event {idx+1}: {pe}")
            print(f"    Total Parsed Events: {len(parsed_events)} | Total GT Events: {len(gt_intervals)}")
            if len(parsed_events) != len(gt_intervals):
                print(f"    [!] WARNING: PARSER DECOMPOSITION MISMATCH!")
                # We won't crash so we can see the rest of the queries, but note it.
                
        # 4. Find target rank
        ranks = [i for i, c in enumerate(candidates) if c['video_id'] == target_video]
        if not ranks:
            print(f"  Result: TARGET VIDEO MISSED (Not in Top 100)")
        else:
            best_rank = ranks[0]
            print(f"  Result: Target Video found at Rank #{best_rank + 1}")
            cand = candidates[best_rank]
            pred_time = cand.get('pts_time', cand.get('frame_idx', 0)/30.0)
            pred_frame_idx = cand.get('frame_idx', 0)
            print(f"  Candidate Source: sampled_keyframe | frame_id: {pred_frame_idx} | pts_time: {pred_time:.2f}s")
            
            if tc["task_type"] == "trake":
                # Recover sim_matrix and pts_times for TARGET VIDEO
                sim_matrix = cand.get("sim_matrix")
                pts_times = cand.get("pts_times")
                
                if sim_matrix is not None and pts_times is not None:
                    # Run DP again just for this target video to get exact debug logs
                    # We have to trim gt_intervals or sim_matrix to match
                    min_events = min(len(gt_intervals), sim_matrix.shape[0])
                    
                    print("  [TRAKE EXACT TARGET DP DEBUG]")
                    # Redo DP logic here to capture variables
                    S_smooth = np.zeros_like(sim_matrix[:min_events])
                    for i in range(min_events):
                        row = sim_matrix[i]
                        left = np.pad(row[:-1], (1, 0), mode='edge')
                        right = np.pad(row[1:], (0, 1), mode='edge')
                        S_smooth[i] = 0.2 * left + 0.6 * row + 0.2 * right
                        
                    S_norm = np.zeros_like(S_smooth)
                    for i in range(min_events):
                        r_min = np.min(S_smooth[i])
                        r_max = np.max(S_smooth[i])
                        if r_max - r_min > 1e-6:
                            S_norm[i] = (S_smooth[i] - r_min) / (r_max - r_min)
                        else:
                            S_norm[i] = S_smooth[i]
                            
                    chosen_j = engine._trake_agent._solve_monotonic_dp(sim_matrix[:min_events], pts_times)
                    
                    for i in range(min_events):
                        st, en = gt_intervals[i]
                        pred_j = chosen_j[i]
                        pred_t = pts_times[pred_j]
                        
                        row_scores = S_norm[i]
                        unconstrained_j = int(np.argmax(row_scores))
                        peak_val = row_scores[unconstrained_j]
                        plateau_indices = np.where(row_scores >= peak_val - 0.01)[0]
                        plat_st = pts_times[plateau_indices[0]]
                        plat_en = pts_times[plateau_indices[-1]]
                        
                        dist = 0.0
                        if pred_t < st: dist = pred_t - st
                        elif pred_t > en: dist = pred_t - en
                            
                        print(f"    Event {i+1}: GT [{st:.2f}s - {en:.2f}s]")
                        print(f"      Pred Time: {pred_t:.2f}s (Signed Dist: {dist:.2f}s)")
                        print(f"      Unconstrained Peak Time: {pts_times[unconstrained_j]:.2f}s")
                        print(f"      Plateau Bounds: [{plat_st:.2f}s - {plat_en:.2f}s] (Size: {len(plateau_indices)} frames)")
            else:
                # QA / KIS
                st, en = gt_intervals[0]
                dist = 0.0
                if pred_time < st: dist = pred_time - st
                elif pred_time > en: dist = pred_time - en
                print(f"  Pred Time: {pred_time:.2f}s (Signed Dist: {dist:.2f}s)")
                
                if tc["task_type"] == "qa":
                    print(f"  [QA VERIFICATION]")
                    gt_ans = gt.get("answer", "")
                    pred_ans = meta.get("generated_qa_answer", "N/A")
                    print(f"    GT Answer: {gt_ans}")
                    print(f"    Pred Answer: {pred_ans}")
                    
        # 5. OCR Check for KIS/QA
        if df_ocr is not None and tc["task_type"] in ["kis", "qa"]:
            v_ocr = df_ocr[df_ocr['video_id'] == target_video]
            if not v_ocr.empty:
                print(f"  [OCR Check] Found {len(v_ocr)} OCR boxes in target video.")
                top_ocr = v_ocr.sort_values('confidence', ascending=False).head(3)
                safe_texts = [str(x).encode('ascii', 'replace').decode('ascii') for x in top_ocr['ocr_text'].tolist()]
                print(f"  Top OCR samples: {safe_texts}")
            else:
                print("  [OCR Check] No OCR data for target video.")
        print("-" * 50)

if __name__ == "__main__":
    main()
