import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent.parent
FRAMES_PATH = BASE_DIR / "data" / "batch_1" / "processed" / "frames.parquet"
GT_PATH = BASE_DIR / "data" / "benchmark" / "ground_truth.json"

def run_audit():
    print("=" * 95)
    print("🔍 BẮT ĐẦU CHẨN ĐOÁN: GT-KEYFRAME COVERAGE AUDIT (11 TEST CASES)")
    print("=" * 95)
    
    df_frames = pd.read_parquet(FRAMES_PATH)
    with open(GT_PATH, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
        
    test_cases = gt_data.get("test_cases", [])
    
    records = []
    
    for case in test_cases:
        qid = case["query_id"]
        ttype = case["task_type"]
        qtext = case["query_text"]
        gt = case["ground_truth"]
        
        target_video = gt.get("video_id", "N/A")
        v_frames = df_frames[df_frames["video_id"] == target_video].sort_values("frame_idx")
        
        if v_frames.empty:
            records.append({
                "Query ID": qid,
                "Task": ttype.upper(),
                "Target Video": target_video,
                "GT Frames": "N/A",
                "Total Keyframes": 0,
                "Keyframes in GT": 0,
                "Coverage Status": "❌ VIDEO NOT FOUND"
            })
            continue
            
        keyframe_indices = v_frames["frame_idx"].values
        
        if ttype in ["kis", "qa"]:
            start_f = gt.get("start_frame", 0)
            end_f = gt.get("end_frame", 0)
            gt_str = f"[{start_f} - {end_f}] ({end_f - start_f + 1}f)"
            
            # Đếm số lượng keyframes nằm trong khoảng [start_f, end_f]
            inside_mask = (keyframe_indices >= start_f) & (keyframe_indices <= end_f)
            inside_count = np.sum(inside_mask)
            inside_frames = keyframe_indices[inside_mask]
            
            # Tìm keyframe gần nhất trước và sau
            before = keyframe_indices[keyframe_indices < start_f]
            after = keyframe_indices[keyframe_indices > end_f]
            nearest_before = before[-1] if len(before) > 0 else "None"
            nearest_after = after[0] if len(after) > 0 else "None"
            
            status = f"✅ Có {inside_count} keyframes" if inside_count > 0 else f"⚠️ MISS GAP (Trước: {nearest_before}, Sau: {nearest_after})"
            
            records.append({
                "Query ID": qid,
                "Task": ttype.upper(),
                "Target Video": target_video,
                "GT Interval": gt_str,
                "Total Video KFs": len(keyframe_indices),
                "KFs in GT": inside_count,
                "KF Frame IDs in GT": str(list(inside_frames[:5])) if inside_count > 0 else "NONE",
                "Status": status
            })
            
        elif ttype == "trake":
            events = gt.get("events", [])
            gt_events_str = f"{len(events)} events"
            inside_counts = []
            for ev in events:
                e_start = ev.get("start_frame", 0)
                e_end = ev.get("end_frame", 0)
                c = np.sum((keyframe_indices >= e_start) & (keyframe_indices <= e_end))
                inside_counts.append(c)
                
            all_covered = all(c > 0 for c in inside_counts)
            status = f"✅ 100% Events có KF ({inside_counts})" if all_covered else f"⚠️ Có Event thiếu KF ({inside_counts})"
            
            records.append({
                "Query ID": qid,
                "Task": ttype.upper(),
                "Target Video": target_video,
                "GT Interval": gt_events_str,
                "Total Video KFs": len(keyframe_indices),
                "KFs in GT": sum(inside_counts),
                "KF Frame IDs in GT": f"Counts per ev: {inside_counts}",
                "Status": status
            })

    df_out = pd.DataFrame(records)
    print(df_out[["Query ID", "Task", "Target Video", "GT Interval", "Total Video KFs", "KFs in GT", "Status"]].to_string(index=False))
    print("=" * 95)
    
    # In chi tiết cho 2 câu 0 điểm
    print("\n🔬 CHI TIẾT CHẨN ĐOÁN 2 CA ĐẶC BIỆT (test-kis-08 & test-trake-03):")
    for r in records:
        if r["Query ID"] in ["test-kis-08", "test-trake-03"]:
            print(f"▶ [{r['Query ID']}] Video: {r['Target Video']} | GT: {r['GT Interval']} | KFs in GT: {r['KFs in GT']} | Chi tiết: {r['KF Frame IDs in GT']}")

if __name__ == "__main__":
    run_audit()
