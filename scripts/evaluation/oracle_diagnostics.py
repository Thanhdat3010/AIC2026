import os
import sys
import json
import time
from pathlib import Path
import pandas as pd

# Force UTF-8 on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.tasks.qa_agent import VisualQAAgent
from src.evaluation.btc_metric import is_qa_match

def run_qa_oracle():
    """Chạy bài test Oracle cho nhóm tác vụ QA (P0 - Diagnostic)"""
    gt_file = BASE_DIR / "data" / "benchmark" / "ground_truth.json"
    with open(gt_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
        
    qa_cases = [c for c in gt_data["test_cases"] if c["task_type"] == "qa"]
    
    print("\n" + "=" * 80)
    print("🧠 BẮT ĐẦU CHẨN ĐOÁN ORACLE LADDER CHO NHÓM QA (14 QUERIES)")
    print("=" * 80)
    
    # Init QA Agent
    qa_agent = VisualQAAgent()
    
    total = len(qa_cases)
    passed_visual_only = 0
    passed_multi_crop = 0
    
    for c_idx, case in enumerate(qa_cases, 1):
        qid = case["query_id"]
        qtext = case["query_text"]
        gt = case["ground_truth"]
        
        vid = gt["video_id"]
        s_frame = gt["start_frame"]
        e_frame = gt["end_frame"]
        gt_answer = gt["answer"]
        
        # Pick the middle frame as the Oracle GT frame
        mid_frame = s_frame + (e_frame - s_frame) // 2
        oracle_candidate = [{"video_id": vid, "frame_idx": mid_frame, "score": 1.0}]
        
        print(f"\n[{c_idx}/{total}] Câu hỏi: {qid} | {qtext[:50]}...")
        print(f"   ▶ Cung cấp Oracle Frame: Video {vid}, Frame {mid_frame}")
        
        # Test 1: Full Frame Only (No Crop)
        ans_full, _ = qa_agent.answer_and_rerank(
            qa_question=qtext,
            candidates=oracle_candidate,
            max_inspect_frames=1,
            use_multi_crop=False
        )
        match_full = is_qa_match(ans_full, gt_answer, qtext)
        if match_full:
            passed_visual_only += 1
            
        # Test 2: Multi-crop (Current Pipeline)
        ans_crop, _ = qa_agent.answer_and_rerank(
            qa_question=qtext,
            candidates=oracle_candidate,
            max_inspect_frames=1,
            use_multi_crop=True
        )
        match_crop = is_qa_match(ans_crop, gt_answer, qtext)
        if match_crop:
            passed_multi_crop += 1
            
        print(f"   - Trả lời (Full Frame) : '{ans_full}' -> {'✅ ĐÚNG' if match_full else '❌ SAI'}")
        print(f"   - Trả lời (Multi-Crop): '{ans_crop}' -> {'✅ ĐÚNG' if match_crop else '❌ SAI'}")
        
    print("\n" + "=" * 80)
    print("📊 TỔNG KẾT BỆNH ÁN ORACLE QA:")
    print(f"   • Khả năng Perception của VLM (Full Frame)  : {passed_visual_only}/{total} ({(passed_visual_only/total)*100:.1f}%)")
    print(f"   • Khả năng Perception của VLM (Multi-Crop)  : {passed_multi_crop}/{total} ({(passed_multi_crop/total)*100:.1f}%)")
    
    if passed_visual_only > passed_multi_crop:
        print("   => CẢNH BÁO: Kỹ thuật Multi-Crop đang LÀM GIẢM khả năng nhận diện của VLM!")
    print("=" * 80)

if __name__ == "__main__":
    run_qa_oracle()
