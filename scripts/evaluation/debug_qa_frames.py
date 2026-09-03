import sys
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.retrieval.unified_search_core import UnifiedSearchCore
from src.query.llm_query_refiner import LLMQueryRefiner
from src.tasks.clean_task_handlers import QAHandler
from src.evaluation.btc_metric import evaluate_query_predictions

search_core = UnifiedSearchCore(engine="siglip2", batch="batch_1")
refiner = LLMQueryRefiner()
qa_handler = QAHandler(search_core, refiner)

with open('data/benchmark/ground_truth_2.json', 'r', encoding='utf-8') as f:
    gt2 = json.load(f)['test_cases']

print('=' * 100)
print('DEBUGGING CHUYÊN SÂU 7 CÂU QA TRÊN GROUND TRUTH 2:')
print('=' * 100)

for tc in gt2:
    if tc['task_type'] == 'qa':
        qid = tc['query_id']
        qtext = tc['query_text']
        gt = tc['ground_truth']
        
        preds, info, lat = qa_handler.search(qtext, top_k=100, config_name="A8")
        eval_res = evaluate_query_predictions(preds, gt, 'qa')
        
        v_ans = info.get("vlm_answer", "N/A")
        print(f"\n🔍 [{qid}] Final Score: {eval_res['final_score']:.4f} | Error: {eval_res.get('error_type', '')}")
        print(f"   • Đề bài: {qtext}")
        print(f"   • VLM Answer sinh ra: '{v_ans}' vs GT Answer: '{gt['answer']}'")
        print(f"   • Target GT: Video={gt['video_id']} | Range=[{gt['start_frame']}, {gt['end_frame']}]")
        print(f"   • Top 10 Predictions (Video & Frame & Answer):")
        for p in preds[:10]:
            is_gt_vid = (p['video_id'] == gt['video_id'])
            is_gt_frame = is_gt_vid and (gt['start_frame'] - 5 <= p['frame_idx'] <= gt['end_frame'] + 5)
            match_str = "🎯 [TRÚNG FRAME & VIDEO]" if is_gt_frame else ("✅ [TRÚNG VIDEO]" if is_gt_vid else "❌")
            print(f"      Rank #{p['rank']:2d} | Vid: {p['video_id']} | Frame: {p['frame_idx']:5d} | Ans: '{p.get('answer', '')}' | {match_str}")
