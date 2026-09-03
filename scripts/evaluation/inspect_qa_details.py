import sys
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

with open('data/benchmark/ground_truth_2.json', 'r', encoding='utf-8') as f:
    gt2 = json.load(f)['test_cases']

with open('data/benchmark/ground_truth_2_results.json', 'r', encoding='utf-8') as f:
    res = json.load(f)

with open('data/benchmark/qa_judge_cache.json', 'r', encoding='utf-8') as f:
    judge_cache = json.load(f)

a8_res = res.get('A8', {}).get('detailed_results', [])
gt_dict = {tc['query_id']: tc for tc in gt2}

print('=' * 100)
print('PHÂN TÍCH CHI TIẾT ĐÁP ÁN VÀ ĐIỂM SỐ CỦA 7 CÂU QA TRÊN GT2:')
print('=' * 100)

for r in a8_res:
    if r.get('task_type') == 'qa':
        qid = r['query_id']
        gt = gt_dict[qid]
        sc = r.get('score', 0.0)
        err = r.get('error_type', '')
        vr = r.get('video_hit_rank', -1)
        pr = r.get('first_pos_rank', -1)
        
        # Tìm đáp án đã nộp
        ans_evals = []
        gt_ans = gt['ground_truth']['answer']
        
        print(f"\n👉 [{qid}] Score: {sc:.4f} | Error: {err} | Video Rank: #{vr} | Pos Rank: #{pr}")
        print(f"   • Câu hỏi: {gt['query_text']}")
        print(f"   • GT Video: {gt['ground_truth']['video_id']} | Khung hình GT: [{gt['ground_truth']['start_frame']}, {gt['ground_truth']['end_frame']}]")
        print(f"   • ĐÁP ÁN GROUND TRUTH: '{gt_ans}'")
        
        # Kiểm tra trong judge cache các đáp án liên quan
        for k_pair, j_score in judge_cache.items():
            if "|||" in k_pair:
                cand_ans, g_ans = k_pair.split("|||", 1)
                if g_ans.strip().lower() == gt_ans.strip().lower():
                    print(f"   • Judge Cache: Pred '{cand_ans}' vs GT '{g_ans}' => Score: {j_score}")
            elif gt_ans.strip().lower() in k_pair.lower():
                print(f"   • Judge Cache Key: {k_pair} => Score: {j_score}")

print('=' * 100)
