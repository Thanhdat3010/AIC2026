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

a8_res = res.get('A8', {}).get('detailed_results', [])
gt_dict = {tc['query_id']: tc for tc in gt2}

print('=' * 100)
print('CHI TIET 7 CAU HOI QA TREN GROUND TRUTH 2 (CAU HINH A8):')
print('=' * 100)
for r in a8_res:
    if r.get('task_type') == 'qa':
        qid = r['query_id']
        gt = gt_dict[qid]
        sc = r.get('score', 0.0)
        err = r.get('error_type', '')
        vr = r.get('video_hit_rank', -1)
        pr = r.get('first_pos_rank', -1)
        min_dist = r.get('min_frame_distance', -1)
        print(f"Query: {qid} | Score: {sc:.4f} | Error: {err} | Video Rank: #{vr} | Pos Rank: #{pr} | Min Frame Dist: {min_dist}")
        print(f"   - De bai: {gt['query_text']}")
        print(f"   - GT Video: {gt['ground_truth']['video_id']} | Range: [{gt['ground_truth']['start_frame']}, {gt['ground_truth']['end_frame']}] | Answer: {gt['ground_truth']['answer']}")
        print('-' * 100)
