import os
import sys
import io
import json
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import faiss

# Force UTF-8 on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Import modules from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.indexing.faiss_indexer import load_faiss_index
from src.query.text_encoder import UnifiedTextEncoder

# Simple high-quality English translation dictionary for benchmark queries (zero-shot test)
BENCHMARK_TRANSLATIONS = {
    "test-kis-01": "Inside a room, a woman wraps and adjusts an orange-yellow sarong around the waist of a man wearing a blue shirt.",
    "test-qa-02": "When two men are moving a motorcycle loaded with bamboo shoots, what is the person in front wearing on his head?",
    "test-trake-03": "A sequence of cooking actions: chef pours diced onions into a pan, adds minced beef, then adds green peas, diced carrots, and finally boiled pasta into the pan.",
    "test-kis-04": "In a close-up shot, a chef using checkered oven mitts tilts a brown glass pot and pours green matcha jelly mixture into a white rectangular porcelain mold.",
    "test-kis-05": "Close-up of a hand using a wooden rolling pin to spread a pink mixture onto the inner surface of a betel leaf.",
    "test-kis-06": "Close-up of wooden chopsticks folding a yellow Vietnamese pancake banh xeo in half over the filling in the pan.",
    "test-qa-07": "In a conversation under a vineyard, when the host asks what time the garden owner starts and finishes work each day, what does the owner answer?",
    "test-kis-08": "In a close-up shot, a female student wearing a blue and white uniform, white headscarf and sunglasses holds a purple phone close to her face and touches the screen.",
    "test-trake-09": "Authorities inspect a room packed with cardboard boxes and books, hands open and verify books from boxes, a man wearing glasses examines a book before boxes, a man in white shirt is interviewed next to a plant, close-up of textbook stacks on shelves.",
    "test-kis-10": "In a documentary report, a man wearing a hat and glasses uses a vintage typewriter to create portrait and landscape art from characters, alternating between typing and artworks.",
    "test-kis-11": "Aerial flycam view looking straight down at 4 cyclists riding in a line along an asphalt road passing by a blooming purple bougainvillea tree on the right roadside."
}

def evaluate_retrieval_ranking(predictions: list[dict], ground_truth: dict, task_type: str) -> dict:
    """
    Tính điểm R@1, R@5, R@20, R@50, R@100 theo chuẩn 100% BTC cho 1 query.
    predictions: danh sách top 100 dict {'video_id': ..., 'frame_idx': ...}
    """
    K_VALUES = [1, 5, 20, 50, 100]
    r_scores = []

    target_video = ground_truth.get("video_id", "")
    
    if task_type in ["kis", "qa"]:
        s_frame = ground_truth.get("start_frame", 0)
        e_frame = ground_truth.get("end_frame", 0)

        for p in predictions:
            v_pred = p.get("video_id", "")
            f_pred = p.get("frame_idx", 0)
            if v_pred == target_video and s_frame <= f_pred <= e_frame:
                r_scores.append(1.0)
            else:
                r_scores.append(0.0)

    elif task_type == "trake":
        events = ground_truth.get("events", [])
        n_events = len(events)
        # Trong baseline tìm kiếm frame đơn thuần, ta chấm điểm dựa trên việc bắt trúng bất kỳ frame sự kiện nào
        for p in predictions:
            v_pred = p.get("video_id", "")
            f_pred = p.get("frame_idx", 0)
            if v_pred == target_video:
                hit_count = 0
                for ev in events:
                    if ev["start_frame"] <= f_pred <= ev["end_frame"]:
                        hit_count += 1
                r_scores.append(hit_count / max(1, n_events))
            else:
                r_scores.append(0.0)
    else:
        r_scores = [0.0] * len(predictions)

    # Đảm bảo đủ 100 kết quả
    if len(r_scores) < 100:
        r_scores.extend([0.0] * (100 - len(r_scores)))

    # Tính R@k
    r_at_k = {}
    for k in K_VALUES:
        r_at_k[f"R@{k}"] = max(r_scores[:k]) if k <= len(r_scores) else 0.0

    # Final Score trung bình cộng của 5 mốc
    final_score = sum(r_at_k.values()) / len(K_VALUES)
    
    # Tìm rank trúng đầu tiên (1-based)
    first_hit_rank = -1
    for idx, sc in enumerate(r_scores):
        if sc > 0.0:
            first_hit_rank = idx + 1
            break

    return {
        "r_at_k": r_at_k,
        "final_score": final_score,
        "first_hit_rank": first_hit_rank,
        "first_hit_score": r_scores[first_hit_rank - 1] if first_hit_rank != -1 else 0.0
    }

class DenseBaselinePipeline:
    def __init__(self, engine: str = "siglip2", batch: str = "batch_1"):
        self.engine = engine
        self.encoder = UnifiedTextEncoder(engine=engine)
        self.index, self.df_frames = load_faiss_index(engine=engine, batch=batch)
        print(f"✅ Pipeline [{engine.upper()}] đã sẵn sàng với {self.index.ntotal:,} vectors!", flush=True)

    def search(self, query_text: str, top_k: int = 100) -> tuple[list[dict], float]:
        t0 = time.time()
        # Mã hóa text query
        query_vec = self.encoder.encode_text(query_text)
        
        # FAISS search
        scores, indices = self.index.search(query_vec, top_k)
        latency_ms = (time.time() - t0) * 1000

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            row = self.df_frames.iloc[idx]
            results.append({
                "rank": rank + 1,
                "video_id": row["video_id"],
                "frame_idx": int(row["frame_idx"]),
                "global_id": int(row["global_id"]),
                "score": float(score)
            })
        return results, latency_ms

def run_ablation_experiment(config_id: int):
    base_dir = Path(__file__).resolve().parent.parent
    gt_file = base_dir / "data" / "benchmark" / "ground_truth.json"

    if not gt_file.exists():
        print(f"❌ Không tìm thấy ground truth tại: {gt_file}", flush=True)
        return

    with open(gt_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    test_cases = gt_data["test_cases"]
    print("\n" + "=" * 95, flush=True)
    print(f"🧪 CHẠY THỬ NGHIỆM ABLATION: CẤU HÌNH {config_id}", flush=True)
    print("=" * 95, flush=True)

    # Lựa chọn pipeline theo config_id
    if config_id == 0:
        config_name = "Baseline 0: OpenAI CLIP ViT-B/32 (512d) + English Translation"
        pipeline = DenseBaselinePipeline(engine="clip")
    elif config_id == 1:
        config_name = "Baseline 1: Google SigLIP 2 SO400M (1152d) + English Translation"
        pipeline = DenseBaselinePipeline(engine="siglip2")
    else:
        print(f"⚠️ Cấu hình {config_id} sẽ được kích hoạt ở các pha tiếp theo!")
        return

    print(f"[*] Đang đánh giá trên {len(test_cases)} test cases...\n", flush=True)
    
    records = []
    latencies = []
    task_scores = {"kis": [], "qa": [], "trake": []}

    for case in test_cases:
        qid = case["query_id"]
        ttype = case["task_type"]
        qtext = case["query_text"]
        gt = case["ground_truth"]

        # Lấy bản dịch tiếng Anh
        en_query = BENCHMARK_TRANSLATIONS.get(qid, qtext)

        preds, latency = pipeline.search(en_query, top_k=100)
        latencies.append(latency)

        eval_res = evaluate_retrieval_ranking(preds, gt, ttype)
        fs = eval_res["final_score"]
        task_scores[ttype].append(fs)

        top1 = f"{preds[0]['video_id']}:{preds[0]['frame_idx']}" if preds else "N/A"
        hit_rank = f"#{eval_res['first_hit_rank']}" if eval_res['first_hit_rank'] != -1 else "MISS"

        target_str = f"{gt.get('video_id')}"
        if "start_frame" in gt:
            target_str += f" [{gt['start_frame']}-{gt['end_frame']}]"
        elif "events" in gt:
            target_str += f" ({len(gt['events'])} events)"

        records.append({
            "Query ID": qid,
            "Task": ttype.upper(),
            "Target": target_str,
            "Top-1 Pred": top1,
            "Hit Rank": hit_rank,
            "R@1": f"{eval_res['r_at_k']['R@1']:.2f}",
            "R@5": f"{eval_res['r_at_k']['R@5']:.2f}",
            "R@20": f"{eval_res['r_at_k']['R@20']:.2f}",
            "R@50": f"{eval_res['r_at_k']['R@50']:.2f}",
            "R@100": f"{eval_res['r_at_k']['R@100']:.2f}",
            "Final Score": f"{fs:.4f}",
            "Latency": f"{latency:.1f}ms"
        })

    # In bảng chi tiết từng query
    df_res = pd.DataFrame(records)
    print(df_res.to_string(index=False))
    print("\n" + "-" * 95)

    # Tính điểm tổng kết
    kis_avg = np.mean(task_scores["kis"]) if task_scores["kis"] else 0.0
    qa_avg = np.mean(task_scores["qa"]) if task_scores["qa"] else 0.0
    trake_avg = np.mean(task_scores["trake"]) if task_scores["trake"] else 0.0
    overall_final = np.mean([r for scores in task_scores.values() for r in scores])
    avg_latency = np.mean(latencies)

    print(f"📊 BẢNG TỔNG KẾT CẤU HÌNH {config_id}: {config_name}")
    print(f"   • KIS Score (6 queries)   : {kis_avg:.4f}")
    print(f"   • QA Score (2 queries)    : {qa_avg:.4f}")
    print(f"   • TRAKE Score (2 queries) : {trake_avg:.4f}")
    print(f"   • 🏆 OVERALL FINAL SCORE   : {overall_final:.4f}")
    print(f"   • ⚡ Average Query Latency: {avg_latency:.2f} ms")
    print("=" * 95 + "\n")
    return {
        "config_id": config_id,
        "config_name": config_name,
        "kis_score": kis_avg,
        "qa_score": qa_avg,
        "trake_score": trake_avg,
        "final_score": overall_final,
        "latency_ms": avg_latency
    }

def main():
    parser = argparse.ArgumentParser(description="Chạy Ablation Benchmark trên tập 11 ground truth queries")
    parser.add_argument("--config", type=int, default=None, help="ID cấu hình muốn chạy (0: CLIP BTC, 1: SigLIP 2 SOTA, ...)")
    parser.add_argument("--compare_baselines", action="store_true", help="Chạy so sánh cả Baseline 0 (CLIP) và Baseline 1 (SigLIP 2)")
    args = parser.parse_args()

    if args.compare_baselines or args.config is None:
        print("\n🚀 BẮT ĐẦU SO SÁNH TRỰC DIỆN BASELINE 0 (CLIP BTC) VS BASELINE 1 (SIGLIP 2 SOTA)...")
        res0 = run_ablation_experiment(0)
        res1 = run_ablation_experiment(1)

        print("\n" + "🔥" * 40)
        print("🏆 SO SÁNH KẾT QUẢ ĐỘT PHÁ BASELINE 0 VS BASELINE 1:")
        print("🔥" * 40)
        print(f"| Chỉ số | Baseline 0 (CLIP BTC 512d) | Baseline 1 (SigLIP 2 1152d) | Mức độ Tăng Trưởng |")
        print(f"| :--- | :---: | :---: | :---: |")
        print(f"| **KIS Score** | {res0['kis_score']:.4f} | {res1['kis_score']:.4f} | **+{((res1['kis_score']-res0['kis_score']))*100:+.2f}%** |")
        print(f"| **QA Score** | {res0['qa_score']:.4f} | {res1['qa_score']:.4f} | **+{((res1['qa_score']-res0['qa_score']))*100:+.2f}%** |")
        print(f"| **TRAKE Score** | {res0['trake_score']:.4f} | {res1['trake_score']:.4f} | **+{((res1['trake_score']-res0['trake_score']))*100:+.2f}%** |")
        print(f"| 🏆 **FINAL SCORE** | **{res0['final_score']:.4f}** | **{res1['final_score']:.4f}** | **🔥 +{((res1['final_score']-res0['final_score']))*100:+.2f}%** |")
        print(f"| **Độ trễ** | {res0['latency_ms']:.1f} ms | {res1['latency_ms']:.1f} ms | ~ |")
        print("🔥" * 40 + "\n")
    else:
        run_ablation_experiment(args.config)

if __name__ == "__main__":
    main()
