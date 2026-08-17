import os
import sys
import io
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import faiss

# Force UTF-8 on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.indexing.faiss_indexer import load_faiss_index
from src.query.text_encoder import UnifiedTextEncoder
from src.retrieval.hybrid_engine import HybridRetrievalEngine
from src.retrieval.task_specialized_engine import TaskSpecializedEngine

# Zero-shot simple English translation dictionary for baseline tests
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
    Tính toán cả 2 nhóm chỉ số:
    1. BTC Official Metric: Frame-level R@1, R@5, R@20, R@50, R@100 & Final Score.
    2. Diagnostic Metrics: Stage-1 Video-level Recall @ 1, 5, 10, 20, 50, 100.
    """
    K_VALUES = [1, 5, 20, 50, 100]
    V_K_VALUES = [1, 5, 10, 20, 50, 100]
    
    target_video = ground_truth.get("video_id", "")
    
    # 1. Đo lường Video-Level Recall (Xác định Stage-1 Retriever đã tìm thấy đúng Video chưa)
    unique_videos = []
    seen_v = set()
    for p in predictions:
        v = p.get("video_id", "")
        if v and v not in seen_v:
            unique_videos.append(v)
            seen_v.add(v)

    video_hit_rank = -1
    for idx, v in enumerate(unique_videos, 1):
        if v == target_video:
            video_hit_rank = idx
            break

    video_recall_at_k = {}
    for k in V_K_VALUES:
        video_recall_at_k[f"V-R@{k}"] = 1.0 if (video_hit_rank != -1 and video_hit_rank <= k) else 0.0

    # 2. Đo lường Frame-Level Recall (Chuẩn 100% BTC)
    r_scores = []
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
        for p in predictions:
            v_pred = p.get("video_id", "")
            if v_pred == target_video:
                hit_count = 0
                event_frames = p.get("event_frames", [])
                if event_frames:
                    for j, ev in enumerate(events):
                        if j < len(event_frames) and ev["start_frame"] <= event_frames[j] <= ev["end_frame"]:
                            hit_count += 1
                else:
                    f_pred = p.get("frame_idx", 0)
                    for ev in events:
                        if ev["start_frame"] <= f_pred <= ev["end_frame"]:
                            hit_count += 1
                r_scores.append(hit_count / max(1, n_events))
            else:
                r_scores.append(0.0)
    else:
        r_scores = [0.0] * len(predictions)

    if len(r_scores) < 100:
        r_scores.extend([0.0] * (100 - len(r_scores)))

    r_at_k = {}
    for k in K_VALUES:
        r_at_k[f"R@{k}"] = max(r_scores[:k]) if k <= len(r_scores) else 0.0

    final_score = sum(r_at_k.values()) / len(K_VALUES)
    
    first_hit_rank = -1
    for idx, sc in enumerate(r_scores):
        if sc > 0.0:
            first_hit_rank = idx + 1
            break

    return {
        "r_at_k": r_at_k,
        "final_score": final_score,
        "first_hit_rank": first_hit_rank,
        "first_hit_score": r_scores[first_hit_rank - 1] if first_hit_rank != -1 else 0.0,
        "video_hit_rank": video_hit_rank,
        "video_recall_at_k": video_recall_at_k
    }

class DenseBaselinePipeline:
    def __init__(self, engine: str = "siglip2", batch: str = "batch_1"):
        self.engine = engine
        self.encoder = UnifiedTextEncoder(engine=engine)
        self.index, self.df_frames = load_faiss_index(engine=engine, batch=batch)

    def search(self, query_text: str, top_k: int = 100) -> tuple[list[dict], float]:
        t0 = time.time()
        query_vec = self.encoder.encode_text(query_text)
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

def run_ablation_experiment(config_code: str, pipeline_cache: dict = None) -> dict:
    gt_file = BASE_DIR / "data" / "benchmark" / "ground_truth.json"
    with open(gt_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    test_cases = gt_data["test_cases"]
    print("\n" + "=" * 100, flush=True)
    print(f"🧪 CHẠY THỬ NGHIỆM ĐO LƯỜNG & CHẨN ĐOÁN: CẤU HÌNH {config_code}", flush=True)
    print("=" * 100, flush=True)

    mode = "hybrid"
    use_multi_prompt = False
    use_dominant_weights = False
    use_adaptive_gating = False
    use_ocr = False
    use_asr = False
    use_temporal = False
    use_soft = False
    use_qa = False

    if config_code == "0":
        config_name = "Baseline 0: BTC CLIP (512d) + Dịch từ điển thô"
        mode = "dense"
        if pipeline_cache and "clip_dense" in pipeline_cache:
            pipeline = pipeline_cache["clip_dense"]
        else:
            pipeline = DenseBaselinePipeline(engine="clip")
            if pipeline_cache is not None:
                pipeline_cache["clip_dense"] = pipeline

    elif config_code == "1":
        config_name = "Baseline 1: Google SigLIP 2 (1152d) + Dịch từ điển thô"
        mode = "dense"
        if pipeline_cache and "siglip_dense" in pipeline_cache:
            pipeline = pipeline_cache["siglip_dense"]
        else:
            pipeline = DenseBaselinePipeline(engine="siglip2")
            if pipeline_cache is not None:
                pipeline_cache["siglip_dense"] = pipeline

    elif config_code == "1b":
        config_name = "Cấu hình 1b: SigLIP 2 + Single Gemini 3.5 Flash Lite Translation"
        mode = "hybrid"
        use_multi_prompt = False
        if pipeline_cache and "hybrid_siglip" in pipeline_cache:
            pipeline = pipeline_cache["hybrid_siglip"]
        else:
            pipeline = HybridRetrievalEngine(engine="siglip2")
            if pipeline_cache is not None:
                pipeline_cache["hybrid_siglip"] = pipeline

    elif config_code == "10":
        config_name = "Cấu hình 10: Full Combo Monolithic Pipeline (SigLIP 2 + MultiPrompt + Gating + QA + SoftFilter)"
        mode = "hybrid"
        use_multi_prompt = True
        use_dominant_weights = True
        use_adaptive_gating = True
        use_ocr = True
        use_asr = True
        use_qa = True
        use_soft = True
        if pipeline_cache and "hybrid_siglip" in pipeline_cache:
            pipeline = pipeline_cache["hybrid_siglip"]
        else:
            pipeline = HybridRetrievalEngine(engine="siglip2")
            if pipeline_cache is not None:
                pipeline_cache["hybrid_siglip"] = pipeline

    elif config_code == "11":
        config_name = "Cấu hình 11: 🚀 TASK-SPECIALIZED SOTA ARCHITECTURE (Chuyên biệt hóa KIS, QA, TRAKE + Gating thông minh)"
        mode = "specialized"
        if pipeline_cache and "task_specialized" in pipeline_cache:
            pipeline = pipeline_cache["task_specialized"]
        else:
            pipeline = TaskSpecializedEngine(engine="siglip2")
            if pipeline_cache is not None:
                pipeline_cache["task_specialized"] = pipeline

    else:
        print(f"⚠️ Cấu hình {config_code} không hợp lệ!")
        return None

    print(f"[*] Đang đánh giá trên {len(test_cases)} test cases...\n", flush=True)
    records = []
    latencies = []
    task_scores = {"kis": [], "qa": [], "trake": []}
    video_recalls = {k: [] for k in [1, 5, 10, 20, 50, 100]}

    for case in test_cases:
        qid = case["query_id"]
        ttype = case["task_type"]
        qtext = case["query_text"]
        gt = case["ground_truth"]

        en_query = BENCHMARK_TRANSLATIONS.get(qid, qtext)

        if mode == "dense":
            preds, latency = pipeline.search(en_query, top_k=100)
        elif mode == "hybrid":
            preds, qinfo, latency = pipeline.search(
                raw_query=qtext,
                top_k=100,
                use_multi_prompt=use_multi_prompt,
                use_dominant_weights=use_dominant_weights,
                use_adaptive_gating=use_adaptive_gating,
                use_ocr=use_ocr,
                use_asr=use_asr,
                use_temporal_smoothing=use_temporal,
                use_soft_filter=use_soft,
                use_qa_agent=use_qa
            )
        elif mode == "specialized":
            preds, qinfo, latency = pipeline.search(
                query_text=qtext,
                task_type=ttype,
                top_k=100
            )

        latencies.append(latency)
        eval_res = evaluate_retrieval_ranking(preds, gt, ttype)
        fs = eval_res["final_score"]
        task_scores[ttype].append(fs)

        for k in [1, 5, 10, 20, 50, 100]:
            video_recalls[k].append(eval_res["video_recall_at_k"][f"V-R@{k}"])

        top1 = f"{preds[0]['video_id']}:{preds[0]['frame_idx']}" if preds else "N/A"
        frame_hit = f"#{eval_res['first_hit_rank']}" if eval_res['first_hit_rank'] != -1 else "MISS"
        video_hit = f"#{eval_res['video_hit_rank']}" if eval_res['video_hit_rank'] != -1 else "MISS"

        target_str = f"{gt.get('video_id')}"
        if "start_frame" in gt:
            target_str += f" [{gt['start_frame']}-{gt['end_frame']}]"
        elif "events" in gt:
            target_str += f" ({len(gt['events'])} ev)"

        records.append({
            "Query ID": qid,
            "Task": ttype.upper(),
            "Target Video": target_str,
            "Video Rank": video_hit,
            "Frame Rank": frame_hit,
            "R@1": f"{eval_res['r_at_k']['R@1']:.2f}",
            "R@5": f"{eval_res['r_at_k']['R@5']:.2f}",
            "R@20": f"{eval_res['r_at_k']['R@20']:.2f}",
            "R@50": f"{eval_res['r_at_k']['R@50']:.2f}",
            "Final Score": f"{fs:.4f}",
            "Latency": f"{latency:.0f}ms"
        })

    df_res = pd.DataFrame(records)
    print(df_res.to_string(index=False))
    print("\n" + "-" * 100)

    kis_avg = np.mean(task_scores["kis"]) if task_scores["kis"] else 0.0
    qa_avg = np.mean(task_scores["qa"]) if task_scores["qa"] else 0.0
    trake_avg = np.mean(task_scores["trake"]) if task_scores["trake"] else 0.0
    overall_final = np.mean([r for scores in task_scores.values() for r in scores])
    avg_latency = np.mean(latencies)

    # Tính toán trung bình Video Recall các cấp độ
    vr_1 = np.mean(video_recalls[1]) * 100
    vr_5 = np.mean(video_recalls[5]) * 100
    vr_10 = np.mean(video_recalls[10]) * 100
    vr_20 = np.mean(video_recalls[20]) * 100
    vr_50 = np.mean(video_recalls[50]) * 100
    vr_100 = np.mean(video_recalls[100]) * 100

    print(f"📊 BẢNG TỔNG KẾT & CHẨN ĐOÁN BOTTLENECK CẤU HÌNH {config_code}:")
    print(f"   • KIS Score (6 queries)       : {kis_avg:.4f}")
    print(f"   • QA Score (2 queries)        : {qa_avg:.4f}")
    print(f"   • TRAKE Score (2 queries)     : {trake_avg:.4f}")
    print(f"   • 🏆 BTC FINAL SCORE (Frame)   : {overall_final:.4f}")
    print(f"   --------------------------------------------------------------")
    print(f"   🔍 STAGE-1 VIDEO-LEVEL RECALL (CHẨN ĐOÁN RETRIEVER):")
    print(f"      ▶ Video Recall@1  : {vr_1:.1f}%")
    print(f"      ▶ Video Recall@5  : {vr_5:.1f}%")
    print(f"      ▶ Video Recall@10 : {vr_10:.1f}%")
    print(f"      ▶ Video Recall@20 : {vr_20:.1f}%")
    print(f"      ▶ Video Recall@50 : {vr_50:.1f}%")
    print(f"      ▶ Video Recall@100: {vr_100:.1f}%")
    print(f"   • ⚡ Average Query Latency    : {avg_latency:.2f} ms")
    print("=" * 100 + "\n")
    return {
        "config_code": config_code,
        "config_name": config_name,
        "kis_score": kis_avg,
        "qa_score": qa_avg,
        "trake_score": trake_avg,
        "final_score": overall_final,
        "video_recall_1": vr_1,
        "video_recall_5": vr_5,
        "video_recall_10": vr_10,
        "video_recall_20": vr_20,
        "video_recall_50": vr_50,
        "video_recall_100": vr_100,
        "latency_ms": avg_latency,
        "records": records
    }

def save_ablation_markdown_report(results: list[dict], output_path: Path):
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    base_score = results[0]["final_score"] if results else 0.0

    lines = []
    lines.append(f"# 📊 BÁO CÁO ĐO LƯỜNG & CHẨN ĐOÁN BOTTLENECK ABLATION STUDY (AIC 2026)")
    lines.append(f"\n> **Thời gian cập nhật:** `{timestamp_str}`  ")
    lines.append(f"> **Tập dữ liệu kiểm chuẩn:** `data/benchmark/ground_truth.json` (11 Test Cases chuẩn BTC)  ")
    lines.append(f"> **Chẩn đoán:** So sánh trực tiếp **Stage-1 Video Recall (Retriever)** vs **Frame Recall (BTC Official Score)**\n")
    lines.append("---\n")

    lines.append("## 🏆 1. MA TRẬN CHẨN ĐOÁN HIỆU SUẤT VÀ NÚT THẮT (BOTTLENECK DIAGNOSIS MATRIX)\n")
    lines.append("| # | Cấu hình Thử nghiệm | 🏆 BTC Final Score | V-R@1 | V-R@5 | V-R@10 | V-R@20 | V-R@50 | Latency | Đánh giá Nút thắt |")
    lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

    for r in results:
        lines.append(f"| **{r['config_code']}** | {r['config_name']} | **{r['final_score']:.4f}** | {r['video_recall_1']:.1f}% | {r['video_recall_5']:.1f}% | {r['video_recall_10']:.1f}% | {r['video_recall_20']:.1f}% | {r['video_recall_50']:.1f}% | {r['latency_ms']:.0f}ms | {'Cần Reranker đưa từ Top 5 lên #1' if r['video_recall_5'] > r['video_recall_1'] else 'Retriever tốt'} |")

    lines.append("\n---\n")
    lines.append("## 🔍 2. CHI TIẾT TỪNG QUERY: SO SÁNH VIDEO RANK VS FRAME RANK\n")

    for r in results:
        lines.append(f"### 🧪 Cấu hình {r['config_code']}: {r['config_name']}\n")
        lines.append(f"- **BTC Final Score:** `{r['final_score']:.4f}` | **Video Recall@5:** `{r['video_recall_5']:.1f}%` | **Video Recall@20:** `{r['video_recall_20']:.1f}%`\n")

        lines.append("| Query ID | Task | Target Video | Video Rank | Frame Rank | R@1 | R@5 | R@20 | R@50 | Final Score | Latency |")
        lines.append("| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for rec in r["records"]:
            lines.append(f"| {rec['Query ID']} | {rec['Task']} | {rec['Target Video']} | **{rec['Video Rank']}** | **{rec['Frame Rank']}** | {rec['R@1']} | {rec['R@5']} | {rec['R@20']} | {rec['R@50']} | {rec['Final Score']} | {rec['Latency']} |")
        lines.append("\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"🎉 [ĐÃ LƯU BÁO CÁO ABLATION & CHẨN ĐOÁN] -> {output_path}", flush=True)

    # Cập nhật Leaderboard
    update_leaderboard(results, BASE_DIR / "docs" / "ABLATION_LEADERBOARD.md", datetime.now().strftime("%Y%m%d_%H%M%S"))

def update_leaderboard(current_results: list[dict], lb_path: Path, run_id: str):
    sorted_results = sorted(current_results, key=lambda x: x["final_score"], reverse=True)
    lines = [
        "# 🏆 BẢNG TỔNG SẮP CÁC CẤU HÌNH ĐẠT ĐIỂM CAO NHẤT (AIC 2026 LEADERBOARD)",
        f"\n> **Cập nhật lần cuối:** `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}` (Run ID: `{run_id}`)\n",
        "| Rank | Cấu hình | Final Score | Video-R@1 | Video-R@5 | Video-R@20 | Latency | Kết luận Chiến thuật |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |"
    ]

    for rank, r in enumerate(sorted_results, 1):
        crown = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else f"#{rank}"))
        lines.append(f"| {crown} | **{r['config_name']}** | **{r['final_score']:.4f}** | {r['video_recall_1']:.1f}% | {r['video_recall_5']:.1f}% | {r['video_recall_20']:.1f}% | {r['latency_ms']:.0f}ms | {'🔥 Cần Reranker kéo V-R@5 lên #1' if rank == 1 else 'Thử nghiệm'} |")

    with open(lb_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"🥇 [ĐÃ CẬP NHẬT BẢNG TỔNG SẮP LEADERBOARD] -> {lb_path}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Chạy Ablation Benchmark kèm chẩn đoán Stage-1 Video Recall")
    parser.add_argument("--config", type=str, default=None, help="Mã cấu hình ('0', '1', '1b', '10', '11')")
    parser.add_argument("--all_configs", action="store_true", help="Chạy toàn bộ các cấu hình để so sánh ma trận")
    args = parser.parse_args()

    pipeline_cache = {}
    md_output_path = BASE_DIR / "docs" / "ABLATION_STUDY_RESULTS.md"

    configs_to_test = ["0", "1", "1b", "10", "11"]
    if args.config is not None:
        configs_to_test = [args.config]

    print("\n🚀 BẮT ĐẦU CHẠY ĐO LƯỜNG CHẨN ĐOÁN BOTTLENECK (STAGE-1 VIDEO RECALL VS FRAME RECALL)...")
    results = []
    for cid in configs_to_test:
        res = run_ablation_experiment(cid, pipeline_cache=pipeline_cache)
        if res:
            results.append(res)

    save_ablation_markdown_report(results, md_output_path)

if __name__ == "__main__":
    main()
