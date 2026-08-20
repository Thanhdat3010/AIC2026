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

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.indexing.faiss_indexer import load_faiss_index
from src.query.text_encoder import UnifiedTextEncoder
from src.retrieval.task_specialized_engine import TaskSpecializedEngine
from src.evaluation.btc_metric import evaluate_query_predictions, summarize_ablation_metrics

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
    "test-kis-11": "Aerial flycam view looking straight down at 4 cyclists riding in a line along an asphalt road passing by a blooming purple bougainvillea tree on the right roadside.",
    "test-trake-12": "E1: Aerial flycam view looking straight down at the road surface capturing the pack of cyclists sprinting towards the finish line; E2: next, a frontal view of the dense pack of cyclists sprinting between two rows of orange-yellow banners; E3: then, a close-up of a cyclist in a blue jersey raising his fist in victory as he rides past cheering spectators; E4: finally, a slow-motion low-angle shot on the asphalt road capturing cyclists crossing the yellow finish line under the orange-red finish arch.",
    "test-kis-13": "A female reporter wearing black sunglasses, a checkered scarf around her neck, and an orange t-shirt is speaking into a microphone in front of the camera.",
    "test-qa-14": "In the scene of the lion dance orchestra in red uniforms with yellow trim playing accompaniment, what musical instrument is the person standing closest to the camera on the left playing?",
    "test-kis-15": "Close-up of a boy wearing a striped shirt with brown flaps using two wooden drumsticks to beat on the surface of a large lion dance drum in front of him.",
    "test-qa-16": "In a close-up shot of the corner of the blue protective mat placed on the stage floor, what yellow flower is placed here?",
    "test-qa-17": "In a university introduction clip, during a wide shot of the auditorium with a speaker on the stage podium, what is the name of the former Prime Minister of Israel appearing on the projection screen?",
    "test-trake-18": "E1: Aerial view looking straight down at the green and red basketball court capturing athletes playing and shooting towards the basket; E2: next, a male student in a white shirt is playing a black grand piano; E3: then, three male students sitting in a row playing acoustic guitars together; E4: finally, a male student with glasses in a white shirt is using drumsticks to play a drum kit.",
    "test-kis-19": "Two men standing side side holding a blue scholarship award banner with the text TRAO TANG QUY HOC BONG TAI NANG TRE.",
    "test-qa-20": "In a wide shot of the technology lab with a female student standing behind looking at her phone, what are the two main color tones of the humanoid robot standing on the wooden floor?"
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
    print("\n" + "=" * 115, flush=True)
    print(f"🧪 CHẠY THỬ NGHIỆM ĐO LƯỜNG & CHẨN ĐOÁN: CẤU HÌNH {config_code}", flush=True)
    print("=" * 115, flush=True)

    mode = "specialized"
    use_intra_reranker = False
    use_neighbor = True
    use_cue = False
    use_multimodal = False
    use_vlm_verification = False
    use_dense_video_refiner = False
    use_rrf = False
    use_neighbor_expansion = False
    use_multi_crop = True
    use_multi_query = False
    use_event_coverage = False
    use_row_norm_dp = False
    use_segmental_dp = False

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

    elif config_code == "11":
        config_name = "Cấu hình 11: Task-Specialized Baseline (ModalityGate + Specialized Routing)"
        mode = "specialized"
        use_intra_reranker = False
        if pipeline_cache and "task_specialized" in pipeline_cache:
            pipeline = pipeline_cache["task_specialized"]
        else:
            pipeline = TaskSpecializedEngine(engine="siglip2")
            if pipeline_cache is not None:
                pipeline_cache["task_specialized"] = pipeline

    elif config_code == "12":
        config_name = "Cấu hình 12 (E1): Config 11 + Gaussian Neighbor Temporal Support (N_i)"
        mode = "specialized"
        use_intra_reranker = True
        use_neighbor = True
        use_cue = False
        use_multimodal = False
        if pipeline_cache and "task_specialized" in pipeline_cache:
            pipeline = pipeline_cache["task_specialized"]
        else:
            pipeline = TaskSpecializedEngine(engine="siglip2")
            if pipeline_cache is not None:
                pipeline_cache["task_specialized"] = pipeline

    elif config_code == "14":
        config_name = "Cấu hình 14 (E1+E2+E3): + Time-Aligned Multi-Modal Timeline Fusion (ASR/OCR/Objects)"
        mode = "specialized"
        use_intra_reranker = True
        use_neighbor = True
        use_cue = True
        use_multimodal = True
        use_vlm_verification = False
        use_dense_video_refiner = False
        if pipeline_cache and "task_specialized" in pipeline_cache:
            pipeline = pipeline_cache["task_specialized"]
        else:
            pipeline = TaskSpecializedEngine(engine="siglip2")
            if pipeline_cache is not None:
                pipeline_cache["task_specialized"] = pipeline

    elif config_code == "15":
        config_name = "Cấu hình 15 (SOTA Master): Task-Specialized + E1 Neighbor + E3 ASR + VLM Verification + Monotonic DP"
        mode = "specialized"
        use_intra_reranker = True
        use_neighbor = True
        use_cue = True
        use_multimodal = True
        use_vlm_verification = True
        use_dense_video_refiner = False
        if pipeline_cache and "task_specialized" in pipeline_cache:
            pipeline = pipeline_cache["task_specialized"]
        else:
            pipeline = TaskSpecializedEngine(engine="siglip2")
            if pipeline_cache is not None:
                pipeline_cache["task_specialized"] = pipeline

    elif config_code == "16":
        config_name = "Cấu hình 16 (Full 3-Layer Master): Config 15 + Layer 3 Gated Dense Video Refinement (OpenCV Vi Sai)"
        mode = "specialized"
        use_intra_reranker = True
        use_neighbor = True
        use_cue = True
        use_multimodal = True
        use_vlm_verification = True
        use_dense_video_refiner = True
        if pipeline_cache and "task_specialized" in pipeline_cache:
            pipeline = pipeline_cache["task_specialized"]
        else:
            pipeline = TaskSpecializedEngine(engine="siglip2")
            if pipeline_cache is not None:
                pipeline_cache["task_specialized"] = pipeline

    elif config_code == "17":
        config_name = "Cấu hình 17 (SOTA Master 2026): Config 15 + RRF Multimodal + Neighbor Expansion + Multi-Crop QA + Vectorized DP"
        mode = "specialized"
        use_intra_reranker = True
        use_neighbor = True
        use_cue = True
        use_multimodal = True
        use_vlm_verification = True
        use_dense_video_refiner = False
        use_rrf = True
        use_neighbor_expansion = True
        use_multi_query = False
        use_event_coverage = False
        use_row_norm_dp = False
        if pipeline_cache and "task_specialized" in pipeline_cache:
            pipeline = pipeline_cache["task_specialized"]
        else:
            pipeline = TaskSpecializedEngine(engine="siglip2")
            if pipeline_cache is not None:
                pipeline_cache["task_specialized"] = pipeline

    elif config_code == "18":
        config_name = "Cấu hình 18 (Baseline Toàn Diện): Config 17 + Optimized DP (TRAKE) + Tri-modal QA (Adaptive Gating)"
        mode = "specialized"
        use_intra_reranker = True
        use_neighbor = True
        use_cue = True
        use_multimodal = True
        use_vlm_verification = True
        use_dense_video_refiner = False
        use_rrf = True
        use_neighbor_expansion = True
        use_multi_crop = True
        use_multi_query = False
        use_event_coverage = False
        use_row_norm_dp = False
        if pipeline_cache and "task_specialized" in pipeline_cache:
            pipeline = pipeline_cache["task_specialized"]
        else:
            pipeline = TaskSpecializedEngine(engine="siglip2")
            if pipeline_cache is not None:
                pipeline_cache["task_specialized"] = pipeline

    elif config_code == "19":
        config_name = "Cấu hình 19 (Ablation 1): Config 18 + Multi-Query FAISS Union (TRAKE Top-50)"
        mode = "specialized"
        use_intra_reranker = True
        use_neighbor = True
        use_cue = True
        use_multimodal = True
        use_vlm_verification = True
        use_dense_video_refiner = False
        use_rrf = True
        use_neighbor_expansion = True
        use_multi_crop = True
        use_multi_query = True
        use_event_coverage = False
        use_row_norm_dp = False
        if pipeline_cache and "task_specialized" in pipeline_cache:
            pipeline = pipeline_cache["task_specialized"]
        else:
            pipeline = TaskSpecializedEngine(engine="siglip2")
            if pipeline_cache is not None:
                pipeline_cache["task_specialized"] = pipeline

    elif config_code == "20":
        config_name = "Cấu hình 20 (Ablation 2): Config 19 + Temporal NMS Event Coverage & Soft-Min (TRAKE)"
        mode = "specialized"
        use_intra_reranker = True
        use_neighbor = True
        use_cue = True
        use_multimodal = True
        use_vlm_verification = True
        use_dense_video_refiner = False
        use_rrf = True
        use_neighbor_expansion = True
        use_multi_crop = True
        use_multi_query = True
        use_event_coverage = True
        use_row_norm_dp = False
        if pipeline_cache and "task_specialized" in pipeline_cache:
            pipeline = pipeline_cache["task_specialized"]
        else:
            pipeline = TaskSpecializedEngine(engine="siglip2")
            if pipeline_cache is not None:
                pipeline_cache["task_specialized"] = pipeline

    elif config_code == "21":
        config_name = "Cấu hình 21 (Ablation 3): Config 20 + Row-Normalized Monotonic DP (TRAKE)"
        mode = "specialized"
        use_intra_reranker = True
        use_neighbor = True
        use_cue = True
        use_multimodal = True
        use_vlm_verification = True
        use_dense_video_refiner = False
        use_rrf = True
        use_neighbor_expansion = True
        use_multi_crop = True
        use_multi_query = True
        use_event_coverage = True
        use_row_norm_dp = True
        if pipeline_cache and "task_specialized" in pipeline_cache:
            pipeline = pipeline_cache["task_specialized"]
        else:
            pipeline = TaskSpecializedEngine(engine="siglip2")
            if pipeline_cache is not None:
                pipeline_cache["task_specialized"] = pipeline

    elif config_code == "22":
        config_name = "Cấu hình 22 (Ablation 4): Config 21 + Segmental Dynamic Programming (TRAKE)"
        mode = "specialized"
        use_intra_reranker = True
        use_neighbor = True
        use_cue = True
        use_multimodal = True
        use_vlm_verification = True
        use_dense_video_refiner = False
        use_rrf = True
        use_neighbor_expansion = True
        use_multi_crop = True
        use_multi_query = True
        use_event_coverage = True
        use_row_norm_dp = True
        use_segmental_dp = True
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
    all_eval_results = []

    for c_idx, case in enumerate(test_cases, 1):
        qid = case["query_id"]
        ttype = case["task_type"]
        qtext = case["query_text"]
        gt = case["ground_truth"]

        print(f"\n[{c_idx:02d}/{len(test_cases):02d}] 🎯 BẮT ĐẦU: {qid:13s} ({ttype.upper():5s}) | \"{qtext[:70]}...\"", flush=True)

        en_query = BENCHMARK_TRANSLATIONS.get(qid, qtext)

        if mode == "dense":
            preds, latency = pipeline.search(en_query, top_k=100)
        elif mode == "specialized":
            if ttype == "kis":
                preds, qinfo, latency = pipeline.search_kis(
                    query_text=qtext,
                    top_k=100,
                    use_intra_reranker=use_intra_reranker,
                    use_neighbor=True,
                    use_cue=False,
                    use_multimodal=False,
                    use_vlm_verification=use_vlm_verification,
                    use_dense_video_refiner=use_dense_video_refiner,
                    use_rrf=use_rrf,
                    use_neighbor_expansion=use_neighbor_expansion
                )
            elif ttype == "qa":
                preds, qinfo, latency = pipeline.search_qa(
                    query_text=qtext,
                    top_k=100,
                    use_intra_reranker=use_intra_reranker,
                    use_neighbor=True,
                    use_cue=True,
                    use_multimodal=True,
                    use_rrf=use_rrf,
                    use_multi_crop=use_multi_crop
                )
            else:
                preds, qinfo, latency = pipeline.search_trake(
                    query_text=qtext,
                    top_k=100,
                    use_multi_query=use_multi_query,
                    use_event_coverage=use_event_coverage,
                    use_row_norm_dp=use_row_norm_dp,
                    use_segmental_dp=use_segmental_dp
                )

        latencies.append(latency)
        eval_res = evaluate_query_predictions(
            predictions=preds,
            ground_truth=gt,
            task_type=ttype,
            check_qa_answer=True,
            question_text=qtext,
            use_llm_judge=True
        )
        eval_res["task"] = ttype.upper()
        all_eval_results.append(eval_res)

        fs = eval_res["final_score"]
        pos_rank = f"#{eval_res['first_pos_rank']}" if eval_res['first_pos_rank'] != -1 else "MISS"
        perf_rank = f"#{eval_res['first_perfect_rank']}" if eval_res['first_perfect_rank'] != -1 else "MISS"
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
            "Pos Rank": pos_rank,
            "Perf Rank": perf_rank,
            "R@1": f"{eval_res['r_at_k']['R@1']:.2f}",
            "R@5": f"{eval_res['r_at_k']['R@5']:.2f}",
            "R@20": f"{eval_res['r_at_k']['R@20']:.2f}",
            "R@50": f"{eval_res['r_at_k']['R@50']:.2f}",
            "R@100": f"{eval_res['r_at_k']['R@100']:.2f}",
            "Final Score": f"{fs:.4f}",
            "Latency": f"{latency:.0f}ms",
            "Error Type": eval_res["error_type"],
            "top_candidates": [
                {
                    "rank": r,
                    "video_id": p.get("video_id", ""),
                    "frame_idx": int(p.get("frame_idx", 0)),
                    "score": float(p.get("score", 0.0)),
                    "answer": p.get("answer", "")
                } for r, p in enumerate(preds[:20], 1)
            ]
        })
        print(f"   ▶ [{len(records):02d}/{len(test_cases):02d}] {qid:13s} ({ttype.upper():5s}) -> Video: {video_hit:4s} | Pos: {pos_rank:4s} | Perf: {perf_rank:4s} | Score: {fs:.4f} | {eval_res['error_type']} ({latency:.0f}ms)", flush=True)

    print("\n" + "=" * 115)
    df_res = pd.DataFrame([{k: v for k, v in r.items() if k != "top_candidates"} for r in records])
    print(df_res.to_string(index=False))
    print("\n" + "-" * 115)

    macro_summary = summarize_ablation_metrics(all_eval_results)
    overall_final = macro_summary["macro_final_score"]
    avg_latency = np.mean(latencies)

    print(f"📊 BẢNG TỔNG KẾT & CHẨN ĐOÁN BOTTLENECK CẤU HÌNH {config_code}:")
    for t_name, t_info in macro_summary.get("task_summary", {}).items():
        print(f"   • {t_name:5s} Score ({t_info['count']:02d} queries)     : {t_info['macro_score']:.4f} (R@1: {t_info['r1']:.2f}, R@5: {t_info['r5']:.2f}, R@20: {t_info['r20']:.2f}, R@50: {t_info['r50']:.2f}, R@100: {t_info['r100']:.2f})")
    print(f"   • 🏆 BTC FINAL SCORE (Macro)   : {overall_final:.4f}")
    print(f"   ---------------------------------------------------------------------------------------")
    print(f"   🔍 STAGE-1 VIDEO-LEVEL RECALL & MRR (RETRIEVER HEALTH):")
    print(f"      ▶ Video MRR       : {macro_summary['video_mrr']:.4f}")
    for k in [1, 5, 10, 20, 50, 100]:
        print(f"      ▶ Video Recall@{k:<3d}: {macro_summary['video_recalls'][f'V-R@{k}']:.1f}%")
    print(f"   ---------------------------------------------------------------------------------------")
    print(f"   🏷️ PHÂN BỐ ĐIỂM SỐ (SCORE BUCKETS):")
    b_str = ", ".join([f"{k}: {v} queries" for k, v in macro_summary["score_buckets"].items()])
    print(f"      ▶ {b_str}")
    print(f"   ---------------------------------------------------------------------------------------")
    print(f"   ❌ PHÂN LOẠI LỖI (ERROR TAXONOMY COUNTS):")
    for err, cnt in macro_summary["error_counts"].items():
        print(f"      ▶ {err:<30s}: {cnt:02d} queries")
    print(f"   • ⚡ Average Query Latency    : {avg_latency:.2f} ms")
    print("=" * 115 + "\n")

    res_dict = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_code": config_code,
        "config_name": config_name,
        "kis_score": float(macro_summary["task_summary"].get("KIS", {}).get("macro_score", 0.0)),
        "qa_score": float(macro_summary["task_summary"].get("QA", {}).get("macro_score", 0.0)),
        "trake_score": float(macro_summary["task_summary"].get("TRAKE", {}).get("macro_score", 0.0)),
        "final_score": float(overall_final),
        "video_mrr": float(macro_summary["video_mrr"]),
        "video_recall_1": float(macro_summary["video_recalls"]["V-R@1"]),
        "video_recall_5": float(macro_summary["video_recalls"]["V-R@5"]),
        "video_recall_10": float(macro_summary["video_recalls"]["V-R@10"]),
        "video_recall_20": float(macro_summary["video_recalls"]["V-R@20"]),
        "video_recall_50": float(macro_summary["video_recalls"]["V-R@50"]),
        "video_recall_100": float(macro_summary["video_recalls"]["V-R@100"]),
        "score_buckets": macro_summary["score_buckets"],
        "error_counts": macro_summary["error_counts"],
        "latency_ms": float(avg_latency),
        "records": records
    }

    # Lưu kết quả Benchmark động cho Streamlit
    latest_json_path = BASE_DIR / "data" / "benchmark" / "latest_ablation_results.json"
    latest_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(latest_json_path, "w", encoding="utf-8") as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=2)
    print(f"💾 [ĐÃ LƯU KẾT QUẢ BENCHMARK ĐỘNG] -> {latest_json_path}", flush=True)

    return res_dict

def save_ablation_markdown_report(results: list[dict], output_path: Path):
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append(f"# 📊 BÁO CÁO ĐO LƯỜNG & CHẨN ĐOÁN BOTTLENECK ABLATION STUDY (AIC 2026)")
    lines.append(f"\n> **Thời gian cập nhật:** `{timestamp_str}`  ")
    total_cases = len(results[0]["records"]) if results else 20
    lines.append(f"> **Tập dữ liệu kiểm chuẩn:** `data/benchmark/ground_truth.json` ({total_cases} Test Cases chuẩn BTC)  ")
    lines.append(f"> **Chẩn đoán:** So sánh trực tiếp **Stage-1 Video Recall (Retriever)** vs **Frame Recall (BTC Official Score)**\n")
    lines.append("---\n")

    lines.append("## 🏆 1. MA TRẬN CHẨN ĐOÁN HIỆU SUẤT VÀ NÚT THẮT (BOTTLENECK DIAGNOSIS MATRIX)\n")
    lines.append("| # | Cấu hình Thử nghiệm | 🏆 BTC Final Score | V-MRR | V-R@1 | V-R@5 | V-R@20 | Latency | Đánh giá Nút thắt |")
    lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

    for r in results:
        lines.append(f"| **{r['config_code']}** | {r['config_name']} | **{r['final_score']:.4f}** | {r.get('video_mrr', 0.0):.4f} | {r['video_recall_1']:.1f}% | {r['video_recall_5']:.1f}% | {r['video_recall_20']:.1f}% | {r['latency_ms']:.0f}ms | {'🔥 KHUYÊN DÙNG THI ĐẤU' if r['final_score'] >= 0.65 else 'Thử nghiệm'} |")

    lines.append("\n---\n")
    lines.append("## 🔍 2. CHI TIẾT TỪNG QUERY: SO SÁNH VIDEO RANK VS FRAME RANK\n")

    for r in results:
        lines.append(f"### 🧪 Cấu hình {r['config_code']}: {r['config_name']}\n")
        lines.append(f"- **BTC Final Score:** `{r['final_score']:.4f}` | **Video MRR:** `{r.get('video_mrr', 0.0):.4f}` | **Video Recall@5:** `{r['video_recall_5']:.1f}%`\n")

        lines.append("| Query ID | Task | Target Video | Video Rank | Pos Rank | Perf Rank | R@1 | R@5 | R@20 | R@50 | R@100 | Final Score | Latency | Error Type |")
        lines.append("| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
        for rec in r["records"]:
            lines.append(f"| {rec['Query ID']} | {rec['Task']} | {rec['Target Video']} | **{rec['Video Rank']}** | {rec['Pos Rank']} | {rec['Perf Rank']} | {rec['R@1']} | {rec['R@5']} | {rec['R@20']} | {rec['R@50']} | {rec['R@100']} | **{rec['Final Score']}** | {rec['Latency']} | `{rec['Error Type']}` |")
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
        "| Rank | Cấu hình | Final Score | Video MRR | Video-R@1 | Video-R@5 | Video-R@20 | Latency | Kết luận Chiến thuật |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |"
    ]

    for rank, r in enumerate(sorted_results, 1):
        crown = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else f"#{rank}"))
        lines.append(f"| {crown} | **{r['config_name']}** | **{r['final_score']:.4f}** | {r.get('video_mrr', 0.0):.4f} | {r['video_recall_1']:.1f}% | {r['video_recall_5']:.1f}% | {r['video_recall_20']:.1f}% | {r['latency_ms']:.0f}ms | {'🔥 KHUYÊN DÙNG THI ĐẤU' if rank == 1 else 'Thử nghiệm'} |")

    with open(lb_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"🥇 [ĐÃ CẬP NHẬT BẢNG TỔNG SẮP LEADERBOARD] -> {lb_path}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Chạy Ablation Benchmark kèm chẩn đoán Stage-1 Video Recall")
    parser.add_argument("--config", type=str, default=None, help="Mã cấu hình ('18', '19', '20', '21', '22')")
    parser.add_argument("--all_configs", "--all", action="store_true", help="Chạy toàn bộ cấu hình từ 0 đến 22")
    parser.add_argument("--v4", "--v4_ablation", action="store_true", help="Chạy ablation study từ Config 18 đến 22")
    args = parser.parse_args()

    pipeline_cache = {}
    md_output_path = BASE_DIR / "docs" / "ABLATION_STUDY_RESULTS.md"

    if args.v4:
        configs_to_test = ["18", "19", "20", "21", "22"]
    elif args.all_configs:
        configs_to_test = ["0", "1", "11", "12", "14", "15", "16", "17", "18", "19", "20", "21", "22"]
    elif args.config is not None:
        configs_to_test = [args.config]
    else:
        configs_to_test = ["21"]

    print(f"\n🚀 BẮT ĐẦU CHẠY THỬ NGHIỆM ABLATION STUDY TRÊN 20 TEST CASES...")
    print(f"[*] Danh sách cấu hình kiểm tra: {configs_to_test}\n")
    results = []
    for cid in configs_to_test:
        res = run_ablation_experiment(cid, pipeline_cache=pipeline_cache)
        if res:
            results.append(res)

    save_ablation_markdown_report(results, md_output_path)

if __name__ == "__main__":
    main()
