import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.retrieval.unified_search_core import UnifiedSearchCore
from src.query.llm_query_refiner import LLMQueryRefiner
from src.tasks.clean_task_handlers import KISHandler, QAHandler, TRAKEHandler
from src.evaluation.btc_metric import evaluate_query_predictions

# Dictionary dịch thô phục vụ các baseline không dùng LLM
RAW_TRANSLATIONS = {
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

def run_configuration(config_id: str, search_core: UnifiedSearchCore, refiner: LLMQueryRefiner, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    print("\n" + "=" * 100, flush=True)
    print(f"🚀 BẮT ĐẦU ĐO LƯỜNG CẤU HÌNH [{config_id}] TRÊN {len(test_cases)} TEST CASES...", flush=True)
    print("=" * 100, flush=True)

    kis_handler = KISHandler(search_core, refiner)
    qa_handler = QAHandler(search_core, refiner)
    trake_handler = TRAKEHandler(search_core, refiner)

    query_results = []
    latencies = []

    for idx, tc in enumerate(test_cases, 1):
        qid = tc["query_id"]
        ttype = tc["task_type"]
        qtext = tc["query_text"]
        gt = tc["ground_truth"]

        raw_en = RAW_TRANSLATIONS.get(qid, qtext)
        t_start = time.time()

        if config_id in ["A0", "B1"]:
            # A0 (Baseline): Pure Standard SigLIP-2 Tiếng Việt gốc
            vec = search_core.encode_text(qtext)
            hits = search_core.search_visual(vec, top_k=100)
            if ttype == "trake":
                preds = []
                for rank, h in enumerate(hits[:100], 1):
                    f0 = h["frame_idx"]
                    preds.append({"video_id": h["video_id"], "events": [str(f0), str(f0+25), str(f0+50)], "event_frames": [f0, f0+25, f0+50], "rank": rank})
            else:
                preds = hits

        elif config_id in ["A1", "A2", "A3"]:
            # A1..A3: Đo lường nâng cấp KIS
            if ttype == "kis":
                preds, _, _ = kis_handler.search(qtext, top_k=100, config_name=config_id)
            elif ttype == "qa":
                preds, _, _ = qa_handler.search(qtext, top_k=100, config_name=config_id)
            elif ttype == "trake":
                preds, _, _ = trake_handler.search(qtext, top_k=100, config_name=config_id)

        elif config_id == "A4":
            # A4: Kích hoạt Unified Multimodal QA Solver
            if ttype == "kis":
                preds, _, _ = kis_handler.search(qtext, top_k=100, config_name="A3")
            elif ttype == "qa":
                preds, _, _ = qa_handler.search(qtext, top_k=100, config_name="A4")
            elif ttype == "trake":
                preds, _, _ = trake_handler.search(qtext, top_k=100, config_name="A4")

        elif config_id in ["A5", "A6", "M5"]:
            # A5..A6 / M5: Kích hoạt toàn bộ KIS + Unified QA + Joint Coverage TRAKE DP
            if ttype == "kis":
                preds, _, _ = kis_handler.search(qtext, top_k=100, config_name="A6")
            elif ttype == "qa":
                preds, _, _ = qa_handler.search(qtext, top_k=100, config_name="A6")
            elif ttype == "trake":
                preds, _, _ = trake_handler.search(qtext, top_k=100, config_name="A6")
            else:
                preds, _, _ = kis_handler.search(qtext, top_k=100, config_name="A6")

        elif config_id in ["A6_1", "A6_2", "A6_3", "A6_4", "A7", "A8", "A9", "A10_FINAL"]:
            # A6_1..A6_4 & A7, A8, A9, A10_FINAL: Các nhánh thực nghiệm độc lập và Grand Master
            if ttype == "kis":
                preds, _, _ = kis_handler.search(qtext, top_k=100, config_name=config_id)
            elif ttype == "qa":
                preds, _, _ = qa_handler.search(qtext, top_k=100, config_name=config_id)
            elif ttype == "trake":
                preds, _, _ = trake_handler.search(qtext, top_k=100, config_name=config_id)
            else:
                preds, _, _ = kis_handler.search(qtext, top_k=100, config_name=config_id)

        latency = (time.time() - t_start) * 1000
        latencies.append(latency)

        # Đánh giá câu hiện tại
        eval_res = evaluate_query_predictions(preds, gt, ttype)
        eval_res["query_id"] = qid
        eval_res["task_type"] = ttype
        eval_res["latency_ms"] = latency
        eval_res["score"] = eval_res["final_score"]
        query_results.append(eval_res)

        vid_rank_str = f"#{eval_res['video_hit_rank']}" if eval_res.get('video_hit_rank', -1) > 0 else "MISS"
        pos_rank_str = f"#{eval_res['first_pos_rank']}" if eval_res.get('first_pos_rank', -1) > 0 else "MISS"
        print(f"[{idx:02d}/{len(test_cases)}] {qid:12s} ({ttype.upper():5s}) -> Video: {vid_rank_str:6s} | Pos: {pos_rank_str:6s} | Score: {eval_res['score']:.4f} | {eval_res.get('error_type', '')} ({latency:.0f}ms)", flush=True)

    # Tổng hợp metrics
    df = pd.DataFrame(query_results)
    kis_df = df[df["task_type"] == "kis"]
    qa_df = df[df["task_type"] == "qa"]
    trake_df = df[df["task_type"] == "trake"]

    kis_score = float(kis_df["score"].mean()) if not kis_df.empty else 0.0
    qa_score = float(qa_df["score"].mean()) if not qa_df.empty else 0.0
    trake_score = float(trake_df["score"].mean()) if not trake_df.empty else 0.0
    macro_score = float(df["score"].mean())

    # Video recall
    v_r1 = float((df["video_hit_rank"] == 1).mean())
    v_r5 = float(((df["video_hit_rank"] > 0) & (df["video_hit_rank"] <= 5)).mean())
    v_r20 = float(((df["video_hit_rank"] > 0) & (df["video_hit_rank"] <= 20)).mean())
    v_r100 = float(((df["video_hit_rank"] > 0) & (df["video_hit_rank"] <= 100)).mean())

    summary = {
        "config_id": config_id,
        "total_queries": len(test_cases),
        "kis_score": kis_score,
        "qa_score": qa_score,
        "trake_score": trake_score,
        "macro_score": macro_score,
        "video_r1": v_r1,
        "video_r5": v_r5,
        "video_r20": v_r20,
        "video_r100": v_r100,
        "avg_latency_ms": float(np.mean(latencies))
    }

    print("\n" + "-" * 80)
    print(f"📊 KẾT QUẢ CẤU HÌNH [{config_id}]:")
    print(f"   • KIS Score   : {kis_score:.4f}")
    print(f"   • QA Score    : {qa_score:.4f}")
    print(f"   • TRAKE Score : {trake_score:.4f}")
    print(f"   • 🏆 MACRO SCORE: {macro_score:.4f}")
    print(f"   • Video Recall@1/5/20/100: {v_r1:.1%} / {v_r5:.1%} / {v_r20:.1%} / {v_r100:.1%}")
    print(f"   • Độ trễ trung bình: {summary['avg_latency_ms']:.1f} ms")
    print("-" * 80)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Chạy Ablation Study trên Ground Truth AIC 2026")
    parser.add_argument("--config", type=str, default="all", help="Mã cấu hình (A1, A2, A3, A4, A5, A6, A7, hoặc all)")
    parser.add_argument("--batch", type=str, default="batch_1", help="Batch dữ liệu")
    args = parser.parse_args()

    gt_file = BASE_DIR / "data" / "benchmark" / "ground_truth.json"
    with open(gt_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    test_cases = gt_data["test_cases"]

    search_core = UnifiedSearchCore(engine="siglip2", batch=args.batch)
    refiner = LLMQueryRefiner()

    if args.config == "incremental":
        configs_to_run = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
    elif args.config == "all":
        configs_to_run = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "M5"]
    else:
        configs_to_run = [args.config]

    out_file = BASE_DIR / "data" / "benchmark" / "ablation_study_summary.json"
    existing_summaries = {}
    if out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                old_list = json.load(f)
                for s in old_list:
                    existing_summaries[s["config_id"]] = s
        except Exception:
            pass

    for cfg in configs_to_run:
        res = run_configuration(cfg, search_core, refiner, test_cases)
        existing_summaries[cfg] = res

    all_summaries = list(existing_summaries.values())

    print("\n" + "=" * 100)
    print("🏆 BẢNG TỔNG SẮP SO SÁNH ĐỐI ĐẦU ABLATION STUDY (GROUND TRUTH 47 CÂU):")
    print("=" * 100)
    print(f"{'Config':8s} | {'KIS Score':10s} | {'QA Score':10s} | {'TRAKE Score':12s} | {'Macro Score':12s} | {'Video-R@20':11s} | {'Latency':10s}")
    print("-" * 100)
    for s in all_summaries:
        print(f"{s['config_id']:8s} | {s['kis_score']:10.4f} | {s['qa_score']:10.4f} | {s['trake_score']:12.4f} | {s['macro_score']:12.4f} | {s['video_r20']:10.1%} | {s['avg_latency_ms']:8.1f}ms")
    print("=" * 100)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    print(f"✅ Đã lưu kết quả tóm tắt vào: {out_file}")

if __name__ == "__main__":
    main()
