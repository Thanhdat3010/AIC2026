import os
import sys
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

# Force UTF-8 on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.indexing.faiss_indexer import load_faiss_index
from src.indexing.bm25_indexer import BM25MultiIndexer
from src.query.text_encoder import UnifiedTextEncoder
from src.query.gemini_router import GeminiQueryRouter
from src.reranking.temporal_smoothing import TemporalSceneSmoother
from src.reranking.soft_filter import SoftVideoFilter
from src.tasks.qa_agent import VisualQAAgent
from src.tasks.trake_agent import TRAKEAlignmentAgent

class HybridRetrievalEngine:
    """
    Bộ máy Tìm kiếm Lai Đa Phương Thức Toàn Diện (SOTA Multimodal Hybrid Retrieval Engine):
    - Động cơ lõi: Google Gemini 3.5 Flash Lite + Adaptive Modality Gating
    - Dominant Prompt Fusion: Kết hợp 70% Primary + 15% Action + 15% Objects
    - 4 nguồn tìm kiếm: Dense FAISS (SigLIP 2/CLIP) + BM25 OCR + BM25 ASR + BM25 Meta
    - Specialized Task Agents: Two-Stage Visual QA Agent & TRAKE Alignment Agent
    """
    def __init__(self, engine: str = "siglip2", batch: str = "batch_1", k_rrf: int = 60):
        self.engine = engine
        self.batch = batch
        self.k_rrf = k_rrf

        print(f"[*] Đang khởi tạo Hybrid Engine [{engine.upper()}]...", flush=True)
        self.text_encoder = UnifiedTextEncoder(engine=engine)
        self.faiss_index, self.df_frames = load_faiss_index(engine=engine, batch=batch)
        self.bm25_indexer = BM25MultiIndexer(batch=batch)
        self.llm_router = GeminiQueryRouter()
        self.smoother = TemporalSceneSmoother()
        self.soft_filter = SoftVideoFilter()
        self.qa_agent = VisualQAAgent(key_pool=self.llm_router.key_pool)
        self.trake_agent = TRAKEAlignmentAgent(engine=engine, batch=batch)
        print(f"✅ Hybrid Retrieval Engine [{engine.upper()}] đã sẵn sàng!", flush=True)

    def search(
        self,
        raw_query: str,
        top_k: int = 100,
        use_multi_prompt: bool = True,
        use_dominant_weights: bool = True,
        use_adaptive_gating: bool = True,
        use_ocr: bool = False,
        use_asr: bool = False,
        use_meta: bool = False,
        use_temporal_smoothing: bool = False,
        use_soft_filter: bool = False,
        use_qa_agent: bool = False,
        use_trake_agent: bool = False,
        custom_en_query: str = None
    ) -> tuple[list[dict], dict, float]:
        """
        Thực hiện tìm kiếm lai đa phương thức với đầy đủ các nâng cấp SOTA.
        Returns: (ranked_results, query_analysis_dict, latency_ms)
        """
        t0 = time.time()

        # 1. Phân rã truy vấn qua Gemini 3.5 Flash Lite
        if custom_en_query:
            query_info = {
                "visual_prompts": [custom_en_query],
                "has_ocr_signal": False,
                "ocr_keywords": [],
                "has_asr_signal": False,
                "asr_keywords": [],
                "is_qa": False,
                "is_trake": False,
                "trake_events": [],
                "weights": {"visual": 1.0, "ocr": 0.0, "asr": 0.0},
                "temporal_hint": "any"
            }
        else:
            query_info = self.llm_router.transform_query(raw_query)

        # 2. Xử lý Trọng số theo Adaptive Modality Gating
        if use_adaptive_gating:
            w_vis = 1.0
            w_ocr = 0.35 if (use_ocr and query_info.get("has_ocr_signal", False)) else 0.0
            w_asr = 0.35 if (use_asr and query_info.get("has_asr_signal", False)) else 0.0
        else:
            w_vis = 1.0
            w_ocr = 0.3 if use_ocr else 0.0
            w_asr = 0.3 if use_asr else 0.0

        frame_scores = defaultdict(lambda: {"score": 0.0, "ranks": {}, "video_id": "", "frame_idx": 0, "global_id": -1})

        # ======================================================================
        # 3. DENSE RETRIEVAL: Dominant Prompt Fusion (70% Main, 15% Action, 15% Props)
        # ======================================================================
        vis_prompts = query_info["visual_prompts"]
        
        if not use_multi_prompt or len(vis_prompts) == 1:
            q_vec = self.text_encoder.encode_text(vis_prompts[0])
        else:
            if use_dominant_weights and len(vis_prompts) >= 3:
                # 70% Primary + 15% Action + 15% Objects
                v_main = self.text_encoder.encode_text(vis_prompts[0])
                v_act = self.text_encoder.encode_text(vis_prompts[1])
                v_obj = self.text_encoder.encode_text(vis_prompts[2])
                combined_vec = 0.70 * v_main + 0.15 * v_act + 0.15 * v_obj
                q_vec = combined_vec / np.linalg.norm(combined_vec, axis=-1, keepdims=True)
            else:
                vecs = [self.text_encoder.encode_text(p) for p in vis_prompts]
                avg_vec = np.mean(vecs, axis=0)
                q_vec = avg_vec / np.linalg.norm(avg_vec, axis=-1, keepdims=True)

        dense_top_k = min(top_k * 4, self.faiss_index.ntotal)
        scores_d, indices_d = self.faiss_index.search(q_vec, dense_top_k)

        for rank, (sim, idx) in enumerate(zip(scores_d[0], indices_d[0]), 1):
            row = self.df_frames.iloc[idx]
            v_id = row["video_id"]
            f_idx = int(row["frame_idx"])
            key = (v_id, f_idx)
            
            rrf_contrib = w_vis / (self.k_rrf + rank)
            frame_scores[key]["score"] += rrf_contrib
            frame_scores[key]["ranks"]["visual"] = rank
            frame_scores[key]["video_id"] = v_id
            frame_scores[key]["frame_idx"] = f_idx
            frame_scores[key]["global_id"] = int(row["global_id"])

        # ======================================================================
        # 4. SPARSE RETRIEVAL: BM25 OCR (Chỉ kích hoạt khi Gating mở)
        # ======================================================================
        if w_ocr > 0.0 and query_info.get("ocr_keywords"):
            ocr_query = " ".join(query_info["ocr_keywords"])
            ocr_results = self.bm25_indexer.search_ocr(ocr_query, top_k=top_k * 2)
            for rank, doc in enumerate(ocr_results, 1):
                v_id = doc["video_id"]
                f_idx = doc["frame_idx"]
                if f_idx < 0:
                    continue
                key = (v_id, f_idx)
                rrf_contrib = w_ocr / (self.k_rrf + rank)
                frame_scores[key]["score"] += rrf_contrib
                frame_scores[key]["ranks"]["ocr"] = rank
                frame_scores[key]["video_id"] = v_id
                frame_scores[key]["frame_idx"] = f_idx

        # ======================================================================
        # 5. SPARSE RETRIEVAL: BM25 ASR (Chỉ kích hoạt khi Gating mở)
        # ======================================================================
        if w_asr > 0.0 and query_info.get("asr_keywords"):
            asr_query = " ".join(query_info["asr_keywords"])
            asr_results = self.bm25_indexer.search_asr(asr_query, top_k=top_k * 2)
            for rank, doc in enumerate(asr_results, 1):
                v_id = doc["video_id"]
                f_idx = doc["frame_idx"]
                key = (v_id, f_idx)
                rrf_contrib = w_asr / (self.k_rrf + rank)
                frame_scores[key]["score"] += rrf_contrib
                frame_scores[key]["ranks"]["asr"] = rank
                frame_scores[key]["video_id"] = v_id
                frame_scores[key]["frame_idx"] = f_idx

        # ======================================================================
        # 6. SẮP XẾP SƠ BỘ CANDIDATES
        # ======================================================================
        candidate_list = list(frame_scores.values())
        candidate_list.sort(key=lambda x: x["score"], reverse=True)
        initial_top = candidate_list[:top_k * 4]

        formatted_candidates = []
        for rank, cand in enumerate(initial_top, 1):
            formatted_candidates.append({
                "rank": rank,
                "video_id": cand["video_id"],
                "frame_idx": cand["frame_idx"],
                "global_id": cand["global_id"],
                "score": float(cand["score"]),
                "matched_modalities": cand["ranks"]
            })

        # ======================================================================
        # 7. HẬU XỬ LÝ (RE-RANKING & SMOOTHING)
        # ======================================================================
        if use_temporal_smoothing:
            formatted_candidates = self.smoother.smooth_and_rerank(formatted_candidates, top_k=top_k * 2)

        if use_soft_filter:
            t_hint = query_info.get("temporal_hint", "any")
            formatted_candidates = self.soft_filter.apply_temporal_hint(formatted_candidates, t_hint)

        # ======================================================================
        # 8. TWO-STAGE TASK AGENTS (Visual QA & TRAKE Alignment)
        # ======================================================================
        if use_qa_agent and query_info.get("is_qa", False):
            ans_text, formatted_candidates = self.qa_agent.answer_and_rerank(
                raw_query, formatted_candidates, max_inspect_frames=4
            )
            query_info["generated_qa_answer"] = ans_text

        final_results = formatted_candidates[:top_k]
        for rank, cand in enumerate(final_results, 1):
            cand["rank"] = rank

        latency = (time.time() - t0) * 1000
        return final_results, query_info, latency

if __name__ == "__main__":
    engine = HybridRetrievalEngine("siglip2")
    q = "Khi 2 người đàn ông đang di chuyển chiếc xe máy chở nhiều măng le, người phía trước đội gì trên đầu?"
    print(f"\n🔎 [TEST FULL PIPELINE SEARCH]: {q}")
    results, q_info, latency = engine.search(
        q,
        top_k=5,
        use_multi_prompt=True,
        use_dominant_weights=True,
        use_adaptive_gating=True,
        use_qa_agent=True
    )
    print(f"⚡ Thời gian xử lý: {latency:.2f} ms")
    print(f"🎯 Câu trả lời QA: {q_info.get('generated_qa_answer')}")
    print(f"🏆 Top 3 kết quả:")
    for r in results[:3]:
        print(f"   + Rank #{r['rank']} | Video: {r['video_id']} | Frame: {r['frame_idx']} | Score: {r['score']:.5f} | QA Ans: {r.get('qa_answer')}")
