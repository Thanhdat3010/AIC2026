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

class HybridRetrievalEngine:
    """
    Bộ máy Tìm kiếm Lai Đa Phương Thức (Multimodal Hybrid Retrieval Engine):
    - Tích hợp 4 nguồn: Dense FAISS (SigLIP 2/CLIP) + BM25 OCR + BM25 ASR + BM25 Meta
    - Thuật toán Late Fusion: Reciprocal Rank Fusion (RRF)
    - Tự động lấy trọng số tối ưu từ Gemini 3.5 Flash Lite
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
        print(f"✅ Hybrid Retrieval Engine [{engine.upper()}] đã sẵn sàng!", flush=True)

    def search(
        self,
        raw_query: str,
        top_k: int = 100,
        use_multi_prompt: bool = True,
        use_ocr: bool = True,
        use_asr: bool = True,
        use_dynamic_weights: bool = True,
        custom_en_query: str = None
    ) -> tuple[list[dict], dict, float]:
        """
        Thực hiện tìm kiếm lai đa phương thức toàn diện:
        Returns: (ranked_results, query_analysis_dict, latency_ms)
        """
        t0 = time.time()

        # 1. Phân rã truy vấn qua Gemini 3.5 Flash Lite
        if custom_en_query:
            query_info = {
                "visual_prompts": [custom_en_query],
                "ocr_keywords": [],
                "asr_keywords": [],
                "weights": {"visual": 1.0, "ocr": 0.0, "asr": 0.0},
                "temporal_hint": "any"
            }
        else:
            query_info = self.llm_router.transform_query(raw_query)

        # Trọng số Fusion
        if use_dynamic_weights:
            w_vis = query_info["weights"].get("visual", 0.8)
            w_ocr = query_info["weights"].get("ocr", 0.1) if use_ocr else 0.0
            w_asr = query_info["weights"].get("asr", 0.1) if use_asr else 0.0
        else:
            w_vis = 1.0
            w_ocr = 0.3 if use_ocr else 0.0
            w_asr = 0.3 if use_asr else 0.0

        # Hồ chứa điểm RRF cho từng frame: key = (video_id, frame_idx) -> {score, matches}
        frame_scores = defaultdict(lambda: {"score": 0.0, "ranks": {}, "video_id": "", "frame_idx": 0, "global_id": -1})

        # ======================================================================
        # 2. DENSE RETRIEVAL (FAISS SigLIP 2 / CLIP)
        # ======================================================================
        vis_prompts = query_info["visual_prompts"] if use_multi_prompt else [query_info["visual_prompts"][0]]
        
        # Mã hóa và trung bình vector các visual prompts (Multi-Prompt Ensembling)
        if len(vis_prompts) == 1:
            q_vec = self.text_encoder.encode_text(vis_prompts[0])
        else:
            vecs = [self.text_encoder.encode_text(p) for p in vis_prompts]
            avg_vec = np.mean(vecs, axis=0)
            q_vec = avg_vec / np.linalg.norm(avg_vec, axis=-1, keepdims=True)

        dense_top_k = min(top_k * 3, self.faiss_index.ntotal)
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
        # 3. SPARSE RETRIEVAL: BM25 OCR (Nếu bật)
        # ======================================================================
        if use_ocr and query_info.get("ocr_keywords"):
            ocr_query = " ".join(query_info["ocr_keywords"])
            ocr_results = self.bm25_indexer.search_ocr(ocr_query, top_k=top_k * 2)
            for rank, doc in enumerate(ocr_results, 1):
                v_id = doc["video_id"]
                f_idx = doc["frame_idx"]
                if f_idx < 0: # Nếu frame_idx không hợp lệ thì bỏ qua
                    continue
                key = (v_id, f_idx)
                rrf_contrib = w_ocr / (self.k_rrf + rank)
                frame_scores[key]["score"] += rrf_contrib
                frame_scores[key]["ranks"]["ocr"] = rank
                frame_scores[key]["video_id"] = v_id
                frame_scores[key]["frame_idx"] = f_idx

        # ======================================================================
        # 4. SPARSE RETRIEVAL: BM25 ASR (Nếu bật)
        # ======================================================================
        if use_asr and query_info.get("asr_keywords"):
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
        # 5. SẮP XẾP VÀ XUẤT KẾT QUẢ TOP K
        # ======================================================================
        sorted_candidates = sorted(frame_scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        
        final_results = []
        for rank, cand in enumerate(sorted_candidates, 1):
            final_results.append({
                "rank": rank,
                "video_id": cand["video_id"],
                "frame_idx": cand["frame_idx"],
                "global_id": cand["global_id"],
                "score": float(cand["score"]),
                "matched_modalities": cand["ranks"]
            })

        latency = (time.time() - t0) * 1000
        return final_results, query_info, latency

if __name__ == "__main__":
    engine = HybridRetrievalEngine("siglip2")
    q = "Trong một căn nhà, người phụ nữ dùng hai tay quấn và chỉnh tấm xà rông màu vàng cam quanh eo người đàn ông mặc áo xanh."
    print(f"\n🔎 [TEST SEARCH]: {q}")
    results, q_info, latency = engine.search(q, top_k=5)
    print(f"⚡ Thời gian xử lý toàn bộ Pipeline: {latency:.2f} ms")
    print(f"🏆 Top 5 kết quả sau RRF Fusion:")
    for r in results:
        print(f"   + Rank #{r['rank']} | Video: {r['video_id']} | Frame: {r['frame_idx']} | Score: {r['score']:.5f} | Modalities: {r['matched_modalities']}")
