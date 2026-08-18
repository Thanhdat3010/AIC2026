import os
import sys
import io
import re
import json
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
from src.query.gemini_router import GeminiQueryRouter, GeminiKeyPool
from src.query.modality_gate import ModalityGate
from src.retrieval.keyframe_loader import KeyframeZipLoader
from src.tasks.qa_agent import VisualQAAgent

class TaskSpecializedEngine:
    """
    Kiến trúc Đột phá: Phân hóa Chiến thuật theo từng Task (Task-Specific Specialist Engine)
    - Không gom chung 1 pipeline để tránh hiện tượng over-engineering & pha loãng vector.
    - Tích hợp ModalityGate để khóa/mở BM25 OCR & ASR thông minh:
      1. KIS Specialist: Tối đa hóa Recall & R@1 bằng Single Best Gemini Translation + SigLIP 2 FAISS.
      2. QA Specialist: Giữ vững Top 100 của SigLIP 2 + Gemini 3.5 Flash Lite Vision soi ảnh sinh <Answer>.
      3. TRAKE Specialist: Bẻ nhỏ N sự kiện + Thuật toán sắp xếp mốc thời gian tăng dần t(E1) <= t(E2) <= ... <= t(En).
      4. Intra-Video Temporal Reranking (E1-E3 optional): Định vị khung hình chính xác sau Stage-1.
    """
    def __init__(self, engine: str = "siglip2", batch: str = "batch_1", k_rrf: int = 60):
        self.engine = engine
        self.batch = batch
        self.k_rrf = k_rrf
        self._intra_reranker = None
        self._dense_refiner = None

        print(f"[*] Đang khởi tạo TaskSpecializedEngine [{engine.upper()}]...", flush=True)
        self.text_encoder = UnifiedTextEncoder(engine=engine)
        self.faiss_index, self.df_frames = load_faiss_index(engine=engine, batch=batch)
        self.bm25_indexer = BM25MultiIndexer(batch=batch)
        self.key_pool = GeminiKeyPool()
        self.router = GeminiQueryRouter()
        self.modality_gate = ModalityGate()
        self.qa_agent = VisualQAAgent(key_pool=self.key_pool)
        print(f"✅ TaskSpecializedEngine [{engine.upper()}] đã sẵn sàng!", flush=True)

    def get_blind_spot_gate(self):
        if not hasattr(self, "_blind_spot_gate") or self._blind_spot_gate is None:
            from src.reranking.blind_spot_gate import MultiSignalBlindSpotGate
            self._blind_spot_gate = MultiSignalBlindSpotGate(
                df_frames=self.df_frames,
                batch=self.batch,
                gemini_api_keys=self.key_pool.api_keys if hasattr(self.key_pool, "api_keys") else []
            )
        return self._blind_spot_gate

    def get_intra_reranker(self):
        if self._intra_reranker is None:
            from src.reranking.intra_video_reranker import IntraVideoTemporalReranker
            self._intra_reranker = IntraVideoTemporalReranker(batch=self.batch, base_dir=BASE_DIR)
        return self._intra_reranker

    def get_dense_refiner(self):
        if self._dense_refiner is None:
            from src.reranking.dense_video_refiner import DenseVideoRefiner
            self._dense_refiner = DenseVideoRefiner(engine=self.engine)
        return self._dense_refiner

    # =========================================================================
    # 1. SPECIALIST 1: TEXTUAL KIS PIPELINE (Tìm kiếm chính xác khung hình)
    # =========================================================================
    def search_kis(
        self,
        query_text: str,
        top_k: int = 100,
        custom_en_query: str = None,
        use_intra_reranker: bool = False,
        use_neighbor: bool = True,
        use_cue: bool = False,
        use_multimodal: bool = False,
        use_vlm_verification: bool = False,
        use_dense_video_refiner: bool = False
    ) -> tuple[list[dict], dict, float]:
        """
        Chiến thuật KIS: 
        1. Dịch 1 câu tiếng Anh chuẩn nhất từ Gemini.
        2. Kiểm tra tín hiệu chữ viết (OCR). Nếu có chữ trong ngoặc kép -> Kết hợp BM25 OCR.
        3. Nếu thuần thị giác -> Chạy 100% SigLIP 2 FAISS (1152d) để đạt R@1 cao nhất.
        4. Tùy chọn: Chạy Intra-Video Temporal Reranker trên Top Candidate Videos.
        """
        t0 = time.time()
        gate_info = self.modality_gate.analyze(query_text)
        
        if custom_en_query:
            en_prompt = custom_en_query
            q_info = {"visual_prompts": [en_prompt]}
        else:
            q_info = self.router.transform_query(query_text)
            en_prompt = q_info["visual_prompts"][0]

        q_vec = self.text_encoder.encode_text(en_prompt)
        scores, indices = self.faiss_index.search(q_vec, top_k * 2)

        # ---------------------------------------------------------------------
        # NẾU KHÔNG DÙNG INTRA-RERANKER (STAGE-1 PURE BASELINE 100% NHƯ CONFIG 11)
        # ---------------------------------------------------------------------
        if not use_intra_reranker:
            if not gate_info["has_ocr"] or not gate_info["ocr_keywords"]:
                results = []
                for rank, (sim, idx) in enumerate(zip(scores[0][:top_k], indices[0][:top_k]), 1):
                    row = self.df_frames.iloc[idx]
                    results.append({
                        "rank": rank,
                        "video_id": row["video_id"],
                        "frame_idx": int(row["frame_idx"]),
                        "global_id": int(row["global_id"]),
                        "score": float(sim)
                    })
            else:
                frame_scores = defaultdict(lambda: {"score": 0.0, "video_id": "", "frame_idx": 0, "global_id": -1})
                w_vis, w_ocr = 1.0, 0.4

                for rank, (sim, idx) in enumerate(zip(scores[0], indices[0]), 1):
                    row = self.df_frames.iloc[idx]
                    key = (row["video_id"], int(row["frame_idx"]))
                    frame_scores[key]["score"] += w_vis / (self.k_rrf + rank)
                    frame_scores[key]["video_id"] = row["video_id"]
                    frame_scores[key]["frame_idx"] = int(row["frame_idx"])
                    frame_scores[key]["global_id"] = int(row["global_id"])

                ocr_query = " ".join(gate_info["ocr_keywords"])
                ocr_results = self.bm25_indexer.search_ocr(ocr_query, top_k=top_k * 2)
                for rank, doc in enumerate(ocr_results, 1):
                    if doc["frame_idx"] < 0:
                        continue
                    key = (doc["video_id"], doc["frame_idx"])
                    frame_scores[key]["score"] += w_ocr / (self.k_rrf + rank)
                    frame_scores[key]["video_id"] = doc["video_id"]
                    frame_scores[key]["frame_idx"] = doc["frame_idx"]

                cand_list = list(frame_scores.values())
                cand_list.sort(key=lambda x: x["score"], reverse=True)
                results = []
                for rank, cand in enumerate(cand_list[:top_k], 1):
                    cand["rank"] = rank
                    results.append(cand)

            latency = (time.time() - t0) * 1000
            return results, {"en_prompt": en_prompt, "gate_info": gate_info}, latency

        # ---------------------------------------------------------------------
        # NẾU BẬT INTRA-VIDEO TEMPORAL RERANKER (STAGE-2 SOTA MOMENT LOCALIZATION)
        # ---------------------------------------------------------------------
        reranker = self.get_intra_reranker()

        # Trích xuất cues nếu E2 được kích hoạt
        cue_vecs = []
        if use_cue:
            cues = q_info.get("visual_prompts", [])[1:]
            if not cues:
                parts = [p.strip() for p in re.split(r"[,;.]", en_prompt) if len(p.strip()) > 3]
                cues = parts[:4]
            cue_vecs = [self.text_encoder.encode_text(c) for c in cues]

        # 1. Thu thập Top 10 Candidate Videos từ Stage-1
        candidate_videos = []
        seen_v = set()
        for idx in indices[0]:
            v = self.df_frames.iloc[idx]["video_id"]
            if v not in seen_v:
                candidate_videos.append(v)
                seen_v.add(v)
                if len(candidate_videos) >= 10:
                    break

        # 2. Rescore từng video và thu thập frame score map
        rescored_frame_deltas = {}
        for v_id in candidate_videos:
            rescored = reranker.rescore_candidate_video(
                video_id=v_id,
                main_query_vec=q_vec,
                cue_vecs=cue_vecs,
                query_text=query_text,
                gate_info=gate_info,
                use_neighbor=use_neighbor,
                use_cue=use_cue,
                use_multimodal=use_multimodal
            )
            for item in rescored:
                key = (item["video_id"], item["frame_idx"])
                # Delta = final_score - raw_score (tác động của neighbor + cue + timeline)
                delta = item["final_score"] - item["raw_score"]
                rescored_frame_deltas[key] = {
                    "delta": delta,
                    "final_score": item["final_score"]
                }

        # 3. Chấm điểm lại toàn bộ candidate pool từ Stage-1
        final_scores = []
        seen_keys = set()

        for sim, idx in zip(scores[0], indices[0]):
            row = self.df_frames.iloc[idx]
            key = (row["video_id"], int(row["frame_idx"]))
            if key in seen_keys:
                continue
            seen_keys.add(key)

            base_sim = float(sim)
            if key in rescored_frame_deltas:
                # Cộng hưởng delta từ Intra-Video Reranker
                delta_info = rescored_frame_deltas[key]
                mod_score = base_sim + 0.40 * delta_info["delta"]
            else:
                mod_score = base_sim

            final_scores.append({
                "video_id": row["video_id"],
                "frame_idx": int(row["frame_idx"]),
                "global_id": int(row["global_id"]),
                "score": mod_score
            })

        # Sắp xếp lại theo điểm sau khi cộng hưởng
        final_scores.sort(key=lambda x: x["score"], reverse=True)

        results = []
        for rank, item in enumerate(final_scores[:top_k], 1):
            item["rank"] = rank
            results.append(item)

        # Layer 3: Multi-Signal Gated Dense Video Refinement (Kính lúp vi sai bằng OpenCV trên GPU)
        if use_dense_video_refiner and results:
            gate = self.get_blind_spot_gate()
            refiner = self.get_dense_refiner()
            for cand in results[:1]:
                vid = cand["video_id"]
                f_idx = cand["frame_idx"]
                c_score = cand.get("score", 0.5)
                
                # Đánh giá đa tín hiệu xem có bị rơi vào Vùng Mù không
                decision = gate.evaluate_blind_spot(
                    video_id=vid,
                    frame_idx=f_idx,
                    score=c_score,
                    query_text=query_text,
                    task_type="kis"
                )
                
                if decision.get("trigger_layer3", False):
                    target_frame = decision.get("target_frame_idx", f_idx)
                    win_sec = decision.get("window_seconds", 2.5)
                    
                    ref_res = refiner.refine_candidate(
                        video_id=vid,
                        approx_frame_idx=target_frame,
                        query_vec=q_vec,
                        window_seconds=win_sec,
                        step=3
                    )
                    if ref_res.get("refined", False) and ref_res["score"] > cand["score"]:
                        cand["frame_idx"] = ref_res["frame_idx"]
                        cand["score"] = ref_res["score"]
                        cand["dense_refined"] = True
                        cand["blind_spot_reason"] = decision.get("reason", "")

        latency = (time.time() - t0) * 1000
        return results, {"en_prompt": en_prompt, "gate_info": gate_info, "top_videos": candidate_videos}, latency

    # =========================================================================
    # 2. SPECIALIST 2: VISUAL Q&A PIPELINE (Hỏi - Đáp Trực quan)
    # =========================================================================
    def search_qa(
        self,
        query_text: str,
        top_k: int = 100,
        custom_en_query: str = None,
        use_intra_reranker: bool = False,
        use_neighbor: bool = True,
        use_cue: bool = False,
        use_multimodal: bool = False
    ) -> tuple[list[dict], dict, float]:
        """
        Chiến thuật QA SOTA:
        1. Sinh tiếng Anh + Modality Gate (ASR / OCR / Visual).
        2. FAISS dense search.
        3. Tái xếp hạng đa phương thức theo mốc thời gian (E3).
        4. Gemini Flash Lite Vision đọc ảnh trả lời câu hỏi và rerank đưa frame chứa đáp án lên Rank #1.
        """
        t0 = time.time()
        gate_info = self.modality_gate.analyze(query_text)

        # 1. Định tuyến
        if custom_en_query:
            en_prompt = custom_en_query
        else:
            q_info = self.router.transform_query(query_text)
            en_prompt = q_info.get("visual_prompts", [query_text])[0]

        q_vec = self.text_encoder.encode_text(en_prompt)

        # 2. Stage-1 FAISS Dense Search
        scores, indices = self.faiss_index.search(q_vec, 300)

        # 3. Candidate frames
        candidates = []
        seen_keys = set()
        for sc, idx in zip(scores[0], indices[0]):
            row = self.df_frames.iloc[idx]
            key = (row["video_id"], int(row["frame_idx"]))
            if key not in seen_keys:
                seen_keys.add(key)
                candidates.append({
                    "video_id": row["video_id"],
                    "frame_idx": int(row["frame_idx"]),
                    "global_id": int(row["global_id"]),
                    "score": float(sc)
                })

        # 4. Intra-Video Reranking (E3 Multi-modal Timeline Sync nếu bật)
        if use_intra_reranker:
            reranker = self.get_intra_reranker()
            top_v_list = list(dict.fromkeys([c["video_id"] for c in candidates[:10]]))
            rescored_deltas = {}
            for v_id in top_v_list:
                rescored = reranker.rescore_candidate_video(
                    video_id=v_id,
                    main_query_vec=q_vec,
                    cue_vecs=[],
                    query_text=query_text,
                    gate_info=gate_info,
                    use_neighbor=use_neighbor,
                    use_cue=use_cue,
                    use_multimodal=use_multimodal
                )
                for item in rescored:
                    key = (item["video_id"], item["frame_idx"])
                    rescored_deltas[key] = item["final_score"] - item["raw_score"]

            for c in candidates:
                key = (c["video_id"], c["frame_idx"])
                if key in rescored_deltas:
                    c["score"] = c["score"] + 0.40 * rescored_deltas[key]

            candidates.sort(key=lambda x: x["score"], reverse=True)

        # 5. Visual QA Agent (Gemini Vision)
        best_answer, reranked = self.qa_agent.answer_and_rerank(
            qa_question=query_text,
            candidates=candidates[:top_k],
            max_inspect_frames=4
        )

        latency = (time.time() - t0) * 1000
        return reranked, {"en_prompt": en_prompt, "generated_qa_answer": best_answer, "gate_info": gate_info}, latency

    # =========================================================================
    # 3. SPECIALIST 3: TRAKE PIPELINE (Monotonic Sequence Dynamic Programming)
    # =========================================================================
    def search_trake(self, query_text: str, top_k: int = 100, custom_en_query: str = None) -> tuple[list[dict], dict, float]:
        """
        Chiến thuật TRAKE SOTA:
        1. Phân rã câu hỏi thành N sự kiện con: E_1 -> E_2 -> ... -> E_N.
        2. Sử dụng TRAKEAlignmentAgent với thuật toán Monotonic Sequence Dynamic Programming.
        3. Xuất danh sách 100 dự đoán chuẩn format BTC: <video>, <f_1>, <f_2>, ..., <f_N>.
        """
        t0 = time.time()

        # 1. Phân rã các sự kiện
        q_info = self.router.transform_query(query_text)
        events = q_info.get("trake_events", [])
        if not events or len(events) < 2:
            parts = [p.strip() for p in re.split(r"[,;]|\bvà\b|\btiếp tục\b|\bcuối cùng\b", query_text) if p.strip()]
            events = parts if len(parts) >= 2 else [query_text, query_text]

        # 2. Chạy Monotonic DP Alignment Agent
        from src.tasks.trake_agent import TRAKEAlignmentAgent
        if not hasattr(self, "_trake_agent") or self._trake_agent is None:
            self._trake_agent = TRAKEAlignmentAgent(engine="siglip2", batch=self.batch, text_encoder=self.text_encoder)

        dp_preds = self._trake_agent.align_events(query_text, events, top_videos=top_k)

        results = []
        for rank, p in enumerate(dp_preds[:top_k], 1):
            f_list = p.get("event_frames", [])
            results.append({
                "rank": rank,
                "video_id": p.get("video_id", ""),
                "frame_idx": f_list[0] if f_list else 0,
                "event_frames": f_list,
                "score": p.get("score", 0.5)
            })

        latency = (time.time() - t0) * 1000
        return results, {"events": events}, latency

    # =========================================================================
    # 4. TỰ ĐỘNG ĐỊNH TUYẾN CHUYÊN BIỆT (AUTOROUTE SEARCH)
    # =========================================================================
    def search(
        self,
        query_text: str,
        task_type: str = "kis",
        top_k: int = 100,
        custom_en_query: str = None,
        use_intra_reranker: bool = False,
        use_neighbor: bool = True,
        use_cue: bool = False,
        use_multimodal: bool = False,
        use_vlm_verification: bool = False
    ) -> tuple[list[dict], dict, float]:
        ttype = task_type.lower()
        if "trake" in ttype:
            return self.search_trake(query_text, top_k=top_k, custom_en_query=custom_en_query)
        elif "qa" in ttype or "q&a" in ttype:
            return self.search_qa(
                query_text,
                top_k=top_k,
                custom_en_query=custom_en_query,
                use_intra_reranker=use_intra_reranker,
                use_neighbor=use_neighbor,
                use_cue=use_cue,
                use_multimodal=use_multimodal
            )
        else:
            return self.search_kis(
                query_text,
                top_k=top_k,
                custom_en_query=custom_en_query,
                use_intra_reranker=use_intra_reranker,
                use_neighbor=use_neighbor,
                use_cue=use_cue,
                use_multimodal=use_multimodal,
                use_vlm_verification=use_vlm_verification
            )
