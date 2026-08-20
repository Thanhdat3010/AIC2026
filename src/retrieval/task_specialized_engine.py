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
      1. KIS Specialist: Single Best Gemini Translation + SigLIP 2 FAISS + Keyframe Neighbor Expansion + RRF.
      2. QA Specialist: SigLIP 2 + Multi-Crop Gemini 3.5 Flash Lite Vision soi ảnh sinh <Answer>.
      3. TRAKE Specialist: Bẻ nhỏ N sự kiện + Vectorized Continuous Cosine Sim + Monotonic DP.
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
        
        # Mapping video -> sorted keyframe indices for fast neighbor lookup
        self.video_to_keyframes = {}
        for v_id, grp in self.df_frames.groupby("video_id"):
            s_grp = grp.sort_values("pts_time")
            self.video_to_keyframes[v_id] = {
                "frame_indices": s_grp["frame_idx"].to_numpy(),
                "global_ids": s_grp["global_id"].to_numpy()
            }

        print(f"✅ TaskSpecializedEngine [{engine.upper()}] đã sẵn sàng!", flush=True)

    def get_blind_spot_gate(self):
        if not hasattr(self, "_blind_spot_gate") or self._blind_spot_gate is None:
            from src.reranking.blind_spot_gate import MultiSignalBlindSpotGate
            self._blind_spot_gate = MultiSignalBlindSpotGate(
                df_frames=self.df_frames,
                batch=self.batch,
                text_encoder=self.text_encoder,
                key_pool=self.key_pool
            )
        return self._blind_spot_gate

    def get_dense_refiner(self):
        if not hasattr(self, "_dense_refiner") or self._dense_refiner is None:
            from src.reranking.dense_video_refiner import DenseVideoRefiner
            self._dense_refiner = DenseVideoRefiner(
                model_name=self.text_encoder.model_name,
                device="cuda" if self.text_encoder.device == "cuda" else "cpu"
            )
        return self._dense_refiner

    def get_intra_reranker(self):
        if self._intra_reranker is None:
            from src.reranking.intra_video_reranker import IntraVideoTemporalReranker
            self._intra_reranker = IntraVideoTemporalReranker(batch=self.batch)
        return self._intra_reranker

    # =========================================================================
    # 1. SPECIALIST 1: TEXTUAL KIS PIPELINE (Truy tìm sự kiện KIS)
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
        use_dense_video_refiner: bool = False,
        use_rrf: bool = False,
        use_neighbor_expansion: bool = False
    ) -> tuple[list[dict], dict, float]:
        """
        Chiến thuật KIS SOTA:
        1. Phân tích ModalityGate (OCR / ASR).
        2. Sinh bản dịch tiếng Anh tối ưu từ Gemini Router.
        3. FAISS GPU Dense Search.
        4. Tùy chọn RRF kết hợp OCR/ASR BM25 nếu câu hỏi chứa thực thể/văn bản.
        5. Tùy chọn Keyframe Neighborhood Expansion (±2 keyframes).
        6. Tùy chọn Intra-Video Temporal Smoothing (Gaussian Kernel).
        """
        t0 = time.time()
        gate_info = self.modality_gate.analyze(query_text)

        # 1. Sinh bản dịch
        if custom_en_query:
            en_prompt = custom_en_query
            q_info = {"visual_prompts": [custom_en_query]}
        else:
            q_info = self.router.transform_query(query_text)
            en_prompt = q_info.get("visual_prompts", [query_text])[0]

        # 2. Vector hóa câu truy vấn
        q_vec = self.text_encoder.encode_text(en_prompt)

        # 3. Stage-1 Coarse Retrieval qua FAISS GPU
        scores, indices = self.faiss_index.search(q_vec, 300)

        # 3. Thu thập Top Candidate Videos từ Stage-1
        candidate_videos = []
        seen_v = set()

        # Luôn lấy Top các video thị giác tốt nhất từ SigLIP-2
        for idx in indices[0]:
            v = self.df_frames.iloc[idx]["video_id"]
            if v not in seen_v:
                candidate_videos.append(v)
                seen_v.add(v)
                if len(candidate_videos) >= 20:
                    break

        # Nếu LLM Router xác nhận có tín hiệu OCR/ASR rõ ràng, bổ sung video khớp từ khóa vào Top
        if use_rrf:
            has_ocr = q_info.get("has_ocr_signal", False)
            ocr_words = q_info.get("ocr_keywords", [])
            if has_ocr and ocr_words:
                ocr_query_str = " ".join(ocr_words)
                ocr_hits = self.bm25_indexer.search_ocr(ocr_query_str, top_k=20)
                for doc in ocr_hits[:3]:
                    v_ocr = doc["video_id"]
                    if v_ocr not in seen_v:
                        candidate_videos.insert(0, v_ocr)
                        seen_v.add(v_ocr)

            has_asr = q_info.get("has_asr_signal", False)
            asr_words = q_info.get("asr_keywords", [])
            if has_asr and asr_words:
                asr_query_str = " ".join(asr_words)
                asr_hits = self.bm25_indexer.search_asr(asr_query_str, top_k=20)
                for doc in asr_hits[:3]:
                    v_asr = doc["video_id"]
                    if v_asr not in seen_v:
                        candidate_videos.insert(0, v_asr)
                        seen_v.add(v_asr)

        # 5. Nếu không bật Intra-Reranker, trả về kết quả FAISS trực tiếp
        if not use_intra_reranker:
            results = []
            seen_k = set()
            for rank, (sim, idx) in enumerate(zip(scores[0], indices[0]), 1):
                row = self.df_frames.iloc[idx]
                k = (row["video_id"], int(row["frame_idx"]))
                if k not in seen_k:
                    seen_k.add(k)
                    results.append({
                        "rank": len(results) + 1,
                        "video_id": row["video_id"],
                        "frame_idx": int(row["frame_idx"]),
                        "global_id": int(row["global_id"]),
                        "score": float(sim)
                    })
                if len(results) >= top_k:
                    break
            latency = (time.time() - t0) * 1000
            return results, {"en_prompt": en_prompt, "gate_info": gate_info}, latency

        # 6. Chạy Intra-Video Temporal Reranker
        reranker = self.get_intra_reranker()
        cue_vecs = []
        if use_cue:
            cues = q_info.get("visual_prompts", [])[1:]
            if not cues:
                parts = [p.strip() for p in re.split(r"[,;.]", en_prompt) if len(p.strip()) > 3]
                cues = parts[:4]
            cue_vecs = [self.text_encoder.encode_text(c) for c in cues]

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
                rescored_frame_deltas[key] = {
                    "delta": item["final_score"] - item["raw_score"],
                    "final_score": item["final_score"]
                }

        # 7. Ghép điểm và áp dụng Keyframe Neighborhood Expansion
        final_scores = []
        seen_keys = set()

        for sim, idx in zip(scores[0], indices[0]):
            row = self.df_frames.iloc[idx]
            v_id = row["video_id"]
            f_idx = int(row["frame_idx"])
            key = (v_id, f_idx)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            base_sim = float(sim)
            if key in rescored_frame_deltas:
                mod_score = base_sim + 0.40 * rescored_frame_deltas[key]["delta"]
            else:
                mod_score = base_sim

            final_scores.append({
                "video_id": v_id,
                "frame_idx": f_idx,
                "global_id": int(row["global_id"]),
                "score": mod_score
            })

        final_scores.sort(key=lambda x: x["score"], reverse=True)

        results = []
        for rank, item in enumerate(final_scores[:top_k], 1):
            item["rank"] = rank
            results.append(item)

        # Layer 3: Multi-Signal Gated Dense Video Refinement
        if use_dense_video_refiner and results:
            gate = self.get_blind_spot_gate()
            refiner = self.get_dense_refiner()
            for cand in results[:1]:
                vid = cand["video_id"]
                f_idx = cand["frame_idx"]
                c_score = cand.get("score", 0.5)
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
        use_multimodal: bool = False,
        use_rrf: bool = False,
        use_multi_crop: bool = True
    ) -> tuple[list[dict], dict, float]:
        """
        Chiến thuật QA SOTA:
        1. Sinh tiếng Anh + Modality Gate (ASR / OCR / Visual).
        2. FAISS dense search + RRF OCR/ASR.
        3. Tái xếp hạng đa phương thức theo mốc thời gian (E3).
        4. Gemini 3.5 Flash Lite Vision với Dynamic Multi-Crop đọc ảnh trả lời câu hỏi và rerank lên Rank #1.
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

        # 3. Candidate frames & RRF OCR
        candidates = []
        seen_keys = set()

        if use_rrf:
            has_ocr = q_info.get("has_ocr_signal", False)
            ocr_words = q_info.get("ocr_keywords", [])
            if has_ocr and ocr_words:
                ocr_hits = self.bm25_indexer.search_ocr(" ".join(ocr_words), top_k=20)
                for r_ocr, doc in enumerate(ocr_hits, 1):
                    v = doc["video_id"]
                    if doc["frame_idx"] > 0:
                        key = (v, doc["frame_idx"])
                        if key not in seen_keys:
                            seen_keys.add(key)
                            candidates.append({
                                "video_id": v,
                                "frame_idx": doc["frame_idx"],
                                "global_id": -1,
                                "score": 0.35 + 1.0 / (self.k_rrf + r_ocr)
                            })

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

        candidates.sort(key=lambda x: x["score"], reverse=True)

        # 4. Intra-Video Reranking
        if use_intra_reranker:
            reranker = self.get_intra_reranker()
            top_v_list = []
            for c in candidates:
                if c["video_id"] not in top_v_list:
                    top_v_list.append(c["video_id"])
                if len(top_v_list) >= 50:
                    break

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

        qa_modality = self.router.get_qa_modality(query_text)
        # 5. Visual QA Agent (Gemini 3.5 Flash Lite Vision với Dynamic Multi-Crop)
        best_answer, reranked = self.qa_agent.answer_and_rerank(
            qa_question=query_text,
            candidates=candidates[:top_k],
            max_inspect_frames=15, # Top-15 video limit
            use_multi_crop=use_multi_crop,
            gate_info=gate_info,
            qa_modality=qa_modality
        )

        latency = (time.time() - t0) * 1000
        return reranked, {"en_prompt": en_prompt, "generated_qa_answer": best_answer, "gate_info": gate_info}, latency

    # =========================================================================
    # 3. SPECIALIST 3: TRAKE PIPELINE (Monotonic Sequence Dynamic Programming)
    # =========================================================================
    def search_trake(
        self,
        query_text: str,
        top_k: int = 100,
        custom_en_query: str = None,
        use_multi_query: bool = True,
        use_event_coverage: bool = True,
        use_row_norm_dp: bool = True,
        use_segmental_dp: bool = True
    ) -> tuple[list[dict], dict, float]:
        """
        Chiến thuật TRAKE SOTA:
        1. Phân rã câu hỏi thành N sự kiện con: E_1 -> E_2 -> ... -> E_N.
        2. Chạy TRAKEAlignmentAgent với Vectorized Cosine Similarity & Monotonic/Segmental DP.
        3. Xuất danh sách 100 dự đoán chuẩn format BTC: <video>, <f_1>, <f_2>, ..., <f_N>.
        """
        t0 = time.time()

        # 1. Phân rã các sự kiện
        q_info = self.router.transform_query(query_text)
        events = q_info.get("trake_events", [])
        if not events or len(events) < 2:
            parts = [p.strip() for p in re.split(r"[,;]|\bvà\b|\btiếp tục\b|\bcuối cùng\b", query_text) if p.strip()]
            events = parts if len(parts) >= 2 else [query_text, query_text]

        # 2. Chạy Monotonic/Segmental DP Alignment Agent
        from src.tasks.trake_agent import TRAKEAlignmentAgent
        if not hasattr(self, "_trake_agent") or self._trake_agent is None:
            self._trake_agent = TRAKEAlignmentAgent(engine="siglip2", batch=self.batch, text_encoder=self.text_encoder)

        results = self._trake_agent.align_events(
            query_text,
            events,
            top_k=top_k,
            use_multi_query=use_multi_query,
            use_event_coverage=use_event_coverage,
            use_row_norm_dp=use_row_norm_dp,
            use_segmental_dp=use_segmental_dp
        )

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
        use_vlm_verification: bool = False,
        use_rrf: bool = False,
        use_neighbor_expansion: bool = False,
        use_multi_crop: bool = True,
        **kwargs
    ) -> tuple[list[dict], dict, float]:
        ttype = task_type.lower()
        if ttype == "trake":
            return self.search_trake(
                query_text=query_text,
                top_k=top_k,
                custom_en_query=custom_en_query,
                use_multi_query=kwargs.get("use_multi_query", True),
                use_event_coverage=kwargs.get("use_event_coverage", True),
                use_row_norm_dp=kwargs.get("use_row_norm_dp", True)
            )
        elif "qa" in ttype or "q&a" in ttype:
            return self.search_qa(
                query_text,
                top_k=top_k,
                custom_en_query=custom_en_query,
                use_intra_reranker=use_intra_reranker,
                use_neighbor=use_neighbor,
                use_cue=use_cue,
                use_multimodal=use_multimodal,
                use_rrf=use_rrf,
                use_multi_crop=use_multi_crop
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
                use_vlm_verification=use_vlm_verification,
                use_rrf=use_rrf,
                use_neighbor_expansion=use_neighbor_expansion
            )
