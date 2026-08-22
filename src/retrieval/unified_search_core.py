import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

from src.indexing.faiss_indexer import load_faiss_index
from src.query.text_encoder import UnifiedTextEncoder
from src.indexing.bm25_indexer import BM25MultiIndexer
from src.retrieval.keyframe_loader import KeyframeZipLoader
from src.query.gemini_router import GeminiKeyPool

class UnifiedSearchCore:
    """
    Lõi tìm kiếm đa phương thức tinh gọn, hiệu năng cao cho AIC 2026.
    Hợp nhất trực tiếp:
    1. SigLIP 2 Dense Vector Index (FAISS FlatIP trên CUDA/CPU)
    2. BM25 OCR (Văn bản trên khung hình)
    3. BM25 ASR (Lời thoại thuyết minh video)
    4. Fast Weighted Reciprocal Rank Fusion (WRRF)
    """
    def __init__(self, engine: str = "siglip2", batch: str = "batch_1"):
        self.engine = engine
        self.batch = batch
        
        print(f"[*] Khởi tạo UnifiedSearchCore [{engine.upper()}] - {batch}...", flush=True)
        self.text_encoder = UnifiedTextEncoder(engine=engine)
        self.faiss_index, self.df_frames = load_faiss_index(engine=engine, batch=batch)
        self.bm25 = BM25MultiIndexer(batch=batch)
        self.loader = KeyframeZipLoader()
        self.key_pool = GeminiKeyPool()
        
        # Fast lookup mapping: video_id -> sorted array of frame_indices
        self.video_to_frames = {}
        for vid, grp in self.df_frames.groupby("video_id"):
            self.video_to_frames[vid] = grp.sort_values("frame_idx")["frame_idx"].to_numpy()
            
        print(f"✅ UnifiedSearchCore [{engine.upper()}] đã sẵn sàng ({len(self.df_frames):,} keyframes)!", flush=True)

    def encode_text(self, text: str) -> np.ndarray:
        """Mã hóa văn bản thành vector chuẩn hóa L2."""
        vec = self.text_encoder.encode_text(text)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        return vec.astype(np.float32)

    def search_visual(self, query_vec: np.ndarray, top_k: int = 150) -> List[Dict[str, Any]]:
        """Tìm kiếm thuần hình ảnh qua FAISS."""
        scores, indices = self.faiss_index.search(query_vec, top_k)
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            if idx < 0 or idx >= len(self.df_frames):
                continue
            row = self.df_frames.iloc[idx]
            results.append({
                "rank": rank,
                "video_id": row["video_id"],
                "frame_idx": int(row["frame_idx"]),
                "pts_time": float(row.get("pts_time", 0.0)),
                "score": float(score),
                "source": "visual"
            })
        return results

    def search_ocr(self, query_text: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Tìm kiếm qua văn bản OCR."""
        raw_hits = self.bm25.search_ocr(query_text, top_k=top_k)
        results = []
        for rank, hit in enumerate(raw_hits, 1):
            results.append({
                "rank": rank,
                "video_id": hit["video_id"],
                "frame_idx": int(hit.get("frame_idx", 0)),
                "score": float(hit.get("score", 0.0)),
                "text": hit.get("text", ""),
                "source": "ocr"
            })
        return results

    def search_asr(self, query_text: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Tìm kiếm qua lời thoại thuyết minh ASR."""
        raw_hits = self.bm25.search_asr(query_text, top_k=top_k)
        results = []
        for rank, hit in enumerate(raw_hits, 1):
            results.append({
                "rank": rank,
                "video_id": hit["video_id"],
                "frame_idx": int(hit.get("frame_idx", 0)),
                "score": float(hit.get("score", 0.0)),
                "text": hit.get("text", ""),
                "source": "asr"
            })
        return results

    def search_multimodal(
        self,
        query_en: str,
        query_vi: str = "",
        weights: Dict[str, float] = None,
        top_k: int = 100,
        k_rrf: int = 60
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
        """
        Tìm kiếm hợp nhất đa phương thức siêu tốc bằng Weighted Reciprocal Rank Fusion (WRRF).
        weights mặc định: {"visual": 0.70, "ocr": 0.18, "asr": 0.12}
        """
        t0 = time.time()
        if weights is None:
            weights = {"visual": 0.70, "ocr": 0.18, "asr": 0.12}
            
        # 1. Visual Search (Dual embedding: Tiếng Việt bản ngữ + Tiếng Anh chi tiết)
        vec_en = self.encode_text(query_en)
        if query_vi and query_vi != query_en:
            vec_vi = self.encode_text(query_vi)
            query_vec = 0.60 * vec_vi + 0.40 * vec_en
            q_norm = np.linalg.norm(query_vec)
            if q_norm > 1e-6:
                query_vec = query_vec / q_norm
        else:
            query_vec = vec_en

        vis_hits = self.search_visual(query_vec, top_k=top_k * 2)
        
        # 2. Text Search (nếu có query_vi hoặc từ khóa)
        search_text_vi = query_vi if query_vi else query_en
        ocr_hits = self.search_ocr(search_text_vi, top_k=top_k) if weights.get("ocr", 0) > 0 else []
        asr_hits = self.search_asr(search_text_vi, top_k=top_k) if weights.get("asr", 0) > 0 else []
        
        # 3. Fast WRRF Fusion at Video & Frame Level
        fused_scores = {}
        meta_lookup = {}
        
        # Nạp Visual hits
        w_vis = weights.get("visual", 0.70)
        for h in vis_hits:
            key = (h["video_id"], h["frame_idx"])
            fused_scores[key] = fused_scores.get(key, 0.0) + w_vis / (k_rrf + h["rank"])
            meta_lookup[key] = h
            
        # Nạp OCR hits (cộng dồn cho frame hoặc frame gần nhất)
        w_ocr = weights.get("ocr", 0.18)
        for h in ocr_hits:
            vid = h["video_id"]
            f = h["frame_idx"]
            if f < 0 and vid in self.video_to_frames:
                f = int(self.video_to_frames[vid][0]) if len(self.video_to_frames[vid]) > 0 else 0
            key = (vid, f)
            fused_scores[key] = fused_scores.get(key, 0.0) + w_ocr / (k_rrf + h["rank"])
            if key not in meta_lookup:
                meta_lookup[key] = h
                
        # Nạp ASR hits
        w_asr = weights.get("asr", 0.12)
        for h in asr_hits:
            vid = h["video_id"]
            f = h["frame_idx"]
            if f < 0 and vid in self.video_to_frames:
                f = int(self.video_to_frames[vid][0]) if len(self.video_to_frames[vid]) > 0 else 0
            key = (vid, f)
            fused_scores[key] = fused_scores.get(key, 0.0) + w_asr / (k_rrf + h["rank"])
            if key not in meta_lookup:
                meta_lookup[key] = h

        # 4. Sắp xếp thứ hạng kết quả dung hợp
        sorted_keys = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)[:top_k]
        
        fused_results = []
        for rank, k in enumerate(sorted_keys, 1):
            base_meta = meta_lookup.get(k, {})
            fused_results.append({
                "rank": rank,
                "video_id": k[0],
                "frame_idx": k[1],
                "score": float(fused_scores[k]),
                "source": base_meta.get("source", "multimodal"),
                "text": base_meta.get("text", "")
            })
            
        latency_ms = (time.time() - t0) * 1000
        info = {
            "num_vis_hits": len(vis_hits),
            "num_ocr_hits": len(ocr_hits),
            "num_asr_hits": len(asr_hits),
            "weights": weights,
            "latency_ms": latency_ms
        }
        return fused_results, info, latency_ms

    def search_tnca(
        self,
        query_vi: str,
        query_en: str = "",
        ocr_keywords: Optional[List[str]] = None,
        asr_keywords: Optional[List[str]] = None,
        config_name: str = "A6",
        top_k: int = 100,
        candidate_k: int = 200,
        temporal_window_sec: float = 30.0,
        lambda_neighbor: float = 0.15,
        alpha_ocr: float = 0.15,
        alpha_asr: float = 0.15
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
        """
        Thuật toán Tìm Kiếm Đa Phương Thức & Tích Hợp Ngữ Cảnh Thời Gian (TNCA).
        Hỗ trợ các cấu hình thực nghiệm bóc tách (Ablation Study):
        - A0: Baseline SigLIP-2 (query_vi nguyên bản, không dual, không neighbor, không multimodal boost).
        - A1: + Dual Text Embedding (0.70 vi + 0.30 en).
        - A2: + Temporal Neighbor Context Aggregation (TNCA [t-30s, t+30s]).
        - A3..A6: + Bounded Multimodal Boost (ASR [t-30s, t+30s] & OCR trên candidate pool).
        """
        t0 = time.time()
        
        # 1. Cấu hình cờ kỹ thuật theo config_name
        use_dual_embedding = config_name in ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A6_1", "A6_2", "A6_3", "A6_4", "M5"]
        use_tnca = config_name in ["A2", "A3", "A4", "A5", "A6", "A7", "A6_1", "A6_2", "A6_3", "A6_4", "M5"]
        use_multimodal_boost = config_name in ["A3", "A4", "A5", "A6", "A7", "A6_1", "A6_2", "A6_3", "A6_4", "M5"]
        actual_cand_k = candidate_k if use_tnca or use_multimodal_boost else top_k

        # 2. Tầng 1: Dense Visual Search
        vec_vi = self.encode_text(query_vi)
        if use_dual_embedding and query_en and query_en != query_vi:
            vec_en = self.encode_text(query_en)
            query_vec = 0.70 * vec_vi + 0.30 * vec_en
            q_norm = np.linalg.norm(query_vec)
            if q_norm > 1e-6:
                query_vec = query_vec / q_norm
        else:
            query_vec = vec_vi

        vis_candidates = self.search_visual(query_vec, top_k=actual_cand_k)
        
        # Nếu cấu hình là A0 hoặc A1 thuần túy -> Trả về trực tiếp
        if not use_tnca and not use_multimodal_boost:
            final_res = vis_candidates[:top_k]
            for rank, r in enumerate(final_res, 1):
                r["rank"] = rank
            latency_ms = (time.time() - t0) * 1000
            info = {
                "config": config_name,
                "mode": "dense_visual_direct",
                "latency_ms": latency_ms
            }
            return final_res, info, latency_ms

        # 3. Bản đồ thời gian và điểm số theo Video cho TNCA
        video_cand_map = {}
        for c in vis_candidates:
            v = c["video_id"]
            if v not in video_cand_map:
                video_cand_map[v] = []
            video_cand_map[v].append(c)

        # 4. Tra cứu Bounded OCR & ASR nếu được kích hoạt
        ocr_map = {}
        asr_map = {}
        has_ocr = False
        has_asr = False
        
        if use_multimodal_boost:
            has_ocr = bool(ocr_keywords and len(ocr_keywords) > 0 and "".join(ocr_keywords).strip())
            has_asr = bool(asr_keywords and len(asr_keywords) > 0 and "".join(asr_keywords).strip())
            
            if has_ocr:
                ocr_hits = self.search_ocr(" ".join(ocr_keywords), top_k=actual_cand_k)
                for h in ocr_hits:
                    ocr_map[(h["video_id"], h["frame_idx"])] = max(ocr_map.get((h["video_id"], h["frame_idx"]), 0.0), h["score"])
                    ocr_map[(h["video_id"], -1)] = max(ocr_map.get((h["video_id"], -1), 0.0), h["score"])
                    
            if has_asr:
                asr_hits = self.search_asr(" ".join(asr_keywords), top_k=actual_cand_k)
                for h in asr_hits:
                    asr_map[(h["video_id"], h["frame_idx"])] = max(asr_map.get((h["video_id"], h["frame_idx"]), 0.0), h["score"])
                    asr_map[(h["video_id"], -1)] = max(asr_map.get((h["video_id"], -1), 0.0), h["score"])

        max_ocr = max(ocr_map.values()) if ocr_map else 1.0
        max_asr = max(asr_map.values()) if asr_map else 1.0
        if max_ocr < 1e-6: max_ocr = 1.0
        if max_asr < 1e-6: max_asr = 1.0

        # Kiểm tra liên từ chuyển cảnh để tự thích ứng trọng số
        has_transition = any(w in query_vi.lower() for w in [
            "tiếp theo", "ngay sau đó", "sau đó", "đầu tiên", "bắt đầu với", "kết thúc bằng", "rồi", "kế đến", "cuối cùng"
        ])
        cur_lambda_neighbor = lambda_neighbor if has_transition else (lambda_neighbor * 0.5)

        # 5. Tầng 2: Tính điểm TNCA và Bounded Multimodal Boost
        scored_list = []
        for c in vis_candidates:
            v = c["video_id"]
            f = c["frame_idx"]
            s_vis = c["score"]
            t_curr = c.get("pts_time", 0.0)
            
            # Tính Temporal Neighbor Support
            s_neighbor = s_vis
            if use_tnca:
                v_cands = video_cand_map.get(v, [])
                neighbor_scores = []
                for other_c in v_cands:
                    f_other = other_c["frame_idx"]
                    if f_other != f:
                        t_other = other_c.get("pts_time", 0.0)
                        dt = abs(t_other - t_curr) if (t_curr > 0 and t_other > 0) else abs(f_other - f) / 25.0
                        if dt <= temporal_window_sec:
                            neighbor_scores.append(other_c["score"])
                if neighbor_scores:
                    s_neighbor = max(neighbor_scores)

            # Tính Bounded Multimodal Support
            lex_bonus = 0.0
            if use_multimodal_boost:
                if has_ocr:
                    raw_ocr = max(ocr_map.get((v, f), 0.0), ocr_map.get((v, -1), 0.0) * 0.8)
                    norm_ocr = raw_ocr / max_ocr
                    if norm_ocr > 0.1:
                        lex_bonus += alpha_ocr * norm_ocr
                if has_asr:
                    raw_asr = max(asr_map.get((v, f), 0.0), asr_map.get((v, -1), 0.0) * 0.8)
                    norm_asr = raw_asr / max_asr
                    if norm_asr > 0.1:
                        lex_bonus += alpha_asr * norm_asr

            # Tổng hợp điểm số thặng dư (Residual SOTA Score)
            if use_tnca:
                s_final = s_vis + (cur_lambda_neighbor * s_neighbor) + (s_vis * lex_bonus)
            else:
                s_final = s_vis + (s_vis * lex_bonus)

            c_new = dict(c)
            c_new["score"] = float(s_final)
            c_new["vis_score"] = float(s_vis)
            c_new["neighbor_score"] = float(s_neighbor)
            c_new["lex_bonus"] = float(lex_bonus)
            scored_list.append(c_new)

        scored_list.sort(key=lambda x: x["score"], reverse=True)
        final_results = scored_list[:top_k]
        for rank, r in enumerate(final_results, 1):
            r["rank"] = rank

        latency_ms = (time.time() - t0) * 1000
        info = {
            "config": config_name,
            "mode": "tnca_multimodal_cascade",
            "has_transition": has_transition,
            "has_ocr": has_ocr,
            "has_asr": has_asr,
            "candidates_fetched": len(vis_candidates),
            "latency_ms": latency_ms
        }
        return final_results, info, latency_ms
