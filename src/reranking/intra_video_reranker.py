import os
import sys
import re
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

# Force UTF-8 on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent.parent.parent

def tokenize_clean(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
    return [w for w in cleaned.split() if len(w) > 0]

class IntraVideoTemporalReranker:
    """
    Intra-Video Temporal Reranker & Multi-Cue Moment Localizer (AIC 2026 SOTA)
    - Tối ưu hóa định vị khung hình và khoảnh khắc thời gian (Moment Localization) bên trong Video ứng viên.
    - Gồm 4 thành phần:
      1. E1: Gaussian Neighbor Temporal Score Aggregation (Lọc nhiễu Spike, tăng cường độ nhất quán thời gian).
      2. E2: Query Cue Decomposition & Sliding Window Coverage (Q2E - Độ bao phủ đa sự kiện con).
      3. E3: Multi-modal Time-Aligned Timeline Fusion (ASR, OCR, Objects gắn đúng mốc thời gian).
      4. Temporal Candidate Expansion & Diverse Hypotheses (Hedging Rank 1, 2, 3...).
    """
    def __init__(self, batch: str = "batch_1", base_dir: Path = None):
        if base_dir is None:
            base_dir = BASE_DIR
        self.batch = batch
        self.processed_dir = base_dir / "data" / batch / "processed"
        
        print(f"[*] Khởi tạo IntraVideoTemporalReranker [{batch}]...", flush=True)
        
        # 1. Tải Frames Mapping và tạo Index theo Video
        frames_path = self.processed_dir / "frames.parquet"
        self.df_frames = pd.read_parquet(frames_path)
        self.df_frames["row_idx"] = np.arange(len(self.df_frames))
        
        # Tạo mapping video_id -> DataFrame slice
        self.video_to_indices = {}
        for v_id, group in self.df_frames.groupby("video_id"):
            sorted_grp = group.sort_values("pts_time")
            self.video_to_indices[v_id] = {
                "row_indices": sorted_grp["row_idx"].to_numpy(),
                "frame_indices": sorted_grp["frame_idx"].to_numpy(),
                "pts_times": sorted_grp["pts_time"].to_numpy(),
                "global_ids": sorted_grp["global_id"].to_numpy()
            }

        # 2. Tải SigLIP Features (Memory-mapped float16 đọc siêu tốc <0.1ms)
        siglip_path = self.processed_dir / "siglip_features.npy"
        if siglip_path.exists():
            n_total = len(self.df_frames)
            self.siglip_features = np.memmap(siglip_path, dtype=np.float16, mode="r", shape=(n_total, 1152))
            print(f"✅ Đã nạp memory-mapped SigLIP features: {self.siglip_features.shape}", flush=True)
        else:
            self.siglip_features = None
            print(f"⚠️ Không tìm thấy {siglip_path.name}", flush=True)

        # 3. Tải ASR Transcripts (Vectorized indexing)
        asr_path = self.processed_dir / "transcripts.parquet"
        self.video_to_asr = defaultdict(list)
        if asr_path.exists():
            df_asr = pd.read_parquet(asr_path)
            for v_id, s_t, e_t, txt in zip(df_asr["video_id"], df_asr["start_time"], df_asr["end_time"], df_asr["transcript"]):
                tokens = tokenize_clean(txt)
                if tokens:
                    self.video_to_asr[v_id].append({
                        "start": float(s_t),
                        "end": float(e_t),
                        "text": str(txt),
                        "tokens": tokens
                    })

        # 4. Tải OCR Results (Vectorized indexing siêu tốc <0.05s)
        ocr_path = self.processed_dir / "ocr_results.parquet"
        self.video_frame_to_ocr = {}
        if ocr_path.exists():
            df_ocr = pd.read_parquet(ocr_path)
            for v_id, f_idx, txt in zip(df_ocr["video_id"], df_ocr["frame_idx"], df_ocr["ocr_text"]):
                if isinstance(txt, str) and txt.strip():
                    self.video_frame_to_ocr[(v_id, int(f_idx))] = {
                        "text": txt,
                        "tokens": tokenize_clean(txt)
                    }

        # 5. Tải Object Summaries (Vectorized indexing)
        obj_path = self.processed_dir / "object_summary.parquet"
        self.gid_to_obj = {}
        if obj_path.exists():
            df_obj = pd.read_parquet(obj_path)
            for gid, p_cnt, entities in zip(df_obj["global_id"], df_obj["person_count"], df_obj["high_conf_entities"]):
                self.gid_to_obj[int(gid)] = {
                    "person_count": int(p_cnt),
                    "entities": [str(e).lower() for e in entities] if hasattr(entities, '__iter__') else []
                }

        print("✅ IntraVideoTemporalReranker đã khởi tạo thành công!", flush=True)

    # =========================================================================
    # E1: GAUSSIAN NEIGHBOR TEMPORAL SMOOTHING
    # =========================================================================
    def compute_gaussian_neighbor_support(self, raw_scores: np.ndarray, pts_times: np.ndarray, sigma_sec: float = 1.5) -> np.ndarray:
        """
        Tính điểm hỗ trợ từ các khung hình lân cận theo hàm Gaussian:
        N_i = \\frac{\\sum_j w(t_i, t_j) * s_j}{\\sum_j w(t_i, t_j)}, với w(t_i, t_j) = exp(-(t_i - t_j)^2 / (2 * sigma^2))
        """
        n_frames = len(pts_times)
        if n_frames <= 1:
            return raw_scores.copy()

        # Ma trận khoảng cách thời gian (N x N)
        time_diffs = pts_times[:, None] - pts_times[None, :]
        weights = np.exp(-(time_diffs ** 2) / (2.0 * (sigma_sec ** 2)))
        
        # Chuẩn hóa tổng trọng số
        weight_sums = np.sum(weights, axis=1, keepdims=True)
        weight_sums[weight_sums == 0] = 1.0
        norm_weights = weights / weight_sums

        # Điểm neighbor support
        neighbor_scores = np.dot(norm_weights, raw_scores)
        return neighbor_scores

    # =========================================================================
    # E2: QUERY DECOMPOSITION & SLIDING WINDOW CUE COVERAGE (Q2E)
    # =========================================================================
    def compute_cue_coverage(self, cue_vecs: list[np.ndarray], video_feats: np.ndarray, pts_times: np.ndarray, window_sec: float = 6.0) -> np.ndarray:
        """
        Đo lường mức độ bao phủ các chi tiết (Cues) trong một cửa sổ thời gian trượt (Sliding Window):
        Coverage_i = \\frac{1}{m} \\sum_{c=1}^m \\max_{j \\in [t_i - W, t_i + W]} Sim(v(C_c), v(f_j))
        """
        n_frames = len(pts_times)
        if not cue_vecs or n_frames == 0:
            return np.zeros(n_frames, dtype=np.float32)

        # Tính toán similarity matrix cho từng cue: (m x n_frames)
        m_cues = len(cue_vecs)
        cue_mat = np.vstack([c.flatten() for c in cue_vecs]).astype(np.float32) # (m, 1152)
        sim_mat = np.dot(cue_mat, video_feats.T) # (m, n_frames)

        coverage_scores = np.zeros(n_frames, dtype=np.float32)
        for i in range(n_frames):
            t_center = pts_times[i]
            in_window_mask = np.abs(pts_times - t_center) <= window_sec
            if np.any(in_window_mask):
                window_sims = sim_mat[:, in_window_mask] # (m, window_size)
                max_per_cue = np.max(window_sims, axis=1) # (m,)
                coverage_scores[i] = np.mean(max_per_cue)
            else:
                coverage_scores[i] = np.mean(sim_mat[:, i])

        return coverage_scores

    # =========================================================================
    # E3: TIME-ALIGNED MULTI-MODAL TIMELINE FUSION (ASR, OCR, OBJECTS)
    # =========================================================================
    def compute_timeline_multimodal_scores(self, video_id: str, pts_times: np.ndarray, frame_indices: np.ndarray, global_ids: np.ndarray, query_text: str, gate_info: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Tính điểm ASR, OCR, Objects gắn trực tiếp vào mốc thời gian thực của từng frame.
        """
        n_frames = len(pts_times)
        asr_scores = np.zeros(n_frames, dtype=np.float32)
        ocr_scores = np.zeros(n_frames, dtype=np.float32)
        obj_scores = np.zeros(n_frames, dtype=np.float32)

        q_tokens = tokenize_clean(query_text)
        if not q_tokens or n_frames == 0:
            return asr_scores, ocr_scores, obj_scores

        # 1. ASR Alignment theo khoảng thời gian [start, end]
        if gate_info and gate_info.get("has_asr", False) and video_id in self.video_to_asr:
            asr_chunks = self.video_to_asr[video_id]
            if asr_chunks:
                corpus = [c["tokens"] for c in asr_chunks]
                bm25_asr = BM25Okapi(corpus)
                chunk_scores = bm25_asr.get_scores(q_tokens)
                max_sc = np.max(chunk_scores) if len(chunk_scores) > 0 and np.max(chunk_scores) > 0 else 1.0
                norm_chunk_scores = chunk_scores / max_sc

                for c_idx, c in enumerate(asr_chunks):
                    sc = norm_chunk_scores[c_idx]
                    if sc > 0.1:
                        mask = (pts_times >= c["start"] - 1.0) & (pts_times <= c["end"] + 1.0)
                        asr_scores[mask] = np.maximum(asr_scores[mask], sc)

        # 2. OCR Alignment theo từng frame_idx cụ thể
        if gate_info and gate_info.get("has_ocr", False):
            ocr_terms = gate_info.get("ocr_keywords", [])
            ocr_q_tokens = tokenize_clean(" ".join(ocr_terms)) if ocr_terms else q_tokens

            for idx, f_idx in enumerate(frame_indices):
                key = (video_id, int(f_idx))
                if key in self.video_frame_to_ocr:
                    doc_tokens = self.video_frame_to_ocr[key]["tokens"]
                    overlap = len(set(ocr_q_tokens).intersection(set(doc_tokens)))
                    if overlap > 0:
                        ocr_scores[idx] = min(1.0, overlap / max(1, len(ocr_q_tokens)))

        # 3. Object & Person Count Alignment
        person_match = re.search(r'(\d+)\s*(người|phụ nữ|đàn ông|cầu thủ|vận động viên)', query_text.lower())
        target_persons = int(person_match.group(1)) if person_match else -1

        for idx, gid in enumerate(global_ids):
            if int(gid) in self.gid_to_obj:
                obj_info = self.gid_to_obj[int(gid)]
                if target_persons > 0:
                    detected_p = obj_info["person_count"]
                    if detected_p == target_persons:
                        obj_scores[idx] += 0.5
                    elif abs(detected_p - target_persons) == 1:
                        obj_scores[idx] += 0.2

        return asr_scores, ocr_scores, obj_scores

    # =========================================================================
    # RESCORE TOÀN DIỆN MỘT CANDIDATE VIDEO
    # =========================================================================
    def rescore_candidate_video(
        self,
        video_id: str,
        main_query_vec: np.ndarray,
        cue_vecs: list[np.ndarray] = None,
        query_text: str = "",
        gate_info: dict = None,
        use_neighbor: bool = True,
        use_cue: bool = True,
        use_multimodal: bool = True,
        sigma_sec: float = 4.0,
        alpha: float = 0.55,
        beta: float = 0.25,
        gamma: float = 0.20
    ) -> list[dict]:
        """
        Chấm điểm lại toàn bộ keyframes của 1 candidate video và trả về danh sách đã sắp xếp.
        """
        if video_id not in self.video_to_indices or self.siglip_features is None:
            return []

        v_info = self.video_to_indices[video_id]
        row_indices = v_info["row_indices"]
        frame_indices = v_info["frame_indices"]
        pts_times = v_info["pts_times"]
        global_ids = v_info["global_ids"]

        n_frames = len(row_indices)
        if n_frames == 0:
            return []

        # 1. Trích xuất ma trận đặc trưng SigLIP của video (cast sang float32): (n_frames, 1152)
        video_feats = self.siglip_features[row_indices].astype(np.float32)

        # 2. Tính điểm Dense Similarity cơ bản
        q_flat = main_query_vec.flatten().astype(np.float32)
        raw_dense_scores = np.dot(video_feats, q_flat)

        # 3. E1: Neighbor Temporal Smoothing
        if use_neighbor:
            neighbor_scores = self.compute_gaussian_neighbor_support(raw_dense_scores, pts_times, sigma_sec=sigma_sec)
        else:
            neighbor_scores = raw_dense_scores.copy()

        # 4. E2: Cue Coverage Score
        if use_cue and cue_vecs and len(cue_vecs) > 0:
            cue_coverage_scores = self.compute_cue_coverage(cue_vecs, video_feats, pts_times, window_sec=6.0)
        else:
            cue_coverage_scores = np.zeros(n_frames, dtype=np.float32)
            gamma = 0.0
            alpha = 0.65
            beta = 0.35

        # 5. E3: Multi-modal Timeline Score
        if use_multimodal and gate_info:
            asr_sc, ocr_sc, obj_sc = self.compute_timeline_multimodal_scores(
                video_id, pts_times, frame_indices, global_ids, query_text, gate_info
            )
            w_asr = 0.35 if gate_info.get("has_asr", False) else 0.0
            w_ocr = 0.35 if gate_info.get("has_ocr", False) else 0.0
            w_obj = 0.10 if np.any(obj_sc > 0) else 0.0
            multi_bonus = w_asr * asr_sc + w_ocr * ocr_sc + w_obj * obj_sc
        else:
            multi_bonus = np.zeros(n_frames, dtype=np.float32)

        # 6. Tổng hợp điểm số cuối cùng cho từng Frame
        final_frame_scores = (
            alpha * raw_dense_scores +
            beta * neighbor_scores +
            gamma * cue_coverage_scores +
            multi_bonus
        )

        results = []
        for i in range(n_frames):
            results.append({
                "video_id": video_id,
                "frame_idx": int(frame_indices[i]),
                "pts_time": float(pts_times[i]),
                "global_id": int(global_ids[i]),
                "raw_score": float(raw_dense_scores[i]),
                "neighbor_score": float(neighbor_scores[i]),
                "cue_score": float(cue_coverage_scores[i]),
                "final_score": float(final_frame_scores[i])
            })

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results

    # =========================================================================
    # TEMPORAL CANDIDATE EXPANSION & DIVERSE HYPOTHESES (HEDGING RANK 1, 2, 3)
    # =========================================================================
    def extract_diverse_temporal_hypotheses(self, rescored_frames: list[dict], min_temporal_gap_sec: float = 8.0, top_k: int = 100) -> list[dict]:
        """
        Trích xuất các giả thuyết khung hình đa dạng theo từng đỉnh thời gian (Temporal Peaks):
        - Rank 1: Frame đỉnh cao nhất (Peak 1).
        - Rank 2: Frame kề cận trong Peak 1 (bảo hiểm lệch sampling interval).
        - Rank 3: Frame đỉnh của khoảnh khắc thời gian thứ hai (Peak 2).
        - Rank 4: Frame thứ 3 trong Peak 1.
        - Rank 5: Frame thứ 2 trong Peak 2...
        """
        if not rescored_frames:
            return []

        # 1. Tìm các đỉnh thời gian (Local Temporal Peaks)
        peaks = []
        for f in rescored_frames:
            t_curr = f["pts_time"]
            is_new_peak = True
            for p in peaks:
                if abs(t_curr - p["pts_time"]) < min_temporal_gap_sec:
                    is_new_peak = False
                    break
            if is_new_peak:
                peaks.append(f)
                if len(peaks) >= 5:
                    break

        if not peaks:
            return rescored_frames[:top_k]

        # 2. Xây dựng danh sách Hedging đa dạng
        selected = []
        seen_frames = set()

        def add_frame(f):
            f_key = (f["video_id"], f["frame_idx"])
            if f_key not in seen_frames:
                selected.append(f)
                seen_frames.add(f_key)

        # Rank 1: Đỉnh số 1
        add_frame(peaks[0])

        # Rank 2: Frame lân cận điểm cao trong cùng vùng đỉnh 1
        t_peak1 = peaks[0]["pts_time"]
        near_peak1 = [f for f in rescored_frames if 0 < abs(f["pts_time"] - t_peak1) <= min_temporal_gap_sec]
        if near_peak1:
            add_frame(near_peak1[0])

        # Rank 3: Đỉnh số 2 (nếu có)
        if len(peaks) > 1:
            add_frame(peaks[1])

        # Rank 4: Frame thứ 3 trong vùng đỉnh 1
        if len(near_peak1) > 1:
            add_frame(near_peak1[1])

        # Rank 5: Frame lân cận đỉnh số 2 (nếu có)
        if len(peaks) > 1:
            t_peak2 = peaks[1]["pts_time"]
            near_peak2 = [f for f in rescored_frames if 0 < abs(f["pts_time"] - t_peak2) <= min_temporal_gap_sec]
            if near_peak2:
                add_frame(near_peak2[0])

        # Điền các frame còn lại theo thứ tự điểm giảm dần
        for f in rescored_frames:
            add_frame(f)
            if len(selected) >= top_k:
                break

        return selected[:top_k]

    # =========================================================================
    # EVIDENCE-GROUNDED VLM REASONING RERANKER (GEMINI VISION VERIFICATION)
    # =========================================================================
    def verify_top_frames_with_vlm(
        self,
        query_text: str,
        candidates: list[dict],
        top_n: int = 4
    ) -> list[dict]:
        """
        Sử dụng Gemini 3.5 Flash Lite Vision soi ảnh trực tiếp các frame Top đầu
        để kiểm chứng độ khớp chi tiết (Multi-Attribute, Object & Action Verification).
        """
        if not candidates or top_n <= 0:
            return candidates

        from src.query.gemini_router import GeminiKeyPool
        from src.retrieval.keyframe_loader import KeyframeZipLoader
        from google import genai
        from google.genai import types

        img_loader = KeyframeZipLoader()
        key_pool = GeminiKeyPool()

        inspect_cands = candidates[:top_n]
        remaining = candidates[top_n:]

        vlm_scores = []
        for cand in inspect_cands:
            v_id = cand["video_id"]
            f_idx = cand["frame_idx"]
            img = img_loader.load_frame(v_id, f_idx)
            if img is None:
                vlm_scores.append(cand)
                continue

            prompt = f"""Bạn là Trợ lý AI giám khảo cuộc thi AI Challenge TP.HCM.
Nhiệm vụ: Hãy quan sát kỹ bức ảnh khung hình video được cung cấp và đánh giá mức độ khớp với mô tả sau:
Mô tả truy vấn: "{query_text}"

Hãy kiểm tra:
1. Đối tượng / Nhân vật chính (Chính xác người, động vật, xe cộ...)
2. Màu sắc, trang phục, phụ kiện (khăn trùm đầu, kính râm, màu áo, điện thoại...)
3. Hành động / Thao tác (quấn sarong, gập bánh xèo, làm việc...)

Hãy cho điểm từ 0.00 đến 1.00 về mức độ khớp thực tế trên ảnh (1.00 là khớp hoàn hảo tất cả chi tiết).
Chỉ trả về 1 số duy nhất từ 0.00 đến 1.00 (ví dụ: 0.95 hoặc 0.20), không giải thích."""

            vlm_sc = 0.5
            for _ in range(3):
                api_key = key_pool.get_next_key()
                if not api_key:
                    break
                try:
                    client = genai.Client(api_key=api_key)
                    resp = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[img, prompt]
                    )
                    text = resp.text.strip()
                    nums = re.findall(r"0\.\d+|1\.00|1\.0|0|1", text)
                    if nums:
                        vlm_sc = float(nums[0])
                    break
                except Exception:
                    time.sleep(1)

            cand_copy = dict(cand)
            cand_copy["vlm_score"] = vlm_sc
            # Calibrated boost: bảo vệ không phá vỡ khoảng cách điểm Stage-1 quá đà
            base_sc = cand.get("final_score", cand.get("score", 0.5))
            cand_copy["score"] = base_sc + 0.35 * (vlm_sc - 0.5)
            vlm_scores.append(cand_copy)

        vlm_scores.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return vlm_scores + remaining
