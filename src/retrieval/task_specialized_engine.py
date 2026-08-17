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
    """
    def __init__(self, engine: str = "siglip2", batch: str = "batch_1", k_rrf: int = 60):
        self.engine = engine
        self.batch = batch
        self.k_rrf = k_rrf

        print(f"[*] Đang khởi tạo TaskSpecializedEngine [{engine.upper()}]...", flush=True)
        self.text_encoder = UnifiedTextEncoder(engine=engine)
        self.faiss_index, self.df_frames = load_faiss_index(engine=engine, batch=batch)
        self.bm25_indexer = BM25MultiIndexer(batch=batch)
        self.key_pool = GeminiKeyPool()
        self.router = GeminiQueryRouter()
        self.modality_gate = ModalityGate()
        self.qa_agent = VisualQAAgent(key_pool=self.key_pool)
        print(f"✅ TaskSpecializedEngine [{engine.upper()}] đã sẵn sàng!", flush=True)

    # =========================================================================
    # 1. SPECIALIST 1: TEXTUAL KIS PIPELINE (Tìm kiếm chính xác khung hình)
    # =========================================================================
    def search_kis(self, query_text: str, top_k: int = 100, custom_en_query: str = None) -> tuple[list[dict], dict, float]:
        """
        Chiến thuật KIS: 
        1. Dịch 1 câu tiếng Anh chuẩn nhất từ Gemini.
        2. Kiểm tra tín hiệu chữ viết (OCR). Nếu có chữ trong ngoặc kép -> Kết hợp BM25 OCR.
        3. Nếu thuần thị giác -> Chạy 100% SigLIP 2 FAISS (1152d) để đạt R@1 cao nhất.
        """
        t0 = time.time()
        gate_info = self.modality_gate.analyze(query_text)
        
        if custom_en_query:
            en_prompt = custom_en_query
        else:
            q_info = self.router.transform_query(query_text)
            en_prompt = q_info["visual_prompts"][0]

        q_vec = self.text_encoder.encode_text(en_prompt)
        scores, indices = self.faiss_index.search(q_vec, top_k * 2)

        # Nếu là thuần thị giác (chiếm 90% trường hợp): Lấy trực tiếp kết quả FAISS
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
            # Nếu có từ khóa OCR rõ ràng: Kết hợp RRF với BM25 OCR
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

    # =========================================================================
    # 2. SPECIALIST 2: VISUAL Q&A PIPELINE (Hỏi - Đáp Trực quan)
    # =========================================================================
    def search_qa(self, query_text: str, top_k: int = 100, custom_en_query: str = None) -> tuple[list[dict], dict, float]:
        """
        Chiến thuật Q&A:
        1. Kiểm tra nếu hỏi về lời thoại phỏng vấn -> Kết hợp BM25 ASR.
        2. SigLIP 2 tìm Top 100 khung hình chuẩn.
        3. Gemini 3.5 Flash Lite Vision soi ảnh Top 3-4 khung hình để sinh câu trả lời ngắn gọn (<100 ký tự).
        4. Ghi câu trả lời vào từng ứng viên để xuất đúng format BTC: <video>, <frame>, "<answer>".
        """
        t0 = time.time()
        gate_info = self.modality_gate.analyze(query_text)

        if custom_en_query:
            en_prompt = custom_en_query
        else:
            q_info = self.router.transform_query(query_text)
            en_prompt = q_info["visual_prompts"][0]

        q_vec = self.text_encoder.encode_text(en_prompt)
        scores, indices = self.faiss_index.search(q_vec, top_k * 2)

        if not gate_info["has_asr"]:
            candidates = []
            for rank, (sim, idx) in enumerate(zip(scores[0][:top_k], indices[0][:top_k]), 1):
                row = self.df_frames.iloc[idx]
                candidates.append({
                    "rank": rank,
                    "video_id": row["video_id"],
                    "frame_idx": int(row["frame_idx"]),
                    "global_id": int(row["global_id"]),
                    "score": float(sim)
                })
        else:
            # Có tín hiệu phỏng vấn/lời thoại -> RRF với ASR
            frame_scores = defaultdict(lambda: {"score": 0.0, "video_id": "", "frame_idx": 0, "global_id": -1})
            w_vis, w_asr = 1.0, 0.45

            for rank, (sim, idx) in enumerate(zip(scores[0], indices[0]), 1):
                row = self.df_frames.iloc[idx]
                key = (row["video_id"], int(row["frame_idx"]))
                frame_scores[key]["score"] += w_vis / (self.k_rrf + rank)
                frame_scores[key]["video_id"] = row["video_id"]
                frame_scores[key]["frame_idx"] = int(row["frame_idx"])
                frame_scores[key]["global_id"] = int(row["global_id"])

            asr_query = " ".join(gate_info["asr_keywords"]) if gate_info["asr_keywords"] else query_text
            asr_results = self.bm25_indexer.search_asr(asr_query, top_k=top_k * 2)
            for rank, doc in enumerate(asr_results, 1):
                if doc["frame_idx"] < 0:
                    continue
                key = (doc["video_id"], doc["frame_idx"])
                frame_scores[key]["score"] += w_asr / (self.k_rrf + rank)
                frame_scores[key]["video_id"] = doc["video_id"]
                frame_scores[key]["frame_idx"] = doc["frame_idx"]

            cand_list = list(frame_scores.values())
            cand_list.sort(key=lambda x: x["score"], reverse=True)
            candidates = []
            for rank, cand in enumerate(cand_list[:top_k], 1):
                cand["rank"] = rank
                candidates.append(cand)

        # Gọi Gemini Vision trả lời câu hỏi
        best_answer, reranked = self.qa_agent.answer_and_rerank(
            qa_question=query_text,
            candidates=candidates,
            max_inspect_frames=4
        )

        for c in reranked:
            if "qa_answer" not in c or not c["qa_answer"] or c["qa_answer"].lower() in ["lỗi api", "n/a"]:
                c["qa_answer"] = best_answer

        latency = (time.time() - t0) * 1000
        return reranked, {"en_prompt": en_prompt, "generated_qa_answer": best_answer, "gate_info": gate_info}, latency

    # =========================================================================
    # 3. SPECIALIST 3: TRAKE PIPELINE (Căn chỉnh chuỗi sự kiện theo thời gian)
    # =========================================================================
    def search_trake(self, query_text: str, top_k: int = 100, custom_en_query: str = None) -> tuple[list[dict], dict, float]:
        """
        Chiến thuật TRAKE:
        1. Phân rã câu hỏi thành N sự kiện con: E_1 -> E_2 -> ... -> E_N.
        2. Tìm kiếm ứng viên trên từng sự kiện và tính điểm tổng hợp cho từng Video.
        3. Trong từng video hàng đầu, áp dụng thuật toán Greedy/DP chọn chuỗi khung hình tăng dần:
           Frame(E_1) <= Frame(E_2) <= ... <= Frame(E_N).
        4. Xuất danh sách 100 dự đoán chuẩn format: <video>, <f_1>, <f_2>, ..., <f_N>.
        """
        t0 = time.time()

        # 1. Phân rã các sự kiện
        q_info = self.router.transform_query(query_text)
        events = q_info.get("trake_events", [])
        if not events or len(events) < 2:
            parts = [p.strip() for p in re.split(r"[,;]|\bvà\b|\btiếp tục\b|\bcuối cùng\b", query_text) if p.strip()]
            events = parts if len(parts) >= 2 else [query_text, query_text]

        n_events = len(events)
        event_vecs = [self.text_encoder.encode_text(ev) for ev in events]

        # 2. Tìm kiếm ứng viên cho từng sự kiện
        video_event_hits = defaultdict(lambda: defaultdict(list))
        for e_idx, vec in enumerate(event_vecs):
            scores, indices = self.faiss_index.search(vec, 200)
            for rank, (sc, idx) in enumerate(zip(scores[0], indices[0]), 1):
                row = self.df_frames.iloc[idx]
                v_id = row["video_id"]
                f_idx = int(row["frame_idx"])
                video_event_hits[v_id][e_idx].append({"frame_idx": f_idx, "score": float(sc), "rank": rank})

        # 3. Chấm điểm video theo độ bao phủ sự kiện và điểm similarity
        video_scores = []
        for v_id, e_dict in video_event_hits.items():
            coverage = len(e_dict)
            avg_sim = np.mean([max([c["score"] for c in cands]) for cands in e_dict.values()])
            total_v_score = (coverage / n_events) * 2.0 + avg_sim
            video_scores.append((v_id, total_v_score))

        video_scores.sort(key=lambda x: x[1], reverse=True)
        top_videos = [v[0] for v in video_scores[:top_k]]

        # 4. Trích xuất chuỗi frame tăng dần
        results = []
        for rank, v_id in enumerate(top_videos, 1):
            e_dict = video_event_hits[v_id]
            df_v = self.df_frames[self.df_frames["video_id"] == v_id].sort_values("frame_idx")
            all_v_frames = df_v["frame_idx"].tolist()

            chosen_frames = []
            last_f = -1
            for e_idx in range(n_events):
                cands = e_dict.get(e_idx, [])
                valid_cands = [c for c in cands if c["frame_idx"] >= last_f]
                if valid_cands:
                    best_c = max(valid_cands, key=lambda x: x["score"])
                    chosen_frames.append(best_c["frame_idx"])
                    last_f = best_c["frame_idx"]
                else:
                    subsequent = [f for f in all_v_frames if f > last_f]
                    if subsequent:
                        chosen_frames.append(subsequent[0])
                        last_f = subsequent[0]
                    elif all_v_frames:
                        chosen_frames.append(all_v_frames[-1])
                    else:
                        chosen_frames.append(0)

            results.append({
                "rank": rank,
                "video_id": v_id,
                "frame_idx": chosen_frames[0] if chosen_frames else 0,
                "event_frames": chosen_frames,
                "score": float(video_scores[rank-1][1])
            })

        latency = (time.time() - t0) * 1000
        return results, {"events": events}, latency

    # =========================================================================
    # 4. TỰ ĐỘNG ĐỊNH TUYẾN CHUYÊN BIỆT (AUTOROUTE SEARCH)
    # =========================================================================
    def search(self, query_text: str, task_type: str = "kis", top_k: int = 100, custom_en_query: str = None) -> tuple[list[dict], dict, float]:
        ttype = task_type.lower()
        if "trake" in ttype:
            return self.search_trake(query_text, top_k=top_k, custom_en_query=custom_en_query)
        elif "qa" in ttype or "q&a" in ttype:
            return self.search_qa(query_text, top_k=top_k, custom_en_query=custom_en_query)
        else:
            return self.search_kis(query_text, top_k=top_k, custom_en_query=custom_en_query)
