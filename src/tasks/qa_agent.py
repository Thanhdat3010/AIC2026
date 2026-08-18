import os
import sys
import io
import json
import time
from pathlib import Path
from PIL import Image

# Force UTF-8 on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from google import genai
from google.genai import types
from src.query.gemini_router import GeminiKeyPool
from src.retrieval.keyframe_loader import KeyframeZipLoader

class VisualQAAgent:
    """
    Two-Stage Visual QA Agent (Chuyên trách bài toán Video Visual Question Answering):
    - Stage 1: Nhận Top candidate frames từ Hybrid Retrieval Engine.
    - Stage 2: Đọc ảnh Keyframe thật từ KeyframeZipLoader và gửi cho Gemini 3.5 Flash Lite Vision.
    - Tích hợp Dynamic Multi-Crop / Dynamic Focus (V* / DyFo style) để phát hiện chi tiết nhỏ ở góc ảnh.
    """
    def __init__(self, key_pool: GeminiKeyPool = None):
        if key_pool is None:
            self.key_pool = GeminiKeyPool()
        else:
            self.key_pool = key_pool

        self.img_loader = KeyframeZipLoader()
        
        import pandas as pd
        from collections import defaultdict
        
        self.frame_to_time = {}
        self.video_to_asr = defaultdict(list)
        self.video_frame_to_ocr = {}
        
        proc_dir = BASE_DIR / "data" / "batch_1" / "processed"
        
        frames_path = proc_dir / "frames.parquet"
        if frames_path.exists():
            df_frames = pd.read_parquet(frames_path)
            for v_id, f_idx, pts in zip(df_frames["video_id"], df_frames["frame_idx"], df_frames["pts_time"]):
                self.frame_to_time[(v_id, int(f_idx))] = float(pts)
                
        asr_path = proc_dir / "transcripts.parquet"
        if asr_path.exists():
            df_asr = pd.read_parquet(asr_path)
            for v_id, s_t, e_t, txt in zip(df_asr["video_id"], df_asr["start_time"], df_asr["end_time"], df_asr["transcript"]):
                self.video_to_asr[v_id].append({"start": float(s_t), "end": float(e_t), "text": str(txt)})
                
        ocr_path = proc_dir / "ocr_results.parquet"
        if ocr_path.exists():
            df_ocr = pd.read_parquet(ocr_path)
            for v_id, f_idx, txt in zip(df_ocr["video_id"], df_ocr["frame_idx"], df_ocr["ocr_text"]):
                if isinstance(txt, str) and txt.strip():
                    self.video_frame_to_ocr[(v_id, int(f_idx))] = txt

    def _generate_multi_crops(self, img: Image.Image) -> list[Image.Image]:
        """Tạo 4 ảnh crop lưới 2x2 với độ phân giải cao để quan sát chi tiết nhỏ."""
        w, h = img.size
        mid_x, mid_y = w // 2, h // 2
        pad_x, pad_y = int(w * 0.05), int(h * 0.05)

        crops = [
            img.crop((0, 0, min(w, mid_x + pad_x), min(h, mid_y + pad_y))),                         # Top-Left
            img.crop((max(0, mid_x - pad_x), 0, w, min(h, mid_y + pad_y))),                         # Top-Right
            img.crop((0, max(0, mid_y - pad_y), min(w, mid_x + pad_x), h)),                         # Bottom-Left
            img.crop((max(0, mid_x - pad_x), max(0, mid_y - pad_y), w, h))                          # Bottom-Right
        ]
        for c in crops:
            if max(c.size) > 1024:
                c.thumbnail((1024, 1024))
        return crops

    def answer_and_rerank(
        self,
        qa_question: str,
        candidates: list[dict],
        max_inspect_frames: int = 5,
        use_multi_crop: bool = True,
        gate_info: dict = None
    ) -> tuple[str, list[dict]]:
        """
        Duyệt qua các khung hình Top đầu, tìm câu trả lời và tái xếp hạng lại danh sách.
        Returns: (best_answer_text, reranked_candidates)
        """
        if not candidates:
            return "Không xác định", []

        inspect_cands = candidates[:max_inspect_frames]
        best_answer = "Không xác định"
        best_confidence = 0.0
        best_cand_idx = 0

        best_cand_idx = 0
        for idx, cand in enumerate(inspect_cands):
            v_id = cand["video_id"]
            f_idx = cand["frame_idx"]
            img = self.img_loader.get_image(v_id, f_idx)
            if img is None:
                continue

            img_full = img.copy()
            if max(img_full.size) > 1024:
                img_full.thumbnail((1024, 1024))

            contents = [img_full]
            
            if use_multi_crop:
                crops = self._generate_multi_crops(img)
                contents.extend(crops)
                
            # Tri-modal Context Injection
            pts = self.frame_to_time.get((v_id, f_idx), 0.0)
            context_str = ""
            
            ocr_txt = self.video_frame_to_ocr.get((v_id, f_idx), "")
            if ocr_txt:
                context_str += f"\n[Hệ thống OCR nhận diện được (có thể chứa nhiễu)]: {ocr_txt}"
                
            asr_texts = []
            for chunk in self.video_to_asr.get(v_id, []):
                if chunk["start"] - 5.0 <= pts <= chunk["end"] + 5.0:
                    asr_texts.append(chunk["text"])
            if asr_texts:
                context_str += f"\n[Hệ thống ASR nghe được (có thể chứa nhiễu)]: {' | '.join(asr_texts)}"
                
            prompt = f"""Bạn là Trợ lý AI tham gia cuộc thi AI Challenge TP.HCM.
Nhiệm vụ: Quan sát kỹ ảnh toàn cảnh (và ảnh crop phóng to nếu có) để trả lời câu hỏi (QA) bằng TIẾNG VIỆT ĐẦY ĐỦ Ý, CHÍNH XÁC VÀ ĐÚNG TRỌNG TÂM (đáp án dưới 100 ký tự).
(Lưu ý Agentic: Dữ liệu OCR/ASR bên dưới có thể chứa nhiễu nhận diện, bạn phải TỰ ĐỐI CHIẾU LẠI với hình ảnh thực tế, không tin tưởng mù quáng vào văn bản).
Câu hỏi: {qa_question}{context_str}

Trả về định dạng JSON thuần túy:
{{
  "has_answer": true/false (ảnh có chứa dữ liệu để trả lời câu hỏi không),
  "answer": "câu trả lời chính xác",
  "confidence": điểm tin cậy từ 0.0 đến 1.0
}}"""
            contents.append(prompt)

            key = self.key_pool.get_next_key()
            client = genai.Client(api_key=key)

            try:
                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1
                    )
                )
                res_json = json.loads(response.text.strip())
                has_ans = res_json.get("has_answer", False)
                ans = res_json.get("answer", "").strip()
                conf = float(res_json.get("confidence", 0.0))

                cand["qa_answer"] = ans
                cand["answer"] = ans
                cand["qa_confidence"] = conf

                if has_ans and conf > best_confidence and ans.lower() not in ["không xác định", "không có", "unknown", "n/a"]:
                    best_confidence = conf
                    best_answer = ans
                    best_cand_idx = idx

            except Exception as e:
                cand["qa_answer"] = "Lỗi API"
                cand["qa_confidence"] = 0.0

        # Nếu tìm thấy khung hình có câu trả lời tự tin, hoán đổi khung hình đó lên Rank #1
        reranked = candidates.copy()
        if best_confidence > 0.4 and best_cand_idx > 0:
            winner = reranked.pop(best_cand_idx)
            winner["score"] = reranked[0]["score"] + 0.1 # Đẩy điểm vượt trội
            reranked.insert(0, winner)

        for r, c in enumerate(reranked, 1):
            c["rank"] = r
            if "answer" not in c or not c["answer"]:
                c["answer"] = best_answer if best_answer not in ["Không xác định", "Lỗi API"] else ""

        return best_answer, reranked

if __name__ == "__main__":
    agent = VisualQAAgent()
    sample_q = "Khi 2 người đàn ông đang di chuyển chiếc xe máy chở nhiều măng le, người phía trước đội gì trên đầu?"
    sample_cands = [
        {"rank": 1, "video_id": "L29_V001", "frame_idx": 24000, "score": 0.015},
        {"rank": 2, "video_id": "L27_V002", "frame_idx": 920, "score": 0.014}, # Ground truth frame
    ]
    print(f"\n🔎 [TEST QA AGENT]: {sample_q}")
    ans, reranked = agent.answer_and_rerank(sample_q, sample_cands, max_inspect_frames=2, use_multi_crop=True)
    print(f"🎯 Câu trả lời sinh ra: '{ans}'")
    print(f"🏆 Thứ hạng sau khi Re-rank bởi Gemini Vision:")
    for r in reranked:
        print(f"   + Rank #{r['rank']} | Video: {r['video_id']} | Frame: {r['frame_idx']} | Answer: {r.get('qa_answer')} | Conf: {r.get('qa_confidence')}")
