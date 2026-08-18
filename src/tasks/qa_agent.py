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
    - Chức năng:
      1. Trả lời chính xác câu hỏi văn bản (1-5 từ).
      2. Chấm điểm độ tin cậy để hoán đổi đưa đúng khung hình chứa câu trả lời lên Rank #1.
    """
    def __init__(self, key_pool: GeminiKeyPool = None):
        if key_pool is None:
            self.key_pool = GeminiKeyPool()
        else:
            self.key_pool = key_pool

        self.img_loader = KeyframeZipLoader()

    def answer_and_rerank(
        self,
        qa_question: str,
        candidates: list[dict],
        max_inspect_frames: int = 5
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

        prompt_template = f"""Bạn là Trợ lý AI giám khảo cuộc thi AI Challenge TP.HCM.
Nhiệm vụ: Hãy quan sát kỹ bức ảnh khung hình video được cung cấp và trả lời câu hỏi sau bằng TIẾNG VIỆT NGẮN GỌN (từ 1 đến 5 từ, không giải thích dài dòng).

Câu hỏi: {qa_question}

Trả về định dạng JSON thuần túy:
{{
  "has_answer": true hoặc false (ảnh có chứa đối tượng/hành động được hỏi không),
  "answer": "câu trả lời cực ngắn gọn",
  "confidence": điểm tin cậy từ 0.0 đến 1.0
}}
"""

        for idx, cand in enumerate(inspect_cands):
            v_id = cand["video_id"]
            f_idx = cand["frame_idx"]
            img = self.img_loader.get_image(v_id, f_idx)
            if img is None:
                continue

            # Nén nhẹ ảnh nếu quá lớn để tối ưu tốc độ gọi API (<0.5s)
            if max(img.size) > 1024:
                img.thumbnail((1024, 1024))

            key = self.key_pool.get_next_key()
            client = genai.Client(api_key=key)

            try:
                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=[img, prompt_template],
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
    ans, reranked = agent.answer_and_rerank(sample_q, sample_cands, max_inspect_frames=2)
    print(f"🎯 Câu trả lời sinh ra: '{ans}'")
    print(f"🏆 Thứ hạng sau khi Re-rank bởi Gemini Vision:")
    for r in reranked:
        print(f"   + Rank #{r['rank']} | Video: {r['video_id']} | Frame: {r['frame_idx']} | Answer: {r.get('qa_answer')} | Conf: {r.get('qa_confidence')}")
