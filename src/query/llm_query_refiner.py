import os
import re
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

from src.query.gemini_router import GeminiKeyPool

class LLMQueryRefiner:
    """
    Bộ Tiền Xử Lý & Điều Phối Trọng Số Đa Phương Thức Thông Minh (AIC 2026).
    Tích hợp:
    1. Thuật toán 1: LLM Structured Semantic Intent Classifier (Gemini 3.5 Flash Lite JSON Schema).
    2. Thuật toán 2: Prototype Vector Cosine Gating (Định tuyến không gian vector mượt mà).
    3. Trích xuất chính xác OCR search keywords, ASR search keywords và TRAKE Sub-Events.
    4. Khóa duy nhất mô hình gemini-3.5-flash-lite bảo toàn quota API.
    """
    def __init__(self, model_name: str = "gemini-3.5-flash-lite"):
        self.model_name = model_name
        self.key_pool = GeminiKeyPool()
        self._cache: Dict[str, Dict[str, Any]] = {}
        
        # Prototype anchors cho Thuật toán 2 (Vector Cosine Gating)
        self.prototypes_text = {
            "vis": "Hình ảnh trực quan, góc quay, màu sắc trang phục, cử chỉ hành động của con người, phong cảnh thiên nhiên, đồ vật chuyển động.",
            "asr": "Nội dung lời nói, bài phát biểu, phỏng vấn nhân vật, lời thuyết minh phóng sự, tin tức thời sự, ca sĩ hát bài hát.",
            "ocr": "Văn bản chữ viết trên màn hình, biển hiệu cửa hàng, logo thương hiệu, băng rôn khẩu hiệu, biển số xe, bảng tên phụ đề."
        }
        self.prototype_vectors: Optional[Dict[str, np.ndarray]] = None

    def init_prototype_vectors(self, text_encoder):
        """Khởi tạo vector đặc trưng cho các Prototype Anchors."""
        if self.prototype_vectors is None:
            self.prototype_vectors = {}
            for k, txt in self.prototypes_text.items():
                vec = text_encoder.encode_text(txt)
                if vec.ndim > 1:
                    vec = vec.squeeze()
                norm = np.linalg.norm(vec)
                self.prototype_vectors[k] = (vec / norm) if norm > 1e-6 else vec

    def compute_prototype_gating(self, query_vec: np.ndarray, temperature: float = 0.5) -> Tuple[float, float, float]:
        """
        Thuật toán 2: Prototype Vector Cosine Gating
        Tính toán trọng số (w_vis, w_ocr, w_asr) dựa trên khoảng cách Cosine Softmax.
        """
        if self.prototype_vectors is None:
            return 0.70, 0.15, 0.15

        q = query_vec.squeeze()
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-6:
            q = q / q_norm

        sims = {}
        for k in ["vis", "ocr", "asr"]:
            p = self.prototype_vectors[k]
            sims[k] = float(np.dot(q, p))

        # Softmax với nhiệt độ temperature
        exp_vals = {k: np.exp(sims[k] / temperature) for k in sims}
        sum_exp = sum(exp_vals.values())
        w_vis = exp_vals["vis"] / sum_exp
        w_ocr = exp_vals["ocr"] / sum_exp
        w_asr = exp_vals["asr"] / sum_exp
        return float(w_vis), float(w_ocr), float(w_asr)

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Gọi Gemini API qua Key Pool với mô hình cố định gemini-3.5-flash-lite."""
        if not self.key_pool.gemini_keys:
            return None

        keys = list(self.key_pool.gemini_keys)
        random.shuffle(keys)

        for key in keys:
            try:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=key)
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json"
                    )
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                # Thử fallback qua google.generativeai nếu cần
                try:
                    import google.generativeai as gai
                    gai.configure(api_key=key)
                    model = gai.GenerativeModel(self.model_name)
                    res = model.generate_content(prompt)
                    if res and res.text:
                        return res.text.strip()
                except Exception:
                    pass
                continue

        return None

    def refine_query(self, raw_query: str, task_type: str = "auto") -> Dict[str, Any]:
        """
        Thuật toán 1: LLM Structured Semantic Intent Classifier
        Phân tích và làm giàu truy vấn từ đề bài BTC bằng Gemini 3.5 Flash Lite.
        """
        raw_clean = raw_query.strip()
        cache_key = f"{task_type}_{raw_clean}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        lower_q = raw_clean.lower()
        if task_type == "auto":
            if any(w in lower_q for w in ["bao nhiêu", "là gì", "ở đâu", "ai là", "màu gì", "như thế nào", "tên của", "cuối cùng trên cân"]):
                task_type = "qa"
            elif any(w in lower_q for w in ["e1", "e2", "chuỗi hành động", "lần lượt", "thứ tự:", "khoảnh khắc đầu tiên"]):
                task_type = "trake"
            else:
                task_type = "kis"

        is_count_query = bool(re.search(r"(bao nhiêu|con số|mấy|số lượng|hiển thị|ghi trên)", lower_q))

        prompt = f"""Bạn là Chuyên gia Xử lý Ngôn ngữ & Truy xuất Video Đa Phương Thức (DIEM CVPR 2024 / ROCLING 2025 Framework).
Nhiệm vụ: Phân tích cấu trúc ngữ nghĩa câu truy vấn tiếng Việt từ Ban tổ chức thành các thành phần thị giác, thoại, chữ viết và câu hỏi trực diện.

Đề bài gốc: "{raw_clean}"
Loại bài toán: {task_type.upper()}

Hãy trả về duy nhất định dạng JSON sau (không thêm văn bản ngoài JSON):
{{
  "cleaned_vi": "Câu tiếng Việt đã chuẩn hóa",
  "visual_scene_vi": "Mô tả bối cảnh thị giác cốt lõi bằng tiếng Việt (loại bỏ từ nghi vấn như 'gì', 'ai', 'mấy giờ', 'như thế nào', chỉ giữ lại Actor, Action, Object, Scene)",
  "visual_scene_en": "Detailed English visual scene description focusing on visual elements for SigLIP-2 retrieval",
  "english_visual": "Detailed English visual scene description",
  "qa_direct_question": "Câu hỏi trực diện ngắn gọn (Ví dụ: 'Người đi phía trước đội gì trên đầu?')",
  "ocr_keywords": ["Từ khóa văn bản trên màn hình/biển hiệu/slide (nếu có)"],
  "asr_keywords": ["Từ khóa nội dung lời thoại/phỏng vấn/thuyết minh (nếu có)"],
  "is_dialogue_query": true,
  "sub_events_vi": ["E1: Sự kiện 1", "E2: Sự kiện 2", "E3: Sự kiện 3"],
  "sub_events_en": ["E1: Event 1 description", "E2: Event 2 description", "E3: Event 3 description"]
}}"""

        resp = self._call_llm(prompt)
        parsed = None

        if resp:
            try:
                clean_json = re.sub(r"^```json\s*", "", resp, flags=re.IGNORECASE)
                clean_json = re.sub(r"^```\s*", "", clean_json)
                clean_json = re.sub(r"\s*```$", "", clean_json)
                parsed = json.loads(clean_json)
            except Exception:
                parsed = None

        if not parsed:
            parsed = self._fallback_refine(raw_clean, task_type)

        # Chuẩn hóa trọng số tổng = 1.0
        v_w = float(parsed.get("visual_relevance", 0.70))
        o_w = float(parsed.get("ocr_relevance", 0.15))
        a_w = float(parsed.get("asr_relevance", 0.15))
        s = v_w + o_w + a_w
        if s > 1e-6:
            parsed["weights"] = {
                "visual": v_w / s,
                "ocr": o_w / s,
                "asr": a_w / s
            }
        else:
            parsed["weights"] = {"visual": 0.80, "ocr": 0.10, "asr": 0.10}

        parsed["task_type"] = task_type
        parsed["is_count_query"] = is_count_query
        self._cache[cache_key] = parsed
        return parsed

    def _fallback_refine(self, raw_query: str, task_type: str) -> Dict[str, Any]:
        """Fallback an toàn khi offline hoặc không có API key."""
        lines = [l.strip() for l in raw_query.split("\n") if l.strip()]
        sub_events_vi = []
        for l in lines:
            if re.match(r"^(?:[eE]|sự kiện|event|bước|cảnh|scene|giai đoạn)\s*\d+[:\s\-\.]", l, re.IGNORECASE) or re.match(r"^\d+[\.\)]\s*", l) or "khoảnh khắc" in l.lower():
                sub_events_vi.append(l)

        if not sub_events_vi and task_type == "trake":
            parts = re.split(r"(đầu tiên|tiếp theo|sau đó|cuối cùng|kế tiếp)", raw_query, flags=re.IGNORECASE)
            sub_events_vi = [p.strip() for p in parts if len(p.strip()) > 10]

        has_ocr_signal = bool(re.search(r'(chữ|biển|bảng|logo|banner|áo in|số hiệu|mã|hiệu|mang dòng chữ|"|\')', raw_query.lower()))
        has_asr_signal = bool(re.search(r'(nói|phát biểu|hát|thuyết minh|giới thiệu|phỏng vấn|kể về|chia sẻ|bình luận)', raw_query.lower()))

        v_w = 0.50 if (has_ocr_signal or has_asr_signal) else 0.90
        o_w = 0.40 if has_ocr_signal else 0.05
        a_w = 0.40 if has_asr_signal else 0.05
        s = v_w + o_w + a_w

        return {
            "cleaned_vi": raw_query,
            "english_visual": raw_query,
            "visual_relevance": v_w / s,
            "ocr_relevance": o_w / s,
            "asr_relevance": a_w / s,
            "weights": {"visual": v_w / s, "ocr": o_w / s, "asr": a_w / s},
            "ocr_keywords": [w for w in raw_query.split() if len(w) > 3][:5] if has_ocr_signal else [],
            "asr_keywords": [w for w in raw_query.split() if len(w) > 3][:5] if has_asr_signal else [],
            "sub_events_en": [f"Event {i+1}: {e}" for i, e in enumerate(sub_events_vi)],
            "sub_events_vi": sub_events_vi if sub_events_vi else [raw_query]
        }

