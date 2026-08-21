import os
import sys
import json
import random
import time
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Force UTF-8 on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Nạp file .env từ thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

class GeminiKeyPool:
    """
    Hồ chứa và quản lý đa Google Gemini API Keys:
    - Random / Round-robin đảo key giữa các request để phân tải.
    - Tự động chuyển key khác nếu 1 key bị Rate Limit (HTTP 429).
    """
    def __init__(self):
        self.gemini_keys: list[str] = []
        
        # 1. Đọc từ GEMINI_API_KEYS (dạng phẩy)
        raw_gemini = os.environ.get("GEMINI_API_KEYS", "")
        if raw_gemini:
            for k in raw_gemini.split(","):
                k_clean = k.strip("\"' \t\r\n")
                if k_clean and not k_clean.startswith("YOUR_"):
                    self.gemini_keys.append(k_clean)

        # 2. Đọc từ GEMINI_API_KEY_1..9
        for i in range(1, 10):
            k = os.environ.get(f"GEMINI_API_KEY_{i}", "").strip("\"' \t\r\n")
            if k and not k.startswith("YOUR_") and k not in self.gemini_keys:
                self.gemini_keys.append(k)

        # 3. Đọc từ GEMINI_API_KEY đơn lẻ
        single_gem = os.environ.get("GEMINI_API_KEY", "").strip("\"' \t\r\n")
        if single_gem and not single_gem.startswith("YOUR_") and single_gem not in self.gemini_keys:
            self.gemini_keys.append(single_gem)

        self._curr_idx = 0
        print(f"🔑 [GEMINI KEY POOL] Đã nạp thành công {len(self.gemini_keys)} Google Gemini API Keys hoạt động!", flush=True)

    def get_next_key(self) -> str:
        """Lấy key kế tiếp theo vòng tròn Round-Robin."""
        if not self.gemini_keys:
            return ""
        key = self.gemini_keys[self._curr_idx % len(self.gemini_keys)]
        self._curr_idx += 1
        return key

    def get_random_key(self) -> str:
        if not self.gemini_keys:
            return ""
        return random.choice(self.gemini_keys)

class GeminiQueryRouter:
    """
    Bộ não điều hướng và làm giàu truy vấn thuần Google Gemini 3.5 Flash Lite:
    - Dominant Multi-Prompt Ensembling: Sinh 3 câu tiếng Anh (Chính 70%, Hành động 15%, Chi tiết 15%).
    - Adaptive Modality Gating: Nhận diện chính xác có tín hiệu OCR/ASR hay không để khóa/mở cổng BM25.
    - Task Identification: Tự động gắn cờ is_qa hoặc is_trake.
    """
    def __init__(self, model_name: str = "gemini-3.5-flash-lite"):
        self.model_name = model_name
        self.key_pool = GeminiKeyPool()

    def _call_gemini(self, prompt: str) -> Optional[str]:
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
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str:
                    print(f"⚠️ Gemini Key (...{key[-6:]}) bị chạm hạn ngạch, chuyển key khác...", flush=True)
                else:
                    print(f"⚠️ Gemini Key (...{key[-6:]}) error: {e}, thử key kế tiếp...", flush=True)
                time.sleep(0.2)
        return None

    def transform_query(self, raw_query: str) -> dict:
        prompt = f"""
You are an expert Multimodal Video Retrieval AI System for AI Challenge 2026.
Analyze this Vietnamese query:
"{raw_query}"

Rules for Semantic Generalization, Entity Expansion & Modality Gating:
1. has_ocr_signal: set to TRUE ONLY if the query explicitly mentions reading written text, signs, banners, titles, text in quotes, awards, numbers, license plates, or specific proper nouns / acronyms (e.g. "COVID-19", "Lausanne", "FANA"). If the query is purely about visual actions/people/objects, set to FALSE.
   - If has_ocr_signal is true: provide `ocr_keywords` containing the normalized Vietnamese text entity AND likely OCR typo / diacritics variants (e.g. ["COVID-19", "covid 19", "covid19"], ["Lausanne", "Đại học Lausanne", "EPFL"], ["FANA", "CLB FANA", "FA NA"]).
2. has_asr_signal: set to TRUE if the query mentions spoken dialogue, interview speech, poems, songs, voice announcements, OR mentions specific proper nouns, acronyms, club/organization/brand/event names (e.g. "FANA", "CLB FANA", "Lausanne", "Nguyễn Trung Trực") because news reportage voiceover narrations frequently speak these entity names aloud.
   - If has_asr_signal is true: provide `asr_keywords` containing normalized spoken phrases, proper noun keywords, AND phonetic variants (e.g. ["fana", "clb fana", "câu lạc bộ fana"], ["Lô-xan", "Lô san", "lausanne"], ["Hỏa hồng Nhự Tảo oanh thiên địa"]).
3. is_qa: set to TRUE if the query is a Question asking for specific entity/action/color/count/time/name.
4. is_trake: set to TRUE if the query describes a chronological sequence of multiple distinct consecutive actions (First... then... then...).
5. If is_trake is true: break down the chronological actions into granular atomic sub-steps in `trake_events` (in concise natural English).
   - CRITICAL: Ensure EVERY SINGLE atomic action is separated, even if they appear in the same sentence (e.g. "thêm đậu Hà Lan rồi cà rốt" MUST be split into two separate events: "add peas" and "add carrots"). Do NOT group actions!

CRITICAL RULE FOR VISUAL PROMPTS:
- Do NOT include prefixes like "Sentence 1:", "Scene 1:", or "-". Just output the raw text.
- visual_prompts[0] MUST BE a "Comprehensive Visual Prompt": A highly detailed and rich English description of the scene. Include the main subjects, actions, specific colors, clothing, background props, and spatial context (e.g. "on the left", "in the background"). Contrastive models like SigLIP thrive on long, dense captions. Do NOT use conversational fillers like "The image shows...".
- visual_prompts[1] MUST BE "Action Focus": A short description isolating only the dynamic actions occurring.
- visual_prompts[2] MUST BE "Entity Focus": A short list of the key objects/people present.

Respond with ONLY a JSON object with this EXACT structure:
{{
  "visual_prompts": [
    "Comprehensive detailed English description of the entire scene including colors and spatial relations",
    "Short English description of actions",
    "Short English list of key entities"
  ],
  "has_ocr_signal": true or false,
  "ocr_keywords": ["normalized entity keywords", "typo/abbreviation variants"],
  "has_asr_signal": true or false,
  "asr_keywords": ["spoken dialogue phrases", "phonetic variants"],
  "is_qa": true or false,
  "is_trake": true or false,
  "trake_events": ["Short English event 1", "Short English event 2"]
}}
"""
        res_str = self._call_gemini(prompt)

        if res_str:
            try:
                cleaned = res_str.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                parsed = json.loads(cleaned.strip())

                if "weights" not in parsed:
                    parsed["weights"] = {"visual": 1.0, "ocr": 0.0, "asr": 0.0}
                # Áp dụng Adaptive Modality Gating an toàn tuyệt đối
                if not parsed.get("has_ocr_signal", False):
                    parsed["ocr_keywords"] = []
                    parsed["weights"]["ocr"] = 0.0
                else:
                    parsed["weights"]["ocr"] = 1.5

                if not parsed.get("has_asr_signal", False):
                    parsed["asr_keywords"] = []
                    parsed["weights"]["asr"] = 0.0
                else:
                    parsed["weights"]["asr"] = 1.2

                return parsed
            except Exception as e:
                print(f"⚠️ JSON parsing error: {e}", flush=True)

        return {
            "visual_prompts": [raw_query, raw_query, raw_query],
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

    def get_qa_modality(self, raw_query: str) -> str:
        """Phân loại độc lập câu hỏi QA thành 4 nhóm để phục vụ Adaptive Evidence."""
        prompt = f"""Phân loại câu hỏi sau đây vào đúng 1 trong 4 nhóm:
Câu hỏi: "{raw_query}"

Các nhóm:
- "count": Nếu câu hỏi yêu cầu đếm số lượng (ví dụ: có bao nhiêu...).
- "ocr": Nếu câu hỏi yêu cầu đọc chữ, số, biển báo, tên riêng trên màn hình.
- "asr": Nếu câu hỏi hỏi về nội dung lời nói, hội thoại.
- "visual": Các câu hỏi thị giác thông thường (hành động, màu sắc, đồ vật).

Trả về ĐÚNG 1 TỪ duy nhất: count, ocr, asr, hoặc visual."""
        res_str = self._call_gemini(prompt)
        if res_str:
            res = res_str.strip().lower()
            if "count" in res: return "count"
            if "ocr" in res: return "ocr"
            if "asr" in res: return "asr"
        return "visual"

if __name__ == "__main__":
    router = GeminiQueryRouter()
    sample = "Khi 2 người đàn ông đang di chuyển chiếc xe máy chở nhiều măng le, người phía trước đội gì trên đầu?"
    print(f"\n🔎 [TEST QUERY]: {sample}")
    res = router.transform_query(sample)
    print("\n📊 [KẾT QUẢ PHÂN RÃ TỰ ĐỘNG TỪ GEMINI 3.5 FLASH LITE]:")
    print(json.dumps(res, indent=2, ensure_ascii=False))
