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

class APIKeyPool:
    """
    Hồ chứa và quản lý đa API Keys (Hỗ trợ cả Google Gemini 'AIza...' và Groq 'gsk_...'):
    - Tự động nhận diện loại Key (Gemini hay Groq).
    - Random / Round-robin đảo key giữa các request để phân tải.
    - Tự động chuyển key khác nếu 1 key bị Rate Limit (HTTP 429).
    """
    def __init__(self):
        self.gemini_keys: list[str] = []
        self.groq_keys: list[str] = []
        
        # 1. Đọc từ GEMINI_API_KEYS / GROQ_API_KEYS (dạng phẩy)
        raw_gemini = os.environ.get("GEMINI_API_KEYS", "")
        if raw_gemini:
            for k in raw_gemini.split(","):
                k_clean = k.strip("\"' \t\r\n")
                if k_clean and not k_clean.startswith("YOUR_"):
                    if k_clean.startswith("gsk_"):
                        self.groq_keys.append(k_clean)
                    else:
                        self.gemini_keys.append(k_clean)

        # 2. Đọc từ GEMINI_API_KEY_1..9 / GROQ_API_KEY_1..9
        for i in range(1, 10):
            k = os.environ.get(f"GEMINI_API_KEY_{i}", "").strip("\"' \t\r\n")
            if k and not k.startswith("YOUR_"):
                if k.startswith("gsk_") and k not in self.groq_keys:
                    self.groq_keys.append(k)
                elif not k.startswith("gsk_") and k not in self.gemini_keys:
                    self.gemini_keys.append(k)

            k_g = os.environ.get(f"GROQ_API_KEY_{i}", "").strip("\"' \t\r\n")
            if k_g and not k_g.startswith("YOUR_") and k_g not in self.groq_keys:
                self.groq_keys.append(k_g)

        # 3. Đọc từ GEMINI_API_KEY / GROQ_API_KEY đơn lẻ
        single_gem = os.environ.get("GEMINI_API_KEY", "").strip("\"' \t\r\n")
        if single_gem and not single_gem.startswith("YOUR_"):
            if single_gem.startswith("gsk_") and single_gem not in self.groq_keys:
                self.groq_keys.append(single_gem)
            elif not single_gem.startswith("gsk_") and single_gem not in self.gemini_keys:
                self.gemini_keys.append(single_gem)

        single_groq = os.environ.get("GROQ_API_KEY", "").strip("\"' \t\r\n")
        if single_groq and not single_groq.startswith("YOUR_") and single_groq not in self.groq_keys:
            self.groq_keys.append(single_groq)

        print(f"🔑 [API KEY POOL] Đã nạp {len(self.gemini_keys)} Gemini Keys & {len(self.groq_keys)} Groq Keys!", flush=True)

class UnifiedLLMRouter:
    """
    Bộ não điều hướng và làm giàu truy vấn:
    - Multi-Prompt Ensembling: Sinh 3 câu tiếng Anh với 3 góc nhìn thị giác độc lập.
    - OCR / ASR Keyword Extraction: Bóc tách từ khóa biển báo và lời thoại.
    - Dynamic Weighting: Cân đối trọng số tự động giữa Visual, OCR, và ASR.
    - Tự động ưu tiên Groq (siêu tốc 500 token/s) hoặc Gemini Flash.
    """
    def __init__(self):
        self.key_pool = APIKeyPool()

    def _call_groq(self, prompt: str) -> Optional[str]:
        if not self.key_pool.groq_keys:
            return None
        keys = list(self.key_pool.groq_keys)
        random.shuffle(keys)
        for key in keys:
            try:
                from groq import Groq
                client = Groq(api_key=key)
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a video retrieval query analyst. You MUST respond with ONLY a valid JSON object matching the requested schema without any markdown formatting."},
                        {"role": "user", "content": prompt}
                    ],
                    model="llama-3.3-70b-versatile",
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                return chat_completion.choices[0].message.content
            except Exception as e:
                print(f"⚠️ Groq Key (...{key[-6:]}) error: {e}, thử key kế tiếp...", flush=True)
                time.sleep(0.3)
        return None

    def _call_gemini(self, prompt: str) -> Optional[str]:
        if not self.key_pool.gemini_keys:
            return None
        keys = list(self.key_pool.gemini_keys)
        random.shuffle(keys)
        for key in keys:
            try:
                import google.generativeai as genai
                genai.configure(api_key=key)
                model = genai.GenerativeModel(
                    model_name="gemini-2.5-flash",
                    generation_config={"temperature": 0.2, "response_mime_type": "application/json"}
                )
                response = model.generate_content(prompt)
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                print(f"⚠️ Gemini Key (...{key[-6:]}) error: {e}, thử key kế tiếp...", flush=True)
                time.sleep(0.3)
        return None

    def transform_query(self, raw_query: str) -> dict:
        prompt = f"""
Analyze this Vietnamese video retrieval query for AIC 2026:
"{raw_query}"

Respond with ONLY a JSON object with this EXACT structure:
{{
  "visual_prompts": [
    "Sentence 1: Full descriptive scene caption in natural English",
    "Sentence 2: English caption focusing strictly on the Main Subjects and their precise Actions",
    "Sentence 3: English caption focusing on Salient Objects, Colors, Clothing, and Background Props"
  ],
  "ocr_keywords": ["Any text, road signs, banner titles, names, numbers that might appear on screen"],
  "asr_keywords": ["Any spoken words, dialogue phrases or voiceover quotes that might be heard"],
  "weights": {{
    "visual": 0.8,
    "ocr": 0.1,
    "asr": 0.1
  }},
  "temporal_hint": "early / middle / late / any"
}}
"""
        # Ưu tiên Groq nếu có keys (gsk_...) hoặc Gemini
        res_str = self._call_groq(prompt)
        if not res_str:
            res_str = self._call_gemini(prompt)

        if res_str:
            try:
                cleaned = res_str.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                return json.loads(cleaned.strip())
            except Exception as e:
                print(f"⚠️ JSON parsing error: {e}", flush=True)

        return {
            "visual_prompts": [raw_query, raw_query, raw_query],
            "ocr_keywords": [],
            "asr_keywords": [],
            "weights": {"visual": 1.0, "ocr": 0.0, "asr": 0.0},
            "temporal_hint": "any"
        }

if __name__ == "__main__":
    router = UnifiedLLMRouter()
    sample = "Trong một căn nhà, người phụ nữ dùng hai tay quấn và chỉnh tấm xà rông màu vàng cam quanh eo người đàn ông mặc áo xanh."
    print(f"\n🔎 [TEST QUERY]: {sample}")
    res = router.transform_query(sample)
    print("\n📊 [KẾT QUẢ PHÂN RÃ TỰ ĐỘNG]:")
    print(json.dumps(res, indent=2, ensure_ascii=False))
