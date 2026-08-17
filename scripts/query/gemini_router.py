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
    Hồ chứa và quản lý nhiều Gemini API Keys:
    - Random / Round-robin đảo key giữa các request để phân tải.
    - Tự động chuyển key khác nếu 1 key bị Rate Limit (HTTP 429).
    """
    def __init__(self):
        self.keys: list[str] = []
        
        # 1. Đọc từ GEMINI_API_KEYS (dạng phẩy)
        raw_keys = os.environ.get("GEMINI_API_KEYS", "")
        if raw_keys:
            self.keys.extend([k.strip("\"' \t\r\n") for k in raw_keys.split(",") if k.strip("\"' \t\r\n") and not k.strip("\"' \t\r\n").startswith("YOUR_")])

        # 2. Đọc từ GEMINI_API_KEY_1, GEMINI_API_KEY_2, GEMINI_API_KEY_3, ...
        for i in range(1, 10):
            k = os.environ.get(f"GEMINI_API_KEY_{i}", "").strip("\"' \t\r\n")
            if k and not k.startswith("YOUR_") and k not in self.keys:
                self.keys.append(k)

        # 3. Đọc từ GEMINI_API_KEY đơn lẻ
        single_k = os.environ.get("GEMINI_API_KEY", "").strip("\"' \t\r\n")
        if single_k and not single_k.startswith("YOUR_") and single_k not in self.keys:
            self.keys.append(single_k)

        self.current_idx = 0
        if self.keys:
            print(f"🔑 [GEMINI KEY POOL] Đã nạp thành công {len(self.keys)} API Keys hoạt động!", flush=True)
        else:
            print(f"⚠️ [GEMINI KEY POOL] Chưa tìm thấy API Key nào trong file .env!", flush=True)

    def get_random_key(self) -> Optional[str]:
        if not self.keys:
            return None
        return random.choice(self.keys)

    def get_next_key(self) -> Optional[str]:
        if not self.keys:
            return None
        key = self.keys[self.current_idx % len(self.keys)]
        self.current_idx += 1
        return key

class GeminiQueryRouter:
    """
    Bộ não điều hướng và làm giàu truy vấn sử dụng Gemini 3.5 Flash Lite:
    - Multi-Prompt Ensembling: Sinh 3 câu tiếng Anh với 3 góc nhìn thị giác độc lập.
    - OCR / ASR Keyword Extraction: Bóc tách từ khóa biển báo và lời thoại.
    - Dynamic Weighting: Cân đối trọng số tự động giữa Visual, OCR, và ASR.
    """
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self.key_pool = GeminiKeyPool()

    def _call_gemini_with_fallback(self, prompt: str) -> Optional[str]:
        """
        Gọi Gemini API với cơ chế tự động thử lần lượt các key trong KeyPool nếu gặp lỗi.
        """
        if not self.key_pool.keys:
            return None

        # Thử tối đa qua tất cả các key có sẵn
        keys_to_try = list(self.key_pool.keys)
        random.shuffle(keys_to_try)

        for key in keys_to_try:
            try:
                import google.generativeai as genai
                genai.configure(api_key=key)
                
                # Cấu hình model
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config={"temperature": 0.2, "response_mime_type": "application/json"}
                )
                response = model.generate_content(prompt)
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str or "exhausted" in err_str:
                    print(f"⚠️ Key kết thúc bằng ...{key[-6:]} bị quá tải hạn ngạch, tự động chuyển sang key khác...", flush=True)
                else:
                    print(f"⚠️ Lỗi gọi Gemini API (...{key[-6:]}): {e}, đang thử key kế tiếp...", flush=True)
                time.sleep(0.5)

        print("❌ Tất cả API Keys trong Key Pool đều không phản hồi.", flush=True)
        return None

    def transform_query(self, raw_query: str) -> dict:
        """
        Phân rã câu truy vấn tiếng Việt thành cấu trúc JSON đa phương thức chuẩn.
        """
        prompt = f"""
Bạn là chuyên gia phân tích truy vấn video cho cuộc thi Video Retrieval AIC 2026.
Hãy phân tích câu truy vấn sau đây từ Ban Tổ Chức:
"{raw_query}"

Hãy trả về một JSON object với đúng định dạng sau:
{{
  "visual_prompts": [
    "Câu 1: Bản dịch tiếng Anh chi tiết, văn phong mô tả toàn cảnh (Full descriptive scene caption)",
    "Câu 2: Bản dịch tiếng Anh tập trung vào Chủ thể và Hành động chính (Subjects and main action)",
    "Câu 3: Bản dịch tiếng Anh tập trung vào Màu sắc, Trang phục và Đồ vật nổi bật (Salient objects, colors, props)"
  ],
  "ocr_keywords": ["Danh sách các từ khóa có thể xuất hiện dưới dạng chữ viết, biển hiệu, tiêu đề trên màn hình (nếu có)"],
  "asr_keywords": ["Danh sách các từ ngữ hoặc câu thoại có thể phát thanh viên hoặc nhân vật nói (nếu có)"],
  "weights": {{
    "visual": 0.8,
    "ocr": 0.1,
    "asr": 0.1
  }},
  "temporal_hint": "early / middle / late / any"
}}

Lưu ý:
- Tổng trọng số visual + ocr + asr phải bằng 1.0.
- Nếu câu hỏi hỏi về lời nói, phỏng vấn, tin tức thời sự -> tăng weight ASR lên 0.4 - 0.6.
- Nếu câu hỏi có tên riêng, địa danh, biển hiệu, số -> tăng weight OCR lên 0.4 - 0.6.
- Nếu câu hỏi mô tả thị giác thuần túy -> weight visual = 0.8 - 0.9.
"""
        res_json_str = self._call_gemini_with_fallback(prompt)
        
        if res_json_str:
            try:
                # Làm sạch markdown json nếu có
                cleaned = res_json_str.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                parsed = json.loads(cleaned.strip())
                return parsed
            except Exception as e:
                print(f"⚠️ Lỗi phân tích JSON từ Gemini: {e}. Sử dụng fallback parser.", flush=True)

        # Fallback Parser nếu không có key hoặc API lỗi
        return {
            "visual_prompts": [raw_query, raw_query, raw_query],
            "ocr_keywords": [],
            "asr_keywords": [],
            "weights": {"visual": 1.0, "ocr": 0.0, "asr": 0.0},
            "temporal_hint": "any"
        }

if __name__ == "__main__":
    router = GeminiQueryRouter()
    sample = "Trong một căn nhà, người phụ nữ dùng hai tay quấn và chỉnh tấm xà rông màu vàng cam quanh eo người đàn ông mặc áo xanh."
    print(f"\n🔎 [TEST QUERY]: {sample}")
    res = router.transform_query(sample)
    print("\n📊 [KẾT QUẢ PHÂN RÃ TỪ GEMINI]:")
    print(json.dumps(res, indent=2, ensure_ascii=False))
