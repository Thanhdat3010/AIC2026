import os
import sys
import re
from pathlib import Path

# Force UTF-8 on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class ModalityGate:
    """
    Bộ Phân Loại & Điều Hướng Kích Hoạt Đa Phương Thức Thông Minh (Intelligent Modality Gate):
    - Tự động nhận diện xem câu hỏi có chứa yêu cầu OCR (chữ viết/biển báo) hay ASR (lời thoại/phỏng vấn) hay không.
    - Quy tắc vàng: Nếu là câu hỏi thuần thị giác -> KHÓA HOÀN TOÀN BM25 (W=0) để triệt tiêu 100% nhiễu.
    - Nếu có từ khóa cụ thể -> MỞ CỔNG BM25 và bóc tách từ khóa tìm kiếm chính xác.
    """
    def __init__(self):
        # Từ khóa kích hoạt OCR
        self.ocr_triggers = [
            r'"([^"]+)"',  # Chữ trong dấu ngoặc kép đôi: "Chúc mừng"
            r"'([^']+)'",  # Chữ trong dấu ngoặc kép đơn
            r'“([^”]+)”',  # Dấu ngoặc kép tiếng Việt
            r'\bbiển số\b',
            r'\bbảng hiệu\b',
            r'\bdòng chữ\b',
            r'\bchữ in\b',
            r'\bchữ viết\b',
            r'\btiêu đề\b',
            r'\blogo\b',
            r'\bbanner\b',
            r'\báp phích\b',
            r'\bsố áo\b',
            r'\btên đường\b'
        ]

        # Từ khóa kích hoạt ASR (Lời thoại / Âm thanh)
        self.asr_triggers = [
            r'\b(hỏi|nói|trả lời|phỏng vấn|lời thoại|thuyết minh|phát biểu|chia sẻ|cho biết)\b',
            r'\b(MC|người dẫn chương trình|phóng viên|nhân vật|chủ vườn)\s+(nói|hỏi|trả lời|kể|cho hay)\b'
        ]

    def analyze(self, query_text: str) -> dict:
        """
        Phân tích câu hỏi và trả về quyết định kích hoạt OCR / ASR.
        """
        q_lower = query_text.lower()

        # 1. Kiểm tra tín hiệu OCR
        has_ocr = False
        ocr_keywords = []

        # Trích xuất các chuỗi trong ngoặc kép
        quotes = re.findall(r'["\'“]([^"\'”]+)["\'”]', query_text)
        if quotes:
            has_ocr = True
            ocr_keywords.extend(quotes)

        for pat in self.ocr_triggers:
            if re.search(pat, q_lower):
                has_ocr = True
                break

        # 2. Kiểm tra tín hiệu ASR
        has_asr = False
        asr_keywords = []
        for pat in self.asr_triggers:
            if re.search(pat, q_lower):
                has_asr = True
                break

        if has_asr:
            # Lấy các từ khóa quan trọng của câu hỏi
            clean_q = re.sub(r'[^\w\s]', ' ', query_text)
            words = [w for w in clean_q.split() if len(w) > 1 and w.lower() not in ["trong", "khi", "người", "đang", "của", "và", "là", "cho", "vào"]]
            asr_keywords = words[:6]

        return {
            "has_ocr": has_ocr,
            "ocr_keywords": ocr_keywords,
            "has_asr": has_asr,
            "asr_keywords": asr_keywords,
            "is_pure_visual": (not has_ocr and not has_asr)
        }

if __name__ == "__main__":
    gate = ModalityGate()
    samples = [
        "Trong một căn nhà, người phụ nữ dùng hai tay quấn và chỉnh tấm xà rông màu vàng cam quanh eo người đàn ông mặc áo xanh.",
        "Chiếc xe tải có biển số '51F-123.45' chạy qua cầu.",
        "Trong cuộc trò chuyện dưới giàn nho, khi người dẫn chương trình hỏi mấy giờ bắt đầu làm việc, chủ vườn trả lời thế nào?"
    ]
    for s in samples:
        res = gate.analyze(s)
        print(f"\n🔎 Query: {s}")
        print(f"   -> OCR: {res['has_ocr']} {res['ocr_keywords']} | ASR: {res['has_asr']} {res['asr_keywords']} | Pure Visual: {res['is_pure_visual']}")
