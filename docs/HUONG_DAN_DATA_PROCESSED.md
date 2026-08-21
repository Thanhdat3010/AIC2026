# 📚 HƯỚNG DẪN SỬ DỤNG DỮ LIỆU ĐÃ TIỀN XỬ LÝ (PROCESSED DATASET GUIDE)
### 🏛️ Cuộc Thi AI Challenge 2026 (AIC 2026) — Nhóm Đồ Án / Đội Tuyển

---

## 🌟 1. TỔNG QUAN HỆ SINH THÁI DỮ LIỆU

Thư mục này (`data/batch_1/processed/`) chứa toàn bộ các ma trận đặc trưng AI, siêu dữ liệu (Metadata), kết quả nhận dạng văn bản (OCR) và nhận dạng giọng nói (ASR/Transcripts) đã được tiền xử lý và tối ưu hóa ở định dạng nhị phân tốc độ cao (**Parquet**, **Numpy NPY**, **JSON**).

### 📊 Thống Kê Quy Mô Dữ Liệu (Batch 1):
* **Số lượng Video:** `873 videos` (Định dạng từ `L20_V001` đến `L30_Vxxx`).
* **Tổng số Keyframes:** `177,321 frames`.
* **Tổng số dòng OCR trích xuất:** `177,605 đoạn văn bản trên màn hình`.
* **Tổng số đoạn lời thoại ASR:** `16,698 câu thuyết minh / phỏng vấn`.
* **Đặc trưng thị giác SigLIP 2:** `177,321 x 1152 chiều` (~408.5 MB).

---

## 📂 2. CẤU TRÚC VÀ CHI TIẾT CÁC FILE DỮ LIỆU

```
data/batch_1/processed/
├── siglip_features.npy         # Ma trận đặc trưng thị giác SigLIP 2 (177321, 1152)
├── clip_features.npy           # Ma trận đặc trưng CLIP ViT-B/32 dự phòng (177321, 512)
├── frames.parquet              # Bảng chỉ mục keyframe (video_id, frame_idx, pts_time, fps)
├── ocr_results.parquet         # Kết quả nhận dạng chữ trên màn hình (CRAFT + VietOCR)
├── transcripts.parquet         # Lời thoại âm thanh bóc băng (VinAI PhoWhisper-large)
├── video_ranges.parquet        # Phạm vi chỉ mục dòng (start_idx, end_idx) của từng video
├── videos.parquet              # Danh mục và metadata tổng thể 873 video
├── video_zip_map.json          # Bảng ánh xạ video_id -> tên file zip keyframes gốc
├── object_summary.parquet      # Bảng tổng hợp đối tượng phát hiện (Object Detection)
└── objects/                    # Dữ liệu bounding box chi tiết theo từng video
```

---

## 🔍 3. SCHEMA CHI TIẾT CỦA TỪNG BẢNG DỮ LIỆU

### 1. `siglip_features.npy` & `clip_features.npy`
* **Định dạng:** Binary Numpy Array (`float32` hoặc `float16`).
* **Kích thước shape:** `(177321, 1152)` đối với SigLIP 2 (`google/siglip2-so400m-patch14-384`).
* **Đặc tính:** Mỗi vector đã được **chuẩn hóa L2 Norm ($\|v\|_2 = 1$)**.
* **Cách dùng:** Tích vô hướng (Cosine Similarity) ma trận:
  $$S = Q \cdot V^T$$
  Tính toán độ tương đồng cho 177,321 frames chỉ mất **~3.5 ms** trên GPU CUDA / Tensor Core.

---

### 2. `frames.parquet` (Bảng Ánh Xạ Chỉ Mục Khung Hình)
Mapping giữa vị trí dòng `i` trong ma trận `siglip_features.npy` với thời gian thực tế của video:

| Tên Cột | Kiểu Dữ Liệu | Ý Nghĩa / Ví Dụ |
| :--- | :--- | :--- |
| `id` | `int64` | Thứ tự toàn cục `0, 1, 2, ..., 177320` tương ứng đúng dòng `i` của npy |
| `video_id` | `string` | Tên mã định danh video (VD: `"L28_V009"`, `"L26_V355"`) |
| `frame_idx` | `int64` | Số thứ tự khung hình trong video gốc (VD: `15866`, `4662`) |
| `pts_time` | `float64` | Thời gian tính bằng giây của keyframe (VD: `634.64` giây) |
| `fps` | `float64` | Tốc độ khung hình của video (VD: `25.0` fps) |

---

### 3. `ocr_results.parquet` (Dữ Liệu Chữ Trên Màn Hình)
Trích xuất từ mô hình SOTA **CRAFT Text Detector + VietOCR Transformer**:

| Tên Cột | Kiểu Dữ Liệu | Ý Nghĩa / Ví Dụ |
| :--- | :--- | :--- |
| `video_id` | `string` | Mã video (VD: `"L25_V044"`) |
| `frame_idx` | `int64` | Keyframe phát hiện thấy chữ |
| `text` | `string` | Văn bản gốc nhận dạng được (VD: `"Trao kinh phí hỗ trợ COVID-19"`) |
| `text_clean` | `string` | Văn bản đã chuẩn hóa Unicode NFC, loại ký tự rác |
| `confidence` | `float64` | Độ tin cậy nhận dạng (0.0 đến 1.0) |
| `bbox` | `list/str` | Tọa độ hộp bao $[x_1, y_1, x_2, y_2]$ của chữ trên khung hình |

---

### 4. `transcripts.parquet` (Dữ Liệu Lời Thoại / Thuyết Minh)
Bóc băng tự động bằng mô hình **VinAI PhoWhisper-large CTranslate2 FP16**:

| Tên Cột | Kiểu Dữ Liệu | Ý Nghĩa / Ví Dụ |
| :--- | :--- | :--- |
| `video_id` | `string` | Mã video (VD: `"L29_V013"`) |
| `start_time` | `float64` | Thời điểm bắt đầu câu thoại (giây) |
| `end_time` | `float64` | Thời điểm kết thúc câu thoại (giây) |
| `text` | `string` | Nội dung lời thoại tiếng Việt đầy đủ |
| `confidence` | `float64` | Xác suất log-probability của Whisper |

---

### 5. `video_zip_map.json` (Bảng Ánh Xạ Đọc Ảnh Không Cần Giải Nén)
Ánh xạ trực tiếp từ `video_id` sang file ZIP chứa ảnh keyframe, giúp hệ thống đọc ảnh tức thời (**< 5ms/ảnh**) bằng module `zipfile` mà không cần giải nén hàng chục GB ra ổ cứng:
```json
{
  "L20_V001": "keyframes/L20_Extra.zip",
  "L26_V355": "keyframes/L26.zip",
  "L28_V009": "keyframes/L28.zip"
}
```

---

## 💻 4. HƯỚNG DẪN TRUY XUẤT NHANH BẰNG PYTHON

### 🚀 Đoạn mã mẫu 1: Đọc nhanh bảng chỉ mục & Memory-map Vector
```python
import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/batch_1/processed")

# 1. Đọc nhanh bảng khung hình (chỉ mất ~50ms với Parquet)
df_frames = pd.read_parquet(PROCESSED_DIR / "frames.parquet")
print(f"Tổng số frames: {len(df_frames):,}")

# 2. Memory-map đặc trưng SigLIP 2 (Không tốn RAM, load tức thì)
features = np.load(PROCESSED_DIR / "siglip_features.npy", mmap_mode="r")
print(f"Shape đặc trưng: {features.shape}, Dtype: {features.dtype}")
```

---

### 🖼️ Đoạn mã mẫu 2: Đọc trực tiếp Keyframe từ file ZIP
```python
import json
import zipfile
from PIL import Image
import io
from pathlib import Path

BASE_DIR = Path(".")
PROCESSED_DIR = BASE_DIR / "data/batch_1/processed"

with open(PROCESSED_DIR / "video_zip_map.json", "r", encoding="utf-8") as f:
    zip_map = json.load(f)

def load_keyframe_image(video_id: str, frame_idx: int) -> Image.Image:
    zip_rel_path = zip_map.get(video_id)
    if not zip_rel_path:
        raise FileNotFoundError(f"Không tìm thấy file zip cho video {video_id}")
    
    zip_path = BASE_DIR / "data" / zip_rel_path
    frame_name = f"{frame_idx:06d}.jpg" # hoặc định dạng chuẩn BTC
    
    with zipfile.ZipFile(zip_path, "r") as z:
        # Tìm đường dẫn file ảnh bên trong zip
        matching = [name for name in z.namelist() if name.endswith(frame_name) and video_id in name]
        if matching:
            img_bytes = z.read(matching[0])
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    raise FileNotFoundError(f"Không tìm thấy frame {frame_idx} trong {zip_path}")
```

---

### 🔎 Đoạn mã mẫu 3: Tìm kiếm BM25 OCR & ASR cho câu truy vấn
```python
from src.indexing.bm25_indexer import BM25Indexer

# Khởi tạo bộ chỉ mục văn bản đa phương thức
indexer = BM25Indexer(processed_dir=Path("data/batch_1/processed"))

# Tìm kiếm thực thể tên riêng trên OCR (Màn hình)
ocr_results = indexer.search_ocr("Trao kinh phí hỗ trợ COVID-19", top_k=5)
for hit in ocr_results:
    print(f"OCR Match: Video {hit['video_id']}, Frame {hit['frame_idx']}, Score: {hit['score']:.4f}, Text: {hit['text']}")

# Tìm kiếm trên Lời thoại thuyết minh (ASR)
asr_results = indexer.search_asr("Thành phố Lausanne Thụy Sĩ", top_k=5)
for hit in asr_results:
    print(f"ASR Match: Video {hit['video_id']}, Time: {hit['start_time']}s, Score: {hit['score']:.4f}")
```

---

## 🛠️ 5. QUY TRÌNH TẠO DỮ LIỆU KHI CÓ BATCH MỚI (BATCH 2 / CHUNG KẾT)

Khi Ban Tổ Chức (BTC) cung cấp thêm dữ liệu mới (`data/batch_2/`):

1. **Bước 1: Trích xuất đặc trưng SigLIP 2:**
   ```bash
   python scripts/indexing/extract_siglip_features.py --batch batch_2
   ```
2. **Bước 2: Bóc băng âm thanh PhoWhisper & trích xuất OCR:**
   ```bash
   python scripts/indexing/extract_ocr_whisper.py --batch batch_2
   ```
3. **Bước 3: Xây dựng chỉ mục tìm kiếm văn bản BM25:**
   ```bash
   python -c "from src.indexing.bm25_indexer import BM25Indexer; BM25Indexer().build_and_save()"
   ```

---
*Tài liệu được biên soạn tự động phục vụ cuộc thi AI Challenge 2026.*
