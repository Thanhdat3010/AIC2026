import os
import sys
import re
import pickle
import time
from pathlib import Path
import pandas as pd
from rank_bm25 import BM25Okapi

# Force UTF-8 on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

def tokenize_text(text: str) -> list[str]:
    """
    Tách từ đơn giản và nhanh cho tiếng Việt và tiếng Anh (lowercase, tách từ).
    """
    if not isinstance(text, str) or not text.strip():
        return []
    # Giữ lại các ký tự chữ cái và số
    cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
    return [token for token in cleaned.split() if len(token) > 0]

class BM25MultiIndexer:
    """
    Hệ thống đánh chỉ mục Sparse Keyword Retrieval (BM25) cho:
    1. OCR (Chữ trên khung hình - ocr_results.parquet)
    2. ASR (Lời thoại phát thanh - transcripts.parquet)
    3. Metadata (Tiêu đề, mô tả video - videos.parquet)
    """
    def __init__(self, batch: str = "batch_1", base_dir: Path = None):
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent.parent.parent

        self.processed_dir = base_dir / "data" / batch / "processed"
        self.indexes_dir = base_dir / "indexes" / batch
        self.indexes_dir.mkdir(parents=True, exist_ok=True)

        self.ocr_bm25_path = self.indexes_dir / "bm25_ocr.pkl"
        self.asr_bm25_path = self.indexes_dir / "bm25_asr.pkl"
        self.meta_bm25_path = self.indexes_dir / "bm25_meta.pkl"

        self.ocr_index = None
        self.ocr_docs = []
        self.asr_index = None
        self.asr_docs = []
        self.meta_index = None
        self.meta_docs = []

    def build_all(self, force_rebuild: bool = False):
        t0 = time.time()
        print("=" * 70, flush=True)
        print("📦 BẮT ĐẦU XÂY DỰNG TOÀN BỘ CHỈ MỤC BM25 (OCR, ASR, METADATA)", flush=True)
        print("=" * 70, flush=True)

        # 1. Build BM25 OCR
        self.build_ocr_index(force_rebuild)

        # 2. Build BM25 ASR
        self.build_asr_index(force_rebuild)

        # 3. Build BM25 Metadata
        self.build_meta_index(force_rebuild)

        print(f"🎉 [HOÀN TẤT] Toàn bộ BM25 Index đã sẵn sàng sau {time.time() - t0:.2f} giây!\n", flush=True)

    def build_ocr_index(self, force_rebuild: bool = False):
        if self.ocr_bm25_path.exists() and not force_rebuild:
            print(f"[*] Đang nạp BM25 OCR từ cache: {self.ocr_bm25_path.name}...", flush=True)
            with open(self.ocr_bm25_path, "rb") as f:
                data = pickle.load(f)
                self.ocr_index = data["index"]
                self.ocr_docs = data["docs"]
            print(f"✅ Đã nạp BM25 OCR: {len(self.ocr_docs):,} documents", flush=True)
            return

        ocr_file = self.processed_dir / "ocr_results.parquet"
        if not ocr_file.exists():
            print(f"⚠️ Không tìm thấy {ocr_file.name}, bỏ qua BM25 OCR", flush=True)
            return

        print(f"[*] Đang xử lý {ocr_file.name} để tạo BM25 OCR...", flush=True)
        df_ocr = pd.read_parquet(ocr_file)
        
        tokenized_corpus = []
        docs = []
        for _, row in df_ocr.iterrows():
            text = str(row.get("ocr_text", "") or "")
            tokens = tokenize_text(text)
            tokenized_corpus.append(tokens)
            docs.append({
                "video_id": row["video_id"],
                "frame_idx": int(row["frame_idx"]),
                "pts_time": float(row.get("pts_time", 0.0)),
                "text": text
            })

        print(f"[*] Đang khởi tạo BM25Okapi trên {len(docs):,} OCR documents...", flush=True)
        self.ocr_index = BM25Okapi(tokenized_corpus)
        self.ocr_docs = docs

        with open(self.ocr_bm25_path, "wb") as f:
            pickle.dump({"index": self.ocr_index, "docs": self.ocr_docs}, f)
        print(f"✅ Đã lưu BM25 OCR vào: {self.ocr_bm25_path}", flush=True)

    def build_asr_index(self, force_rebuild: bool = False):
        if self.asr_bm25_path.exists() and not force_rebuild:
            print(f"[*] Đang nạp BM25 ASR từ cache: {self.asr_bm25_path.name}...", flush=True)
            with open(self.asr_bm25_path, "rb") as f:
                data = pickle.load(f)
                self.asr_index = data["index"]
                self.asr_docs = data["docs"]
            print(f"✅ Đã nạp BM25 ASR: {len(self.asr_docs):,} documents", flush=True)
            return

        asr_file = self.processed_dir / "transcripts.parquet"
        if not asr_file.exists():
            print(f"⚠️ Không tìm thấy {asr_file.name}, bỏ qua BM25 ASR", flush=True)
            return

        print(f"[*] Đang xử lý {asr_file.name} để tạo BM25 ASR...", flush=True)
        df_asr = pd.read_parquet(asr_file)
        
        tokenized_corpus = []
        docs = []
        for _, row in df_asr.iterrows():
            text = str(row.get("transcript", "") or "")
            tokens = tokenize_text(text)
            tokenized_corpus.append(tokens)
            docs.append({
                "video_id": row["video_id"],
                "start_frame": int(row.get("start_frame", 0)),
                "end_frame": int(row.get("end_frame", 0)),
                "start_time": float(row.get("start_time", 0.0)),
                "end_time": float(row.get("end_time", 0.0)),
                "text": text
            })

        print(f"[*] Đang khởi tạo BM25Okapi trên {len(docs):,} ASR sentences...", flush=True)
        self.asr_index = BM25Okapi(tokenized_corpus)
        self.asr_docs = docs

        with open(self.asr_bm25_path, "wb") as f:
            pickle.dump({"index": self.asr_index, "docs": self.asr_docs}, f)
        print(f"✅ Đã lưu BM25 ASR vào: {self.asr_bm25_path}", flush=True)

    def build_meta_index(self, force_rebuild: bool = False):
        if self.meta_bm25_path.exists() and not force_rebuild:
            print(f"[*] Đang nạp BM25 Metadata từ cache: {self.meta_bm25_path.name}...", flush=True)
            with open(self.meta_bm25_path, "rb") as f:
                data = pickle.load(f)
                self.meta_index = data["index"]
                self.meta_docs = data["docs"]
            print(f"✅ Đã nạp BM25 Metadata: {len(self.meta_docs):,} documents", flush=True)
            return

        meta_file = self.processed_dir / "videos.parquet"
        if not meta_file.exists():
            print(f"⚠️ Không tìm thấy {meta_file.name}, bỏ qua BM25 Metadata", flush=True)
            return

        print(f"[*] Đang xử lý {meta_file.name} để tạo BM25 Metadata...", flush=True)
        df_meta = pd.read_parquet(meta_file)
        
        tokenized_corpus = []
        docs = []
        for _, row in df_meta.iterrows():
            title = str(row.get("title", "") or "")
            desc = str(row.get("description", "") or "")
            full_text = f"{title} {desc}"
            tokens = tokenize_text(full_text)
            tokenized_corpus.append(tokens)
            docs.append({
                "video_id": row["video_id"],
                "title": title,
                "description": desc
            })

        print(f"[*] Đang khởi tạo BM25Okapi trên {len(docs):,} Video Metadatas...", flush=True)
        self.meta_index = BM25Okapi(tokenized_corpus)
        self.meta_docs = docs

        with open(self.meta_bm25_path, "wb") as f:
            pickle.dump({"index": self.meta_index, "docs": self.meta_docs}, f)
        print(f"✅ Đã lưu BM25 Metadata vào: {self.meta_bm25_path}", flush=True)

    def search_ocr(self, query_text: str, top_k: int = 100) -> list[dict]:
        if self.ocr_index is None:
            self.build_ocr_index()
        tokens = tokenize_text(query_text)
        if not tokens:
            return []
        scores = self.ocr_index.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            if scores[idx] <= 0:
                break
            doc = self.ocr_docs[idx].copy()
            doc["rank"] = rank
            doc["score"] = float(scores[idx])
            results.append(doc)
        return results

    def search_asr(self, query_text: str, top_k: int = 100) -> list[dict]:
        if self.asr_index is None:
            self.build_asr_index()
        tokens = tokenize_text(query_text)
        if not tokens:
            return []
        scores = self.asr_index.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            if scores[idx] <= 0:
                break
            doc = self.asr_docs[idx].copy()
            doc["rank"] = rank
            doc["score"] = float(scores[idx])
            # Chọn frame đại diện ở giữa đoạn thoại
            mid_frame = (doc["start_frame"] + doc["end_frame"]) // 2
            doc["frame_idx"] = mid_frame
            results.append(doc)
        return results

if __name__ == "__main__":
    indexer = BM25MultiIndexer()
    indexer.build_all(force_rebuild=True)

    # Test OCR Search
    print("\n🔍 Test Tìm Kiếm OCR (Ví dụ: 'Cần Thơ'):")
    res_ocr = indexer.search_ocr("Cần Thơ", top_k=3)
    for r in res_ocr:
        print(f"   + [Score: {r['score']:.2f}] Video: {r['video_id']} | Frame: {r['frame_idx']} | Text: '{r['text']}'")

    # Test ASR Search
    print("\n🔍 Test Tìm Kiếm ASR (Ví dụ: 'thời tiết'):")
    res_asr = indexer.search_asr("thời tiết", top_k=3)
    for r in res_asr:
        print(f"   + [Score: {r['score']:.2f}] Video: {r['video_id']} | Frames: [{r['start_frame']}-{r['end_frame']}] | Text: '{r['text']}'")
