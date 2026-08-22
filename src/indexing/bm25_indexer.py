import os
import sys
import re
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

# Force UTF-8 on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import unicodedata

def remove_vietnamese_accents(text: str) -> str:
    """Chuyển đổi chuỗi tiếng Việt có dấu thành không dấu nhanh và chính xác."""
    text = unicodedata.normalize('NFD', text)
    text = re.sub(r'[\u0300-\u036f]', '', text)
    text = text.replace('đ', 'd').replace('Đ', 'd')
    return unicodedata.normalize('NFC', text)

def tokenize_text(text: str, include_unaccented: bool = True) -> list[str]:
    """
    Chuẩn hóa và tách từ nâng cao cho OCR / ASR theo SOTA VBS/TRECVID:
    1. Chuẩn hóa Unicode NFC và lowercase.
    2. Tách từ ghép, mã hiệu đặc biệt (VD: COVID-19 -> ['covid-19', 'covid19', 'covid', '19']).
    3. Lọc bỏ ký tự rác 1 chữ cái (nhiễu texture OCR).
    4. Sinh song song tokens không dấu để chống lỗi OCR mất dấu (Dual Inverted Indexing).
    """
    if not isinstance(text, str) or not text.strip():
        return []
    
    # 1. NFC Normalization & Lowercase
    text = unicodedata.normalize('NFC', text).lower()
    
    # 2. Bóc tách các mã hiệu đặc biệt (như covid-19, fana-clb, q1-2026)
    special_tokens = set()
    hyphen_matches = re.findall(r'[a-z0-9]+-[a-z0-9]+', text)
    for m in hyphen_matches:
        special_tokens.add(m)
        special_tokens.add(m.replace('-', ''))

    # 3. Làm sạch ký tự đặc biệt, chỉ giữ chữ cái và số
    cleaned = re.sub(r'[^\w\s]', ' ', text)
    raw_tokens = [t for t in cleaned.split() if len(t) > 0]
    
    tokens = set()
    for t in raw_tokens:
        # Lọc bỏ ký tự rác 1 chữ cái (trừ khi là chữ số 0-9)
        if len(t) == 1 and not t.isdigit() and t not in ('a', 'y', 'ở', 'ổ', 'ố', 'á', 'à'):
            continue
        tokens.add(t)
        
        # 4. Thêm dạng không dấu nếu cần (Dual Inverted Indexing)
        if include_unaccented:
            unacc = remove_vietnamese_accents(t)
            if unacc != t and len(unacc) > 1:
                tokens.add(unacc)
                
    tokens.update(special_tokens)
    return list(tokens)

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
        
        texts = df_ocr["ocr_text"].fillna("").astype(str).tolist()
        vids = df_ocr["video_id"].tolist()
        fidxs = df_ocr["frame_idx"].tolist()
        pts = df_ocr["pts_time"].fillna(0.0).tolist() if "pts_time" in df_ocr.columns else [0.0] * len(texts)

        tokenized_corpus = [tokenize_text(t) for t in texts]
        docs = [{"video_id": v, "frame_idx": int(f), "pts_time": float(p), "text": t} for v, f, p, t in zip(vids, fidxs, pts, texts)]

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
        
        texts = df_asr["transcript"].fillna("").astype(str).tolist()
        vids = df_asr["video_id"].tolist()
        sf = df_asr["start_frame"].fillna(0).tolist() if "start_frame" in df_asr.columns else [0] * len(texts)
        ef = df_asr["end_frame"].fillna(0).tolist() if "end_frame" in df_asr.columns else [0] * len(texts)
        st = df_asr["start_time"].fillna(0.0).tolist() if "start_time" in df_asr.columns else [0.0] * len(texts)
        et = df_asr["end_time"].fillna(0.0).tolist() if "end_time" in df_asr.columns else [0.0] * len(texts)

        tokenized_corpus = [tokenize_text(t) for t in texts]
        docs = [{"video_id": v, "start_frame": int(s), "end_frame": int(e), "start_time": float(stt), "end_time": float(ett), "text": t} 
                for v, s, e, stt, ett, t in zip(vids, sf, ef, st, et, texts)]

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
        scores = np.asarray(self.ocr_index.get_scores(tokens), dtype=np.float32)
        if len(scores) == 0 or np.max(scores) <= 0:
            return []
        k = min(top_k, len(scores))
        top_indices = np.argpartition(-scores, k)[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            sc = float(scores[idx])
            if sc <= 0:
                break
            doc = self.ocr_docs[idx].copy()
            doc["rank"] = rank
            doc["score"] = sc
            results.append(doc)
        return results

    def search_asr(self, query_text: str, top_k: int = 100) -> list[dict]:
        if self.asr_index is None:
            self.build_asr_index()
        tokens = tokenize_text(query_text)
        if not tokens:
            return []
        scores = np.asarray(self.asr_index.get_scores(tokens), dtype=np.float32)
        if len(scores) == 0 or np.max(scores) <= 0:
            return []
        k = min(top_k, len(scores))
        top_indices = np.argpartition(-scores, k)[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            sc = float(scores[idx])
            if sc <= 0:
                break
            doc = self.asr_docs[idx].copy()
            doc["rank"] = rank
            doc["score"] = sc
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
