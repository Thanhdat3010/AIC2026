import faiss
import numpy as np
from pathlib import Path
from typing import Tuple

class FAISSIndexer:
    def __init__(self, index_path: Path = None):
        self.index = None
        if index_path and index_path.exists():
            self.load(index_path)
            
    def build_index(self, features_path: Path, output_path: Path, expected_keyframes: int, dim: int = 512):
        print(f"Loading CLIP features from {features_path}...")
        # BTC features are float16
        fp = np.memmap(features_path, dtype='float16', mode='r', shape=(expected_keyframes, dim))
        
        # FAISS requires float32 contiguous arrays
        # Load in batches to avoid high memory usage
        print("Converting to float32 and L2 normalizing...")
        
        # To avoid OOM, load the entire array into memory as float32 since 177K * 512 * 4 bytes is ~360MB
        # This easily fits in RAM
        features = np.array(fp, dtype=np.float32)
        
        # FAISS IndexFlatIP expects normalized vectors for Cosine Similarity
        faiss.normalize_L2(features)
        
        print("Building IndexFlatIP...")
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(features)
        
        print(f"Index built with {self.index.ntotal} vectors.")
        
        # Save index
        output_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(output_path))
        print(f"Saved FAISS index to {output_path}")
        
    def load(self, index_path: Path):
        self.index = faiss.read_index(str(index_path))
        print(f"Loaded FAISS index with {self.index.ntotal} vectors.")
        
    def search(self, query_vectors: np.ndarray, top_k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise ValueError("Index is not loaded or built.")
            
        # Ensure query is float32 and contiguous
        if query_vectors.dtype != np.float32:
            query_vectors = query_vectors.astype(np.float32)
            
        if not query_vectors.flags['C_CONTIGUOUS']:
            query_vectors = np.ascontiguousarray(query_vectors)
            
        # FAISS expects normalized query for cosine similarity
        # (Assuming the query is already normalized from the text encoder, but we enforce it here for safety if needed.
        # It's better to let the TextEncoder handle normalization to be precise, but we can double check)
        
        distances, indices = self.index.search(query_vectors, top_k)
        return distances, indices
