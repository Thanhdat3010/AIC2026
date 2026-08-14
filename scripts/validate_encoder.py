import argparse
from pathlib import Path
import sys
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.query.text_encoder import CLIPTextEncoder, MultiLingualQueryEncoder
from src.query.translator import VietnameseTranslator
from src.config import settings
from src.indexing.faiss_indexer import FAISSIndexer

def main():
    print("=== Text Encoder Validation ===")
    
    translator = VietnameseTranslator(model_name=settings.models.translator)
    clip_encoder = CLIPTextEncoder(model_name=settings.models.text_encoder)
    
    encoder = MultiLingualQueryEncoder(
        encoder=clip_encoder,
        translator=translator,
        use_translation=settings.models.use_translation,
        fusion_alpha=settings.models.fusion_alpha
    )
    
    test_query = "một con chó màu vàng đang chạy trên cỏ"
    print(f"\nTest Query: {test_query}")
    
    if settings.models.use_translation:
        en_query = translator.translate(test_query)
        print(f"Translated: {en_query}")
        
    vec = encoder.encode(test_query)
    
    print(f"\nEncoded vector shape: {vec.shape}")
    print(f"Vector dtype: {vec.dtype}")
    
    norm = np.linalg.norm(vec, axis=1)
    print(f"Vector L2 Norm (should be ~1.0): {norm[0]:.4f}")
    
    if abs(norm[0] - 1.0) > 1e-4:
        print("[WARNING] Vector is not properly normalized!")
    else:
        print("[SUCCESS] Vector is properly L2 normalized.")
        
    # Test FAISS search if index exists
    index_path = settings.directories.indexes / "clip.faiss"
    if index_path.exists():
        print(f"\nLoading FAISS index to test search...")
        indexer = FAISSIndexer(index_path)
        
        distances, indices = indexer.search(vec, top_k=5)
        print("\nTop 5 results for test query:")
        for i in range(5):
            print(f"Rank {i+1} | Global ID: {indices[0][i]} | Cosine Sim: {distances[0][i]:.4f}")
            
        # The cosine similarity should typically be > 0.15 for a valid match in CLIP space
        if distances[0][0] > 0.15:
            print("\n[SUCCESS] Vector alignment looks valid (Cosine similarity > 0.15 for top match).")
        else:
            print("\n[WARNING] Cosine similarities are very low. The text encoder might not align well with BTC image features.")
            
    else:
        print("\n[WARNING] FAISS index not found. Skipping vector alignment test.")

if __name__ == "__main__":
    main()
