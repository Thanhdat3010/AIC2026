import argparse
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.indexing.faiss_indexer import FAISSIndexer
from src.config import settings

def main():
    parser = argparse.ArgumentParser(description="Build FAISS Index for AIC 2026 KIS")
    parser.add_argument("--features", type=str, default=str(settings.directories.processed / "clip_features.npy"))
    parser.add_argument("--out", type=str, default=str(settings.directories.indexes / "clip.faiss"))
    
    args = parser.parse_args()
    
    features_path = Path(args.features)
    out_path = Path(args.out)
    
    if not features_path.exists():
        print(f"[ERROR] Features file {features_path} does not exist. Did you run preprocess_all.py?")
        sys.exit(1)
        
    indexer = FAISSIndexer()
    
    try:
        indexer.build_index(
            features_path=features_path,
            output_path=out_path,
            expected_keyframes=settings.data.expected_keyframes,
            dim=settings.data.clip_dim
        )
    except Exception as e:
        print(f"[ERROR] Failed to build index: {e}")
        sys.exit(1)
        
    print("\n[SUCCESS] Vector Indexing completed successfully!")

if __name__ == "__main__":
    main()
