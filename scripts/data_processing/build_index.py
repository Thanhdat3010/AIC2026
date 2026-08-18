import argparse
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.indexing.faiss_indexer import FAISSIndexer
from src.config import settings

def main():
    parser = argparse.ArgumentParser(description="Build FAISS Index for AIC 2026 KIS")
    parser.add_argument("--features", type=str, default=str(settings.directories.processed / "clip_features.npy"))
    parser.add_argument("--out", type=str, default=str(settings.directories.indexes / "clip.faiss"))
    
    parser.add_argument("--dim", type=int, default=None, help="Embedding dimension (e.g. 512, 1152). If None, auto-detected.")
    
    args = parser.parse_args()
    
    features_path = Path(args.features)
    out_path = Path(args.out)
    
    if not features_path.exists():
        print(f"[ERROR] Features file {features_path} does not exist.")
        sys.exit(1)
        
    expected_keyframes = settings.data.expected_keyframes
    if args.dim is not None:
        dim = args.dim
    else:
        # Auto-detect dim: filesize in bytes / (expected_keyframes * 2 bytes per float16)
        filesize = features_path.stat().st_size
        detected_dim = filesize // (expected_keyframes * 2)
        dim = detected_dim if detected_dim > 0 else settings.data.clip_dim
        print(f"ℹ️ Auto-detected embedding dimension: {dim} from {features_path.name}")
        
    indexer = FAISSIndexer()
    
    try:
        indexer.build_index(
            features_path=features_path,
            output_path=out_path,
            expected_keyframes=expected_keyframes,
            dim=dim
        )
    except Exception as e:
        print(f"[ERROR] Failed to build index: {e}")
        sys.exit(1)
        
    print("\n[SUCCESS] Vector Indexing completed successfully!")

if __name__ == "__main__":
    main()
