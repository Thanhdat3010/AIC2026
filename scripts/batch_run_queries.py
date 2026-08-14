import argparse
import sys
import time
from pathlib import Path
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import settings
from src.query.translator import VietnameseTranslator
from src.query.text_encoder import CLIPTextEncoder, MultiLingualQueryEncoder
from src.query.query_decomposer import QueryDecomposer
from src.retrieval.faiss_retriever import FAISSRetriever
from src.retrieval.multi_cue_retriever import MultiCueRetriever
from src.retrieval.video_aggregator import VideoAggregator
from src.reranking.metadata_reranker import MetadataReranker
from src.reranking.fusion import FusionEngine
from src.submission.writer import SubmissionWriter

def main():
    parser = argparse.ArgumentParser(description="Batch Run KIS Queries")
    parser.add_argument("--query_dir", type=str, required=True, help="Path to the directory containing query text files")
    args = parser.parse_args()
    
    query_dir = Path(args.query_dir)
    if not query_dir.exists():
        print(f"[ERROR] Query directory not found: {query_dir}")
        sys.exit(1)
        
    print("=== Initializing KIS Pipeline ===")
    
    translator = VietnameseTranslator(model_name=settings.models.translator)
    clip_encoder = CLIPTextEncoder(model_name=settings.models.text_encoder)
    encoder = MultiLingualQueryEncoder(
        encoder=clip_encoder,
        translator=translator,
        use_translation=settings.models.use_translation,
        fusion_alpha=settings.models.fusion_alpha
    )
    
    decomposer = QueryDecomposer()
    
    faiss_path = settings.directories.indexes / "clip.faiss"
    frames_path = settings.directories.processed / "frames.parquet"
    retriever = FAISSRetriever(faiss_path, frames_path)
    
    multi_retriever = MultiCueRetriever(decomposer, encoder, retriever)
    
    videos_path = settings.directories.processed / "videos.parquet"
    metadata_reranker = MetadataReranker(str(videos_path))
    fusion = FusionEngine(metadata_reranker)
    
    aggregator = VideoAggregator(max_frames_per_video=settings.reranking.diversification.max_frames_per_video)
    writer = SubmissionWriter(settings.directories.outputs)
    
    # Find all KIS queries
    query_files = sorted(list(query_dir.glob("*kis*.txt")))
    if not query_files:
        print(f"No KIS query files found in {query_dir}.")
        sys.exit(0)
        
    print(f"\nFound {len(query_files)} KIS queries. Processing...")
    
    for qf in tqdm(query_files, desc="Batch Search"):
        query_id = qf.stem
        # read text
        with open(qf, "r", encoding="utf-8") as f:
            query_text = f.read().strip()
            
        if not query_text:
            continue
            
        # 1. Retrieval
        candidate_frames = multi_retriever.search(query_text, top_k_per_cue=settings.retrieval.top_k_per_cue)
        
        # 2. Aggregation
        aggregated_frames = aggregator.aggregate(candidate_frames, top_k_videos=settings.retrieval.top_k_videos)
        
        # 3. Reranking
        final_frames = fusion.rerank(query_text, aggregated_frames)
        
        # 4. Write CSV submission
        # We need to extract the exact query name. E.g. query-p1-1-kis.txt -> query_id = query-p1-1
        writer.write(query_id, final_frames, top_k=100)

    print("\n[SUCCESS] All queries processed and submissions generated in outputs/submission!")

if __name__ == "__main__":
    main()
