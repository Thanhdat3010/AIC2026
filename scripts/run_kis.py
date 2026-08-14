import argparse
import sys
import time
from pathlib import Path

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

def main():
    parser = argparse.ArgumentParser(description="Run AIC 2026 KIS Pipeline (M6)")
    parser.add_argument("--query", type=str, required=True, help="Vietnamese textual query")
    args = parser.parse_args()
    
    print("=== Initializing KIS Pipeline ===")
    start_init = time.time()
    
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
    
    print(f"Initialization took {time.time() - start_init:.2f}s")
    
    print(f"\n=== Processing Query ===")
    print(f"Query: {args.query}")
    
    start_search = time.time()
    
    # 1. Retrieval
    candidate_frames = multi_retriever.search(args.query, top_k_per_cue=settings.retrieval.top_k_per_cue)
    print(f"Retrieved {len(candidate_frames)} candidate frames across cues.")
    
    # 2. Aggregation
    aggregated_frames = aggregator.aggregate(candidate_frames, top_k_videos=settings.retrieval.top_k_videos)
    print(f"Aggregated down to {len(aggregated_frames)} frames across {settings.retrieval.top_k_videos} videos.")
    
    # 3. Reranking
    final_frames = fusion.rerank(args.query, aggregated_frames)
    
    print(f"\nSearch took {time.time() - start_search:.2f}s")
    
    print("\n=== Top 10 Results ===")
    for i, frame in enumerate(final_frames[:10]):
        vid = frame['video_id']
        f_idx = frame['frame_idx']
        pts = frame['pts_time']
        f_score = frame['final_score']
        print(f"{i+1:2d}. {vid}, Frame {f_idx:5d} (PTS: {pts:6.2f}s) | Score: {f_score:.4f}")

if __name__ == "__main__":
    main()
