import sys
from pathlib import Path
from typing import List, Dict, Any

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
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

class KISTaskRunner:
    """
    Task 1: Textual Known Item Search (Textual KIS)
    Output format: <video_id>,<frame_idx>
    """
    def __init__(self):
        print("[KIS] Initializing Textual KIS Pipeline...")
        translator = VietnameseTranslator(model_name=settings.models.translator)
        clip_encoder = CLIPTextEncoder(model_name=settings.models.text_encoder)
        self.encoder = MultiLingualQueryEncoder(
            encoder=clip_encoder,
            translator=translator,
            use_translation=settings.models.use_translation,
            fusion_alpha=settings.models.fusion_alpha
        )
        self.decomposer = QueryDecomposer()
        
        faiss_path = settings.directories.indexes / "clip.faiss"
        frames_path = settings.directories.processed / "frames.parquet"
        self.retriever = FAISSRetriever(faiss_path, frames_path)
        self.multi_retriever = MultiCueRetriever(self.decomposer, self.encoder, self.retriever)
        
        videos_path = settings.directories.processed / "videos.parquet"
        self.metadata_reranker = MetadataReranker(str(videos_path))
        self.fusion = FusionEngine(self.metadata_reranker)
        self.aggregator = VideoAggregator(max_frames_per_video=settings.reranking.diversification.max_frames_per_video)
        self.writer = SubmissionWriter(settings.directories.outputs)

    def run_query(self, query_id: str, query_text: str, top_k: int = 100) -> Path:
        candidate_frames = self.multi_retriever.search(query_text, top_k_per_cue=settings.retrieval.top_k_per_cue)
        aggregated_frames = self.aggregator.aggregate(candidate_frames, top_k_videos=settings.retrieval.top_k_videos)
        final_frames = self.fusion.rerank(query_text, aggregated_frames)
        output_path = self.writer.write(query_id, final_frames, top_k=top_k)
        return output_path

    def run_batch(self, query_dir: Path, top_k: int = 100):
        query_files = sorted(list(query_dir.glob("*kis*.txt")))
        print(f"[KIS] Found {len(query_files)} KIS queries in {query_dir}.")
        for qf in query_files:
            with open(qf, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                self.run_query(qf.stem, text, top_k=top_k)
