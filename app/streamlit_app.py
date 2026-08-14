import streamlit as st
import sys
import time
from pathlib import Path
import pandas as pd

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

st.set_page_config(page_title="AIC 2026 KIS Search", layout="wide")

@st.cache_resource
def init_pipeline():
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
    
    return multi_retriever, aggregator, fusion

st.title("AIC 2026 KIS Search Engine (Baseline)")

# Initialize pipeline
with st.spinner("Đang tải các mô hình (FAISS, CLIP, Translator)..."):
    multi_retriever, aggregator, fusion = init_pipeline()

# Query input
query = st.text_input("Nhập câu truy vấn Tiếng Việt:", placeholder="ví dụ: một người đàn ông mặc áo đỏ đang đi bộ...")

col1, col2, col3 = st.columns(3)
top_k_cues = col1.slider("Top K (per cue)", min_value=100, max_value=2000, value=settings.retrieval.top_k_per_cue)
top_k_videos = col2.slider("Top K Videos", min_value=10, max_value=200, value=settings.retrieval.top_k_videos)
max_frames = col3.slider("Max Frames per Video", min_value=1, max_value=5, value=settings.reranking.diversification.max_frames_per_video)

if st.button("Tìm kiếm", type="primary"):
    if query:
        start_time = time.time()
        
        with st.spinner("Đang trích xuất đặc trưng và tìm kiếm..."):
            # Update aggregator setting
            aggregator.max_frames = max_frames
            
            candidate_frames = multi_retriever.search(query, top_k_per_cue=top_k_cues)
            aggregated_frames = aggregator.aggregate(candidate_frames, top_k_videos=top_k_videos)
            final_frames = fusion.rerank(query, aggregated_frames)
            
        st.success(f"Tìm kiếm hoàn tất trong {time.time() - start_time:.2f}s")
        
        # Format results
        results_list = []
        for rank, frame in enumerate(final_frames[:100]):
            results_list.append({
                "Rank": rank + 1,
                "Video ID": frame["video_id"],
                "Frame Index": frame["frame_idx"],
                "PTS Time (s)": round(frame["pts_time"], 2),
                "Final Score": round(frame["final_score"], 4),
                "Meta Score": round(frame["meta_score"], 4),
                "Cue Coverage": round(frame.get("cue_coverage", 1.0), 2)
            })
            
        df = pd.DataFrame(results_list)
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("Vui lòng nhập câu truy vấn!")
