import pandas as pd
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MetadataReranker:
    def __init__(self, videos_parquet_path: str):
        print(f"Loading video metadata for reranking from {videos_parquet_path}...")
        self.videos_df = pd.read_parquet(videos_parquet_path)
        self.videos_df = self.videos_df.set_index("video_id")
        
        # Build TF-IDF index over the combined search_text
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # Replace NaNs with empty string
        corpus = self.videos_df['search_text'].fillna("").tolist()
        if corpus:
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
            self.video_ids = self.videos_df.index.tolist()
            # Map video_id to tfidf row index
            self.vid_to_idx = {vid: i for i, vid in enumerate(self.video_ids)}
        else:
            self.tfidf_matrix = None
            
    def score(self, query: str, video_id: str) -> float:
        """
        Returns a normalized score [0, 1] for metadata text match.
        """
        if self.tfidf_matrix is None or not query.strip():
            return 0.0
            
        vid_idx = self.vid_to_idx.get(video_id)
        if vid_idx is None:
            return 0.0
            
        query_vec = self.vectorizer.transform([query.lower()])
        sim = cosine_similarity(query_vec, self.tfidf_matrix[vid_idx])
        return float(sim[0][0])
