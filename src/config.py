from pathlib import Path
from pydantic import BaseModel
import yaml

class DirectoriesConfig(BaseModel):
    raw: Path
    processed: Path
    indexes: Path
    outputs: Path

class DataConfig(BaseModel):
    clip_dim: int
    expected_videos: int
    expected_keyframes: int

class RerankingWeights(BaseModel):
    clip_score: float
    metadata_score: float
    object_score: float
    temporal_score: float
    cue_coverage_multiplier: float

class DiversificationConfig(BaseModel):
    max_frames_per_video: int
    target_results: int

class RerankingConfig(BaseModel):
    weights: RerankingWeights
    diversification: DiversificationConfig

class RetrievalConfig(BaseModel):
    top_k_per_cue: int
    top_k_videos: int

class ModelsConfig(BaseModel):
    text_encoder: str
    translator: str
    use_translation: bool
    fusion_alpha: float

class AppConfig(BaseModel):
    directories: DirectoriesConfig
    data: DataConfig
    retrieval: RetrievalConfig
    reranking: RerankingConfig
    models: ModelsConfig

def load_config(config_path: str = "config/config.yaml") -> AppConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return AppConfig(**data)

# Singleton instance
settings = load_config()
