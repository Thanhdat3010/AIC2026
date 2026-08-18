import pytest
from pathlib import Path
import sys
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import settings
from src.query.query_decomposer import QueryDecomposer
from src.submission.validator import SubmissionValidator

def test_config_paths_exist():
    assert settings.directories.raw.exists()
    assert settings.directories.processed.exists()
    assert (settings.directories.processed / "frames.parquet").exists()
    assert (settings.directories.processed / "siglip_features.npy").exists()

def test_query_decomposer():
    decomposer = QueryDecomposer()
    cues = decomposer.decompose("người đi bộ, mặc áo đỏ")
    assert len(cues) == 2
    assert cues[0] == "người đi bộ"
    assert cues[1] == "mặc áo đỏ"
    
    cues = decomposer.decompose("một con chó và một con mèo")
    assert len(cues) == 2
    assert cues[0] == "một con chó"
    assert cues[1] == "một con mèo"

def test_submission_validator(tmp_path):
    validator = SubmissionValidator()
    
    valid_csv = tmp_path / "valid.csv"
    valid_csv.write_text("L21_V001,123\nL22_V015,456\n")
    is_valid, msg = validator.validate(valid_csv)
    assert is_valid
    
    invalid_csv = tmp_path / "invalid.csv"
    invalid_csv.write_text("video_id,frame_idx\nL21_V001,123\n")
    is_valid, msg = validator.validate(invalid_csv)
    assert not is_valid
    assert "Expected format" in msg

def test_frames_parquet_schema():
    frames_path = settings.directories.processed / "frames.parquet"
    df = pd.read_parquet(frames_path)
    expected_cols = {"global_id", "video_id", "keyframe_index", "pts_time", "fps", "frame_idx", "clip_row", "position_ratio"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 177321
