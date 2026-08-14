import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.submission.writer import SubmissionWriter
from src.submission.validator import SubmissionValidator
from src.config import settings

def main():
    print("=== Submission Validation ===")
    
    # 1. Generate a dummy submission
    writer = SubmissionWriter(settings.directories.outputs)
    
    dummy_results = [
        {"video_id": "L21_V001", "frame_idx": 123},
        {"video_id": "L21_V001", "frame_idx": 456},
        {"video_id": "L22_V015", "frame_idx": 789},
        {"video_id": "L30_V112", "frame_idx": 1024},
    ]
    
    out_path = writer.write("dummy01", dummy_results, top_k=100)
    
    # 2. Validate it
    validator = SubmissionValidator()
    is_valid, message = validator.validate(out_path)
    
    if is_valid:
        print(f"\n[SUCCESS] {message}")
    else:
        print(f"\n[ERROR] {message}")
        sys.exit(1)

if __name__ == "__main__":
    main()
