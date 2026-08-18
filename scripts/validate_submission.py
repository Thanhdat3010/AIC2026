import sys
from pathlib import Path

# Fix Windows console UTF-8 output encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.submission.submission_validator import SubmissionValidator

def main():
    print("=" * 80)
    print("[*] KIEM TRA DINH DANG TEP NOP BAI CHUAN BTC (AIC 2026 VALIDATION)")
    print("=" * 80)
    
    validator = SubmissionValidator()
    output_dir = PROJECT_ROOT / "output" / "batch_1"
    
    if not output_dir.exists():
        print(f"[!] Khong tim thay thu muc: {output_dir}")
        sys.exit(1)
        
    summary = validator.validate_directory(output_dir)
    print(f"[*] Thu muc kiem tra: {output_dir}")
    print(f"[*] Tong so file CSV: {summary['total_files']}")
    print(f"[*] Ket qua kiem chuan: {'[SUCCESS] 100% HOP LE QUY CHE BTC' if summary['all_valid'] else '[FAILED] PHAT HIEN LOI DINH DANG'}")
    
    if not summary["all_valid"]:
        for f, err in summary.get("details", {}).items():
            if not err["valid"]:
                print(f"   - {f}: {err['reason']}")
        sys.exit(1)
    else:
        print("\n[SUCCESS] Toan bo cac file CSV da san sang de dong goi submission.zip!")

if __name__ == "__main__":
    main()
