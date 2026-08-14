import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from tasks.kis import KISTaskRunner
from tasks.qa import QATaskRunner
from tasks.trake import TRAKETaskRunner

def main():
    parser = argparse.ArgumentParser(description="Run AIC 2026 Tasks (KIS, QA, TRAKE)")
    parser.add_argument("--task", type=str, choices=["kis", "qa", "trake", "all"], default="kis",
                        help="Which task pipeline to run (kis, qa, trake, all)")
    parser.add_argument("--query_dir", type=str, default="query/query-p1-groupA",
                        help="Directory containing query files")
    parser.add_argument("--top_k", type=int, default=100,
                        help="Number of results per query")
    args = parser.parse_args()
    
    query_dir = Path(args.query_dir)
    if not query_dir.exists():
        print(f"[ERROR] Directory '{query_dir}' does not exist.")
        sys.exit(1)

    if args.task in ["kis", "all"]:
        print("\n" + "="*50)
        print(">> RUNNING TASK 1: TEXTUAL KIS")
        print("="*50)
        kis_runner = KISTaskRunner()
        kis_runner.run_batch(query_dir, top_k=args.top_k)

    if args.task in ["qa", "all"]:
        print("\n" + "="*50)
        print(">> RUNNING TASK 2: VISUAL QA")
        print("="*50)
        qa_runner = QATaskRunner()
        qa_runner.run_batch(query_dir, top_k=args.top_k)

    if args.task in ["trake", "all"]:
        print("\n" + "="*50)
        print(">> RUNNING TASK 3: TRAKE (TEMPORAL ALIGNMENT)")
        print("="*50)
        trake_runner = TRAKETaskRunner()
        trake_runner.run_batch(query_dir, top_k=args.top_k)

if __name__ == "__main__":
    main()
