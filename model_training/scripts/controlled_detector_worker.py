"""Isolated subprocess entry point for one frozen detector run."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    from naartjie_cv.detector_worker import run_one
    run_one(args.manifest, args.run_id)
