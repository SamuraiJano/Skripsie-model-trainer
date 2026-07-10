from __future__ import annotations

import argparse
import json

from naartjie_cv.classical import run_classical_baselines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classical colour/texture baselines.")
    parser.add_argument("--config", default="config/project.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["hsv_thresholding", "lab_kmeans", "hsv_texture_classifier"],
        default=["hsv_thresholding", "lab_kmeans", "hsv_texture_classifier"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = run_classical_baselines(args.config, args.split, args.methods)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()

