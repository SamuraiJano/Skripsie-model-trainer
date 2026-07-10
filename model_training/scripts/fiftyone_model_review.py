from __future__ import annotations

import argparse
import json

from naartjie_cv.fiftyone_tools import run_fiftyone_review


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load the synthetic dataset into FiftyOne, apply/evaluate a model, and export hard examples."
    )
    parser.add_argument("--config", default="config/project.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--weights", help="Optional Ultralytics .pt weights to apply and evaluate.")
    parser.add_argument("--dataset-name", help="Optional FiftyOne dataset name.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing FiftyOne dataset of the same name.")
    parser.add_argument("--no-launch", action="store_true", help="Do not launch the FiftyOne App.")
    parser.add_argument("--no-hard-export", action="store_true", help="Do not export hard examples after evaluation.")
    parser.add_argument("--wait", action="store_true", help="Keep the process open until the FiftyOne App is closed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_fiftyone_review(
        config_path=args.config,
        split=args.split,
        weights=args.weights,
        dataset_name=args.dataset_name,
        overwrite=args.overwrite,
        launch=not args.no_launch,
        export_hard=not args.no_hard_export,
        wait=args.wait,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

