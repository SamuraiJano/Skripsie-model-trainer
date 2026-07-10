from __future__ import annotations

import argparse
import json

from naartjie_cv.evaluate import evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO26 segmentation model.")
    parser.add_argument("--config", default="config/project.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--variant", choices=["rgb", "rgb_hsv"], default="rgb")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_model(args.config, args.weights, args.variant, args.split)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

