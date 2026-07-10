from __future__ import annotations

import argparse

from naartjie_cv.runtime import assert_training_runtime, print_runtime_report
from naartjie_cv.train import train_yolo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or resume a YOLO26 segmentation model.")
    parser.add_argument("--config", default="config/project.yaml")
    parser.add_argument("--variant", choices=["rgb", "rgb_hsv"], default="rgb")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_runtime_report(assert_training_runtime())
    train_yolo(args.config, args.variant)


if __name__ == "__main__":
    main()

