from __future__ import annotations

import argparse
import json

from naartjie_cv.multispectral import create_rgb_hsv_dataset
from naartjie_cv.synthetic import generate_synthetic_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic naartjie defect datasets.")
    parser.add_argument("--config", default="config/project.yaml")
    parser.add_argument("--skip-rgb-hsv", action="store_true", help="Only generate the RGB dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rgb_summary = generate_synthetic_dataset(args.config)
    print(json.dumps({"rgb": rgb_summary}, indent=2))
    if not args.skip_rgb_hsv:
        rgb_hsv_summary = create_rgb_hsv_dataset(args.config)
        print(json.dumps({"rgb_hsv": rgb_hsv_summary}, indent=2))


if __name__ == "__main__":
    main()

