from __future__ import annotations

import argparse

from train_common import add_training_args, train, training_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the HSV naartjie YOLO segmentation model on HSV-converted image files."
    )
    add_training_args(
        parser,
        default_config=training_root() / "configs" / "naartjie_hsv.yaml",
        default_name="naartjie_hsv",
    )
    return parser.parse_args()


def main() -> None:
    train(parse_args(), disable_color_aug=True)


if __name__ == "__main__":
    main()
