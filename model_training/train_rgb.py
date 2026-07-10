from __future__ import annotations

import argparse

from train_common import add_training_args, train, training_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the RGB naartjie YOLO segmentation model.")
    add_training_args(
        parser,
        default_config=training_root() / "configs" / "naartjie_rgb.yaml",
        default_name="naartjie_rgb",
    )
    return parser.parse_args()


def main() -> None:
    train(parse_args(), disable_color_aug=False)


if __name__ == "__main__":
    main()
