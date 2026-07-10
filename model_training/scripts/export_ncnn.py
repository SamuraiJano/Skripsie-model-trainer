from __future__ import annotations

import argparse

from naartjie_cv.export import export_ncnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a YOLO26 model to Pi 4-friendly NCNN.")
    parser.add_argument("--config", default="config/project.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--variant", choices=["rgb", "rgb_hsv"], default="rgb")
    parser.add_argument("--imgsz", type=int, default=None, help="Override export image size; default is 640 in config.")
    parser.add_argument("--int8", action=argparse.BooleanOptionalAction, default=None, help="Enable or disable INT8 NCNN export.")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=None, help="Enable or disable FP16 export.")
    parser.add_argument("--simplify", action=argparse.BooleanOptionalAction, default=None, help="Enable or disable graph simplification.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exported = export_ncnn(
        args.config,
        args.weights,
        args.variant,
        imgsz=args.imgsz,
        half=args.half,
        int8=args.int8,
        simplify=args.simplify,
    )
    print(exported)


if __name__ == "__main__":
    main()
