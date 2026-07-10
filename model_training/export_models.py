from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def training_root() -> Path:
    return Path(__file__).resolve().parent


def project_root() -> Path:
    return training_root().parent


def export_one(weights: Path, output_path: Path, imgsz: int, opset: int, simplify: bool) -> Path:
    from ultralytics import YOLO

    weights = weights.resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = YOLO(str(weights))
    exported = model.export(format="onnx", imgsz=imgsz, opset=opset, simplify=simplify)
    exported_path = Path(str(exported)).resolve()
    if not exported_path.exists():
        raise FileNotFoundError(f"Ultralytics did not produce an ONNX file at: {exported_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported_path, output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    root = training_root()
    backend_models = project_root() / "background" / "models"
    parser = argparse.ArgumentParser(description="Export trained RGB and HSV naartjie YOLO models to ONNX for the C++ backend.")
    parser.add_argument(
        "--rgb-weights",
        type=Path,
        default=root / "runs" / "segment" / "naartjie_rgb" / "weights" / "best.pt",
    )
    parser.add_argument(
        "--hsv-weights",
        type=Path,
        default=root / "runs" / "segment" / "naartjie_hsv" / "weights" / "best.pt",
    )
    parser.add_argument("--rgb-output", type=Path, default=backend_models / "rgb_yolo_seg.onnx")
    parser.add_argument("--hsv-output", type=Path, default=backend_models / "hsv_yolo_seg.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--no-simplify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    simplify = not args.no_simplify
    rgb = export_one(args.rgb_weights, args.rgb_output, args.imgsz, args.opset, simplify)
    hsv = export_one(args.hsv_weights, args.hsv_output, args.imgsz, args.opset, simplify)
    print(f"RGB ONNX: {rgb}")
    print(f"HSV ONNX: {hsv}")


if __name__ == "__main__":
    main()
