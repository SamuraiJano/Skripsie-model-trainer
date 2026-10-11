"""Prepare a separate frozen experiment, inspect it, run it or aggregate results."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))


def main():
    from naartjie_cv.detector_experiment import prepare_experiment, matrix_text, verify_experiment, execute_experiment, aggregate_results
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Pair and verify data; write the matrix without training")
    prepare.add_argument("--dataset", default=str(PROJECT / "dataset_detect_rgb"))
    prepare.add_argument("--dfine-root", default=str(PROJECT / "third_party/D-FINE"))
    prepare.add_argument("--output-root", default=str(PROJECT / "runs/controlled_detection"))
    prepare.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    prepare.add_argument("--fewer-seeds-reason")
    prepare.add_argument("--horizontal-flip", type=float, choices=[0., 0.5], default=0.)
    prepare.add_argument("--tuning-ledger", help="JSON ledger with equal validation-only trials and selected configurations")
    for name in ("verify", "run", "aggregate"):
        command = commands.add_parser(name)
        command.add_argument("manifest", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        ledger = json.loads(Path(args.tuning_ledger).read_text(encoding="utf-8")) if args.tuning_ledger else None
        path = prepare_experiment(args.dataset, args.dfine_root, args.output_root, seeds=args.seeds,
                                  fewer_seeds_reason=args.fewer_seeds_reason, horizontal_flip=args.horizontal_flip,
                                  tuning_ledger=ledger)
        print(matrix_text(json.loads(path.read_text(encoding="utf-8"))))
        print(f"\nFrozen configuration: {path}\nNo training was started.")
    elif args.command == "verify":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        print(matrix_text(manifest))
        print(json.dumps(verify_experiment(manifest), indent=2))
    elif args.command == "run":
        print(execute_experiment(args.manifest))
    else:
        print(aggregate_results(args.manifest))


if __name__ == "__main__":
    main()
