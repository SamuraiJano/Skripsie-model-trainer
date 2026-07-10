from __future__ import annotations

from naartjie_cv.runtime import assert_training_runtime, print_runtime_report


def main() -> None:
    report = assert_training_runtime()
    print_runtime_report(report)


if __name__ == "__main__":
    main()

