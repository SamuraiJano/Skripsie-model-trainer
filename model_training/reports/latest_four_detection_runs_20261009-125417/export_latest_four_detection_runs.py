"""Collect the latest four colour detector runs without changing their files."""
from __future__ import annotations

import csv
import json
import re
import shutil
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
LOCAL_TIME = timezone(timedelta(hours=2))
STAMP = datetime.now(LOCAL_TIME).strftime("%Y%m%d-%H%M%S")
OUTPUT = PROJECT / "reports" / f"latest_four_detection_runs_{STAMP}"
GROUPS = (("yolo_rgb", "YOLO RGB", "ultralytics", "rgb"),
          ("yolo_hsv", "YOLO HSV", "ultralytics", "hsv"),
          ("dfine_rgb", "D-FINE RGB", "ArgoHA/D-FINE-seg", "rgb"),
          ("dfine_hsv", "D-FINE HSV", "ArgoHA/D-FINE-seg", "hsv"))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def csv_data(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return {"columns": reader.fieldnames, "rows": rows}


def link(path, label):
    return f"[{label}](<{path.resolve().as_posix()}>)"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def value_string(value):
    return str(value) if value is not None else "Not saved"


def metric_table(metrics):
    lines = ["| Saved metric | Value |", "|---|---:|"]
    lines.extend(f"| `{key}` | {value_string(value)} |" for key, value in metrics.items())
    return lines


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    candidates = []
    for folder in (PROJECT / "runs/yolo26", PROJECT / "runs/dfine-seg"):
        for path in folder.glob("*/run_metadata.json"):
            metadata = read_json(path)
            if metadata.get("task") == "detect":
                candidates.append((path.parent, metadata))
    selected = []
    for key, label, trainer, colour in GROUPS:
        matches = [(directory, metadata) for directory, metadata in candidates
                   if metadata.get("trainer") == trainer
                   and metadata.get("experiment", {}).get("colour_space") == colour]
        if not matches:
            raise RuntimeError(f"Missing run: {label}")
        # Directory stamps record run start. created_at can record final export instead.
        directory, metadata = max(matches, key=lambda item: re.search(r"(\d{8}-\d{6})$", item[0].name).group(1))
        selected.append((key, label, directory, metadata))

    bundle = {"generated_at": datetime.now(LOCAL_TIME).isoformat(),
              "selection_rule": "Latest run-start directory timestamp in each trainer/colour group; detection only",
              "source_files_unchanged": True, "runs": []}
    inventory = []
    original_files = []
    reference_configs = []
    warnings = []
    report = ["# Latest four detection runs", "",
              "Saved metrics are reproduced at their stored precision. No training or inference was run.", "",
              "D-FINE here is the ArgoHA/D-FINE-seg implementation in detection mode.", "",
              "## Test comparison", "",
              "All four saved test evaluations contain 504 images. Values below are fractions, not percentages.", "",
              "| Run | Precision | Recall | mAP50 | mAP50-95 | Saved F1 | Mean IoU | TP | FP | FN |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

    for key, label, directory, metadata in selected:
        test_path = directory / "test_metrics.json"
        test = read_json(test_path) if test_path.is_file() else None
        metrics = test.get("metrics", {}) if test else {}
        pick = lambda native, yolo=None: metrics.get(native, metrics.get(yolo) if yolo else None)
        report.append("| " + " | ".join([label, *(value_string(v) for v in (
            pick("precision", "metrics/precision(B)"), pick("recall", "metrics/recall(B)"),
            pick("mAP_50", "metrics/mAP50(B)"), pick("mAP_50_95", "metrics/mAP50-95(B)"),
            pick("f1"), pick("iou"), pick("TPs"), pick("FPs"), pick("FNs")))]) + " |")
        if not test:
            warnings.append(f"{label}: no saved test evaluation.")
        if metadata.get("stop_reason") not in {"early_stopping", "max_epochs"}:
            warnings.append(f"{label}: stop reason is {metadata.get('stop_reason')!r}.")

        files = sorted(p for p in directory.rglob("*") if p.is_file())
        record = {"id": key, "label": label, "run_id": directory.name,
                  "source_directory": str(directory), "metadata": metadata,
                  "saved_test": test, "metric_json_files": {}, "metric_csv_files": {},
                  "validation_history": [], "config_files": {}, "notes": [], "file_count": len(files)}
        for source in files:
            relative = source.relative_to(directory)
            prepared = "prepared_dataset" in relative.parts
            is_image = source.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
            archive_path = Path("runs") / key / directory.name / relative
            stat = source.stat()
            entry = {"run": key, "relative_path": relative.as_posix(), "absolute_path": str(source),
                     "archive_path": archive_path.as_posix(), "bytes": stat.st_size,
                     "mtime_ns": stat.st_mtime_ns, "included_in_complete_archive": True,
                     "included_in_artifacts_archive": not (prepared and is_image)}
            inventory.append(entry)
            original_files.append((source, archive_path.as_posix(), entry))
            if prepared:
                continue
            if source.suffix.lower() == ".json":
                record["metric_json_files"][relative.as_posix()] = read_json(source)
            elif source.suffix.lower() == ".csv":
                record["metric_csv_files"][relative.as_posix()] = csv_data(source)
            elif source.suffix.lower() == ".jsonl":
                record["validation_history"] = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
            elif source.suffix.lower() in {".yaml", ".yml"}:
                record["config_files"][relative.as_posix()] = source.read_text(encoding="utf-8-sig")

        if "results.csv" in record["metric_csv_files"]:
            rows = record["metric_csv_files"]["results.csv"]["rows"]
            record["training_csv_rows"] = len(rows)
            record["last_training_row"] = rows[-1] if rows else None
            record["best_training_csv_row"] = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"])) if rows else None
            record["recorded_training_seconds"] = float(rows[-1]["time"]) if rows else None
            recorded = metadata.get("completed_epochs")
            if recorded != len(rows):
                record["notes"].append(f"Metadata reports {recorded} completed epochs; results.csv contains {len(rows)} training rows. The validation history contains a final best-checkpoint re-evaluation as an additional entry. Both source counts are preserved.")
            if record["best_training_csv_row"] and int(record["best_training_csv_row"]["epoch"]) != metadata.get("selection", {}).get("best_epoch"):
                record["notes"].append(f"Best training CSV epoch is {record['best_training_csv_row']['epoch']}; metadata records best epoch {metadata.get('selection', {}).get('best_epoch')}. The latter corresponds to the final checkpoint re-evaluation entry. No source metadata was corrected.")
            record["notes"].append("final_metrics.json combines final best-checkpoint validation P/R/AP with losses, learning rates and time from the last training CSV row. These are different evaluation contexts.")
            reference = metadata.get("artifacts", {}).get("training_data_yaml")
            if reference and Path(reference).is_file():
                reference_configs.append((Path(reference), f"referenced_configs/{key}/{Path(reference).name}"))
        else:
            text = (directory / "model/train_log.txt").read_text(encoding="utf-8")
            duration = re.findall(r"Full training time:\s*([\d.]+)\s*hours", text)
            record["recorded_training_hours"] = float(duration[-1]) if duration else None
            record["training_duration_precision"] = "Native log rounded to two decimal hours"
            extended = record["metric_csv_files"].get("model/extended_metrics.csv", {}).get("rows", [])
            record["saved_operating_thresholds"] = {row.get("", row.get("Unnamed: 0")): row.get("optimal_thresh") for row in extended}
            options = metadata.get("options", {})
            record["operating_metric_confidence"] = options.get("conf_threshold")
            record["operating_metric_iou"] = options.get("iou_threshold")
            record["notes"].append(f"The native D-FINE P/R/F1 and counts are computed at the configured confidence {options.get('conf_threshold')} and matching IoU {options.get('iou_threshold')}. Separate validation/test F1-optimal thresholds in extended_metrics.csv describe the plotted confidence sweeps; they are not the thresholds used for the reported P/R/F1. AP uses the native full ranking.")
        warnings.extend(f"{label}: {note}" for note in record["notes"])
        bundle["runs"].append(record)
        print(json.dumps({"selected": label, "run": directory.name, "files": len(files), "test_metrics": metrics}), flush=True)

    report.extend(["", "YOLO does not save scalar F1, IoU or TP/FP/FN in its test JSON; these are marked Not saved rather than invented.", "",
                   "## Training and selection", "",
                   "| Run | Metadata completed epochs | Training CSV rows | Metadata best epoch | Best training CSV epoch | Selected validation mAP50-95 | Native training duration | Optimiser updates |",
                   "|---|---:|---:|---:|---:|---:|---|---:|"])
    for record in bundle["runs"]:
        metadata = record["metadata"]
        selection = metadata.get("selection", {})
        duration = (f"{record['recorded_training_seconds']} seconds" if record.get("recorded_training_seconds") is not None
                    else f"{record.get('recorded_training_hours')} hours (rounded)")
        best_csv = record.get("best_training_csv_row") or {}
        values = [record["label"], metadata.get("completed_epochs"), record.get("training_csv_rows"),
                  selection.get("best_epoch"), best_csv.get("epoch"), selection.get("best_validation_metric"),
                  duration, metadata.get("actual_optimizer_steps")]
        report.append("| " + " | ".join(value_string(v) for v in values) + " |")

    report.extend(["", "## Source details", ""])
    for record in bundle["runs"]:
        directory = Path(record["source_directory"])
        report.extend([f"### {record['label']}", "", f"Run: `{record['run_id']}`", "",
                       f"Source: {link(directory, 'Original run folder')}", "",
                       f"Stop reason: `{record['metadata'].get('stop_reason')}`. Test images: {record['saved_test'].get('images') if record['saved_test'] else 'Not saved'}.", ""])
        for note in record["notes"]:
            report.extend([note, ""])
        for filename, metrics in record["metric_json_files"].items():
            if filename == "run_metadata.json":
                continue
            report.extend([f"#### {filename}", "", link(directory / filename, "Original file"), ""])
            report.extend(metric_table(metrics.get("metrics", metrics)))
            report.append("")
        for filename, data in record["metric_csv_files"].items():
            report.extend([f"#### {filename}", "", link(directory / filename, "Original file"), ""])
            if filename == "results.csv":
                report.extend([f"All {len(data['rows'])} rows and every column are included in all_metrics.json and both archives.", "",
                               "Last training row:", "", *metric_table(record["last_training_row"]), "",
                               "Best training CSV row:", "", *metric_table(record["best_training_csv_row"]), ""])
            else:
                headers = data["columns"]
                report.append("| " + " | ".join(h or "Split" for h in headers) + " |")
                report.append("| " + " | ".join("---" for _ in headers) + " |")
                for row in data["rows"]:
                    report.append("| " + " | ".join(value_string(row[h]) for h in headers) + " |")
                report.append("")
        report.extend([f"All {len(record['validation_history'])} validation-history records are retained in all_metrics.json.", ""])

    bundle["notes"] = warnings
    bundle["file_count"] = len(inventory)
    bundle["source_bytes"] = sum(entry["bytes"] for entry in inventory)
    bundle["prepared_image_bytes"] = sum(entry["bytes"] for entry in inventory if not entry["included_in_artifacts_archive"])
    index = ["# Complete file inventory", "", "Every file in the four selected run folders is listed below.", "",
             "The complete archive includes every listed file. The smaller artifacts archive omits only the 1,851 prepared dataset images; its annotation JSON files are included.", ""]
    for key, label, directory, _ in selected:
        entries = [entry for entry in inventory if entry["run"] == key]
        index.extend([f"## {label}: {directory.name}", "", f"{len(entries)} files; {sum(e['bytes'] for e in entries):,} bytes.", "",
                      "| File | Bytes | Artifacts archive |", "|---|---:|---|"])
        for entry in entries:
            index.append(f"| {link(Path(entry['absolute_path']), entry['relative_path'])} | {entry['bytes']} | {'Included' if entry['included_in_artifacts_archive'] else 'Complete archive only'} |")
        index.append("")
    write_json(OUTPUT / "all_metrics.json", bundle)
    write_json(OUTPUT / "file_inventory.json", inventory)
    (OUTPUT / "metrics_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUTPUT / "file_inventory.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    shutil.copy2(__file__, OUTPUT / "export_latest_four_detection_runs.py")
    support_files = sorted(p for p in OUTPUT.iterdir() if p.is_file())

    archived = []
    for name, complete in (("latest_four_run_artifacts.zip", False), ("latest_four_runs_complete.zip", True)):
        target = OUTPUT / name
        started = time.monotonic()
        members = [(source, arc, entry) for source, arc, entry in original_files if complete or entry["included_in_artifacts_archive"]]
        if shutil.disk_usage(OUTPUT).free < sum(entry["bytes"] for _, _, entry in members) + 1024**3:
            raise RuntimeError("Insufficient space to safely create the complete archive")
        print(json.dumps({"archive_started": str(target), "source_files": len(members)}), flush=True)
        with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            for number, (source, arc, entry) in enumerate(members, 1):
                before = source.stat()
                if (before.st_size, before.st_mtime_ns) != (entry["bytes"], entry["mtime_ns"]):
                    raise RuntimeError(f"Source changed during export: {source}")
                compressed = zipfile.ZIP_STORED if source.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} else zipfile.ZIP_DEFLATED
                archive.write(source, arc, compress_type=compressed, compresslevel=1)
                after = source.stat()
                if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
                    raise RuntimeError(f"Source changed during archive write: {source}")
                if number % 250 == 0:
                    print(json.dumps({"archive": name, "written": number, "total": len(members)}), flush=True)
            for source in support_files:
                archive.write(source, f"reports/{source.name}")
            for source, arc in reference_configs:
                archive.write(source, arc)
        print(json.dumps({"archive_written": name, "bytes": target.stat().st_size, "seconds": round(time.monotonic()-started, 2), "checking_crc": True}), flush=True)
        with zipfile.ZipFile(target) as archive:
            expected = {arc for _, arc, _ in members} | {f"reports/{p.name}" for p in support_files} | {arc for _, arc in reference_configs}
            if set(archive.namelist()) != expected or len(archive.namelist()) != len(expected):
                raise RuntimeError(f"Archive file list mismatch: {name}")
            for _, arc, entry in members:
                if archive.getinfo(arc).file_size != entry["bytes"]:
                    raise RuntimeError(f"Archive file size mismatch: {arc}")
            damaged = archive.testzip()
            if damaged:
                raise RuntimeError(f"Archive CRC failure: {damaged}")
        archived.append({"file": str(target), "bytes": target.stat().st_size, "source_files": len(members),
                         "crc_check": "PASS", "member_list_check": "PASS", "size_check": "PASS"})
        print(json.dumps({"archive_verified": name, "elapsed_seconds": round(time.monotonic()-started, 2)}), flush=True)
    write_json(OUTPUT / "export_verification.json", {"source_file_count": len(inventory), "source_bytes": bundle["source_bytes"],
                                                   "archives": archived, "source_files_unchanged": True})
    print(json.dumps({"output": str(OUTPUT), "verification": "PASS", "source_files": len(inventory)}), flush=True)


if __name__ == "__main__":
    main()
