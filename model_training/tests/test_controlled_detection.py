"""Scientific controls, common metrics and real model-loader representation checks."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from naartjie_cv.detection_data import prepare_detector_pairs, detection_ground_truth, letterbox_rgb, map_boxes
from naartjie_cv.detection_metrics import coco_ap, clean_predictions, select_confidence, evaluate_test
from naartjie_cv.detector_experiment import prepare_experiment, verify_experiment, digest_json, baseline_recipes, aggregate_results
from naartjie_cv.detector_worker import final_evaluation, protocol_for, official_config


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source_rgb"
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        pixels = np.tile(np.array([140, 30, 210], dtype=np.uint8), (32, 73, 1))
        Image.fromarray(pixels).save(root / "images" / split / f"{split}.png")
        (root / "labels" / split / f"{split}.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    (root / "data.yaml").write_text(yaml.safe_dump({"path": str(root), "train": "images/train", "val": "images/val",
                                                   "test": "images/test", "names": {0: "fruit"}, "colour_space": "rgb"}))
    return root


@pytest.fixture
def frozen(source, tmp_path):
    upstream = tmp_path / "official"
    (upstream / "configs/dfine/custom").mkdir(parents=True)
    (upstream / "configs/dfine/custom/dfine_hgnetv2_n_custom.yml").write_text("task: detection\n")
    (upstream / "pretrained").mkdir()
    (upstream / "pretrained/dfine_n_coco.pth").write_bytes(b"official nano fixture")
    yolo = tmp_path / "yolo26n.pt"
    yolo.write_bytes(b"yolo nano fixture")
    path = prepare_experiment(source, upstream, tmp_path / "results", yolo_checkpoint=yolo)
    return path, json.loads(path.read_text())


@pytest.fixture
def truth():
    return {"info": {}, "images": [{"id": 1, "width": 100, "height": 100}],
            "categories": [{"id": 0, "name": "fruit"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [20., 20., 40., 40.],
                             "area": 1600., "iscrowd": 0}]}


def prediction(score=.8, bbox=None):
    return {"image_id": 1, "category_id": 0, "score": score, "bbox": bbox or [20., 20., 40., 40.]}


def test_common_perfect_and_empty_metrics(truth):
    result = evaluate_test(truth, [prediction()], select_confidence(truth, [prediction()]))
    assert set(("mAP50-95", "mAP50", "precision", "recall", "F1", "selected_confidence")) <= result.keys()
    assert result["mAP50-95"] == pytest.approx(1)
    assert result["mAP50"] == pytest.approx(1)
    assert result["precision"] == result["recall"] == result["F1"] == 1
    empty = evaluate_test(truth, [], select_confidence(truth, []))
    assert empty["mAP50-95"] == empty["F1"] == 0


def test_confidence_uses_validation_only_and_ap_ignores_threshold(truth):
    predictions = [prediction(.2), prediction(.1, [0., 0., 5., 5.])]
    selection = select_confidence(truth, predictions)
    assert selection["confidence"] == .2
    assert selection["metrics"]["F1"] == 1
    with pytest.raises(ValueError, match="validation"):
        select_confidence(truth, predictions, split="test")
    with pytest.raises(ValueError, match="validation"):
        evaluate_test(truth, predictions, {"split": "test", "confidence": .2})
    a = evaluate_test(truth, predictions, selection)
    b = evaluate_test(truth, predictions, {**selection, "confidence": .9})
    assert a["mAP50-95"] == b["mAP50-95"] == pytest.approx(1)
    assert a["F1"] == 1 and b["F1"] == 0


def test_score_ties_include_all_detections(truth):
    selection = select_confidence(truth, [prediction(.8), prediction(.8)])
    assert selection["metrics"]["TP"] == 1 and selection["metrics"]["FP"] == 1
    assert selection["metrics"]["F1"] == pytest.approx(2 / 3)


def test_zero_f1_tie_selects_highest_threshold(truth):
    assert select_confidence(truth, [prediction(.8, [0, 0, 1, 1])])["confidence"] == 1


def test_ap_detection_cap_floor_and_matching(truth):
    predictions = [prediction(.5, [0, 0, 2, 2]) for _ in range(101)] + [prediction(.1), prediction(.0005)]
    assert len(clean_predictions(truth, predictions)) == 100
    assert coco_ap(truth, predictions)["mAP50-95"] == 0


@pytest.mark.parametrize("change", [{"score": float("nan")}, {"category_id": 12}, {"image_id": 9}, {"bbox": [0, 0, -1, 2]}])
def test_invalid_predictions_rejected(truth, change):
    with pytest.raises(ValueError):
        clean_predictions(truth, [{**prediction(), **change}])


def test_exact_pair_pixels_labels_geometry_and_full_hue(source, tmp_path):
    pair = prepare_detector_pairs(source, tmp_path / "cache")
    assert pair["labels_identical"] and pair["dimensions_identical"] and pair["HSV_FULL_pixels_verified"]
    assert pair["split_integrity"]["fruit_track_split_integrity"] == "NOT PROVABLE"
    from naartjie_cv.detection_data import load_canonical_rgb, to_representation
    rgb = load_canonical_rgb(source / "images/train/train.png")
    hsv = to_representation(rgb, "hsv")
    audit = json.loads(Path(pair["paired_sample_audit"]).read_text())
    import hashlib
    sample = next(s for s in audit if s["split"] == "train")
    assert sample["hsv_full_pixel_sha256"] == hashlib.sha256(hsv.tobytes()).hexdigest()
    assert sample["dimensions"] == [73, 32]
    assert hsv[0, 0, 0] > 179
    np.testing.assert_array_equal(hsv, cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV_FULL))
    for split in ("train", "val", "test"):
        assert detection_ground_truth(pair["rgb"], split)["annotations"] == detection_ground_truth(pair["hsv"], split)["annotations"]
    padded, geometry = letterbox_rgb(rgb)
    assert padded.shape == (640, 640, 3)
    boxes = np.array([[0, 0, 73, 32], [10, 3, 50, 29]], dtype=float)
    np.testing.assert_allclose(map_boxes(map_boxes(boxes, geometry), geometry, inverse=True), boxes, atol=1e-5)


@pytest.mark.parametrize("kind", ["pixels", "labels", "dimensions"])
def test_pair_tampering_blocks_training(source, tmp_path, kind):
    pair = prepare_detector_pairs(source, tmp_path / "cache")
    if kind == "labels":
        config = yaml.safe_load(Path(pair["hsv"]).read_text())
        config["names"] = {0: "changed-class"}
        Path(pair["hsv"]).write_text(yaml.safe_dump(config))
    else:
        audit = Path(pair["paired_sample_audit"])
        samples = json.loads(audit.read_text())
        samples[0]["dimensions" if kind == "dimensions" else "hsv_full_pixel_sha256"] = [1, 1] if kind == "dimensions" else "changed"
        audit.write_text(json.dumps(samples))
    with pytest.raises(ValueError):
        prepare_detector_pairs(source, tmp_path / "cache")


def test_duplicate_annotations_removed_consistently_without_changing_export(source, tmp_path):
    original = source / "labels/train/train.txt"
    content = "0 0.5 0.5 0.5 0.5\n0 .50 .50 .50 .50\n"
    original.write_text(content)
    pair = prepare_detector_pairs(source, tmp_path / "cache")
    assert original.read_text() == content
    assert pair["duplicate_annotations_removed"] == {"train": 1, "val": 0, "test": 0}
    for colour in ("rgb", "hsv"):
        assert len(detection_ground_truth(pair[colour], "train")["annotations"]) == 1
    source_identity = pair["source_identity"]["splits"]["train"]
    resolved = pair["dataset_identity"]["splits"]["train"]
    assert source_identity["identity_sha256"] == resolved["identity_sha256"]
    assert source_identity["annotation_sha256"] != resolved["annotation_sha256"]
    # Corruption of the generated label view fails verification; it is not repaired.
    generated = Path(pair["rgb"]).parent / "labels/train/train.txt"
    generated.write_text("0 0.5 0.5 0.1 0.1\n")
    with pytest.raises(ValueError, match="shared labels changed"):
        prepare_detector_pairs(source, tmp_path / "cache")


@pytest.mark.parametrize("field", ["fruit_identity", "track_identity", "sequence_identity", "recording_sequence_identity"])
def test_group_leakage_blocks_preserved_splits(source, tmp_path, field):
    data = yaml.safe_load((source / "data.yaml").read_text())
    data[field] = {"train/train": "same-fruit-or-sequence", "val/val": "same-fruit-or-sequence"}
    (source / "data.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="crosses splits"):
        prepare_detector_pairs(source, tmp_path / "cache")


def test_frozen_twenty_run_matrix_and_pair_recipes(frozen):
    path, manifest = frozen
    assert len(manifest["matrix"]) == 20
    assert manifest["seeds"] == [0, 1, 2, 3, 4]
    assert verify_experiment(manifest)["status"] == "PASS"
    assert "max_epochs" in (path.parent / "experiment_matrix.csv").read_text()
    assert not list(path.parent.glob("*/run_metadata.json"))
    for model in ("yolo26n", "dfine_n"):
        jobs = [j for j in manifest["matrix"] if j["model"] == model]
        assert len({j["recipe_sha256"] for j in jobs}) == len({j["checkpoint"] for j in jobs}) == 1
        assert manifest["recipes"][model]["model_size"] == "Nano"


@pytest.mark.parametrize("setting,value", [("epochs", 299), ("imgsz", 512), ("hsv_h", .01), ("mosaic", .5)])
def test_invalid_controlled_recipe_rejected_even_if_manifest_rehashed(frozen, setting, value):
    _, m = frozen
    m["recipes"]["yolo26n"]["training"][setting] = value
    for job in m["matrix"]:
        if job["model"] == "yolo26n":
            job["recipe_sha256"] = digest_json(m["recipes"]["yolo26n"])
    m["manifest_sha256"] = digest_json({k: v for k, v in m.items() if k != "manifest_sha256"})
    with pytest.raises(ValueError):
        verify_experiment(m, check_files=False)


def test_frozen_checkpoint_changes_block(frozen):
    _, m = frozen
    Path(m["recipes"]["yolo26n"]["checkpoint"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        verify_experiment(m)


def test_pair_recipe_mismatch_blocks(frozen):
    _, m = frozen
    m["matrix"][1]["recipe_sha256"] = "another-recipe"
    m["manifest_sha256"] = digest_json({k: v for k, v in m.items() if k != "manifest_sha256"})
    with pytest.raises(ValueError, match="recipes differ"):
        verify_experiment(m, check_files=False)


def test_tuning_opportunity_equal_and_selected_recipe_applied():
    from naartjie_cv.detector_experiment import validate_tuning, apply_selected_trials
    ledger = {"maximum_trials_per_architecture": 3, "test_split_used": False,
              "trial_protocol": {"colour": "rgb", "seed": 0, "max_epochs": 60},
              "trials": {m: [{"configuration": {"lr0": .0005} if m == "yolo26n" else {"head_lr": .0006},
                              "selection_split": "val", "selected": True, "validation_mAP50-95": .5}]
                         for m in ("yolo26n", "dfine_n")}}
    validate_tuning(ledger)
    recipes = baseline_recipes()
    apply_selected_trials(recipes, ledger)
    assert recipes["yolo26n"]["training"]["lr0"] == .0005
    assert recipes["dfine_n"]["training"]["head_lr"] == .0006
    ledger["trials"]["dfine_n"] = []
    with pytest.raises(ValueError, match="same completed tuning"):
        validate_tuning(ledger)
    ledger["test_split_used"] = True
    with pytest.raises(ValueError, match="no test data"):
        validate_tuning(ledger)


def test_final_phase_freezes_confidence_before_once_only_test(tmp_path, truth):
    metadata = tmp_path / "run_metadata.json"
    metadata.write_text(json.dumps({"completed_epochs": 100}))
    calls = []
    def collect(split):
        calls.append(split)
        if split == "test":
            saved = json.loads(metadata.read_text())
            assert saved["validation_confidence_selection"]["confidence"] == .2
            assert saved["test_evaluation"]["status"] == "started"
        return truth, [prediction(.2 if split == "val" else .8)]
    result = final_evaluation(collect, metadata)
    assert calls == ["val", "test"] and result["selected_confidence"] == .2
    saved = json.loads(metadata.read_text())
    assert saved["test_evaluation"]["inference_count"] == 1
    with pytest.raises(FileExistsError):
        final_evaluation(collect, metadata)
    assert calls == ["val", "test"]


def test_final_phase_never_opens_test_before_minimum(tmp_path):
    path = tmp_path / "run_metadata.json"
    path.write_text('{"completed_epochs": 99}')
    with pytest.raises(ValueError, match="completed controlled"):
        final_evaluation(lambda split: pytest.fail("Inference should not run"), path)


def test_result_aggregation_sample_std_and_partial_groups(frozen):
    path, m = frozen
    for seed, ap in ((0, .2), (1, .4)):
        job = next(j for j in m["matrix"] if j["model"] == "yolo26n" and j["colour"] == "rgb" and j["seed"] == seed)
        run = path.parent / job["run_id"]
        run.mkdir()
        (run / "run_metadata.json").write_text(json.dumps({"manifest_sha256": m["manifest_sha256"],
            "recipe_sha256": job["recipe_sha256"], "test_evaluation": {"status": "complete"}, "completed_epochs": 100,
            "best_metric": .5, "training_seconds": 12., "optimisation": {"physical_batch_size": 16,
            "gradient_accumulation": 1, "effective_batch_size": 16}, "final_test_metrics": {
                "mAP50-95": ap, "mAP50": ap, "precision": .8, "recall": .9, "F1": .85, "selected_confidence": .2}}))
    result = json.loads(aggregate_results(path).read_text())
    summary = result["summary"][0]
    assert not result["complete"] and summary["completed_seeds"] == 2
    assert summary["statistics"]["test_mAP50-95"]["mean"] == pytest.approx(.3)
    assert summary["statistics"]["test_mAP50-95"]["std"] == pytest.approx(np.std([.2, .4], ddof=1))
    assert (path.parent / "comparison_per_seed.csv").is_file()


@pytest.mark.parametrize("colour", ["rgb", "hsv"])
def test_real_yolo_training_loader_uses_same_geometry_then_colour(source, tmp_path, colour):
    from ultralytics.cfg import get_cfg
    from naartjie_cv.yolo_detection import ControlledDetectionTrainer, convert_batch_representation
    pair = prepare_detector_pairs(source, tmp_path / "cache")
    trainer = ControlledDetectionTrainer.__new__(ControlledDetectionTrainer)
    training = baseline_recipes()["yolo26n"]["training"]
    trainer.args = get_cfg(overrides={k: v for k, v in training.items() if v is not None})
    trainer.data = {"names": {0: "fruit"}, "nc": 1, "channels": 3}
    trainer.model = torch.nn.Module()
    trainer.model.stride = torch.tensor([32])
    dataset = trainer.build_dataset(pair["dataset_identity"]["splits"]["train"]["path"], "train", batch=1)
    batch = dataset.collate_fn([dataset[0]])
    actual = convert_batch_representation(batch, colour)["img"].float() / 255
    expected_rgb = np.array([140, 30, 210], dtype=np.uint8).reshape(1, 1, 3)
    expected = expected_rgb[0, 0] if colour == "rgb" else cv2.cvtColor(expected_rgb, cv2.COLOR_RGB2HSV_FULL)[0, 0]
    np.testing.assert_allclose(actual[0, :, 320, 320], expected / 255, atol=1e-7)
    assert actual.shape == (1, 3, 640, 640)
    assert not dataset.rect


def test_yolo_common_ap_controls_fitness(truth, monkeypatch):
    from ultralytics.models.yolo.detect import DetectionValidator
    from naartjie_cv.yolo_detection import CommonDetectionValidator
    monkeypatch.setattr(DetectionValidator, "get_stats", lambda self: {"metrics/mAP50-95(B)": .01, "fitness": .01})
    validator = CommonDetectionValidator.__new__(CommonDetectionValidator)
    validator.protocol = {}
    validator.common_ground_truth, validator.prediction_records = truth, [prediction()]
    validator.common_seen, validator.common_image_ids = {1}, {"image": 1}
    result = validator.get_stats()
    assert result["fitness"] == result["metrics/mAP50-95(B)"] == pytest.approx(1)


def test_actual_official_geometry_scaled_once(source, tmp_path):
    from naartjie_cv.official_detection import ControlledLetterbox
    rgb = np.array(Image.open(source / "images/val/val.png"))
    target = {"boxes": torch.tensor([[18.25, 8, 54.75, 24]]), "area": torch.tensor([584.])}
    geometries, boxes = [], []
    for colour in ("rgb", "hsv"):
        image, transformed, _ = ControlledLetterbox(colour=colour)((Image.fromarray(rgb), target, None))
        assert image.size == (640, 640)
        expected = rgb[0, 0] if colour == "rgb" else cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV_FULL)[0, 0]
        np.testing.assert_array_equal(np.array(image)[320, 320], expected)
        geometries.append(transformed["letterbox_geometry"])
        boxes.append(transformed["boxes"])
    assert torch.equal(geometries[0], geometries[1]) and torch.equal(boxes[0], boxes[1])


@pytest.mark.parametrize("colour", ["rgb", "hsv"])
def test_official_resolved_loader_preserves_recipe_and_scales_once(source, tmp_path, colour):
    import sys
    upstream = Path(__file__).resolve().parents[1] / "third_party/D-FINE"
    sys.path.insert(0, str(upstream))
    from src.core import YAMLConfig
    from naartjie_cv.official_detection import install_official_geometry
    install_official_geometry()
    pair = prepare_detector_pairs(source, tmp_path / "cache")
    manifest = {"recipes": baseline_recipes(), "paired_dataset": pair, "dfine_root": str(upstream)}
    job = {"model": "dfine_n", "colour": colour, "seed": 0}
    directory = tmp_path / colour
    protocol = protocol_for(manifest, job, directory)
    config = official_config(manifest, job, directory, protocol)
    cfg = YAMLConfig(str(config), device="cpu", use_amp=False)
    for split in ("train", "val"):
        loader = cfg.train_dataloader if split == "train" else cfg.val_dataloader
        images, targets = next(iter(loader))
        expected = np.array([140, 30, 210], dtype=np.uint8).reshape(1, 1, 3)
        if colour == "hsv":
            expected = cv2.cvtColor(expected, cv2.COLOR_RGB2HSV_FULL)
        np.testing.assert_allclose(images[0, :, 320, 320], expected[0, 0] / 255, atol=1e-7)
        assert images.dtype == torch.float32 and images.shape == (1, 3, 640, 640)
        assert not loader.drop_last and loader.num_workers == 0
        assert "letterbox_geometry" in targets[0]
    assert cfg.yaml_cfg["epochs"] == 300
    assert cfg.yaml_cfg["lr_scheduler"]["milestones"] == [500]
    assert cfg.yaml_cfg["train_dataloader"]["collate_fn"]["stop_epoch"] == 280


def test_dashboard_has_frozen_detection_controls():
    from naartjie_cv.detector_dashboard import create_detector_dashboard
    view = create_detector_dashboard()
    text = view.children[0].value
    assert "Minimum 100" in text and "Patience 40" in text and "D-FINE-N" in text
    assert "colour" in text.lower() and "Nominal effective batch" in text
    assert view.children[5].children[1].disabled


def test_dashboard_console_stays_bounded_through_long_training():
    from naartjie_cv.detector_dashboard import ConsoleTail
    import ipywidgets as widgets
    output = widgets.Output()
    console = ConsoleTail(output)
    for i in range(1000):
        console.write(f"Epoch/batch {i}: " + "progress" * 100 + "\n")
    console.flush()
    assert len(output.outputs) == 1
    assert len(output.outputs[0]["text"]) <= 60000
    assert "Epoch/batch 999:" in output.outputs[0]["text"]


def test_yolo_strict_best_keeps_first_tie_and_records_actual_values(tmp_path, monkeypatch):
    from naartjie_cv.yolo_detection import ControlledDetectionTrainer
    from ultralytics.models.yolo.detect import DetectionTrainer
    from naartjie_cv.yolo_training import ValidationEarlyStopping
    trainer = ControlledDetectionTrainer.__new__(ControlledDetectionTrainer)
    trainer.protocol = {}
    trainer.metadata_path = tmp_path / "run_metadata.json"
    trainer.metadata_path.write_text("{}")
    trainer.save_dir = tmp_path
    trainer.last = tmp_path / "weights/last.pt"
    trainer.last.parent.mkdir()
    trainer.stopper = ValidationEarlyStopping(40, 100, 300)
    trainer.actual_updates, trainer.observed_accumulation, trainer.batch_size = 12, {1}, 16
    trainer.optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=.001)
    def native_save(self):
        self.last.write_bytes(str(self.epoch).encode())
        return True
    monkeypatch.setattr(DetectionTrainer, "save_model", native_save)
    for epoch, score in enumerate((.1, .3, .3, .2)):
        trainer.epoch, trainer.metrics = epoch, {"metrics/mAP50-95(B)": score}
        trainer.stopper(epoch + 1, score)
        trainer.save_model()
    assert (tmp_path / "weights/best_controlled.pt").read_bytes() == b"1"
    recorded = json.loads(trainer.metadata_path.read_text())
    assert recorded["best_epoch"] == 2 and recorded["completed_epochs"] == 4
    assert recorded["actual_optimizer_steps"] == 12
    assert recorded["observed_nominal_effective_batches"] == [16]
    assert recorded["final_learning_rates"] == [.001]


def test_official_native_fit_uses_common_validation_and_stops_at_100(source, tmp_path, monkeypatch):
    import sys
    upstream = Path(__file__).resolve().parents[1] / "third_party/D-FINE"
    sys.path.insert(0, str(upstream))
    from src.solver import det_solver
    from src.solver._solver import BaseSolver
    from naartjie_cv.official_detection import install_controlled_solver
    metadata = tmp_path / "run_metadata.json"
    metadata.write_text("{}")
    model = torch.nn.Identity()
    class Loader(list):
        batch_size, num_workers, drop_last = 1, 0, False
        dataset = SimpleNamespace(img_folder=source / "images")
    loader = Loader([(torch.zeros(1, 3, 640, 640), [{"image_path": str(source / "images/val/val.png"),
                                                 "letterbox_geometry": torch.tensor([1., 1., 0., 0.])}])])
    solver = det_solver.DetSolver(SimpleNamespace(epochs=300, checkpoint_freq=301, clip_max_norm=.1, print_freq=10,
                        yaml_cfg={"lr_scheduler": {"type": "MultiStepLR"}, "use_ema": False}))
    def setup(self):
        self.optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
        self.model, self.criterion, self.ema = model, model, None
        self.postprocessor = lambda output, sizes: [{"boxes": torch.zeros(0, 4), "scores": torch.zeros(0), "labels": torch.zeros(0)}]
        self.device, self.writer, self.use_wandb, self.evaluator, self.scaler = "cpu", None, False, None, None
        self.lr_scheduler, self.lr_warmup_scheduler = SimpleNamespace(step=lambda: None, get_last_lr=lambda: [.001]), None
        self.train_dataloader = SimpleNamespace(batch_size=1, num_workers=0, drop_last=False, set_epoch=lambda epoch: None,
                                               collate_fn=SimpleNamespace(stop_epoch=280, ema_restart_decay=.9999))
        self.val_dataloader, self.output_dir, self.last_epoch = loader, tmp_path, -1
    monkeypatch.setattr(BaseSolver, "train", setup)
    # Track these attributes so monkeypatch restores the native functions after installation.
    monkeypatch.setattr(det_solver.DetSolver, "fit", det_solver.DetSolver.fit)
    monkeypatch.setattr(det_solver, "evaluate", det_solver.evaluate)
    updates = []
    monkeypatch.setattr(det_solver, "train_one_epoch", lambda *args, **kwargs: updates.append(args[5]) or {})
    monkeypatch.setattr(det_solver, "stats", lambda cfg: (0, "bounded native-loop adapter test"))
    solver.state_dict = lambda: {"last_epoch": solver.last_epoch}
    install_controlled_solver({"rgb_yaml": str(source / "data.yaml"), "augmentation": {"colour": "off"}}, metadata)
    solver.fit()
    recorded = json.loads(metadata.read_text())
    assert len(updates) == recorded["completed_epochs"] == 100
    assert recorded["stop_reason"] == "early_stopping"
    assert recorded["best_epoch"] == 1 and recorded["best_metric"] == 0
    assert torch.load(tmp_path / "best.pth", weights_only=True)["last_epoch"] == 0
    assert torch.load(tmp_path / "last.pth", weights_only=True)["last_epoch"] == 99
    assert recorded["optimisation"]["gradient_accumulation"] == 1


@pytest.mark.parametrize("trajectory,expected", [(lambda epoch: 0., 100), (lambda epoch: epoch / 1000, 300),
                                                 (lambda epoch: .5 if epoch < 90 else .6, 130)])
def test_both_native_stopping_adapters_share_minimum_patience_and_metric(trajectory, expected):
    from naartjie_cv.yolo_training import ValidationEarlyStopping
    from naartjie_cv.training_policy import ValidationStoppingState
    # Official solver uses the same state directly; YOLO uses its native interface.
    official = ValidationStoppingState(300, 100, 40)
    yolo = ValidationEarlyStopping(40, 100, 300)
    for epoch in range(1, 301):
        stop = official.observe(epoch, trajectory(epoch))
        assert yolo(epoch, trajectory(epoch)) == stop
        if stop:
            break
    assert epoch == expected
    assert official.best_epoch == yolo.state.best_epoch
