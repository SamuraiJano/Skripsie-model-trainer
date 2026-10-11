"""Check model input tensors, paired dataset integrity and recipe parity."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from naartjie_cv.colour import prepare_colour_dataset, verify_paired_datasets
from naartjie_cv.datasets import prepare_dfine_seg_dataset
from naartjie_cv.runtime import seed_training
from naartjie_cv.train import TrainingOptions, _dfine_seg_config, final_dfine_seg_options, final_yolo_options, dfine_seg_loader_smoke
from scripts.model_trainer import parser
from test_training_export import make_yolo_dataset


def hsv_fixture(tmp_path, task, order="HSV"):
    import shutil
    rgb = make_yolo_dataset(tmp_path, task)
    source = np.tile(np.array([140, 30, 210], dtype=np.uint8), (640, 640, 1))
    expected = cv2.cvtColor(source[..., ::-1], cv2.COLOR_BGR2HSV_FULL)[0, 0]
    assert expected[0] > 179
    for image in rgb.rglob("*.png"):
        Image.fromarray(source).save(image)
    root = tmp_path / "historical_hsv"
    shutil.copytree(rgb, root)
    for image in root.rglob("*.png"):
        values = expected if order == "HSV" else expected[::-1]
        Image.fromarray(np.tile(values, (640, 640, 1))).save(image)
    data = yaml.safe_load((root / "data.yaml").read_text())
    data.update(colour_space="hsv", hsv_storage_order=order, source_data_yaml=str(rgb / "data.yaml"))
    (root / "data.yaml").write_text(yaml.safe_dump(data))
    return root, expected


@pytest.mark.parametrize("task", ["detect", "segment"])
@pytest.mark.parametrize("order", ["HSV", "VSH"])
@pytest.mark.parametrize("colour", ["rgb", "hsv"])
def test_yolo_actual_training_tensor_has_h_s_v(tmp_path, task, order, colour):
    from ultralytics.cfg import get_cfg
    from ultralytics.models.yolo.detect import DetectionTrainer
    from naartjie_cv.yolo_training import MaskSelectionTrainer

    root, expected = hsv_fixture(tmp_path, task, order)
    if colour == "rgb":
        root = Path(yaml.safe_load((root / "data.yaml").read_text())["source_data_yaml"]).parent
        expected = np.array(Image.open(root / "images/train/train.png"))[0, 0]
    canonical = prepare_colour_dataset(root, task, tmp_path / "cache")
    options = asdict(final_yolo_options(task))
    options.pop("min_epochs_before_early_stop")  # Our gate is installed separately from native YOLO args.
    options.update(options.pop("extra_args"))
    options.update(task=task, imgsz=640, seed=0)
    trainer_type = DetectionTrainer if task == "detect" else MaskSelectionTrainer
    trainer = trainer_type.__new__(trainer_type)
    trainer.args = get_cfg(overrides=options)
    trainer.data = {"names": {0: "Naartjie"}, "nc": 1, "channels": 3}
    trainer.model = torch.nn.Module()
    trainer.model.stride = torch.tensor([32])
    trainer.device = torch.device("cpu")
    trainer.stride = 32
    seed_training()
    dataset = trainer.build_dataset(str(canonical / "images/train"), mode="train", batch=1)
    batch = dataset.collate_fn([dataset[0]])
    tensor = trainer.preprocess_batch(batch)["img"]
    torch.testing.assert_close(tensor[0, :, 320, 320], torch.tensor(expected.astype(float) / 255, dtype=torch.float32))
    assert tensor.shape == (1, 3, 640, 640)


@pytest.mark.parametrize("task", ["detect", "segment"])
def test_hsv_conversion_preserves_source_splits_labels_and_pixels(tmp_path, task):
    rgb = make_yolo_dataset(tmp_path, task)
    hsv = prepare_colour_dataset(rgb, task, tmp_path / "cache", "hsv")
    verify_paired_datasets(rgb / "data.yaml", hsv / "data.yaml")
    for split in ("train", "val"):
        expected = cv2.cvtColor(cv2.imread(str(rgb / f"images/{split}/{split}.png")), cv2.COLOR_BGR2HSV_FULL)
        np.testing.assert_array_equal(np.array(Image.open(hsv / f"images/{split}/{split}.png")), expected)
        assert (rgb / f"labels/{split}/{split}.txt").read_bytes() == (hsv / f"labels/{split}/{split}.txt").read_bytes()
    assert prepare_colour_dataset(rgb, task, tmp_path / "cache", "hsv") == hsv
    (rgb / "labels/train/train.txt").write_text("0 0.5 0.5 0.25 0.25\n" if task == "detect" else "0 0.1 0.1 0.9 0.1 0.9 0.9\n")
    with pytest.raises(ValueError, match="annotation_sha256"):
        verify_paired_datasets(rgb / "data.yaml", hsv / "data.yaml")
    assert prepare_colour_dataset(rgb, task, tmp_path / "cache", "hsv") != hsv


def test_legacy_vsh_dataset_is_checked_and_corrected(tmp_path):
    rgb = make_yolo_dataset(tmp_path, "detect")
    rgb = rgb.rename(tmp_path / "dataset_rgb")
    hsv = tmp_path / "dataset_hsv"
    import shutil
    shutil.copytree(rgb, hsv)
    for image in hsv.rglob("*.png"):
        expected = cv2.cvtColor(cv2.imread(str(rgb / image.relative_to(hsv))), cv2.COLOR_BGR2HSV)
        cv2.imwrite(str(image), expected)  # legacy OpenCV save stores V,S,H
    canonical = prepare_colour_dataset(hsv, "detect", tmp_path / "cache")
    report = json.loads((canonical / "colour_preparation.json").read_text())
    assert report["source_storage_orders"] == ["VSH"]
    assert report["paired_source_verified"]
    image = canonical / "images/train/train.png"
    assert report["source_encodings"] == ["HSV_180"]
    np.testing.assert_array_equal(np.array(Image.open(image)), cv2.cvtColor(cv2.imread(str(rgb / "images/train/train.png")), cv2.COLOR_BGR2HSV_FULL))


def test_unpaired_hsv_provenance_is_rejected(tmp_path):
    root, _ = hsv_fixture(tmp_path, "detect")
    data = yaml.safe_load((root / "data.yaml").read_text())
    del data["source_data_yaml"]
    (root / "data.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="RGB source provenance"):
        prepare_colour_dataset(root, "detect", tmp_path / "cache")


@pytest.mark.parametrize("corruption", ["encoding", "pixels", "version"])
def test_old_or_altered_hsv_cache_is_regenerated(tmp_path, corruption):
    root, expected = hsv_fixture(tmp_path, "detect")
    rgb = Path(yaml.safe_load((root / "data.yaml").read_text())["source_data_yaml"]).parent
    hsv = prepare_colour_dataset(rgb, "detect", tmp_path / "cache", "hsv")
    image = hsv / "images/train/train.png"
    report_path = hsv / "colour_preparation.json"
    report = json.loads(report_path.read_text())
    if corruption == "pixels":
        old = cv2.cvtColor(cv2.imread(str(rgb / "images/train/train.png")), cv2.COLOR_BGR2HSV)
        Image.fromarray(old).save(image)
    else:
        report["hsv_encoding" if corruption == "encoding" else "cache_version"] = "HSV_180" if corruption == "encoding" else "hsv-v1"
        report_path.write_text(json.dumps(report))
    assert prepare_colour_dataset(rgb, "detect", tmp_path / "cache", "hsv") == hsv
    np.testing.assert_array_equal(np.array(Image.open(image))[320, 320], expected)
    report = json.loads(report_path.read_text())
    assert report["hsv_encoding"] == "HSV_FULL" and report["hue_range"] == [0, 255]
    assert report["tensor_scaling"] == "divide_by_255" and report["tensor_channel_order"] == "HSV"


def test_resolved_config_diff_reports_every_unapproved_difference():
    from naartjie_cv.final_validation import configuration_differences
    rgb = {"data": "rgb", "optimizer": "AdamW", "lr0": 0.001, "seed": 0, "model": "yolo26n.pt"}
    hsv = {**rgb, "data": "hsv"}
    assert configuration_differences(rgb, hsv, "yolo") == {}
    for field, value in (("lr0", 0.002), ("seed", 1), ("model", "yolo26s.pt")):
        assert field in configuration_differences(rgb, {**hsv, field: value}, "yolo")


@pytest.mark.parametrize("task", ["detect", "segment"])
def test_yolo_cold_import_preserves_rgb_hsv_geometric_labels(tmp_path, task):
    # A separate process starts without Albumentations in sys.modules. Its
    # first lazy import must not shift the seeded geometry of only the RGB run.
    rgb = make_yolo_dataset(tmp_path, task)
    hsv = prepare_colour_dataset(rgb, task, tmp_path / "cache", "hsv")
    code = r'''
import sys
from pathlib import Path
from naartjie_cv.final_validation import yolo_training_batch, verify_batch_pair
assert 'albumentations' not in sys.modules
rgb = yolo_training_batch(Path(sys.argv[1]), sys.argv[3])
hsv = yolo_training_batch(Path(sys.argv[2]), sys.argv[3])
verify_batch_pair(rgb, hsv)
'''
    completed = subprocess.run([sys.executable, "-c", code, str(rgb), str(hsv), task],
                               capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_fruit_track_metadata_blocks_leakage_without_rewriting_splits(tmp_path):
    from naartjie_cv.experiment_tracking import dataset_split_integrity
    root = make_yolo_dataset(tmp_path, "detect")
    path = root / "data.yaml"
    assert dataset_split_integrity(path)["fruit_track_split_integrity"] == "NOT PROVABLE"
    payload = yaml.safe_load(path.read_text())
    payload["fruit_identity"] = {"train/train": "fruit-a", "val/val": "fruit-a"}
    path.write_text(yaml.safe_dump(payload))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Fruit/track identity crosses splits"):
        prepare_colour_dataset(root, "detect", tmp_path / "cache", "hsv")
    assert path.read_bytes() == before
    payload["fruit_identity"]["val/val"] = "fruit-b"
    path.write_text(yaml.safe_dump(payload))
    assert dataset_split_integrity(path)["fruit_track_split_integrity"] == "PASS"


def test_export_manifest_fruit_identity_is_checked_from_dataset_parent(tmp_path):
    from naartjie_cv.experiment_tracking import dataset_split_integrity
    root = make_yolo_dataset(tmp_path, "detect")
    manifest = {"images": [{"split": "train", "image_uuid": "train", "track_id": "track-a"},
                           {"split": "val", "file_name": "images/val/val.png", "track_id": "track-a"}]}
    (tmp_path / "export_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Fruit/track identity crosses splits"):
        dataset_split_integrity(root / "data.yaml")


@pytest.mark.parametrize("task", ["detect", "segment"])
@pytest.mark.parametrize("colour", ["rgb", "hsv"])
def test_active_dfine_actual_training_tensor_has_h_s_v(tmp_path, task, colour):
    project = Path(__file__).resolve().parents[1]
    upstream = project / "third_party/D-FINE-seg"
    if not (upstream / "config.yaml").exists():
        pytest.skip("Optional D-FINE-seg checkout unavailable")
    root, expected = hsv_fixture(tmp_path, task, "VSH")
    if colour == "rgb":
        root = Path(yaml.safe_load((root / "data.yaml").read_text())["source_data_yaml"]).parent
        expected = np.array(Image.open(root / "images/train/train.png"))[0, 0]
    result = dfine_seg_loader_smoke(root, task, upstream, final_dfine_seg_options(task))
    assert result["tensor_shape"] == [1, 3, 640, 640]
    np.testing.assert_allclose(result["tensor_centre_channels"], expected / 255, atol=1e-7)


@pytest.mark.parametrize("colour", ["rgb", "hsv"])
def test_official_dfine_actual_loader_tensor_has_h_s_v(tmp_path, colour):
    project = Path(__file__).resolve().parents[1]
    upstream = project / "third_party/D-FINE"
    if not (upstream / "train.py").exists():
        pytest.skip("Optional official D-FINE checkout unavailable")
    from naartjie_cv.train import train_dfine_export
    root, expected = hsv_fixture(tmp_path, "detect", "VSH")
    if colour == "rgb":
        root = Path(yaml.safe_load((root / "data.yaml").read_text())["source_data_yaml"]).parent
        expected = np.array(Image.open(root / "images/train/train.png"))[0, 0]
    result = train_dfine_export(root, upstream, TrainingOptions(device="cpu", workers=0, output_dir=str(tmp_path / "runs")), execute=False)
    script = r'''
import sys, torch
from src.core import YAMLConfig
from naartjie_cv.runtime import seed_training
seed_training(0)
cfg = YAMLConfig(sys.argv[1])
image, target = cfg.train_dataloader.dataset[0]
import json
cx, cy = target['boxes'][0, :2]
torch.testing.assert_close(image[:,int(cy*640),int(cx*640)], torch.tensor(json.loads(sys.argv[2]),dtype=torch.float32)/255)
assert list(image.shape) == [3,640,640]
print('Official H,S,V verified')
'''
    completed = subprocess.run([sys.executable, "-c", script, result["config"], json.dumps(expected.tolist())], cwd=upstream, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("task", ["detect", "segment"])
def test_rgb_hsv_dfine_configs_differ_only_in_representation_and_paths(tmp_path, task):
    project = Path(__file__).resolve().parents[1]
    upstream = project / "third_party/D-FINE-seg"
    if not (upstream / "config.yaml").exists():
        pytest.skip("Optional D-FINE-seg checkout unavailable")
    rgb = make_yolo_dataset(tmp_path, task)
    hsv = prepare_colour_dataset(rgb, task, tmp_path / "cache", "hsv")
    configs = []
    options = final_dfine_seg_options(task)
    for colour, source in (("rgb", rgb), ("hsv", hsv)):
        prepared = prepare_dfine_seg_dataset(source, task, tmp_path / colour / "prepared")
        config = yaml.safe_load(_dfine_seg_config(upstream, tmp_path / colour, prepared, task, options).read_text())
        # Compare every setting, excluding representation and run/dataset paths.
        for name in ("exp_name", "exp"):
            config.pop(name)
        for name in ("root", "data_path", "path_to_test_data", "path_to_save", "debug_img_path", "eval_preds_path", "bench_img_path", "infer_path", "colour_space", "input_representation"):
            config["train"].pop(name)
        configs.append(config)
    assert configs[0] == configs[1]


@pytest.mark.parametrize("name", ["yolo-detect", "yolo-segment", "dfine-detect", "dfine-seg-detect", "dfine-seg-segment"])
def test_controlled_cli_defaults_and_colour_parity(name):
    upstream = ["--dfine-root", "upstream"] if name == "dfine-detect" else ["--dfine-seg-root", "upstream"] if name.startswith("dfine-seg") else []
    rgb = vars(parser().parse_args([name, "dataset", *upstream, "--colour-space", "rgb"]))
    hsv = vars(parser().parse_args([name, "dataset", *upstream, "--colour-space", "hsv"]))
    assert rgb.pop("colour_space") == "rgb" and hsv.pop("colour_space") == "hsv"
    assert rgb == hsv
    assert rgb["epochs"] == 300 and rgb["seed"] == 0 and rgb["imgsz"] == 640 and rgb["deterministic"]
    assert rgb["batch"] > 0
    assert rgb["workers"] == (0 if name.startswith("dfine-seg") else 4)
    if name.startswith("yolo"):
        assert rgb["batch"] == 16 and rgb["optimizer"] == "AdamW" and rgb["patience"] == 40
        assert rgb["min_epochs_before_early_stop"] == 100
        assert rgb["hsv_h"] == rgb["hsv_s"] == rgb["hsv_v"] == rgb["bgr"] == 0
        assert rgb["yolo_args_json"] == {"augmentations": []}
    elif name.startswith("dfine-seg"):
        assert rgb["early_stopping"] == 40
        assert rgb["min_epochs_before_early_stop"] == 100


def test_common_seeding_is_repeatable():
    import random
    seed_training()
    first = (random.random(), np.random.rand(), torch.rand(3))
    seed_training()
    second = (random.random(), np.random.rand(), torch.rand(3))
    assert first[:2] == second[:2]
    torch.testing.assert_close(first[2], second[2])
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark


def test_default_early_stopping_cannot_stop_before_100_epochs():
    from naartjie_cv.yolo_training import ValidationEarlyStopping
    stopper = ValidationEarlyStopping()
    assert not any(stopper(epoch, 0.0) for epoch in range(1, 100))


def test_dfine_transform_rng_is_seeded_independently_of_global_numpy():
    import albumentations as A
    from scripts.dfine_seg_runner import _install_reproducible_transforms

    class Dataset:
        def _init_augs(self, cfg):
            self.transform = A.Compose([A.HorizontalFlip(p=0.5)])
    module = SimpleNamespace(CustomDataset=Dataset)
    _install_reproducible_transforms(module)
    _install_reproducible_transforms(module)
    cfg = SimpleNamespace(train=SimpleNamespace(seed=0))
    a, b = Dataset(), Dataset()
    a._init_augs(cfg)
    np.random.seed(999)
    b._init_augs(cfg)
    image = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    for _ in range(10):
        np.testing.assert_array_equal(a.transform(image=image)["image"], b.transform(image=image)["image"])
    assert module.seed_worker.__name__ == "seed_dfine_worker"


@pytest.mark.parametrize("task", ["detect", "segment"])
def test_yolo_rgb_hsv_use_identical_train_arguments_and_checkpoint(tmp_path, monkeypatch, task):
    import ultralytics
    from naartjie_cv.train import train_yolo_export

    models, arguments = [], []
    class FakeYOLO:
        def __init__(self, checkpoint):
            models.append(checkpoint)
        def train(self, **kwargs):
            arguments.append(kwargs.copy())
            self.trainer = SimpleNamespace(save_dir=Path(kwargs["project"]) / kwargs["name"])
            return SimpleNamespace(results_dict={})
    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    root = make_yolo_dataset(tmp_path, task)
    for colour in ("rgb", "hsv"):
        train_yolo_export(root, task, TrainingOptions(device="cpu", workers=0,
                          colour_space=colour, output_dir=str(tmp_path / "runs"), run_name=colour,
                          yolo=final_yolo_options(task)))
    for value in arguments:
        assert value.pop("data")
        assert value.pop("name")
    assert arguments[0] == arguments[1]
    assert models[0] == models[1] == ("yolo26n.pt" if task == "detect" else "yolo26n-seg.pt")


@pytest.mark.parametrize("task", ["detect", "segment"])
def test_dashboard_colour_selection_preserves_all_training_controls(task):
    from naartjie_cv.notebook_ui import create_training_dashboard
    dashboard = create_training_dashboard()
    def descendants(widget):
        yield widget
        for child in getattr(widget, "children", ()):
            yield from descendants(child)
    fields = {control.description: control for control in descendants(dashboard) if getattr(control, "description", "")}
    fields["Task"].value = task
    before = {name: control.value for name, control in fields.items() if name != "Colour space" and hasattr(control, "value")}
    fields["Colour space"].value = "hsv"
    for name, value in before.items():
        assert fields[name].value == value, name


@pytest.mark.parametrize("task", ["dfine_seg_detect", "dfine_seg_segment"])
def test_dfine_dashboard_colour_selection_preserves_recipe(task):
    from naartjie_cv.dfine_notebook_ui import create_dfine_training_dashboard
    dashboard = create_dfine_training_dashboard()
    def descendants(widget):
        yield widget
        for child in getattr(widget, "children", ()):
            yield from descendants(child)
    fields = list(descendants(dashboard))
    task_control = next(control for control in fields if getattr(control, "description", "") == "Trainer")
    task_control.value = task
    selector = next(control for control in fields if getattr(control, "description", "") == "Colour space")
    before = [(control, control.value) for control in fields if getattr(control, "description", "") and hasattr(control, "value") and control is not selector]
    selector.value = "hsv"
    for control, value in before:
        assert control.value == value, control.description
