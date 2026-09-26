"""Published configuration and upstream dataset layout regressions."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from spacergbbenchmark.datasets.prepare import Builder, prepare_swisscube


def test_published_mesh_patterns_use_the_builder_object_id(tmp_path):
    root = Path(__file__).resolve().parents[1]
    configs = json.loads((root / "configs/datasets/native.json").read_text())["datasets"]
    for name, cfg in configs.items():
        mesh = tmp_path / Path(cfg["mesh"]).name.format(object_id=2)
        mesh.write_text("mesh fixture")
        cfg["mesh"] = str(tmp_path / Path(cfg["mesh"]).name)
        builder = Builder(name, cfg, tmp_path / name)
        builder.add("test", 2, [], {}, 8, 8, np.eye(3))
        assert builder.streams


@pytest.mark.parametrize("nested", [False, True])
def test_swisscube_preparation_accepts_official_and_flat_layouts(tmp_path, nested):
    root = tmp_path / "raw"
    record = {"cam_R_m2c": np.eye(3).reshape(-1).tolist(), "cam_t_m2c": [0, 0, 1000]}
    for number in range(400, 500):
        seq = root / f"seq_{number:06d}"
        if nested:
            seq /= "000000"
        (seq / "rgb").mkdir(parents=True)
        (seq / "mask_visib").mkdir()
        Image.new("RGB", (8, 8)).save(seq / "rgb/000000.jpg")
        Image.new("L", (8, 8), 255).save(seq / "mask_visib/000000_000000.png")
        (seq / "scene_camera.json").write_text(
            json.dumps({"0": {"cam_K": np.eye(3).reshape(-1).tolist()}})
        )
        (seq / "scene_gt.json").write_text(json.dumps({"0": [record]}))
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_swiss_boxes.py"
    spec = importlib.util.spec_from_file_location("prepare_swiss_boxes", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    boxes = tmp_path / "boxes"
    module.prepare(root, boxes)
    mesh = tmp_path / "mesh.ply"
    mesh.write_text("mesh fixture")
    builder = Builder(
        "swisscube",
        {
            "root": str(root),
            "boxes": str(boxes),
            "mesh": str(mesh),
            "localization_policy": "annotation-conditioned",
        },
        tmp_path / "prepared",
    )
    prepare_swisscube(builder)
    assert len(builder.streams) == len(builder.targets) == 100
    for entry in builder.streams:
        seq = json.loads((builder.output / entry["manifest"]).read_text())
        assert Path(seq["frames"][0]["slam_mask"]).is_file()


@pytest.mark.parametrize("units,scale", [("m", 1), ("mm", 1000)])
@pytest.mark.parametrize("custom_calibration", [False, True])
def test_shirt_preparation_resolves_the_published_mesh_pattern(
    tmp_path, units, scale, custom_calibration
):
    import trimesh

    from spacergbbenchmark.datasets.prepare import prepare_shirt

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "camera.json").write_text(
        json.dumps({"cameraMatrix": [[40, 0, 16], [0, 40, 12], [0, 0, 1]], "Nu": 32, "Nv": 24})
    )
    labels = [
        {"q_vbs2tango_true": [1, 0, 0, 0], "r_Vo2To_vbs_true": [0, 0, 2], "filename": "000000.png"}
    ]
    for trajectory in ("roe1", "roe2"):
        (raw / trajectory).mkdir()
        (raw / trajectory / f"{trajectory}.json").write_text(json.dumps(labels))
    calibration = raw / "calibration.json"
    calibration.write_text(
        json.dumps({"R_align": np.eye(3).tolist(), "delta": [0, 0, 0], "convention": "H2"})
    )
    trimesh.creation.box(extents=np.array([0.1, 0.1, 0.1]) * scale).export(
        raw / "object_000001.ply"
    )
    builder = Builder(
        "shirt",
        {
            "root": str(raw),
            "mesh": str(raw / "object_{object_id:06d}.ply"),
            "mesh_units": units,
            **({"calibration": str(calibration)} if custom_calibration else {}),
            "localization_policy": "annotation-conditioned",
        },
        tmp_path / "prepared",
    )
    prepare_shirt(builder)
    assert len(builder.streams) == len(builder.targets) == 4
    sequence = json.loads((builder.output / builder.streams[0]["manifest"]).read_text())
    np.testing.assert_allclose(
        sequence["frames"][0]["box"], [14.56410256, 10.56410256, 17.43589744, 13.43589744]
    )
    cal = json.loads(Path(builder.config["calibration"]).read_text())
    target = builder.targets[0][3]
    np.testing.assert_allclose(target[:3, :3], cal["R_align"])
    np.testing.assert_allclose(
        target[:3, 3], np.array([0, 0, 2]) - np.asarray(cal["R_align"]) @ cal["delta"]
    )


@pytest.mark.parametrize("custom_calibration", [False, True])
def test_relocated_dataset_config_loads_calibration(tmp_path, monkeypatch, custom_calibration):
    from spacergbbenchmark.datasets import prepare as preparation

    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/datasets/native.json").read_text())
    expected = json.loads((root / "protocols/shirt_calibration.json").read_text())
    if custom_calibration:
        config["datasets"]["shirt"]["calibration"] = "custom.json"
        expected = {"R_align": np.eye(3).tolist(), "delta": [0, 0, 0], "convention": "H2"}
    loaded = []

    def read_calibration(builder):
        loaded.append(json.loads(builder.source(builder.config["calibration"]).read_text()))

    monkeypatch.setattr(preparation, "prepare_shirt", read_calibration)
    monkeypatch.setattr(Builder, "finish", lambda self: {"targets": 0})
    monkeypatch.setenv("SPACERGB_DATA_ROOT", str(tmp_path / "data"))
    for index, name in enumerate(("configs/data.local.json", "elsewhere/nested/data.json")):
        path = tmp_path / name
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(config))
        if custom_calibration:
            (path.parent / "custom.json").write_text(json.dumps(expected))
        preparation.prepare(path, tmp_path / f"prepared{index}", ["shirt"])
    assert loaded == [expected, expected]
