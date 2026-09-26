import numpy as np
import pytest
from PIL import Image

from spacergbbenchmark.datasets import Dataset, Sequence
from spacergbbenchmark.demo import make_fixture
from spacergbbenchmark.io import atomic_json, read_json, sha256
from spacergbbenchmark.runner.engine import sequence_job
from spacergbbenchmark.runner.resources import output_lease
from spacergbbenchmark.schema import read_predictions


def edit_sequence(manifest, change):
    path = manifest.parent / "sequence.json"
    config = read_json(path)
    change(config)
    atomic_json(path, config)
    ds = read_json(manifest)
    ds["streams"][0]["sha256"] = sha256(path)
    atomic_json(manifest, ds)
    return Sequence(path)


def test_missing_detection_does_not_carry_box(tmp_path):
    manifest = make_fixture(tmp_path / "data", frames=3)
    sequence = edit_sequence(manifest, lambda c: c["frames"][1].update(box=None))
    assert sequence.localization(0).any()
    assert sequence.localization(1) is None
    assert sequence.localization(2).any()


def test_colored_masks_need_declared_encoding(tmp_path):
    manifest = make_fixture(tmp_path / "data", frames=1)
    mask = np.zeros((24, 32, 3), np.uint8)
    mask[4:10, 5:12, 1] = 75
    path = tmp_path / "mask.png"
    Image.fromarray(mask).save(path)

    def change(c):
        c["frames"][0].update(mask=str(path), box=None)
        c["mask_encoding"] = "nonzero_any_channel"

    sequence = edit_sequence(manifest, change)
    assert sequence.localization(0).sum() == 42
    np.testing.assert_array_equal(sequence.bbox(0), [5, 4, 12, 10])


def test_rle_uses_object_detection_mask(tmp_path):
    coco = pytest.importorskip("pycocotools.mask")
    manifest = make_fixture(tmp_path / "data", frames=1)
    mask = np.zeros((24, 32), np.uint8, order="F")
    mask[4:10, 5:12] = 1
    rle = coco.encode(mask)
    rle["counts"] = rle["counts"].decode()
    sequence = edit_sequence(manifest, lambda c: c["frames"][0].update(mask_rle=rle))
    np.testing.assert_array_equal(sequence.localization(0), mask.astype(bool))


def test_independent_schedule_contains_only_declared_targets(tmp_path):
    manifest = make_fixture(tmp_path / "data", frames=5)
    edit_sequence(manifest, lambda c: [f.update(target=f["id"] in [1, 4]) for f in c["frames"]])
    ds = Dataset(manifest)
    sequence_job(ds, ds.streams[0], "fixture", {}, tmp_path / "job", None)
    assert read_predictions(tmp_path / "job/predictions.npz")["frames"].tolist() == [1, 4]


def test_increasing_variable_frame_ids_are_preserved(tmp_path):
    manifest = make_fixture(tmp_path / "data", frames=3)
    sequence = edit_sequence(
        manifest, lambda c: [f.update(id=i) for f, i in zip(c["frames"], [1, 7, 12])]
    )
    assert [f["id"] for f in sequence.frames] == [1, 7, 12]


def test_output_is_owned_by_one_writer(tmp_path):
    with output_lease(tmp_path / "job"):
        with pytest.raises(RuntimeError, match="owns this output"):
            with output_lease(tmp_path / "job"):
                pytest.fail("Two writers acquired one output")


def test_keyboard_interrupt_preserves_restartable_receipt(tmp_path, monkeypatch):
    from spacergbbenchmark.models.fixture import FixtureModel

    manifest = make_fixture(tmp_path / "data")
    ds = Dataset(manifest)
    original = FixtureModel.step

    def interrupt(self, index):
        if index == 1:
            raise KeyboardInterrupt()
        return original(self, index)

    monkeypatch.setattr(FixtureModel, "step", interrupt)
    with pytest.raises(KeyboardInterrupt):
        sequence_job(ds, ds.streams[0], "fixture", {}, tmp_path / "job", None)
    assert read_json(tmp_path / "job/state.json")["status"] == "interrupted"
    monkeypatch.setattr(FixtureModel, "step", original)
    result = sequence_job(ds, ds.streams[0], "fixture", {}, tmp_path / "job", None)
    assert result["status"] == "complete" and result["attempt"] == "attempt-002"


def test_giga_template_conversion_keeps_depth_units_and_object_frame(tmp_path):
    from spacergbbenchmark.assets import convert_giga
    from spacergbbenchmark.io import atomic_npz

    template = tmp_path / "pico"
    template.mkdir()
    poses = np.eye(4)[None]
    poses[0, 2, 3] = 2.5
    atomic_npz(
        template / "templates.npz",
        rgb=np.zeros((1, 2, 2, 3), np.uint8),
        depth=np.ones((1, 2, 2), np.float32) * 2.25,
        poses=poses,
        K=np.array([[572.4114, 0, 320], [0, 573.57043, 240], [0, 0, 1]]),
    )
    convert_giga(template, tmp_path / "giga")
    with Image.open(tmp_path / "giga/templates/custom/000001/000000_depth.png") as image:
        np.testing.assert_array_equal(np.asarray(image), np.full((2, 2), 2250))
    converted = np.load(tmp_path / "giga/templates/custom/object_poses/000001.npy")
    assert converted[0, 2, 3] == 2500
    np.testing.assert_array_equal(converted[0, :3, :3], np.eye(3))
    convert_giga(template, tmp_path / "giga")
