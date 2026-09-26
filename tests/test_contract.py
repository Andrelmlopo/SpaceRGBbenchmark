import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from spacergbbenchmark.datasets import Dataset
from spacergbbenchmark.demo import make_fixture
from spacergbbenchmark.evaluation.importer import import_predictions
from spacergbbenchmark.evaluation.metrics import add_s, metric_values
from spacergbbenchmark.evaluation.score import score
from spacergbbenchmark.io import atomic_json, atomic_npz, read_json
from spacergbbenchmark.schema import Prediction, read_predictions, write_predictions


def test_metric_units_and_rotation():
    truth, prediction = np.eye(4), np.eye(4)
    truth[2, 3] = 2
    prediction[2, 3] = 2.2
    prediction[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    result = metric_values(prediction, truth)
    assert result["E_t_m"] == pytest.approx(0.2)
    assert result["E_q_deg"] == pytest.approx(90)
    assert result["E_p"] == pytest.approx(np.pi / 2 + 0.1)


def test_shirt_origin_correction():
    align = Rotation.from_euler("z", 15, degrees=True).as_matrix()
    delta = np.array([0.1, 0.2, 0.3])
    P, G = np.eye(4), np.eye(4)
    P[2, 3], G[2, 3] = 2.3, 2
    cal = dict(R_align=align.tolist(), delta=delta.tolist())
    result = metric_values(P, G, cal)
    assert result["E_p"] == pytest.approx(0.3 / np.linalg.norm(G[:3, 3] + delta))


def test_adds_translation():
    G, P = np.eye(4), np.eye(4)
    P[2, 3] = 0.1
    assert add_s(P, G, [[0, 0, 0], [1, 0, 0]]) == pytest.approx(0.1)


def test_missing_predictions_preserve_denominator(tmp_path):
    manifest = make_fixture(tmp_path / "data", frames=4)
    predictions = tmp_path / "predictions"
    pose = np.eye(4)
    pose[2, 3] = 2
    write_predictions(predictions / "fixture.npz", [0], [Prediction(pose)])
    result = score(manifest, predictions, tmp_path / "score", "partial")
    assert result["targets"] == 4
    assert result["valid"] == 1
    assert result["coverage"] == 0.25
    assert result["E_p"] == 0


def test_invalid_pose_never_marked_valid(tmp_path):
    P = np.eye(4)
    P[0, 0] = 2
    write_predictions(tmp_path / "p.npz", [5], [Prediction(P)])
    assert not read_predictions(tmp_path / "p.npz")["valid"][0]


def test_duplicate_frame_rejected(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        write_predictions(tmp_path / "p.npz", [1, 1], [Prediction(), Prediction()])


def test_partial_swiss_never_full(tmp_path):
    path = make_fixture(tmp_path / "swiss", name="swisscube")
    cfg = read_json(path)
    cfg["scope"] = "full"
    atomic_json(path, cfg)
    with pytest.raises(ValueError, match="frozen partition"):
        Dataset(path).validate()


def test_manifest_tampering_rejected(tmp_path):
    path = make_fixture(tmp_path / "data")
    sequence = path.parent / "sequence.json"
    sequence.write_text(sequence.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        Dataset(path).validate()


def test_truth_not_in_inference_manifest(tmp_path):
    path = make_fixture(tmp_path / "data")
    seq = read_json(path.parent / "sequence.json")
    assert "truth" not in seq
    assert all("pose" not in f for f in seq["frames"])


def test_import_millimetres_and_explicit_object_frame(tmp_path):
    path = make_fixture(tmp_path / "data", frames=1)
    raw = tmp_path / "raw"
    raw.mkdir()
    P = np.eye(4)
    P[2, 3] = 2000
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
    atomic_npz(raw / "fixture.npz", frames=np.array([0]), poses=P[None])
    import_predictions(path, raw, tmp_path / "out", units="mm", source_to_object=T)
    actual = read_predictions(tmp_path / "out/fixture.npz")["poses"][0]
    assert actual[2, 3] == 2
    np.testing.assert_allclose(actual[:3, :3], T[:3, :3].T)


def test_unknown_prediction_frame_rejected(tmp_path):
    manifest = make_fixture(tmp_path / "data")
    write_predictions(tmp_path / "p/fixture.npz", [100], [Prediction(np.eye(4))])
    with pytest.raises(ValueError, match="unknown frame"):
        score(manifest, tmp_path / "p", tmp_path / "score", "bad")


def test_archived_scoring_does_not_require_original_rgb(tmp_path):
    manifest = make_fixture(tmp_path / "data", frames=1)
    (manifest.parent / "000000.png").unlink()
    result = score(manifest, tmp_path / "missing", tmp_path / "score", "archived")
    assert result["targets"] == 1 and result["valid"] == 0
