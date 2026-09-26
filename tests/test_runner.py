import pytest

from spacergbbenchmark.datasets import Dataset
from spacergbbenchmark.demo import make_fixture
from spacergbbenchmark.io import read_json
from spacergbbenchmark.runner.engine import run_suite, sequence_job


def test_resume_reuses_verified_completed_sequence(tmp_path):
    manifest = make_fixture(tmp_path / "data")
    dataset = Dataset(manifest)
    output = tmp_path / "job"
    first = sequence_job(dataset, dataset.streams[0], "fixture", {}, output, None)
    second = sequence_job(dataset, dataset.streams[0], "fixture", {}, output, None)
    assert first == second
    assert len(list(output.glob("attempt-*"))) == 1


def test_changed_pixels_invalidate_resume(tmp_path):
    manifest = make_fixture(tmp_path / "data")
    dataset = Dataset(manifest)
    output = tmp_path / "job"
    sequence_job(dataset, dataset.streams[0], "fixture", {}, output, None)
    image = manifest.parent / "000000.png"
    image.write_bytes(image.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed"):
        sequence_job(dataset, dataset.streams[0], "fixture", {}, output, None)


def test_failed_sequence_restarts_from_zero(tmp_path, monkeypatch):
    from spacergbbenchmark.models.fixture import FixtureModel

    manifest = make_fixture(tmp_path / "data", frames=5)
    dataset = Dataset(manifest)
    output = tmp_path / "job"
    native = FixtureModel.step
    visited = []

    def fail_once(self, index):
        visited.append(index)
        if index == 2:
            raise RuntimeError("interruption")
        return native(self, index)

    monkeypatch.setattr(FixtureModel, "step", fail_once)
    with pytest.raises(RuntimeError):
        sequence_job(dataset, dataset.streams[0], "fixture", {}, output, None)
    assert read_json(output / "state.json")["status"] == "failed"

    def good(self, index):
        visited.append(index)
        return native(self, index)

    monkeypatch.setattr(FixtureModel, "step", good)
    sequence_job(dataset, dataset.streams[0], "fixture", {}, output, None)
    assert visited == [0, 1, 2, 0, 1, 2, 3, 4]
    assert len(list(output.glob("attempt-*"))) == 2


def test_suite_continues_independent_jobs_after_failure(tmp_path):
    manifest = make_fixture(tmp_path / "data")
    config = dict(
        datasets={"spark": str(manifest)},
        models={
            "good": dict(adapter="fixture", settings={}),
            "bad": dict(adapter="fixture", settings={"fail_at": 2}),
        },
    )
    result = run_suite(config, tmp_path / "run", jobs=2)
    assert result["status"] == "failed"
    good = read_json(tmp_path / "run/metrics/good/spark/metrics.json")
    bad = read_json(tmp_path / "run/metrics/bad/spark/metrics.json")
    assert good["valid"] == 8
    assert bad["valid"] == 0 and bad["targets"] == 8


def test_fixture_cannot_claim_fresh_fps(tmp_path):
    dataset = Dataset(make_fixture(tmp_path / "data"))
    with pytest.raises(ValueError, match="fresh streaming"):
        sequence_job(
            dataset, dataset.streams[0], "fixture", {}, tmp_path / "job", None, timing=True
        )


def test_lost_completed_archive_is_recomputed(tmp_path):
    dataset = Dataset(make_fixture(tmp_path / "data"))
    output = tmp_path / "job"
    sequence_job(dataset, dataset.streams[0], "fixture", {}, output, None)
    (output / "predictions.npz").unlink()
    result = sequence_job(dataset, dataset.streams[0], "fixture", {}, output, None)
    assert result["status"] == "complete" and result["attempt"] == "attempt-002"


def test_sustained_timing_excludes_setup_but_includes_recovery(tmp_path, monkeypatch):
    from spacergbbenchmark.models.fixture import FixtureModel
    from spacergbbenchmark.runner import engine

    monkeypatch.setenv("SPACERGB_LOCK_ROOT", str(tmp_path / "locks"))

    dataset = Dataset(make_fixture(tmp_path / "data", frames=6))
    clock = [0.0]
    original = FixtureModel.step

    def setup(self):
        clock[0] += 80.0

    def step(self, index):
        clock[0] += 10.0 if index == 4 else 2.0
        return original(self, index)

    monkeypatch.setattr(FixtureModel, "measurement_kind", "fresh_inference")
    monkeypatch.setattr(FixtureModel, "start", setup)
    monkeypatch.setattr(FixtureModel, "step", step)
    monkeypatch.setattr(engine.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(engine.os, "getloadavg", lambda: (0, 0, 0))
    sequence_job(
        dataset, dataset.streams[0], "fixture", {}, tmp_path / "job", None, warmup=3, timing=True
    )
    measured = read_json(tmp_path / "job/timing.json")
    assert measured["setup_seconds"] == 80
    assert measured["warmup_seconds"] == 6
    assert measured["seconds"] == 14
    assert measured["fps"] == pytest.approx(3 / 14)
    assert measured["timed_valid_outputs"] == 3


def test_resume_rejects_changed_scoring_protocol(tmp_path):
    from spacergbbenchmark.io import atomic_json

    manifest = make_fixture(tmp_path / "data")
    config = dict(
        datasets={"spark": str(manifest)}, models={"fixture": dict(adapter="fixture", settings={})}
    )
    run_suite(config, tmp_path / "run")
    changed = read_json(manifest)
    changed["metric_calibration"] = {"changed": True}
    atomic_json(manifest, changed)
    with pytest.raises(ValueError, match="scoring protocol changed"):
        run_suite(config, tmp_path / "run", resume=True)


@pytest.mark.parametrize("location", ["default", "environment", "explicit"])
def test_resource_lease_location_and_exclusion(tmp_path, monkeypatch, location):
    import fcntl

    from spacergbbenchmark.runner import resources

    monkeypatch.setattr(resources.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.delenv("SPACERGB_LOCK_ROOT", raising=False)
    expected = tmp_path / "spacergbbenchmark-locks"
    explicit = None
    if location in ("environment", "explicit"):
        expected = tmp_path / "environment"
        monkeypatch.setenv("SPACERGB_LOCK_ROOT", str(expected))
    if location == "explicit":
        explicit = expected = tmp_path / "explicit"

    with resources.lease(lock_root=explicit):
        with (expected / "timing.lock").open("a+") as competing:
            with pytest.raises(BlockingIOError):
                fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (expected / "timing.lock").open("a+") as competing:
        fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(competing, fcntl.LOCK_UN)
