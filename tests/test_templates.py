import importlib.util
from pathlib import Path

import pytest


def test_cookiecutter_fixture_runs_through_all_dataset_interfaces(tmp_path):
    from cookiecutter.main import cookiecutter

    from spacergbbenchmark.datasets import Dataset
    from spacergbbenchmark.demo import make_fixture
    from spacergbbenchmark.evaluation.score import score
    from spacergbbenchmark.schema import write_predictions

    template = Path(__file__).resolve().parents[1] / "templates/model_adapter"
    if not template.exists():
        pytest.skip("Source template integration requires the source distribution")
    root = Path(
        cookiecutter(
            str(template),
            no_input=True,
            output_dir=str(tmp_path),
            extra_context={
                "adapter_name": "toy_pose",
                "package_name": "toy_pose",
                "class_name": "ToyPose",
                "implementation": "fixture",
            },
        )
    )
    spec = importlib.util.spec_from_file_location("toy_pose", root / "src/toy_pose/__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ["spark", "ycbv", "swisscube", "shirt"]:
        manifest = make_fixture(tmp_path / name, name, frames=2)
        dataset = Dataset(manifest)
        seq = dataset.sequence(dataset.streams[0])
        model = module.ToyPose({}, seq, tmp_path / "out")
        model.start()
        predictions = [model.step(i) for i in range(len(seq))]
        model.close()
        destination = tmp_path / "predictions" / name
        write_predictions(destination / "fixture.npz", [0, 1], predictions)
        result = score(manifest, destination, tmp_path / "scores" / name, "toy_pose")
        assert result["valid"] == result["targets"] == 2
        assert result["E_p"] == 0
