"""Small synthetic fixtures exercise the complete CLI without benchmark assets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .io import atomic_json, atomic_npz, sha256


def make_fixture(root, name="spark", frames=8):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    mesh = root / "object.ply"
    mesh.write_text(
        "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n"
    )
    points = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]], dtype=float)
    np.save(root / "points.npy", points)
    rows = []
    for i in range(frames):
        image = root / f"{i:06d}.png"
        Image.new("RGB", (32, 24), color=(i * 10 % 255, 40, 80)).save(image)
        rows.append(
            dict(id=i, image=str(image.absolute()), target=True, box=[8, 5, 24, 20], mask=None)
        )
    sequence = dict(
        schema="spacergbbenchmark.sequence.v1",
        dataset=name,
        id="fixture",
        object_id=1,
        width=32,
        height=24,
        K=[[40, 0, 16], [0, 40, 12], [0, 0, 1]],
        mesh=str(mesh.absolute()),
        mesh_units="m",
        raw_to_object=np.eye(4).tolist(),
        object_frame="fixture",
        localization_policy="synthetic fixture boxes",
        frames=rows,
    )
    atomic_json(root / "sequence.json", sequence)
    truth = np.repeat(np.eye(4)[None], frames, axis=0)
    truth[:, 2, 3] = 2
    atomic_npz(
        root / "truth.npz",
        streams=np.array(["fixture"] * frames),
        frames=np.arange(frames),
        objects=np.ones(frames, dtype=int),
        poses=truth,
    )
    cfg = dict(
        schema="spacergbbenchmark.dataset.v1",
        dataset=name,
        scope="fixture",
        protocol="fixture-v1",
        object_frame="fixture",
        expected_targets=frames,
        localization_policy="synthetic fixture boxes",
        streams=[
            dict(id="fixture", manifest="sequence.json", sha256=sha256(root / "sequence.json"))
        ],
        truth=dict(path="truth.npz", sha256=sha256(root / "truth.npz")),
        mesh_points={"1": dict(path="points.npy", sha256=sha256(root / "points.npy"))},
    )
    atomic_json(root / "dataset.json", cfg)
    return root / "dataset.json"


def demo(output):
    from .evaluation.report import report
    from .runner.engine import run_suite

    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError(output)
    datasets = {
        name: str(make_fixture(output / "data" / name, name))
        for name in ("spark", "ycbv", "swisscube", "shirt")
    }
    pose = np.eye(4)
    pose[2, 3] = 2.1
    suite = dict(
        datasets=datasets,
        models={"fixture": dict(adapter="fixture", settings={"pose": pose.tolist()})},
    )
    result = run_suite(suite, output / "run")
    result.update(report(output / "run"))
    return result
