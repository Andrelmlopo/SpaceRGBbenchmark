from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .datasets import Dataset
from .io import atomic_json, read_json
from .runner.resources import lease


def prepare_templates(suite, dataset_name, model, objects=None, gpu=0):
    dataset = Dataset(suite["datasets"][dataset_name])
    settings = suite["models"][model]["settings"]
    if model not in {"picopose", "pico_hat", "gigapose_refined"}:
        raise ValueError("This template command supports PicoPose/Pico-HAT/GigaPose")
    seen, receipts = set(), []
    for stream in dataset.streams:
        sequence = dataset.sequence(stream)
        obj = sequence.object_id
        if obj in seen or objects and obj not in objects:
            continue
        seen.add(obj)
        local = dict(settings)
        local.update(settings.get("objects", {}).get(f"{dataset_name}:{obj}", {}))
        destination = Path(local["templates"])
        if destination.exists():
            from .io import sha256

            metadata = read_json(destination / "metadata.json")
            if metadata["mesh_sha256"] != sha256(sequence.file(sequence.config["mesh"])):
                raise ValueError("Existing templates belong to another mesh")
            if model == "gigapose_refined":
                convert_giga(destination, local["giga_root"], local.get("giga_dataset", "custom"))
            receipts.append(dict(object_id=obj, status="verified_existing", path=str(destination)))
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYOPENGL_PLATFORM="egl")
        if local.get("package_paths"):
            env["PYTHONPATH"] = os.pathsep.join(local["package_paths"])
        with lease(str(gpu)):
            subprocess.run(
                [
                    local["pico_python"],
                    "-m",
                    "pico_hat.cli",
                    "prepare",
                    "--mesh",
                    str(sequence.file(sequence.config["mesh"])),
                    "--units",
                    sequence.config["mesh_units"],
                    "--pico-source",
                    local["pico_source"],
                    "--out",
                    str(destination),
                ],
                env=env,
                check=True,
            )
        if model == "gigapose_refined":
            convert_giga(destination, local["giga_root"], local.get("giga_dataset", "custom"))
        receipts.append(dict(object_id=obj, status="prepared", path=str(destination)))
    return receipts


def convert_giga(templates, output, dataset="custom"):
    import numpy as np
    from PIL import Image

    output = Path(output)
    bank = output / "templates" / dataset
    from .io import sha256

    provenance = dict(
        format="spacergbbenchmark.giga-templates.v2",
        template_sha256=sha256(Path(templates) / "templates.npz"),
        template_pose_units="mm",
        object_frame="source mesh, no recentering",
    )
    if bank.exists():
        if read_json(output / "provenance.json") != provenance:
            raise ValueError("GigaPose template cache differs; select a fresh directory")
        return
    bank.mkdir(parents=True, exist_ok=False)
    images = bank / "000001"
    images.mkdir()
    (bank / "object_poses").mkdir()
    with np.load(Path(templates) / "templates.npz", allow_pickle=False) as archive:
        expected_K = np.array([[572.4114, 0, 320], [0, 573.57043, 240], [0, 0, 1]])
        if not np.allclose(archive["K"], expected_K, rtol=0, atol=1e-5):
            raise ValueError("Template intrinsics differ from the pinned GigaPose camera")
        for i, (rgb, depth) in enumerate(zip(archive["rgb"], archive["depth"])):
            alpha = ((depth > 0) * 255).astype(np.uint8)
            Image.fromarray(np.dstack([rgb, alpha])).save(images / f"{i:06d}.png")
            depth_mm = np.rint(depth * 1000)
            if not np.isfinite(depth_mm).all() or depth_mm.max() > 65535:
                raise ValueError("GigaPose uint16 template depth cannot represent this mesh scale")
            Image.fromarray(depth_mm.astype(np.uint16)).save(images / f"{i:06d}_depth.png")
        poses = archive["poses"].copy()
        poses[:, :3, 3] *= 1000
        np.save(bank / "object_poses/000001.npy", poses)
    atomic_json(output / dataset / "models/models_info.json", {"1": {}})
    atomic_json(
        output / "provenance.json",
        provenance,
    )


def mesh_samples(mesh_path, units, transform, destination, seed=20260908):
    """New deterministic 2000-point surface bank in benchmark object coordinates."""
    import numpy as np
    import trimesh

    from .io import sha256

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    rng = np.random.default_rng(seed)
    triangles = np.asarray(mesh.triangles)
    areas = np.asarray(mesh.area_faces)
    selected = rng.choice(len(triangles), size=2000, p=areas / areas.sum())
    u, v = np.sqrt(rng.random(2000)), rng.random(2000)
    tri = triangles[selected]
    points = (
        (1 - u[:, None]) * tri[:, 0]
        + (u * (1 - v))[:, None] * tri[:, 1]
        + (u * v)[:, None] * tri[:, 2]
    )
    points *= {"m": 1, "cm": 0.01, "mm": 0.001}[units]
    transform = np.asarray(transform)
    points = points @ transform[:3, :3].T + transform[:3, 3]
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, points, allow_pickle=False)
    atomic_json(
        destination.with_suffix(".json"),
        dict(
            seed=seed,
            points=2000,
            mesh_sha256=sha256(mesh_path),
            points_sha256=sha256(destination),
            recipe="surface-v1; new sample bank, not the historical frozen ADD-S sample",
        ),
    )
    return destination
