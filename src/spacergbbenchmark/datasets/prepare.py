"""Convert native benchmark layouts to explicit inference and scoring manifests."""

from __future__ import annotations

import csv
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..io import absolute, atomic_json, atomic_npz, read_config, read_json, sha256
from .manifest import COUNTS, Dataset

EXCLUDED_YCB = {1, 13, 16, 18, 19, 20, 21}
SPARK_K = [[1744.922, 0, 720], [0, 1746.586, 540], [0, 0, 1]]


def pose(rotation, translation):
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, translation
    return result


def bop_pose(record):
    return pose(
        np.asarray(record["cam_R_m2c"]).reshape(3, 3), np.asarray(record["cam_t_m2c"]) / 1000
    )


def image_paths(root):
    return sorted(p for p in Path(root).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def padded(box, fraction, width, height):
    if box is None:
        return None
    box = np.asarray(box, dtype=float)
    extent = box[2:] - box[:2]
    return np.clip(
        box + np.r_[-fraction * extent, fraction * extent],
        [0, 0, 0, 0],
        [width, height, width, height],
    ).tolist()


class Builder:
    def __init__(self, dataset, config, output):
        self.dataset, self.config = dataset, config
        self.output = Path(output)
        if self.output.exists():
            raise FileExistsError(f"Preserving existing manifest directory: {self.output}")
        self.output.mkdir(parents=True)
        self.streams, self.targets = [], []
        self.sources = {}

    def source(self, path):
        path = Path(path)
        self.sources[str(path)] = sha256(path)
        return path

    def add(self, name, obj, frames, truths, width, height, K=None):
        cfg = self.config
        mesh = str(cfg["mesh"]).format(object_id=obj)
        self.source(mesh)
        sequence = dict(
            schema="spacergbbenchmark.sequence.v1",
            dataset=self.dataset,
            id=name,
            object_id=obj,
            width=width,
            height=height,
            mesh=mesh,
            mesh_units=cfg.get("mesh_units", "m"),
            raw_to_object=cfg.get("raw_to_object", np.eye(4).tolist()),
            object_frame=cfg.get("object_frame", "native_cad"),
            localization_policy=cfg["localization_policy"],
            mask_encoding=cfg.get(
                "mask_encoding", "nonzero_any_channel" if self.dataset == "spark" else "binary"
            ),
            slam_input=cfg.get("slam_input", "rgb"),
            slam_resolution=cfg.get("slam_resolution", 196608),
            frames=frames,
        )
        if K is not None:
            sequence["K"] = np.asarray(K).tolist()
        path = self.output / "streams" / f"{name}.json"
        atomic_json(path, sequence)
        self.streams.append(
            dict(id=name, manifest=str(path.relative_to(self.output)), sha256=sha256(path))
        )
        for frame in frames:
            if frame["target"]:
                self.targets.append((name, frame["id"], obj, truths[frame["id"]]))

    def finish(self):
        targets = sorted(self.targets, key=lambda x: x[:3])
        if len(targets) != COUNTS[self.dataset] or len({t[:3] for t in targets}) != len(targets):
            raise ValueError(f"Incomplete or duplicate {self.dataset} targets: {len(targets)}")
        atomic_npz(
            self.output / "truth.npz",
            streams=np.array([t[0] for t in targets]),
            frames=np.array([t[1] for t in targets]),
            objects=np.array([t[2] for t in targets]),
            poses=np.array([t[3] for t in targets]),
        )
        points = {}
        for obj in sorted({t[2] for t in targets}):
            if self.config.get("mesh_points"):
                source = self.source(self.config["mesh_points"].format(object_id=obj))
            else:
                from ..assets import mesh_samples

                source = mesh_samples(
                    self.config["mesh"].format(object_id=obj),
                    self.config["mesh_units"],
                    self.config.get("raw_to_object", np.eye(4)),
                    self.output / "mesh_points" / f"obj{obj:06d}.npy",
                )
            points[str(obj)] = dict(path=str(source.absolute()), sha256=sha256(source))
        manifest = dict(
            schema="spacergbbenchmark.dataset.v1",
            dataset=self.dataset,
            scope="full",
            protocol="four-datasets-v1",
            expected_targets=COUNTS[self.dataset],
            streams=self.streams,
            truth=dict(path="truth.npz", sha256=sha256(self.output / "truth.npz")),
            mesh_points=points,
            localization_policy=self.config["localization_policy"],
            object_frame=self.config.get("object_frame", "native_cad"),
            sources=self.sources,
        )
        if self.config.get("calibration"):
            manifest["metric_calibration"] = read_json(self.source(self.config["calibration"]))
        atomic_json(self.output / "dataset.json", manifest)
        return Dataset(self.output / "dataset.json").validate()


def prepare_spark(b):
    cfg = b.config
    grouped = defaultdict(list)
    with b.source(cfg["truth"]).open() as stream:
        for row in csv.DictReader(stream):
            if row["sequence"] in {f"RT{i}" for i in range(500, 600)}:
                grouped[row["sequence"]].append(row)
    for name in (f"RT{i}" for i in range(500, 600)):
        boxes = read_json(b.source(Path(cfg["boxes"]) / f"{name}.json"))
        frames, truths = [], {}
        for row in sorted(grouped[name], key=lambda x: x["filename"]):
            image = Path(cfg["images"]) / name / row["filename"]
            fid = int(image.stem.split("_")[0][3:])
            mask = (
                Path(cfg["masks"]) / name / "masks" / (image.stem + ".png")
                if cfg.get("masks")
                else None
            )
            frames.append(
                dict(
                    id=fid,
                    image=str(image),
                    target=True,
                    box=boxes.get(image.stem),
                    mask=str(mask) if mask is not None and mask.is_file() else None,
                )
            )
            truths[fid] = pose(
                Rotation.from_quat([float(row[k]) for k in ("Qx", "Qy", "Qz", "Qw")]).as_matrix(),
                [float(row[k]) for k in ("Tx", "Ty", "Tz")],
            )
        if [f["id"] for f in frames] != list(range(300)):
            raise ValueError(f"Incomplete SPARK sequence: {name}")
        b.add(name, 1, frames, truths, 1440, 1080, SPARK_K)


def prepare_ycbv(b):
    cfg = b.config
    groups = defaultdict(set)
    for target in read_json(b.source(cfg["targets"])):
        if target["obj_id"] not in EXCLUDED_YCB:
            groups[(target["scene_id"], target["obj_id"])].add(target["im_id"])
    detections = {}
    for detection in read_json(b.source(cfg["detections"])):
        key = (detection["scene_id"], detection["image_id"], detection["category_id"])
        if key not in detections or detection["score"] > detections[key]["score"]:
            detections[key] = detection
    for (scene, obj), targets in sorted(groups.items()):
        root = Path(cfg["images"]) / f"{scene:06d}"
        cameras = read_json(b.source(root / "scene_camera.json"))
        truth = read_json(b.source(Path(cfg["truth"]) / f"{scene:06d}" / "scene_gt.json"))
        frames, truths = [], {}
        for image in image_paths(root / "rgb"):
            fid = int(image.stem)
            detection = detections.get((scene, fid, obj)) if fid in targets else None
            box = None
            if detection is not None:
                x, y, width, height = detection["bbox"]
                box = padded([x, y, x + width, y + height], 0, 640, 480)
            frames.append(
                dict(
                    id=fid,
                    image=str(image),
                    K=np.asarray(cameras[str(fid)]["cam_K"]).reshape(3, 3).tolist(),
                    target=fid in targets,
                    box=box,
                    mask=None,
                    mask_rle=detection.get("segmentation") if detection else None,
                )
            )
            if fid in targets:
                records = [x for x in truth[str(fid)] if x["obj_id"] == obj]
                if len(records) != 1:
                    raise ValueError(f"Ambiguous target {scene}/{fid}/{obj}")
                truths[fid] = bop_pose(records[0])
        if set(truths) != targets:
            raise ValueError("Missing YCB target images")
        b.add(f"{scene:06d}_obj{obj:06d}", obj, frames, truths, 640, 480)


def swiss_sequence_root(root, name):
    """Accept both the official nested BOP export and existing flat layouts."""
    sequence = Path(root) / name
    if (sequence / "scene_camera.json").is_file():
        return sequence
    if (sequence / "000000" / "scene_camera.json").is_file():
        return sequence / "000000"
    raise FileNotFoundError(f"Missing SwissCube scene_camera.json under {sequence}")


def prepare_swisscube(b):
    cfg = b.config
    for number in range(400, 500):
        name = f"seq_{number:06d}"
        root = swiss_sequence_root(cfg["root"], name)
        cameras = read_json(b.source(root / "scene_camera.json"))
        truth = read_json(b.source(root / "scene_gt.json"))
        boxes = read_json(b.source(Path(cfg["boxes"]) / f"{name}.json"))
        frames, truths = [], {}
        for image in image_paths(root / "rgb"):
            fid = int(image.stem)
            box = boxes.get(f"{fid:06d}", boxes.get(str(fid)))
            frames.append(
                dict(
                    id=fid,
                    image=str(image),
                    target=True,
                    K=np.asarray(cameras[str(fid)]["cam_K"]).reshape(3, 3).tolist(),
                    box=padded(box, cfg.get("box_padding", 0.15), 1024, 1024),
                    mask=None,
                    slam_mask=str(root / "mask_visib" / f"{fid:06d}_000000.png"),
                )
            )
            truths[fid] = bop_pose(truth[str(fid)][0])
        if {f["id"] for f in frames} != {int(k) for k in truth}:
            raise ValueError(f"SwissCube image/label coverage mismatch: {name}")
        b.add(name, 1, frames, truths, 1024, 1024)


def prepare_shirt(b):
    import trimesh

    cfg = b.config
    camera = read_json(b.source(Path(cfg["root"]) / "camera.json"))
    K = np.asarray(camera["cameraMatrix"], dtype=np.float32)
    width, height = camera["Nu"], camera["Nv"]
    cal = read_json(b.source(cfg["calibration"]))
    align, delta = np.asarray(cal["R_align"]), np.asarray(cal["delta"])
    # Explicit annotation-conditioned input generation. This runs before inference.
    mesh = trimesh.load(b.source(cfg["mesh"].format(object_id=1)), force="mesh")
    mesh.apply_scale({"m": 1.0, "cm": 0.01, "mm": 0.001}[cfg.get("mesh_units", "m")])
    lo, hi = mesh.bounds
    corners = np.array(list(itertools.product(*zip(lo, hi))), dtype=np.float32)
    for trajectory, domain in itertools.product(("roe1", "roe2"), ("synthetic", "lightbox")):
        labels = read_json(b.source(Path(cfg["root"]) / trajectory / f"{trajectory}.json"))
        frames, truths = [], {}
        for fid, label in enumerate(labels):
            q = label["q_vbs2tango_true"]
            R = Rotation.from_quat(q[1:] + q[:1]).as_matrix()
            if cal["convention"] != "H2":
                R = R.T
            t = np.asarray(label["r_Vo2To_vbs_true"])
            Rcad = R @ align
            truths[fid] = pose(Rcad, t - Rcad @ delta)
            xyz = corners @ R.T + t
            xyz[:, 2] = np.clip(xyz[:, 2], 1e-3, None)
            uv = (xyz @ K.T)[:, :2] / xyz[:, 2:3]
            box = padded(np.r_[uv.min(0), uv.max(0)], cfg.get("box_padding", 0.2), width, height)
            image = Path(cfg["root"]) / trajectory / domain / "images" / label["filename"]
            frames.append(dict(id=fid, image=str(image), target=True, box=box, mask=None))
        b.add(f"{trajectory}_{domain}", 1, frames, truths, width, height, K)


def prepare(config_path, output, datasets=None):
    config_path = Path(config_path).absolute()
    config = read_config(config_path)
    selected = list(COUNTS) if datasets is None else datasets
    reports = []
    for name in selected:
        cfg = dict(config["datasets"][name])
        for key in (
            "root",
            "images",
            "truth",
            "mesh",
            "mesh_points",
            "boxes",
            "masks",
            "calibration",
            "detections",
            "targets",
        ):
            if cfg.get(key):
                cfg[key] = str(absolute(cfg[key], config_path.parent))
        builder = Builder(name, cfg, Path(output) / name)
        {
            "spark": prepare_spark,
            "ycbv": prepare_ycbv,
            "swisscube": prepare_swisscube,
            "shirt": prepare_shirt,
        }[name](builder)
        reports.append(builder.finish())
        print(f"Prepared {name}: {reports[-1]['targets']} targets", flush=True)
    return reports
