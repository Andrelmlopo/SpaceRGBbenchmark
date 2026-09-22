"""Inference-only sequence readers. Ground truth belongs to the evaluator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..io import absolute, identity, read_json, safe_name, sha256

COUNTS = {"spark": 30000, "ycbv": 3073, "swisscube": 8522, "shirt": 9484}


class Sequence:
    def __init__(self, path):
        self.path = Path(path).absolute()
        self.config = read_json(self.path)
        if self.config.get("schema") != "spacergbbenchmark.sequence.v1":
            raise ValueError(f"Unknown sequence schema: {path}")
        self.name = safe_name(self.config["id"])
        self.frames = self.config["frames"]
        self.object_id = int(self.config["object_id"])
        ids = [frame["id"] for frame in self.frames]
        if not ids or any(type(i) is not int for i in ids) or ids != sorted(set(ids)):
            raise ValueError("Frames must have unique, increasing integer IDs")
        self.width, self.height = self.config["width"], self.config["height"]
        self.K = self.intrinsics(0)
        self.slam_input = self.config.get("slam_input", "rgb")
        self.slam_resolution = self.config.get("slam_resolution", 196608)
        if self.slam_input not in {"rgb", "masked"}:
            raise ValueError("slam_input must be rgb or masked")
        transform = np.asarray(self.config["raw_to_object"], dtype=float)
        from ..schema import valid_pose

        if not valid_pose(transform) or self.config["mesh_units"] not in {"m", "cm", "mm"}:
            raise ValueError("Declare mesh units and a rigid raw_to_object transform")
        if any("pose" in f or "gt" in f for f in self.frames):
            raise ValueError("Inference manifests must not contain pose labels")

    def __len__(self):
        return len(self.frames)

    def file(self, value):
        return absolute(value, self.path.parent)

    def intrinsics(self, index):
        K = np.asarray(self.frames[index].get("K", self.config.get("K")), dtype=float)
        if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
            raise ValueError("Invalid camera intrinsics")
        if not np.allclose(K[2], [0, 0, 1]) or K[0, 1] != 0 or K[1, 0] != 0:
            raise ValueError("Expected zero-skew pinhole intrinsics")
        return K

    def rgb(self, index):
        with Image.open(self.file(self.frames[index]["image"])) as image:
            if image.size != (self.width, self.height):
                raise ValueError(f"Image/calibration size mismatch: {image.filename}")
            return np.asarray(image.convert("RGB")).copy()

    def localization(self, index):
        frame = self.frames[index]
        if frame.get("mask_rle"):
            from pycocotools import mask as coco_mask

            rle = frame["mask_rle"]
            mask = (
                coco_mask.decode(
                    coco_mask.frPyObjects(rle, *rle["size"])
                    if isinstance(rle["counts"], list)
                    else rle
                )
                != 0
            )
            if mask.shape != (self.height, self.width):
                raise ValueError("Detection RLE must match the RGB dimensions")
        elif frame.get("mask"):
            with Image.open(self.file(frame["mask"])) as image:
                mask = np.asarray(image)
            if mask.ndim == 3:
                if self.config.get("mask_encoding") == "nonzero_any_channel":
                    mask = np.any(mask != 0, axis=2)
                elif np.all(mask == mask[..., :1]):
                    mask = mask[..., 0]
            if mask.shape != (self.height, self.width):
                raise ValueError("Mask must match the RGB dimensions")
            mask = mask != 0
        elif frame.get("box") is not None:
            box = np.asarray(frame["box"], dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all():
                raise ValueError("Expected finite xyxy box")
            x1, y1, x2, y2 = box
            if not 0 <= x1 < x2 <= self.width or not 0 <= y1 < y2 <= self.height:
                raise ValueError("Box is empty or outside the image")
            mask = np.zeros((self.height, self.width), dtype=bool)
            mask[int(np.floor(y1)) : int(np.ceil(y2)), int(np.floor(x1)) : int(np.ceil(x2))] = True
        else:
            return None
        return mask if mask.sum() >= 8 else None

    def bbox(self, index):
        box = self.frames[index].get("box")
        if box is not None:
            box = np.asarray(box, dtype=float)
            if (
                box.shape != (4,)
                or not np.isfinite(box).all()
                or not (0 <= box[0] < box[2] <= self.width and 0 <= box[1] < box[3] <= self.height)
            ):
                raise ValueError("Box is invalid or outside the image")
            return box
        mask = self.localization(index)
        if mask is None:
            return None
        y, x = np.nonzero(mask)
        return np.array([x.min(), y.min(), x.max() + 1, y.max() + 1], dtype=float)

    def validate(self, deep=False):
        for index, frame in enumerate(self.frames):
            self.intrinsics(index)
            for key in ("image", "mask", "slam_mask"):
                if frame.get(key) and not self.file(frame[key]).is_file():
                    raise FileNotFoundError(self.file(frame[key]))
            if deep:
                self.rgb(index)
                self.localization(index)
        if not self.file(self.config["mesh"]).is_file():
            raise FileNotFoundError(self.config["mesh"])
        return {
            "id": self.name,
            "frames": len(self),
            "targets": sum(f["target"] for f in self.frames),
        }


class Dataset:
    def __init__(self, path):
        self.path = Path(path).absolute()
        self.config = read_json(self.path)
        if self.config.get("schema") != "spacergbbenchmark.dataset.v1":
            raise ValueError("Unknown dataset manifest schema")
        self.name = safe_name(self.config["dataset"])
        self.streams = self.config["streams"]
        if len({s["id"] for s in self.streams}) != len(self.streams):
            raise ValueError("Duplicate stream IDs")

    def sequence(self, stream):
        if isinstance(stream, str):
            stream = next(s for s in self.streams if s["id"] == stream)
        path = absolute(stream["manifest"], self.path.parent)
        if sha256(path) != stream["sha256"]:
            raise ValueError(f"Sequence manifest changed: {path}")
        return Sequence(path)

    def validate(self, deep=False, check_files=True):
        rows = []
        for stream in self.streams:
            sequence = self.sequence(stream)
            rows.append(
                sequence.validate(deep)
                if check_files
                else dict(
                    id=sequence.name,
                    frames=len(sequence),
                    targets=sum(f["target"] for f in sequence.frames),
                )
            )
        count = sum(row["targets"] for row in rows)
        if count != self.config["expected_targets"]:
            raise ValueError(f"Target count mismatch: {count} != {self.config['expected_targets']}")
        if self.config.get("scope") == "full":
            if self.name not in COUNTS or count != COUNTS[self.name]:
                raise ValueError("Full benchmark target count differs from the frozen partition")
            names = {row["id"] for row in rows}
            if self.name == "swisscube" and names != {f"seq_{i:06d}" for i in range(400, 500)}:
                raise ValueError("SwissCube requires all 100 official test sequences")
            if self.name == "spark" and names != {f"RT{i}" for i in range(500, 600)}:
                raise ValueError("SPARK requires RT500 through RT599")
            frozen = read_json(Path(__file__).resolve().parents[1] / "protocols/partitions.json")[
                self.name
            ]
            actual = {}
            for stream in self.streams:
                seq = self.sequence(stream)
                targets = [f["id"] for f in seq.frames if f["target"]]
                actual[seq.name] = dict(
                    object_id=seq.object_id,
                    targets=len(targets),
                    target_ids_sha256=identity(targets),
                    inputs=len(seq),
                    input_ids_sha256=identity([f["id"] for f in seq.frames]),
                )
            if actual != frozen:
                raise ValueError("Stream/object/frame schedule differs from the frozen partition")
        return dict(
            dataset=self.name,
            scope=self.config["scope"],
            targets=count,
            streams=len(rows),
            input_frames=sum(r["frames"] for r in rows),
        )
