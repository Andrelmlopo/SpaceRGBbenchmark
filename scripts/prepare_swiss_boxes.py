"""Generate explicitly annotation-conditioned SwissCube xyxy boxes from visible masks."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from spacergbbenchmark.datasets.prepare import swiss_sequence_root
from spacergbbenchmark.io import atomic_json, read_json


def prepare(root, output):
    for number in range(400, 500):
        name = f"seq_{number:06d}"
        masks = sorted((swiss_sequence_root(root, name) / "mask_visib").glob("*_000000.png"))
        if not masks:
            raise ValueError(f"Missing official test masks: {name}")
        boxes = {}
        for path in masks:
            with Image.open(path) as image:
                mask = np.asarray(image) != 0
            y, x = np.nonzero(mask)
            boxes[path.stem.split("_")[0]] = (
                [int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)] if len(x) else None
            )
        path = Path(output) / f"{name}.json"
        if path.exists() and read_json(path) != boxes:
            raise ValueError(f"Preserving existing different boxes: {path}")
        atomic_json(path, boxes)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    prepare(a.root, a.out)
