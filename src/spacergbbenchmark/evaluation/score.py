"""Scorer-only access to pose labels, full target accounting and metric receipts."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..datasets import Dataset
from ..io import absolute, atomic_json, sha256
from ..schema import read_predictions, valid_pose
from .metrics import add_s, metric_values, summarize


def score(dataset_path, prediction_root, output, method, adds=True):
    dataset = Dataset(dataset_path)
    dataset.validate(check_files=False)
    cfg = dataset.config
    truth_path = absolute(cfg["truth"]["path"], dataset.path.parent)
    if sha256(truth_path) != cfg["truth"]["sha256"]:
        raise ValueError("Ground-truth archive changed")
    with np.load(truth_path, allow_pickle=False) as archive:
        truth = {
            (str(s), int(f), int(o)): p
            for s, f, o, p in zip(
                archive["streams"],
                archive["frames"],
                archive["objects"],
                archive["poses"],
                strict=True,
            )
        }
    targets, predictions, prediction_hashes = set(), {}, {}
    for stream in dataset.streams:
        sequence = dataset.sequence(stream)
        for frame in sequence.frames:
            if frame["target"]:
                targets.add((sequence.name, frame["id"], sequence.object_id))
        path = Path(prediction_root) / f"{sequence.name}.npz"
        if not path.exists():
            continue
        loaded = read_predictions(path)
        allowed = {f["id"] for f in sequence.frames}
        if not set(loaded["frames"]) <= allowed:
            raise ValueError(f"Predictions contain unknown frame IDs: {path}")
        prediction_hashes[sequence.name] = sha256(path)
        for f, p, ok, failure in zip(
            loaded["frames"], loaded["poses"], loaded["valid"], loaded["failure"]
        ):
            predictions[(sequence.name, int(f), sequence.object_id)] = (p, ok, str(failure))
    if set(truth) != targets or len(targets) != cfg["expected_targets"]:
        raise ValueError("Truth and declared target sets differ")
    points = {}
    if adds:
        for obj in {t[2] for t in targets}:
            item = cfg["mesh_points"].get(str(obj))
            if not item:
                raise ValueError(
                    f"Missing frozen mesh samples for object {obj}; prepare them or use --no-adds"
                )
            path = absolute(item["path"], dataset.path.parent)
            if sha256(path) != item["sha256"]:
                raise ValueError(f"Mesh samples changed: {path}")
            points[obj] = np.load(path, allow_pickle=False)
    rows, grouped = [], defaultdict(list)
    for key in sorted(targets):
        stream, frame, obj = key
        if not valid_pose(truth[key]):
            raise ValueError(f"Invalid ground-truth pose: {key}")
        p, ok, failure = predictions.get(key, (None, False, "missing_prediction"))
        row = dict(
            dataset=dataset.name,
            method=method,
            stream=stream,
            frame=frame,
            object_id=obj,
            valid=bool(ok),
            failure="" if ok else failure,
            E_p=None,
            E_t_m=None,
            E_q_deg=None,
            ADD_S_m=None,
        )
        if ok:
            row.update(metric_values(p, truth[key], cfg.get("metric_calibration")))
            if adds:
                row["ADD_S_m"] = add_s(p, truth[key], points[obj])
        rows.append(row)
        grouped[stream].append(row)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "targets.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(
        summarize(rows),
        dataset=dataset.name,
        method=method,
        scope=cfg["scope"],
        localization_policy=cfg["localization_policy"],
        dataset_sha256=sha256(dataset.path),
        truth_sha256=sha256(truth_path),
        prediction_sha256=prediction_hashes,
        targets_sha256=sha256(output / "targets.csv"),
        protocol=cfg["protocol"],
        object_frame=cfg["object_frame"],
    )
    atomic_json(output / "metrics.json", result)
    atomic_json(output / "sequences.json", {k: summarize(v) for k, v in grouped.items()})
    return result
