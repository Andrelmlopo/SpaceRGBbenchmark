from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..datasets import Dataset
from ..io import atomic_json, sha256
from ..schema import Prediction, valid_pose, write_predictions


def import_predictions(
    dataset_path,
    source,
    output,
    *,
    pattern="{stream}.npz",
    format="npz",
    pose_key="poses",
    frame_key="frames",
    schedule=None,
    units="m",
    source_to_object=None,
    candidate_scores=None,
):
    dataset, source, output = Dataset(dataset_path), Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"Preserving existing predictions: {output}")
    output.mkdir(parents=True)
    scale = {"m": 1.0, "mm": 0.001}[units]
    transform = np.eye(4) if source_to_object is None else np.asarray(source_to_object, dtype=float)
    if not valid_pose(transform):
        raise ValueError("source_to_object must be a rigid metric transform")
    inverse = np.linalg.inv(transform)
    bop = {}
    hashes = {}
    if format == "bop":
        hashes[str(source)] = sha256(source)
        with source.open() as stream:
            for row in csv.DictReader(stream):
                key = (f"{int(row['scene_id']):06d}_obj{int(row['obj_id']):06d}", int(row["im_id"]))
                confidence = float(row["score"])
                if key not in bop or confidence > bop[key][0]:
                    P = np.eye(4)
                    P[:3, :3] = np.fromstring(row["R"], sep=" ").reshape(3, 3)
                    P[:3, 3] = np.fromstring(row["t"], sep=" ") * scale
                    bop[key] = (confidence, P @ inverse)
    total = 0
    for item in dataset.streams:
        sequence = dataset.sequence(item)
        target_ids = [f["id"] for f in sequence.frames if f["target"]]
        all_ids = [f["id"] for f in sequence.frames]
        mapped = {}
        if format == "bop":
            mapped = {
                fid: Prediction(
                    bop[(sequence.name, fid)][1], confidence=bop[(sequence.name, fid)][0]
                )
                for fid in all_ids
                if (sequence.name, fid) in bop
            }
        else:
            path = source / pattern.format(stream=sequence.name, object_id=sequence.object_id)
            if path.exists():
                hashes[str(path)] = sha256(path)
                with np.load(path, allow_pickle=False) as archive:
                    poses = archive[pose_key].copy()
                    if frame_key in archive:
                        frames = archive[frame_key].tolist()
                    elif schedule in {"inputs", "targets"}:
                        frames = all_ids if schedule == "inputs" else target_ids
                    else:
                        raise ValueError(
                            f"{path}: frame IDs absent; declare --schedule inputs or targets"
                        )
                    if len(frames) != len(poses) or len(set(frames)) != len(frames):
                        raise ValueError(f"{path}: mismatched or duplicate frames")
                    if not set(frames) <= set(all_ids):
                        raise ValueError(f"{path}: unknown frame IDs")
                    valid = (
                        archive["valid"].copy()
                        if "valid" in archive
                        else np.ones(len(frames), bool)
                    )
                    scores = archive[candidate_scores].copy() if candidate_scores else None
                if poses.ndim == 4:
                    if scores is None or scores.shape != poses.shape[:2]:
                        raise ValueError("Candidate archives require an explicit score array")
                    chosen = []
                    for candidates, logits in zip(poses, scores):
                        eligible = np.array(
                            [valid_pose(p) and np.isfinite(s) for p, s in zip(candidates, logits)]
                        )
                        chosen.append(
                            candidates[np.argmax(np.where(eligible, logits, -np.inf))]
                            if eligible.any()
                            else np.full((4, 4), np.nan)
                        )
                    poses = np.asarray(chosen)
                    if valid.ndim == 2:
                        valid = valid.any(axis=1)
                if poses.shape != (len(frames), 4, 4) or valid.shape != (len(frames),):
                    raise ValueError("Expected N x 4 x 4 poses and N validity flags")
                for fid, P, ok in zip(frames, poses, valid):
                    P = np.asarray(P, dtype=float).copy()
                    P[:3, 3] *= scale
                    mapped[int(fid)] = Prediction(
                        P @ inverse if ok else None,
                        status="imported",
                        failure="" if ok else "source_invalid",
                    )
        records = [
            mapped.get(fid, Prediction(status="missing", failure="missing_source_prediction"))
            for fid in target_ids
        ]
        write_predictions(output / f"{sequence.name}.npz", target_ids, records)
        total += sum(valid_pose(p.pose) for p in records)
    receipt = dict(
        kind="saved_prediction_import",
        fresh_inference=False,
        dataset_sha256=sha256(dataset.path),
        sources=hashes,
        units=units,
        source_to_object=transform.tolist(),
        valid=total,
        targets=dataset.config["expected_targets"],
    )
    atomic_json(output / "import.json", receipt)
    return receipt
