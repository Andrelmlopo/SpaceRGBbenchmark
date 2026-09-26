from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import atomic_npz

SCHEMA = "spacergbbenchmark.predictions.v1"


def valid_pose(pose):
    if pose is None:
        return False
    pose = np.asarray(pose)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        return False
    rotation = pose[:3, :3]
    return bool(
        np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5, rtol=0)
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3, rtol=0)
        and abs(np.linalg.det(rotation) - 1) < 1e-3
    )


@dataclass
class Prediction:
    pose: object = None
    status: str = "estimated"
    confidence: float | None = None
    failure: str = ""


def write_predictions(path, frames, predictions):
    if len(frames) != len(predictions) or len(set(frames)) != len(frames):
        raise ValueError("Predictions require one record per unique input frame")
    valid = np.array([valid_pose(p.pose) for p in predictions], dtype=bool)
    poses = np.full((len(frames), 4, 4), np.nan)
    for index, record in enumerate(predictions):
        if valid[index]:
            poses[index] = record.pose
    atomic_npz(
        path,
        schema=np.array(SCHEMA),
        frames=np.asarray(frames, dtype=np.int64),
        poses=poses,
        valid=valid,
        status=np.asarray([p.status for p in predictions], dtype=str),
        failure=np.asarray(
            [
                p.failure or ("" if ok else "missing_or_invalid_SE3")
                for p, ok in zip(predictions, valid)
            ],
            dtype=str,
        ),
        confidence=np.asarray(
            [np.nan if p.confidence is None else p.confidence for p in predictions]
        ),
    )


def read_predictions(path, expected_frames=None):
    with np.load(path, allow_pickle=False) as archive:
        frames, poses, valid = (archive[k].copy() for k in ("frames", "poses", "valid"))
        if "schema" not in archive or str(archive["schema"]) != SCHEMA:
            raise ValueError(f"Unrecognized prediction schema: {path}")
        status = archive["status"].copy()
        failure = archive["failure"].copy()
    count = len(frames)
    if (
        frames.shape != (count,)
        or not np.issubdtype(frames.dtype, np.integer)
        or len(set(frames.tolist())) != count
        or poses.shape != (count, 4, 4)
        or valid.shape != (count,)
        or valid.dtype != np.bool_
        or status.shape != (count,)
        or failure.shape != (count,)
    ):
        raise ValueError(f"Invalid prediction dimensions or identifiers: {path}")
    if expected_frames is not None and frames.tolist() != list(expected_frames):
        raise ValueError(f"Prediction frame schedule differs: {path}")
    if any(ok and not valid_pose(pose) for ok, pose in zip(valid, poses)):
        raise ValueError(f"A valid prediction is not a finite SE(3) pose: {path}")
    return dict(frames=frames, poses=poses, valid=valid, status=status, failure=failure)
