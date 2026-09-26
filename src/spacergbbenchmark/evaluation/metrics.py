from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from ..schema import valid_pose


def metric_values(prediction, truth, calibration=None):
    if not valid_pose(prediction) or not valid_pose(truth):
        raise ValueError("Metrics require finite SE(3) poses")
    P, G = np.asarray(prediction).copy(), np.asarray(truth).copy()
    if calibration:
        align, delta = np.asarray(calibration["R_align"]), np.asarray(calibration["delta"])
        for X in (P, G):
            X[:3, 3] += X[:3, :3] @ delta
            X[:3, :3] = X[:3, :3] @ align.T
    et = np.linalg.norm(P[:3, 3] - G[:3, 3])
    q1, q2 = Rotation.from_matrix(P[:3, :3]).as_quat(), Rotation.from_matrix(G[:3, :3]).as_quat()
    eq = 2 * np.arccos(np.clip(abs(q1 @ q2), 0, 1))
    return {
        "E_t_m": float(et),
        "E_q_deg": float(np.degrees(eq)),
        "E_p": float(eq + et / max(np.linalg.norm(G[:3, 3]), 1e-6)),
    }


def add_s(prediction, truth, points):
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("ADD-S needs finite metric CAD points in the prediction object frame")
    predicted = points @ prediction[:3, :3].T + prediction[:3, 3]
    target = points @ truth[:3, :3].T + truth[:3, 3]
    return float(cKDTree(target).query(predicted, k=1)[0].mean())


def summarize(rows):
    good = [row for row in rows if row["valid"]]
    summary = dict(
        targets=len(rows),
        valid=len(good),
        missing=len(rows) - len(good),
        coverage=len(good) / len(rows) if rows else 0,
        aggregation="pooled targets; error conditional on valid predictions",
    )
    for key in ("E_p", "E_t_m", "E_q_deg", "ADD_S_m"):
        values = [r[key] for r in good if r.get(key) is not None]
        summary[key] = float(np.mean(values)) if values else None
        summary[key + "_SD"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary[key + "_count"] = len(values)
    summary["rotation_over90_count"] = sum(r["E_q_deg"] > 90 for r in good)
    return summary
