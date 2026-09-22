"""Replay recorded HAT observations and check prefix invariance without pose labels."""

import argparse
import importlib
from pathlib import Path

import numpy as np

from spacergbbenchmark.io import atomic_json, sha256
from spacergbbenchmark.schema import read_predictions


def check(observations, predictions, model, output):
    with np.load(observations, allow_pickle=False) as a:
        motion = a["slam"].copy()
        anchors = {
            int(f): (p.copy(), s.copy())
            for f, p, s in zip(a["frames"], a["candidates"], a["scores"])
        }
    tracker_class = importlib.import_module(model).Tracker

    def replay(count):
        tracker = tracker_class(count)
        try:
            return np.array([tracker.step(motion[i], anchors.get(i))[0] for i in range(count)])
        finally:
            tracker.close()

    expected = read_predictions(predictions)["poses"]
    actual = replay(len(motion))
    np.testing.assert_array_equal(actual, expected)
    lengths = sorted(
        {1, min(6, len(motion)), min(12, len(motion)), min(24, len(motion)), len(motion)}
    )
    for length in lengths:
        np.testing.assert_array_equal(replay(length), actual[:length])
    result = dict(
        model=model,
        frames=len(motion),
        anchors=len(anchors),
        replay_exact=True,
        prefix_lengths=lengths,
        prefixes_exact=True,
        observations_sha256=sha256(observations),
        predictions_sha256=sha256(predictions),
    )
    atomic_json(output, result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--model", choices=["pico_hat", "mega_hat"], required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    print(check(a.observations, a.predictions, a.model, a.out))
