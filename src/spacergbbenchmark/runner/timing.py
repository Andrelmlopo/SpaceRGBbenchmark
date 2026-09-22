"""Fresh resident-model timings are separate from accuracy and cached replay."""

from pathlib import Path

import numpy as np

from ..datasets import Dataset
from ..io import atomic_json, read_json
from .engine import hardware, sequence_job


def benchmark(suite, dataset_name, method, stream_name, gpu, warmup, output):
    if warmup < 1:
        raise ValueError("Exclude initialization with at least one warm-up input")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    dataset = Dataset(suite["datasets"][dataset_name])
    stream = next(s for s in dataset.streams if s["id"] == stream_name)
    model = suite["models"][method]
    repetitions = []
    for repeat in range(3):
        destination = output / f"repeat-{repeat + 1}"
        sequence_job(
            dataset,
            stream,
            model["adapter"],
            model["settings"],
            destination,
            str(gpu),
            warmup=warmup,
            timing=True,
        )
        repetitions.append(read_json(destination / "timing.json"))
    result = dict(
        dataset=dataset_name,
        method=method,
        stream=stream_name,
        hardware=hardware(),
        repetitions=repetitions,
        fps_mean=float(np.mean([x["fps"] for x in repetitions])),
        fps_sd=float(np.std([x["fps"] for x in repetitions], ddof=1)),
        scope="three warmed fresh passes on the named stream; not full-partition timing",
    )
    atomic_json(output / "timing.json", result)
    return result
