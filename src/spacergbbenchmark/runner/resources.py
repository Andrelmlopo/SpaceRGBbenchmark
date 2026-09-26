from __future__ import annotations

import fcntl
import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def output_lease(output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / ("." + output.name + ".lock")).open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process owns this output: {output}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def gpu_processes():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


@contextmanager
def lease(gpu=None, timing=False, lock_root=None):
    root = Path(
        lock_root
        or os.environ.get("SPACERGB_LOCK_ROOT")
        or Path(tempfile.gettempdir()) / "spacergbbenchmark-locks"
    )
    root.mkdir(parents=True, exist_ok=True)
    handles = []
    try:
        if timing and os.getloadavg()[0] > max(1, len(os.sched_getaffinity(0))) * 0.5:
            raise RuntimeError("Host CPU load is too high for an isolated timing interval")
        interval = (root / "timing.lock").open("a+")
        handles.append(interval)
        fcntl.flock(interval, fcntl.LOCK_EX if timing else fcntl.LOCK_SH)
        if gpu is not None:
            if not str(gpu).isdigit():
                raise ValueError("GPU must be a physical integer index")
            handle = (root / f"gpu-{gpu}.lock").open("a+")
            handles.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX)
            info = subprocess.run(
                ["nvidia-smi", f"--id={gpu}", "--query-gpu=uuid", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            busy = gpu_processes()
            if (timing and busy) or any(line.split(",")[0].strip() == info for line in busy):
                raise RuntimeError("GPU occupied by external work; no process was interrupted")
        yield
    finally:
        for handle in reversed(handles):
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
