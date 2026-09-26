from __future__ import annotations

import importlib.metadata
import random
import sys
import traceback
from multiprocessing.connection import Connection

import numpy as np

from ..datasets import Sequence


def main():
    connection, model = Connection(int(sys.argv[1])), None
    try:
        request = connection.recv()
        settings = request["settings"]
        import cv2
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the selected model environment")
        seed = settings.get("seed", 0)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cv2.setRNGSeed(seed % (2**31))
        cv2.setNumThreads(settings.get("cpu_threads", 1))
        sequence = Sequence(request["sequence"])
        from pathlib import Path

        from ..io import atomic_json

        atomic_json(
            Path(request["output"]) / (request["backend"] + "-environment.json"),
            dict(
                python=sys.version,
                executable=sys.executable,
                torch=torch.__version__,
                cuda=torch.version.cuda,
                device=torch.cuda.get_device_name(0),
                seed=seed,
                packages={
                    d.metadata["Name"]: d.version
                    for d in importlib.metadata.distributions()
                    if d.metadata.get("Name")
                },
            ),
        )
        from .backends import make_backend

        model = make_backend(request["backend"], settings, sequence, request["output"])
        torch.cuda.synchronize()
        connection.send({"status": "ready"})
        while True:
            command = connection.recv()
            if command is None:
                break
            result = None if command.get("synchronize") else model.step(command["index"], command)
            torch.cuda.synchronize()
            connection.send({"status": "ok", "value": result})
    except BaseException as error:
        traceback.print_exc()
        try:
            connection.send({"status": "error", "error": f"{type(error).__name__}: {error}"})
        except (OSError, EOFError):
            pass
        raise
    finally:
        if model is not None and hasattr(model, "close"):
            model.close()
        connection.close()


if __name__ == "__main__":
    main()
