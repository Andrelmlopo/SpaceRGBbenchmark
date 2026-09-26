from __future__ import annotations

import os
import signal
import subprocess
import sys
from multiprocessing import Pipe
from pathlib import Path


class Worker:
    def __init__(self, backend, settings, sequence, output, gpu):
        self.timeout = settings.get("timeout_seconds", 600)
        self.log_path = Path(output) / f"{backend}.log"
        self.log = self.log_path.open("w")
        self.connection, child = Pipe()
        self.process = None
        env = dict(os.environ)
        package_paths = [str(Path(__file__).resolve().parents[2])] + settings.get(
            "package_paths", []
        )
        env["PYTHONPATH"] = os.pathsep.join(package_paths)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[key] = str(settings.get("cpu_threads", 1))
        env.update(settings.get("environment", {}))
        python = settings.get(f"{backend}_python", settings.get("python", sys.executable))
        try:
            self.process = subprocess.Popen(
                [python, "-m", "spacergbbenchmark.models.worker", str(child.fileno())],
                pass_fds=[child.fileno()],
                stdout=self.log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            child.close()
            self.connection.send(
                dict(
                    backend=backend,
                    settings=settings,
                    sequence=str(sequence.path),
                    output=str(output),
                )
            )
            self.receive()
        except BaseException:
            child.close()
            self.close()
            raise

    def receive(self):
        if not self.connection.poll(self.timeout):
            raise TimeoutError(f"Model worker timeout; see {self.log_path}")
        try:
            result = self.connection.recv()
        except EOFError as error:
            raise RuntimeError(f"Model worker exited; see {self.log_path}") from error
        if result["status"] == "error":
            raise RuntimeError(f"{result['error']}; see {self.log_path}")
        return result.get("value")

    def call(self, index, **values):
        self.connection.send(dict(index=index, **values))
        return self.receive()

    def synchronize(self):
        self.connection.send({"synchronize": True})
        self.receive()

    def close(self):
        if self.process is not None and self.process.poll() is None:
            try:
                self.connection.send(None)
                self.process.wait(timeout=10)
            except (OSError, EOFError, subprocess.TimeoutExpired):
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait()
        self.connection.close()
        self.log.close()
