from __future__ import annotations

import json
import os
import selectors
import shutil
import signal
import subprocess
import time
from pathlib import Path

import numpy as np

from ..io import identity, sha256
from ..schema import Prediction, valid_pose
from .base import Model
from .process import Worker


class SRT3DModel(Model):
    temporal_mode = "causal"
    needs_gpu = True  # Automatic initialization is charged as setup/warm-up.

    def start(self):
        self.process, self.initializer, self.initial_index = None, None, None
        self.initializer = Worker("mega", self.settings, self.sequence, self.output, self.gpu)

    def initialize(self, index, pose):
        import trimesh

        s = self.sequence
        mesh = trimesh.load(s.file(s.config["mesh"]), force="mesh", process=False)
        mesh.apply_scale({"m": 1.0, "mm": 0.001, "cm": 0.01}[s.config["mesh_units"]])
        mesh.apply_transform(np.asarray(s.config["raw_to_object"]))
        output = Path(self.output)
        geometry = output / "object.obj"
        geometry.write_text(trimesh.exchange.obj.export_obj(mesh, include_texture=False))
        diameter = float(1.1 * 2 * np.linalg.norm(mesh.vertices, axis=1).max())
        key = identity(
            dict(mesh=sha256(geometry), executable=sha256(self.settings["srt3d_executable"]))
        )
        cache = Path(self.settings.get("srt3d_model_cache", output / "models")) / key
        cache.mkdir(parents=True, exist_ok=True)
        images = output / "images"
        images.mkdir()
        for ordinal, frame in enumerate(s.frames[index:]):
            (images / f"frame_{ordinal:06d}.png").symlink_to(s.file(frame["image"]))
            if not np.array_equal(s.intrinsics(index), s.intrinsics(index + ordinal)):
                raise ValueError("This SRT3D camera adapter requires constant sequence intrinsics")
        np.savetxt(output / "initial.txt", pose)
        K = s.intrinsics(index)
        config = dict(
            images=str(images),
            geometry=str(geometry),
            model_dir=str(cache),
            init_pose=str(output / "initial.txt"),
            output=str(output / "native_poses.txt"),
            n_frames=len(s) - index,
            width=s.width,
            height=s.height,
            fu=K[0, 0],
            fv=K[1, 1],
            ppu=K[0, 2],
            ppv=K[1, 2],
            geometry_unit=1.0,
            diameter=diameter,
            interactive=1,
        )
        path = output / "native.cfg"
        path.write_text(
            "".join(
                f"{k} {json.dumps(v) if isinstance(v, str) else v}\n" for k, v in config.items()
            )
        )
        self.log = (output / "srt3d.log").open("w")
        env = dict(os.environ, OMP_NUM_THREADS=str(self.settings.get("cpu_threads", 1)))
        command = [self.settings["srt3d_executable"], str(path)]
        if not env.get("DISPLAY"):
            if not shutil.which("xvfb-run"):
                raise RuntimeError("Headless SRT3D requires xvfb-run (install xvfb and xauth)")
            command = ["xvfb-run", "-a", *command]
            env["LIBGL_ALWAYS_SOFTWARE"] = "1"
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            env=env,
            start_new_session=True,
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.buffer = b""
        self.initial_index = index
        return self.receive(0)

    def receive(self, expected):
        deadline = time.monotonic() + self.settings.get("timeout_seconds", 600)
        while True:
            while b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                text = line.decode(errors="replace")
                if text.startswith("SPACERGB_POSE "):
                    values = np.fromstring(text[len("SPACERGB_POSE ") :], sep=" ")
                    if len(values) != 17 or int(values[0]) != expected:
                        raise ValueError("SRT3D output schedule mismatch")
                    return values[1:].reshape(4, 4)
                self.log.write(text + "\n")
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.selector.select(remaining):
                raise TimeoutError("SRT3D timed out; inspect srt3d.log")
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError("SRT3D exited before emitting a pose; inspect srt3d.log")
            self.buffer += chunk

    def step(self, index):
        if self.initial_index is None:
            if self.sequence.localization(index) is None:
                return Prediction(status="missing", failure="no_initial_detection")
            result = self.initializer.call(index)
            poses, scores = np.asarray(result["poses"]), np.asarray(result["scores"])
            good = [valid_pose(p) and np.isfinite(s) for p, s in zip(poses, scores)]
            if not any(good):
                return Prediction(status="missing", failure="initializer_failed")
            initial = poses[np.argmax(np.where(good, scores, -np.inf))]
            self.initializer.close()
            self.initializer = None
            return Prediction(self.initialize(index, initial), status="initialization")
        ordinal = index - self.initial_index
        self.process.stdin.write(f"{ordinal}\n".encode())
        self.process.stdin.flush()
        return Prediction(self.receive(ordinal))

    def close(self):
        if getattr(self, "initializer", None) is not None:
            self.initializer.close()
        if getattr(self, "process", None) is not None:
            if self.process.poll() is None:
                self.process.stdin.close()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGTERM)
                    self.process.wait(timeout=5)
            self.selector.close()
            self.process.stdout.close()
            self.log.close()
