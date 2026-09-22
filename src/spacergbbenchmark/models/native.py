"""Full inference adapters and HAT composition using resident native workers."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from ..io import atomic_npz
from ..schema import Prediction, valid_pose
from .base import Model
from .process import Worker


class NativePoseModel(Model):
    needs_gpu = True
    backend = "pico"

    def start(self):
        self.worker = Worker(self.backend, self.settings, self.sequence, self.output, self.gpu)

    def step(self, index):
        if self.sequence.localization(index) is None:
            return Prediction(status="missing", failure="no_detection")
        result = self.worker.call(index)
        poses, scores = np.asarray(result["poses"]), np.asarray(result["scores"])
        eligible = np.array([valid_pose(p) and np.isfinite(s) for p, s in zip(poses, scores)])
        if not eligible.any():
            return Prediction(status="failed", failure="no_valid_hypothesis")
        selected = int(np.argmax(np.where(eligible, scores, -np.inf)))
        return Prediction(poses[selected], confidence=float(scores[selected]))

    def synchronize(self):
        self.worker.synchronize()

    def close(self):
        if hasattr(self, "worker"):
            self.worker.close()


class PicoPoseModel(NativePoseModel):
    backend = "pico"


class MegaPoseModel(NativePoseModel):
    backend = "mega"


class GigaPoseModel(NativePoseModel):
    backend = "giga"

    def start(self):
        super().start()
        try:
            self.refiner = Worker("mega", self.settings, self.sequence, self.output, self.gpu)
        except BaseException:
            self.worker.close()
            raise

    def step(self, index):
        if self.sequence.localization(index) is None:
            return Prediction(status="missing", failure="no_detection")
        candidates = self.worker.call(index)
        result = self.refiner.call(index, priors=candidates["poses"])
        poses, scores = np.asarray(result["poses"]), np.asarray(result["scores"])
        good = [valid_pose(p) and np.isfinite(s) for p, s in zip(poses, scores)]
        if not any(good):
            return Prediction(status="failed", failure="no_valid_refined_hypothesis")
        selected = int(np.argmax(np.where(good, scores, -np.inf)))
        return Prediction(poses[selected], confidence=float(scores[selected]))

    def synchronize(self):
        super().synchronize()
        self.refiner.synchronize()

    def close(self):
        if hasattr(self, "refiner"):
            self.refiner.close()
        super().close()


class HatModel(NativePoseModel):
    temporal_mode = "causal"
    tracker_package = "pico_hat"

    def start(self):
        self.tracker = importlib.import_module(self.tracker_package).Tracker(len(self.sequence))
        self.observations, self.motion = [], []
        super().start()
        try:
            self.droid = Worker("droid", self.settings, self.sequence, self.output, self.gpu)
        except BaseException:
            self.worker.close()
            self.tracker.close()
            raise

    def step(self, index):
        motion = np.asarray(self.droid.call(index)["pose"])
        candidates = None
        if self.tracker.anchor_due and self.sequence.localization(index) is not None:
            result = self.worker.call(index)
            candidates = np.asarray(result["poses"]), np.asarray(result["scores"])
            self.observations.append((index, *candidates))
        self.motion.append(motion)
        output, diagnostics = self.tracker.step(motion, candidates)
        return Prediction(
            output,
            status="held" if diagnostics.get("held") else "estimated",
            failure="" if diagnostics.get("valid") else "not_initialized",
        )

    def synchronize(self):
        super().synchronize()
        self.droid.synchronize()

    def close(self):
        if hasattr(self, "observations"):
            count = self.settings.get("candidates", 5 if self.backend == "pico" else 10)
            atomic_npz(
                Path(self.output) / "observations.npz",
                slam=np.asarray(self.motion),
                frames=np.asarray([x[0] for x in self.observations], dtype=int),
                candidates=np.asarray([x[1] for x in self.observations]).reshape(-1, count, 4, 4),
                scores=np.asarray([x[2] for x in self.observations]).reshape(-1, count),
            )
        if hasattr(self, "droid"):
            self.droid.close()
        if hasattr(self, "tracker"):
            self.tracker.close()
        super().close()


class PicoHatModel(HatModel):
    backend = "pico"


class MegaHatModel(HatModel):
    backend = "mega"
    tracker_package = "mega_hat"


class RGBTrackModel(NativePoseModel):
    backend = "rgbtrack"
    temporal_mode = "causal"

    def step(self, index):
        result = self.worker.call(index)
        return Prediction(
            result.get("pose"),
            status=result.get("phase", "estimated"),
            failure=result.get("failure", ""),
        )
