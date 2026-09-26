import numpy as np

from ..schema import Prediction
from .base import Model


class FixtureModel(Model):
    measurement_kind = "fixture"

    def step(self, index):
        if index == self.settings.get("fail_at"):
            raise RuntimeError("Deliberate fixture failure")
        if (
            self.settings.get("require_detection", True)
            and self.sequence.localization(index) is None
        ):
            return Prediction(status="missing", failure="no_detection")
        pose = np.asarray(self.settings.get("pose", np.eye(4).tolist()), dtype=float)
        return Prediction(pose, status="fixture")
