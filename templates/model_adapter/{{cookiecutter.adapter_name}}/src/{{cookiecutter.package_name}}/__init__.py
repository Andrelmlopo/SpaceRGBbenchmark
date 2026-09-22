import numpy as np
from spacergbbenchmark.models.base import Model
from spacergbbenchmark.schema import Prediction


class {{cookiecutter.class_name}}(Model):
    temporal_mode = "{{cookiecutter.temporal_mode}}"
{% if cookiecutter.implementation == "fixture" %}
    needs_gpu = False
    measurement_kind = "fixture"

    def start(self):
        self.pose = np.eye(4)
        self.pose[2, 3] = self.settings.get("distance_m", 2.0)

    def step(self, index):
        return Prediction(self.pose.copy())

    def run_sequence(self, indices):
        return [self.step(i) for i in indices]
{% else %}
    needs_gpu = True

    def start(self):
        """Load a resident model using self.settings, with no label access."""
        raise NotImplementedError("Connect the native model before running this adapter")

    def step(self, index):
        rgb = self.sequence.rgb(index)
        K = self.sequence.intrinsics(index)
        mask = self.sequence.localization(index)
        return Prediction(self.model.predict(rgb, K, mask))

    def run_sequence(self, indices):
        """Implement joint sequence inference if temporal_mode is offline."""
        raise NotImplementedError("Connect the offline sequence model here")

    def synchronize(self):
        import torch
        torch.cuda.synchronize()

    def close(self):
        """Release resources owned by this adapter."""
{% endif %}
