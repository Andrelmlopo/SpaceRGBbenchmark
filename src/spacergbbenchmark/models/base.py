"""Minimal public adapter interface."""


class Model:
    temporal_mode = "framewise"  # framewise, causal, or offline
    needs_gpu = False
    measurement_kind = "fresh_inference"
    required_modalities = {"rgb", "intrinsics", "cad", "localization"}

    def __init__(self, settings, sequence, output, gpu=None):
        self.settings, self.sequence, self.output, self.gpu = settings, sequence, output, gpu

    def start(self):
        """Load resident models. Excluded from sustained timing."""

    def step(self, index):
        """Consume this frame and return schema.Prediction."""
        raise NotImplementedError

    def run_sequence(self, indices):
        """Offline methods override this, returning one Prediction per index."""
        return [self.step(index) for index in indices]

    def synchronize(self):
        """Wait for all asynchronous computation before returning."""

    def close(self):
        """Release only this adapter's owned resources."""
