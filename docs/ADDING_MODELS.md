# Add a model

Generate and install an entry-point adapter without changing dataset or metric code:

```bash
.venv/bin/cookiecutter templates/model_adapter
.venv/bin/python -m pip install -e ./my_pose_model
```

The template generates a Python package, `model.yaml`, dependency file and test.
Choose `native` for an actual integration. Connect its intentionally unimplemented
loader and prediction method. Choose `fixture` for an immediately executable
constant-pose example, restricted to fixture datasets. It cannot enter a full
benchmark or claim fresh FPS.

Add the generated `model.yaml` to the `models` mapping of a suite. A suite alias
may differ from the registered adapter name. The coordinator discovers installed
`spacergbbenchmark.models` entry points. Model-specific CUDA dependencies belong
in a separate environment. The resident `Worker` interface used by the built-in
models is an example of an isolated process adapter.

`Model.start()` loads once. `step(index)` receives only the current input and
returns `Prediction(pose, status, confidence, failure)`. A causal model preserves
state between steps and emits an immutable output. An offline model implements
`run_sequence(indices)` and declares `temporal_mode = "offline"`. Framewise models
receive only target inputs. Causal/offline models receive the full input schedule.

Use `self.sequence.rgb(index)`, `.intrinsics(index)`, `.localization(index)` and
`.bbox(index)`. `None` localization means no detection. The manifest supplies
CAD units and the transform to its object frame. Outputs are metric object-to-camera
poses. Do not expose evaluation labels or choose parameters from their errors.
Declare unsupported inputs and reject them clearly. A relative-only model needs
a metric pose pipeline, and a depth-dependent model needs a separate protocol.

Implement `synchronize()` for accurate timing and `close()` for all owned workers.
Ordinary reinitialization belongs inside `step`, so its cost stays in sustained
FPS. Never skip missing frames in temporal models. Restarting an interrupted
sequence from zero is the default recovery policy.

Before claiming support, test object/camera axes, units, absent detections,
invalid poses, initialization, interruption and sequence reset. A causal method
also needs fixed-recording replay and prefix invariance. Run a small native
inference-to-score check on each claimed dataset. Keep the compatibility status
visible in your adapter documentation. The template test is only a starting point.

Declare `required_modalities` when a model needs more than the default RGB,
intrinsics, CAD and optional localization contract. Unsupported modalities are
rejected before execution. Put additional checkpoint/source locations in
`asset_paths` so the generic runner fingerprints them without a core code change.
