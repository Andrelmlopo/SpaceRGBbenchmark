# {{cookiecutter.adapter_name}}

Benchmark adapter. Install with `python -m pip install -e .` in the
coordinator environment, then reference `model.yaml` in a suite. No dataset or
scorer changes are needed. Native model libraries belong in a separate environment.

The `fixture` implementation is an executable interface example restricted to
fixture datasets. The `native` implementation needs a model loader and prediction method
before it can run. Declare units, object frame, initialization, missing
detection behavior and supported modalities in this document.

Run `pytest tests` after installing pytest. Add behavioral checks for your native
model, especially frame conversion and reset/recovery behavior.
