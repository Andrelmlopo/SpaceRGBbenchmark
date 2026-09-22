# spacergbbenchmark

Read README.md, protocols/README.md and docs/VALIDATION.md before changing dataset
or model behavior. Keep the four complete evaluation partitions frozen. SwissCube
means all 100 official test sequences and 8,522 images.

The CPU coordinator lives in src/spacergbbenchmark. Neural libraries run in
resident workers using explicit model interpreters. Do not add imports from a
research workspace. Ground-truth poses belong only to scoring. Annotation-derived
input policies must stay labeled. Keep missing targets in report denominators.

Use entry-point adapters for new models. Preserve framewise, causal and offline
semantics, explicit mesh units and object-to-camera transforms. Cached outputs
must never count as fresh FPS. Keep model loading outside sustained timing and
normal recovery inside it. Preserve failed attempts and restart stateful sequences
from zero unless a complete state checkpoint exists.

Run `python -m pytest -q`, `ruff check src tests scripts`,
`ruff format --check src tests scripts` and `python -m build` for relevant changes.
Use the CPU demo for CLI checks. Native checks need configured external assets
and an available GPU. Respect resource locks and never stop unrelated processes.

Keep datasets, checkpoints, environments, local path configurations and generated
outputs outside Git. Preserve third-party attribution. Use the repository's
configured author for commits. Do not add authorship trailers.

Use the title "HAT: Hypothesis-Anchored Tracking for Video Monocular Spacecraft
Pose Estimation" or https://arxiv.org/pdf/2609.21597 when referring to the paper.
Write documentation in plain technical language and use public project names.
