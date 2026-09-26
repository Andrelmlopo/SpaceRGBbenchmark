# Sources and attribution

The benchmark's original coordinator, interfaces and tests use the repository's
MIT license. External models, datasets and checkpoints retain their own licenses
and attribution requirements. They are fetched separately, not relicensed here.

| Component | Source |
|---|---|
| Cookiecutter Data Science scaffold | https://github.com/drivendataorg/cookiecutter-data-science |
| PicoPose | https://github.com/foollh/PicoPose |
| Pico-HAT | https://github.com/Andrelmlopo/Pico-HAT |
| MegaPose | https://github.com/megapose6d/megapose6d |
| Mega-HAT | https://github.com/Andrelmlopo/Mega-HAT |
| DROID-SLAM | https://github.com/princeton-vl/DROID-SLAM |
| GigaPose | https://github.com/nv-nguyen/gigapose |
| DINOv2 | https://github.com/facebookresearch/dinov2 |
| RGBTrack, building on FoundationPose and XMem | https://github.com/GreatenAnoymous/RGBTrack |
| SRT3D | https://github.com/DLR-RM/3DObjectTracking |

Pinned revisions and patches are in `environments/sources.json`. Checkpoint
sources and hashes are recorded separately. GigaPose retrieval glue was ported
from the author's research adapter and calls upstream network and pose-recovery
APIs. SRT3D's interactive executable was ported from the research integration and
links the upstream library. Scoring and dataset conversion preserve the recorded
thesis conventions. Cite the model and dataset papers when reporting results.

The initial scaffold used Cookiecutter Data Science 2.3.0, revision
`0f6b163cdbe3918a2c65ab57ad9fefda93976d9e`.

Pico-HAT and Mega-HAT implement
[HAT: Hypothesis-Anchored Tracking for Video Monocular Spacecraft Pose Estimation](https://arxiv.org/pdf/2609.21597)
by André Lopo, Atabak Dehban and Rodrigo Ventura (2026).

The MIT notices for GigaPose-derived retrieval glue and the SRT3D integration
are retained in `LICENSES/`. RGBTrack/FoundationPose and their checkpoints retain
the upstream NVIDIA terms referenced by the fetched source repository.
