# Design and scope

`spacergbbenchmark` evaluates RGB pose models through a common dataset interface,
runner and scorer. The project uses Cookiecutter Data Science for its structure
and a separate Cookiecutter template for model adapters.

Pico-HAT and Mega-HAT implement
[HAT: Hypothesis-Anchored Tracking for Video Monocular Spacecraft Pose Estimation](https://arxiv.org/pdf/2609.21597).
The benchmark also supports PicoPose, MegaPose, GigaPose with MegaPose refinement,
RGBTrack and SRT3D.

## Evaluation partitions

| Dataset | Partition | Targets |
|---|---|---:|
| SPARK 2024 Stream-2 | RT500–RT599, research development partition | 30,000 |
| YCB-V | Declared non-symmetric object targets across 41 object streams | 3,073 |
| SwissCube | Official test sequences 400–499, with variable lengths | 8,522 |
| SHIRT | ROE1/ROE2, synthetic and lightbox | 9,484 |

SwissCube uses the full official test split. YCB-V uses the non-symmetric
comparison, not the complete BOP evaluation. Independent methods infer at target
timestamps. Temporal methods receive the dense input streams and are scored on
the same targets.

Versioned manifests specify frame and object IDs, calibration, mesh units,
coordinate transforms, localization sources and checksums. SPARK/YCB-V use
predicted localization. SwissCube/SHIRT use annotation-conditioned localization.
These input policies are reported with the results. Pose labels are stored
separately and read only by the scorer. Model initialization and fallback decisions
must not use evaluation poses.

## Model interface

Dataset adapters provide RGB frames, intrinsics, CAD assets, optional localization
and the target schedule. Missing detections are distinct from missing images.
Predictions contain frame and object identity, object-to-camera pose, validity,
failure reason, optional confidence and tracking status. Translations are in
metres and camera axes follow OpenCV. CAD transforms must be explicit.

Framewise models estimate individual target poses. Causal models preserve state
between steps and cannot revise past outputs. Offline models process complete
sequences. Each adapter declares its temporal mode, required inputs, environment
and supported outputs. Unsupported modalities are rejected before inference.
Relative motion needs an explicit metric-pose pipeline to enter this benchmark.

Pico-HAT and Mega-HAT use pinned standalone packages. Their main configurations
retain DROID-SLAM, causal hypothesis selection, both fusion stages and gap filling.
Component removals belong in separately named ablations.

New models are installed through entry points. The adapter template supplies a
package, model configuration, dependency file and test example. Adding a model
should not require changes to dataset loaders or metric code.

## Execution and recovery

The CPU coordinator starts resident model workers in separate environments.
Source revisions, patches, checkpoint hashes, templates, package versions, seeds
and hardware are recorded with each run. Installation recipes build native
extensions from source. Datasets, weights, environments and outputs stay outside
Git.

A suite expands into model/dataset/sequence jobs. The runner validates inputs,
saves the resolved configuration, runs inference, validates outputs, scores all
targets and writes reports. Asset acquisition and template preparation are explicit
setup commands. Assets requiring credentials can be placed manually.

Jobs use atomic state files, bounded retries and GPU locks. Failed attempts retain
their logs while independent jobs continue. Completed outputs are reused only when
inputs, code, settings, environments and output hashes match. Interrupted stateful
sequences restart from frame zero. Future adapters may resume from a complete
state checkpoint, but skipping frames without restoring state is invalid.

Accuracy jobs can run on separate GPUs. Timing requires an isolated interval and
checks for external GPU and CPU contention. The runner cleans up only its own
workers.

## Metrics and timing

The pose metric is rotation error in radians plus translation error divided by
ground-truth range, with a 1e-6 range guard. Reports also include translation error
in metres, rotation error in degrees, ADD-S, mean, sample SD, coverage and separate
sequence results. SHIRT retains its calibrated origin transform. ADD-S uses fixed
mesh samples in the declared object frame.

Every declared target has a result row, including failures. Error means use valid
predictions. Matched-target comparisons show results on common valid targets.
Official dataset metrics, if added later, need their own evaluation profiles.

Sustained FPS uses three fresh resident-model runs after warm-up, with device
synchronization at the timing boundaries. Loading, initial setup, external
detection and scoring are excluded. Required input preparation, tracking,
recovery, DROID and fusion remain in the timed loop. Initialization costs are
recorded separately. Receipts include input count, wall time, resolution,
precision, CPU allocation and warm-up. Sparse-target throughput and dense-video
FPS use explicit denominators. Cached predictions cannot count as fresh FPS.

## Acceptance checks

The implementation was checked with full-partition saved-result scoring,
interruption and resume tests, input-change invalidation, independent job failures,
short native inference on each model/dataset pair, and HAT causal-prefix replay.
A separate installation and a generated fixture adapter exercise the package and
extension interfaces. Exact counts, tolerances and installation limits are in
[VALIDATION.md](VALIDATION.md).

Full fresh accuracy runs and sustained FPS measurements follow adapter validation.
New renderers or frontend environments require new accuracy measurements even
when temporal settings are unchanged. Historical test-informed choices remain
recorded in the reproduction protocol.

Version 1 evaluates existing models. Training, fine-tuning, parameter search,
additional datasets and manuscript editing are outside its scope. See the
[README](../README.md) for commands and [protocol](../protocols/README.md) for the
complete comparison rules.
