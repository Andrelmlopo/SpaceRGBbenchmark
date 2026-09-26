# Validation

The checks below were completed on 22 September 2026. Native adapter receipts
are in [native_validation.json](native_validation.json). Full fresh accuracy and
FPS experiments have not been run.

## Full-partition scoring

[scoring_validation.json](scoring_validation.json) records per-target comparison
against the existing reference CSVs, including hashes of both CSVs.

| Dataset | Declared targets | Valid saved predictions | Metric agreement |
|---|---:|---:|---|
| SPARK | 30,000 | 30,000 | All four metrics exactly equal |
| YCB-V | 3,073 | 3,048 | All four metrics exactly equal |
| SwissCube | 8,522 | 8,522 | All four metrics exactly equal |
| SHIRT | 9,484 | 9,484 | ADD-S exact; other metrics within float32/float64 tolerance |

SHIRT maximum differences are 7.70e-8 pose error, 4.70e-7 m translation and
2.17e-6 degrees rotation. The original scorer performed an in-place calibrated
origin conversion on float32 poses. The portable scorer uses float64. Acceptance
tolerances are 1e-6 pose error/metres and 1e-5 degrees for that conversion.
These checks use frozen historical mesh samples. They do not establish accuracy
for fresh template rendering or a new model environment.

## Infrastructure

A fresh clone outside the research workspace installs and runs its demo and
27 CPU tests on Python 3.10. The final expanded suite passes 29 tests both
against an installed wheel on Python 3.10 and the source on Python 3.13.
Pinned Pico-HAT, Mega-HAT and patched MegaPose sources were fetched into fresh
directories. The native SRT3D library and executable were built from source.
The [installation receipt](installation_validation.json) records the scope.

CPU tests exercise metric units, rotations, calibrated SHIRT origin, ADD-S,
invalid SE(3), missing-output denominators, frame conversion, unknown timestamps,
manifest tampering, subset rejection, absent detections, colored masks, COCO RLE,
target-only framewise scheduling, variable timestamps, exclusive output ownership,
restart after a simulated exception and a real KeyboardInterrupt, verified reuse,
changed-image invalidation and failure isolation across independent jobs.

The synthetic demo runs all four dataset identities through inference, scoring
and reporting. The generated Cookiecutter fixture adapter is also installed and
run through the same interface as an extensibility check. This is a software
fixture check, not native-dataset inference.

## Native adapter checks

All seven methods pass fresh inference-to-score checks on each dataset.
Each check uses the first declared stream. These short checks keep
the complete target denominator in scoring and label their outputs as development
runs. They establish integration, not full-partition accuracy.

| Method | SPARK | YCB-V | SwissCube | SHIRT |
|---|---|---|---|---|
| PicoPose | Pass | Pass | Pass | Pass |
| Pico-HAT | Pass | Pass | Pass | Pass |
| MegaPose | Pass | Pass | Pass | Pass |
| Mega-HAT | Pass | Pass | Pass | Pass |
| GigaPose + MegaPose refiner | Pass | Pass | Pass | Pass |
| RGBTrack | Pass | Pass | Pass | Pass |
| SRT3D | Pass | Pass | Pass | Pass |

Framewise methods process three target timestamps. RGBTrack and SRT3D process
three consecutive inputs. Each HAT processes 36 inputs on SPARK/SwissCube/SHIRT
and 84 on YCB-V. All emitted poses in these checks are valid. Scored YCB targets
are fewer than dense temporal inputs, as the protocol requires. The JSON receipt
records processed inputs, valid scored targets, stream IDs and prediction hashes.
RGBTrack's SPARK/SHIRT/YCB checks exercise native tracking recovery. SRT3D uses
automatic MegaPose initialization. A separate PicoPose/SwissCube CLI run verifies
the complete command-to-inference-to-scoring-to-report path.

## Causality

[pico_causality_validation.json](pico_causality_validation.json) records fresh
Pico-HAT observation replay on four datasets: 36 frames each on SPARK,
SwissCube and SHIRT, and 84 YCB frames. Replay matches the emitted poses exactly.
Five prefixes per stream match exactly, including missing detections between
YCB anchors. [Mega-HAT](mega_causality_validation.json) also passes four exact
replays and 20 prefixes at the same per-dataset input counts. This tests the temporal method against recorded observations;
it does not claim a new bit-exact proof of every neural backend.

## Limits

Full fresh seven-model accuracy runs and isolated three-repeat FPS are separate
experiments. Historical test-informed recipes and annotation-conditioned localization remain
explicit in the protocol.

The coordinator is portable. GPU checks use explicitly configured model
environments and pinned sources. The checks reused existing CUDA
dependencies. A complete installation of all CUDA environments on another machine
remains untested. The installation instructions build native extensions from source.

## Revision check, 23 September 2026

The HAT dependencies now use their published shared-configuration names. Both
revisions were fetched and installed again. Their numerical settings are unchanged.
All eight saved observation replays and 40 prefix checks remain exact. The 29 CPU
tests, package builds and SRT3D build/self-check pass. The SRT3D usage and timing
schema strings now use the benchmark's name. See the
[revision receipt](release_validation_20260923.json).

## Fresh installation and subset checks, 26 September 2026

A new checkout and new Python environments were tested on an 8 GB RTX 3070
Laptop GPU. PicoPose used Python 3.9/PyTorch 2.0.0+cu118; MegaPose and DROID
used Python 3.10/PyTorch 2.7.1+cu118. All three DROID extensions were built
from pinned sources with CUDA 11.8 and GCC 11.5. Checkpoint hashes and repeated
source-helper checks passed. See the [dated receipt](release_validation_20260926.json).

All 34 CPU tests pass from source and against a newly installed wheel. Lint,
formatting, package builds and the four-identity CPU demo pass. Five new dataset
preparation regressions cover mesh patterns, official/flat SwissCube layouts and
SHIRT mesh units. Fresh inference also reproduced and fixed MegaPose dependency
and sparse-mesh failures. YCB-V resume preserves prediction, observation and
state artifacts exactly.

Both HAT methods ran through SpaceRGBbenchmark on 300 dense YCB-V inputs
(scene 48/object 6; five scored targets), 36 SwissCube inputs (seq_000400),
and 36 inputs from each of SHIRT's four streams. Each method emitted valid
SE(3) poses for all 480 inputs. Saved observations reproduce every pose exactly
through the API and standalone replay CLI. Five causal prefixes per stream
match exactly (60 checks across both methods).

These checks have material limits:

- SPARK 2024 was not rerun because the restricted dataset and matching inputs
  were unavailable. Earlier validation receipts are separate evidence.
- SwissCube's declared 393216-pixel DROID resolution exhausted this GPU.
  The successful development run explicitly uses 196608 pixels and buffer 64;
  this does not validate the published default on 8 GB. The frozen schedule
  and published resolution remain unchanged.
- SHIRT used a local Tango CAD whose correspondence to the historical calibrated
  origin is unverified. Its accuracy scores are not release evidence.
- YCB-V produced 138 valid DROID motion observations per method but too few
  usable anchors for fusion: 295/300 outputs were held. Pico-HAT also held
  35/36 outputs on SHIRT roe2_synthetic without initializing fusion. Valid
  poses and exact replay do not establish useful tracking accuracy.
- This is not a full-partition accuracy or repeated-FPS experiment. Metadata
  matches all 41 YCB-V streams, 100 SwissCube sequences (8,522 targets) and
  four SHIRT streams (9,484 targets), but neural inference used subsets only.
