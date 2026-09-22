# spacergbbenchmark

Benchmark RGB 6-DoF pose models on SPARK, YCB-V, SwissCube and SHIRT. The runner
handles inference, interrupted runs, scoring and reports. Model libraries run in
separate environments. Use the Cookiecutter adapter template to add a model.

Pico-HAT and Mega-HAT implement
[HAT: Hypothesis-Anchored Tracking for Video Monocular Spacecraft Pose Estimation](https://arxiv.org/pdf/2609.21597)
by André Lopo, Atabak Dehban and Rodrigo Ventura.

| Dataset | Complete evaluation partition | Targets |
|---|---|---:|
| SPARK 2024 Stream-2 | RT500–RT599 | 30,000 |
| YCB-V | Declared non-symmetric object targets in full video streams | 3,073 |
| SwissCube | Official test sequences 400–499, variable lengths | 8,522 |
| SHIRT | ROE1/ROE2, synthetic and lightbox | 9,484 |

SPARK uses the research development partition. YCB-V covers the declared
non-symmetric objects. SwissCube uses the **full official test**.
Exact target and input schedules are frozen in the
[protocol inventory](src/spacergbbenchmark/protocols/partitions.json).

## Try it without datasets or a GPU

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r environments/requirements-core.txt
.venv/bin/python -m pip install '.[data,plots,test]'
.venv/bin/spacergbbenchmark doctor
.venv/bin/spacergbbenchmark demo --out results/demo
```

The demo uses synthetic fixtures to check missing-output handling, CSV/JSON
metrics and plots. Its report is saved to `results/demo/run/report/REPORT.md`.
It does not measure model accuracy or FPS.

## Run your models

Follow [installation](docs/INSTALL.md) and [data preparation](docs/DATA.md) once.
Set data, checkpoint and interpreter paths in local configurations. Prepare
PicoPose/GigaPose templates using `prepare-model`. Then run a single model or the
[seven-model suite](configs/suites/four_datasets.json):

```bash
.venv/bin/spacergbbenchmark data prepare \
  --config configs/data.local.json --out results/data
.venv/bin/spacergbbenchmark data validate --datasets all
.venv/bin/spacergbbenchmark run --suite configs/suites/four_datasets.json \
  --model picopose --dataset spark --gpus 0 --run results/runs/pico-spark
.venv/bin/spacergbbenchmark run --suite configs/suites/four_datasets.json \
  --gpus 0 1 --jobs 2 --run results/runs/all-models
.venv/bin/spacergbbenchmark resume --run results/runs/all-models
.venv/bin/spacergbbenchmark report --suite-run results/runs/all-models
```

Inference automatically scores every declared target and writes a report.
Missing predictions remain visible in coverage. Completed sequences are reused
only when their inputs, code, configuration, environments and outputs match.
Interrupted stateful sequences restart from frame zero. Failed jobs retain logs
and do not prevent independent jobs from running.

Adapters are provided for **PicoPose, Pico-HAT, MegaPose, Mega-HAT, GigaPose with
MegaPose refinement, RGBTrack and SRT3D**. Read the
[validation record](docs/VALIDATION.md) for the checks completed for each adapter.
A short inference check is distinct from a full benchmark measurement.

## Accuracy and speed

Reports contain pose error, translation error, rotation error, ADD-S, sample SD,
coverage, per-target failures and matched-target comparisons. See the
[protocol](protocols/README.md) for units, object frames and input privileges.
SPARK/YCB-V use predicted localization. SwissCube/SHIRT use explicitly labeled
annotation-conditioned localization in the reproduction profile.

Fresh FPS is measured separately in three resident-model runs, after warm-up,
with exclusive resource ownership. Model loading, external detection and scoring
are excluded. Normal recovery, DROID and fusion stay inside the timed loop.

```bash
.venv/bin/spacergbbenchmark benchmark --suite configs/suites/four_datasets.json \
  --model pico_hat --dataset spark --stream RT500 --gpu 0 --warmup 30 \
  --out results/runs/all-models/timings/pico_hat/spark/RT500
.venv/bin/spacergbbenchmark report --suite-run results/runs/all-models
```

The timing command refuses occupied GPUs. It never stops other users' jobs.
A named-stream timing is reported with that scope, not as full-partition FPS.

## Extend and reproduce

- [Add a model with Cookiecutter](docs/ADDING_MODELS.md).
- [Import saved predictions and reproduce scoring](docs/SCORING.md).
- [Validation and known limits](docs/VALIDATION.md).
- [Third-party sources and attribution](THIRD_PARTY.md).
- [Design and scope](docs/IMPLEMENTATION_PLAN.md).

The project structure is based on Cookiecutter Data Science 2.3.0. The
[scaffold receipt](docs/scaffold.json) records its revision. Datasets, weights,
model environments and outputs stay outside Git. Configure these assets before
running a suite. Version 1 supports evaluation of existing models.

## Citation

For experiments using Pico-HAT or Mega-HAT, cite the [HAT paper](https://arxiv.org/pdf/2609.21597)
and the relevant model and dataset papers. [CITATION.cff](CITATION.cff) contains
the HAT reference.
