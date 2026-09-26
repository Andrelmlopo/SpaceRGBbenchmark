# Benchmark setup

SpaceRGBbenchmark needs Python 3.10 or newer. Dataset preparation, scoring and
the demo run on CPU.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r environments/requirements-core.txt
.venv/bin/python -m pip install '.[data,plots]'
.venv/bin/spacergbbenchmark doctor
```

Next, [prepare the datasets](DATA.md). To evaluate saved predictions, use the
[import and scoring commands](DATA.md#import-saved-predictions).

## Connect installed models

Install each model using its own documentation, then point the benchmark at
those environments and checkpoints:

| Models | Installation guide |
|---|---|
| PicoPose, Pico-HAT and DROID-SLAM | [Pico-HAT](https://github.com/Andrelmlopo/Pico-HAT/blob/main/docs/INSTALL.md) |
| MegaPose, Mega-HAT and DROID-SLAM | [Mega-HAT](https://github.com/Andrelmlopo/Mega-HAT/blob/main/docs/INSTALL.md) |
| GigaPose | [GigaPose](https://github.com/nv-nguyen/gigapose) |
| RGBTrack | [RGBTrack](https://github.com/GreatenAnoymous/RGBTrack) |
| SRT3D | [3DObjectTracking](https://github.com/DLR-RM/3DObjectTracking) |

Set interpreter, source and checkpoint fields in `configs/models/`. Paths
resolve from the configuration file. Reuse existing model environments.
`mega_data` points to the directory containing `megapose-models/`.

The HAT adapters also need their Python tracker packages in the benchmark
coordinator. Install the pinned versions with:

```bash
python3 scripts/fetch_sources.py --sources Pico-HAT Mega-HAT
.venv/bin/python -m pip install ./third_party/Pico-HAT ./third_party/Mega-HAT
```

The PicoPose and DROID workers use Pico-HAT's runtime, so `pico-hat` must also
be installed in their selected environments. YCB mask decoding needs
`pycocotools` in workers that read localization. The coordinator supplies the
benchmark worker code, so it does not need installing in each model environment.

GigaPose uses a MegaPose refinement worker and the PicoPose template renderer.
SRT3D uses a MegaPose initializer. Source revisions and the required MegaPose
patch are recorded in `environments/sources.json`. The fetch helper can obtain
selected sources with `--sources`. The GigaPose and RGBTrack dependency
inventories in `environments/` record the adapter versions tested.

## Prepare benchmark adapters

After configuring a model, prepare its templates from the dataset manifests:

```bash
.venv/bin/spacergbbenchmark prepare-model --suite configs/suites/four_datasets.json \
  --dataset spark --model picopose --gpu 0
```

Repeat for each dataset you will evaluate. PicoPose and Pico-HAT share template
banks. Use `--model gigapose_refined` for GigaPose's converted banks. MegaPose
renders its coarse views during model setup.

For SRT3D, build the benchmark's native adapter after installing its dependencies:

```bash
python3 scripts/build_srt3d.py --source third_party/SRT3D \
  --build results/build/srt3d --jobs 4
```

Set `srt3d_executable` to the resulting executable. Headless runs also need
`xvfb` and `xauth`. See the [README](../README.md#run-your-models) for running,
resuming and reporting benchmark suites.
