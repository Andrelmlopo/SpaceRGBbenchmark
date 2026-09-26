# Installation

The coordinator needs Python 3.10 or newer. Native GPU models are Linux/CUDA
integrations with separate interpreters. The short GPU checks use an NVIDIA
A100. See [validation](VALIDATION.md) for what was tested on a fresh installation
and what was exercised in an existing model environment.

## Coordinator and pinned sources

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r environments/requirements-core.txt
.venv/bin/python -m pip install '.[data,plots,test]'
python3 scripts/fetch_sources.py
.venv/bin/python -m pip install ./third_party/Pico-HAT ./third_party/Mega-HAT
.venv/bin/spacergbbenchmark doctor
```

`environments/sources.json` pins every upstream revision. `fetch_sources.py`
fetches their recorded submodules, applies the MegaPose patch and verifies any
existing checkout before reuse. Select only needed sources with `--sources`.
The MegaPose patch covers interpreter lookup, NumPy 2, EGL rendering and explicit
glTF axis preservation. Sparse CAD meshes use vertex sampling with replacement
when MegaPose requests more points than the mesh contains; dense-mesh sampling
is unchanged. The patch changes no benchmark labels or model weights.

## PicoPose and templates

```bash
python3.9 -m venv .venv-pico
.venv-pico/bin/python -m pip install 'pip<25' 'setuptools<81' wheel
.venv-pico/bin/python -m pip install torch==2.0.0 torchvision==0.15.1 \
  --index-url https://download.pytorch.org/whl/cu118
.venv-pico/bin/python -m pip install xformers==0.0.18
.venv-pico/bin/python -m pip install -r environments/requirements-pico.txt
.venv-pico/bin/python -m pip install ./third_party/Pico-HAT
.venv-pico/bin/python -m pip check
```

The coordinator sends the benchmark worker package on `PYTHONPATH`; do not
install the Python ≥3.10 coordinator into the Python 3.9 environment. The worker
code used by PicoPose supports Python 3.9. `pycocotools` decodes YCB CNOS masks.
The template renderer requires EGL/OpenGL driver libraries. PyOpenGL used by
pyrender is incompatible with Python 3.12+, hence the separate environment.

## DROID-SLAM

Install a CUDA toolkit with `nvcc`, a C++ compiler and Ninja. Set `CUDA_HOME` to
that toolkit and put its `bin` directory on `PATH`. A driver alone cannot build
these extensions. Limit build concurrency with `MAX_JOBS=4` if necessary.

```bash
python3.10 -m venv .venv-droid
.venv-droid/bin/python -m pip install 'setuptools<81' wheel ninja
.venv-droid/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu118
.venv-droid/bin/python -m pip install -r environments/requirements-droid.txt
.venv-droid/bin/python -m pip install --no-build-isolation \
  ./third_party/DROID-SLAM/thirdparty/lietorch
.venv-droid/bin/python -m pip install --no-build-isolation \
  ./third_party/DROID-SLAM/thirdparty/pytorch_scatter
.venv-droid/bin/python -m pip install --no-build-isolation ./third_party/DROID-SLAM
.venv-droid/bin/python -m pip install ./third_party/Pico-HAT
```

Build these extensions on the target installation. No compiled research-workspace
binaries are distributed. Both HAT methods use the pinned causal DROID frontend.
`droid_buffer` is explicit and must hold a sequence's retained keyframes. A full
buffer raises an error with a concrete remedy, rather than silently truncating.

## MegaPose and Mega-HAT

```bash
python3.10 -m venv .venv-mega
.venv-mega/bin/python -m pip install 'setuptools<81' wheel
.venv-mega/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu118
.venv-mega/bin/python -m pip install -r environments/requirements-mega.txt
.venv-mega/bin/python -m pip check
.venv-mega/bin/python -c "import pinocchio, png"
```

The adapter imports the pinned MegaPose checkout directly. Keep the NVIDIA EGL
driver available. If the machine needs `__EGL_VENDOR_LIBRARY_FILENAMES`, use its
own vendor JSON. Set `mega_data` to the directory containing `megapose-models/`.
The named RGB multi-hypothesis model needs both coarse and refiner weights.
Keep the TinyXML 10 pin: the Pinocchio/urdfdom binaries require
`libtinyxml2.so.10`. The bundled BOP toolkit also imports `pypng`. The import
check detects these runtime requirements in addition to package metadata checks.

## GigaPose with MegaPose refinement

Use Python 3.9 and the pinned upstream environment. The observed compatible
package inventory is `environments/giga-validation-freeze.txt`. It records the
specific torch 2.0.0/cu118, torchvision 0.15.1, xformers 0.0.18 and Lightning
1.8.1 environment used for validation. To reproduce it:

```bash
python3.9 -m venv .venv-giga
.venv-giga/bin/python -m pip install 'pip<25' 'setuptools<81' wheel
.venv-giga/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  -r environments/giga-validation-freeze.txt
```

The frozen source tree supplies native retrieval and pose recovery. The pinned
DINOv2 source is loaded locally. GigaPose's five candidates are refined and
rescored in the separate MegaPose worker. This method's costs include both
workers. `prepare-model --model gigapose_refined` creates per-object banks from
the metric Pico template renderer. This is new renderer provenance, not a claim
of historical GigaPose output equality.

## RGBTrack

The observed dependency inventory is `environments/rgbtrack-validation-freeze.txt`.
Use Python 3.10, the pinned upstream source and its XMem submodule. The inventory
pins nvdiffrast and PyTorch3D source revisions. Compilation requires CUDA, Eigen3,
CMake and a C++ compiler.

```bash
python3.10 -m venv .venv-rgbtrack
.venv-rgbtrack/bin/python -m pip install 'pip<25' 'setuptools<81' wheel ninja
.venv-rgbtrack/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  -r environments/rgbtrack-validation-freeze.txt
cmake -S third_party/RGBTrack/mycpp -B third_party/RGBTrack/mycpp/build \
  -DPYTHON_EXECUTABLE="$PWD/.venv-rgbtrack/bin/python"
cmake --build third_party/RGBTrack/mycpp/build -j 4
.venv-rgbtrack/bin/python -m pip install --no-build-isolation \
  ./third_party/RGBTrack/bundlesdf/mycuda
```

Place FoundationPose's published scorer/refiner weight directories under
`third_party/RGBTrack/weights/` and the XMem checkpoint at
`third_party/RGBTrack/XMem/saves/XMem.pth`, following the pinned upstream READMEs.
The adapter uses RGB-only initialization and XMem tracking, without ground-truth
pose initialization or depth images. Internal recovery is included in timing.

## SRT3D

Install Eigen3, OpenCV development libraries, GLFW, GLEW, OpenGL, CMake, g++,
`pkg-config`, `xvfb` and `xauth`. Build the library and interactive adapter:

```bash
python3 scripts/build_srt3d.py --source third_party/SRT3D \
  --build results/build/srt3d --jobs 4
```

Set `srt3d_executable` in its model configuration. The CPU tracker receives a
MegaPose initializer, so this adapter still needs a GPU for initialization. On
headless hosts it starts its own Xvfb display and software OpenGL renderer.
No existing display or unrelated process is stopped.

## Weights and runtime configuration

```bash
python3 scripts/fetch_assets.py --catalog environments/downloadable_assets.json --out weights
```

That catalog contains the official MegaPose files and GigaPose checkpoint with
validated SHA-256 values. PicoPose and DROID authors distribute their weights
through Google Drive; their links and digests are in
`environments/manual_assets.json`. Place those files at `weights/picopose.ckpt`
and `weights/droid.pth`. Manual placement is supported for every asset. Run
identities record their actual hashes, so changed checkpoints invalidate reuse.

Edit the JSON files in `configs/models/`, or copy them to `*.local.json` and
reference those copies from a local suite. Paths resolve from each model
configuration, not from the shell's working directory. Keep copies in the same
directory when using the example relative paths. Additional untracked model
assets can be declared in `asset_paths` for content hashing. Package search
roots belong in `package_paths` and are also hashed.

After [preparing data](DATA.md), prepare each required object bank:

```bash
.venv/bin/spacergbbenchmark prepare-model --suite configs/suites/four_datasets.json \
  --dataset spark --model picopose --gpu 0
```

Repeat for the other three datasets. The command handles every object in a
dataset and verifies existing banks. PicoPose and Pico-HAT share banks. GigaPose
preparation additionally converts them to its template format. MegaPose renders
its coarse views during resident-model setup.

If public HTTPS Git access is unavailable but GitHub SSH is configured, use
`python3 scripts/fetch_sources.py --ssh`. Interrupted downloads preserve their
directories for diagnosis; select a fresh `--out` directory for a clean fetch.
The `*-validation-freeze.txt` files are observed inventories, not a claim that
all platform-specific wheels remain available for every OS and Python version.

## GPU memory for development subsets

An 8 GB RTX 3070 Laptop GPU completed both HAT subset checks with DROID
resolution 196608 pixels and buffer 64. SwissCube's declared 393216-pixel
setting ran out of memory on that GPU even with buffer 64. Use a GPU with
more available memory to validate that declared setting. For a compatibility
check, change the resolution only in an explicitly labeled local development
configuration and regenerate its manifest; this changes the experiment. Keep
full-run buffer sizing and the published protocol separate from short-run
memory settings. See [validation limits](VALIDATION.md).
