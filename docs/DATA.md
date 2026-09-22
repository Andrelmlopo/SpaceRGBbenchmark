# Prepare the four datasets

Obtain RGB images, labels, camera calibration and CAD assets from the dataset
owners. Asset access and licensing remain with those sources. Preparation accepts
existing files and never downloads a multi-gigabyte dataset implicitly.

- SPARK 2024: [official release](https://zenodo.org/records/10908215), Stream-2.
- YCB-V: [BOP YCB-Video](https://bop.felk.cvut.cz/datasets/#YCB-V), including dense
  `test_all` video inputs, BOP targets and category-specific CNOS/FastSAM detections.
- SwissCube: the authors' [Wide-Depth-Range-Pose](https://github.com/cvlab-epfl/wide-depth-range-pose)
  release, complete test sequences 400–499.
- SHIRT: the Stanford [Space Rendezvous Laboratory datasets](https://slab.stanford.edu/projects/datasets),
  ROE1 and ROE2 in both synthetic and lightbox domains.

Copy `configs/datasets/native.json` to `configs/data.local.json`. Because this
moves the file up a directory, change the SHIRT calibration path to
`../protocols/shirt_calibration.json`. Replace `${SPACERGB_DATA_ROOT}` with an
exported data root, or replace each field with the actual existing path. Keep
mesh units and the object-frame transform explicit. The shipped model filenames
are examples; no file rename is required when you configure its real path.

```bash
export SPACERGB_DATA_ROOT=/path/to/data
.venv/bin/spacergbbenchmark data prepare --config configs/data.local.json --out results/data
.venv/bin/spacergbbenchmark data validate --datasets all
```

`--deep` additionally opens every RGB/mask and validates dimensions/localization.
The normal check verifies file existence, calibration, manifest checksums and
the exact frozen frame schedule. Inference identities hash image and mask bytes
before executing a sequence. Existing output directories are preserved.

## Native layout contract

SPARK uses `images/RT500/img000.jpg` through RT599, with actual filenames from
`train.csv`. The CSV needs sequence, filename, Tx/Ty/Tz and Qx/Qy/Qz/Qw columns.
Boxes are one JSON per sequence, mapping the full image stem to exclusive xyxy
pixels. Masks are `masks/RT500/masks/<image-stem>.png`. The stock Mask2Former
prediction files are required preprocessing inputs. The adapter does not generate
or tune detections. Colored masks explicitly use any nonzero channel as foreground.

YCB uses `<scene>/rgb/<six-digit-frame>.png`, `scene_camera.json`, ground truth
`scene_gt.json`, `test_targets_bop19.json`, and the CNOS JSON containing scene_id,
image_id, category_id, bbox, score and optional COCO mask RLE. CAD files follow
`obj_<six-digit-object>.ply` in millimetres. The non-symmetric object list is
frozen by the code and target inventory. Only the matching object's highest-score
detection is eligible. Missing detection is not a missing RGB frame.

SwissCube uses `seq_000400/rgb/<six-digit-frame>.png`, `scene_camera.json`,
`scene_gt.json` and `mask_visib/<frame>_000000.png`. Provide per-sequence box JSON
files under `boxes`, mapping frame IDs to tight xyxy silhouette boxes. The loader
adds the explicit padding. The separate silhouette remains DROID's mask input.
All 100 variable-length streams must be present. Extra historical subset folders
are ignored because the exact stream IDs are explicit.

SHIRT uses `camera.json`, `roe1/roe1.json`, `roe2/roe2.json`, and
`<trajectory>/<synthetic-or-lightbox>/images/<filename>`. The included historical
calibration maps the configured Tango CAD to its label origin. Preparation derives
annotation-conditioned boxes before writing inference manifests. Reports label
that privilege, and adapters never read the label pose archive.

## Manifests and mesh samples

Preparation writes `dataset.json`, hashed `streams/*.json`, a separate
`truth.npz`, and ADD-S sample references. Sequence manifests expose only images,
intrinsics, localization, CAD, object-frame information and target flags.

For exact historical ADD-S reproduction, configure `mesh_points` as a path pattern
such as `/path/to/frozen/obj_{object_id:06d}.npy`, containing the original 2,000
metric points in the benchmark object frame. If absent, preparation generates a
new deterministic surface bank and records its recipe. It is suitable for a new
shared comparison but may change historical ADD-S values. The other three metrics
are independent of mesh sampling.

Data files are not copied into Git. Generated manifests may reference absolute
local paths and live under ignored `results/`. To move an experiment to another
machine, prepare manifests there from the same assets and archive the resulting
input/asset receipts with the run.

Generate the SwissCube box JSON files directly from its visible masks with:

```bash
.venv/bin/python scripts/prepare_swiss_boxes.py --root /path/to/swisscube \
  --out /path/to/swisscube/boxes
```

This command deliberately uses annotated masks and preserves the protocol's
annotation-conditioned label. It refuses to overwrite differing box files.
