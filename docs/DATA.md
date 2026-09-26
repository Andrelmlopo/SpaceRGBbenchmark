# Prepare the four datasets

Obtain RGB images, labels, camera calibration and CAD assets from the dataset
owners. Asset access and licensing remain with those sources. Preparation accepts
existing files and never downloads a multi-gigabyte dataset implicitly.

- SPARK 2024: [official release](https://zenodo.org/records/10908215), Stream-2.
  Requires external access approval from the dataset owners.
- YCB-V: [BOP YCB-Video](https://bop.felk.cvut.cz/datasets/#YCB-V), including dense
  `test_all` video inputs, BOP targets and category-specific CNOS/FastSAM detections.
- SwissCube: the authors' [Wide-Depth-Range-Pose](https://github.com/cvlab-epfl/wide-depth-range-pose)
  release, complete test sequences 400–499.
- SHIRT: the Stanford [Space Rendezvous Laboratory datasets](https://slab.stanford.edu/projects/datasets),
  ROE1 and ROE2 in both synthetic and lightbox domains.

Use the supplied `configs/datasets/native.json` directly. Set `SPACERGB_DATA_ROOT`
to the directory containing your datasets. The SHIRT calibration is included in
the package and loaded automatically.

```bash
export SPACERGB_DATA_ROOT="$PWD/data"
.venv/bin/spacergbbenchmark data prepare --config configs/datasets/native.json --out results/data
.venv/bin/spacergbbenchmark data validate --datasets all
```

If your asset layout differs, customize the corresponding fields in a local
configuration. Mesh units and object-frame transforms remain explicit. The model
filenames are examples; configure the real filename instead of renaming an asset.
An explicit SHIRT `calibration` field overrides the included default and resolves
relative to your configuration file.

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

SwissCube accepts `testing/seq_000400/000000/` and flat `seq_000400/` layouts,
with `rgb/<six-digit-frame>.jpg` (or PNG), `scene_camera.json`,
`scene_gt.json` and `mask_visib/<frame>_000000.png`. Provide per-sequence box JSON
files under `boxes`, mapping frame IDs to tight xyxy silhouette boxes. The loader
adds the explicit padding. The separate silhouette remains DROID's mask input.
All 100 variable-length streams must be present. Extra historical subset folders
are ignored because the exact stream IDs are explicit.

SHIRT uses `camera.json`, `roe1/roe1.json`, `roe2/roe2.json`, and
`<trajectory>/<synthetic-or-lightbox>/images/<filename>`. The included historical
calibration maps the original uncoloured Tango CAD to its label origin. Preparation derives
annotation-conditioned boxes before writing inference manifests. Reports label
that privilege, and adapters never read the label pose archive.

## Manifests and mesh samples

Preparation writes `dataset.json`, hashed `streams/*.json`, a separate
`truth.npz`, and ADD-S sample references. Sequence manifests expose only images,
intrinsics, localization, CAD, object-frame information and target flags.

For exact historical ADD-S reproduction, configure `mesh_points` as a path pattern
such as `${SPACERGB_DATA_ROOT}/frozen/obj_{object_id:06d}.npy`, containing the original 2,000
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
.venv/bin/python scripts/prepare_swiss_boxes.py --root "$SPACERGB_DATA_ROOT/swisscube/testing" \
  --out "$SPACERGB_DATA_ROOT/swisscube/boxes"
```

This command deliberately uses annotated masks and preserves the protocol's
annotation-conditioned label. It refuses to overwrite differing box files.

## Evaluation

The [partition inventory](../src/spacergbbenchmark/protocols/partitions.json)
fixes all stream, object and frame IDs. Framewise methods process scored targets
only. Temporal methods receive full video and are scored at the same timestamps.
YCB-V has 41 object streams and 72,306 input frames. Missing category-specific
detections remain missing. Pose labels are stored separately from inference inputs.

| Dataset | Pose localization | DROID input |
|---|---|---|
| SPARK | Predicted Mask2Former masks and boxes | Masked RGB |
| YCB-V | Category-specific CNOS/FastSAM detections | Full RGB |
| SwissCube | Annotated mask box, 15% padding | Annotated silhouette-masked RGB |
| SHIRT | Annotation-projected CAD box, 20% padding | Full RGB |

Predictions are object-to-camera transforms `T_CO`, with translations in metres
and OpenCV axes: x right, y down, z forward. Mesh origins are preserved. A source
pose converts as `T_benchmark = T_source @ inverse(raw_to_object)`.

Pose error is `E_p = E_q_radians + ||t_pred - t_gt|| / max(||t_gt||, 1e-6)`.
Reports also give translation error in metres, rotation error in degrees, ADD-S,
sample SD (`ddof=1`) and coverage. Error means use valid predictions, while
coverage includes every declared target. Matched comparisons use common valid
targets. These metrics differ from official BOP symmetry-aware recall.

SHIRT's pose metric applies `t += R @ delta`, then `R = R @ R_align.T`, to both
predicted and true CAD poses. ADD-S uses the CAD origin. The bundled calibration
and historical settings are test-informed. Annotation-conditioned localization
is reported explicitly. Adapters do not receive ground-truth pose initialization.

## Import saved predictions

Use one NPZ per stream, with `poses` shaped `(N, 4, 4)` and integer `frames`:

```bash
.venv/bin/spacergbbenchmark import \
  --dataset results/data/spark/dataset.json --source results/saved_predictions \
  --out results/imported/spark --format npz --units m \
  --pattern '{stream}/poses.npz' --pose-key poses --frame-key frames
.venv/bin/spacergbbenchmark score \
  --dataset results/data/spark/dataset.json --predictions results/imported/spark \
  --out results/saved/metrics/method/spark --method method
.venv/bin/spacergbbenchmark report --suite-run results/saved
```

Without frame IDs, specify `--schedule inputs` or `--schedule targets`. Use
`--format bop` for BOP CSVs, `--candidate-scores` for top-K archives, and
`--source-to-object` for a different CAD frame. Imported results cannot supply
fresh FPS. Missing predictions remain in coverage. ADD-S needs mesh samples,
or `--no-adds` explicitly omits it.
