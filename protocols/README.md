# Four-dataset protocol v1

The packaged `partitions.json` freezes stream names, object IDs, target counts,
input counts and SHA-256 digests of ordered frame IDs. Full manifests must match
all fields, not just their total counts. Sequence JSON files contain inference
inputs only. Label poses live in a separate scorer-only archive. This is a
software interface boundary, not an operating-system security sandbox.

SPARK has 100 streams and 30,000 targets. YCB-V has 41 object streams, 72,306
input frames and 3,073 scored targets. Independent models process only declared
target timestamps. Temporal models process dense video. CNOS detections are
category-specific and available only at declared timestamps. An absent detection
remains absent. No previous box or GT box is substituted.

SwissCube has 100 official test streams, IDs 400–499, totaling 8,522 frames.
Lengths vary. There is no 500-image benchmark profile. SHIRT comprises four
ROE1/ROE2 synthetic/lightbox streams and 9,484 targets.

## Input privileges

| Dataset | Pose frontend | Relative-motion input |
|---|---|---|
| SPARK | Predicted Mask2Former masks and boxes | RGB masked by those predictions |
| YCB-V | Category-specific CNOS/FastSAM detections | Full RGB |
| SwissCube | Annotation mask bounding box, 15% padding each side | RGB masked by annotated visible silhouette |
| SHIRT | Projected annotation-conditioned CAD bounding box, 20% padding each side | Full RGB |

These policies reproduce the declared research input classes. They are not an
all-datasets automatic detection experiment. No adapter receives ground-truth
pose initialization. SRT3D initializes from MegaPose and RGBTrack initializes
from RGB, localization and CAD geometry. New frontend/template rendering is
recorded separately from historical predictions.

## Pose and metric contract

Every prediction is `T_CO`: object coordinates to OpenCV camera coordinates
(x right, y down, z forward), translation in metres. A manifest explicitly
names mesh units and `raw_to_object`. A source-mesh pose converts with
`T_benchmark = T_source @ inverse(raw_to_object)`. SPARK uses the recorded
Panda z-up frame. No mesh is silently recentered. Rotation must be a proper
orthonormal matrix. Missing outputs are invalid, with NaN poses and a reason.

`E_p = E_q_radians + ||t_pred - t_gt|| / max(||t_gt||, 1e-6)`.
Report translation error in metres, rotation error in degrees, and ADD-S in
metres. ADD-S uses a fixed 2,000-point mesh bank in the named object frame.
Historical exact replay requires the historical point bank. Without one,
preparation generates a new deterministic bank and records its hash. The new
bank can change ADD-S while leaving the other metrics unchanged.

The SHIRT pose metric converts both predicted and true CAD poses to the
calibrated label origin using `t += R @ delta`, `R = R @ R_align.T`.
ADD-S uses the CAD origin. The [calibration](shirt_calibration.json) is a frozen
historical, test-informed reproduction artifact, not a newly fitted parameter.

Means pool valid targets and SD uses `ddof=1`. Coverage always uses every
declared target, including failures. Reports additionally compute the intersection
of valid targets across methods. Different input manifests cannot be merged as
one comparison. These metrics are distinct from official BOP symmetry-aware
recall, which is not implemented in v1.

## Reproduction scope

Pico-HAT and Mega-HAT implement the [HAT paper](https://arxiv.org/pdf/2609.21597).
They use pinned standalone packages with DROID, causal
hypothesis selection, both fusion stages and hold fill. Their numerical recipes
are defined by their pinned package versions. Generic frontend rendering and
localization conversion can differ from earlier research scripts, so fresh
outputs require new accuracy measurements. Historical test-informed development
must remain disclosed. No manuscript is changed by the benchmark runner.
