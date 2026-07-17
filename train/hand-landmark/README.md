# Sensorimotor Hand Tower — Review Before Running

This experiment does **not** recreate MediaPipe by throwing many unrelated RGB
datasets at a small CNN. It uses the frozen MediaPipe hand constellation as a
teacher for a compact, gradient-free sensorimotor geometry model, following the
same learning pattern as `CPU · Monty (3D evidence)`.

Read this file and the generated `plan.json` before every distillation or
training run. Expensive commands require the reviewed plan's SHA-256 token.

## 1. What the MediaPipe paper actually establishes

The procedure is grounded in *MediaPipe Hands: On-device Real-time Hand
Tracking* (`arXiv:2006.10214`):

- MediaPipe is a two-stage pipeline: a full-frame palm detector supplies an
  oriented crop to a separate 21-landmark model.
- Accurate cropping lets the landmark network spend capacity on geometry
  instead of translation, scale and rotation invariance.
- During video tracking, the previous landmark constellation predicts the next
  crop. The palm detector runs initially and only when tracking confidence
  falls below a threshold.
- The landmark model outputs 21 x/y/relative-z coordinates, crop-presence and
  handedness.
- The paper used 6K diverse in-the-wild images, 10K in-house gesture images,
  and 100K rendered images. Real images supervised x/y; synthetic renders
  supplied depth supervision. The combined dataset performed best and reduced
  temporal jitter.

Source: <https://ar5iv.labs.arxiv.org/html/2006.10214>

This matters because relative depth cannot be recovered merely by relabelling
Jester or HaGRID RGB frames. For this experiment, MediaPipe's frozen z output
is the teacher sensation. It is not presented as independent ground truth.

## 2. What is inside `hand_landmarker.task`

`train/weights/hand_landmarker.task` is a ZIP-compatible inference bundle:

| member | bytes | purpose |
|---|---:|---|
| `hand_detector.tflite` | 2,339,878 | palm/ROI detector |
| `hand_landmarks_detector.tflite` | 5,478,949 | cropped 21-point model |

Inspect and optionally extract it without executing inference:

```bash
.venv/bin/python train/hand-landmark/offline.py teacher
.venv/bin/python train/hand-landmark/offline.py teacher \
  --extract train/hand-landmark/teacher-models
```

The bundle contains weights and inference operators, not Google's training
images, synthetic renderer, losses, optimizer state or training graph. There
is no reliable operation that turns TFLite inference weights back into the
original trainer. “Reverse engineering” here therefore means reproducible
behavioral distillation of the teacher's constellations.

## 3. Target architecture

```text
initial RGB frame
  -> retained lightweight palm detector
  -> canonical hand crop
  -> Sensor Module samples a small RGB patch
  -> sensorimotor policy chooses the next predicted joint location
  -> patch search confirms or rejects that sensation
  -> Monty Learning Module updates hand/pose hypotheses
  -> completed 21-point constellation
  -> next video frame ROI predicted from the previous constellation
  -> palm detector reruns only when evidence collapses
```

The run implemented now learns the middle **geometry substrate**:

```text
MediaPipe 21 x 2.5D constellation
  -> ordered sensorimotor episode
  -> palm-anchor scan
  -> finger-chain movements
  -> additive hand reference frames
  -> predict next joint before teacher reveals it
```

It does not claim to replace pixels→landmarks yet. The later pixel Sensor
Module must search patches around Monty's predicted next locations. Until that
passes camera validation, MediaPipe remains the reference.

For the real TBP implementation, follow
[`train/tbp-monty-guide.md`](../tbp-monty-guide.md). On Apple Silicon it lives
in a separate osx-64/Rosetta conda environment. The current experiment uses the
compatible lightweight protocol in `train/monty_lab/`, allowing the learned
`objects.npz` reference frames to cross that process boundary later.

## 4. Sensorimotor observation contract

One MediaPipe hand detection becomes one episode. Coordinates are converted to
a wrist-origin, scale-normalized object reference frame; orientation is kept.

An observation contains:

```text
location: current joint x/y/relative-z in the hand reference frame
feature:  joint/pixel-patch evidence supplied by the Sensor Module
action:   3D displacement from the previous sensed joint
pose:     rotation aligning the partial observation to a hand hypothesis
```

The current offline run teaches locations and actions from MediaPipe. The RGB
patch feature adapter is deliberately a later stage, so geometry can be proved
before introducing appearance ambiguity.

## 5. Exploration policies

### Palm-anchor scan

Begin with wrist and MCP anchors `[0, 5, 9, 13, 17]`. These define the palm's
reference frame and give enough non-collinear locations for a pose hypothesis.

### Finger-chain scan

Move outward along the fixed anatomical chains:

```text
thumb  0 -> 1 -> 2 -> 3 -> 4
index  5 -> 6 -> 7 -> 8
middle 9 -> 10 -> 11 -> 12
ring   13 -> 14 -> 15 -> 16
pinky  17 -> 18 -> 19 -> 20
```

Before each movement completes, the current best reference-frame hypothesis
predicts the next 3D joint location. The teacher location is then revealed and
becomes new evidence. Prediction error—not reconstruction after seeing the
whole hand—is the learning metric.

### Random walk

After deterministic scans work, a seeded random walk over the same anatomical
graph tests whether the model learned geometry rather than memorizing one
ordering. Random policy results must remain reproducible by storing the seed.

### Hypothesis testing

When two hand-pose prototypes have similar evidence, move toward the joint
whose predicted locations disagree most. This is the later active-inference
policy; it is preferable to blindly scanning all 21 points.

“Spiral scan” and curvature-following are appropriate for continuous object
surfaces. A hand constellation is a sparse articulated graph, so the analogous
policy is palm anchors followed by anatomical chains—not a fake image-space
spiral.

## 6. Learning rule

Learning is additive and uses no backpropagation:

1. normalize a teacher constellation to the wrist reference frame;
2. align it against every stored prototype with batched Kabsch pose solves;
3. if an existing prototype explains it below the novelty RMS threshold, do
   not store a duplicate;
4. otherwise append it as a new articulated hand reference frame;
5. store mean/std movement vectors for each anatomical edge.

Changing one prototype does not retrain or damage the others. The artifact is:

```text
train/runs/sensorimotor_hand/objects.npz
train/runs/sensorimotor_hand/meta.json
```

## 7. Compact data curriculum

This is intentionally bounded. We need pose and camera variation, not every
frame in every dataset.

| role | source | planned maximum | reason |
|---|---|---:|---|
| target-domain exploration | `train/datasets/crude/prepared.npz` | 600 | exact camera/lighting and held gestures |
| cross-subject pose exploration | `train/datasets/hagrid_shapes/prepared.npz` | 800 | different people, backgrounds and static articulations |
| temporal exploration | `train/datasets/jester/prepared.npz` | 800 | ordered motion and viewpoint change |
| untouched referee | `train/datasets/yolo26/prepared.npz` | 500 | foreign-domain prediction test |

Not used for geometry learning:

- `train/datasets/nvgesture/raw/` is empty.
- `crops_classifier.onnx` predicts motion phase, not 3D joint locations. It may
  gate mid-motion frames later but is not a geometry teacher.
- COCO body keypoints are irrelevant to this hand-constellation experiment.
- A massive InterHand/FreiHAND mixture is not required for this Monty
  substrate experiment; it would be relevant only for a separately trained
  gradient-based pixel landmarker.

Run the authoritative audit:

```bash
.venv/bin/python train/hand-landmark/offline.py audit
```

## 8. Review-gated run procedure

### A. Generate and review the exact plan

```bash
.venv/bin/python train/hand-landmark/offline.py plan \
  --out train/hand-landmark/plan.json
```

Record the printed `PLAN_SHA256`. Any edit invalidates it.

### B. Bounded teacher harvest

Use exactly the limits and strides recorded in the plan:

```bash
.venv/bin/python train/hand-landmark/offline.py distill \
  --plan train/hand-landmark/plan.json --approve REVIEWED_SHA256 \
  --source crude --limit 600 --stride 3
.venv/bin/python train/hand-landmark/offline.py distill \
  --plan train/hand-landmark/plan.json --approve REVIEWED_SHA256 \
  --source hagrid_shapes --limit 800 --stride 5
.venv/bin/python train/hand-landmark/offline.py distill \
  --plan train/hand-landmark/plan.json --approve REVIEWED_SHA256 \
  --source jester --limit 800 --stride 10
.venv/bin/python train/hand-landmark/offline.py distill \
  --plan train/hand-landmark/plan.json --approve REVIEWED_SHA256 \
  --source yolo26 --limit 500 --stride 2
```

Each frame is also evaluated after horizontal flipping. Inverted landmark
coordinates must agree within six pixels and handedness confidence must exceed
the plan threshold. A miss is discarded as unknown; it is never made negative.

The resulting coordinate-only shards live under
`train/hand-landmark/curriculum/*.teacher.npz` and reference the source frame
indices rather than duplicating images.

### C. Gradient-free sensorimotor learning

```bash
.venv/bin/python train/hand-landmark/offline.py train \
  --plan train/hand-landmark/plan.json --approve REVIEWED_SHA256 \
  --input train/hand-landmark/curriculum/crude.teacher.npz \
  --input train/hand-landmark/curriculum/hagrid_shapes.teacher.npz \
  --input train/hand-landmark/curriculum/jester.teacher.npz \
  --referee train/hand-landmark/curriculum/yolo26.teacher.npz \
  --novelty-rms 0.10 --max-prototypes 256
```

The referee shard is never considered by the additive prototype learner. The
command prints and saves this role, every provenance hash, prototype count,
next-location NME/PCK, and per-step latency.

## 9. Acceptance gates

The geometry substrate is accepted only if held-out episodes achieve:

- PCK@0.10 ≥ 0.50 for **next-location prediction**;
- PCK@0.20 ≥ 0.80;
- median hypothesis/prediction step ≤ 1 ms;
- no more than 256 stored prototypes.

These are not pixel-tracking acceptance gates. Promoting a MediaPipe-free GUI
runtime additionally requires:

- RGB patch Sensor Module validated on live camera frames;
- tracking dropout ≤ 5%;
- p95 visible jitter ≤ 4 pixels;
- wrong robot action ≤ 5%;
- detector fallback proven to recover lost tracks.

If the substrate fails, inspect failures by exploration step and pose. Do not
add unrelated datasets reflexively. Append constellations specifically for the
weak articulation or viewpoint, then rerun the held-out prediction proof.

## 10. Recorded bounded proof (2026-07-13)

The reviewed plan admitted 730 exploration episodes and produced 255
prototypes. The evaluator used 181 held-out exploration episodes plus all 263
accepted YOLO26 referee episodes:

| evaluation | PCK@0.10 | PCK@0.20 | median step |
|---|---:|---:|---:|
| exploration holdout | 0.577 | 0.914 | 0.671 ms |
| untouched YOLO26 referee | 0.521 | 0.862 | 0.672 ms |
| combined | 0.544 | 0.883 | 0.674 ms |

The geometry substrate therefore passes the gates above. This proves bounded
next-sensation prediction and a small recurrent hypothesis core; it does not
yet prove RGB-to-landmark perception. That promotion remains blocked on the
pixel Sensor Module and live-camera gates listed above.

## 10. Reproducibility and cost controls

- No command downloads data.
- Teacher bundle, plan and curriculum shards are SHA-256 hashed.
- Expensive commands require the reviewed plan hash.
- Limits and strides bound teacher wall-clock time.
- Validation frames never become prototypes.
- Learning is CPU/NumPy and runs on macOS, Raspberry Pi-class CPUs, or a cloud
  machine; no GPU is required for the Monty geometry stage.
- The real `tbp.monty` environment remains separate from OpenPAVE's MLX venv as
  required by `train/tbp-monty-guide.md`.
