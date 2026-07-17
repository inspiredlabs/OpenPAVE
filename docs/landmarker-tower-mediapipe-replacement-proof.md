# Proving a Lightweight Landmarker Tower Can Replace MediaPipe

## Purpose

This document defines the shortest credible path from the current OpenPAVE hand
gesture demonstration to a reusable tower architecture that no longer depends
on MediaPipe at inference time.

The immediate goal is deliberately narrower than a VLM replacement:

> Prove that a small CPU landmarker tower can produce sufficiently stable,
> directionally informative hand geometry for independently replaceable gesture,
> trajectory, and Monty-style evidence towers.

If this boundary works, the same architecture can later support VLM-like feature
extraction: small sensory towers emit structured features and uncertainty, while
a recurrent/fusion layer decides whether templates, a reasoning model, or a full
vision model are required.

## Current system: what is actually running

There are three materially different paths in the repository.

### MediaPipe reference path

```text
RGB frame
  -> MediaPipe hand detector + 21-point landmarker
  -> normalized 63D landmark geometry
  -> SVM or Monty evidence classifier
  -> gesture intent
```

MediaPipe currently performs both localization and landmark estimation. SVM and
Monty consume its geometry; they do not detect hands in pixels.

### Tiny gesture v3 path

`train/runs/tiny_gesture` currently contains:

```text
trunk.onnx  128x128 RGB -> 42 landmark coordinates + presence
crop.onnx   64x64 crop  -> six gesture tower probabilities
seq.onnx    8x42 track  -> six temporal tower probabilities
```

The artifact is approximately:

- `trunk.onnx`: 3.4 MB;
- `crop.onnx`: 95 KB;
- `seq.onnx`: 172 KB;
- roughly 959k parameters total;
- crop validation accuracy recorded as 0.645.

This is already the correct *shape* of a MediaPipe-free tower stack. Its weakness
is landmark quality and generalization, not architectural intent.

### HGDet hybrid path

The GUI entry `CPU · HGDet hybrid (detector A/B)` currently runs:

```text
hand_detector.onnx 320x240 -> bounding box
  -> resize box to 64x64
  -> tiny_gesture/crop.onnx -> gesture probabilities
```

`hand_detector.onnx` is a 1.2 MB hand **box detector**, not a 21-point
landmarker. The current `HgDetRuntime` returns a box and never emits landmark
coordinates or a trajectory. Calling it a MediaPipe replacement conflates two
different jobs.

This explains the observed “worst of both” outcome:

1. missed or loose HGDet boxes destroy the crop classifier's canonical view;
2. the crop tower was trained largely from a different crop distribution;
3. no landmark geometry reaches the sequence or Monty towers;
4. detector cost is paid on every frame rather than amortized by tracking;
5. the detector's known domain failure on blurred local captures remains
   upstream of every other decision.

## The proposition to prove

The proof is not “a tiny CNN sometimes recognizes my gesture.” It is:

> Given identical held-out RGB sequences, a lightweight landmarker produces a
> stable landmark stream whose frozen downstream towers remain acceptably close
> to the MediaPipe-reference outcomes, at materially lower sustained CPU cost.

The proof has four independent claims:

1. **Presence:** it finds hands and rejects non-hands.
2. **Localization:** it places landmarks close enough to the reference geometry.
3. **Temporal geometry:** displacement and orientation remain directionally
   correct across frames.
4. **Downstream equivalence:** frozen gesture and Monty towers retain useful
   intent accuracy without being retrained to hide landmark errors.

All four must pass. A good gesture score alone can conceal a bad landmarker; a
good average landmark distance can conceal left/right inversions.

## Proposed permanent tower boundary

```text
                      +---------------- presence tower
RGB frame -> ROI -> shared landmarker trunk
                      +---------------- 21-point geometry tower
                      +---------------- visibility/quality tower
                                           |
                    +----------------------+----------------------+
                    |                      |                      |
              pose/shape tower      trajectory tower       Monty evidence
                    |                      |                      |
                    +----------------------+----------------------+
                                           |
                                 recurrent/fusion policy
                                           |
                            intent + features + uncertainty
```

The landmarker contract must be task-neutral. It cannot output `TROT` or `HOME`.
It outputs observable hand geometry and quality; downstream towers attach robot
semantics.

### Versioned runtime contract

Recommended `landmarker-v1` ONNX contract:

```text
input:
  image       float32 or uint8 [N,3,96,96] ROI

outputs:
  landmarks   float32 [N,21,2] normalized ROI coordinates
  visibility  float32 [N,21]
  presence    float32 [N,1]
  quality     float32 [N,1]
```

Optional later outputs:

```text
  handedness  float32 [N,2]
  relative_z  float32 [N,21]
  roi_delta   float32 [N,4]
```

Do not require `relative_z` for the first proof. Stable 2D geometry and motion
are sufficient for shape, left/right, crop generation, and a first Monty 2D
adapter. Adding weak pseudo-depth early increases teacher noise and CPU output
without proving the critical boundary.

Every artifact must include metadata:

```json
{
  "contract": "openpave.landmarker.v1",
  "input_px": 96,
  "landmark_order": "mediapipe-21-compatible",
  "coordinate_space": "roi-normalized",
  "precision": "fp32",
  "teacher": "mediapipe+verified-dataset-landmarks",
  "training_sources": [],
  "metrics": {}
}
```

Keeping MediaPipe's 21-point ordering is a compatibility choice, not a runtime
dependency. It lets all existing feature, SVM, sequence, visualization, and
Monty code consume either provider behind one adapter.

## Do not make HGDet the landmarker

HGDet can be useful as one ROI proposal source, but it should not define the
landmark architecture.

Test three proposal modes independently:

1. **Oracle ROI:** crop from reference landmarks. Measures pure landmarker
   quality.
2. **Tiny presence/ROI tower:** crop from the student itself. Measures the
   proposed deployable stack.
3. **HGDet ROI:** crop from `hand_detector.onnx`. Measures whether HGDet adds
   value for a particular domain.

If oracle ROI fails, improve the landmarker. If oracle passes but HGDet fails,
the detector is the bottleneck. If student ROI passes, HGDet is unnecessary.

This separation prevents detector errors, crop errors, landmark errors, and
gesture errors from being blended into one misleading GUI outcome.

## Recommended landmarker architecture

The current v3 trunk uses four stride-2 convolution blocks, global pooling, and
a linear layer that directly regresses 42 coordinates. This is compact, but
global coordinate regression often becomes noisy under translation, partial
occlusion, and unfamiliar backgrounds.

### Proof model: heatmap-lite landmarker

Use a 96x96 or 112x112 ROI with a depthwise-separable encoder and a small spatial
decoder:

```text
96x96 RGB
 -> stem, stride 2
 -> depthwise blocks: 48x48 -> 24x24 -> 12x12
 -> lightweight upsample/fusion to 24x24
 -> 21 heatmaps at 24x24
 -> soft-argmax -> 21x2 coordinates
 -> pooled presence/quality heads
```

Initial budget:

- 150k-400k parameters;
- under 1.5 MB FP32 ONNX;
- under 1 MB after INT8 if accuracy survives;
- 96x96 input;
- one hand per ROI;
- no decoder more expensive than the encoder.

Heatmaps preserve spatial evidence and allow visibility supervision. Export
soft-argmax inside ONNX only after verifying it maps efficiently to the target
runtime; otherwise export heatmaps and decode them in small C++/NumPy code.

### Alternative proof model

If heatmap export is awkward, retain direct coordinate regression but add:

- spatial feature maps rather than immediate global pooling;
- coordinate convolution channels;
- a bounding-box/ROI residual head;
- landmark visibility;
- bone and equivariance losses;
- detector-jittered ROI training.

Do not enlarge the network until the loss and crop-domain experiments establish
that capacity, rather than supervision, is the limiting factor.

## Training data strategy

### Teacher use is allowed offline

The goal is removing MediaPipe from deployment, not pretending it never existed.
MediaPipe can supply pseudo-landmarks during training, provided its errors are
filtered and the final evaluation contains independently reviewed examples.

Use the existing prepared v2 shards:

- `crude`: target camera and operator behavior;
- `hagrid_shapes`: static shape and subject/background diversity;
- `yolo26`: independently labeled gesture images;
- `ipn`: continuous motion, clutter, illumination, and temporal continuity;
- `custom`: new target-hardware recordings;
- Jester when licensed locally: swipes and temporal negatives.

HaGRID's own landmark annotations should be preferred where available. Use
MediaPipe pseudo-labels when verified labels are absent.

### Required sampling balance

Landmark training is not gesture-class training. Sample to cover geometry and
failure modes:

| Bucket | Suggested share |
|---|---:|
| clean visible hand | 30% |
| small/distant hand | 15% |
| edge-clipped or partial hand | 10% |
| motion blur | 10% |
| low/high illumination | 10% |
| clutter/skin-like background | 10% |
| two hands; choose/track one consistently | 5% |
| hard no-hand negatives | 10% |

Do not let thousands of near-identical IPN frames dominate. Sample by subject,
recording, hand scale, illumination, and pose cluster.

### Teacher-label filtering

Keep a pseudo-landmark example only when:

- teacher hand presence exceeds a strict threshold;
- all required landmarks are finite and inside a reasonable padded frame;
- palm and finger bone lengths are anatomically plausible;
- adjacent-frame displacement is plausible or corroborated by image motion;
- optional horizontal-flip inference agrees after inverse transformation.

Rejected teacher frames remain useful as presence or hard-negative examples;
they simply do not receive coordinate loss.

### Human-reviewed proof set

Create a small frozen `landmarker_proof_v1` set before tuning:

- 300-500 hand-positive frames;
- 200 no-hand frames;
- at least 10 people;
- target camera plus external sources;
- stationary, directional movement, blur, occlusion, and difficult light;
- 50-100 frames with manually corrected landmark coordinates;
- complete sequences retained for temporal scoring.

Never train on this set. Version its manifest and hashes. The 22-probe gesture
matrix remains a smoke test, not the landmarker proof set.

## Loss design to prevent noisy or collapsed landmarks

Use a multi-objective loss whose terms can be ablated:

```text
L = 1.0 * L_presence
  + 4.0 * L_coordinate_or_heatmap
  + 0.5 * L_visibility
  + 0.5 * L_bone
  + 0.5 * L_equivariance
  + 0.25 * L_temporal
  + 0.25 * L_quality
```

### Presence loss

Use weighted BCE or focal loss with real hard negatives. Do not label every
teacher miss as `no hand`; that teaches the student to copy the teacher's blind
spots.

### Landmark loss

Use Smooth L1, Wing loss, or heatmap KL/MSE only on valid visible landmarks.
Normalize by ROI size, not full-frame size.

### Bone/topology loss

Penalize implausible relative bone lengths and broken finger chains. Normalize
against palm scale so different hand sizes remain valid.

### Equivariance loss

Apply known crop transforms and require predictions to transform identically:

- horizontal flip with correct landmark remapping;
- translation;
- scale;
- small rotation;
- moderate perspective/affine distortion.

This directly trains the property that current global regression lacks.

### Temporal loss

For continuous recordings, penalize frame-to-frame landmark acceleration unless
supported by image motion. Do not over-smooth genuine swipes or throws. The
objective is stable geometry, not frozen geometry.

### Quality head

Train quality to predict downstream landmark error or teacher/student
disagreement. This gives the fusion architecture a principled rejection signal.
A low-quality hand must become `unknown`, not a confident wrong intent.

## Train on the crop distribution used at inference

This is the most important correction to the HGDet experiment.

For every training hand, create multiple ROIs:

1. ideal box from teacher landmarks;
2. translated boxes;
3. boxes that are too tight or too loose;
4. aspect-ratio errors;
5. boxes sampled from the actual HGDet error distribution;
6. boxes sampled from the student's previous checkpoint.

Transform landmark targets into each ROI's coordinate space. This makes the
landmarker tolerant of realistic proposal error and prevents a clean-crop model
from collapsing behind an imperfect detector.

The crop classifier must similarly train on the **student landmarker's** boxes,
as v3 already attempts. Keep an oracle-crop score and a student-crop score so
crop mismatch remains visible.

## Training curriculum

### Phase 0: freeze contracts and tests

Before retraining:

1. freeze `landmarker_proof_v1`;
2. store MediaPipe outputs on it;
3. store HGDet boxes and current v3 landmarks;
4. freeze gesture/Monty/sequence downstream artifacts;
5. define the metrics and pass thresholds below.

### Phase 1: presence plus oracle-ROI landmarks

Train only on oracle/padded ROIs. This proves whether the compact landmark model
has sufficient capacity independent of detection.

Stop if oracle-ROI accuracy is inadequate. Do not compensate by retraining the
gesture tower yet.

### Phase 2: proposal-jitter robustness

Mix ideal, synthetic-jitter, HGDet, and previous-student crops. Re-evaluate the
same frozen downstream towers.

### Phase 3: student ROI and tracking

Use the landmarker/presence head to initialize and update ROI. Run the expensive
full-frame proposal path only:

- at startup;
- after tracking quality falls;
- after several no-hand frames;
- after a scene-change gate fires.

Between proposals, predict ROI from prior landmarks and velocity. This is the
important speed property HGDet hybrid currently misses: detection should not run
on every frame.

### Phase 4: downstream tower retraining

Only after frozen-tower equivalence is measured may gesture/sequence/Monty
towers be retrained on student landmarks. Report both scores:

- frozen downstream score: landmarker replacement quality;
- adapted downstream score: achievable complete-system quality.

Otherwise a retrained classifier can hide a weak and non-general landmarker.

### Phase 5: quantization

Establish FP32 accuracy first. Then compare:

- FP16 where the runtime benefits;
- static INT8 with representative calibration frames;
- mixed precision for heatmap/soft-argmax if INT8 coordinate error is excessive.

Quantization must pass downstream equivalence, not merely output cosine
similarity.

## Proof matrix

Every run must publish this matrix from the same frozen data:

| Provider | ROI mode | Presence F1 | NME | PCK@0.05 | direction agreement | frozen gesture accuracy | p50 ms | p95 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MediaPipe | internal | | | | | | | |
| student | oracle | | | | | | | |
| student | student ROI | | | | | | | |
| student | HGDet ROI | | | | | | | |
| HGDet hybrid | box only | | n/a | n/a | n/a | | | |

### Landmark metrics

Normalized mean error:

```text
NME = mean(||pred_i - target_i|| / palm_scale)
```

Also report:

- PCK at 0.05 and 0.10 palm-normalized distance;
- wrist error;
- fingertip error;
- per-landmark visibility recall;
- left/right sign agreement for wrist and index-tip displacement;
- landmark loss by hand scale and illumination bucket;
- jitter: stationary landmark velocity in pixels/second;
- track-loss events per minute;
- recovery frames after occlusion.

### Downstream metrics

Use the permanent equivalence probes, expanded beyond 22 cases:

- intent recall per HOME/LEFT/RIGHT/STOP/TROT;
- NOOP false-positive rate;
- hand presence false-positive/negative rate;
- feature equality with the reference;
- Monty object-evidence agreement;
- sequence direction agreement;
- unknown/rejection rate.

Never summarize only with total accuracy. The existing matrix already shows why:
one tower can appear acceptable while completely missing TROT or HOME.

## Initial pass/fail gates

The student may replace MediaPipe in an experimental GUI path when all of these
hold on the frozen proof set:

### Geometry

- positive-hand presence recall >= 95%;
- no-hand specificity >= 98%;
- PCK@0.10 >= 95%;
- PCK@0.05 >= 85%;
- fingertip direction-sign agreement >= 95%;
- no systematic left/right inversion;
- no single-class or center-point landmark collapse.

### Downstream equivalence

- frozen gesture macro-F1 within five points of MediaPipe;
- LEFT/RIGHT recall within five points of MediaPipe;
- Monty evidence top-1 agreement >= 90% on clear examples;
- NOOP false-positive rate no worse than MediaPipe plus two points;
- rejection is preferred to confident disagreement.

### Performance

Measure the complete path, not one ONNX call:

- proposal + landmark + crop + gesture;
- p50 and p95 latency;
- sustained 15-minute CPU utilization;
- memory;
- temperature/throttling;
- joules/minute where hardware measurement is available.

Suggested targets:

| Target | Complete hand pipeline target |
|---|---:|
| M4 development reference | <2 ms average after tracking is warm |
| Raspberry Pi 5 | <15 ms average, <30 ms p95 |
| Orion O6 | <10 ms average or comfortably inside the robot control budget |

The target is a sustained average. A 10 ms proposal once every 15 frames plus a
2 ms tracked landmarker may be better than a 4 ms detector on every frame.

## Benchmark the latency decomposition

Publish this timing record per frame:

```json
{
  "scene_gate_us": 80,
  "proposal_ms": 0.0,
  "landmarker_ms": 1.8,
  "crop_tower_ms": 0.2,
  "sequence_us": 30,
  "fusion_us": 20,
  "total_ms": 2.13,
  "proposal_ran": false,
  "quality": 0.94
}
```

This prevents the current ambiguity where `hand_detector.onnx` appears fast in
isolation but the complete hybrid feels slow and recognizes few hands.

## Artifact layout and independent retraining

```text
train/runs/
  landmarker_tower/
    model.onnx
    model.int8.onnx
    meta.json
    proof-report.json
  tiny_gesture/
    crop.onnx
    seq.onnx
    meta.json
  monty_gestures/
    objects.npz
  fusion/
    policy.json
```

Rules:

- retraining the landmarker does not retrain gesture or Monty by default;
- retraining a gesture tower does not change the landmarker;
- changing ROI proposal policy requires no model retraining;
- changing HGDet does not overwrite the student landmarker;
- ensemble metadata pins artifact hashes and contract versions;
- every replacement is evaluated against frozen upstream/downstream artifacts.

This is the flexible Monty architecture proof: sensory providers and learning
modules can change independently while their contracts remain stable.

## PyQt integration strategy

### Stage 1: offline only

Do not select the new landmarker as authoritative until the proof report passes.

### Stage 2: GUI shadow mode

Run MediaPipe and student side by side at a low diagnostic cadence. Display:

```text
MP: hand yes | LEFT | 5.2 ms
LM: hand yes | LEFT | 1.9 ms | PCK proxy 0.93 | quality 0.91
```

Draw skeletons in different styles and log disagreement frames to a harvest
directory. Never post duplicate robot intents from shadow output.

### Stage 3: student authoritative, MediaPipe audit

The student drives speech/intent while MediaPipe samples perhaps one frame every
one to five seconds for audit only. Persist disagreement examples.

### Stage 4: MediaPipe absent

Remove MediaPipe from the runtime environment and run the complete regression,
thermal, and restart tests. A hidden import or fallback means the replacement
has not been proven.

## How this points toward VLM-like feature extraction

The landmarker experiment establishes the pattern required for broader visual
towers:

```text
pixels
 -> cheap proposal/tracking
 -> task-neutral structured sensory feature
 -> specialist evidence towers
 -> temporal/recurrent fusion
 -> language template or conditional VLM escalation
```

After the hand proof, analogous towers can expose:

- person presence and body landmarks;
- motion vector and track state;
- torso/lower-body appearance histograms;
- object proposals and compact embeddings;
- illumination/camera-motion state;
- novelty and quality.

Each tower must emit:

```text
feature values + coordinate/reference frame + confidence + quality + age
```

The fusion layer then reasons over stable structured observations rather than
rerunning a general VLM over every nearly identical RGB frame.

The hand landmarker is therefore not merely a gesture optimization. It is the
smallest serious demonstration that OpenPAVE can replace one monolithic visual
dependency with a measurable, independently trainable sensory tower.

## Recommended next implementation slice

Implementation status (2026-07-13): the HGDet GUI entry has been removed and
replaced by `CPU · Landmarker Tower`. The standalone trainer/exporter is
`train/landmark-tower.sh`; its artifact is `train/runs/landmark_tower/model.onnx`.
The GUI also exposes `CPU · Landmark + Monty (3D evidence)`, a separate
MediaPipe-free experimental controller:

```text
RGB -> landmark tower (21 x/y points)
    -> class-neutral anatomical depth lift
    -> frozen Monty 3D evidence exemplars
    -> STOP | TROT | HOME | LEFT | RIGHT
    -> existing ingress safety gates
```

It emits the same skeleton and evidence-box video overlay and the same
`observation.json` speech contract as the other gesture controllers. The
original `CPU · Monty (3D evidence)` controller remains unchanged as the
MediaPipe reference. The remaining items below are the empirical acceptance
sequence for the student path.

Do this before acquiring another large dataset or expanding the VLM proposal:

1. rename the GUI concept from `HGDet landmarker` to `HGDet box hybrid` wherever
   ambiguity remains;
2. create `train/runs/landmarker_tower/` as a first-class artifact independent
   of `tiny_gesture`;
3. extract a frozen 500-700-frame `landmarker_proof_v1` dataset;
4. add oracle-ROI evaluation for the current `trunk.onnx`;
5. report PCK/NME, direction agreement, presence F1, and downstream frozen-tower
   equivalence;
6. train a heatmap-lite or spatial coordinate-regression student with detector
   jitter and the losses above;
7. compare student ROI, oracle ROI, and HGDet ROI;
8. add GUI shadow mode only after offline geometry is understood;
9. keep the crop and sequence towers frozen until replacement quality is known;
10. deploy to Raspberry Pi/Orion only after the complete sustained-latency test.

The decisive milestone is not a visually pleasing skeleton. It is a report
showing that a frozen Monty/gesture stack receives interchangeable MediaPipe and
student landmark streams with bounded accuracy loss and a measured CPU/energy
win.
