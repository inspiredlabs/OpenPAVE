# Replacing MediaPipe hand pose with four simple classifiers derived from `tbp.monty`

Status: proposal, revised 2026-07-16. Supersedes the earlier "make live Monty
match training" framing. The design decision this revision locks in:

> **Monty is the teacher, not the runtime.** Temporal Monty evidence/CMP is
> used ONLY during training. Live inference is the frozen 71k acquirer plus
> four simple, independently calibrated gesture classifiers whose supervision,
> inputs, and calibration are all derived from the HanCo × Monty training
> stack. Nothing named `EvidenceLM`, `Episode`, Kabsch, or a depth prior runs
> in `./mlx-runtime.sh`.

## 1. Why the previous framing was wrong

The measured blocker (unchanged, see
`docs/hand-pose-sensorimotor-lab.md` §5 "Known inference-side gaps"):

- `LandmarkerMontyWorker` (`pave_ui/perception.py`) lifts predicted 2D
  landmarks with a class-blind median-depth prior, then runs one-shot batched
  Kabsch against every frozen exemplar.
- Training (`monty/tutorials/hand_pose_lab.py`, driven from
  `polyglot.py`) sees HanCo metric XYZ, RGB patch appearance, motor deltas,
  temporal continuity, and multi-camera viewpoint change.
- The archived HAND POSE LAB run only swaps `objects.npz`; the observation
  front end stays incompatible with what those exemplars were built from.

The earlier proposal tried to close this gap by pushing training richness
INTO the runtime (sequential Monty observations, live RGB patches, real
depth). This revision closes it in the opposite direction — and the repo's
own measurements say that direction is right:

| Measured fact | Source | Consequence |
|---|---|---|
| 71k incumbent: 36.27% live acquisition; every retrained acquisition candidate ≤10.35% | `training-with-monty.md`, alternation reports | The acquirer stays FROZEN. Front-end replacement is a separate, already-gated programme; this proposal does not touch it. |
| One binary geometry gate on 71k landmarks: 94.55% presence F1, 83.65% target acquisition, 2.17% no-hand false acquisition, 0.0 s first lock | `train/runs/hanco_target_poc/` | Existence proof: a simple per-gesture classifier over the frozen acquirer's output is live-viable. |
| One 5-way softmax geometry head over the same landmarks: 52.1% correct, 46.9% wrong-gesture | `train/runs/hanco_crop_gesture/` | A single multiclass head is NOT viable — motivates four one-vs-rest gates with independent calibration and a strict arbitration rule. |
| Monty pose-space propagation precision 0.897; multi-view consistency term reduced wrong-gesture 0.508→0.449 | `train/runs/hanco_crop_curriculum/` | Monty's graph memory measurably improves supervision quality. Its value is training-time, exactly where this proposal spends it. |
| Runtime Kabsch abstains on everything unless exemplar normalisation matches `_normalise` exactly; z is a fabricated prior; right-hand exemplars cannot mirror-match | `hand-pose-sensorimotor-lab.md` §5 | The Kabsch/depth-prior stage is the fragile component. Remove it from the runtime rather than repairing it. |

Better gesture exemplars cannot compensate for an incompatible observation
contract — that diagnosis stands. The fix is that the runtime should stop
consuming exemplars at all.

## 2. Explicit target architecture

### Training stack (osx-64 `tbp.monty` env + arm64 `.venv`, bridged by files)

```text
HanCo (all existing, cache-first):
  rgb/<seq>/cam0..7      mask_hand  mask_fg
  xyz/<seq>/<frame>.json (metric 21×3, world)
  calib (per-frame K, M) shape (MANO)   gesture_manifest.json (reviewed)
        │
        ▼
HAND POSE SENSORIMOTOR LAB (monty/tutorials/hand_pose_lab.py, polyglot.py tab)
  virtual patch sensor saccades over 21 joints
  features-at-poses: joint_id + RGB patch appearance @ canonical 3D pose
  motor deltas; curriculum = joint walk → temporal continuity → viewpoint change
  two EvidenceGraphLM columns (distal/proximal), CMP voting, TEMPORAL evidence
        │  model.pt  +  objects.npz  +  per-frame evidence scores   (file bridge)
        ▼
DISTILLATION (openpave/.venv, new: train/monty_distill_gates.py)
  for every labelled HanCo (seq, frame, cam):
    input  = EXACTLY what live computes (see §3: parity by construction)
    target = reviewed label ∪ Monty-propagated pseudo-label (precision 0.897)
    weight/soft target = Monty evidence margin for that constellation
    regularisers = multi-view consistency KL (weight 0.2, measured win),
                   mask_hand-inpainted same-frame negatives,
                   crude no_hand frames
  output = 4 one-vs-rest gates + per-gate thresholds calibrated on live replay
```

### Inference stack (`./mlx-runtime.sh`, arm64, CPU)

```text
webcam BGR frame
  → frozen 71k acquirer (train/landmark_tower.LandmarkerRuntime, UNCHANGED:
    detector global search → crop → 21 2D landmarks, presence/quality gates)
  → canonical 2D feature (hanco_target_poc.feature: wrist-origin,
    palm-axis rotation, unit scale — 42 coords + 20 bone lengths)
  → FOUR independent binary gates, one artifact each:
      palm.npz   fist.npz   like.npz   point.npz
    each: linear/logistic (or ≤26k-param MLP) + its OWN calibrated threshold
  → arbitration: exactly one gate above threshold → that gesture;
    zero or ≥2 gates above threshold → abstain (no intent, display only)
  → point only: geometry rule decides direction
    (subject-centric, mediapipe_svm.point_direction convention; vertical → no-op)
  → intent map (unchanged): palm→STOP  fist→HOME  like→TROT  point→LEFT/RIGHT
```

What is explicitly ABSENT from the live path: `EvidenceLM`, `objects.npz`,
`Episode`/`Observation`, Kabsch (full or partial), the median-depth prior,
any z coordinate, RGB patch inspection, CMP voting, and temporal Monty
evidence. The only temporal state at runtime is what the frozen acquirer
already keeps (ROI tracking) plus the existing debounce in the viewer.

Four artifacts, not one: each gate is trained one-vs-rest and thresholded
independently (the `hanco_target_poc` recipe, which passed, rather than the
softmax recipe, which failed). Adding a fifth gesture later means training
one new gate and re-running arbitration calibration — no retraining of the
other four, which preserves the few-shot-extension property Monty was chosen
for.

## 3. Inference parity by construction (the HanCo corpus rule)

Training sees HanCo metric XYZ and RGB patches; live sees 71k-predicted 2D
landmarks. Parity is achieved at DATA GENERATION time, not by upgrading the
runtime:

1. **Geometry channel.** Project HanCo `xyz` through the per-frame `K`/`M`
   into each valid camera view (`hanco_target_poc.project` — per-frame `K`,
   never cached). The result is exactly the 2D constellation family live
   inference produces, with zero fabricated depth.
2. **Deployment noise channel.** Additionally run the FROZEN 71k acquirer
   over the actual HanCo RGB (the `hanco_crop_gesture` path already does
   this). Train on a declared mixture of clean projections and real acquirer
   outputs; out-of-crop or rejected joints get visibility 0. This is the
   lesson the alternation rounds paid for: distribution match beats
   component quality.
3. **Negative channel.** `mask_hand`-inpainted same-frame negatives,
   `mask_fg` person-rejection composites, MANO-pose-nearest hard negatives
   (the `hanco_target_poc` mechanism), and explicit `crude` `no_hand`
   frames.
4. **Split discipline.** Sequence-level splits, all eight cameras of a
   timestep together, temporal splits within target sequences —
   calibration and evaluation frames never shared (existing rules, kept).

Monty's exclusive training-time contributions ride on top:

- **Labels:** reviewed manifest (19,705 observations) expanded by
  pose-space propagation from the 1,502-graph checkpoint (measured
  precision 0.897; the 557 accepted frames are additional supervision, not
  a corpus replacement).
- **Soft targets:** per-frame evidence margins from the graph memory weight
  ambiguous frames down instead of poisoning the gates.
- **Consistency:** the symmetric multi-view KL term between synchronized
  cameras (weight 0.2) — the one curriculum ingredient with a measured
  wrong-gesture reduction.
- **Temporal evidence, training-only:** curriculum stage 2 (frame striding)
  teaches per-object articulation drift; its effect reaches the runtime only
  through the distilled weights.

## 4. Code changes (small, contained)

1. `train/monty_distill_gates.py` (new): builds the parity corpus (§3),
   consumes the lab checkpoint/evidence via the existing file bridge, trains
   and calibrates the four gates, writes
   `train/runs/monty_gates/{palm,fist,like,point}.npz` + `meta.json`
   (per-gate thresholds, confusion, abstention, provenance, SHA-256 of the
   HanCo shard and the Monty checkpoint).
2. `pave_ui/perception.py`: new `MontyGateWorker` cloned from the
   `HanCoTargetWorker` pattern (frozen `LandmarkerRuntime` + npz gates), with
   four-gate arbitration and the point-direction rule. It replaces
   `LandmarkerMontyWorker` as the promoted variant once gates pass;
   `LandmarkerMontyWorker` stays available as the historical diagnostic
   dropdown entry. `discover_landmarker_monty_models()` gains the
   `monty-gates · 4×1-vs-rest · distilled` entry.
3. Delete from the promoted live path (not from the repo): the
   `_depth_prior` computation, `Episode` construction, and
   `infer`/`infer_partial` calls. `objects.npz` becomes a distillation input
   only.
4. `./train/monty-landmarks.sh gates` wrapper + entries in the runbook.

## 5. Promotion gates (unchanged in substance, wrong-action first)

New variant ships display-only (`commands off`) until ALL of:

- live acquisition ≥60% on the crude replay set, all four videos locking;
- wrong-intent rate ≤3% (reported FIRST — a wrong command moves the robot;
  an abstention only delays);
- `no_hand` false acceptance <8% (sequential, through the real PyQt worker
  via `replay_crude_videos.py`);
- end-to-end p95 ≤50 ms on the deployment CPU;
- per-gesture confusion matrix and abstention rate reported per gate, never
  as one aggregate;
- `./train/gesture-lab.sh eval-v3` (yolo26 referee) and
  `.venv/bin/python train/equivalence_probes.py run` both pass;
- three-seed reproduction with pre-declared selection on calibration
  macro-F1 (the MPS ±4 pp variance caveat is measured — single-run wins
  don't promote);
- command isolation: empty command log in every training/benchmark mode.

An OpenPAVE-only ablation (gates trained without Monty-derived labels, soft
targets, and consistency) is mandatory before attributing any gain to
`tbp.monty` itself.

## 6. What this deliberately does not attempt

- Retraining or replacing the 71k acquirer (measured dead end for now; the
  proposer/landmarker programme continues separately under its own gates).
- Runtime 3D lifting, depth estimation, or surface normals — no consumer
  remains once Kabsch leaves the live path.
- Runtime Monty evidence accumulation or CMP voting — by design, per this
  revision's opening decision, not as a temporary shortcut. If a future
  candidate wants temporal evidence back at runtime, it re-enters through
  these same promotion gates as a new variant.
