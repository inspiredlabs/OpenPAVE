# Hand Pose Sensorimotor Lab

Status: specification + first implementation (2026-07-16). The lab is the
`HAND POSE SENSORIMOTOR LAB` tab of `monty/tutorials/polyglot.py`; its
headless core is `monty/tutorials/hand_pose_lab.py`. The empirical test path
is `openpave/mlx-runtime.sh` via the exported `objects.npz` bridge.

This document refines the original plan against tbp.monty's
[application criteria](https://docs.thousandbrains.org/docs/application-criteria)
and binds every abstract requirement to a concrete, already-existing asset.

## Why this is a sensorimotor problem, not a dataset problem

The criteria are explicit: "Monty is designed for sensorimotor applications.
It is not designed to learn from static datasets," and any application needs
moving sensors — which the criteria say may be physical, simulated, **or
"cropped sensor patches in images."** That last allowance is the entire legal
basis of this lab: HanCo is pre-recorded, but the *sensor* we model is not
the camera — it is a small patch that saccades across the hand, and we know
its pose at every step because HanCo gives metric 3D for all 21 joints.

The mindset shift from the draft plan survives intact: no `model.fit()`, no
backpropagation, no static batches. Learning is associative — features at
poses accumulate into reference-frame graphs.

## 1. Data: from HanCo sequences to features-at-poses

Everything below already exists on disk; the lab downloads nothing.

| Ingredient | Source (existing) |
|---|---|
| RGB views | `~/.cache/hanco/rgb/<seq>/cam0..7/` (224², all 1,502 sequences) |
| Metric 3D joints | `~/.cache/hanco/xyz/<seq>/<frame>.json` (21×3, metres, world frame) |
| Per-frame intrinsics + extrinsics | `~/.cache/hanco/HanCo_calib_meta/calib/` (`K` varies per frame — read per frame, never cache) |
| Canonical constellations | `monty/tutorials/results/hand_landmarks/ingest/sequences/<seq>.npz` (wrist/MCP-centroid frame, built by the resumable ingester) |
| Gesture labels | `openpave/train/datasets/hanco/gesture_manifest.json` (reviewed whole-sequence labels; file-bridge read) |

**The virtual sensor.** One observation = one numbered joint visit:

- **Feature (non-morphological):** the 21-way `joint_id` one-hot plus a small
  appearance vector sampled from the RGB patch centred on the joint's
  projected pixel (patch mean colour + local intensity contrast). The
  appearance term is what makes this a *vision* lab rather than a pure
  geometry replay; the criteria's "features at poses" needs both.
- **Pose (morphological):** the joint's location in the canonical hand frame
  (true 3D — this is where the earlier `z=0` ablation is finally retired)
  plus bone-derived `pose_vectors` (parent-bone tangent frame), exactly the
  contract `landmark_contract.local_pose_vectors` already implements.
- **Motor signal:** the displacement between consecutive joint visits in the
  canonical frame. The sensor "moves"; Monty is told how. Proprioception
  reports the sensor at the joint location.

**Correction to the draft:** the draft proposed "(x, y, θ) within the frame"
as the pose. That would rebuild the 2-D ablation the project already
falsified. HanCo's `xyz` gives metric 3D joint positions, so the pose is 3-D
location + 3-D orientation vectors in the hand's reference frame, and image
(x, y) is only used to *cut the patch*, never as the pose.

**Correction to the draft:** "Feature ID via a small CNN" is deferred, not
required. Monty's LMs accept continuous feature vectors with per-feature
tolerances; a trained CNN encoder can replace the hand-crafted appearance
vector later without touching the architecture (swap inside the
SensorModule).

## 2. Architecture (mirrors the verified adapter, extends to voting)

The lab reuses the pattern already proven in
`openpave/train/monty_lab/tbp_adapter/` (a real
`MontySupervisedObjectPretrainingExperiment` with a custom environment,
programmatic config, no-command motor system) and extends it to two columns:

- **Environment** `HandPatchEnvironment`: one labelled (sequence, frame,
  camera) constellation per episode; 21 joint visits per episode; implements
  `step/reset/close` and emits `Observations` + `ProprioceptiveState`.
  Deterministic replay actions are sanctioned — the docs allow returning
  current observations for empty action sequences.
- **SensorModules** (2): `distal` (fingertips + DIP joints) and `proximal`
  (wrist, MCPs, PIPs). Each converts its joint visits into CMP `Message`s
  (location, pose vectors, `joint_id`, appearance, confidence, `use_state`).
- **LearningModules** (2): `EvidenceGraphLM` instances with per-feature
  tolerances; `max_graph_size 8.0` (the four-unit grid silently rejected
  85.7% of point observations — measured, do not shrink).
- **Coupling:** `sm_to_lm_matrix = [[0], [1]]`,
  `lm_to_lm_vote_matrix = [[1], [0]]`. During supervised pretraining votes
  are inert; they exist so the same checkpoint can be evaluated later with
  CMP voting between the distal and proximal columns.
- **Motor system:** no-command replay (`commands_enabled: false` in every
  state dict). Nothing in this lab can ever emit a robot intent.

## 3. The training loop the Play button runs

One press of **▶ Play training** runs the real experiment — not an
animation of one — with the GUI as a spectator:

1. The experiment thread runs `MontySupervisedObjectPretrainingExperiment`
   over the curriculum below.
2. Every environment step publishes (frame RGB, patch box, joint id,
   canonical location, episode label) to a queue; the GUI drains it on a
   timer and draws the saccading patch, the visited-joint trail, and the
   growing canonical constellation.
3. Episode ends → supervised association: the LMs bind the accumulated
   features-at-poses to the target object (`palm`, `fist`, `like`, `point`).
4. Checkpoint is written to
   `monty/tutorials/results/hand_pose_lab/pretrained/model.pt`, and the
   openpave bridge `objects.npz` (`label → (n_exemplars, 21, 3)`) is exported
   next to it.

**Curriculum order** (each stage is one epoch group; all bounded):

1. **Joint walk, single view** — skeleton-order saccade over cam0 frames of
   each labelled sequence: builds the within-frame reference frame.
2. **Temporal continuity** — the same sequences, striding frames: the object
   is re-visited as articulation drifts, teaching pose variation per object.
3. **Viewpoint change** — the same frames through different cameras:
   canonical locations are camera-invariant, so this teaches that appearance
   varies while geometry votes stay stable (HanCo's unique multi-view gift).

## 4. Application-criteria compliance and workarounds

| Criterion / limitation | Status here | Workaround |
|---|---|---|
| "Not designed to learn from static datasets" | Compliant | Virtual patch sensor with per-step pose + motor delta ("cropped sensor patches in images" is an explicitly sanctioned movement class) |
| Moving sensors with known poses | Compliant | Saccade across 21 joints; pose from metric `xyz` in the canonical hand frame; camera hops use known extrinsics |
| Features at poses, not feature bags | Compliant | `joint_id` + patch appearance at 3-D location + bone-frame orientation |
| No per-pixel depth in HanCo RGB | Limitation | Sensor pose comes from triangulated joint `xyz`, not from a depth map; patch appearance stays 2-D. A future MANO mesh tier adds surface normals |
| No live motor control (pre-recorded data) | Limitation | Deterministic replay policy; the environment owns the trajectory and the motor system emits no commands. Documented allowance for empty action sequences |
| "Research project… API not stable, could change at any time" | Limitation | Programmatic configs pinned to the local clone (both venvs import the same source tree); the lab mirrors the already-verified adapter classes rather than inventing new entry points |
| Supervised pretraining saves only at the end | Limitation (measured: the omniglot alphabet failure) | Bounded episode counts per Play run; each run saves; the GUI excepthook prevents the qFatal abort that killed long runs; re-running extends the existing checkpoint |
| Ideal: little data, no labels, continual learning | Aligned | ~30 reviewed sequences suffice; each Play run continues from the previous checkpoint; unlabelled sequences can join later as unsupervised episodes |

## 5. Empirical test path (`openpave/mlx-runtime.sh`)

The exported `objects.npz` follows the EvidenceLM `_normalise` convention —
wrist at the origin, max-abs coordinate scaled to 1, orientation kept for
Kabsch recovery (`objects_convention: evidence-lm-normalised.v1` in
`meta.json`). This matters: canonical palm-median scaling (max-abs ≈ 2.2)
looks fine on disk but makes every runtime Kabsch residual huge, so the
evidence stage abstains on everything. Round-trip verified: each exported
exemplar re-infers its own label at evidence 1.000.

Every completed export is frozen under Monty's
`tutorials/results/hand_pose_lab/runs/<UTC timestamp>/`. `./mlx-runtime.sh`
hot-discovers every run under `CPU · Landmark + Monty (3D evidence)` as
`landmark+monty · HAND POSE LAB · <timestamp> · <FRESH|EXTENDED> · <episodes>ep · …`.
Newly completed runs appear without restarting OpenPAVE and the newest new run
is selected automatically. Deleting an archived run in Monty's learned-memory
tree removes it from this dropdown on the next two-second refresh. The
`?objects=` binding in `pave_ui/perception.py` keeps every selection tied to
its immutable `objects.npz`; override the runs location with
`PAVE_HAND_POSE_LAB_RUNS_DIR`. The frozen 71k acquirer, gates, and command
isolation are unchanged; only the object memory differs. Definition of done
stays the project standard: wrong-action rate first, then the
equivalence-probe matrix.

Known inference-side gaps the runtime does NOT close (measured in
`pave_ui/perception.py` + `train/monty_lab/evidence_lm.py`): observation z is
a fixed anatomical depth prior (the median exemplar profile — deliberately
class-blind), so articulation-dependent depth is unmodelled; no patch pixels
are inspected at inference (geometry-only evidence); recognition is one
batched Kabsch per frame, not step-accumulated hypothesis narrowing or CMP
voting; sigma/evidence_floor (0.16/0.50) were calibrated for the incumbent
exemplars; and right-hand HanCo exemplars cannot mirror-match a left hand
(Kabsch enforces det +1). These are the frontier items, in that order.

## 6. Summary checklist (refined from the draft)

| Component | Requirement for HanCo → Monty | Delivered by |
|---|---|---|
| Input | Patch appearance + `joint_id` at canonical 3-D pose with motor deltas | `HandPatchEnvironment` + two SensorModules |
| Model | Two `EvidenceGraphLM` columns under `MontyForEvidenceGraphMatching` | programmatic config, mirrored from the verified adapter |
| Learning | Associative reference-frame building, supervised object binding, no backprop | `MontySupervisedObjectPretrainingExperiment` |
| Output | Object ID (gesture) + graph memory; `model.pt` + `objects.npz` bridge | lab save + export step |
| Test | Live webcam through the unchanged evidence runtime | `./mlx-runtime.sh` |
