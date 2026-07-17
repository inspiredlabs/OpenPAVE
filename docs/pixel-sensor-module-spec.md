# Patch-Sensing Pixel Sensor Module — engineering specification

> Rev 2 (2026-07-15): incorporates three review corrections — shards are 2D
> (no z; 1.7% out-of-range coords), `predicted_uv` needs an explicit
> similarity-transform projector (Kabsch rotation alone is insufficient), and
> palm-anchor cold start is unsolved (v3-trunk-initialiser spike recommended
> first). Sections 2.2, 5 and 6.3 carry the details.

Audience: a data scientist / developer picking this up cold. Everything cited
by path exists in this repo; every number was measured on the M4 MacBook that
will do the training.

## 1. Objective and current state

Goal: remove MediaPipe from the runtime hand pipeline by replacing
`pixels → 21 landmarks` with a Monty-native sensorimotor loop:

```
frame ─→ lightweight palm/ROI detector          (kept, ~1-3ms)
      ─→ canonical hand crop
      ─→ PATCH SENSOR MODULE          ←── the missing piece this doc specifies
            samples a small pixel patch at Monty's PREDICTED next joint,
            confirms/rejects, emits (feature, 3D location) percepts
      ─→ Monty learning module (hypotheses over gesture objects + pose)
      ─→ completed constellation → next frame's ROI prediction
```

Both ends of this loop are already proven:

- **Geometry side (done)**: the real `tbp.monty` EvidenceGraphLM learns and
  recognises gesture constellations at 75% held-out from 16 episodes/0.4s —
  `train/monty_lab/tbp_adapter/run_gestures.py`. The clean-room equivalent
  (`train/monty_lab/evidence_lm.py`) ships in the GUI at ~0.5ms.
- **Teacher side (done)**: MediaPipe landmarker produces supervision
  (landmarks stored in every training shard; ~13.5k supervised frames on
  disk). It stays the reference until this module passes camera validation —
  see `train/hand-landmark/README.md`, which is the architectural charter
  this spec implements.

What five weeks of measurements established: gesture *classification* is
solved in landmark space; the open frontier is exactly and only
**pixels → localised geometry**. Every pixel-global student plateaued
(~60% foreign accuracy); the two systems that work (MediaPipe, YOLO)
both *localise before classifying*. The patch SM is that principle taken to
its per-joint limit.

## 2. The interface ("API")

### 2.1 What the SM must emit — the CMP Message (validated contract)

One percept per sensed joint. These exact fields drive the real framework
today (see `run_gestures.py::episode_to_percepts`):

**Stage-1 dimensionality (review-corrected):** the patch model observes
pixels, so it can only *confirm 2D*. The SM emits confirmed crop-frame (u,v);
a separate **coordinate adapter** composes `Message.location` by combining
that 2D evidence with the current depth prior (teacher-z statistics from the
geometry tower / episodes.npz). The z component must carry a provenance flag
(`"z_source": "prior"`) — it is NEVER to be presented as observed pixel
evidence. Genuine metric 3D does not exist anywhere in this stack.

```python
Message(
    location=np.ndarray(3,),                # [x, y from patches; z from PRIOR]
    morphological_features={
        "pose_vectors": np.ndarray(3,3),    # surface/point frame; constant OK at stage 1
        "pose_fully_defined": bool,
        "on_object": 1,                     # 0 => percept off-object (rejection)
    },
    non_morphological_features={
        "joint_id": [i / 20.0],             # which landmark (float-encoded)
        # stage 2+: patch appearance features go here (e.g. embedding, contrast)
    },
    confidence=float,                       # patch-match confidence, 0..1
    use_state=bool,                         # False => Monty ignores this percept
    sender_id="joint_sensor",
    sender_type="SM",
)
```

### 2.2 The SM class to implement

```python
class PatchSensorModule:
    def reset(self, frame_bgr, roi) -> None
        """New frame/crop. roi from palm detector or previous constellation."""

    def sense(self, predicted_uv, joint_id) -> Message | None
        """Crop a P×P patch at the PREDICTED image location; run the patch
        model; return a percept with the CONFIRMED location (predicted +
        regressed offset) or None / use_state=False on rejection.
        Coordinate duty: image (u,v) -> hand reference frame (see §3)."""

    def fallback_needed(self) -> bool
        """True when accumulated confidence collapses -> rerun palm detector."""
```

The driving loop (policy) lives outside the SM, per
`train/hand-landmark/README.md` §5: palm-anchor scan `[0, 5, 9, 13, 17]`
first (defines the pose hypothesis), then finger chains.

**`predicted_uv` requires a projection module that does not exist yet
(review finding — do not underestimate it).** Monty's hypothesis lives in a
wrist-centred, scale-normalised hand frame; Kabsch supplies rotation ONLY.
Projecting a graph node into the crop needs the full similarity transform:

```
hand-frame point → R (rotation) → s (scale) → t (translation) → crop u,v
```

Build `HandFrameProjector` as an explicit module: an Umeyama similarity
solve (rotation+scale+translation, the closed-form extension of Kabsch) over
the ALREADY-CONFIRMED joints of the current episode, refreshed after each
confirmation. Ship it with round-trip tests (hand-frame → uv → hand-frame,
sub-pixel tolerance) BEFORE the patch model trains — otherwise projection
error will be misattributed to the patch model.

### 2.3 Environment plumbing (already written)

`train/monty_lab/tbp_adapter/hand_episodes_env.py` implements their
`SimulatedEnvironment` protocol (`step/reset/close`) and is smoke-tested
inside the tbp.monty env. Swapping its NPZ-replay `_observe()` for live
`PatchSensorModule.sense()` calls is the integration point — the protocol
and types don't change.

### 2.4 OpenPAVE runtime contract (unchanged)

The GUI consumes any recogniser through one seam: a worker emitting
`results_ready(dets, intent, timing, hands)` (see
`pave_ui/perception.py::MediaPipeSvmWorker`) and artifacts under
`train/runs/` discovered by title. A successful patch SM slots in as one
more `CPU · …` runtime; nothing upstream changes.

## 3. What the hand landmarks expect (conventions — violations have bitten)

- **Topology**: 21 joints. 0 = wrist; thumb 1–4; index 5–8 (5 = MCP, 8 =
  tip); middle 9–12; ring 13–16; pinky 17–20. Bone graph:
  `perception.py::HAND_CONNECTIONS`.
- **Image coordinates**: normalised u,v with x → image-right, y → image-DOWN.
- **z**: MediaPipe's z is *relative depth, wrist-anchored, teacher sensation
  not ground truth* (see hand-landmark README §1 — real depth supervision in
  the original came from 100k synthetic renders; do not pretend z is metric).
- **Hand reference frame** (what Monty learns in): subtract landmark 0,
  divide by max |coord|, **orientation kept** — pose is solved at recognition
  time (`monty_lab/evidence_lm.py::_normalise`).
- **Direction semantics**: subject-centric; unmirrored camera means "their
  left" = image-right (`mediapipe_svm.point_direction`, PAVE_POINT_MIRROR).
  Calibrate empirically for any new capture chain — this inverted our turns
  once.
- **Index vector** 5→8 defines pointing; the ±35° horizontal cone gates
  LEFT/RIGHT.

## 4. Models to reverse-engineer / warm-start from (all on disk)

| artifact | size | what it teaches |
|---|---|---|
| `train/weights/hand_landmarker.task` → `hand_landmarks_detector.tflite` | 5.48MB | THE reference: consumes a 224² canonical crop → 21×(x,y,z) + presence + handedness. Extract via `train/hand-landmark/offline.py teacher --extract`; inspect graph in Netron. Its regression head over a rotation-normalised crop is precisely what the patch model decomposes per-joint. |
| same bundle → `hand_detector.tflite` | 2.34MB | BlazePalm-style ROI stage — the "fallback_needed" path. |
| `train/datasets/dynamic_gestures/models/hand_detector.onnx` | 1.2MB / 2.35ms | Alternative ROI detector (Apache-licensed, HaGRIDv2-trained, 1M images). Strong on clean scenes, blind on blurred crude frames — measured. |
| `train/runs/tiny_gesture/trunk.onnx` | 959k params / ~1ms | Our own landmark-regression trunk (v3). Its conv features are the natural warm start for the patch model; its full-frame landmark accuracy is the baseline the per-patch approach must beat. |

"Reverse engineering" here means behavioural distillation (the bundle holds
inference weights only — no trainer comes back out of a TFLite file) plus
architecture reading via Netron for input sizes, anchor schemes, and head
shapes.

## 5. Dataset dependencies

**Already on disk (nothing to download) — with a review-verified caveat:**
every shard in `~/.cache/openpave/datasets/*/prepared.npz` stores full frames
*and* teacher landmarks (`landmarks`, `has_lm`): crude 4.8k, hagrid_shapes
7.25k (GT quotas), jester 21.3k, ipn 2.1k, yolo26 1.5k (GT), swipe_phases
1.2k ≈ 38k supervised frames. BUT the stored landmarks are `(N, 42)` =
**21×(x,y) ONLY — no z** (verified 2026-07-15), and ~1.7% of coordinate
values fall outside [0,1] (hand partially out of frame — the patch generator
must clip or drop those joints, decided per-joint not per-frame). The shards
therefore train exactly the Stage-1 target — `patch → Δu, Δv, match` — and
nothing more. The only xyz store is `train/runs/monty_gestures/episodes.npz`
(96 episodes, teacher-z, bounded accepted subset); joining it back to RGB
frames via source indices is possible but supplies *teacher-relative z*, not
metric depth. Three tiers, never to be conflated: (1) 2D patch corrections =
pixel evidence; (2) teacher-relative z = prior; (3) metric 3D = absent.

**Must be generated (one script, teacher-driven, unattended):** the patch
dataset. For each supervised frame and each joint: crop P×P (32–64px) at the
teacher landmark **plus a random jitter δ**, label = (−δ as the regression
target, match=1); plus off-joint/background crops (match=0). Yields
~38k × 21 ≈ 800k patch samples from existing shards. Store per-source, same
non-destructive shard pattern.

**Optional, only if metric z ever matters:** synthetic hand renders (the
paper's approach). Not needed while z remains "teacher sensation".

## 6. What it will take (workplan, calibrated to this machine)

1. **Patch shard generator** (~1 day incl. verification montages). Reuses the
   `gesture_lab` prepare pattern.
2. **Patch model** (~2–3 days of runs): tiny CNN, 32–64px in → (Δu, Δv,
   match-logit[, z]). Budget: ≤50k params, target ≤50µs/patch so a 21-joint
   walk stays ≤1–2ms. Train torch-MPS, export ONNX fp32 (the established
   pipeline; MLX has no export path and can't run on the Orion).
3. **Sensing policy + coordinate duty** (~2–3 days): palm-anchor scan →
   finger chains; `HandFrameProjector` (Umeyama, round-trip-tested) for
   predicted_uv; confidence-collapse fallback to the ROI detector.

   **Cold start (review finding — the anchors have a bootstrapping problem):**
   the first frame has no confirmed joints to solve a projection from, and a
   box-only detector (`hand_detector.onnx`: boxes/labels/scores, no
   keypoints — inspected) cannot seed `[0, 5, 9, 13, 17]`. Options, in
   recommended order:
   a. **First spike: v3 trunk as coarse initialiser** (exists, ~1ms, outputs
      all 21 rough positions) + patch refinement on top. This cheaply answers
      the only question that matters first: does sensorimotor patch
      refinement add accuracy over its initialiser?
   b. BlazePalm's palm stage regresses 7 palm keypoints per the MediaPipe
      paper — the bundled `hand_detector.tflite` may expose them; verify its
      output tensors in Netron before building anything new.
   c. Train a five-anchor palm model (only if a and b both disappoint).
   d. Fixed anchor templates in a rotation-normalised ROI (weakest; accept
      degraded cold-start only as a stopgap).
4. **Tracking** (~1 day): previous constellation → next ROI (the MediaPipe
   trick); detector only on collapse — this is where the latency win lives.
5. **Validation** (hours, harnesses exist): the yolo26 referee
   (`gesture-lab.sh eval-v3` pattern), the equivalence-probe matrix
   (`equivalence_probes.py run`), plus a NEW landmark-error eval vs teacher
   (mean px error per joint on held-out frames). Gates: wrong-action ≤ the
   MediaPipe pipeline's, landmark error ≤ ~5px @384, end-to-end ≤2ms steady
   state, and honest degradation (confidence collapse → abstain, never
   hallucinate — remember the teacher itself hallucinates gestures on empty
   frames; probes cover this).
6. **GUI runtime + A/B** (~half day: the seam is standard by now).

Realistic total: **1–2 focused weeks**, mostly unattended compute. The risk
concentrated in step 3: patch-local ambiguity (knuckles look alike) is
exactly what Monty's evidence accumulation is supposed to absorb — if it
doesn't, the honest fallback is keeping the v3 trunk as a coarse initialiser
with patch refinement on top.

## 7. Reading list (ordered by usefulness)

1. *MediaPipe Hands: On-device Real-time Hand Tracking* — arXiv:2006.10214.
   The two-stage design, crop canonicalisation, tracking-not-detecting, and
   the real/synthetic data split. The charter document.
2. Thousand Brains docs — "Using Monty in a Custom Application" +
   `tests/unit/frameworks/models/evidence_matching/evidence_lm_test.py` in
   the tbp.monty clone (the API ground truth; docs lag the code).
3. *BlazeFace / BlazePalm* (arXiv:1907.05047 and the MediaPipe model cards) —
   anchor scheme for the ROI stage you keep.
4. *Recurrent Models of Visual Attention* (Mnih et al., 2014) and glimpse-
   network follow-ups — the ML lineage of "sample a patch where you predict,
   integrate over fixations"; useful for the policy and for failure-mode
   expectations.
5. *HaGRIDv2* (arXiv:2412.01508) — evidence that dynamic recognition can be
   built on static training + tracking; motivates keeping the temporal layer
   out of the SM.
6. This repo's own ledger: `train/hand-landmark/README.md` (charter),
   `docs/monty-3d-evidence-tower.md` (the geometry stage, built),
   `train/datasets/README.md` (data facts, dead ends already ruled out).
