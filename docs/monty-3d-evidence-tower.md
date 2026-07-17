# `CPU · Monty (3D evidence)` — how it was built, and how to rebuild it offline

## Part 1 — Terse: how we built it (July 2026)

The goal was Thousand-Brains-style recognition (docs.thousandbrains.org,
"Using Monty in a Custom Application") for the gesture tower. The real
`tbp.monty` framework would not install into the repo venv (`torch-sparse`
build failure — it ships a pinned conda env), so we implemented the **method**
clean-room in ~150 lines ([train/monty_gesture.py](../train/monty_gesture.py),
generalised later into [train/monty_lab/](../train/monty_lab/)):

1. **An object is a reference frame, not a picture.** Each gesture is its 3D
   landmark constellation: 21 MediaPipe points (x, y, z), wrist moved to the
   origin, scale-normalised, **orientation kept** (pose is solved at
   recognition time, never erased).
2. **Learning is few-shot and gradient-free.** ~10 exemplar constellations per
   object, stride-sampled from the middle 40% of each capture video (the held
   gesture, not the transitions). Total learning time: **~8 seconds**.
3. **Recognition is evidence over pose hypotheses.** Every exemplar is aligned
   to the observation with a Kabsch rotation solve; the residual RMS becomes
   evidence `exp(-(rms/σ)²)`. Best evidence under a floor → answer `noop`:
   **unknown gestures abstain by construction.**
4. **All pointing is ONE object.** In a rotation-solving recogniser, direction
   is pose, not identity — LEFT/RIGHT come from the index-finger vector
   afterwards (subject-centric, empirically calibrated).
5. Iterations, each costing seconds: σ/evidence-floor tuning (0.35→0.50),
   exemplars 6→10 per source, and **+12 cross-subject exemplars per object**
   harvested from ground-truth foreign data (yolo26 train split).
6. The evidence loop was then vectorised (one batched SVD over all exemplars:
   1,890µs → ~500µs) inside `monty_lab.EvidenceLM`, and wired into the PyQt
   GUI as the `CPU · Monty (3D evidence)` runtime (MediaPipe landmarker in
   front, Monty recognition behind, same intent gates as every tower).

Referee result: **66% correct on foreign ground truth from 96 exemplars
learned in 9.6 seconds** — beating CNN students trained for hours on 18k
frames, and abstaining (never mis-commanding) on out-of-vocabulary gestures.

---

## Part 2 — Comprehensive: training it offline, from scratch, unassisted

This describes a single self-contained script — call it `monty_offline.py` —
that takes a machine from nothing to a validated `objects.npz` tower. Each
section names the code the script must contain and why.

### 0. Environment and inputs

- Python ≥3.10 venv with: `mediapipe` (≥0.10.30, Tasks API), `numpy`,
  `opencv-python`. Nothing else — no torch, no sklearn.
- **Capture videos** (the only training data you must create): one `.mp4`
  per gesture, 15–30s, the gesture *held* through the middle of the clip,
  filmed on the same class of camera the robot will use. Naming carries the
  label: `palm.mp4`, `fist.mp4`, `like.mp4`, `up.mp4`, `down.mp4`,
  `right-to-left.mp4` (= point toward the SUBJECT's left), `left-to-right.mp4`.
- **A referee set the model never learns from**: ~100 labelled images of
  OTHER people making the gestures (a small Roboflow export works). Non-
  negotiable — validating on your own frames only measures memorisation.

### 1. `setup()` — downloads and checks

- Fetch the MediaPipe hand-landmark bundle (~7.8MB) if absent:
  `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`
- Instantiate `HandLandmarker` (num_hands=1, detection confidence 0.5) and
  smoke-test it on one frame of each capture video; abort with a clear error
  if any video yields no hands (bad lighting/framing is cheaper to fix now).

### 2. Geometry primitives (the entire "model")

```
constellation(hand):                      # 21x3 reference frame
    pts = [[q.x, q.y, q.z] for q in hand]
    pts -= pts[0]                         # wrist -> origin
    return pts / max(|pts|)               # scale-norm; ORIENTATION KEPT

kabsch_rms(obs, ex):                      # one pose hypothesis
    H = obs.T @ ex;  U,S,Vt = svd(H)
    R = U @ diag(1,1,sign(det(U@Vt))) @ Vt
    return rms(obs @ R - ex)

evidence(rms) = exp(-(rms/σ)²)            # σ ≈ 0.16 to start
```

Vectorise for production: stack all exemplars `(E,21,3)`, batch the SVD
(`np.linalg.svd` on `(E,3,3)`) — one call recognises against every exemplar
of every object in ~0.5ms for E≈100.

### 3. `learn()` — the curriculum, in order

**Step A — self exemplars.** For each video: take frames from the 30–70%
band only (transitions at the ends poison exemplars — we measured a
point-down teardown classifying as fist), run the landmarker, keep the first
~10 constellations. Map sources → objects: all four point videos feed ONE
`point` object.

**Step B — cross-subject exemplars.** Append ~12 constellations per object
from labelled images of other people (your referee set's *train* split, or
HaGRID-class data: `like`→like, `palm/stop`→palm, `fist`→fist, `one`→point).
This is the single highest-value step: exemplars are additive, so it costs
seconds and can never damage Step A (Monty's non-destructive property — a
new person or gesture is an append, never a retrain).

**Step C — direction calibration (do not skip).** Run the landmarker over
`right-to-left.mp4`, compute the index vector (landmark 5 → 8), and check
which image-space direction dominates. On an unmirrored camera, pointing to
YOUR left appears as image-RIGHT — we shipped an inverted turn before
learning this. Persist the resolved sign in the metadata; expose a
mirror flag for camera chains that flip.

**Step D — threshold calibration.** Sweep σ ∈ [0.10, 0.24] and
evidence-floor ∈ [0.3, 0.7] against the referee set, choosing by
**wrong-action rate first, hit rate second** (an abstention delays a robot; a
wrong command moves it). Our landed values: σ=0.16, floor=0.50. Include
out-of-vocabulary probes (a gesture you never taught, e.g. thumbs-down) and
require ~0 commands on them.

### 4. `validate()` — what the script must report before saving

Outcome-level scoring on the referee set, through the exact runtime path
(landmarker → constellation → evidence → direction cone):

- per-class hits vs ground truth (`STOP/HOME/TROT/LEFT/RIGHT/NOOP`),
- **wrong-action rate** (commands ≠ ground truth; the gate metric),
- abstention rate (safe, but watch it — a mute tower is useless),
- OOV probe behaviour (must abstain),
- recognition latency (µs; if >1ms, vectorise per §2),
- and a **stability check**: on a held video, the intended outcome must
  appear on ≥3 consecutive frames within the first ~10 (mirrors the live
  debounce; single-frame flickers are suppressed downstream anyway).

Acceptance gates that served us: wrong-action ≤5%, OOV commands = 0,
recognition ≤1ms, every commanded class ≥50% hit on the referee.

### 5. `save()` — the artifact

- `objects.npz`: one array per object, shape `(E, 21, 3)` float32.
- `meta.json`: exemplar counts per object, σ, evidence floor, direction
  sign/mirror flag, calibration scores, timestamp. The GUI's discovery reads
  this to title the dropdown entry (`monty · 96ex · 3D-evidence-4obj · fp32`).

### 6. Operating it afterwards

- **New person struggles?** Append 5–10 of their constellations to the weak
  object and re-run `validate()` — seconds, nothing else moves.
- **New gesture?** New capture video → new object key in the store → add its
  intent mapping. Existing objects untouched.
- **Deployment note:** the tower is numpy-only and runs anywhere (the Orion
  included). Its input stage — pixels→landmarks — is the part with a compute
  budget (MediaPipe ≈5ms CPU, TFLite GPU delegate on Mali, or a distilled
  trunk); Monty itself never touches pixels and never will.

### Known failure modes (all observed, all handled above)

| symptom | cause | fix baked into the curriculum |
|---|---|---|
| turns inverted | subject-centric vs image-space direction | Step C calibration + mirror flag |
| fist fires during point teardown | exemplars from transition frames | 30–70% band sampling + live debounce |
| one person's gestures weak | single-subject exemplars | Step B cross-subject appends |
| unknown gesture commands robot | floor too low | Step D sweep with OOV probes |
