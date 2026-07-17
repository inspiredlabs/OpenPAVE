# Training hand landmarks with `tbp.monty`

Status: diagnostic integration verified; MediaPipe replacement is **not ready
for runtime promotion**. The 2026-07-14 retraining (below) replaced the
full-frame regressor with the oracle-ROI cold-start landmarker this document
prescribed, cutting mean landmark error roughly 4–5× on both frozen splits.

## What we built

OpenPAVE and `tbp.monty` cannot share one Python environment:

- OpenPAVE: Python 3.12 arm64 in `.venv`.
- Monty: Python 3.8 osx-64/Rosetta in
  `/opt/anaconda3/envs/tbp.monty`.
- Bridge: `train/runs/monty_landmark_alignment/episodes.npz`.

`prepare_landmark_pairs.py` creates paired observations from the same stored
frames: frozen MediaPipe landmarks are the teacher and — since 2026-07-14 —
the oracle-ROI soft-argmax heatmap landmarker is the student
(`--student pixel-sensor` restores the legacy v3 trunk plus patch refiner for
comparison). It does not invoke MediaPipe. Each observation contains the image
source/index, split, gesture label, 21 teacher coordinates, 21 student
coordinates, and student confidence.

`HandLandmarkSensorModule` is a real Monty `SensorModule`. For each numbered
joint it emits a CMP `Message` containing:

- a 21-way categorical `joint_id` feature;
- a location in a wrist/palm-centred common frame;
- local pose vectors derived from the parent bone;
- confidence and `use_state` acceptance.

Teacher and student pass through identical transforms, but the learning module
can see only the selected stream. Raw pairs remain available solely for
accountability and measurement.

`run_framework_pretraining.py` runs the actual
`MontySupervisedObjectPretrainingExperiment`. Monty owns the exploratory
episode lifecycle, supplied object/pose target, graph update, checkpoint, and
framework logging through `BasicCSVStatsHandler` and `DetailedJSONHandler`.
The detailed output loads with Monty's official `load_stats` helper. A
no-command motor-system shim records the fixed joint traversal and can never
produce robot actions.

The earlier `run_landmark_alignment.py` is retained as a programmatic
student-evaluation diagnostic. It uses real Monty model/message APIs but is not
presented as the supervised experiment lifecycle.

## Reference frame and environment lessons

The common-frame transform is computed independently for each constellation:

1. landmark 0 is the origin;
2. wrist-to-middle-MCP (0→9) defines the longitudinal axis;
3. the median wrist-to-palm-anchor distance defines scale;
4. image coordinates are embedded in 3D with `z=0`.

One-hot joint identity replaced the incorrect ordinal `joint_id / 20` feature.
The Monty graph extent had to increase from 4 to 8 canonical units: the smaller
grid rejected 85.7% of the valid `point` observations and silently omitted that
object model. The verified checkpoint contains `fist`, `like`, `palm`, and
`point`.

The inherited shell `PYTHONPATH` points at Homebrew Python 3.11 packages and
causes Monty's Python 3.8 to load an incompatible `cv2`. Framework runs must
unset it; `train/monty-landmarks.sh pretrain` does this.

## Results and visual accountability

When MediaPipe supplies the landmarks, the earlier gesture evidence experiment
learns four objects and reaches 15/20 (75%) held-out recognition. This shows
that the downstream geometry/evidence idea works when its input geometry is
credible.

The student landmark stream does not yet meet that condition:

| Evaluation | Frames | Mean error at 384 px | Gesture recognition |
|---|---:|---:|---:|
| Exploration holdout | 89 | 81.86 px | 30.34% |
| Untouched referee | 111 | 60.28 px | 30.63% |

The 35,175-parameter patch refiner improves the trunk average (78.61→73.25 px
and 68.11→64.26 px in the earlier benchmark), but that improvement is too small
for a trustworthy constellation.

`render_framework_replay.py` renders Monty's own detailed sensor log over the
stored RGB frame:

- red: MediaPipe teacher;
- green: student/replacement;
- yellow: teacher-to-student residual vector;
- joint numbers: landmark identity.

The four bounded proof episodes show mean errors of 22.4, 58.5, 32.6, and
34.3 px. They are supervised pretraining examples, not substitutes for the
held-out benchmark. The green skeleton visibly collapses, changes scale, and
misplaces individual fingers. This is an accountable failure, not a candidate
for GUI or robot promotion.

## 2026-07-14 retraining: oracle-ROI cold-start landmarker

The strategy sections below were applied literally and the offline procedure
was re-run end to end:

1. **Oracle-ROI student (§4–5 of the contract).** A 93,546-parameter
   soft-argmax heatmap landmarker (`train_oracle_student.py`, 96 px crop,
   24×24 heatmaps) sees only the teacher-defined oriented ROI — the
   acquisition-free benchmark. Training uses controlled ROI
   translation/scale/rotation error (curriculum step 3), photometric jitter,
   label-preserving horizontal flips, bone-length consistency loss, fingertip
   oversampling (×1.5), and a per-joint confidence head trained against
   actual localization success, P(error < 5% ROI). The selected acceptance
   threshold (0.275, F1 0.59) is recorded in
   `oracle_student/meta.json`.
2. **Multi-anchor hand frame.** `landmark_contract.canonicalize` now fits the
   longitudinal axis from the confidence-weighted MCP centroid [5, 9, 13, 17]
   with a 0→9 fallback, and excludes zero-weight anchors from the scale
   median. One bad anchor now shifts the frame instead of rotating and
   rescaling every joint. The sensor module passes student confidences as
   weights; the teacher stream uses uniform weights.
3. **Grid-extent consistency.** The diagnostic runner's `max_graph_size` was
   raised 4→8 to match the framework run (the four-unit grid had silently
   rejected 85.7% of valid `point` observations).

Results against the same frozen splits (student stream, teacher-defined ROI;
oracle-ROI benchmark, so acquisition error is excluded by construction):

| Evaluation | Frames | Mean error at 384 px | PCK@10px | Gesture recognition |
|---|---:|---:|---:|---:|
| Exploration holdout (was 81.86 px / 30.34%) | 89 | 17.61 px | 45% | 41.6% |
| Untouched referee (was 60.28 px / 30.63%) | 111 | 11.20 px | 59% | 40.5% |

Full landmarker benchmark over all frozen frames (not just Monty episodes):
holdout mean 17.28 px / median 11.68 px / p95 51.9 px; referee mean 11.66 px /
median 8.36 px / p95 32.2 px. The legacy full-frame student measured
73.25/64.26 px mean on the same protocol — the poorly conditioned full-frame
problem, not regression itself, was the dominant failure, exactly as §"Why the
current strategy fails" argued. The re-rendered proof episodes drop to 12.0,
32.7, 17.9 and 24.6 px and the green skeleton stays hand-shaped at the correct
scale.

### §9 frontier: proposed-ROI acquisition and GUI empirical testing

The first post-oracle gate was implemented the same day. New artifacts, all
under `train/runs/monty_landmark_alignment/oracle_student/`:

- `proposer.onnx` (+ `proposer_meta.json`): a 78k-parameter full-frame
  oriented-ROI proposer (§4 candidates 1–2) — centre heatmap soft-argmax,
  log-size, +v orientation vector, presence — supervised entirely by
  teacher-derived oracle ROIs, with `no_hand` frames as presence negatives.
  Holdout: centre error 28 px median, size ratio median 1.05, orientation
  median 21°. Its raw presence head is weak (77% false accepts alone) and is
  therefore never trusted alone.
- `oracle_runtime.py`: the §7 state machine as a webcam runtime —
  GLOBAL_SEARCH (proposer, legacy landmark_tower detector as fallback) →
  COLD_START (multi-scale probe, keep the proposal with strongest landmark
  evidence, one oriented re-inspection) → TRACKING (next ROI from the
  previous accepted constellation) → back to GLOBAL_SEARCH after sustained
  evidence collapse. Acceptance requires ≥60% of joints above the trained
  confidence threshold; lowering that gate was measured to convert
  abstentions into wrong intents and was rejected.
- `benchmark_proposed_roi.py` (`./train/monty-landmarks.sh proposed`):
  landmark metrics reported twice per §4. Current acquisition penalty
  (single-frame cold start, no tracking): holdout 18.8→55.3 px mean
  (+36.5 px), referee 12.1→82.1 px mean (+70.0 px); end-to-end false-proposal
  rate on `no_hand` frames 18.75%; cold-start latency p95 8.5 ms. The
  landmarker was retrained with crop-error augmentation matched to the
  measured proposer noise (±17°, 0.7–1.45 scale, ±12% translation), costing
  ~0.4 px of oracle-ROI accuracy. That robustness trade-off is not free
  downstream: with the crop-robust weights the Monty diagnostic recognition
  reads 24.7% holdout / 36.0% referee (landmark error 18.9 / 11.6 px),
  versus 41.6% / 40.5% for the narrow-augmentation weights — constellation
  sharpness and crop tolerance currently trade against each other, so the
  results table above belongs to the narrow model and the deployed
  `landmarker.onnx` is the crop-robust one chosen for live acquisition.
- `replay_crude_videos.py`: webcam simulation driving the actual PyQt worker
  over `train/crude` videos. stop.mp4: 127 STOP vs 3 wrong-intent frames on
  accepted hands. fist/like/point videos are dominated by abstention — the
  cold-start acquisition, not the gesture evidence, is the bottleneck, and
  wrong actions stay rare by construction.

The GUI dropdown (`CPU · Landmark + Monty (3D evidence)`) now lists the
`landmark+monty · oracle-roi` candidate alongside the legacy tower; selecting
it loads `OracleLandmarkerRuntime` behind the unchanged Monty evidence stage
(palm→STOP, fist→HOME, like→TROT, point→LEFT/RIGHT by geometry).

Remaining gaps before promotion (unchanged in kind, smaller in degree):

- PCK@5%-ROI is 28% on the referee against the ≥95% `v1` gate; fingertips
  (4, 20) and the pinky chain carry the worst per-joint means (13–15 px).
- Gesture recognition through Monty (≈41%) still trails the teacher-stream
  ceiling (75%), so constellation shape — not just mean error — remains
  imperfect.
- Acquisition is still oracle-supplied: the proposed-ROI benchmark, occlusion
  and recovery, live replay, and command-isolation gates are untouched.
- Wrong-action framing: this model emits no commands anywhere
  (`commands_enabled: false` in every summary); the capability × tower
  equivalence matrix is unchanged because no runtime artifact was modified.

### Session record (2026-07-14): reality check and the next training round

Everything executed this session, in order, with the artifact each step froze:

| Step | Command / change | Key result | Artifact |
|---|---|---|---|
| Contract fix | multi-anchor `canonicalize` (MCP centroid [5,9,13,17], weighted, 0→9 fallback) + 6 unit tests | one bad anchor shifts instead of rotating the frame | `landmark_contract.py` |
| Student v1 (narrow aug ±12°, 0.85–1.2, ±6%) | `monty-landmarks.sh student` | oracle-ROI: holdout 17.28 px / referee **11.66 px** mean (was 73.25/64.26) | overwritten |
| Monty diagnostic (narrow) | `prepare` + `run` + `analyze` | 17.61 px / 41.6% holdout; 11.20 px / 40.5% referee | `tbp_run/` (superseded) |
| Replay-log bug | consolidated `detailed_run_stats.json` must outrank the stale per-episode dir | proof episodes 12.0/32.7/17.9/24.6 px (were 22.4/58.5/32.6/34.3) | `render_framework_replay.py` |
| Proposed-ROI benchmark v0 | `monty-landmarks.sh proposed` (legacy detector) | penalty **+49/+96 px**; detector box 2.1–3.3× too large, centre err 47–89 px | `proposed_roi_benchmark.json` |
| Detector calibration | measured on train split only: oracle size = 0.52 × detector side | penalty barely moved → detector, not calibration, is the limit | `oracle_runtime.py` |
| Student v2 (wide aug ±17°, 0.7–1.45, ±12%) | retrain | oracle-ROI 18.46/12.06 px (−1 px cost); **deployed weights** | `oracle_student/landmarker.onnx` |
| Oriented ROI proposer (78k) | `train_roi_proposer.py` | centre 28 px median, size ratio 1.05, angle 21° median; presence head weak (77% false accepts alone) | `oracle_student/proposer.onnx` |
| Proposed-ROI benchmark v1 | proposer + multi-scale cold start + refinement pass | penalty **+36.5/+70.0 px**; end-to-end false proposals 18.75%; p95 8.5 ms | `proposed_roi_benchmark.json` |
| Webcam simulation | `replay_crude_videos.py` through the real PyQt worker | stop.mp4: 127 STOP / 3 wrong intents; fist/like/point: mostly abstention | script kept in repo |
| Gate sensitivity | min_joint_fraction 0.4→0.3→0.25 | recovers nothing; converts abstentions into palm/STOP false intents → rejected | default stays 0.4 |
| Monty diagnostic (wide, deployed) | re-run for consistency | 18.9 px / **24.7%** holdout; 11.6 px / 36.0% referee | `tbp_run/` |
| GUI | dropdown lists `landmark+monty · NEW oracle-roi · 94k` first (default), `legacy tower · 71k` second | both load through `LandmarkerMontyWorker` | `pave_ui/perception.py` |

**Reality check (user-verified in the GUI): the legacy 71k tower currently
infers better live than the new 94k student — the 94k rejects too much.**
This is the correct reading of the data above, and the mechanism matters for
the next round:

1. **Co-training beats component quality.** The 71k landmarker was trained on
   crops jittered exactly the way its own detector proposes them
   (`_hand_crops`: side = bbox × 1.65–2.2, centre noise ∝ side). Its training
   distribution IS its deployment distribution. The 94k student is 5× more
   accurate on oracle crops but was never trained on the proposer's actual
   error distribution — measured centre error has a heavy tail reaching
   30–100% of ROI size, far beyond the ±12% translation it saw. Component-wise
   excellence lost to distribution match.
2. **Honest gates read as failure.** The 94k confidence head estimates
   P(error < 5% ROI) and was calibrated on oracle crops; under proposer crops
   that probability genuinely is low, so the ≥60%-of-joints gate abstains.
   The legacy path gates only on presence ≥ 0.5 / quality ≥ 0.15 and always
   emits a constellation. Downstream, Monty's evidence accumulation tolerates
   moderate landmark noise far better than absence — an imperfect skeleton
   still votes; a rejection votes for nothing.
3. **Cold-start lock-in.** Tracking (which restores near-oracle ROIs) engages
   only after one frame passes the gate. On compact gestures (fist, like) the
   cold start rarely passes, so the runtime never leaves GLOBAL_SEARCH and
   abstains forever — the penalty compounds temporally.

**Reorganized next training round — maximize penalty reduction and the live
GUI metric, in this order:**

1. **Train the landmarker on the proposer's real outputs, not synthetic
   jitter.** Run the frozen proposer over every training frame, keep its
   proposed ROI, and train the student on those crops (teacher labels
   projected through the proposed ROI; joints outside the crop become
   visibility 0). Mix ~30% oracle crops to preserve the precision ceiling.
   This is the single highest-leverage change: it is exactly what
   `landmark_tower` does implicitly and why 71k wins live today.
2. **Recalibrate confidence and gates on the deployment distribution.** The
   confidence target stays P(error < 5% ROI) but must be trained/thresholded
   on proposer crops; select `min_joint_fraction` and the per-joint threshold
   on the live-replay metric (crude videos), not on frozen oracle frames.
   Report acquisition rate, wrong-intent rate, and median time-to-first-lock
   per video as the selection criteria.
3. **Make the live replay an explicit gate.** `replay_crude_videos.py` output
   (per-video accepted/abstained/wrong-intent counts) joins the frozen
   benchmarks in meta.json; a candidate that improves oracle PCK but worsens
   stop/fist/like acquisition is rejected automatically.
4. **Iterate proposer ↔ landmarker.** After step 1, mine the frames where the
   refined constellation disagrees most with the proposer ROI (hard
   examples), retrain the proposer on them, then repeat step 1 once. Two
   rounds of this alternation is the budget; measure the penalty after each.
5. **Soften the abstention cliff, not the gate.** Keep per-joint honesty but
   let the runtime emit a *partial* constellation when ≥60% of joints pass
   (Monty already accepts per-joint `use_state`); a lock then upgrades to
   tracking. Never fabricate rejected joints — emit them as missing, exactly
   as the contract (§5) requires.
6. **Keep two eval columns forever.** Oracle-ROI (capability ceiling) and
   proposed-ROI/live (deployment truth) — this session's central lesson is
   that optimizing the first column alone does not move the second.

The narrow/wide synthetic-jitter results remain historical baselines. The
acquisition-matched round deliberately replaces that flavour choice with the
measured proposer distribution; old artifacts, not the current training entry
point, are the source of reproduction for those superseded runs.

### Implemented acquisition-matched round contract

The next-round code now makes that reorganization enforceable rather than
advisory:

- `train_oracle_student.py` freezes the configured proposer over train,
  validation and referee frames. Seventy percent of training crops are its
  exact outputs and 30% are oracle crops by default. Out-of-crop teacher joints
  have visibility 0. Checkpoint selection and confidence calibration use the
  proposed-ROI validation set.
- Every `meta.json` uses `evaluation_columns.oracle_roi_capability_ceiling` and
  `evaluation_columns.proposed_roi_deployment_truth`. These keys are a stable
  artifact contract; neither may be replaced by one aggregate score.
- `replay_crude_videos.py` compares the candidate with the incumbent 71k path
  through the exact PyQt worker. It calibrates per-joint confidence and minimum
  accepted-joint fraction using acquisition rate, wrong-intent rate, videos
  locked and time-to-first-lock. It writes the selected gates and its complete
  report into the deployment-truth column. Any live regression sets
  `runtime_promotion=false`, even if oracle PCK improved.
- At runtime, accepted joints are emitted and rejected joints are `NaN`.
  `EvidenceLM.infer_partial` aligns only the observed joint IDs; missing joints
  do not receive topology-filled coordinates and do not vote. Full-frame
  rejection remains when too few joints pass.
- `run_alternating_rounds.py` performs exactly two rounds. Round 1 trains on
  the initial proposer; the hard-example miner separately ranks combined
  proposed-crop landmark/ROI-centre errors and false-positive `no_hand`
  presence. Round 2 balances repeated hard positives and hard negatives,
  traces the frozen multi-scale/re-inspection acquisition path, retrains the
  proposer, then retrains the landmarker. Each round retains its oracle/live
  columns, hashes, acquisition penalty and live selection result in
  `alternation_report.json`.

Promotion therefore has a deliberate asymmetry: oracle-ROI can demonstrate
capacity, but only proposed-ROI/live can authorize deployment. A candidate
that is five times better on oracle crops and worse live is correctly rejected.

### Executed two-round result (2026-07-14)

The complete alternation was executed, including 18 live gate configurations
per round over 1,478 scored crude-video frames. Neither round passed the 71k
incumbent acquisition gate, so `selected_round` is deliberately `null` and
both candidates remain GUI-visible as rejected experiments.

| Metric | Round 1 | Round 2 | 71k incumbent |
|---|---:|---:|---:|
| Oracle holdout mean | 21.92 px | **20.70 px** | — |
| Oracle referee mean | 14.22 px | **13.99 px** | — |
| Proposed holdout mean | **53.42 px** | 54.97 px | — |
| Proposed referee mean | **83.32 px** | 86.01 px | — |
| Holdout acquisition penalty | **+30.83 px** | +33.53 px | — |
| Referee acquisition penalty | **+69.10 px** | +72.02 px | — |
| Live acquisition rate | 8.86% | **10.35%** | **36.27%** |
| Live wrong-intent rate | 1.62% | **1.35%** | 3.52% |
| Median time-to-first-lock | 1.47 s | **1.43 s** | 1.67 s |
| Videos obtaining a lock | 4/4 | 4/4 | 4/4 |
| Live selection | REJECT | REJECT | incumbent |

Round 2 improved oracle accuracy, live acquisition and wrong-intent rate, but
worsened both frozen proposed-ROI penalties. Hard-positive oversampling made
the proposer more sensitive (96.7% hand recall) while its standalone
false-proposal rate rose to 88.9%; the complete landmark gate reduced the
observed no-hand false-proposal rate to 19.75%. This is evidence that the next
iteration must balance hard positives with hard negatives and train against
the actual multi-scale/refinement selection path. It is not evidence for
lowering the live gate or promoting either candidate.

The immutable experiment record is
`train/runs/monty_landmark_alignment/oracle_student/alternation/alternation_report.json`.

### Next offline curriculum: preserve Round 2's gains without repeating its bias

Round 2 is useful as a frozen acquisition prior, not as a promoted model. The
next run should start from its proposer and use its landmarker to record which
crop the *complete* cold-start path selects after the three scale probes and
optional oriented re-inspection. Training only on the proposer's raw box was
still not fully acquisition matched: deployment applies a learned crop
selection step after that box.

`run_alternating_rounds.py` now encodes this curriculum:

1. Round 1 uses 30% oracle crops. The other 70% is split evenly between raw
   proposer crops and crops selected by a frozen prior runtime. Point targets
   outside each crop remain invisible.
2. Live recalibration runs through the exact PyQt worker. Oracle metrics cannot
   choose thresholds or authorize promotion.
3. The miner records two balanced pools: difficult hand frames ranked by
   landmark plus ROI error, and difficult explicit `no_hand` frames ranked by
   false hand-presence probability.
4. Proposer training repeats hard positives ×3 and hard negatives ×6, then
   weights binary cross-entropy so the total positive and negative presence
   contributions remain equal after repetition. This directly addresses the
   88.9% false-proposal failure rather than merely adding more arbitrary
   backgrounds.
5. Round 2 uses the newly trained proposer and the complete frozen Round 1
   acquisition path as its E-step. It again retains raw proposer crops so the
   landmarker can still score the first cold-start probes; training only on
   successful refined crops would create another oracle-like shortcut.
6. Both oracle-ROI and proposed-ROI/live columns are immutable selection
   outputs. A live improvement with a frozen proposed-ROI regression is kept
   as an experiment, never silently promoted.

#### Incumbent-beating acquisition extension

The curriculum now includes the bounded-search and partial-lock changes needed
to target the 71k incumbent rather than merely improve Round 2:

- The 16×16 centre heatmap was always trained but was previously discarded at
  ONNX export in favour of one soft-argmax coordinate. New proposers export the
  normalized heatmap without adding parameters. Runtime non-maximum
  suppression retains three separated centre hypotheses. Old four-output
  proposer artifacts remain compatible and behave as one-hypothesis models.
- Cold start tests three scales for each centre and scores candidates with 75%
  weight on `[0,5,9,13,17]` palm-anchor confidence and 25% on all-joint
  confidence. Only the winning hypothesis receives one oriented re-inspection:
  at most ten landmarker passes, not an unbounded image scan.
- A partial constellation can initialize tracking only when wrist 0 and at
  least three MCPs pass their per-joint confidence gates and form a
  non-collapsed palm with bounded spread/length ratio. This is distinct from
  lowering the generic ≥60% joint gate. Rejected joints remain `NaN` and do not
  vote. Palm-only lock is internal: it improves the next frame's ROI but emits
  nothing to Monty until at least six joints are accepted. Its next crop is
  extrapolated from wrist/MCP orientation, palm width and wrist-to-MCP length,
  shifted toward the fingers; weak distal predictions do not define that crop.
  Acquisition and command evidence therefore remain separate.
- Failed but geometrically associated cold starts persist for at most five
  frames. Three frames are required before temporal evidence can help. Because
  webcam frames are correlated, confidence receives only a bounded +0.025 per
  additional frame; probabilities are never combined as independent events.
  Temporal evidence may unlock only through the anatomical palm gate, never by
  making a generic fraction of weak joints appear accepted.
- Proposer presence now uses class-balanced focal BCE (`gamma=2`) after hard
  repeats. Hard `no_hand` examples are ranked by both presence and palm-anchor
  evidence, so hand-shaped hallucinations are mined even when the raw presence
  score alone is inconclusive.
- During landmarker training, a raw-crop sample is drawn across all exported
  top-k peaks, not just the best centre. A crop that misses every teacher joint
  gives visibility 0 and trains confidence rejection. The runtime-selected
  crop mixture still teaches successful refinement.

The benchmark now reports top-k oracle-centre coverage, number of ROI
hypotheses, landmarker passes per cold start, palm-anchor versus generic locks,
independent-frame and sequential `no_hand` false proposals, and landmark
latency. Live promotion additionally requires landmark p95 ≤5 ms. Total
proposer plus landmarker parameters are recorded in `alternation_report.json`;
"71k versus 94k" is not treated as a valid whole-pipeline size comparison.

Pre-registered interpretation for the next full run:

| Live acquisition | Interpretation |
|---:|---|
| <15% | balanced mining/search failed to improve the acquisition regime |
| 15–25% | useful iteration, still materially behind the incumbent |
| 25–36.27% | strong candidate; acquisition works but does not beat 71k |
| ≥60%, wrong intent ≤3%, p95 ≤5 ms | meaningful incumbent-beating result |

The gap between 36.27% and 60% is intentional: a rounding-level win is not
enough evidence to replace the incumbent. The result must also lock all four
videos, retain end-to-end `no_hand` acceptance below 8%, and reproduce across
three seeds before promotion.

To seed a new run from the useful parts of the executed Round 2 without
overwriting it:

```bash
./train/monty-landmarks.sh alternate \
  --out train/runs/monty_landmark_alignment/oracle_student/alternation_balanced \
  --initial-proposer train/runs/monty_landmark_alignment/oracle_student/alternation/round_2/proposer.onnx \
  --initial-acquisition-model-dir train/runs/monty_landmark_alignment/oracle_student/alternation/round_2
```

The right hard-negative mix is *balanced and deployment-derived*: explicit
empty frames, sleeves/arms, faces, high-contrast clutter, and partial hands
that the proposer scores highly. Random easy negatives quickly contribute
near-zero loss and do not repair the tail. Keep negative subclasses and source
recording IDs in reports; do not let multiple adjacent frames from one clip
leak across training and validation.

### Monty repository detail: saccades and grid-cell anchoring

The archived `monty_lab` example names
`SaccadeOnImageDataLoader` and `SaccadeOnImageFromStreamDataLoader`; in current
`tbp.monty` the corresponding design is split into
`SaccadeOnImageEnvironment`/`SaccadeOnImageInterface` and their stream
variants. The important detail is behavioural, not the old class name: a
bounded patch moves over an RGB-D image through look/turn actions; each sensor
observation carries the local colour/depth patch, pixel location, point-cloud
patch and camera transform, while proprioception reports the sensor's 3-D
position. The stream variant changes to sequential RGB/depth scenes; it is not
a special training loss. See the
[Monty Meets World processing example](https://github.com/thousandbrainsproject/monty_lab/tree/main/monty_meets_world#3-processing-data-in-monty)
and the [current 2-D environment source](https://github.com/thousandbrainsproject/tbp.monty/blob/main/src/tbp/monty/frameworks/environments/two_d_data.py).

Our previous `PairedHandEnvironment` replayed joint coordinates with a zero
sensor position. That proved logging and graph accountability but omitted the
key sensorimotor signal. Prepared paired episodes now contain a `joint_order`,
the visited sensor locations, and motor deltas; the environment publishes the
current visited location as proprioception and logs the delta. This is still a
2-D ablation because its `z` is explicitly zero.

`prepare_saccade_simulation.py` adds the missing local visual contract. Every
21-step hand episode contains a joint-centred patch, sensor location, movement
delta, joint identity, and one matched background patch per positive step. Its
curriculum is 60% clean skeleton-only renders, 30% high-contrast skeleton plus
bounded noise, then 30% real RGB. This addresses the shortage of isolated
structural views without pretending that clean renders alone represent
deployment:

```bash
./train/monty-landmarks.sh saccade-prepare --cap 100 --patch 24
```

The output is `saccade_simulation.npz` plus a schema manifest. It is the input
contract for the next pixel-inspecting Sensor Module; it is not yet fed to the
coordinate-only `HandLandmarkSensorModule`, and that limitation is deliberate
in the artifact metadata.

The grid-cell experiment supplies the update order that the future Sensor
Module should follow:

1. apply the motor displacement to the location layer (path integration);
2. process the local sensory patch with the predicted location as context;
3. anchor/correct the location representation from sensory evidence;
4. associate the stabilized location representation with the learned hand.

During inference, freeze learned object graphs, begin from an uncertain
location set, perform the same saccades, and narrow hypotheses only when local
appearance and movement agree. This can reduce finger swaps and preserve joint
identity across motion. It cannot acquire the initial hand ROI, infer real
depth from `z=0`, or replace the landmark estimator. The archived
[grid-cells experiment](https://github.com/thousandbrainsproject/monty_lab/tree/main/grid_cells)
uses L6a grid cells for path integration and L4 sensory anchoring; its mesh
paths are generated over nearby surface points, which is why real 3-D hand
training still needs RGB-D or a synthetic rig.

### Training and inference in simulation

Use two simulation levels and never merge their metrics:

- **2-D sensor simulation (available now):** train local appearance/on-hand
  evidence over the saccade artifact, first with structure-only patches, then
  controlled noise, then real RGB. At inference, initialize from the proposed
  ROI, saccade to predicted joint neighbourhoods, update location by the motor
  delta, accept/reject each patch, and robustly refit the palm frame. Teacher
  coordinates are forbidden from the inference observation.
- **3-D embodied simulation (next data source):** place an articulated hand
  rig in Habitat or another renderer, randomize camera, skin, lighting,
  articulation, occlusion and background, and retain exact depth, joint
  visibility, mesh normals and segmentation. A distant-like policy acquires
  the hand; a surface-like policy visits adjacent mesh/joint neighbourhoods.
  Train graph/object memory with ground-truth poses, then evaluate with labels
  hidden and identical motor/sensor transforms. Only this branch can supervise
  physical normals.

Simulation passes only when a held-out randomization seed and held-out hand rig
improve *live* acquisition/recovery without worsening wrong intent. Synthetic
oracle success is a capability ceiling, exactly like oracle-ROI.

### CPU, memory and thermal policy

The Activity Monitor capture does not show a two-core job: all 14 displayed
CPU cores are at roughly 96–100%, package power is about 59.5 W, the reported
hot sensor reaches 104°C, fan speed is about 5,368 RPM, and swap is 9 GB. The
machine is already using every core. Adding more unrestricted Python/ORT/BLAS
threads would increase oversubscription, memory pressure and sustained heat;
it would not expose unused compute.

The refactored defaults use one bounded pool per process:

- neural CPU training: 8 intra-op threads and 1 inter-op thread;
- live replay: 4 ONNX Runtime threads and 2 OpenCV threads;
- `OMP`, Accelerate/vecLib, OpenBLAS, MKL and NumExpr pools receive the same
  per-stage bound so nested pools do not multiply;
- EvidenceGraphLM parallel hypothesis evaluation is opt-in with
  `monty-landmarks.sh pretrain --multithreading`; for four small hand graphs,
  its overhead may exceed its gain, so benchmark it rather than assuming;
- MPS remains the preferred neural-training device when the installed arm64
  PyTorch reports it available. Thread settings accelerate/bound the CPU
  fallback; they do not make ONNX Runtime use the Apple GPU.

Recalibration now defaults to a 0.90 thermal duty cycle: after each video it
idles for one second per nine seconds of measured work. This is a 10% idle
fraction and changes no frames or rankings. The two-round driver also cools
for 30 seconds after each round's live calibration. The settings are recorded
in `live_replay.json` and `alternation_report.json`:

```bash
./train/monty-landmarks.sh live \
  --ort-threads 4 --opencv-threads 2 --thermal-duty-cycle 0.90
```

This is workload control, not a guarantee about hardware safety. If macOS
reports thermal pressure, repeated throttling, rising swap, or instability,
stop the run, let the machine cool, lower `--train-threads`, and reduce caps or
batch size. Avoid running the neural trainers, Rust analysis and video replay
concurrently; the screenshot's 9 GB of swap is at least as concerning for
throughput as core count.

## Why the current strategy fails—and how MediaPipe avoided it

Monty currently receives already-decoded coordinates. It can learn and compare
their geometry, but it cannot recover image evidence discarded by an incorrect
regressor. The earlier 0→9 normalization also amplified one bad anchor; the
current multi-MCP frame reduces that failure mode but cannot repair bad pixels
or a badly acquired crop.

MediaPipe did not eliminate all of these difficulties; it organized the pipeline
so the landmark model saw a much narrower, better-supervised problem:

| Concern in our pipeline | What MediaPipe Hands does | What OpenPAVE should copy |
|---|---|---|
| Coordinate regression can average or collapse an unconstrained hand | MediaPipe also uses direct regression. It first supplies an accurately oriented palm crop, reducing translation, rotation, and scale variation so network capacity can focus on 21-point localization. | Do not reject regression solely because it is regression. First prove it on oracle-aligned ROIs; compare it fairly with heatmaps or integral regression. |
| Full-frame background variation overwhelms hand structure | Only the palm detector searches the full image. The landmark model sees a normalized hand ROI. Detector training uses varied in-the-wild scenes; landmark training combines real gesture crops with a large rendered hand dataset over varied backgrounds. | Separate acquisition data from landmark data. Train the ROI model for scene diversity and the landmarker for articulation, topology, occlusion, and realistic crop error. |
| Our stored targets are only 2D and Monty receives `z=0` | The paper trains `x,y` from real and synthetic examples and learns wrist-relative depth from synthetic 3D hands. Current Tasks output both normalized image landmarks and a separate `hand_world_landmarks` stream. | Preserve and supervise depth explicitly. Never manufacture a physical normal from `z=0`; export both image-relative and world-coordinate teacher streams. |
| Occlusion and confidence are weakly represented | MediaPipe trains on partially visible/self-occluded hands and learns a consistent pose representation. It exposes a whole-crop hand-presence signal and handedness, but the standard Hand Landmarker result does **not** expose a reliable per-joint visibility score. | Add per-joint visibility/uncertainty to our student even though MediaPipe's public result does not. Use whole-hand presence only for crop acceptance and reacquisition. |
| Our frames are independent | MediaPipe derives the next ROI from the previous frame's landmarks. It runs global palm detection on the first frame and again only when hand-presence/tracking confidence fails. The paper does not claim a recurrent landmark network; continuity is primarily pipeline-level crop tracking. | Track the accepted hand frame and ROI, inspect local pixels, and reacquire globally only after sustained evidence collapse. |
| Gesture accuracy can hide malformed landmarks | MediaPipe evaluates palm detection and landmark error separately; gesture recognition is a downstream application over the skeleton. | Keep per-joint NME/PCK, topology, depth, normal, and jitter gates primary. Gesture equivalence remains necessary but cannot certify landmark quality. |

The correction is important: our regressor is failing because it is solving a
poorly conditioned full-frame/crop-mismatch problem with insufficient structural
and depth supervision—not because direct coordinate regression is inherently
invalid.

### MediaPipe outputs relevant to normal extraction

The current Hand Landmarker result contains three public streams per detected
hand: `handedness`, normalized `hand_landmarks`, and
`hand_world_landmarks`. The generic landmark container can hold visibility and
presence, but those fields should be treated as unset unless the specific model
actually supplies them.

- Normalized landmarks: `x` and `y` are image-normalized; `z` is wrist-relative,
  with magnitude roughly on the scale of `x`. This is a 2.5D image/crop
  representation, not metric camera depth.
- World landmarks: 21 predicted `x,y,z` coordinates in metres with origin near
  the hand's geometric centre. These are the better input for geometric normals,
  but they are still model estimates—not measured surface depth.
- A pure 2D constellation cannot determine a unique 3D surface normal. Rotating
  a 2D bone vector by 90 degrees gives only an image-plane perpendicular.

For a stable **palm-frame normal**, use world landmarks when available:

1. collect palm anchors `[0, 5, 9, 13, 17]`;
2. fit a plane with confidence-weighted PCA/SVD and take the smallest-variance
   eigenvector as the unsigned normal;
3. define the longitudinal axis from wrist 0 toward the MCP centroid and the
   lateral axis from pinky MCP 17 toward index MCP 5;
4. use their cross product to establish a handedness-aware orientation, then
   resolve the remaining sign ambiguity using calibrated camera direction and
   temporal continuity (`dot(n_t, n_{t-1}) >= 0`);
5. reject or hold the previous frame when palm anchors are collinear, strongly
   occluded, or the plane residual is excessive;
6. smooth on the rotation manifold rather than averaging normal components and
   renormalizing without an angular gate.

For a **joint-local frame**, let the parent-to-child bone be the tangent. Cross
that tangent with the accepted palm normal to obtain a binormal, then recompute
an orthogonal local normal. At an articulated joint, the cross product of its
incoming and outgoing bones describes the joint's bending plane, but becomes
unstable when the bones are nearly collinear. These are skeleton-frame
pseudo-normals; 21 centreline landmarks do not contain enough information to
recover independent skin-surface normals at every landmark.

Log the normalized landmarks, world landmarks, crop affine transform,
handedness, timestamp, plane-fit residual, normal sign decision, and tracking
state together. This makes a normal reproducible in the red teacher stream and
allows the student/Monty pipeline to be graded on angular error as well as point
error.

## Where to use Distant-Agent and Surface-Agent?

### What the Distant Agent actually contributes

Monty's Distant Agent remains away from an object and points or tilts its
sensors across the visible surface. Its `GetGoodView` positioning procedure can
use a wide-field viewfinder to put a simulated object at a useful scale.
However, `GetGoodView` may use privileged simulator semantic/depth information,
runs before an episode, and is explicitly unavailable to the learning module
during normal inference. A webcam provides neither ground-truth semantics nor
metric depth.

Therefore a Distant Agent does **not** solve foreground exclusion by itself. In
OpenPAVE, its useful role is a **distant-like acquisition controller** around a
small learned hand-presence/oriented-ROI model:

1. inspect the full frame or a low-resolution feature map;
2. propose and score one or more oriented palm/hand regions;
3. select a normalized crop and initialize the landmark model;
4. saccade to alternative proposals when evidence is ambiguous;
5. rerun global acquisition only when tracked hand confidence collapses.

This preserves MediaPipe's important separation between full-frame acquisition
and crop-local landmark estimation without requiring BlazePalm specifically.
The foreground decision must come from our own learned presence/mask/ROI
evidence, not from the Monty agent name.

### What the Surface Agent actually contributes

Monty's Surface Agent is designed to stay close and perpendicular to a physical
object surface, move tangentially, and optionally follow principal-curvature
directions. Its depth transform clips distant observations and constructs an
on-surface mask around the centre of a sensor patch. Those behaviours can
exclude unrelated geometry in a depth-enabled simulator, but they do not
automatically transfer to a monocular RGB hand image.

For OpenPAVE, implement a **virtual surface inspector** after a hand ROI has
already been acquired:

- its “surface” is the confidence-weighted hand mask and current skeleton graph;
- its position is a small RGB/feature patch around a proposed joint;
- tangential movement follows a finger bone or searches the local heatmap;
- “on surface” means local appearance, topology, and confidence agree that the
  patch belongs to the same hand;
- curvature becomes image/skeleton evidence such as ridge orientation, bone
  direction, joint angle, and fingertip response—not simulated metric curvature;
- leaving the hand reverses or rejects the step rather than inventing a joint.

With an RGB-D camera, Monty's literal depth clipping, surface separation, and
normal estimation could become relevant. With the current RGB webcam they must
be replaced by learned visual evidence.

### Correct combined mapping

```text
RGB frame
  → distant-like global hand/palm proposal (our model, not BlazePalm)
  → oriented, normalized ROI
  → heatmap landmark cold start
  → surface-like local joint inspection and correction
  → multi-anchor robust hand-frame fit
  → Monty evidence update and accept/reject log
  → next-frame ROI prediction
  → global proposal again only after confidence collapse
```

This mirrors the key MediaPipe lesson: the full-frame detector supplies an
oriented crop, the landmark model spends its capacity inside that crop, and
subsequent frames normally reuse landmark-derived tracking rather than running
global detection every time. Monty can improve the active sensing, local
verification, and recurrence around that design; it does not remove the need
to learn the initial hand-versus-background decision.

## `landmarker-v2` contract and oracle-ROI benchmark

The five previously open contracts are fixed below. The browser-side reference
implementation is
`train/landmark-observation-tower/src/routes/handOracleGeometry.js`; the live
recorder is `HandLandmarkChart.svelte`. Exports use
`openpave.oracle-roi.dataset.v1`, containing frames with schema
`openpave.oracle-roi.frame.v1`.

### 1. Coordinate and normal definitions

The required geometric outputs are:

1. one robust palm coordinate frame;
2. one skeleton-local coordinate frame for each of the 21 landmarks/bones;
3. bending-plane pseudo-normals for the 15 articulated finger joints;
4. explicit invalidity where any frame is degenerate.

Skin-surface normals are not an output of the 21-point landmarker. A future
surface-normal head must be supervised from synthetic mesh geometry or RGB-D,
not from these skeleton frames.

The palm frame uses world anchors `[0, 5, 9, 13, 17]`. Its origin is the wrist.
The longitudinal reference runs from wrist to MCP centroid; the lateral
reference runs from pinky MCP 17 to index MCP 5. A plane is fitted to all five
anchors with a symmetric-covariance eigensolve. The smallest-eigenvalue vector
is the unsigned plane normal. Its reproducible sign rule is:

```text
dot(pca_normal, cross(pinky_to_index, wrist_to_mcp_centroid)) >= 0
```

The axes are re-orthogonalized after the plane fit. A palm frame is invalid if
its normalized RMS plane residual exceeds `0.15`, if an axis is degenerate, or
if an anchor is non-finite. This topology sign is deterministic and therefore
appropriate for a frozen dataset. Temporal sign continuity may be used by the
runtime tracker, but it must not rewrite exported teacher labels.

For each bone, tangent is parent→child, binormal is
`palm_normal × tangent`, and local normal is `tangent × binormal`. At an
articulated joint, the bending normal is `incoming × outgoing`; it is invalid
when `sin(angle) < 0.05`, and otherwise signed to have non-negative dot product
with the palm normal. These are explicitly named skeleton pseudo-normals.

### 2. Teacher-data extraction and supervision classes

Every label carries a supervision class:

| Output | Reference | Meaning |
|---|---|---|
| Normalized landmarks | MediaPipe Tasks | teacher compatibility pseudo-label |
| World landmarks | MediaPipe Tasks | teacher compatibility pseudo-label, metres as reported |
| Palm/bone/bending frames | derived from MediaPipe world landmarks | geometric teacher pseudo-label |
| Rig joints and mesh normals | synthetic rig | geometric ground truth |
| RGB-D points/normals | calibrated depth camera | measured validation reference |
| Audited RGB landmarks | human review | real-image validation reference |

Teacher compatibility and geometric accuracy are separate benchmark columns.
A student may pass teacher compatibility while failing synthetic/RGB-D normal
accuracy; that result cannot promote a normal-dependent Surface Agent.

MediaPipe result buffers are copied once, synchronously, inside the inference
tick. The resulting snapshot is never mutated. The shared channel retains only
the newest snapshot, assigns a strictly increasing sequence, rejects duplicate
or out-of-order timestamps, and publishes synchronously to subscribers.

### 3. Coordinate contract

Each oracle frame contains:

- monotonic `capturedAt` and wall-clock `timestampUnixMs`;
- source-video and inference dimensions;
- unmirrored normalized image landmarks (`+x` right, `+y` down);
- MediaPipe wrist-relative 2.5D `z` without reinterpretation as metric depth;
- raw MediaPipe world landmarks in metres and their original axis convention;
- handedness categories and score;
- `inferenceMirrored=false` and `displayMirrored=true` as separate facts;
- a reversible normalized-image↔unit-ROI affine transform;
- every derived frame, sign rule, residual, uncertainty source and validity;
- an explicit declaration that per-joint uncertainty is unavailable from the
  public teacher result when visibility/presence is unset.

The oracle ROI is determined only from teacher normalized landmarks. Its `+v`
axis points from MCPs toward the wrist, so fingers face the top of the crop.
All landmarks are projected into this basis; a square covering the maximum
oriented extent is expanded by `1.25`. If `p` is a normalized source point,
`c` the ROI centre, `s` its square size, and `(x_axis,y_axis)` its basis:

```text
u = dot(p-c, x_axis) / s + 0.5
v = dot(p-c, y_axis) / s + 0.5
```

Both `sourceToRoi` and `roiToSource` 2×3 affine matrices are exported. Existing
episodes with `z=0` remain valid only for legacy 2D alignment and must never be
upgraded into this dataset by reconstruction.

### 4. Distant-like ROI acquisition

The first benchmark is acquisition-free: extract the teacher-defined oriented
ROI and run only the proposed student landmarker. This answers one question:
can the student infer acceptable landmarks when localization, rotation and
scale are supplied perfectly?

Do not choose the full-frame proposal model until that test passes. Once it
does, benchmark these acquisition candidates against the same frozen frames:

1. tiny oriented palm/ROI regression head;
2. hand-presence centre/scale/orientation heatmap;
3. previous-student-frame ROI prediction plus a low-rate global fallback;
4. the combination with lowest miss rate under the Raspberry Pi budget.

Acquisition metrics are ROI centre error, scale error, angular error, recall,
false proposals per frame, reacquisition latency and CPU p95. Landmark metrics
must be reported twice: oracle ROI and proposed ROI. Their difference is the
measured acquisition penalty.

### 5. Landmark cold-start model

The cold-start model receives only the normalized oracle crop. Candidate model
families are heatmaps, integral regression and direct regression under identical
data and compute budgets. It must emit 21 coordinates, per-joint confidence or
visibility, whole-hand presence, and handedness. Missing evidence produces a
missing joint, never a topology-filled coordinate.

Losses are confidence-masked coordinate/heatmap loss plus bone-length,
palm-shape, joint-angle, handedness and temporal-consistency terms. Distal
joints, occlusions and crop perturbations are stratified rather than averaged
away.

### 6. Virtual Surface Agent

One Monty movement is a transition from the current accepted joint neighbourhood
to a topologically adjacent candidate or a bounded local-search offset.

One observation contains:

- local RGB or frozen backbone feature patch;
- joint identity and candidate ROI coordinate;
- heatmap/confidence and visibility;
- parent/child bone directions and current robust palm frame;
- appearance agreement with the tracked hand;
- topology residual and teacher residual during training only.

`on_surface` means that appearance, hand mask/presence, topology and local
confidence jointly support membership in the same hand. Failure rejects the
candidate or triggers local search. It never synthesizes a plausible joint.
Termination requires sufficient accepted palm anchors, the task-required
finger anchors, a valid frame fit, and stable evidence. Otherwise the episode
abstains or requests reacquisition.

### 7. Monty evidence, recurrence and temporal state machine

The runtime states are:

```text
GLOBAL_SEARCH → ORACLE/CANDIDATE_ROI → COLD_START → LOCAL_INSPECTION
→ FRAME_FIT → TRACKING
```

Low evidence first rejects an individual joint, then revisits locally. Sustained
loss of palm anchors or invalid frame fit enters `DEGRADED`; exceeding a frozen
consecutive-frame/timeout threshold enters `GLOBAL_SEARCH`. The last accepted
frame may predict search locations but cannot be emitted as a new observation.
Only newly observed evidence advances the observation sequence.

Monty evidence combines local appearance, candidate confidence, topology,
frame-fit residual and temporal innovation. Command isolation remains absolute:
training, benchmark and diagnostic modes have no ROS2/ingres publisher and
must produce an empty command log.

### 8. Losses and curriculum

Train in this order:

1. synthetic rig crops with exact 3D joints and mesh normals;
2. teacher oracle crops with MediaPipe compatibility labels;
3. controlled ROI translation/scale/rotation error;
4. blur, lighting, background, skin-tone and camera variation;
5. structured self-occlusion and missing-joint targets;
6. temporal clips with motion, loss and reacquisition;
7. real audited/RGB-D validation without fitting to it.

Synthetic geometry supervises true normals. Teacher frames supervise compatible
landmarks and skeleton frames. Loss reports remain separated by supervision
class and are never merged into one misleading aggregate.

### 9. Frozen benchmarks and promotion gates

Freeze subject, gesture, camera, lighting and occlusion splits before model
selection. For the oracle-ROI landmarker, report per-joint NME, median/p95 pixel
error, PCK@5/PCK@10, missing rate, finger-swap/topology violations, world-joint
error where supported, palm-normal angular error, valid-normal coverage,
temporal jitter and CPU latency/RSS.

Initial `v1` oracle-ROI gates are:

- PCK at 5% ROI size ≥95% overall and ≥90% on fingertips;
- missing-joint rate ≤1% on unoccluded hands;
- zero finger-identity swaps on the frozen referee;
- palm-normal median angular error ≤10° and p95 ≤25° against the relevant
  reference, with coverage reported separately;
- held-pose jitter no worse than `1.5×` the teacher baseline;
- Raspberry Pi landmarker p95 ≤5 ms and bounded RSS with no positive trend
  over a 30-minute soak;
- oracle-ROI downstream wrong-action rate ≤5%, OOV commands zero, and command
  output empty in benchmark mode.

Passing oracle ROI permits acquisition work; it does not promote runtime use.
Runtime promotion additionally requires proposed-ROI, occlusion/recovery,
cross-subject, live replay and ROS2 command-isolation gates.

### 10. Diagnostic GUI and command isolation

The Svelte widget shows sequence, handedness, normalized/world coordinates,
oracle ROI, palm normal, plane residual, valid bone/bending-frame counts and
exportability. Recording is explicit and bounded (default 1,200 frames). Only
frames with at least one exportable hand enter the dataset; invalid frames are
counted rather than repaired. Export releases its temporary object URL and the
component clears subscriptions, timers, pending frames and recordings on
unmount.

The export contains geometry and metadata, not RGB pixels. A paired RGB capture
must key images by frame sequence/timestamp and record consent/retention policy
separately. The benchmark widget has no OpenPAVE command/ingres/ROS2 dependency
and must remain on a diagnostic route or behind a build-time diagnostic flag in
the PyQt integration.

## Better training and inference strategy

### 1. Prove the visual estimator independently

Freeze a manually audited proof set before further tuning. Separate oracle-ROI
landmark error from detector/crop error. Report per-joint mean, median, p95,
PCK@5, PCK@10, missing-joint rate, scale/rotation error, and temporal jitter.
Keep the exploration holdout and untouched referee immutable.

Replace full-frame coordinate regression with a small crop-based heatmap or
integral-regression model. Train it first on high-contrast, hand-only structural
views, then introduce controlled backgrounds, occlusion, blur, lighting, scale,
and real clutter. Preserve joint visibility masks. Combine robust coordinate or
heatmap loss with bone-length, palm-shape, joint-angle, left/right, and temporal
consistency losses. Oversample distal finger joints and poses that currently
produce the worst errors.

### 2. Make Monty inspect pixels, not merely accept coordinates

Use the visual model only for a cold-start constellation. For every proposed
joint, the Monty sensor module should receive a local RGB/feature patch plus
candidate position, heatmap confidence, and joint identity. Its recurrence is:

```text
predict joint
  → inspect local pixels/features
  → accept, reject, or search locally
  → robustly refit the hand reference frame
  → revisit uncertain joints
```

Fit the reference frame from multiple palm anchors `[0, 5, 9, 13, 17]` using a
confidence-weighted robust fit rather than trusting only 0→9. Store local
appearance evidence and anatomical relationships in the graph. A rejected
joint must remain missing; it must not be replaced by plausible-looking weak
geometry.

### 3. Add temporal sensorimotor evidence

Track the accepted hand frame between webcam frames. Predict the next ROI and
joint neighbourhoods from the previous accepted state, inspect only those
regions, and reset after sustained low evidence. Log every proposal,
accept/reject decision, correction, reference-frame update, and teacher
residual during diagnostic mode.

### 4. Gate promotion

Do not remove MediaPipe from the runtime until the student passes all of these
on frozen data and live replay:

- landmark error and PCK targets, globally and per joint;
- stable topology with no finger swaps or collapsed constellations;
- temporal jitter and recovery after occlusion;
- downstream gesture equivalence without retraining the gesture classifier to
  hide upstream errors;
- bounded latency on the deployment CPU;
- diagnostic command output remains empty.

## Commands and artifacts

From the repository root:

```bash
./train/monty-landmarks.sh hanco-prepare # bounded metric-3D auxiliary shard
./train/monty-landmarks.sh hanco-target  # binary HanCo_tester/no_hand geometry gate
./train/monty-landmarks.sh hanco-gestures-prepare # expand reviewed gesture manifest
./train/monty-landmarks.sh hanco-gestures # train HanCo-only crop diagnostic
./train/monty-landmarks.sh hanco-curriculum # Monty-graph propagation + consistency curriculum
./train/monty-landmarks.sh student   # train on frozen proposer crops + 30% oracle mix
./train/monty-landmarks.sh proposed  # update frozen-frame proposed-ROI column
./train/monty-landmarks.sh live      # calibrate gates + update live promotion gate
./train/monty-landmarks.sh hard      # mine acquisition/landmarker hard examples
./train/monty-landmarks.sh alternate # run and retain both two-round candidates
./train/monty-landmarks.sh saccade-prepare # build local-patch/motor simulation episodes
./train/monty-landmarks.sh prepare   # create paired teacher/student episodes
./train/monty-landmarks.sh run       # diagnostic student evaluation
./train/monty-landmarks.sh analyze   # load logs and report per-joint metrics
./train/monty-landmarks.sh pretrain  # actual supervised Monty experiment
./train/monty-landmarks.sh replay    # render red/green/yellow accountability view
```

HanCo is training-only auxiliary supervision. The prepared shard keeps all
eight cameras from a timestep in the same temporally contiguous split, reads
the changing `K` for every frame, and retains world/camera xyz provenance.
`run_alternating_rounds.py` includes `hanco` in landmarker training by default
but excludes it from deployment validation and gate calibration. The proposer
continues to train only on OpenPAVE-domain sources. Use `--seed` on `alternate`
for genuinely distinct reproduction runs; round 2 uses `seed + 1` and both
values are recorded in `alternation_report.json`.

### HanCo seed-37 revision result (2026-07-16)

The bounded HanCo revision completed both alternating rounds under seeds 37 and
38. It used 432 HanCo training views (all eight cameras across 54 timesteps) as
metric-3D auxiliary supervision, alongside 11,677 OpenPAVE-domain frames. The
104 temporally held-out HanCo views were excluded from deployment validation
and gate calibration. The prepared shard SHA-256 is
`0dfcbce7af16cbd221cf283cb4c241025ccf1da3e147d42500f5dfafd9cd45b4`.

Neither round is eligible for promotion:

| metric | 71k incumbent | round 1 | round 2 |
|---|---:|---:|---:|
| live acquisition rate | 36.27% | 10.01% | 10.22% |
| live wrong-intent rate | 3.52% | 1.69% | 3.52% |
| median first lock | 1.67 s | 1.03 s | 1.57 s |
| landmarker p95 latency | 3.26/3.12 ms | 6.06 ms | 9.94 ms |
| frozen holdout proposed-ROI mean | — | 54.28 px | 68.99 px |
| sequential no-hand false proposals | — | 6.50% | 14.50% |

`selected_round` is therefore `null`; the 71k model remains the runtime
incumbent. Round 2's top-3 search increased centre coverage but also increased
misses, false proposals, error, and compute. Do not spend the next run on more
landmarker capacity or more HanCo duplication. The measured bottleneck remains
deployment-distribution ROI acquisition, especially the heavy-tailed centre
error. The next bounded experiment should keep this landmarker/data contract
fixed and improve proposer presence/centre calibration on real OpenPAVE video
crops. Only after one seed clears the live gate should seeds 41 and 73 be run;
an OpenPAVE-only ablation is still required before attributing any gain to
HanCo itself.

The complete evidence is in
`train/runs/monty_landmark_alignment/oracle_student/alternation_hanco_seed37/alternation_report.json`.

### Single-target HanCo acquisition proof (2026-07-16)

The binary `HanCo_tester`/`no_hand` proof keeps the legacy 71k pixel acquirer
frozen. Sequence `0110` supplies the sole target gesture. Its synchronized RGB
views provide runtime-domain positives; its MANO pose parameters select the
closest non-target poses from the full local HanCo corpus; `xyz` and per-frame
calibration project those hard negatives into all valid camera views. Explicit
`crude` validation frames supply `no_hand`. Target frames are split temporally,
and other poses are held out by sequence; calibration and evaluation are
separate.

The held-out result is **94.55% Presence F1**, **83.65% target acquisition**,
**2.17% no-hand false acquisition**, **0.28% other-pose false acquisition**,
and **0.0 s median first lock**, with 8/8 cameras locking. This meets the
presence proof point without retraining or replacing the 71k front end. It is
display-only in the GUI and cannot issue a robot command. The artifact and
complete provenance are in `train/runs/hanco_target_poc/`.

### Reviewed HanCo gesture mix (2026-07-16)

The reviewed manifest at `train/datasets/hanco/gesture_manifest.json` expands
the confirmed palm, like/thumbs-up, fist, and point selections into **19,705**
camera observations. Every reviewed sequence uses every available frame and
synchronized camera. `0032` is included as point by the latest review;
`0012` and `0018` remain excluded as ambiguous counting sequences.

Sequences `0006` and `0019` have conflicting whole-sequence labels. The offline
preparer assigns every frame to the closest reviewed MANO-pose anchor (`0006`:
fist frame 2 versus point frame 10; `0019`: palm frames 21/58 versus point
frame 35), so no frame/camera observation is duplicated across classes.
Sequence-level train/calibration/evaluation splits keep all eight cameras
together and prevent camera leakage.

The revised head trains on HanCo only: RGB positives plus `mask_hand`-inpainted
negatives from the same frame. It consumes crops supplied by the frozen 71k
acquirer, has 26,413 parameters, and records an empty
`external_training_sources` list. On held-out sequences it reaches **74.50%
overall accuracy**, **55.90% macro F1**, **52.08% correct gesture acquisition**,
and **3.07% no-hand false acquisition**. Per-class correct rates are 65.99%
palm, 59.12% like, 64.11% fist, and 37.05% point. This improves the prior
geometry-only head but does not meet the 71k promotion standard: wrong-gesture
rate is still **46.93%**. It remains a CPU display-only diagnostic; the worker
emits no intent. The ONNX model and full confusion matrix are in
`train/runs/hanco_crop_gesture/`.

### Monty-graph curriculum: unsupervised HanCo leverage (2026-07-16)

`train/hanco_crop_curriculum.py` (`./train/monty-landmarks.sh hanco-curriculum`)
is the first offline curriculum that consumes the ENTIRE tbp.monty sparse
checkpoint. The bridge is
`train/datasets/hanco/monty_reference_graphs.npz`, exported from
`~/Documents/GitHub/monty/tutorials/results/hand_landmarks/pretrained/model.pt`
(1,502 reference-frame graphs, 102,480 canonical constellations) by
`tutorials/hand_landmarks.py export` inside the monty venv — the two stacks
still bridge only via files.

Three mechanisms were built and each was measured, not assumed:

1. **Pose-space label propagation.** Every frame's canonical constellation is
   reduced to a 26-D articulation descriptor (finger curls, splay, fingertip
   reach/elevation, pinch — subject-shape-robust, unlike raw coordinates).
   Reviewed frames are anchors; unreviewed frames take a k-NN vote gated by a
   distance threshold calibrated with leave-one-sequence-out over the reviewed
   sequences: measured propagation precision 0.897 at 8.2% reviewed-frame
   coverage (per-class: palm 0.966, point 0.867, fist 0.836, like 0.833).
   Only **557 of 100,016** unreviewed frames pass (fist 123 / like 92 /
   palm 200 / point 142). That small number is a *finding*: most of HanCo's
   articulation space simply is not one of our four gestures, so HanCo cannot
   be pseudo-labelled into a large gesture corpus. Its unsupervised value is
   geometry supervision and regularization, not free gesture labels.
2. **Foundational-skeleton pretraining (negative result).** Pretraining the
   crop trunk on crop → canonical-3D-pose regression before gesture
   fine-tuning HURT every variant tried (6 or 12 epochs; trunk LR scale 0.25,
   0.5, 1.0) — at 26k parameters the pose task crowds out
   gesture-discriminative features and monocular 64 px crops cannot resolve
   the depth the target contains. The stage remains available
   (`--pose-epochs`) but defaults effectively off in the shipped
   configuration.
3. **Multi-view consistency.** A symmetric-view KL term between synchronized
   cameras of the same frame (weight 0.2) is the ingredient that reduces
   wrong-gesture rate; removing it raised wrong-gesture from 0.449 to 0.508
   in the paired ablation.

Single-run ablations on the frozen reviewed evaluation split (same 26,413-
parameter head and frozen 71k acquirer as the reviewed baseline):

| config | acq | wrong | macro F1 |
|---|---:|---:|---:|
| reviewed baseline (`hanco_crop_gesture.py`) | 0.521 | 0.469 | 0.559 |
| full curriculum (pose 6, trunk 0.25) | 0.389 | 0.517 | 0.455 |
| pose 6, trunk LR 1.0 | 0.429 | 0.502 | 0.511 |
| pose 12, 20 epochs, trunk 0.5 | 0.426 | 0.459 | 0.481 |
| propagation + consistency, no pose | 0.491 | **0.449** | 0.556 |
| … + 8 pseudo cameras | 0.495 | 0.470 | **0.583** |
| … without consistency | 0.487 | 0.508 | 0.540 |

**Variance caveat (measured, decisive):** re-running the winning
configuration with an identical seed reproduced acq 0.454 / wrong 0.486
versus 0.491 / 0.449 — MPS training is nondeterministic and run-to-run
spread (~±4 pp) is the same size as the ablation deltas. Single-run
rankings above are therefore indicative only; the shipped artifact is
selected across three seeds by best calibration macro-F1 (a pre-declared
rule that never touches the evaluation split), and no claim of beating the
52.1% reviewed baseline is made.

Three-seed spread of the shipped configuration (propagation + consistency,
no pose pretraining):

| seed | calibration F1 | acq | wrong | macro F1 |
|---:|---:|---:|---:|---:|
| **37 (selected)** | **0.442** | 0.454 | 0.486 | 0.547 |
| 38 | 0.401 | 0.496 | 0.445 | 0.560 |
| 39 | 0.381 | 0.470 | 0.479 | 0.545 |

Against the single-run reviewed baseline (acq 0.521 / wrong 0.469 / F1
0.559) the curriculum is statistically indistinguishable on wrong-gesture
rate and macro-F1 and slightly behind on acquisition. The honest summary:
the curriculum roughly matches the reviewed baseline while adding measured
propagation and a consistency term, and it falsifies both the "pseudo-label
HanCo at scale" and the "skeleton-pretrain then classify" shortcuts at this
capacity.

The GUI lists the artifact as `hanco-crop-ssl · monty-propagated +
view-consistent · …` in `CPU · Landmarker Tower` (launch `./mlx-runtime.sh`);
it reuses `HanCoCropGestureWorker` unchanged — display-only, commands off.
Artifacts and the full per-config reports are in
`train/runs/hanco_crop_curriculum/`.

Important outputs:

- `train/runs/monty_landmark_alignment/oracle_student/landmarker.onnx` (+ meta.json)
- `train/runs/monty_landmark_alignment/episodes.npz`
- `train/runs/monty_landmark_alignment/tbp_run/summary.json`
- `train/runs/monty_landmark_alignment/tbp_run/comparison.npz`
- `train/runs/monty_landmark_alignment/framework_verified/pretrained/model.pt`
- `train/runs/monty_landmark_alignment/framework_verified/pretrained/train_stats.csv`
- `train/runs/monty_landmark_alignment/framework_verified/pretrained/detailed_run_stats.json`
- `train/runs/monty_landmark_alignment/framework_verified/pretrained/replay/contact_sheet.png`

Implementation lives in `train/monty_lab/tbp_adapter/`. The design follows the
Thousand Brains documentation for
[pretraining](https://docs.thousandbrains.org/docs/pretraining-a-model),
[sensor modules](https://docs.thousandbrains.org/docs/sensor-module), and
[logging/analysis](https://docs.thousandbrains.org/docs/logging-and-analysis#analyzing-data-from-monty_handlers),
plus Monty's
[Distant/Surface Agent policy description](https://docs.thousandbrains.org/docs/policy).
The acquisition/tracking mapping is grounded in the
[MediaPipe Hands paper](https://arxiv.org/abs/2006.10214) and its two-stage
palm-detector/crop-local-landmarker pipeline. Current output and confidence
contracts are checked against the official
[HandLandmarkerResult](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarkerResult)
and
[HandLandmarkerOptions](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarkerOptions)
APIs; the implementation exposes handedness, normalized landmarks, world
landmarks, and separate detection/presence/tracking thresholds. Coordinate and
training details are also recorded in MediaPipe's official
[Hands solution documentation](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md).
