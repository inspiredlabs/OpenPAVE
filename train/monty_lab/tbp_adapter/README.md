# tbp.monty gesture adapter — status & remaining wiring

The concise project-level record, results, failure analysis, and next training
strategy are maintained in `docs/training-with-monty.md`.

## Paired landmark preparation and analysis

The MediaPipe-replacement experiment now uses the real framework boundary:

0. `train_oracle_student.py` runs in OpenPAVE's arm64 `.venv`. It trains the
   oracle-ROI cold-start landmarker (docs/training-with-monty.md §4–5): a
   soft-argmax heatmap model that sees only the teacher-defined oriented hand
   crop, with ROI-perturbation augmentation, bone-length loss, fingertip
   oversampling, and a per-joint confidence head calibrated against actual
   localization success (P(error < 5% ROI)). `oracle_roi.py` holds the shared
   crop geometry (`+v` from MCPs toward the wrist, ×1.25 expansion,
   reversible source↔ROI mapping).
1. `prepare_landmark_pairs.py` runs in OpenPAVE's arm64 `.venv`. It reads the
   frozen MediaPipe coordinates already present in prepared shards and runs
   the oracle-ROI student on the same frames (`--student pixel-sensor`
   restores the legacy full-frame trunk plus patch refiner). It does not
   invoke or import MediaPipe.
2. `HandLandmarkSensorModule` runs in the separate osx-64 `tbp.monty` env. It
   converts one numbered landmark per step into a CMP `Message`, using a
   21-way categorical `joint_id` and a wrist/palm reference-frame transform.
   The frame's longitudinal axis is now fitted from the confidence-weighted
   MCP centroid [5, 9, 13, 17] rather than the single 0→9 pair, so one bad
   anchor shifts the frame instead of rotating and rescaling every joint.
   Teacher and student streams use identical code, but only the selected
   stream is visible to the learning module.
3. `run_framework_pretraining.py` runs the actual
   `MontySupervisedObjectPretrainingExperiment`. The experiment owns the
   exploratory lifecycle, target object/pose supervision, graph update,
   `BasicCSVStatsHandler`, and `DetailedJSONHandler`. The motor system is a
   no-command logger shim; joint traversal belongs to the stored environment.
4. `run_landmark_alignment.py` remains a diagnostic student-evaluation runner.
   It reports mean/median/p95/PCK and compact per-joint comparisons; it is not
   presented as the framework pretraining lifecycle.
5. `render_framework_replay.py` reads the bounded Monty detailed log and draws
   MediaPipe teacher landmarks in red, student landmarks in green, and their
   residual vectors in yellow on the original stored frame.

Run the complete file-bridged workflow from the OpenPAVE root:

```bash
./train/monty-landmarks.sh student   # OpenPAVE arm64: train oracle-ROI landmarker
./train/monty-landmarks.sh prepare   # OpenPAVE arm64: build the file bridge
./train/monty-landmarks.sh pretrain  # Monty Python 3.8: real supervised lifecycle
./train/monty-landmarks.sh replay    # OpenPAVE arm64: render logged comparison
```

Outputs live under `train/runs/monty_landmark_alignment/`. The evaluation is
diagnostic only and cannot emit robot commands.

The framework checkpoint and logs are under
`framework_verified/pretrained/`; the visual contact sheet is
`framework_verified/pretrained/replay/contact_sheet.png`. The detailed log is
limited to the four supervised proof episodes and uses the consolidated format
accepted by tbp.monty's `load_stats`.

## Working now (verified 2026-07-14)
- OpenPAVE side: `.venv/bin/python -m monty_lab.runner export --task gestures`
  -> `train/runs/monty_gestures/episodes.npz` (96 episodes, 21x3 locations +
  object labels, 80/20 split).
- tbp side: `hand_episodes_env.py` implements their SimulatedEnvironment
  protocol over that file; smoke-tested INSIDE the tbp.monty env with real
  framework types:
  `conda run -n tbp.monty python train/monty_lab/tbp_adapter/smoke.py`

## RUNNING (2026-07-14): real-framework gesture learning works
`conda run -n tbp.monty python train/monty_lab/tbp_adapter/run_gestures.py`
- SensorModule = `episode_to_percepts` (21 CMP Messages: joint-id feature at
  3D pose); config = EvidenceGraphLM constructor (max_graph_size rescaled to
  4.0 for hand-frame units, max_match_distance 0.15).
- Lifecycle mirrors their own unit tests (the supported programmatic entry).
- Result: 4 gesture objects learned few-shot (16 episodes, **0.4s**);
  held-out recognition **15/20 (75%)** at 64ms/episode under Rosetta.
- Two API facts learned (cost three tracebacks): extending an existing graph
  needs `detected_rotation_r = Rotation.identity()` (None only builds the
  FIRST graph), and `buffer.stats["detected_location_on_model"]` must be set
  alongside `detected_location_rel_body`.

## Deliberate boundary

This proof does not promote the model into the webcam runtime. It first makes
the structural failure observable and reproducible. Promotion should occur
only after a held-out student run improves the per-joint/PCK measures and the
logged green skeleton remains stable across background, scale, and pose changes.

Session rule (also in repo CLAUDE.md): every Monty session begins
`cd ~/Documents/GitHub/monty && conda activate tbp.monty`.
