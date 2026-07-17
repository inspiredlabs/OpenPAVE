# OpenPAVE — instructions for AI agents

## Cache-first rule — NEVER duplicate a download
Before downloading ANY dataset or model, check in this order:
1. `~/.cache/huggingface/` — community models (MLX VLM checkpoints, the HaGRID zip)
2. `~/.cache/openpave/` — project datasets. `train/datasets`, `train/yolo26`,
   and `train/insect-poc/raw` are SYMLINKS into it; new projects should link
   the same cache rather than re-download.
3. `./train/gesture-lab.sh list` — prepared training shards (never re-prepare
   what exists; `FORCE=1` is a deliberate user choice)
4. `train/datasets/README.md` — the source catalogue AND past research
   findings (dead links, license gates, verified facts). Do not re-research
   anything recorded there.

`train/crude/` is user-authored capture footage — source of truth. Never
overwrite, regenerate, or delete it.

## Monty training sessions
To begin training towers with Monty, at the start of every session (if not
already active):

    cd ~/Documents/GitHub/monty && conda activate tbp.monty

- The env lives at `/opt/anaconda3/envs/tbp.monty` and imports the package
  EDITABLY from that clone — the clone must not be moved or deleted.
- `pytest` ONLY from inside that clone (elsewhere it collects the wrong
  project → ~1,388 false errors; this burned us twice).
- Any install into that env needs the `CONDA_SUBDIR=osx-64` prefix (Intel
  under Rosetta).
- NEVER install tbp.monty or its deps into `./.venv` (arm64/MLX). The two
  stacks bridge only via files: `train/monty_lab` episode exports and
  `objects.npz` artifacts.

## Definition of done for ANY tower/model change
1. `./train/gesture-lab.sh eval-v3` (or the tower's own eval) — the yolo26
   referee, outcome-level.
2. `.venv/bin/python train/equivalence_probes.py run` — the capability ×
   tower matrix.
Report **wrong-action rate first** (a wrong command moves the robot; an
abstention only delays). Artifacts under `train/runs/` are what the GUI
loads — the runtime dropdown picks up retrains on next activation.

## Conventions that have bitten before
- Pointing direction is SUBJECT-centric (`mediapipe_svm.point_direction`);
  the camera is unmirrored. Calibrate empirically before mapping any new
  dataset's Left/Right labels.
- Directional classes + label-blind flip augmentation poison training; use
  label-swapping flips only.
- GUI: no coloured text labels unless the user explicitly asks; streaming
  status chips must never change character length.
- Temporal data: split by video/recording, never random frames (leakage).
