# tbp.monty install guide (Apple Silicon)

> **VERDICT (2026-07-14): INSTALLED AND WORKING** on macOS 15.5 / Apple
> Silicon — env built with conda 26 (`CONDA_SUBDIR=osx-64 conda env create -f
> environment_arm64.yml`), verified **586 passed, 1 skipped, 0 errors** and
> `cv2 4.13.0` present (issue #931 pre-empted).
>
> **The 1388-error false alarm (hit twice):** `pytest` was run from OUTSIDE
> the clone, so it collected the wrong project's tests under monty's Python
> 3.8. ALWAYS `cd` into the tbp.monty clone before `pytest`. The earlier
> "do not install" verdict was this same artifact.

**Why our pip attempt failed (2026-07-13):** `pip install git+…tbp.monty` into
the repo venv dies building `torch-sparse`. That is not fixable in-venv: on
Apple Silicon, tbp.monty officially runs as an **Intel (osx-64) conda
environment under Rosetta** — its pinned deps (torch-sparse et al.) only
resolve there. There is no PyPI package, and their docs say do NOT use `uv`.

## FAST PATH (use this one — 2026-07-13)

Anaconda's conda 23.9 classic solver hung >1h on this env. micromamba solves
it in minutes:

```bash
brew install micromamba
cd ~/Documents/GitHub/monty
CONDA_SUBDIR=osx-64 micromamba env create -f environment_arm64.yml -y
micromamba run -n tbp.monty pytest          # all pass/skip = installed
micromamba shell init -s zsh && exec zsh    # one-time; then:
micromamba activate tbp.monty
```

Rules: env name is `tbp.monty` (from the yml); prefix any FUTURE install into
it with `CONDA_SUBDIR=osx-64`. Skip YCB/pretrained downloads (Habitat only).

## Downloads — only if you want their simulator benchmarks

Not needed for our landmark-gesture custom app (we feed real sensor data, not
Habitat). Skip unless reproducing their YCB experiments:

```bash
python -m habitat_sim.utils.datasets_download --uids ycb --data-path ~/tbp/data/habitat
mkdir -p ~/tbp/results/monty/pretrained_models/ && cd $_
curl -L https://tbp-pretrained-models-public-c9c24aef2e49b897.s3.us-east-2.amazonaws.com/tbp.monty/pretrained_ycb_v13.tgz | tar -xzf -
```

Env vars if you relocate things: `MONTY_MODELS`, `MONTY_DATA`, `MONTY_LOGS`.

## How it plugs into OpenPAVE

- **Never install it into `.venv`** — the repo venv is arm64/MLX; tbp.monty is
  osx-64/Rosetta. They cannot share a process.
- The integration seam is already built: `train/monty_lab/` mirrors the
  tbp.monty protocol (Environment / features-at-poses / LearningModule). To
  use the real framework, write an adapter in the tbp.monty env that consumes
  our `Episode` streams (21 landmarks as features-at-locations — see
  `monty_lab/tasks/gestures.py`) and exports learned objects back to
  `objects.npz`. The GUI tower (`CPU · Monty (3D evidence)`) reads that file
  and does not care which implementation produced it.
- Custom-app entry point in their docs: "Using Monty in a Custom Application"
  — implement their `Environment`/`Interface` protocols; run experiments via
  `python run.py experiment=<config>`.

## Observed failures (2026-07-13, this machine — do not retry these)

- **`uv sync`** → `torch-scatter==2.1.2` build error
  (`ModuleNotFoundError: setuptools.build_meta` under pyenv CPython 3.9).
  Expected: their docs mark uv "experimental and not supported". Same class
  of failure as pip. Conda-only project.
- **`conda env create -f environment_arm64.yml`** (no `--subdir=osx-64`) →
  `ResolvePackageNotFound: pyg, pytorch-scatter, pytorch-sparse,
  habitat-sim, mkl<2022`. Those packages have no osx-arm64 builds; the
  subdir flag is what makes conda pull Intel builds for Rosetta.
- **`conda activate monty`** → `EnvironmentNameNotFound`. The env name comes
  from the yml's `name:` field and is **`tbp.monty`**, regardless of the
  clone directory's name.

## Gotchas

- Rosetta means Intel emulation: fine for this CPU-bound research code, but
  benchmark before assuming latency numbers transfer.
- `conda env create` without `--subdir=osx-64` on Apple Silicon will resolve
  arm64 packages and fail the same way pip did.
- Re-entering the env later: `conda activate tbp.monty` (the subdir pin is
  stored per-env by step 3).
