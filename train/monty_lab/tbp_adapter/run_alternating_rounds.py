#!/usr/bin/env python3
"""Run two proposer↔landmarker rounds and retain both evaluation columns."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "train" / "runs" / "monty_landmark_alignment" / "oracle_student"


def run(*args, threads=8):
    print("[alternation]", " ".join(map(str, args)), flush=True)
    # The interactive shell can inject foreign-architecture packages through
    # PYTHONPATH (notably Qt). The repository interpreter and cwd already
    # provide every required import.
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    # One bounded pool per process. Unbounded OpenMP/Accelerate/ORT pools can
    # multiply each other and cause the all-core saturation and swap pressure
    # seen during the first experiment.
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS"):
        environment[variable] = str(threads)
    environment["PAVE_ORT_THREADS"] = str(threads)
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True,
                   env=environment)


def cooldown(seconds, label):
    if seconds > 0:
        print(f"[alternation] cooldown {seconds:.0f}s after {label}", flush=True)
        time.sleep(seconds)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_landmarker(
    cfg, round_dir, proposer, acquisition_model_dir=None, *, seed
):
    trained = (round_dir / "landmarker.onnx").exists() and (
        round_dir / "meta.json"
    ).exists()
    if not cfg.resume or not trained:
        args = ["train/monty_lab/tbp_adapter/train_oracle_student.py",
            "--proposer", proposer, "--out", round_dir, "--oracle-mix", cfg.oracle_mix,
            "--epochs", cfg.landmarker_epochs, "--train-cap", cfg.train_cap,
            "--val-cap", cfg.val_cap, "--referee-cap", cfg.referee_cap,
            "--cpu-threads", cfg.train_threads,
            "--sources", cfg.landmarker_sources,
            "--auxiliary-sources", cfg.auxiliary_sources,
            "--seed", seed,
            "--raw-proposer-mix", cfg.raw_proposer_mix,
            "--thermal-duty-cycle", cfg.thermal_duty_cycle]
        if acquisition_model_dir is not None:
            args += ["--acquisition-model-dir", acquisition_model_dir]
        run(*args, threads=cfg.train_threads)
    # Keep the exact frozen acquisition artifact beside its student runtime.
    if proposer != round_dir / "proposer.onnx":
        shutil.copy2(proposer, round_dir / "proposer.onnx")
        proposer_meta = proposer.parent / "proposer_meta.json"
        if proposer_meta.exists():
            shutil.copy2(proposer_meta, round_dir / "proposer_meta.json")
    benchmark = round_dir / "proposed_roi_benchmark.json"
    if not cfg.resume or not benchmark.exists():
        run("train/monty_lab/tbp_adapter/benchmark_proposed_roi.py",
            "--model-dir", round_dir, "--update-meta", "--val-cap", cfg.benchmark_cap,
            "--referee-cap", cfg.referee_cap)
    meta = json.loads((round_dir / "meta.json").read_text())
    live = (meta.get("evaluation_columns", {}).get("proposed_roi_deployment_truth", {})
            .get("live_replay"))
    if not cfg.resume or not live:
        run("train/monty_lab/tbp_adapter/replay_crude_videos.py",
            "--model", round_dir / "landmarker.onnx", "--calibrate", "--update-meta",
            "--stride", cfg.replay_stride, "--ort-threads", cfg.replay_threads,
            "--thermal-duty-cycle", cfg.thermal_duty_cycle,
            threads=cfg.replay_threads)
        cooldown(cfg.round_cooldown_seconds, f"{round_dir.name} recalibration")


def main(argv=None):
    cfg = parser().parse_args(argv)
    cfg.out.mkdir(parents=True, exist_ok=True)
    initial = cfg.initial_proposer
    if not initial.exists():
        raise SystemExit(f"initial proposer missing: {initial}")

    round1 = cfg.out / "round_1"
    train_landmarker(
        cfg,
        round1,
        initial,
        cfg.initial_acquisition_model_dir,
        seed=cfg.seed,
    )
    hard = round1 / "hard_examples.json"
    if not cfg.resume or not hard.exists():
        run("train/monty_lab/tbp_adapter/mine_acquisition_hard_examples.py",
            "--model-dir", round1, "--cap", cfg.hard_cap,
            "--negative-cap", cfg.hard_negative_cap,
            "--fraction", cfg.hard_fraction,
            "--negative-fraction", cfg.hard_negative_fraction,
            "--thermal-duty-cycle", cfg.thermal_duty_cycle,
            "--out", hard, threads=cfg.replay_threads)
        cooldown(cfg.round_cooldown_seconds, "hard-example mining")

    round2 = cfg.out / "round_2"
    proposer_trained = ((round2 / "proposer.onnx").exists()
                        and (round2 / "proposer_meta.json").exists())
    if not cfg.resume or not proposer_trained:
        run("train/monty_lab/tbp_adapter/train_roi_proposer.py",
            "--out", round2, "--hard-examples", hard,
            "--hard-repeat", cfg.hard_repeat, "--epochs", cfg.proposer_epochs,
            "--hard-negative-repeat", cfg.hard_negative_repeat,
            "--train-cap", cfg.proposer_train_cap, "--val-cap", cfg.val_cap,
            "--seed", cfg.seed + 1,
            "--cpu-threads", cfg.train_threads, threads=cfg.train_threads)
    train_landmarker(
        cfg,
        round2,
        round2 / "proposer.onnx",
        round1,
        seed=cfg.seed + 1,
    )

    rounds = []
    for number, directory in enumerate((round1, round2), 1):
        meta = json.loads((directory / "meta.json").read_text())
        proposer_meta_path = directory / "proposer_meta.json"
        proposer_meta = (json.loads(proposer_meta_path.read_text())
                         if proposer_meta_path.exists() else {})
        benchmark = json.loads((directory / "proposed_roi_benchmark.json").read_text())
        rounds.append({
            "round": number,
            "directory": str(directory),
            "proposer_sha256": digest(directory / "proposer.onnx"),
            "landmarker_sha256": digest(directory / "landmarker.onnx"),
            "parameter_accounting": {
                "landmarker": meta.get("params"),
                "proposer": proposer_meta.get("params"),
                "pipeline_total": (
                    meta.get("params", 0) + proposer_meta.get("params", 0)
                    if proposer_meta.get("params") is not None else None),
            },
            "evaluation_columns": meta["evaluation_columns"],
            "acquisition_penalty": {
                split: benchmark[split]["acquisition_penalty_mean_px"]
                for split in ("exploration_holdout", "untouched_referee")},
            "selection_gate": meta["selection_gate"],
        })
    eligible = [r for r in rounds if r["selection_gate"]["passed"]]
    report = {
        "contract": "openpave.two-round-acquisition-alternation.v1",
        "rounds": rounds,
        "selected_round": eligible[-1]["round"] if eligible else None,
        "promotion_rule": "a round is eligible only when its live-replay gate passes",
        "acquisition_policy": {
            "topk_centres": 3,
            "scales_per_centre": 3,
            "maximum_cold_start_landmarker_passes": 10,
            "palm_anchor_bootstrap": True,
            "temporal_cold_start_frames": 5,
        },
        "compute_policy": {
            "seed": cfg.seed,
            "round_seeds": [cfg.seed, cfg.seed + 1],
            "landmarker_sources": cfg.landmarker_sources.split(","),
            "auxiliary_sources": cfg.auxiliary_sources.split(","),
            "train_threads": cfg.train_threads,
            "replay_threads": cfg.replay_threads,
            "recalibration_duty_cycle": cfg.thermal_duty_cycle,
            "round_cooldown_seconds": cfg.round_cooldown_seconds,
        },
    }
    (cfg.out / "alternation_report.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=BASE / "alternation")
    p.add_argument("--initial-proposer", type=Path, default=BASE / "proposer.onnx")
    p.add_argument("--initial-acquisition-model-dir", type=Path,
                   help="optional prior candidate used to trace round-1 live crops")
    p.add_argument("--oracle-mix", type=float, default=0.30)
    p.add_argument("--raw-proposer-mix", type=float, default=0.50)
    p.add_argument(
        "--landmarker-sources",
        default="crude,hagrid_shapes,jester,ipn,hanco",
        help=(
            "training sources; auxiliary sources are excluded from deployment "
            "validation"
        ),
    )
    p.add_argument("--auxiliary-sources", default="hanco")
    p.add_argument("--seed", type=int, default=37)
    p.add_argument("--landmarker-epochs", type=int, default=24)
    p.add_argument("--proposer-epochs", type=int, default=30)
    p.add_argument("--train-cap", type=int, default=4000)
    p.add_argument("--proposer-train-cap", type=int, default=8000)
    p.add_argument("--val-cap", type=int, default=600)
    p.add_argument("--referee-cap", type=int, default=0)
    p.add_argument("--benchmark-cap", type=int, default=300)
    p.add_argument("--hard-cap", type=int, default=4000)
    p.add_argument("--hard-negative-cap", type=int, default=4000)
    p.add_argument("--hard-fraction", type=float, default=0.20)
    p.add_argument("--hard-negative-fraction", type=float, default=0.20)
    p.add_argument("--hard-repeat", type=int, default=3)
    p.add_argument("--hard-negative-repeat", type=int, default=6)
    p.add_argument("--replay-stride", type=int, default=2)
    p.add_argument("--train-threads", type=int, default=8)
    p.add_argument("--replay-threads", type=int, default=4)
    p.add_argument("--thermal-duty-cycle", type=float, default=0.90)
    p.add_argument("--round-cooldown-seconds", type=float, default=30.0)
    p.add_argument("--resume", action="store_true",
                   help="reuse completed round artifacts and continue at the first missing gate")
    return p


if __name__ == "__main__":
    main()
