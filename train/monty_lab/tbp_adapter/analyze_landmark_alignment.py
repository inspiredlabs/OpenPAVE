"""Analyze paired landmark observations emitted through monty_handlers."""
from __future__ import print_function

import argparse
import json
from pathlib import Path

import numpy as np

from tbp.monty.frameworks.utils.logging_utils import load_stats


def metrics(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean_px": float(np.mean(values)),
        "median_px": float(np.median(values)),
        "p95_px": float(np.percentile(values, 95)),
        "pck_5px": float(np.mean(values <= 5.0)),
        "pck_10px": float(np.mean(values <= 10.0)),
    }


def analyze(exp_path):
    _train, _eval, detailed, _models = load_stats(
        exp_path, load_train=False, load_eval=False,
        load_detailed=True, load_models=False)
    episodes = [detailed[key] for key in sorted(detailed, key=int)]
    compact_path = exp_path / "comparison.npz"
    if compact_path.exists():
        with np.load(compact_path, allow_pickle=True) as stored:
            compact = {key: stored[key] for key in stored.files}
        roles = sorted(set(compact["role"].tolist()))
    else:
        compact = None
        roles = sorted(set(ep["role"] for ep in episodes))
    report = {"contract": "openpave.monty-handler-landmark-analysis.v1", "roles": {}}
    for role in roles:
        if compact is not None:
            keep = compact["role"] == role
            error = np.asarray(compact["pixel_error_384"][keep], dtype=np.float64)
            accepted = np.asarray(compact["accepted"][keep], dtype=bool)
            correct = np.asarray(compact["correct"][keep], dtype=bool)
            count = int(keep.sum())
        else:
            selected = [ep for ep in episodes if ep["role"] == role]
            error = np.asarray([
                ep["landmark_comparison"]["pixel_error_384"] for ep in selected
            ], dtype=np.float64)
            accepted = np.asarray([
                ep["landmark_comparison"]["accepted"] for ep in selected
            ], dtype=bool)
            correct = np.asarray([ep["correct"] for ep in selected], dtype=bool)
            count = len(selected)
        report["roles"][role] = {
            "episodes": count,
            "recognition_accuracy": float(np.mean(correct)),
            "all_joints": metrics(error.reshape(-1)),
            "accepted_joint_fraction": float(accepted.mean()),
            "per_joint": {str(joint): metrics(error[:, joint]) for joint in range(21)},
        }
    report["detailed_logged_episodes"] = len(episodes)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("exp_path", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    report = analyze(args.exp_path)
    output = json.dumps(report, indent=2) + "\n"
    destination = args.out or args.exp_path / "analysis.json"
    destination.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
