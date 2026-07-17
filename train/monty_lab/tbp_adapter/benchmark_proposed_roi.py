#!/usr/bin/env python3
"""Proposed-ROI benchmark — the first §9 runtime gate after oracle ROI.

Runs the oracle-ROI landmarker behind the real acquisition path (top-k centre
hypotheses -> bounded scale search -> palm-evidence selection -> one oriented
refinement pass) on the frozen exploration-holdout and untouched-referee splits, and
reports landmark metrics twice — oracle ROI and proposed ROI — so the
difference is the measured acquisition penalty (docs/training-with-monty.md §4).

Also reports acquisition metrics: ROI centre/scale error against the oracle
ROI, top-k centre coverage, detector recall, independent/sequential no_hand
false proposals, search passes, and per-frame CPU latency. Run in the
OpenPAVE arm64 environment.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.monty_lab.tbp_adapter.oracle_roi import oracle_roi
from train.monty_lab.tbp_adapter.oracle_runtime import OracleLandmarkerRuntime
from train.pixel_sensor.train import DATASETS, evenly_spaced

OUT = ROOT / "train" / "runs" / "monty_landmark_alignment" / "oracle_student"


def load(source, split, cap, want_hand=True):
    with np.load(DATASETS / source / "prepared.npz", allow_pickle=True) as d:
        has = np.asarray(d["has_lm"], bool)
        is_val = np.asarray(d["is_val"], bool)
        labels = np.asarray(d["labels"]).astype(str)
        if want_hand:
            mask = has & (labels != "no_hand")
        else:
            mask = labels == "no_hand"
        if split == "val":
            mask &= is_val
        elif split == "train":
            mask &= ~is_val
        rows = evenly_spaced(np.where(mask)[0], cap)
        return (np.asarray(d["imgs"])[rows],
                np.asarray(d["landmarks"], np.float32)[rows].reshape(-1, 21, 2))


def metrics(errors_px):
    flat = errors_px[np.isfinite(errors_px)]
    if not len(flat):
        return {"joints": 0}
    return {"joints": int(len(flat)),
            "mean_px_384": float(flat.mean()),
            "median_px_384": float(np.median(flat)),
            "p95_px_384": float(np.percentile(flat, 95)),
            "pck_5px": float((flat <= 5.0).mean()),
            "pck_10px": float((flat <= 10.0).mean())}


def run_split(runtime, images, truth):
    proposed = np.full((len(images), 21), np.nan)
    oracle = np.full((len(images), 21), np.nan)
    centre_err, scale_err, latency = [], [], []
    topk_hits, hypothesis_counts, landmarker_passes = [], [], []
    missed = 0
    for row, (image, target) in enumerate(zip(images, truth)):
        valid = np.isfinite(target).all(-1)
        try:
            true_roi = oracle_roi(target)
        except ValueError:
            continue

        # Oracle-ROI reference on the identical frame subset.
        points, _conf = runtime._landmark_pass(image, true_roi)
        err = np.linalg.norm(points - target, axis=-1) * 384.0
        oracle[row, valid] = err[valid]

        # Proposed ROI: cold start from the detector, every frame independent.
        runtime.reset()
        t0 = time.perf_counter()
        detector_rois, presence = runtime._detector_rois(image)
        if presence < runtime.presence_gate:
            missed += 1
            latency.append((time.perf_counter() - t0) * 1000)
            continue
        points, conf, selected_roi = runtime.cold_start_hypotheses(
            image, detector_rois)
        latency.append((time.perf_counter() - t0) * 1000)
        err = np.linalg.norm(points - target, axis=-1) * 384.0
        proposed[row, valid] = err[valid]
        centre_err.append(
            np.linalg.norm(selected_roi["center"] - true_roi["center"]) * 384.0)
        scale_err.append(selected_roi["size"] / true_roi["size"])
        hit = False
        for candidate in detector_rois:
            offset = true_roi["center"] - candidate["center"]
            local = np.asarray([np.dot(offset, candidate["x_axis"]),
                                np.dot(offset, candidate["y_axis"])])
            if np.max(np.abs(local)) <= 0.5 * candidate["size"]:
                hit = True
                break
        topk_hits.append(hit)
        hypothesis_counts.append(len(detector_rois))
        landmarker_passes.append(runtime.last_acquisition["landmarker_passes"])

    return {
        "frames": int(len(images)),
        "detector_miss_rate": float(missed / max(len(images), 1)),
        "oracle_roi": metrics(oracle),
        "proposed_roi": metrics(proposed),
        "acquisition_penalty_mean_px": float(
            np.nanmean(proposed) - np.nanmean(oracle)),
        "roi_centre_error_px_384": {
            "mean": float(np.mean(centre_err)), "p95": float(np.percentile(centre_err, 95))},
        "roi_scale_ratio": {
            "median": float(np.median(scale_err)), "p95": float(np.percentile(scale_err, 95))},
        "topk_oracle_centre_coverage": float(np.mean(topk_hits)),
        "roi_hypotheses_per_frame": float(np.mean(hypothesis_counts)),
        "landmarker_passes_per_cold_start": float(np.mean(landmarker_passes)),
        "latency_ms": {"median": float(np.median(latency)),
                       "p95": float(np.percentile(latency, 95))},
    }


def false_proposals(runtime, images):
    independent = 0
    for image in images:
        runtime.reset()
        lm, _presence, _quality = runtime.step(image)
        if lm is not None:
            independent += 1
    runtime.reset()
    sequential = 0
    palm_locks = 0
    for image in images:
        lm, _presence, _quality = runtime.step(image)
        if lm is not None:
            sequential += 1
            palm_locks += int(runtime.last_lock_reason == "palm_anchors")
    return {"frames": int(len(images)),
            "false_proposal_rate": float(independent / max(len(images), 1)),
            "independent_frame_false_proposal_rate":
                float(independent / max(len(images), 1)),
            "sequential_false_proposal_rate":
                float(sequential / max(len(images), 1)),
            "sequential_palm_anchor_locks": int(palm_locks)}


def main(args=None):
    cfg = parser().parse_args(args)
    model_dir = Path(cfg.model_dir)
    runtime = OracleLandmarkerRuntime(model_dir=model_dir)
    report = {"contract": "openpave.proposed-roi-benchmark.v3",
              "acquisition": "frozen deployment proposer -> top-k centre/scale "
                             "search -> palm-scored selection -> one oriented "
                             "refinement; landmark splits are independent, while "
                             "no_hand reports independent and sequential modes"}

    holdout_images, holdout_truth = [], []
    for source in cfg.sources.split(","):
        images, truth = load(source, "val", cfg.val_cap)
        holdout_images.append(images)
        holdout_truth.append(truth)
    report["exploration_holdout"] = run_split(
        runtime, np.concatenate(holdout_images), np.concatenate(holdout_truth))

    referee_images, referee_truth = load(cfg.referee, "all", cfg.referee_cap)
    report["untouched_referee"] = run_split(runtime, referee_images, referee_truth)

    negatives, _ = load("crude", "all", cfg.negative_cap, want_hand=False)
    report["no_hand_frames"] = false_proposals(runtime, negatives)

    model_dir.mkdir(parents=True, exist_ok=True)
    out = model_dir / "proposed_roi_benchmark.json"
    out.write_text(
        json.dumps(report, indent=2) + "\n")
    if cfg.update_meta:
        meta_path = model_dir / "meta.json"
        meta = json.loads(meta_path.read_text())
        columns = meta.setdefault("evaluation_columns", {})
        deployment = columns.setdefault("proposed_roi_deployment_truth", {})
        deployment["frozen_frame_benchmark"] = report
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", default="crude,hagrid_shapes,jester,ipn")
    p.add_argument("--referee", default="yolo26")
    p.add_argument("--val-cap", type=int, default=300)
    p.add_argument("--referee-cap", type=int, default=0)
    p.add_argument("--negative-cap", type=int, default=400)
    p.add_argument("--model-dir", type=Path, default=OUT)
    p.add_argument("--update-meta", action="store_true")
    return p


if __name__ == "__main__":
    main()
