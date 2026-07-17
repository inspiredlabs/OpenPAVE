#!/usr/bin/env python3
"""Mine proposer/landmarker failures for the next alternation round."""
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


def thermal_pause(started, processed, duty_cycle, interval=100):
    if duty_cycle < 1.0 and processed and processed % interval == 0:
        work = time.perf_counter() - started
        time.sleep(work * (1.0 - duty_cycle) / duty_cycle)
        return time.perf_counter()
    return started


def main(argv=None):
    cfg = parser().parse_args(argv)
    if not 0.0 < cfg.thermal_duty_cycle <= 1.0:
        raise SystemExit("--thermal-duty-cycle must be in (0, 1]")
    if cfg.negative_evidence_multiplier < 1:
        raise SystemExit("--negative-evidence-multiplier must be >= 1")
    runtime = OracleLandmarkerRuntime(model_dir=cfg.model_dir)
    report = {
        "contract": "openpave.acquisition-hard-examples.v2",
        "sources": {},
        "negative_sources": {},
        "scores": {
            "positive": "proposed landmark mean error + normalized ROI centre error",
            "negative": "mean of proposer presence and palm-anchor evidence on no_hand",
        },
    }
    for source in [s for s in cfg.sources.split(",") if s]:
        with np.load(DATASETS / source / "prepared.npz", allow_pickle=True) as data:
            has = np.asarray(data["has_lm"], bool)
            is_val = np.asarray(data["is_val"], bool)
            labels = np.asarray(data["labels"]).astype(str)
            rows = evenly_spaced(np.where(has & (labels != "no_hand") & ~is_val)[0], cfg.cap)
            negative_rows = evenly_spaced(
                np.where((labels == "no_hand") & ~is_val)[0], cfg.negative_cap)
            all_images = np.asarray(data["imgs"])
            images = all_images[rows]
            negative_images = all_images[negative_rows]
            truth = np.asarray(data["landmarks"], np.float32)[rows].reshape(-1, 21, 2)
        scored = []
        throttle_started = time.perf_counter()
        for processed, (row, image, target) in enumerate(
                zip(rows, images, truth), 1):
            try:
                target_roi = oracle_roi(target)
            except ValueError:
                continue
            detector_rois, presence = runtime._detector_rois(image)
            points, _confidence, proposed_roi = runtime.cold_start_hypotheses(
                image, detector_rois)
            landmark_error = float(np.linalg.norm(points - target, axis=1).mean())
            centre_error = float(
                np.linalg.norm(proposed_roi["center"] - target_roi["center"])
                / max(target_roi["size"], 1e-6))
            score = landmark_error + centre_error
            scored.append((score, int(row), landmark_error, centre_error, presence))
            throttle_started = thermal_pause(
                throttle_started, processed, cfg.thermal_duty_cycle)
        scored.sort(reverse=True)
        count = max(1, int(round(len(scored) * cfg.fraction))) if scored else 0
        selected = scored[:count]
        report["sources"][source] = [x[1] for x in selected]
        # Presence is cheap and safely removes obvious background. Run the
        # ten-pass palm-evidence search only on the highest-presence tail.
        negative_prefilter = []
        for row, image in zip(negative_rows, negative_images):
            detector_rois, presence = runtime._detector_rois(image)
            negative_prefilter.append(
                (float(presence), int(row), image, detector_rois))
        negative_prefilter.sort(reverse=True, key=lambda item: item[0])
        negative_count = (max(1, int(round(len(negative_prefilter)
                                           * cfg.negative_fraction)))
                          if negative_prefilter else 0)
        evidence_count = min(len(negative_prefilter), max(
            negative_count, negative_count * cfg.negative_evidence_multiplier))
        negative_scored = []
        throttle_started = time.perf_counter()
        for processed, (presence, row, image, detector_rois) in enumerate(
                negative_prefilter[:evidence_count], 1):
            _points, confidence, _roi = runtime.cold_start_hypotheses(
                image, detector_rois)
            evidence = runtime._candidate_score(confidence)
            score = 0.5 * presence + 0.5 * float(evidence)
            negative_scored.append((score, row, presence, evidence))
            throttle_started = thermal_pause(
                throttle_started, processed, cfg.thermal_duty_cycle)
        negative_scored.sort(reverse=True)
        negative_selected = negative_scored[:negative_count]
        report["negative_sources"][source] = [x[1] for x in negative_selected]
        report.setdefault("statistics", {})[source] = {
            "frames_scored": len(scored), "selected": len(selected),
            "selected_score_mean": float(np.mean([x[0] for x in selected]))
            if selected else None,
            "selected_landmark_error_mean_roi":
                float(np.mean([x[2] for x in selected])) if selected else None,
            "selected_centre_error_over_roi":
                float(np.mean([x[3] for x in selected])) if selected else None,
            "negative_frames_scored": len(negative_scored),
            "negative_frames_presence_prefiltered": len(negative_prefilter),
            "negative_selected": len(negative_selected),
            "negative_selected_presence_mean":
                float(np.mean([x[2] for x in negative_selected]))
                if negative_selected else None,
            "negative_selected_palm_evidence_mean":
                float(np.mean([x[3] for x in negative_selected]))
                if negative_selected else None,
        }
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    cfg.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=OUT)
    p.add_argument("--sources", default="crude,hagrid_shapes,jester,ipn")
    p.add_argument("--cap", type=int, default=4000)
    p.add_argument("--negative-cap", type=int, default=4000)
    p.add_argument("--fraction", type=float, default=0.20)
    p.add_argument("--negative-fraction", type=float, default=0.20)
    p.add_argument("--negative-evidence-multiplier", type=int, default=3)
    p.add_argument("--thermal-duty-cycle", type=float, default=0.90)
    p.add_argument("--out", type=Path, default=OUT / "hard_examples.json")
    return p


if __name__ == "__main__":
    main()
