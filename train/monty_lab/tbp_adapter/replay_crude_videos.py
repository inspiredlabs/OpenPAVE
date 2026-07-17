#!/usr/bin/env python3
"""Live-replay promotion gate for the landmark acquisition pipeline.

This drives the exact GUI worker over stored crude videos and compares a
candidate with the incumbent 71k pipeline. It measures what deployment needs:
acquisition, wrong intent, and time to first correct lock. With ``--update-meta``
the result becomes the proposed-ROI/live column and may veto promotion even
when the oracle-ROI column improved. No robot commands are emitted.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import types
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_VIDEOS = ("stop.mp4", "like.mp4", "fist.mp4", "right-to-left-muted.mp4")
EXPECTED = {
    "stop.mp4": {"STOP"},
    "like.mp4": {"TROT"},
    "fist.mp4": {"HOME"},
    "right-to-left-muted.mp4": {"LEFT", "RIGHT"},
}


def replay(worker, path: Path, stride: int):
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    expected = EXPECTED.get(path.name, set())
    labels, intents = Counter(), Counter()
    lock_reasons = Counter()
    acquisition_passes = []
    frames = acquired = wrong = correct = 0
    landmark_latency_ms = []
    first_lock_source_frame = None
    source_index = 0
    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        source_index += 1
        if source_index % stride:
            continue
        frames += 1
        runtime = getattr(worker, "_landmark_runtime", None)
        state_before = getattr(runtime, "state", None)
        dets, intent, _timing = worker._detect(frame_bgr)
        if isinstance(_timing, dict) and "lm_ms" in _timing:
            landmark_latency_ms.append(float(_timing["lm_ms"]))
        state_after = getattr(runtime, "state", None)
        if state_before != "TRACKING" and state_after == "TRACKING":
            lock_reasons[getattr(runtime, "last_lock_reason", None) or "unknown"] += 1
        acquisition = getattr(runtime, "last_acquisition", None)
        if state_before != "TRACKING" and acquisition:
            acquisition_passes.append(int(acquisition.get("landmarker_passes", 0)))
        acquired += int(bool(dets))
        label = dets[0].label.rsplit(" ", 1)[0] if dets else "no_hand"
        labels[label] += 1
        intents[intent or "-"] += 1
        if intent:
            if intent in expected:
                correct += 1
                if first_lock_source_frame is None:
                    first_lock_source_frame = source_index
            else:
                wrong += 1
    capture.release()
    return {
        "frames_scored": frames,
        "expected_intents": sorted(expected),
        "accepted_frames": acquired,
        "acquisition_rate": acquired / max(frames, 1),
        "correct_intent_frames": correct,
        "wrong_intent_frames": wrong,
        "wrong_intent_rate": wrong / max(frames, 1),
        "time_to_first_lock_s": (
            first_lock_source_frame / fps
            if first_lock_source_frame is not None and fps > 0 else None),
        "labels": dict(labels.most_common()),
        "intents": dict(intents.most_common()),
        "lock_reasons": dict(lock_reasons),
        "cold_start_attempts": len(acquisition_passes),
        "mean_landmarker_passes_per_cold_start": (
            float(statistics.mean(acquisition_passes)) if acquisition_passes else None),
        "landmark_latency_ms": ({
            "median": float(statistics.median(landmark_latency_ms)),
            "p95": float(np.percentile(landmark_latency_ms, 95)),
        } if landmark_latency_ms else None),
    }


def summarize(videos):
    valid = [v for v in videos.values() if isinstance(v, dict)]
    frames = sum(v["frames_scored"] for v in valid)
    accepted = sum(v["accepted_frames"] for v in valid)
    wrong = sum(v["wrong_intent_frames"] for v in valid)
    locks = [v["time_to_first_lock_s"] for v in valid
             if v["time_to_first_lock_s"] is not None]
    latency_p95 = [v["landmark_latency_ms"]["p95"] for v in valid
                   if v.get("landmark_latency_ms")]
    return {
        "videos_present": len(valid),
        "frames_scored": frames,
        "acquisition_rate": accepted / max(frames, 1),
        "wrong_intent_rate": wrong / max(frames, 1),
        "median_time_to_first_lock_s": statistics.median(locks) if locks else None,
        "videos_locked": len(locks),
        "landmark_latency_p95_ms": max(latency_p95) if latency_p95 else None,
    }


def install_headless_qt():
    # Offline replay calls the exact worker's synchronous _detect method and
    # never starts a Qt thread. Avoid importing a platform-sensitive Qt binary
    # solely for QThread/pyqtSignal declarations in headless training jobs.
    if "PyQt6.QtCore" not in sys.modules:
        qt_package = types.ModuleType("PyQt6")
        qt_core = types.ModuleType("PyQt6.QtCore")

        class _Signal:
            def connect(self, *_args, **_kwargs):
                pass

            def emit(self, *_args, **_kwargs):
                pass

        class _QThread:
            def __init__(self, *_args, **_kwargs):
                pass

            def isInterruptionRequested(self):
                return False

        qt_core.QThread = _QThread
        qt_core.pyqtSignal = lambda *_args, **_kwargs: _Signal()
        qt_package.QtCore = qt_core
        sys.modules["PyQt6"] = qt_package
        sys.modules["PyQt6.QtCore"] = qt_core


def make_worker(model: Path):
    install_headless_qt()
    from pave_ui import perception

    worker = perception.LandmarkerMontyWorker(str(model))
    worker._ensure_model()
    return worker


def evaluate_worker(worker, model: Path, videos, stride, gates=None,
                    thermal_duty_cycle=1.0):
    if gates and hasattr(worker._landmark_runtime, "joint_gate"):
        worker._landmark_runtime.joint_gate = gates[0]
        worker._landmark_runtime.min_joint_fraction = gates[1]
    results = {}
    for name in videos:
        path = ROOT / "train" / "crude" / name
        if not path.exists():
            results[name] = "missing"
            continue
        reset = getattr(worker._landmark_runtime, "reset", None)
        if reset is not None:
            reset()
        started = time.perf_counter()
        results[name] = replay(worker, path, stride)
        work_seconds = time.perf_counter() - started
        # A 0.90 duty cycle inserts one second idle for each nine seconds of
        # inference. Rest between videos breaks the sustained calibration load
        # without changing frames, thresholds, rankings, or metrics.
        if thermal_duty_cycle < 1.0:
            time.sleep(work_seconds * (1.0 - thermal_duty_cycle)
                       / thermal_duty_cycle)
    return {"model": str(model), "gates": ({
                "joint_confidence_threshold": gates[0],
                "minimum_joint_fraction": gates[1]} if gates else None),
            "summary": summarize(results), "videos": results}


def live_failures(candidate, incumbent):
    c, b = candidate["summary"], incumbent["summary"]
    failures = []
    if c["acquisition_rate"] + 1e-12 < b["acquisition_rate"]:
        failures.append("candidate acquisition rate is below incumbent")
    if c["wrong_intent_rate"] > b["wrong_intent_rate"] + 1e-12:
        failures.append("candidate wrong-intent rate is above incumbent")
    if c["videos_locked"] < b["videos_locked"]:
        failures.append("candidate locks on fewer videos than incumbent")
    ct, bt = c["median_time_to_first_lock_s"], b["median_time_to_first_lock_s"]
    if bt is not None and (ct is None or ct > bt + 1e-12):
        failures.append("candidate median time-to-first-lock is slower than incumbent")
    if (c.get("landmark_latency_p95_ms") is not None
            and c["landmark_latency_p95_ms"] > 50.0):
        failures.append("candidate landmark p95 latency exceeds 50 ms")
    return failures


def parse_grid(value):
    return [float(x) for x in value.split(",") if x.strip()]


def main(argv=None):
    cfg = parser().parse_args(argv)
    if not 0.0 < cfg.thermal_duty_cycle <= 1.0:
        raise SystemExit("--thermal-duty-cycle must be in (0, 1]")
    os.environ["PAVE_ORT_THREADS"] = str(cfg.ort_threads)
    cv2.setNumThreads(cfg.opencv_threads)
    install_headless_qt()
    from pave_ui import perception

    candidate_path = Path(cfg.model or
                          perception._ORACLE_STUDENT_DIR / "landmarker.onnx")
    incumbent_path = Path(cfg.incumbent_model or
                          perception._LANDMARK_RUN_DIR / "model.onnx")
    incumbent_worker = make_worker(incumbent_path)
    incumbent = evaluate_worker(
        incumbent_worker, incumbent_path, cfg.videos, cfg.stride,
        thermal_duty_cycle=cfg.thermal_duty_cycle)

    model_meta = candidate_path.parent / "meta.json"
    meta = json.loads(model_meta.read_text())
    calibrated = float(meta.get("confidence_threshold", 0.25))
    gate_pairs = [(calibrated, 0.4)]
    if cfg.calibrate:
        thresholds = sorted(set([calibrated] + parse_grid(cfg.joint_thresholds)))
        fractions = sorted(set(parse_grid(cfg.minimum_joint_fractions)))
        gate_pairs = [(t, f) for t in thresholds for f in fractions]

    candidate_worker = make_worker(candidate_path)
    trials = [evaluate_worker(candidate_worker, candidate_path,
                              cfg.videos, cfg.stride, pair,
                              cfg.thermal_duty_cycle)
              for pair in gate_pairs]
    # Live selection: prefer any configuration that clears every incumbent
    # rule, then maximize acquisition before wrong-intent and lock speed.
    # Oracle metrics are deliberately not consulted.
    def rank(trial):
        s = trial["summary"]
        lock = s["median_time_to_first_lock_s"]
        return (len(live_failures(trial, incumbent)), -s["acquisition_rate"],
                s["wrong_intent_rate"],
                -s["videos_locked"], lock if lock is not None else math.inf)

    candidate = min(trials, key=rank)
    failures = live_failures(candidate, incumbent)
    report = {
        "contract": "openpave.live-replay-selection-gate.v2",
        "selection_distribution": "stored crude videos through exact GUI worker",
        "compute_policy": {
            "ort_threads": cfg.ort_threads,
            "opencv_threads": cfg.opencv_threads,
            "thermal_duty_cycle": cfg.thermal_duty_cycle,
            "idle_fraction": 1.0 - cfg.thermal_duty_cycle,
        },
        "candidate": candidate,
        "incumbent": incumbent,
        "calibration_trials": [{"gates": t["gates"], "summary": t["summary"]}
                               for t in trials],
        "gate": {
            "passed": not failures,
            "failures": failures,
            "rules": [
                "acquisition_rate >= incumbent",
                "wrong_intent_rate <= incumbent",
                "videos_locked >= incumbent",
                "median_time_to_first_lock <= incumbent when incumbent locks",
                "landmark p95 latency <= 50 ms",
            ],
        },
    }
    out = Path(cfg.out) if cfg.out else candidate_path.parent / "live_replay.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    if cfg.update_meta:
        columns = meta.setdefault("evaluation_columns", {})
        live = columns.setdefault("proposed_roi_deployment_truth", {})
        live["live_replay"] = report
        meta["runtime_gates"] = candidate["gates"]
        meta["selection_gate"] = report["gate"]
        meta["live_replay_eligible"] = bool(report["gate"]["passed"])
        if failures:
            meta["runtime_promotion"] = False
            meta["runtime_blocker"] = "; ".join(failures)
        else:
            # Passing this selection gate makes the candidate eligible; it
            # does not silently waive the separate recovery/latency/safety
            # gates documented for final runtime promotion.
            meta["runtime_promotion"] = bool(meta.get("runtime_promotion", False))
            meta["runtime_blocker"] = "live replay passed; remaining promotion gates pending"
        model_meta.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if not cfg.fail_on_reject or not failures else 2


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--videos", nargs="*", default=list(DEFAULT_VIDEOS))
    p.add_argument("--model", default=None)
    p.add_argument("--incumbent-model", default=None)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--joint-thresholds", default="0.10,0.20,0.30,0.40,0.50")
    p.add_argument("--minimum-joint-fractions", default="0.20,0.30,0.40")
    p.add_argument("--out", default=None)
    p.add_argument("--update-meta", action="store_true")
    p.add_argument("--fail-on-reject", action="store_true")
    p.add_argument("--ort-threads", type=int, default=4)
    p.add_argument("--opencv-threads", type=int, default=2)
    p.add_argument("--thermal-duty-cycle", type=float, default=0.90,
                   help="fraction of wall time spent working; 0.90 adds 10%% idle")
    return p


if __name__ == "__main__":
    raise SystemExit(main())
