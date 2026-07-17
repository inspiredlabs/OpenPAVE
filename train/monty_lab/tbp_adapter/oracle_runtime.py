"""Webcam runtime for the oracle-ROI landmarker (docs/training-with-monty.md §7).

Implements the runtime state machine around the crop-based student:

    GLOBAL_SEARCH -> CANDIDATE_ROI -> COLD_START -> TRACKING

- GLOBAL_SEARCH: the landmark_tower hand detector proposes an axis-aligned
  centre/size (distant-like acquisition; §4 candidate 1).
- COLD_START: the landmarker runs on that unoriented crop, then once more on
  the oriented ROI derived from its own first-pass constellation (the local
  re-inspection recurrence).
- TRACKING: the next frame's oriented ROI comes from the previous accepted
  constellation (§4 candidate 3). Sustained low evidence re-enters
  GLOBAL_SEARCH rather than emitting stale geometry.

The class mirrors ``train.landmark_tower.LandmarkerRuntime.step``'s
``(lm42, presence, quality) | (None, p, q)`` contract so the PyQt worker can
swap it in without touching the Monty evidence stage. It emits no commands.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

try:
    from .oracle_roi import crop as roi_crop, oracle_roi, project_to_source
except ImportError:  # direct script execution
    from oracle_roi import crop as roi_crop, oracle_roi, project_to_source

ROOT = Path(__file__).resolve().parents[3]
ORACLE_DIR = ROOT / "train" / "runs" / "monty_landmark_alignment" / "oracle_student"
DETECTOR = ROOT / "train" / "runs" / "landmark_tower" / "detector.onnx"
DETECTOR_INPUT = 128
CROP = 96
PALM_ANCHORS = np.asarray([0, 5, 9, 13, 17], dtype=np.int64)
PALM_MCPS = np.asarray([5, 9, 13, 17], dtype=np.int64)
MIN_PARTIAL_EMISSION_JOINTS = 6


class OracleLandmarkerRuntime:
    """Two-stage acquisition + oracle-ROI landmarker with temporal tracking."""

    def __init__(self, presence_gate=0.5, quality_gate=None,
                 min_joint_fraction=None, max_track_misses=5,
                 model_dir=ORACLE_DIR, proposer_path=None,
                 max_roi_hypotheses=3, cold_history_frames=5,
                 cold_min_history=3):
        import onnxruntime as ort

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = max(
            1, int(os.environ.get("PAVE_ORT_THREADS", "4")))
        session_options.inter_op_num_threads = 1

        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "meta.json").read_text())
        selected_gates = meta.get("runtime_gates", {})
        # Prefer the oriented ROI proposer trained against teacher-derived
        # oracle ROIs; fall back to the legacy landmark_tower detector.
        proposer = Path(proposer_path) if proposer_path else model_dir / "proposer.onnx"
        if proposer.exists():
            self.proposer = ort.InferenceSession(
                str(proposer), sess_options=session_options,
                providers=["CPUExecutionProvider"])
            self.detector = None
        else:
            self.proposer = None
            self.detector = ort.InferenceSession(
                str(DETECTOR), sess_options=session_options,
                providers=["CPUExecutionProvider"])
        self.landmarker = ort.InferenceSession(
            str(model_dir / "landmarker.onnx"), sess_options=session_options,
            providers=["CPUExecutionProvider"])
        self.presence_gate = float(presence_gate)
        # per-joint acceptance uses the trained threshold, P(err < 5% ROI)
        self.joint_gate = float(quality_gate if quality_gate is not None
                                else selected_gates.get(
                                    "joint_confidence_threshold",
                                    meta["confidence_threshold"]))
        self.min_joint_fraction = float(
            min_joint_fraction if min_joint_fraction is not None
            else selected_gates.get("minimum_joint_fraction", 0.4))
        self.max_track_misses = int(max_track_misses)
        self.max_roi_hypotheses = max(1, int(max_roi_hypotheses))
        self.cold_history_frames = max(1, int(cold_history_frames))
        self.cold_min_history = max(1, int(cold_min_history))
        self.reset()

    def reset(self):
        self.state = "GLOBAL_SEARCH"
        self._tracked = None       # last accepted (21, 2) source-normalized points
        self._misses = 0
        self._cold_history = []
        self.last_lock_reason = None
        self.last_acquisition = None
        # Public accountability mask: only these joints may reach Monty/the GUI.
        self.last_joint_mask = None

    def _landmark_pass(self, frame, roi):
        patch = roi_crop(frame, roi, CROP).astype(np.float32) / 255.0 - 0.5
        uv, conf = self.landmarker.run(
            ["landmarks_uv", "confidence"],
            {"crops": np.transpose(patch[None], (0, 3, 1, 2))})
        return project_to_source(uv[0], roi), conf[0]

    # The landmark_tower detector was trained with side ~ 1.9x the hand bbox
    # while oracle ROIs are 1.25x the oriented extent; the median mapping
    # measured on the training split (never the frozen benchmarks) is 0.52.
    DETECTOR_SIZE_CALIBRATION = 0.52
    # The detector's centre/size estimates are noisy (p95 centre offset ~0.3
    # normalized), so cold start probes several scales and keeps the proposal
    # with the strongest landmark evidence — the distant-like saccade.
    COLD_START_SCALES = (0.75, 1.0, 1.4)

    @staticmethod
    def _heatmap_centres(probability, count, suppression_cells=2):
        work = np.asarray(probability, np.float64).copy()
        height, width = work.shape
        centres = []
        for _ in range(count):
            flat = int(np.argmax(work))
            y, x = np.unravel_index(flat, work.shape)
            score = float(work[y, x])
            if not np.isfinite(score) or score < 0:
                break
            centres.append((np.asarray([(x + 0.5) / width,
                                        (y + 0.5) / height]), score))
            y0 = max(0, y - suppression_cells)
            y1 = min(height, y + suppression_cells + 1)
            x0 = max(0, x - suppression_cells)
            x1 = min(width, x + suppression_cells + 1)
            work[y0:y1, x0:x1] = -np.inf
        return centres

    def _detector_rois(self, frame):
        import cv2
        resized = cv2.resize(frame, (DETECTOR_INPUT, DETECTOR_INPUT))
        x = (resized.astype(np.float32) / 255.0 - 0.5).transpose(2, 0, 1)[None]
        if self.proposer is not None:
            outputs = self.proposer.run(None, {"image": x})
            centre, size, axis, presence = outputs[:4]
            y_axis = axis[0].astype(np.float64)
            norm = float(np.linalg.norm(y_axis))
            y_axis = y_axis / norm if norm > 1e-6 else np.asarray([0.0, 1.0])
            centres = [(centre[0].astype(np.float64), 1.0)]
            if len(outputs) >= 5:
                centres = self._heatmap_centres(
                    outputs[4][0], self.max_roi_hypotheses)
            side = float(np.clip(np.asarray(size).reshape(-1)[0], 0.05, 1.5))
            rois = [{"center": candidate.astype(np.float64),
                     "x_axis": np.asarray([y_axis[1], -y_axis[0]]),
                     "y_axis": y_axis,
                     "size": side,
                     "proposal_score": float(score)}
                    for candidate, score in centres]
            return rois, float(np.asarray(presence).reshape(-1)[0])
        centre, size, presence = self.detector.run(None, {"image": x})
        side = float(np.clip(size[0, 0], 0.08, 1.2)) * self.DETECTOR_SIZE_CALIBRATION
        roi = {"center": centre[0].astype(np.float64),
               "x_axis": np.asarray([1.0, 0.0]),
               "y_axis": np.asarray([0.0, 1.0]),
               "size": side, "proposal_score": 1.0}
        return [roi], float(presence[0, 0])

    def _detector_roi(self, frame):
        """Backward-compatible best raw proposal used by older diagnostics."""
        rois, presence = self._detector_rois(frame)
        return rois[0], presence

    def _refine_roi(self, points):
        try:
            return oracle_roi(points[:, :2])
        except ValueError:
            return None

    @staticmethod
    def _palm_tracking_roi(points):
        """Extrapolate a full-hand crop from reliable wrist/MCP geometry."""
        points = np.asarray(points, np.float64)
        wrist = points[0]
        mcp_centre = points[PALM_MCPS].mean(axis=0)
        toward_wrist = wrist - mcp_centre
        length = float(np.linalg.norm(toward_wrist))
        if length < 1e-6:
            return None
        y_axis = toward_wrist / length
        x_axis = np.asarray([y_axis[1], -y_axis[0]])
        spread = float(np.ptp(points[PALM_MCPS] @ x_axis))
        # Shift toward the fingers and cover roughly wrist-to-tip length.
        centre = mcp_centre - 0.40 * length * y_axis
        size = float(np.clip(2.7 * max(length, spread), 0.08, 1.5))
        return {"center": centre, "x_axis": x_axis, "y_axis": y_axis,
                "size": size, "proposal_score": 1.0}

    @staticmethod
    def _candidate_score(confidence):
        confidence = np.asarray(confidence, np.float64)
        return float(0.75 * confidence[PALM_ANCHORS].mean()
                     + 0.25 * confidence.mean())

    def cold_start_hypotheses(self, frame, detector_rois):
        """Probe top-k centres and scales, then refine only the best result."""
        best = None
        passes = 0
        for hypothesis, det_roi in enumerate(detector_rois):
            for scale in self.COLD_START_SCALES:
                candidate = dict(det_roi, size=det_roi["size"] * scale)
                points, conf = self._landmark_pass(frame, candidate)
                passes += 1
                score = self._candidate_score(conf)
                if best is None or score > best[0]:
                    best = (score, candidate, points, conf, hypothesis, scale)
        score, selected_roi, points, conf, hypothesis, scale = best
        refined = self._refine_roi(points)
        if refined is not None:
            re_points, re_conf = self._landmark_pass(frame, refined)
            passes += 1
            if self._candidate_score(re_conf) >= score:
                selected_roi, points, conf = refined, re_points, re_conf
                score = self._candidate_score(conf)
        self.last_acquisition = {
            "roi_hypotheses": len(detector_rois),
            "landmarker_passes": passes,
            "selected_hypothesis": int(hypothesis),
            "selected_scale": float(scale),
            "evidence_score": float(score),
        }
        return points, conf, selected_roi

    def cold_start(self, frame, det_roi):
        """Backward-compatible single-centre cold start."""
        points, conf, _roi = self.cold_start_hypotheses(frame, [det_roi])
        return points, conf

    def trace_acquisition_roi(self, frame):
        """Return the crop selected by the frozen live cold-start path.

        This is the E-step used by the next alternation round: train against
        what multi-scale selection and one re-inspection actually delivered,
        not merely against the proposer's raw box.
        """
        detector_rois, presence = self._detector_rois(frame)
        _points, conf, selected_roi = self.cold_start_hypotheses(
            frame, detector_rois)
        return selected_roi, float(presence), float(conf.mean())

    @staticmethod
    def _palm_signature(points):
        points = np.asarray(points, np.float64)
        return np.concatenate((points[0], points[PALM_MCPS].mean(axis=0)))

    @staticmethod
    def _palm_bootstrap_ok(points, accepted):
        """Require a coherent wrist plus at least three accepted MCP anchors."""
        if not accepted[0] or int(accepted[PALM_MCPS].sum()) < 3:
            return False
        mcps = points[PALM_MCPS[accepted[PALM_MCPS]]]
        wrist = points[0]
        centre = mcps.mean(axis=0)
        longitudinal = float(np.linalg.norm(centre - wrist))
        spread = float(np.linalg.norm(mcps[:, None] - mcps[None, :], axis=-1).max())
        if not (0.025 <= longitudinal <= 0.8 and spread >= 0.02):
            return False
        ratio = spread / max(longitudinal, 1e-6)
        return bool(0.20 <= ratio <= 3.0 and np.isfinite(points[PALM_ANCHORS]).all())

    @staticmethod
    def _partial_emission_ok(points, accepted):
        return bool(OracleLandmarkerRuntime._palm_bootstrap_ok(points, accepted)
                    and int(accepted.sum()) >= MIN_PARTIAL_EMISSION_JOINTS)

    def _accumulate_cold_start(self, points, confidence):
        signature = self._palm_signature(points)
        if self._cold_history:
            previous = self._cold_history[-1][2]
            if float(np.linalg.norm(signature - previous)) > 0.30:
                self._cold_history.clear()
        self._cold_history.append((points.copy(), confidence.copy(), signature))
        self._cold_history = self._cold_history[-self.cold_history_frames:]
        if len(self._cold_history) < self.cold_min_history:
            return points, confidence, len(self._cold_history)
        point_stack = np.stack([entry[0] for entry in self._cold_history])
        confidence_stack = np.clip(
            np.stack([entry[1] for entry in self._cold_history]), 0.0, 1.0)
        weights = np.maximum(confidence_stack, 1e-4)[..., None]
        combined_points = (point_stack * weights).sum(axis=0) / weights.sum(axis=0)
        # Frames are correlated, so a probability union would be dangerously
        # overconfident. Repetition receives only a bounded +2.5 points per
        # extra associated frame; the topology gate below must still pass.
        combined_confidence = np.clip(
            confidence_stack.mean(axis=0)
            + 0.025 * (len(self._cold_history) - 1), 0.0, 1.0)
        return combined_points, combined_confidence, len(self._cold_history)

    def step(self, rgb, apply_gate=True):
        """rgb -> (lm42 float32 in [0,1], presence, quality) or (None, p, q)."""
        import cv2

        # Resize (not crop) to square, exactly like LandmarkerRuntime: the
        # returned coordinates stay normalized to the full input frame so the
        # GUI overlay and downstream geometry need no aspect bookkeeping.
        if rgb.shape[0] != rgb.shape[1]:
            frame = cv2.resize(rgb, (DETECTOR_INPUT, DETECTOR_INPUT))
        else:
            frame = rgb

        presence = 1.0
        if self.state == "TRACKING" and self._tracked is not None:
            tracked_roi = (self._palm_tracking_roi(self._tracked)
                           if self.last_lock_reason == "palm_anchors"
                           else self._refine_roi(self._tracked))
        else:
            tracked_roi = None
        if tracked_roi is not None:
            points, conf = self._landmark_pass(frame, tracked_roi)
            refined = self._refine_roi(points)
            if refined is not None:
                re_points, re_conf = self._landmark_pass(frame, refined)
                if re_conf.mean() >= conf.mean():
                    points, conf = re_points, re_conf
        else:
            detector_rois, presence = self._detector_rois(frame)
            if presence < self.presence_gate:
                self.reset()
                return None, presence, 0.0
            points, conf, _selected_roi = self.cold_start_hypotheses(
                frame, detector_rois)

        accepted = conf >= self.joint_gate
        quality = float(conf.mean())
        fraction_ok = float(accepted.mean()) >= self.min_joint_fraction
        palm_ok = self._palm_bootstrap_ok(points, accepted)
        if tracked_roi is None and not (fraction_ok or palm_ok):
            points, conf, history_frames = self._accumulate_cold_start(points, conf)
            accepted = conf >= self.joint_gate
            quality = float(conf.mean())
            # Temporal accumulation may bootstrap only an anatomical palm,
            # never the generic fraction gate; correlated weak joints must not
            # become a complete hand merely by recurring.
            fraction_ok = False
            palm_ok = self._palm_bootstrap_ok(points, accepted)
        else:
            history_frames = len(self._cold_history)
        lock_ok = fraction_ok or palm_ok
        emit_ok = fraction_ok or self._partial_emission_ok(points, accepted)
        if not lock_ok:
            self._misses += 1
            if self._misses > self.max_track_misses:
                self.reset()
            if apply_gate:
                self.last_joint_mask = None
                return None, presence, quality
        else:
            self._misses = 0
            self.state = "TRACKING"
            self._tracked = points.copy()
            self.last_lock_reason = "joint_fraction" if fraction_ok else "palm_anchors"
            if self.last_acquisition is not None:
                self.last_acquisition["history_frames"] = history_frames
                self.last_acquisition["lock_reason"] = self.last_lock_reason
                self.last_acquisition["emitted_on_lock"] = bool(emit_ok)
            self._cold_history.clear()

        # A coherent palm is enough to improve the next frame's ROI, but not
        # enough evidence to expose an intent-bearing constellation. Tracking
        # may therefore lock internally while this frame remains an abstention.
        if apply_gate and not emit_ok:
            self.last_joint_mask = None
            return None, presence, quality

        # Benchmarks request raw complete predictions. Deployment emits only
        # accepted joints; rejected joints are NaN, never plausible-looking
        # fabricated coordinates. Monty consumes last_joint_mask per joint.
        if apply_gate:
            emitted = np.clip(points, 0.0, 1.0).astype(np.float32)
            emitted[~accepted] = np.nan
            self.last_joint_mask = accepted.copy()
        else:
            emitted = np.clip(points, 0.0, 1.0).astype(np.float32)
            self.last_joint_mask = np.ones(21, dtype=bool)
        lm = emitted.reshape(-1)
        return lm, presence, quality
