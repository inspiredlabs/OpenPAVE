import numpy as np

from train.monty_lab.tbp_adapter.oracle_runtime import OracleLandmarkerRuntime
from train.monty_lab.tbp_adapter.train_roi_proposer import heatmap_peaks


def palm_points():
    points = np.full((21, 2), 0.5, dtype=np.float64)
    points[0] = [0.50, 0.82]
    points[5] = [0.30, 0.52]
    points[9] = [0.44, 0.40]
    points[13] = [0.59, 0.44]
    points[17] = [0.71, 0.54]
    return points


def test_heatmap_peaks_are_spatially_separated():
    heatmap = np.zeros((16, 16), dtype=np.float32)
    heatmap[2, 2] = 1.0
    heatmap[3, 3] = 0.9  # suppressed with the first peak
    heatmap[10, 12] = 0.8
    heatmap[14, 4] = 0.7
    peaks = heatmap_peaks(heatmap, count=3, suppression_cells=2)
    assert len(peaks) == 3
    assert peaks[0][:2] == ((2.5 / 16), (2.5 / 16))
    assert peaks[1][:2] == ((12.5 / 16), (10.5 / 16))


def test_palm_bootstrap_requires_wrist_and_three_coherent_mcps():
    points = palm_points()
    accepted = np.zeros(21, dtype=bool)
    accepted[[0, 5, 9, 13]] = True
    assert OracleLandmarkerRuntime._palm_bootstrap_ok(points, accepted)
    assert not OracleLandmarkerRuntime._partial_emission_ok(points, accepted)
    accepted[[1, 2]] = True
    assert OracleLandmarkerRuntime._partial_emission_ok(points, accepted)
    accepted[13] = False
    assert not OracleLandmarkerRuntime._palm_bootstrap_ok(points, accepted)

    roi = OracleLandmarkerRuntime._palm_tracking_roi(points)
    assert roi is not None
    assert 0.08 <= roi["size"] <= 1.5
    assert np.isclose(np.linalg.norm(roi["y_axis"]), 1.0)


def test_temporal_confidence_bonus_is_bounded_and_resets_on_jump():
    runtime = OracleLandmarkerRuntime.__new__(OracleLandmarkerRuntime)
    runtime._cold_history = []
    runtime.cold_history_frames = 5
    runtime.cold_min_history = 3
    points = palm_points()
    confidence = np.full(21, 0.20, dtype=np.float64)

    runtime._accumulate_cold_start(points, confidence)
    runtime._accumulate_cold_start(points, confidence)
    _combined_points, combined_confidence, frames = runtime._accumulate_cold_start(
        points, confidence)
    assert frames == 3
    assert np.allclose(combined_confidence, 0.25)

    jumped = points + np.asarray([0.35, 0.0])
    _points, _confidence, frames = runtime._accumulate_cold_start(
        jumped, confidence)
    assert frames == 1
