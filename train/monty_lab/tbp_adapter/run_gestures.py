"""Gesture learning + recognition with the REAL tbp.monty EvidenceGraphLM.

RUN INSIDE THE tbp.monty ENV (Python 3.8, osx-64/Rosetta):

  conda run -n tbp.monty python train/monty_lab/tbp_adapter/run_gestures.py

The SensorModule here is `episode_to_percepts`: it converts one exported hand
constellation (21 landmarks in the hand reference frame) into the framework's
CMP Messages — features (joint identity) at 3D poses — which is precisely the
sensor-module contract. Train/eval lifecycle mirrors the framework's own unit
tests (tests/unit/frameworks/models/evidence_matching/evidence_lm_test.py),
which is the supported programmatic entry (`run.py` + Hydra is a config shell
around these same objects).

Input : train/runs/monty_gestures/episodes.npz   (monty_lab runner export)
Output: accuracy report + learned graphs summary.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from tbp.monty.cmp import Message
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)

SENSOR_ID = "joint_sensor"
NPZ = Path(__file__).resolve().parents[2] / "runs" / "monty_gestures" / "episodes.npz"
TRAIN_PER_OBJECT = 4          # few-shot, like everything Monty
POSE_VECTORS = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)


def episode_to_percepts(locations):
    """The SensorModule: one constellation -> 21 CMP Messages.

    Feature = joint identity (which landmark), pose = its 3D location in the
    hand reference frame. Constant pose vectors: geometry-only for now, per
    train/hand-landmark/README.md stage 1."""
    percepts = []
    for i, loc in enumerate(locations):
        percepts.append(Message(
            location=np.asarray(loc, dtype=float),
            morphological_features={
                "pose_vectors": POSE_VECTORS.copy(),
                "pose_fully_defined": True,
                "on_object": 1,
            },
            non_morphological_features={"joint_id": [i / 20.0]},
            confidence=1.0,
            use_state=True,
            sender_id=SENSOR_ID,
            sender_type="SM",
        ))
    return percepts


def build_lm():
    return EvidenceGraphLM(
        max_match_distance=0.15,          # hand frame spans ~[-1, 1]
        tolerances={SENSOR_ID: {"joint_id": [0.03]}},
        feature_weights={SENSOR_ID: {"joint_id": np.array([1.0])}},
        max_graph_size=4.0,               # NOT the default 30cm — our units differ
        hypotheses_updater_args=dict(initial_possible_poses="informed"),
    )


def target(name):
    return {"object": name, "quat_rotation": [1, 0, 0, 0]}


def main():
    d = np.load(NPZ, allow_pickle=True)
    locs, labels, is_val = d["locations"], d["labels"], d["is_val"]
    objects = sorted(set(labels.tolist()))
    ctx = RuntimeContext(rng=np.random.RandomState(7))
    lm = build_lm()

    # ── train: few-shot exploratory episodes per gesture object ─────────────
    t0 = time.perf_counter()
    for obj in objects:
        idx = np.where((labels == obj) & ~is_val)[0][:TRAIN_PER_OBJECT]
        for ep_i in idx:
            lm.mode = ExperimentMode.TRAIN
            lm.reset_stm()
            lm.fixme_reset_ground_truth(primary_target=target(obj))
            for percept in episode_to_percepts(locs[ep_i]):
                lm.exploratory_step(ctx, [percept])
            lm.detected_object = obj
            # constellations are ALREADY in the hand reference frame, so the
            # detected rotation for graph extension is identity (None only
            # works for the very first graph build)
            lm.detected_rotation_r = Rotation.identity()
            cur = lm.buffer.get_current_location(input_channel="first")
            lm.buffer.stats["detected_location_rel_body"] = cur
            # observations are already in the model (hand) frame + identity
            # rotation, so rel-model location == rel-body location
            lm.buffer.stats["detected_location_on_model"] = cur
            lm.update_ltm_from_stm()
            lm.fixme_update_ground_truth()
    known = lm.get_all_known_object_ids()
    print("[tbp] learned objects:", list(known),
          f"in {time.perf_counter() - t0:.1f}s "
          f"({TRAIN_PER_OBJECT} episodes each, 21 percepts per episode)")

    # ── eval: held-out episodes through matching_step -> MLH ────────────────
    hits, total, per = 0, 0, {}
    t0 = time.perf_counter()
    for ep_i in np.where(is_val)[0]:
        true = str(labels[ep_i])
        lm.mode = ExperimentMode.EVAL
        lm.reset_stm()
        lm.fixme_reset_ground_truth(primary_target=target("placeholder"))
        for percept in episode_to_percepts(locs[ep_i]):
            lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            lm.matching_step(ctx, [percept])
            for obj in known:
                lm.get_unique_pose_if_available(obj)
        mlh = lm.get_current_mlh()
        pred = str(mlh["graph_id"]) if mlh else "none"
        ok = pred == true
        hits += ok
        total += 1
        h, t = per.get(true, (0, 0))
        per[true] = (h + ok, t + 1)
    dt = time.perf_counter() - t0
    print(f"[tbp] held-out recognition: {hits}/{total} ({hits / max(total,1):.0%}) "
          f"in {dt:.1f}s ({dt / max(total,1) * 1000:.0f}ms/episode incl. 21 matching steps)")
    for k, (h, t) in sorted(per.items()):
        print(f"    {k:8s} {h}/{t}")


if __name__ == "__main__":
    sys.exit(main())
