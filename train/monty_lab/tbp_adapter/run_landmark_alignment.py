"""Pretrain a real tbp.monty graph on MediaPipe and evaluate student points.

Run inside the osx-64 tbp.monty environment after prepare_landmark_pairs.py.
The output follows DetailedJSONHandler/load_stats conventions.
"""
from __future__ import print_function

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.loggers.monty_handlers import DetailedJSONHandler
from tbp.monty.frameworks.models.evidence_matching.learning_module import EvidenceGraphLM

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from hand_landmark_sensor import HandLandmarkSensorModule  # noqa: E402
from landmark_contract import canonicalize, paired_errors  # noqa: E402

ROOT = HERE.parents[2]
DEFAULT_EPISODES = ROOT / "train" / "runs" / "monty_landmark_alignment" / "episodes.npz"
DEFAULT_OUT = ROOT / "train" / "runs" / "monty_landmark_alignment" / "tbp_run"
SENSOR_ID = "hand_landmark_sensor"


def build_lm():
    return EvidenceGraphLM(
        max_match_distance=0.20,
        tolerances={SENSOR_ID: {"joint_id": np.full(21, 0.05)}},
        feature_weights={SENSOR_ID: {"joint_id": np.ones(21)}},
        # A four-unit grid rejected 85.7% of valid `point` observations and
        # silently omitted that object model; eight units contains the set.
        max_graph_size=8.0,
        num_model_voxels_per_dim=80,
        use_multithreading=False,
        hypotheses_updater_args=dict(initial_possible_poses="informed"),
    )


def target(name):
    return {"object": str(name), "quat_rotation": [1, 0, 0, 0]}


def observation(data, episode, joint):
    return {
        "frame_id": int(episode),
        "source": str(data["sources"][episode]),
        "source_index": int(data["source_indices"][episode]),
        "role": str(data["roles"][episode]),
        "label": str(data["labels"][episode]),
        "joint_id": int(joint),
        "teacher_uv": data["teacher_uv"][episode],
        "student_uv": data["student_uv"][episode],
        "student_confidence": data["student_confidence"][episode],
    }


def choose_training(data, per_object):
    selected = []
    train = np.where((data["roles"] == "exploration") & ~data["is_val"])[0]
    for label in sorted(set(data["labels"][train].tolist())):
        rows = train[data["labels"][train] == label]
        if len(rows):
            selected.extend(rows[np.linspace(0, len(rows) - 1,
                                             min(per_object, len(rows)), dtype=np.int64)])
    return np.asarray(selected, dtype=np.int64)


def train_graphs(data, rows, lm, sensor, ctx):
    for episode in rows:
        label = str(data["labels"][episode])
        sensor.stream = "teacher"
        sensor.reset()
        lm.mode = ExperimentMode.TRAIN
        lm.reset_stm()
        lm.fixme_reset_ground_truth(primary_target=target(label))
        for joint in range(21):
            message = sensor.step(ctx, observation(data, episode, joint))
            lm.exploratory_step(ctx, [message])
        lm.detected_object = label
        lm.detected_rotation_r = Rotation.identity()
        current = lm.buffer.get_current_location(input_channel="first")
        lm.buffer.stats["detected_location_rel_body"] = current
        lm.buffer.stats["detected_location_on_model"] = current
        lm.update_ltm_from_stm()
        lm.fixme_update_ground_truth()


def comparison(data, episode, confidence_threshold):
    teacher = data["teacher_uv"][episode]
    student = data["student_uv"][episode]
    raw = paired_errors(teacher[None], student[None])[0]
    try:
        teacher_c, _ = canonicalize(teacher)
        student_c, _ = canonicalize(
            student, weights=np.asarray(data["student_confidence"][episode]))
        canonical = np.linalg.norm(student_c - teacher_c, axis=1)
    except ValueError:
        canonical = np.full(21, np.nan)
    return {
        "joint_id": np.arange(21),
        "pixel_error_384": raw,
        "canonical_error": canonical,
        "mean_pixel_error_384": float(np.nanmean(raw)),
        "accepted": (np.asarray(data["student_confidence"][episode]) >=
                     float(confidence_threshold)),
    }


def log_episode(handler, output_dir, local_episode, global_episode, detail, basic):
    pool = {
        "BASIC": {"eval_stats": {local_episode: basic}},
        "DETAILED": {local_episode: detail},
    }
    handler.report_episode(
        pool, str(output_dir), local_episode, mode=ExperimentMode.EVAL,
        eval_episodes_to_total={local_episode: global_episode},
        train_episodes_to_total={},
    )


def evaluate(data, rows, lm, sensor, ctx, out, detailed_cap):
    detailed_ids = set(range(min(int(detailed_cap), len(rows))))
    handler = DetailedJSONHandler(detailed_episodes_to_save=detailed_ids)
    records, comparisons = [], []
    for local_episode, episode in enumerate(rows):
        save_detailed = local_episode in detailed_ids
        lm.has_detailed_logger = save_detailed
        label = str(data["labels"][episode])
        sensor.stream = "student"
        sensor.reset()
        lm.mode = ExperimentMode.EVAL
        lm.reset_stm()
        lm.fixme_reset_ground_truth(primary_target=target(label))
        for joint in range(21):
            message = sensor.step(ctx, observation(data, episode, joint))
            lm.add_lm_processing_to_buffer_stats(lm_processed=message.use_state)
            if message.use_state:
                lm.matching_step(ctx, [message])
        mlh = lm.get_current_mlh()
        prediction = str(mlh["graph_id"]) if mlh else "none"
        cmp = comparison(data, episode, sensor.confidence_threshold)
        record = {
            "episode": int(episode), "target": label, "prediction": prediction,
            "role": str(data["roles"][episode]),
            "source": str(data["sources"][episode]),
            "correct": prediction == label,
            "mean_pixel_error_384": cmp["mean_pixel_error_384"],
        }
        records.append(record)
        comparisons.append((int(episode), str(data["roles"][episode]), label,
                            prediction, prediction == label,
                            cmp["pixel_error_384"], cmp["canonical_error"],
                            cmp["accepted"]))
        if save_detailed:
            detail = {
                "LM_0": {
                    "locations": lm.buffer.locations,
                    "features": lm.buffer.features,
                    "displacements": lm.buffer.displacements,
                    "possible_matches": lm.buffer.stats.get("possible_matches", []),
                    "current_mlh": lm.buffer.stats.get("current_mlh", []),
                    "possible_locations": lm.buffer.stats.get("possible_locations", []),
                    "evidences": lm.buffer.stats.get("evidences", []),
                },
                "SM_0": sensor.state_dict(),
                "landmark_comparison": cmp,
            }
            log_episode(handler, out, local_episode, local_episode, detail, record)
    handler.close()
    return records, comparisons


def summarize(records):
    roles = sorted(set(row["role"] for row in records))
    summary = {}
    for role in roles:
        rows = [row for row in records if row["role"] == role]
        summary[role] = {
            "episodes": len(rows),
            "recognition_accuracy": float(np.mean([row["correct"] for row in rows])),
            "mean_landmark_error_384": float(np.mean(
                [row["mean_pixel_error_384"] for row in rows])),
        }
    return summary


def main(argv=None):
    args = parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    with np.load(args.episodes, allow_pickle=True) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    ctx = RuntimeContext(rng=np.random.RandomState(args.seed))
    lm = build_lm()
    sensor = HandLandmarkSensorModule(confidence_threshold=args.confidence)
    training = choose_training(data, args.train_per_object)
    train_graphs(data, training, lm, sensor, ctx)

    pretrained = args.out / "pretrained"
    pretrained.mkdir(exist_ok=True)
    torch.save({"lm_dict": {0: lm.state_dict()}, "sm_dict": {0: sensor.state_dict()}},
               pretrained / "model.pt")

    evaluation = np.where(data["is_val"])[0]
    if args.eval_cap > 0 and len(evaluation) > args.eval_cap:
        evaluation = evaluation[np.linspace(0, len(evaluation) - 1,
                                            args.eval_cap, dtype=np.int64)]
    records, comparisons = evaluate(
        data, evaluation, lm, sensor, ctx, args.out, args.detailed_cap)
    np.savez_compressed(
        args.out / "comparison.npz",
        episode=np.asarray([row[0] for row in comparisons], np.int64),
        role=np.asarray([row[1] for row in comparisons]),
        target=np.asarray([row[2] for row in comparisons]),
        prediction=np.asarray([row[3] for row in comparisons]),
        correct=np.asarray([row[4] for row in comparisons], bool),
        pixel_error_384=np.asarray([row[5] for row in comparisons], np.float32),
        canonical_error=np.asarray([row[6] for row in comparisons], np.float32),
        accepted=np.asarray([row[7] for row in comparisons], bool),
    )
    summary = {
        "contract": "openpave.tbp-monty-landmark-alignment.v1",
        "training_episodes": int(len(training)),
        "known_objects": list(lm.get_all_known_object_ids()),
        "evaluation": summarize(records),
        "commands_enabled": False,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--train-per-object", type=int, default=16)
    p.add_argument("--eval-cap", type=int, default=200)
    p.add_argument("--detailed-cap", type=int, default=8,
                   help="episodes retained by DetailedJSONHandler")
    p.add_argument("--confidence", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=7)
    return p


if __name__ == "__main__":
    main()
