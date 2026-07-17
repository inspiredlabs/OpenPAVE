"""Run the real Monty supervised-pretraining experiment over paired hands."""
from __future__ import print_function

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from tbp.monty.frameworks.loggers.monty_handlers import (
    BasicCSVStatsHandler,
    DetailedJSONHandler,
)
from tbp.monty.frameworks.models.evidence_matching.learning_module import EvidenceGraphLM
from tbp.monty.frameworks.models.evidence_matching.model import MontyForEvidenceGraphMatching

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from framework_pretraining import (  # noqa: E402
    HandLandmarkPretrainingExperiment,
    LandmarkReplayMotorSystem,
    PairedHandEnvironment,
    PairedHandInterface,
)
from hand_landmark_sensor import HandLandmarkSensorModule  # noqa: E402

ROOT = HERE.parents[2]
DEFAULT_EPISODES = ROOT / "train" / "runs" / "monty_landmark_alignment" / "episodes.npz"
DEFAULT_OUT = ROOT / "train" / "runs" / "monty_landmark_alignment" / "framework_run"
SENSOR_ID = "hand_landmark_sensor"


def make_config(args):
    lm = {
        "_target_": EvidenceGraphLM,
        "max_match_distance": 0.20,
        "tolerances": {SENSOR_ID: {"joint_id": np.full(21, 0.05)}},
        "feature_weights": {SENSOR_ID: {"joint_id": np.ones(21)}},
        # Canonical wrist-to-middle-MCP distance is 1. Some foreshortened
        # teacher frames extend beyond +/-2 units, so a four-unit grid rejects
        # otherwise valid joint sets. Eight units contains the measured set.
        "max_graph_size": 8.0,
        "num_model_voxels_per_dim": 80,
        "use_multithreading": bool(args.multithreading),
        "hypotheses_updater_args": {"initial_possible_poses": "informed"},
    }
    monty = {
        "monty_class": MontyForEvidenceGraphMatching,
        "monty_args": {
            "min_eval_steps": 3,
            "min_train_steps": 3,
            "num_exploratory_steps": 21,
            "max_total_steps": 21,
        },
        "learning_modules": {"learning_module_0": lm},
        "sensor_modules": {"sensor_module_0": {
            "_target_": HandLandmarkSensorModule,
            "sensor_module_id": SENSOR_ID,
            "stream": "teacher",
            "confidence_threshold": args.confidence,
        }},
        "motor_system_config": {"_target_": LandmarkReplayMotorSystem},
        "sm_to_agent_dict": {SENSOR_ID: "agent_id_0"},
        "sm_to_lm_matrix": [[0]],
        "lm_to_lm_matrix": None,
        "lm_to_lm_vote_matrix": None,
    }
    config = {
        "show_sensor_output": False,
        "do_train": True,
        "do_eval": False,
        "max_train_steps": 21,
        "max_eval_steps": 21,
        "max_total_steps": 21,
        "n_train_epochs": int(args.per_object) * 4,
        "n_eval_epochs": 0,
        "model_name_or_path": "",
        "min_lms_match": 1,
        "seed": args.seed,
        "supervised_lm_ids": "all",
        "monty_config": monty,
        "environment": {
            "env_init_func": PairedHandEnvironment,
            "env_init_args": {
                "episodes_npz": str(args.episodes),
                "role": "exploration",
                "per_object": args.per_object,
            },
            "transform": None,
        },
        "train_env_interface_class": PairedHandInterface,
        "train_env_interface_args": {},
        "eval_env_interface_class": PairedHandInterface,
        "eval_env_interface_args": {},
        "logging": {
            "output_dir": str(args.out),
            "run_name": "hand_landmark_supervised_pretraining",
            "python_log_level": "INFO",
            "python_log_to_file": True,
            "python_log_to_stderr": False,
            "monty_log_level": "DETAILED",
            "monty_handlers": [BasicCSVStatsHandler, DetailedJSONHandler],
            "wandb_handlers": [],
            "detailed_episodes_to_save": list(range(int(args.per_object) * 4)),
            # Four episodes are deliberately bounded; use the consolidated
            # format consumed by tbp.monty's documented load_stats helper.
            "detailed_save_per_episode": False,
        },
    }
    return OmegaConf.create(config, flags={"allow_objects": True})


def main(argv=None):
    args = parser().parse_args(argv)
    cfg = make_config(args)
    experiment = HandLandmarkPretrainingExperiment(cfg)
    # tbp.monty's Hydra entry point normally performs this dehydration step.
    experiment._monty_cfg = cfg["monty_config"]
    with experiment:
        experiment.run()
        known = list(experiment.model.learning_modules[0].get_all_known_object_ids())
        episodes = int(experiment.train_episodes)
    actual_out = args.out / "pretrained"
    summary = {
        "contract": "openpave.tbp-monty-framework-pretraining.v1",
        "experiment_class": "MontySupervisedObjectPretrainingExperiment",
        "training_episodes": episodes,
        "known_objects": known,
        "monty_handlers": ["BasicCSVStatsHandler", "DetailedJSONHandler"],
        "commands_enabled": False,
        "evidence_graph_multithreading": bool(args.multithreading),
        "output_dir": str(actual_out),
    }
    (actual_out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--per-object", type=int, default=1)
    p.add_argument("--confidence", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--multithreading", action="store_true",
                   help="parallelize EvidenceGraphLM hypothesis evaluation")
    return p


if __name__ == "__main__":
    main()
