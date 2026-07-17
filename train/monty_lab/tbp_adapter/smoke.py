"""Protocol smoke test — run INSIDE the tbp.monty env, from anywhere:

  conda run -n tbp.monty python train/monty_lab/tbp_adapter/smoke.py

Verifies: real tbp.monty types import, the environment satisfies the
step/reset/close contract, and a full 21-landmark walk of the first episode
round-trips through Observations/ProprioceptiveState.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hand_episodes_env import AGENT_ID, SENSOR_ID, HandEpisodesEnvironment  # noqa: E402

NPZ = Path(__file__).resolve().parents[2] / "runs" / "monty_gestures" / "episodes.npz"


def main() -> None:
    env = HandEpisodesEnvironment(str(NPZ), split="train")
    obs, state = env.reset()
    walked = 1
    while env.steps_remaining > 0:
        obs, state = env.step(["visit_next"])
        walked += 1
    loc = obs[AGENT_ID][SENSOR_ID]["location"]
    print("object:", env.current_label)
    print("landmarks walked:", walked)
    print("final observation:", obs[AGENT_ID][SENSOR_ID]["landmark_index"], loc.round(3))
    print("proprioceptive agent position:", state[AGENT_ID].position)
    assert walked == 21 and obs[AGENT_ID][SENSOR_ID]["landmark_index"] == 20
    env.close()
    print("SMOKE OK — protocol conformance verified inside tbp.monty env")


if __name__ == "__main__":
    main()
