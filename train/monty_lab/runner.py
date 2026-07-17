"""Task-agnostic learn/eval runner. Any Task in monty_lab/tasks registers
itself; nothing here knows about gestures.

  .venv/bin/python -m monty_lab.runner learn --task gestures
  .venv/bin/python -m monty_lab.runner eval  --task gestures
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .evidence_lm import EvidenceLM
from .tasks import TASKS

RUNS = Path(__file__).resolve().parents[1] / "runs"


def model_path(task_name: str) -> Path:
    return RUNS / f"monty_{task_name}" / "objects.npz"


def learn(task_name: str) -> None:
    task = TASKS[task_name]()
    lm = EvidenceLM()
    t0 = time.perf_counter()
    n = 0
    for ep in task.learning_episodes():
        lm.learn_episode(ep)
        n += 1
    lm.save(model_path(task_name))
    print(f"[monty_lab] {task_name}: learned {n} episodes in "
          f"{time.perf_counter() - t0:.1f}s -> {model_path(task_name)}")


def evaluate(task_name: str) -> None:
    task = TASKS[task_name]()
    lm = EvidenceLM.load(model_path(task_name))
    per: dict[str, list[int]] = {}
    t_rec = []
    for ep in task.eval_episodes():
        t0 = time.perf_counter()
        obj, e, _pose = lm.infer(ep)
        out = task.outcome(obj, ep)
        t_rec.append((time.perf_counter() - t0) * 1e6)
        want = ep.label
        key = ep.meta.get("gt_class", want)
        hit = out == want
        wrong = out not in (want, "ABSTAIN", "")
        h, w_, t = per.get(key, [0, 0, 0])
        per[key] = [h + hit, w_ + wrong, t + 1]
    hits = sum(v[0] for v in per.values()); wrongs = sum(v[1] for v in per.values())
    total = sum(v[2] for v in per.values())
    t_rec.sort()
    print(f"[monty_lab] {task_name} eval: {hits}/{total} correct ({hits / total:.0%}), "
          f"{wrongs / total:.1%} wrong-action, recognition {t_rec[len(t_rec) // 2]:.0f}µs median")
    for k, (h, w_, t) in sorted(per.items()):
        print(f"    {k:14s} {h:>3}/{t:<3} hit  {w_:>2} wrong-action")


def export(task_name: str) -> None:
    """Serialise learning episodes for the tbp.monty side of the bridge
    (train/monty_lab/tbp_adapter/ — runs in the osx-64 conda env)."""
    import numpy as np
    task = TASKS[task_name]()
    locs, labels = [], []
    for ep in task.learning_episodes():
        locs.append(ep.locations())
        labels.append(ep.label)
    locs = np.stack(locs).astype("float32")
    is_val = np.zeros(len(labels), bool)
    is_val[int(len(labels) * 0.8):] = True
    out = model_path(task_name).with_name("episodes.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, locations=locs, labels=np.array(labels), is_val=is_val)
    print(f"[monty_lab] exported {len(labels)} episodes "
          f"({int(is_val.sum())} val) -> {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["learn", "eval", "export"])
    p.add_argument("--task", default="gestures", choices=list(TASKS))
    a = p.parse_args()
    {"learn": learn, "eval": evaluate, "export": export}[a.stage](a.task)


if __name__ == "__main__":
    main()
