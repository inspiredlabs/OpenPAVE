"""EvidenceLM: few-shot object learning + pose-hypothesis evidence, vectorised.

Monty's evidence-based learning module, reduced to its geometric core: an
object is a set of exemplar reference frames (location constellations); a new
episode is recognised by solving the optimal rotation (Kabsch) against EVERY
exemplar of EVERY object IN ONE BATCHED CALL and converting residuals to
evidence. Unknowns abstain: best evidence under the floor returns 'noop'.

Learning is gradient-free and additive — `learn_episode` appends a graph, so
a new object or a new person costs seconds and never retrains the rest
(exactly the insect-poc contract, one abstraction up).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .protocol import Episode


def _normalise(locs: np.ndarray) -> np.ndarray:
    """Object reference frame: first observation is the origin, scale-norm,
    ORIENTATION KEPT — pose is recovered at recognition time, not erased."""
    pts = locs.astype(np.float32) - locs[0]
    return pts / (np.abs(pts).max() or 1.0)


class EvidenceLM:
    def __init__(self, sigma: float = 0.16, evidence_floor: float = 0.50) -> None:
        self.sigma = sigma
        self.evidence_floor = evidence_floor
        self._graphs: dict[str, list[np.ndarray]] = {}
        self._stack: np.ndarray | None = None      # (E, N, 3) all exemplars
        self._owners: list[str] = []               # exemplar -> object name

    # ── learning (few-shot, additive) ────────────────────────────────
    def learn_episode(self, ep: Episode) -> None:
        assert ep.label, "learning episodes must be labelled"
        self._graphs.setdefault(ep.label, []).append(_normalise(ep.locations()))
        self._stack = None                          # invalidate the batch cache

    # ── recognition (batched pose-hypothesis evidence) ───────────────
    def _ensure_stack(self) -> None:
        if self._stack is None:
            self._owners = [n for n, g in self._graphs.items() for _ in g]
            self._stack = np.stack([e for g in self._graphs.values() for e in g])

    def infer(self, ep: Episode) -> tuple[str, float, np.ndarray | None]:
        """(object | 'noop', evidence, pose R of the winning hypothesis).
        One batched Kabsch over all exemplars — microseconds, not millis."""
        self._ensure_stack()
        obs = _normalise(ep.locations())            # (N, 3)
        ex = self._stack                             # (E, N, 3)
        h = np.einsum("ni,enj->eij", obs, ex)        # (E, 3, 3) covariance
        u, _, vt = np.linalg.svd(h)                  # batched SVD
        det = np.linalg.det(np.einsum("eij,ejk->eik", u, vt))
        s = np.repeat(np.eye(3)[None], len(ex), axis=0).astype(np.float32)
        s[:, 2, 2] = np.sign(det)
        r = np.einsum("eij,ejk,ekl->eil", u, s, vt)  # (E, 3, 3) optimal rotations
        aligned = np.einsum("ni,eij->enj", obs, r)
        rms = np.sqrt(((aligned - ex) ** 2).sum(-1).mean(-1))
        evidence = np.exp(-(rms / self.sigma) ** 2)
        best = int(evidence.argmax())
        e = float(evidence[best])
        if e < self.evidence_floor:
            return "noop", e, None
        return self._owners[best], e, r[best]

    def infer_partial(self, ep: Episode, joint_ids) -> tuple[str, float, np.ndarray | None]:
        """Infer from an explicitly observed subset of the reference graph.

        ``joint_ids`` maps each observation to its MediaPipe-compatible joint
        index. Missing joints do not receive placeholder coordinates and do
        not vote. At least three finite, distinct observations are required
        to constrain a pose hypothesis.
        """
        self._ensure_stack()
        ids = np.asarray(joint_ids, dtype=np.int64).reshape(-1)
        obs = np.asarray(ep.locations(), np.float32)
        if len(ids) != len(obs):
            raise ValueError("joint_ids and observations must have equal length")
        valid = (np.isfinite(obs).all(axis=1) & (ids >= 0) &
                 (ids < self._stack.shape[1]))
        ids, obs = ids[valid], obs[valid]
        if len(ids) < 3 or len(np.unique(ids)) != len(ids):
            return "noop", 0.0, None

        ex = self._stack[:, ids, :].astype(np.float32, copy=True)
        obs = obs - obs.mean(axis=0, keepdims=True)
        ex -= ex.mean(axis=1, keepdims=True)
        obs /= max(float(np.abs(obs).max()), 1e-6)
        ex /= np.maximum(np.abs(ex).max(axis=(1, 2), keepdims=True), 1e-6)

        h = np.einsum("ni,enj->eij", obs, ex)
        u, _, vt = np.linalg.svd(h)
        det = np.linalg.det(np.einsum("eij,ejk->eik", u, vt))
        s = np.repeat(np.eye(3, dtype=np.float32)[None], len(ex), axis=0)
        s[:, 2, 2] = np.sign(det)
        r = np.einsum("eij,ejk,ekl->eil", u, s, vt)
        aligned = np.einsum("ni,eij->enj", obs, r)
        rms = np.sqrt(((aligned - ex) ** 2).sum(-1).mean(-1))
        evidence = np.exp(-(rms / self.sigma) ** 2)
        best = int(evidence.argmax())
        value = float(evidence[best])
        if value < self.evidence_floor:
            return "noop", value, None
        return self._owners[best], value, r[best]

    # ── persistence ──────────────────────────────────────────────────
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{k: np.stack(v) for k, v in self._graphs.items()})
        meta = {"objects": {k: len(v) for k, v in self._graphs.items()},
                "sigma": self.sigma, "evidence_floor": self.evidence_floor,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        path.with_name("meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "EvidenceLM":
        meta_p = Path(path).with_name("meta.json")
        kw = {}
        if meta_p.exists():
            m = json.loads(meta_p.read_text())
            kw = {"sigma": m.get("sigma", 0.16),
                  "evidence_floor": m.get("evidence_floor", 0.50)}
        lm = cls(**kw)
        d = np.load(path, allow_pickle=True)
        lm._graphs = {k: list(d[k]) for k in d.files}
        return lm
