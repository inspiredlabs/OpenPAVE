"""MLX trainer for a fixed 256-unit temporal fusion core and its ablations."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

STATE_CLASSES = ["absent", "present_still", "present_left", "present_right"]


def prepare_core_demo(path: Path, minutes: float = 15.0, fps: int = 15, seed: int = 41) -> None:
    """Generate a temporal contract test with occlusion, flicker and light shifts.

    This proves the trainer and temporal task, not camera accuracy. Real capture
    tooling must write the identical arrays described in README.md.
    """
    rng = np.random.default_rng(seed)
    total = int(minutes * 60 * fps)
    dims = 12
    X, y, recording, timestamps = [], [], [], []
    # Short independent recordings make group-held-out evaluation possible.
    clip_frames = fps * 10
    for rid, start in enumerate(range(0, total, clip_frames)):
        n = min(clip_frames, total - start)
        phase = np.arange(n) % (fps * 15)
        truth = np.zeros(n, dtype=np.int32)
        truth[(phase >= fps * 3) & (phase < fps * 7)] = 3
        truth[(phase >= fps * 7) & (phase < fps * 11)] = 1
        truth[(phase >= fps * 11)] = 2
        feat = rng.normal(0, 0.40, (n, dims)).astype(np.float32)
        present = truth > 0
        feat[:, 0] += np.where(present, 1.0, -0.9)       # presence margin
        feat[:, 1] += np.where(truth == 1, 0.9, -0.2)   # still
        feat[:, 2] += np.where(truth == 2, 0.9, -0.2)   # left
        feat[:, 3] += np.where(truth == 3, 0.9, -0.2)   # right
        feat[:, 4] = present.astype(np.float32) + rng.normal(0, 0.15, n)  # track quality
        feat[:, 5] = np.minimum(np.arange(n) / fps, 5.0) / 5.0            # track age proxy
        # The instantaneous evidence is deliberately corrupted. Temporal state
        # should bridge 2-5-frame dropouts without smearing real boundaries.
        dropout = rng.random(n) < 0.06
        feat[dropout, :5] = rng.normal(0, 0.2, (dropout.sum(), 5))
        # Adversarial single-frame specialist errors are indistinguishable from
        # truth without history. This is the bounded temporal capability the
        # matched zero-recurrence ablation is expected to expose.
        flipped = rng.random(n) < 0.10
        feat[flipped, :4] *= -1.0
        light = (phase >= fps * 5) & (phase < fps * 5 + 3)
        feat[light, 6] = 2.5                         # illumination delta
        feat[light, 0] *= -0.8                       # fools raw presence evidence
        shake = (phase >= fps * 12) & (phase < fps * 12 + 2)
        feat[shake, 7] = 2.0                         # camera motion
        feat[shake, 2:4] += rng.normal(0, 1.5, (shake.sum(), 2))
        feat[:, 8] = np.r_[0.0, np.abs(np.diff(feat[:, 0]))]
        feat[:, 9] = np.r_[0.0, np.abs(np.diff(feat[:, 2]))]
        feat[:, 10] = np.r_[0.0, np.abs(np.diff(feat[:, 3]))]
        feat[:, 11] = 1.0
        X.append(feat); y.append(truth)
        recording.extend([rid] * n)
        timestamps.extend(np.arange(n) / fps)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=np.concatenate(X), y=np.concatenate(y),
                        recording_id=np.asarray(recording, np.int32),
                        timestamp_s=np.asarray(timestamps, np.float32), fps=np.int32(fps),
                        classes=np.asarray(STATE_CLASSES))
    print(f"[core-data] {path}: {total} frames, {minutes:g} minutes, {dims} features")


def _power_radius(A: np.ndarray, iterations: int = 80) -> float:
    v = np.ones(A.shape[0], dtype=np.float32) / math.sqrt(A.shape[0])
    for _ in range(iterations):
        v = np.einsum("ij,j->i", A, v, optimize=True)
        norm = float(np.linalg.norm(v)) or 1.0
        v /= norm
    return float(np.linalg.norm(np.einsum("ij,j->i", A, v, optimize=True)))


def fixed_adjacency(units: int, seed: int, density: float = 0.02,
                    spectral_radius: float = 0.85) -> np.ndarray:
    """Signed block-shaped sparse core, scaled to a stable spectral radius."""
    rng = np.random.default_rng(seed)
    A = np.zeros((units, units), dtype=np.float32)
    cuts = (0, units // 4, 3 * units // 4, units)
    # sensory->internal, internal recurrent, internal->output, feedback
    blocks = ((1, 0, 1.5), (1, 1, 1.0), (2, 1, 1.5), (1, 2, 0.35))
    for dst, src, multiplier in blocks:
        r0, r1 = cuts[dst], cuts[dst + 1]
        c0, c1 = cuts[src], cuts[src + 1]
        mask = rng.random((r1 - r0, c1 - c0)) < min(1.0, density * multiplier)
        values = rng.normal(0, 1, mask.shape).astype(np.float32)
        A[r0:r1, c0:c1] = values * mask
    radius = _power_radius(A)
    if radius > 0:
        A *= spectral_radius / radius
    return A


def _group_split(groups: np.ndarray, seed: int):
    unique = np.unique(groups)
    if len(unique) < 3:
        raise ValueError("core training needs at least three recording_id groups")
    rng = np.random.default_rng(seed); rng.shuffle(unique)
    ntest = max(1, int(round(len(unique) * 0.2)))
    nval = max(1, int(round(len(unique) * 0.2)))
    test_g, val_g = unique[:ntest], unique[ntest:ntest + nval]
    train_g = unique[ntest + nval:]
    return tuple(np.flatnonzero(np.isin(groups, g)) for g in (train_g, val_g, test_g))


def _windows(indices: np.ndarray, groups: np.ndarray, length: int):
    result = []
    for group in np.unique(groups[indices]):
        rows = indices[groups[indices] == group]
        rows.sort()
        for start in range(0, max(1, len(rows) - length + 1), length):
            window = rows[start:start + length]
            if len(window) >= 4:
                result.append(window)
    return result


def _metrics(y: np.ndarray, pred: np.ndarray, groups: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    transitions_true = transitions_pred = 0
    for group in np.unique(groups):
        yt, yp = y[groups == group], pred[groups == group]
        transitions_true += int(np.count_nonzero(np.diff(yt)))
        transitions_pred += int(np.count_nonzero(np.diff(yp)))
    counts = np.bincount(pred, minlength=len(STATE_CLASSES))
    active_classes = int(np.count_nonzero(counts))
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "transitions_true": transitions_true, "transitions_pred": transitions_pred,
            "flicker_ratio": transitions_pred / max(1, transitions_true),
            "predicted_class_counts": counts.tolist(),
            "collapsed": active_classes < len(np.unique(y))}


def _predict_numpy(X, groups, A, pin, bin_, pout, bout, leak):
    pred = np.empty(len(X), dtype=np.int32)
    for group in np.unique(groups):
        h = np.zeros(A.shape[0], dtype=np.float32)
        for row in np.flatnonzero(groups == group):
            candidate = np.tanh(np.einsum("d,du->u", X[row], pin, optimize=True)
                                + np.einsum("u,vu->v", h, A, optimize=True) + bin_)
            h = leak * h + (1.0 - leak) * candidate
            pred[row] = int(np.argmax(np.einsum("u,uc->c", h, pout, optimize=True) + bout))
    return pred


def _fit_variant(X, y, groups, tr, va, A_np, units, epochs, seed, leak,
                 window_length=32, patience=18):
    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    rng = np.random.default_rng(seed)
    classes = int(y.max()) + 1
    pin = mx.array(rng.normal(0, 0.04, (X.shape[1], units)).astype(np.float32))
    bin_ = mx.zeros((units,), dtype=mx.float32)
    pout = mx.array(rng.normal(0, 0.04, (units, classes)).astype(np.float32))
    bout = mx.zeros((classes,), dtype=mx.float32)
    A = mx.array(A_np)
    counts = np.bincount(y[tr], minlength=classes).astype(np.float32)
    weights = mx.array(len(tr) / np.maximum(counts * classes, 1.0))
    windows = _windows(tr, groups, window_length)
    # Rare directional events occupy ~1% of IPN frames. Repeat windows that
    # contain them so both recurrent and ablation models receive enough useful
    # gradient without changing held-out distributions or fabricating frames.
    rare_windows = [rows for rows in windows if np.any(y[rows] >= 2)]
    windows = windows + rare_windows * 4

    def loss_fn(pin, bin_, pout, bout, xb, yb):
        h = mx.zeros((units,), dtype=mx.float32)
        losses = []
        for step in range(xb.shape[0]):
            candidate = mx.tanh(xb[step] @ pin + h @ A.T + bin_)
            h = leak * h + (1.0 - leak) * candidate
            logits = h @ pout + bout
            logp = logits - mx.logsumexp(logits)
            losses.append(-logp[yb[step]] * weights[yb[step]])
        l2 = 1e-5 * (mx.sum(pin * pin) + mx.sum(pout * pout))
        return mx.mean(mx.stack(losses)) + l2

    loss_grad = mx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3))
    velocity = [mx.zeros_like(p) for p in (pin, bin_, pout, bout)]
    best = None; best_f1 = -1.0; stale = 0
    for epoch in range(epochs):
        rng.shuffle(windows)
        lr = 0.025 / (1.0 + epoch * 0.025)
        for rows in windows:
            xb, yb = mx.array(X[rows]), mx.array(y[rows])
            _, grads = loss_grad(pin, bin_, pout, bout, xb, yb)
            norm = mx.sqrt(sum(mx.sum(g * g) for g in grads))
            factor = mx.minimum(1.0, 1.0 / (norm + 1e-6))
            params = [pin, bin_, pout, bout]
            for i in range(4):
                velocity[i] = 0.9 * velocity[i] + grads[i] * factor
                params[i] = params[i] - lr * velocity[i]
            pin, bin_, pout, bout = params
        mx.eval(pin, bin_, pout, bout)
        arrays = tuple(np.asarray(p, dtype=np.float32) for p in (pin, bin_, pout, bout))
        val_pred = _predict_numpy(X[va], groups[va], A_np, *arrays, leak)
        score = _metrics(y[va], val_pred, groups[va])["macro_f1"]
        if score > best_f1 + 1e-4:
            best_f1, best, stale = score, arrays, 0
        else:
            stale += 1
        if stale >= patience:
            break
    return best, {"epochs_completed": epoch + 1, "best_validation_macro_f1": best_f1}


def train_core(dataset: Path, output: Path, units: int = 256, epochs: int = 120,
               seed: int = 73, max_minutes: float = 15.0) -> dict:
    """Train recurrent and no-recurrence variants and produce an honest verdict."""
    ds = np.load(dataset, allow_pickle=False)
    X, y = ds["X"].astype(np.float32), ds["y"].astype(np.int32)
    groups = ds["recording_id"].astype(np.int32)
    fps = int(ds["fps"])
    cap = int(max_minutes * 60 * fps)
    if len(X) > cap:
        # Keep complete recording groups until the time budget is exhausted.
        keep, used = [], 0
        for group in np.unique(groups):
            rows = np.flatnonzero(groups == group)
            if used and used + len(rows) > cap:
                break
            keep.extend(rows.tolist()); used += len(rows)
        keep = np.asarray(keep)
        X, y, groups = X[keep], y[keep], groups[keep]
    tr, va, te = _group_split(groups, seed)
    mean, scale = X[tr].mean(0), X[tr].std(0); scale[scale < 1e-6] = 1.0
    X = ((X - mean) / scale).astype(np.float32)
    A = fixed_adjacency(units, seed)
    variants = {"recurrent": A, "no_recurrence": np.zeros_like(A)}
    report = {"schema_version": 1, "dataset": str(dataset), "units": units,
              "frames_used": len(X), "minutes_used": len(X) / fps / 60,
              "train_frames": len(tr), "validation_frames": len(va), "test_frames": len(te),
              "seed": seed, "variants": {}}
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for name, adjacency in variants.items():
        params, training = _fit_variant(X, y, groups, tr, va, adjacency, units, epochs,
                                        seed + (name == "no_recurrence"), leak=0.65)
        pred = _predict_numpy(X[te], groups[te], adjacency, *params, 0.65)
        metrics = _metrics(y[te], pred, groups[te])
        report["variants"][name] = {**training, **metrics}
        np.savez_compressed(output / f"{name}.npz", adjacency=adjacency, mean=mean, scale=scale,
                            pin=params[0], bin=params[1], pout=params[2], bout=params[3],
                            leak=np.float32(0.65), classes=np.asarray(STATE_CLASSES))
        print(f"[core-train] {name}: val={training['best_validation_macro_f1']:.3f} "
              f"test={metrics['macro_f1']:.3f} flicker={metrics['flicker_ratio']:.2f}")
    recurrent, static = report["variants"]["recurrent"], report["variants"]["no_recurrence"]
    report["improvement"] = {"macro_f1_delta": recurrent["macro_f1"] - static["macro_f1"],
                             "flicker_delta": recurrent["flicker_ratio"] - static["flicker_ratio"]}
    report["substrate_improves_bounded_perception"] = (
        report["improvement"]["macro_f1_delta"] > 0.005
        and recurrent["flicker_ratio"] <= static["flicker_ratio"]
        and not recurrent["collapsed"])
    report["train_seconds"] = time.perf_counter() - started
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    verdict = "PASS" if report["substrate_improves_bounded_perception"] else "NOT PROVEN"
    print(f"[core-verdict] {verdict}: F1 delta={report['improvement']['macro_f1_delta']:+.3f}, "
          f"flicker delta={report['improvement']['flicker_delta']:+.2f}")
    print(f"[core-train] wrote {output / 'report.json'} in {report['train_seconds']:.1f}s")
    return report
