"""CLI for independent GPU training, assembly, and CPU inference benchmarking."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from .runtime import PortableRbfSpecialist

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.json"
DEFAULT_DATA = ROOT / "data"
DEFAULT_RUNS = ROOT / "runs"


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("unsupported config schema")
    return config


def selected(config: dict, names: list[str]) -> list[str]:
    available = list(config["specialists"])
    if not names or names == ["all"]:
        return available
    bad = sorted(set(names) - set(available))
    if bad:
        raise ValueError(f"unknown specialist(s): {', '.join(bad)}")
    return names


def _demo_centres(name: str, classes: list[str], dims: int, rng) -> np.ndarray:
    centres = rng.normal(0, 1.2, (len(classes), dims)).astype(np.float32)
    # Make demo classes deliberately learnable but non-linear/noisy.
    centres += np.eye(len(classes), dims, dtype=np.float32) * 3.0
    if name == "motion" and dims >= 2:
        mapping = {"still": (0, 0), "left": (-4, 0), "right": (4, 0),
                   "up": (0, -4), "down": (0, 4)}
        centres[:, :2] = np.asarray([mapping[c] for c in classes], dtype=np.float32)
    return centres


def prepare_demo(config: dict, names: list[str], data_dir: Path, samples: int, seed: int) -> None:
    """Create deterministic contract/smoke data; real extractors overwrite the same NPZ files."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for pos, name in enumerate(names):
        spec = config["specialists"][name]
        rng = np.random.default_rng(seed + pos)
        classes, dims = spec["classes"], int(spec["feature_count"])
        centres = _demo_centres(name, classes, dims, rng)
        X, y, groups = [], [], []
        per_class = max(8, samples // len(classes))
        for ci, centre in enumerate(centres):
            raw = centre + rng.normal(0, 0.65, (per_class, dims))
            # A curved nuisance dimension makes RBF materially useful in the smoke dataset.
            raw[:, -1] += 0.35 * np.square(raw[:, 0])
            X.append(raw.astype(np.float32))
            y.extend([classes[ci]] * per_class)
            groups.extend(np.arange(per_class) // 8 + ci * 10000)
        target = data_dir / f"{name}.npz"
        np.savez_compressed(target, X=np.concatenate(X), y=np.asarray(y), groups=np.asarray(groups))
        print(f"[prepare] {name}: {target} ({len(y)} samples, {dims} features)")


def _backend(requested: str):
    """Resolve Apple Metal first, NVIDIA CUDA second, CPU last."""
    if requested in {"auto", "mlx"}:
        try:
            import mlx.core as mx
            # Force a tiny evaluated Metal operation now. Importing MLX alone
            # can succeed in a headless session that has no usable GPU.
            mx.set_default_device(mx.gpu)
            mx.eval(mx.array([1.0], dtype=mx.float32) + 1.0)
            return "mlx", None
        except Exception as exc:
            if requested == "mlx":
                raise RuntimeError("BACKEND=mlx requested but an MLX Metal GPU is unavailable") from exc
    if requested in {"auto", "cuml"}:
        try:
            from cuml.svm import SVC as SVC
            return "cuml", SVC
        except Exception as exc:
            if requested == "cuml":
                raise RuntimeError("BACKEND=cuml requested but RAPIDS cuML is unavailable") from exc
    from sklearn.svm import SVC
    return "sklearn", SVC


def _gamma_value(gamma, X: np.ndarray) -> float:
    if gamma == "scale":
        return 1.0 / (X.shape[1] * max(float(X.var()), 1e-12))
    if gamma == "auto":
        return 1.0 / X.shape[1]
    return float(gamma)


def _fit_mlx_binary(X: np.ndarray, binary: np.ndarray, spec: dict, seed: int):
    """Fit an RBF squared-hinge classifier entirely with MLX/Metal operations.

    MLX has no packaged SVC/QP solver. This optimizes the kernel SVM objective
    over a bounded anchor set, then prunes the smallest coefficients. The
    resulting decision function is the same portable sum(alpha_i*K(x,sv_i))+b
    consumed by runtime.py.
    """
    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    rng = np.random.default_rng(seed)
    limit = min(int(spec.get("max_support_vectors", 512)), len(X))
    # Preserve both sides of the one-vs-rest problem in the anchor set.
    pos, neg = np.flatnonzero(binary), np.flatnonzero(1 - binary)
    if not len(pos) or not len(neg):
        raise ValueError("each one-vs-rest MLX problem needs positive and negative samples")
    per_pos = min(len(pos), max(1, limit // 2))
    chosen = list(rng.choice(pos, per_pos, replace=False))
    remaining = min(len(neg), limit - len(chosen))
    chosen.extend(rng.choice(neg, remaining, replace=False).tolist())
    if len(chosen) < limit:
        rest = np.setdiff1d(np.arange(len(X)), np.asarray(chosen), assume_unique=False)
        chosen.extend(rng.choice(rest, min(len(rest), limit - len(chosen)), replace=False).tolist())
    anchors_np = X[np.asarray(chosen)].astype(np.float32)
    gamma = _gamma_value(spec["gamma"], X)

    Xm, Am = mx.array(X), mx.array(anchors_np)
    y = mx.array(binary.astype(np.float32) * 2.0 - 1.0)
    dist2 = mx.sum((Xm[:, None, :] - Am[None, :, :]) ** 2, axis=2)
    K = mx.exp(-gamma * dist2)
    anchor_dist2 = mx.sum((Am[:, None, :] - Am[None, :, :]) ** 2, axis=2)
    Kaa = mx.exp(-gamma * anchor_dist2)
    alpha = mx.zeros((len(anchors_np),), dtype=mx.float32)
    bias = mx.array(0.0, dtype=mx.float32)
    lr = float(spec.get("mlx_learning_rate", 0.03))
    regularization = 1.0 / max(float(spec["C"]), 1e-6)
    pos_weight = float(len(binary)) / max(2.0 * float(binary.sum()), 1.0)
    neg_weight = float(len(binary)) / max(2.0 * float((1 - binary).sum()), 1.0)
    weights = mx.where(y > 0, pos_weight, neg_weight)

    for epoch in range(int(spec.get("mlx_epochs", 300))):
        score = K @ alpha + bias
        hinge = mx.maximum(0.0, 1.0 - y * score)
        residual = weights * y * hinge
        grad_alpha = regularization * (Kaa @ alpha) \
            - 2.0 * (K.T @ residual) / len(X)
        grad_bias = -2.0 * mx.mean(residual)
        grad_norm = mx.sqrt(mx.sum(grad_alpha * grad_alpha) + grad_bias * grad_bias)
        grad_scale = mx.minimum(1.0, 1.0 / (grad_norm + 1e-6))
        # Smooth decay stabilizes the dense kernel optimization without an
        # optimizer dependency and keeps every update on Metal.
        step = lr / (1.0 + 0.01 * epoch)
        alpha = mx.clip(alpha - step * grad_alpha * grad_scale, -20.0, 20.0)
        bias = mx.clip(bias - step * grad_bias * grad_scale, -10.0, 10.0)
        if epoch % 25 == 0:
            mx.eval(alpha, bias)
    mx.eval(alpha, bias)
    coef = np.asarray(alpha, dtype=np.float32)
    intercept = float(np.asarray(bias))
    if not np.isfinite(coef).all() or not np.isfinite(intercept):
        raise RuntimeError("MLX RBF optimization became non-finite; no artifact was written")
    # Remove numerically irrelevant anchors; retain at least the strongest one.
    keep = np.flatnonzero(np.abs(coef) >= max(1e-5, np.max(np.abs(coef)) * 1e-4))
    if not len(keep):
        keep = np.asarray([int(np.argmax(np.abs(coef)))])
    return anchors_np[keep], coef[keep], intercept, gamma


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    return np.asarray(value)


def _split(X, y, groups, seed: int):
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    indices = np.arange(len(y))
    if groups is not None and len(np.unique(groups)) > 1:
        tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                        random_state=seed).split(indices, y, groups))
    else:
        tr, va = train_test_split(indices, test_size=0.2, random_state=seed, stratify=y)
    return tr, va


def train_one(name: str, spec: dict, data_dir: Path, runs_dir: Path,
              requested_backend: str, seed: int) -> None:
    from sklearn.metrics import accuracy_score, classification_report

    source = data_dir / f"{name}.npz"
    if not source.exists():
        raise FileNotFoundError(f"missing {source}; run prepare-demo or provide extracted features")
    ds = np.load(source, allow_pickle=False)
    X, y = ds["X"].astype(np.float32), ds["y"].astype(str)
    groups = ds["groups"] if "groups" in ds.files else None
    if X.shape[1] != int(spec["feature_count"]):
        raise ValueError(f"{name}: config expects {spec['feature_count']} features, dataset has {X.shape[1]}")
    tr, va = _split(X, y, groups, seed)
    mean, scale = X[tr].mean(0), X[tr].std(0)
    scale[scale < 1e-6] = 1.0
    Xtr, Xva = (X[tr] - mean) / scale, (X[va] - mean) / scale
    backend, SVC = _backend(requested_backend)

    supports, coefs, intercepts, gammas, offsets = [], [], [], [], [0]
    t0 = time.perf_counter()
    for class_index, cls in enumerate(spec["classes"]):
        binary = (y[tr] == cls).astype(np.int32)
        if backend == "mlx":
            support, coef, intercept, gamma = _fit_mlx_binary(
                Xtr, binary, spec, seed + class_index)
        else:
            kwargs = dict(kernel="rbf", C=float(spec["C"]), gamma=spec["gamma"])
            if backend == "sklearn":
                kwargs["class_weight"] = "balanced"
            model = SVC(**kwargs)
            model.fit(Xtr, binary)
            support = _to_numpy(model.support_vectors_).astype(np.float32)
            coef = _to_numpy(model.dual_coef_).reshape(-1).astype(np.float32)
            intercept = float(_to_numpy(model.intercept_).reshape(-1)[0])
            gamma = getattr(model, "_gamma", None)
            if gamma is None:
                gamma = _gamma_value(spec["gamma"], Xtr)
        supports.append(support); coefs.append(coef); intercepts.append(intercept); gammas.append(float(gamma))
        offsets.append(offsets[-1] + len(support))
    elapsed = time.perf_counter() - t0

    with tempfile.TemporaryDirectory(dir=runs_dir) as tmp:
        out = Path(tmp)
        np.savez_compressed(out / "model.npz", mean=mean.astype(np.float32), scale=scale.astype(np.float32),
                            support=np.concatenate(supports), coef=np.concatenate(coefs),
                            intercept=np.asarray(intercepts, np.float32), gamma=np.asarray(gammas, np.float32),
                            offsets=np.asarray(offsets, np.int32))
        meta = {"schema_version": 1, "name": name, "classes": spec["classes"],
                "feature_extractor": spec["feature_extractor"], "feature_count": X.shape[1],
                "accept_margin": spec.get("accept_margin", 0.0), "backend": backend,
                "train_samples": len(tr), "validation_samples": len(va),
                "support_vectors": int(offsets[-1]), "train_seconds": elapsed,
                "source": str(source), "created_at": time.time()}
        (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        portable = PortableRbfSpecialist(out)
        pred = portable.predict(Xva)
        meta["validation_accuracy"] = float(accuracy_score(y[va], pred))
        meta["validation_report"] = classification_report(y[va], pred, zero_division=0)
        (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        target = runs_dir / "specialists" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        backup = target.with_name(target.name + ".previous")
        staged = target.with_name(target.name + ".next")
        if staged.exists():
            shutil.rmtree(staged)
        shutil.copytree(out, staged)
        if backup.exists():
            shutil.rmtree(backup)
        if target.exists():
            target.rename(backup)
        staged.rename(target)
    print(f"[train] {name}: {backend}, {offsets[-1]} SV, {elapsed:.2f}s, val={meta['validation_accuracy']:.3f}")


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def assemble(config: dict, runs_dir: Path) -> Path:
    manifest = {"schema_version": 1, "created_at": time.time(), "specialists": {}}
    for name, spec in config["specialists"].items():
        model_dir = runs_dir / "specialists" / name
        if not (model_dir / "model.npz").exists():
            continue
        meta = json.loads((model_dir / "meta.json").read_text())
        if meta["feature_extractor"] != spec["feature_extractor"]:
            raise ValueError(f"{name}: trained extractor does not match config; retrain this specialist")
        manifest["specialists"][name] = {
            "path": str(Path("specialists") / name), "sha256": checksum(model_dir / "model.npz"),
            "feature_extractor": meta["feature_extractor"], "validation_accuracy": meta["validation_accuracy"]}
    runs_dir.mkdir(parents=True, exist_ok=True)
    target = runs_dir / "ensemble.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[assemble] {target}: {', '.join(manifest['specialists']) or 'no specialists'}")
    return target


def _rapl_uj() -> int | None:
    paths = list(Path("/sys/class/powercap").glob("intel-rapl*/energy_uj"))
    try:
        return sum(int(p.read_text()) for p in paths) if paths else None
    except OSError:
        return None


def bench(names: list[str], data_dir: Path, runs_dir: Path, iterations: int) -> None:
    report = {"created_at": time.time(), "iterations": iterations, "specialists": {}}
    for name in names:
        model = PortableRbfSpecialist(runs_dir / "specialists" / name)
        ds = np.load(data_dir / f"{name}.npz", allow_pickle=False)
        X = ds["X"].astype(np.float32)
        batch = X[np.arange(iterations) % len(X)]
        model.predict(batch[:min(8, len(batch))])  # warmup
        e0, cpu0, wall0 = _rapl_uj(), time.process_time(), time.perf_counter()
        for row in batch:
            model.predict(row)
        wall, cpu, e1 = time.perf_counter() - wall0, time.process_time() - cpu0, _rapl_uj()
        item = {"mean_latency_ms": wall * 1000 / iterations,
                "throughput_fps": iterations / wall, "cpu_seconds": cpu,
                "model_bytes": sum(p.stat().st_size for p in (runs_dir / "specialists" / name).iterdir())}
        if e0 is not None and e1 is not None and e1 >= e0:
            item["measured_joules"] = (e1 - e0) / 1e6
            item["joules_per_inference"] = item["measured_joules"] / iterations
        report["specialists"][name] = item
        print(f"[bench] {name}: {item['mean_latency_ms']:.3f} ms, {item['throughput_fps']:.1f} infer/s, {item['model_bytes']/1024:.1f} KiB")
    out = runs_dir / "benchmark.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    if not any("measured_joules" in x for x in report["specialists"].values()):
        print("[bench] energy counter unavailable; latency/model size recorded, use an inline power meter on Pi")
    print(f"[bench] wrote {out}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("list", "prepare-demo", "train", "assemble", "bench", "all",
                                       "prepare-core-demo", "train-core", "fetch-ipn", "prepare-ipn-core",
                                       "prepare-ipn-direction"))
    p.add_argument("--specialist", action="append", default=[], help="repeatable name, or all")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    p.add_argument("--backend", choices=("auto", "mlx", "cuml", "sklearn"), default="auto")
    p.add_argument("--samples", type=int, default=1200)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--minutes", type=float, default=15.0)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--core-units", type=int, default=256)
    p.add_argument("--core-epochs", type=int, default=120)
    p.add_argument("--raw-dir", type=Path, default=ROOT / "raw" / "ipn")
    p.add_argument("--ipn-shard", type=int, default=1)
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    config = load_config(args.config)
    names = selected(config, args.specialist)
    if args.command == "prepare-core-demo":
        from .core_train import prepare_core_demo
        prepare_core_demo(args.data_dir / "core_sequences.npz", args.minutes, args.fps, args.seed)
        return 0
    if args.command == "train-core":
        from .core_train import train_core
        train_core(args.data_dir / "core_sequences.npz", args.runs_dir / "shaped-core",
                   args.core_units, args.core_epochs, args.seed, args.minutes)
        return 0
    if args.command == "fetch-ipn":
        from .real_data import fetch_ipn
        fetch_ipn(args.raw_dir, args.ipn_shard)
        return 0
    if args.command == "prepare-ipn-core":
        from .real_data import prepare_ipn_core
        prepare_ipn_core(args.raw_dir, args.data_dir / "core_sequences.npz", args.minutes, args.fps)
        return 0
    if args.command == "prepare-ipn-direction":
        from .real_data import prepare_ipn_direction
        prepare_ipn_direction(args.raw_dir, args.data_dir / "motion.npz", args.minutes, args.fps)
        return 0
    if args.command == "list":
        for name in names:
            spec = config["specialists"][name]
            state = "trained" if (args.runs_dir / "specialists" / name / "model.npz").exists() else "missing"
            print(f"{name:10s} {state:7s} {spec['feature_extractor']} {spec['feature_count']}f")
        return 0
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    if args.command in {"prepare-demo", "all"}:
        prepare_demo(config, names, args.data_dir, args.samples, args.seed)
    if args.command in {"train", "all"}:
        for name in names:
            train_one(name, config["specialists"][name], args.data_dir, args.runs_dir,
                      args.backend, args.seed)
    if args.command in {"assemble", "all"}:
        assemble(config, args.runs_dir)
    if args.command in {"bench", "all"}:
        bench(names, args.data_dir, args.runs_dir, args.iterations)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"insect-poc: {exc}", file=sys.stderr)
        raise SystemExit(2)
