"""One training CLI for every backend (responsibility C in §3.2).

The loop is identical across models — collect labeled frames, run the backend
featurizer, fit the head, save weights + manifest — so it lives in one place and is
parameterized by `--backend`. Only the featurizer changes (already in backends.py).

Usage:
    # train DINOv3 probe from a labeled image folder (data/<LABEL>/*.png|jpg)
    python -m pave_mlx.train_intent_head --backend dino --data data

    # prove the training loop with synthetic separable features (no images/model)
    python -m pave_mlx.train_intent_head --backend dino --synthetic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pave_mlx.backends import make_backend
from pave_mlx.heads.base import INTENT_LABELS, PKG_DIR, HeadManifest
from pave_mlx.heads.embedding_probe import EmbeddingProbe

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def _load_dataset(backend, data_dir: Path, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Featurize data/<LABEL>/*.img into (X, y). Labels not in INTENT_LABELS are skipped."""
    from pave_mlx.openai_shim import _decode_image
    import base64

    feats: list[np.ndarray] = []
    ys: list[int] = []
    for label_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        label = label_dir.name.upper()
        if label not in labels:
            print(f"  skip '{label}' (not in intent vocab {labels})")
            continue
        idx = labels.index(label)
        n = 0
        for img_path in sorted(label_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXT:
                continue
            data_url = base64.b64encode(img_path.read_bytes()).decode("ascii")
            image = _decode_image(data_url)
            if image is None:
                continue
            feats.append(backend.embed(image))
            ys.append(idx)
            n += 1
        print(f"  {label}: {n} frames")
    if not feats:
        raise SystemExit(f"no labeled images found under {data_dir} (expect data/<LABEL>/*.png)")
    return np.stack(feats).astype(np.float32), np.asarray(ys, dtype=np.int64)


def _synthetic_dataset(feature_dim: int, labels: list[str], per_class: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Separable Gaussian blobs per label — exercises the loop with no model/images."""
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((len(labels), feature_dim)).astype(np.float32) * 3.0
    X, y = [], []
    for idx in range(len(labels)):
        X.append(centers[idx] + rng.standard_normal((per_class, feature_dim)).astype(np.float32))
        y.extend([idx] * per_class)
    return np.concatenate(X).astype(np.float32), np.asarray(y, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train an OpenPAVE intent head")
    ap.add_argument("--backend", default="dino", choices=["dino", "vjepa", "lingbot"])
    ap.add_argument("--data", default=None, help="labeled image folder: data/<LABEL>/*.png")
    ap.add_argument("--synthetic", action="store_true", help="train on synthetic features")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--config", default=None, help="output manifest path")
    args = ap.parse_args()

    backend = make_backend(args.backend)
    labels = list(INTENT_LABELS)
    feature_dim = backend.feature_dim or 384
    print(f"[train] backend={backend.name} mode={getattr(backend, 'mode', '?')} feature_dim={feature_dim}")

    if args.synthetic:
        X, y = _synthetic_dataset(feature_dim, labels)
        pooling = "synthetic"
    elif args.data:
        X, y = _load_dataset(backend, Path(args.data), labels)
        pooling = "mean_patch"
    else:
        raise SystemExit("provide --data <dir> or --synthetic")

    probe = EmbeddingProbe(feature_dim=X.shape[1], n_labels=len(labels))
    loss = probe.fit(X, y, steps=args.steps, lr=args.lr)
    acc = float((np.argmax(probe.logits(X), axis=1) == y).mean())
    print(f"[train] samples={len(X)} final_loss={loss:.4f} train_acc={acc:.3f}")

    weights_rel = f"heads/weights/{backend.name}.npz"
    probe.save(PKG_DIR / weights_rel)
    manifest = HeadManifest(
        backend=backend.name,
        model_id=getattr(backend, "model_id", backend.name),
        feature_dim=X.shape[1],
        pooling=pooling,
        weights=weights_rel,
        labels=labels,
        trained=True,
    )
    cfg_path = Path(args.config) if args.config else (PKG_DIR / "heads" / "configs" / f"{backend.name}.json")
    manifest.save(cfg_path)
    print(f"[train] saved weights -> {PKG_DIR / weights_rel}")
    print(f"[train] saved manifest -> {cfg_path} (trained=true)")


if __name__ == "__main__":
    main()
