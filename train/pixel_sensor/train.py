#!/usr/bin/env python3
"""Train and judge a trunk-initialised, 2D-only patch landmark refiner."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "train"
DATASETS = TRAIN / "datasets"
TRUNK = TRAIN / "runs" / "tiny_gesture" / "trunk.onnx"
OUT = TRAIN / "runs" / "pixel_sensor"
PATCH = 48
RADIUS = PATCH / 2.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def evenly_spaced(rows: np.ndarray, cap: int) -> np.ndarray:
    if cap <= 0 or len(rows) <= cap:
        return rows
    return rows[np.linspace(0, len(rows) - 1, cap, dtype=np.int64)]


@dataclass
class Frames:
    images: np.ndarray
    truth: np.ndarray
    initial: np.ndarray
    sources: np.ndarray
    indices: np.ndarray


def trunk_predict(images: np.ndarray, session, batch: int = 256) -> np.ndarray:
    result = []
    for start in range(0, len(images), batch):
        x = images[start:start + batch].astype(np.float32) / 255.0 - 0.5
        x = np.transpose(x, (0, 3, 1, 2))
        result.append(session.run(["landmarks"], {"frames": x})[0].reshape(-1, 21, 2))
    return np.concatenate(result).astype(np.float32)


def load_frames(sources: list[str], split: str, cap: int, trunk_session) -> Frames:
    images, truth, names, indices = [], [], [], []
    for source in sources:
        path = DATASETS / source / "prepared.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=True) as d:
            has = np.asarray(d["has_lm"], bool)
            is_val = np.asarray(d["is_val"], bool)
            if split == "train":
                rows = np.where(has & ~is_val)[0]
            elif split == "val":
                rows = np.where(has & is_val)[0]
            elif split == "all":
                rows = np.where(has)[0]
            else:
                raise ValueError(split)
            rows = evenly_spaced(rows, cap)
            images.append(np.asarray(d["imgs"])[rows])
            truth.append(np.asarray(d["landmarks"], np.float32)[rows].reshape(-1, 21, 2))
            names.extend([source] * len(rows)); indices.extend(rows.tolist())
            print(f"[data] {split:5s} {source:15s} {len(rows):5d}")
    all_images = np.concatenate(images)
    return Frames(all_images, np.concatenate(truth), trunk_predict(all_images, trunk_session),
                  np.asarray(names), np.asarray(indices, np.int64))


def crop_patch(image: np.ndarray, centre_px: np.ndarray) -> np.ndarray:
    import cv2
    pad = PATCH
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    centre = (float(centre_px[0] + pad), float(centre_px[1] + pad))
    return cv2.getRectSubPix(padded, (PATCH, PATCH), centre)


def visible(points: np.ndarray) -> np.ndarray:
    return np.isfinite(points).all(-1) & (points >= 0).all(-1) & (points <= 1).all(-1)


def make_samples(frames: Frames, per_frame: int, seed: int,
                 negative_fraction: float = 0.25) -> tuple[np.ndarray, ...]:
    """Create deterministic patches; invalid joints are skipped, never frames."""
    rng = np.random.default_rng(seed)
    patches, joint_ids, offsets, matches = [], [], [], []
    for image, truth, initial in zip(frames.images, frames.truth, frames.initial):
        valid = np.where(visible(truth))[0]
        if not len(valid):
            continue
        for _ in range(per_frame):
            joint = int(rng.choice(valid)); true_px = truth[joint] * image.shape[0]
            make_negative = rng.random() < negative_fraction
            if make_negative:
                other = valid[valid != joint]
                if len(other) and rng.random() < 0.65:
                    centre = truth[int(rng.choice(other))] * image.shape[0]
                else:
                    centre = rng.uniform(-0.1, 1.1, size=2) * image.shape[0]
                # A negative accidentally close to the requested joint is not
                # mislabeled; push it to the opposite side of the patch.
                if np.max(np.abs(true_px - centre)) < RADIUS * 0.85:
                    centre = true_px + rng.choice([-1.0, 1.0], size=2) * RADIUS * 1.1
                match = 0.0
            else:
                if rng.random() < 0.70:
                    centre = initial[joint] * image.shape[0] + rng.normal(0, 2.0, size=2)
                else:
                    centre = true_px + rng.uniform(-RADIUS * 0.7, RADIUS * 0.7, size=2)
                match = float(np.max(np.abs(true_px - centre)) <= RADIUS * 0.85)
            delta = np.clip((true_px - centre) / RADIUS, -1.0, 1.0)
            patches.append(crop_patch(image, centre)); joint_ids.append(joint)
            offsets.append(delta); matches.append(match)
    return (np.asarray(patches, np.uint8), np.asarray(joint_ids, np.int64),
            np.asarray(offsets, np.float32), np.asarray(matches, np.float32))


def build_model():
    import torch
    import torch.nn as nn

    class PatchNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 12, 3, 2, 1), nn.BatchNorm2d(12), nn.SiLU(),
                nn.Conv2d(12, 20, 3, 2, 1), nn.BatchNorm2d(20), nn.SiLU(),
                nn.Conv2d(20, 24, 3, 2, 1), nn.BatchNorm2d(24), nn.SiLU())
            self.joint = nn.Embedding(21, 8)
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(24 * 6 * 6 + 8, 32),
                                      nn.SiLU(), nn.Linear(32, 3))

        def forward(self, patches, joint_id):
            features = self.features(patches)
            joined = torch.cat((features.reshape(features.shape[0], -1),
                                self.joint(joint_id)), 1).contiguous()
            output = self.head[1:](joined)
            return torch.tanh(output[:, :2]).contiguous(), output[:, 2].contiguous()

    return PatchNet()


def tensors(samples: tuple[np.ndarray, ...]):
    import torch
    patches, joints, offsets, matches = samples
    x = torch.from_numpy(patches.astype(np.float32) / 255.0 - 0.5).permute(0, 3, 1, 2)
    return x, torch.from_numpy(joints), torch.from_numpy(offsets), torch.from_numpy(matches)


def validation_loss(model, samples, device: str, batch: int = 512) -> float:
    import torch
    import torch.nn.functional as F
    x, joints, offsets, matches = tensors(samples)
    total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for rows in torch.split(torch.arange(len(x)), batch):
            delta, logit = model(x[rows].to(device), joints[rows].to(device))
            m = matches[rows].to(device)
            loss = F.binary_cross_entropy_with_logits(logit, m)
            if bool((m > 0.5).any()):
                loss += 2.0 * F.smooth_l1_loss(delta[m > 0.5], offsets[rows].to(device)[m > 0.5])
            total += float(loss) * len(rows); count += len(rows)
    return total / max(count, 1)


def infer_offsets(model, frames: Frames, device: str,
                  batch: int = 1024) -> tuple[np.ndarray, np.ndarray, float]:
    import torch
    patches, joints, owners = [], [], []
    t0 = time.perf_counter()
    for owner, (image, initial) in enumerate(zip(frames.images, frames.initial)):
        for joint in range(21):
            patches.append(crop_patch(image, initial[joint] * image.shape[0]))
            joints.append(joint); owners.append(owner)
    crop_ms = (time.perf_counter() - t0) * 1000
    x = torch.from_numpy(np.asarray(patches, np.float32) / 255.0 - 0.5).permute(0, 3, 1, 2)
    j = torch.tensor(joints, dtype=torch.long)
    deltas, confidence = [], []
    model.eval(); t0 = time.perf_counter()
    with torch.no_grad():
        for rows in torch.split(torch.arange(len(x)), batch):
            d, logit = model(x[rows].to(device), j[rows].to(device))
            deltas.append(d.cpu().numpy()); confidence.append(torch.sigmoid(logit).cpu().numpy())
    model_ms = (time.perf_counter() - t0) * 1000
    deltas = np.concatenate(deltas).reshape(len(frames.images), 21, 2)
    confidence = np.concatenate(confidence).reshape(len(frames.images), 21)
    return deltas, confidence, (crop_ms + model_ms) / max(len(frames.images), 1)


def apply_offsets(frames: Frames, deltas: np.ndarray, confidence: np.ndarray,
                  threshold: float) -> np.ndarray:
    corrected = frames.initial.copy()
    use = confidence >= threshold
    corrected[use] += deltas[use] * (RADIUS / frames.images.shape[1])
    return corrected


def landmark_metrics(prediction: np.ndarray, frames: Frames) -> dict:
    mask = visible(frames.truth)
    error_px = np.linalg.norm(prediction - frames.truth, axis=-1)[mask] * 384.0
    return {"joints": int(mask.sum()), "mean_px_384": float(error_px.mean()),
            "median_px_384": float(np.median(error_px)),
            "pck_5px": float((error_px <= 5.0).mean()),
            "pck_10px": float((error_px <= 10.0).mean()),
            "p95_px_384": float(np.percentile(error_px, 95))}


def evaluate(frames: Frames, threshold: float, inference) -> tuple[dict, np.ndarray]:
    deltas, confidence, latency = inference
    refined = apply_offsets(frames, deltas, confidence, threshold)
    baseline = landmark_metrics(frames.initial, frames)
    result = {"frames": int(len(frames.images)), "baseline": baseline,
              "refined": landmark_metrics(refined, frames),
              "accepted_joint_fraction": float((confidence >= threshold).mean()),
              "batched_eval_ms_per_frame": float(latency)}
    result["mean_error_improvement"] = float(
        (baseline["mean_px_384"] - result["refined"]["mean_px_384"])
        / max(baseline["mean_px_384"], 1e-9))
    return result, refined


def export_onnx(model, out: Path) -> None:
    import torch

    class Export(torch.nn.Module):
        def __init__(self, inner):
            super().__init__(); self.inner = inner

        def forward(self, patches, joint_id):
            delta, logit = self.inner(patches, joint_id)
            return delta, torch.sigmoid(logit)

    torch.onnx.export(Export(model.cpu().eval()),
                      (torch.zeros(1, 3, PATCH, PATCH), torch.zeros(1, dtype=torch.long)),
                      str(out), input_names=["patches", "joint_id"],
                      output_names=["delta", "match_probability"],
                      dynamic_axes={"patches": {0: "n"}, "joint_id": {0: "n"},
                                    "delta": {0: "n"}, "match_probability": {0: "n"}},
                      opset_version=17, dynamo=False)


def cmd_train(args) -> None:
    import onnxruntime as ort
    import torch
    import torch.nn.functional as F

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() and not args.cpu_train else "cpu"
    trunk = ort.InferenceSession(str(TRUNK), providers=["CPUExecutionProvider"])
    exploration = args.sources.split(",")
    train_frames = load_frames(exploration, "train", args.train_cap, trunk)
    val_frames = load_frames(exploration, "val", args.val_cap, trunk)
    referee_frames = load_frames([args.referee], "all", args.referee_cap, trunk)
    val_samples = make_samples(val_frames, 2, args.seed + 10_000)
    model = build_model().to(device)
    params = sum(p.numel() for p in model.parameters())
    if params > 50_000:
        raise SystemExit(f"parameter budget exceeded: {params}")
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_loss, best_state = float("inf"), None
    started = time.perf_counter()
    for epoch in range(args.epochs):
        samples = make_samples(train_frames, args.samples_per_frame, args.seed + epoch)
        x, joints, offsets, matches = tensors(samples)
        order = torch.randperm(len(x)); model.train(); running = 0.0
        for rows in torch.split(order, args.batch):
            xb = x[rows].to(device).contiguous(); jb = joints[rows].to(device).contiguous()
            ob = offsets[rows].to(device).contiguous(); mb = matches[rows].to(device).contiguous()
            if torch.rand(()) < 0.5:
                xb = torch.flip(xb, dims=[3]).contiguous()
                ob = ob.clone(); ob[:, 0] *= -1; ob = ob.contiguous()
            xb = xb * (0.85 + 0.30 * torch.rand(len(xb), 1, 1, 1, device=device))
            delta, logit = model(xb, jb)
            loss = F.binary_cross_entropy_with_logits(logit, mb)
            if bool((mb > 0.5).any()):
                loss += 2.0 * F.smooth_l1_loss(delta[mb > 0.5], ob[mb > 0.5])
            optimiser.zero_grad(); loss.backward(); optimiser.step()
            running += float(loss.detach()) * len(rows)
        score = validation_loss(model, val_samples, device)
        if score < best_loss:
            best_loss = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[train] epoch={epoch + 1}/{args.epochs} loss={running/len(x):.4f} val={score:.4f}", flush=True)
    model.load_state_dict(best_state)

    # Select one global confidence threshold on exploration holdout only.
    threshold_results = []
    holdout_inference = infer_offsets(model, val_frames, device)
    for threshold in np.linspace(0.25, 0.75, 11):
        metrics, _ = evaluate(val_frames, float(threshold), holdout_inference)
        threshold_results.append((metrics["refined"]["mean_px_384"], float(threshold), metrics))
    _, threshold, holdout = min(threshold_results, key=lambda row: row[0])
    referee, _ = evaluate(referee_frames, threshold,
                          infer_offsets(model, referee_frames, device))

    OUT.mkdir(parents=True, exist_ok=True)
    export_onnx(model, OUT / "patch_refiner.onnx")
    accepted = (holdout["mean_error_improvement"] > 0
                and referee["mean_error_improvement"] > 0)
    meta = {
        "contract": "openpave.pixel-sensor-stage1.v1", "dimensionality": "2d-only",
        "z_source": "prior", "metric_3d": False, "initialiser": str(TRUNK.relative_to(ROOT)),
        "initialiser_sha256": sha256(TRUNK), "patch_px": PATCH, "params": params,
        "precision": "fp32", "training_device": device, "epochs": args.epochs,
        "sources": exploration, "referee": args.referee,
        "caps": {"train_per_source": args.train_cap, "val_per_source": args.val_cap,
                 "referee": args.referee_cap},
        "confidence_threshold": threshold, "validation_loss": best_loss,
        "exploration_holdout": holdout, "untouched_referee": referee,
        "accepted": accepted, "acceptance_rule": "mean landmark error improves over v3 trunk on both splits",
        "runtime_promotion": False,
        "runtime_blocker": "Requires independent tail-error and end-to-end latency gates after benchmarking.",
        "training_seconds": time.perf_counter() - started,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    print(f"[pixel-sensor] {'ACCEPTED' if accepted else 'REJECTED'} -> {OUT}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", default="crude,hagrid_shapes,jester,ipn")
    p.add_argument("--referee", default="yolo26")
    p.add_argument("--train-cap", type=int, default=3000)
    p.add_argument("--val-cap", type=int, default=600)
    p.add_argument("--referee-cap", type=int, default=0)
    p.add_argument("--samples-per-frame", type=int, default=2)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=37)
    p.add_argument("--cpu-train", action="store_true")
    p.set_defaults(fn=cmd_train)
    return p


if __name__ == "__main__":
    cmd_train(parser().parse_args())
