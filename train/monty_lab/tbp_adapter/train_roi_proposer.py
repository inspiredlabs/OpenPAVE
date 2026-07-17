#!/usr/bin/env python3
"""Train the oriented hand-ROI proposer (docs/training-with-monty.md §4 cand. 1-2).

A small full-frame model that proposes the oriented oracle-style ROI the
cold-start landmarker was trained on: centre (16x16 heatmap soft-argmax),
log-size, orientation (unit vector of the +v axis, MCPs toward wrist) and
hand presence. Supervision is derived entirely from the frozen teacher
landmarks via oracle_roi(); no_hand frames supervise presence negatives.

Run in the OpenPAVE arm64 environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.monty_lab.tbp_adapter.oracle_roi import oracle_roi
from train.pixel_sensor.train import DATASETS, evenly_spaced

OUT = ROOT / "train" / "runs" / "monty_landmark_alignment" / "oracle_student"
INPUT = 128
HEAT = 16
TOP_K = 3


def load_split(sources, split, cap, negative_cap, hard_examples=None, hard_repeat=0,
               hard_negatives=None, hard_negative_repeat=0):
    images, centres, sizes, axes, presence = [], [], [], [], []
    for source in sources:
        with np.load(DATASETS / source / "prepared.npz", allow_pickle=True) as d:
            has = np.asarray(d["has_lm"], bool)
            is_val = np.asarray(d["is_val"], bool)
            labels = np.asarray(d["labels"]).astype(str)
            landmarks = np.asarray(d["landmarks"], np.float32).reshape(-1, 21, 2)
            split_mask = ~is_val if split == "train" else is_val
            hand_rows = evenly_spaced(
                np.where(has & (labels != "no_hand") & split_mask)[0], cap)
            if split == "train" and hard_examples and hard_repeat:
                hard = np.asarray(hard_examples.get(source, []), dtype=np.int64)
                eligible = set(np.where(has & (labels != "no_hand") & split_mask)[0])
                hard = np.asarray([r for r in hard if int(r) in eligible], dtype=np.int64)
                if len(hard):
                    hand_rows = np.concatenate(
                        [hand_rows, np.repeat(hard, hard_repeat)])
            neg_rows = evenly_spaced(
                np.where((labels == "no_hand") & split_mask)[0], negative_cap)
            if split == "train" and hard_negatives and hard_negative_repeat:
                hard_neg = np.asarray(hard_negatives.get(source, []), dtype=np.int64)
                eligible_neg = set(np.where((labels == "no_hand") & split_mask)[0])
                hard_neg = np.asarray(
                    [r for r in hard_neg if int(r) in eligible_neg], dtype=np.int64)
                if len(hard_neg):
                    neg_rows = np.concatenate(
                        [neg_rows, np.repeat(hard_neg, hard_negative_repeat)])
            # decompress the image array once; per-row d["imgs"][row] access
            # re-decompresses the whole npz member every time
            imgs = np.asarray(d["imgs"])
            for row in hand_rows:
                try:
                    roi = oracle_roi(landmarks[row])
                except ValueError:
                    continue
                images.append(imgs[row])
                centres.append(roi["center"])
                sizes.append(roi["size"])
                axes.append(roi["y_axis"])
                presence.append(1.0)
            for row in neg_rows:
                images.append(imgs[row])
                centres.append([0.5, 0.5])
                sizes.append(1.0)
                axes.append([0.0, 1.0])
                presence.append(0.0)
            print(f"[data] {split:5s} {source:15s} hands={len(hand_rows)} neg={len(neg_rows)}")
    return (np.asarray(images, np.uint8), np.asarray(centres, np.float32),
            np.asarray(sizes, np.float32), np.asarray(axes, np.float32),
            np.asarray(presence, np.float32))


def build_model():
    import torch
    import torch.nn as nn

    def block(cin, cout, stride):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1),
                             nn.BatchNorm2d(cout), nn.SiLU())

    class RoiProposer(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1 = block(3, 16, 2)    # 64
            self.e2 = block(16, 32, 2)   # 32
            self.e3 = block(32, 48, 2)   # 16
            self.e4 = block(48, 64, 2)   # 8
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
            self.fuse = block(64, 48, 1)  # 16, + skip e3
            self.heat = nn.Conv2d(48, 1, 1)
            self.head = nn.Sequential(nn.Linear(64, 48), nn.SiLU(), nn.Linear(48, 4))
            grid = (torch.arange(HEAT, dtype=torch.float32) + 0.5) / HEAT
            self.register_buffer("grid", grid)

        def forward(self, x):
            f3 = self.e3(self.e2(self.e1(x)))
            f4 = self.e4(f3)
            fused = self.fuse(self.up(f4)) + f3
            logits = self.heat(fused)[:, 0]                  # (B, 16, 16)
            prob = torch.softmax(logits.flatten(1), dim=1).reshape(-1, HEAT, HEAT)
            cx = (prob.sum(dim=1) * self.grid).sum(dim=1)
            cy = (prob.sum(dim=2) * self.grid).sum(dim=1)
            pooled = f4.mean(dim=(2, 3))
            head = self.head(pooled)                         # log_size, dx, dy, presence
            axis = torch.nn.functional.normalize(head[:, 1:3], dim=1)
            return (torch.stack((cx, cy), dim=1), head[:, 0], axis,
                    head[:, 3], logits)

    return RoiProposer()


def heat_target(centres, sigma=1.0):
    cells = (np.arange(HEAT, dtype=np.float32) + 0.5) / HEAT
    dx = centres[:, 0:1] - cells[None, :]
    dy = centres[:, 1:2] - cells[None, :]
    s = sigma / HEAT
    g = (np.exp(-0.5 * (dy / s) ** 2)[:, :, None]
         * np.exp(-0.5 * (dx / s) ** 2)[:, None, :])
    flat = g.reshape(len(centres), -1)
    return (flat / np.maximum(flat.sum(-1, keepdims=True), 1e-12)).astype(np.float32)


def heatmap_peaks(probability, count=TOP_K, suppression_cells=2):
    """Return separated cell centres from a normalized 2-D heatmap."""
    work = np.asarray(probability, np.float64).copy()
    peaks = []
    for _ in range(count):
        flat = int(np.argmax(work))
        y, x = np.unravel_index(flat, work.shape)
        score = float(work[y, x])
        if not np.isfinite(score) or score < 0:
            break
        peaks.append(((x + 0.5) / HEAT, (y + 0.5) / HEAT, score))
        y0, y1 = max(0, y - suppression_cells), min(HEAT, y + suppression_cells + 1)
        x0, x1 = max(0, x - suppression_cells), min(HEAT, x + suppression_cells + 1)
        work[y0:y1, x0:x1] = -np.inf
    return peaks


def evaluate(model, data, device):
    import torch
    images, centres, sizes, axes, presence = data
    pred_c, pred_s, pred_a, pred_p, pred_heat = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for rows in np.array_split(np.arange(len(images)), max(1, len(images) // 512)):
            x = torch.from_numpy(images[rows].astype(np.float32) / 255.0
                                 - 0.5).permute(0, 3, 1, 2).contiguous().to(device)
            c, s, a, p, logits = model(x)
            pred_c.append(c.cpu().numpy()); pred_s.append(s.cpu().numpy())
            pred_a.append(a.cpu().numpy()); pred_p.append(torch.sigmoid(p).cpu().numpy())
            pred_heat.append(torch.softmax(logits.flatten(1), dim=1)
                             .reshape(-1, HEAT, HEAT).cpu().numpy())
    pred_c = np.concatenate(pred_c); pred_s = np.exp(np.concatenate(pred_s))
    pred_a = np.concatenate(pred_a); pred_p = np.concatenate(pred_p)
    pred_heat = np.concatenate(pred_heat)
    hand = presence > 0.5
    centre_err = np.linalg.norm(pred_c[hand] - centres[hand], axis=1) * 384.0
    angle_err = np.degrees(np.arccos(np.clip(
        (pred_a[hand] * axes[hand]).sum(1), -1.0, 1.0)))
    topk_hit, topk_best_error = [], []
    for probability, target, predicted_size in zip(
            pred_heat[hand], centres[hand], pred_s[hand]):
        candidates = np.asarray([peak[:2] for peak in heatmap_peaks(probability)])
        error = np.linalg.norm(candidates - target[None], axis=1)
        best = float(error.min())
        topk_best_error.append(best * 384.0)
        # A candidate is useful when the target ROI centre lies inside its
        # predicted square. Scale is shared by the lightweight global head.
        topk_hit.append(best <= 0.5 * float(predicted_size))
    return {
        "frames": int(len(images)), "hand_frames": int(hand.sum()),
        "centre_error_px_384": {"mean": float(centre_err.mean()),
                                "median": float(np.median(centre_err)),
                                "p95": float(np.percentile(centre_err, 95))},
        "size_ratio": {"median": float(np.median(pred_s[hand] / sizes[hand])),
                       "p95": float(np.percentile(pred_s[hand] / sizes[hand], 95))},
        "angular_error_deg": {"median": float(np.median(angle_err)),
                              "p95": float(np.percentile(angle_err, 95))},
        "presence_recall": float((pred_p[hand] >= 0.5).mean()),
        "false_proposal_rate": float((pred_p[~hand] >= 0.5).mean()) if (~hand).any() else None,
        "topk": TOP_K,
        "topk_oracle_centre_coverage": float(np.mean(topk_hit)),
        "topk_best_centre_error_px_384": {
            "mean": float(np.mean(topk_best_error)),
            "p95": float(np.percentile(topk_best_error, 95)),
        },
    }


def export_onnx(model, out):
    import torch

    class Export(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, image):
            centre, log_size, axis, presence, logits = self.inner(image)
            heatmap = torch.softmax(logits.flatten(1), dim=1).reshape(
                -1, HEAT, HEAT)
            return (centre, torch.exp(log_size), axis,
                    torch.sigmoid(presence), heatmap)

    torch.onnx.export(Export(model.cpu().eval()),
                      (torch.zeros(1, 3, INPUT, INPUT),),
                      str(out), input_names=["image"],
                      output_names=["centre", "size", "axis", "presence",
                                    "centre_heatmap"],
                      dynamic_axes={"image": {0: "n"}, "centre": {0: "n"},
                                    "size": {0: "n"}, "axis": {0: "n"},
                                    "presence": {0: "n"},
                                    "centre_heatmap": {0: "n"}},
                      opset_version=17, dynamo=False)


def main(args=None):
    import torch
    import torch.nn.functional as F

    cfg = parser().parse_args(args)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(max(1, cfg.cpu_threads))
    torch.set_num_interop_threads(1)
    device = "mps" if torch.backends.mps.is_available() and not cfg.cpu_train else "cpu"
    sources = [s for s in cfg.sources.split(",") if s]
    hard_examples, hard_negatives = {}, {}
    if cfg.hard_examples:
        payload = json.loads(cfg.hard_examples.read_text())
        hard_examples = payload.get("sources", payload)
        hard_negatives = payload.get("negative_sources", {})
    train_data = load_split(sources, "train", cfg.train_cap, cfg.negative_cap,
                            hard_examples, cfg.hard_repeat,
                            hard_negatives, cfg.hard_negative_repeat)
    val_data = load_split(sources, "val", cfg.val_cap, cfg.negative_cap // 4)

    model = build_model().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[model] {params} parameters, device={device}")
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=cfg.epochs)

    images, centres, sizes, axes, presence = train_data
    positive_count = int((presence > 0.5).sum())
    negative_count = int((presence <= 0.5).sum())
    # Equal total positive/negative contribution after hard-example repeats.
    # This prevents hard-positive mining from teaching the presence head to
    # emit a hand on almost every frame.
    presence_pos_weight = negative_count / max(positive_count, 1)
    log_sizes = np.log(sizes)
    best_score, best_state = float("inf"), None
    started = time.perf_counter()
    for epoch in range(cfg.epochs):
        order = np.random.permutation(len(images))
        model.train()
        running = 0.0
        for rows in np.array_split(order, max(1, len(order) // cfg.batch)):
            imgs = images[rows].astype(np.float32) / 255.0 - 0.5
            c = centres[rows].copy(); a = axes[rows].copy()
            flip = np.random.random(len(rows)) < 0.5
            imgs[flip] = imgs[flip, :, ::-1]
            c[flip, 0] = 1.0 - c[flip, 0]
            a[flip, 0] = -a[flip, 0]
            xb = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous().to(device)
            xb = xb * (0.85 + 0.30 * torch.rand(len(xb), 1, 1, 1, device=device))
            xb = xb + 0.08 * (torch.rand(len(xb), 1, 1, 1, device=device) - 0.5)
            t_c = torch.from_numpy(c).to(device)
            t_ls = torch.from_numpy(log_sizes[rows]).to(device)
            t_a = torch.from_numpy(a).to(device)
            t_p = torch.from_numpy(presence[rows]).to(device)
            t_heat = torch.from_numpy(heat_target(c)).to(device)
            mask = t_p

            p_c, p_ls, p_a, p_pl, logits = model(xb)
            log_prob = torch.log_softmax(logits.flatten(1), dim=1)
            heat_loss = -((t_heat * log_prob).sum(1) * mask).sum() / mask.sum().clamp(min=1.0)
            centre_loss = (F.smooth_l1_loss(p_c, t_c, beta=0.02, reduction="none").sum(1)
                           * mask).sum() / mask.sum().clamp(min=1.0)
            size_loss = (F.smooth_l1_loss(p_ls, t_ls, beta=0.05, reduction="none")
                         * mask).sum() / mask.sum().clamp(min=1.0)
            axis_loss = ((1.0 - (p_a * t_a).sum(1)) * mask).sum() / mask.sum().clamp(min=1.0)
            presence_bce = F.binary_cross_entropy_with_logits(
                p_pl, t_p,
                pos_weight=torch.as_tensor(presence_pos_weight, device=device),
                reduction="none")
            presence_probability = torch.sigmoid(p_pl)
            presence_pt = (t_p * presence_probability
                           + (1.0 - t_p) * (1.0 - presence_probability))
            presence_loss = (((1.0 - presence_pt) ** cfg.presence_focal_gamma)
                             * presence_bce).mean()
            loss = heat_loss + 2.0 * centre_loss + size_loss + axis_loss + presence_loss
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running += float(loss.detach()) * len(rows)
        schedule.step()
        holdout = evaluate(model, val_data, device)
        score = holdout["centre_error_px_384"]["mean"]
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[train] epoch={epoch + 1}/{cfg.epochs} loss={running / len(images):.4f} "
              f"val_centre_px={score:.2f} val_angle_med="
              f"{holdout['angular_error_deg']['median']:.1f}", flush=True)
    model.load_state_dict(best_state)

    holdout = evaluate(model, val_data, device)
    cfg.out.mkdir(parents=True, exist_ok=True)
    export_onnx(model, cfg.out / "proposer.onnx")
    meta = {
        "contract": "openpave.oriented-roi-proposer.v2",
        "model": "full-frame centre heatmap + log-size + orientation + presence",
        "inference_outputs": {
            "soft_argmax_centre": True,
            "centre_heatmap": [HEAT, HEAT],
            "runtime_topk": TOP_K,
            "peak_suppression_cells": 2,
        },
        "input_px": INPUT, "params": params, "sources": sources,
        "caps": {"train_per_source": cfg.train_cap, "val_per_source": cfg.val_cap,
                 "negatives_per_source": cfg.negative_cap},
        "epochs": cfg.epochs, "seed": cfg.seed, "training_device": device,
        "cpu_threads": cfg.cpu_threads,
        "exploration_holdout": holdout,
        "supervision": "teacher-derived oracle ROIs; no_hand presence negatives",
        "hard_examples": str(cfg.hard_examples) if cfg.hard_examples else None,
        "hard_repeat": cfg.hard_repeat,
        "hard_negative_repeat": cfg.hard_negative_repeat,
        "presence_balance": {
            "positive_examples_after_repeats": positive_count,
            "negative_examples_after_repeats": negative_count,
            "bce_positive_weight": presence_pos_weight,
            "focal_gamma": cfg.presence_focal_gamma,
        },
        "training_seconds": time.perf_counter() - started,
    }
    meta["onnx_sha256"] = hashlib.sha256((cfg.out / "proposer.onnx").read_bytes()).hexdigest()
    (cfg.out / "proposer_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(holdout, indent=2))
    print(f"[roi-proposer] wrote {cfg.out / 'proposer.onnx'}")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", default="crude,hagrid_shapes,jester,ipn")
    p.add_argument("--train-cap", type=int, default=8000)
    p.add_argument("--val-cap", type=int, default=600)
    p.add_argument("--negative-cap", type=int, default=1500)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--seed", type=int, default=37)
    p.add_argument("--cpu-train", action="store_true")
    p.add_argument("--cpu-threads", type=int, default=8)
    p.add_argument("--hard-examples", type=Path)
    p.add_argument("--hard-repeat", type=int, default=3)
    p.add_argument("--hard-negative-repeat", type=int, default=6)
    p.add_argument("--presence-focal-gamma", type=float, default=2.0)
    p.add_argument("--out", type=Path, default=OUT)
    return p


if __name__ == "__main__":
    main()
