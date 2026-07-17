#!/usr/bin/env python3
"""Train the acquisition-matched ROI landmarker (training-with-monty.md).

The frozen proposer is run over the actual training frames and its ROIs form
the primary training distribution.  A minority of oracle ROIs preserves the
capability ceiling.  Oracle and proposed-ROI evaluation columns are both
written to meta.json and checkpoint selection uses proposed-ROI validation.
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

from train.monty_lab.tbp_adapter.oracle_roi import (
    crop as roi_crop,
    oracle_roi,
    project_to_roi,
    project_to_source,
)
from train.pixel_sensor.train import DATASETS, evenly_spaced

OUT = ROOT / "train" / "runs" / "monty_landmark_alignment" / "oracle_student"
DEFAULT_PROPOSER = OUT / "proposer.onnx"
CROP = 96
HEAT = 24
PCK_GOOD = 0.05  # confidence target: joint localized within 5% of ROI size
PARENT = np.asarray([
    9, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19,
], dtype=np.int64)
FINGERTIPS = np.asarray([4, 8, 12, 16, 20], dtype=np.int64)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_split(sources, split, cap):
    images, truth, names, indices = [], [], [], []
    for source in sources:
        with np.load(DATASETS / source / "prepared.npz", allow_pickle=True) as d:
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
            names.extend([source] * len(rows))
            indices.extend(rows.tolist())
            print(f"[data] {split:5s} {source:15s} {len(rows):5d}")
    return (np.concatenate(images), np.concatenate(truth),
            np.asarray(names), np.asarray(indices, np.int64))


def _heatmap_centres(probability, count=3, suppression_cells=2):
    work = np.asarray(probability, np.float64).copy()
    height, width = work.shape
    centres = []
    for _ in range(count):
        y, x = np.unravel_index(int(np.argmax(work)), work.shape)
        if not np.isfinite(work[y, x]):
            break
        centres.append(np.asarray([(x + 0.5) / width, (y + 0.5) / height]))
        work[max(0, y - suppression_cells):min(height, y + suppression_cells + 1),
             max(0, x - suppression_cells):min(width, x + suppression_cells + 1)] = -np.inf
    return centres


def infer_proposer_rois(images, proposer_path=DEFAULT_PROPOSER, batch=512,
                        cpu_threads=4, return_hypotheses=False):
    """Freeze the deployment proposer into one deterministic ROI per frame."""
    import cv2
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, int(cpu_threads))
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(proposer_path), sess_options=options, providers=["CPUExecutionProvider"])
    rois, presences, hypotheses = [], [], []
    for start in range(0, len(images), batch):
        chunk = images[start:start + batch]
        resized = np.stack([cv2.resize(im, (128, 128), interpolation=cv2.INTER_AREA)
                            for im in chunk])
        x = (resized.astype(np.float32) / 255.0 - 0.5).transpose(0, 3, 1, 2)
        outputs = session.run(None, {"image": x})
        centre, size, axis, presence = outputs[:4]
        heatmaps = outputs[4] if len(outputs) >= 5 else [None] * len(centre)
        for c, s, y, p, heatmap in zip(centre, size, axis, presence, heatmaps):
            y = np.asarray(y, np.float64)
            norm = float(np.linalg.norm(y))
            y = y / norm if norm > 1e-6 else np.asarray([0.0, 1.0])
            centres = (_heatmap_centres(heatmap) if heatmap is not None
                       else [np.asarray(c, np.float64)])
            candidate_rois = [{
                "center": np.asarray(candidate, np.float64),
                "x_axis": np.asarray([y[1], -y[0]], np.float64),
                "y_axis": y,
                "size": float(np.clip(np.asarray(s).reshape(-1)[0], 0.05, 1.5)),
            } for candidate in centres]
            rois.append(candidate_rois[0])
            hypotheses.append(candidate_rois)
            presences.append(float(np.asarray(p).reshape(-1)[0]))
    result = (rois, np.asarray(presences, np.float32))
    return result + (hypotheses,) if return_hypotheses else result


def infer_acquisition_rois(images, model_dir, proposer_path,
                           thermal_duty_cycle=1.0):
    """Freeze the exact multi-scale/re-inspection crop selected at runtime."""
    from train.monty_lab.tbp_adapter.oracle_runtime import OracleLandmarkerRuntime

    runtime = OracleLandmarkerRuntime(
        model_dir=model_dir, proposer_path=proposer_path)
    rois, presences, qualities = [], [], []
    throttle_started = time.perf_counter()
    for processed, image in enumerate(images, 1):
        roi, presence, quality = runtime.trace_acquisition_roi(image)
        rois.append(roi)
        presences.append(presence)
        qualities.append(quality)
        if (thermal_duty_cycle < 1.0 and processed % 100 == 0):
            work = time.perf_counter() - throttle_started
            time.sleep(work * (1.0 - thermal_duty_cycle) / thermal_duty_cycle)
            throttle_started = time.perf_counter()
    return (rois, np.asarray(presences, np.float32),
            np.asarray(qualities, np.float32))


def make_crops(images, truth, rng=None, flip_fraction=0.5,
               proposed_rois=None, oracle_fraction=0.0,
               raw_proposed_rois=None, raw_proposer_fraction=0.0):
    """Crops and targets from the real proposer with an optional oracle mix.

    ``proposed_rois=None`` preserves the pure oracle evaluation contract.
    When proposals are supplied, every non-oracle sample uses the proposer's
    exact frozen output; no synthetic translation tail is substituted for it.
    """
    crops, targets, visible_mask, rois, keep, kinds = [], [], [], [], [], []
    for row, (image, points) in enumerate(zip(images, truth)):
        try:
            oracle = oracle_roi(points)
        except ValueError:
            continue
        use_oracle = proposed_rois is None or (
            rng is not None and rng.random() < float(oracle_fraction))
        use_raw = (not use_oracle and raw_proposed_rois is not None
                   and rng is not None
                   and rng.random() < float(raw_proposer_fraction))
        if use_oracle:
            roi = oracle
        elif use_raw:
            candidates = raw_proposed_rois[row]
            if isinstance(candidates, (list, tuple)):
                roi = candidates[int(rng.integers(0, len(candidates)))]
            else:
                roi = candidates
        else:
            roi = proposed_rois[row]
        patch = roi_crop(image, roi, CROP)
        uv = project_to_roi(points, roi).astype(np.float32)
        vis = (np.isfinite(uv).all(-1) & (uv >= 0.0).all(-1) & (uv <= 1.0).all(-1))
        if rng is not None and rng.random() < flip_fraction:
            patch = patch[:, ::-1].copy()
            uv = uv.copy()
            uv[:, 0] = 1.0 - uv[:, 0]
        crops.append(patch)
        targets.append(np.nan_to_num(uv, nan=0.5))
        visible_mask.append(vis)
        rois.append(roi)
        keep.append(row)
        kinds.append("oracle" if use_oracle else
                     "raw_proposer" if use_raw else "runtime_selected")
    return (np.asarray(crops, np.uint8), np.asarray(targets, np.float32),
            np.asarray(visible_mask, bool), rois, np.asarray(keep, np.int64),
            np.asarray(kinds))


def build_model():
    import torch
    import torch.nn as nn

    def block(cin, cout, stride):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1),
                             nn.BatchNorm2d(cout), nn.SiLU())

    class RoiLandmarker(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1 = block(3, 16, 2)    # 48
            self.e2 = block(16, 32, 2)   # 24
            self.e3 = block(32, 48, 2)   # 12
            self.e4 = block(48, 64, 2)   # 6
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
            self.d1 = block(64, 48, 1)   # 12
            self.d2 = block(48, 32, 1)   # 24
            self.heat = nn.Conv2d(32, 21, 1)
            self.conf = nn.Sequential(nn.Linear(64, 48), nn.SiLU(), nn.Linear(48, 21))
            grid = (torch.arange(HEAT, dtype=torch.float32) + 0.5) / HEAT
            self.register_buffer("grid", grid)

        def forward(self, x):
            f1 = self.e1(x)
            f2 = self.e2(f1)
            f3 = self.e3(f2)
            f4 = self.e4(f3)
            d = self.d1(self.up(f4)) + f3
            d = self.d2(self.up(d)) + f2
            logits = self.heat(d)                              # (B, 21, 24, 24)
            prob = torch.softmax(logits.flatten(2), dim=2)
            prob_map = prob.reshape(-1, 21, HEAT, HEAT)
            u = (prob_map.sum(dim=2) * self.grid).sum(dim=2)
            v = (prob_map.sum(dim=3) * self.grid).sum(dim=2)
            coords = torch.stack((u, v), dim=2)                # (B, 21, 2)
            conf_logits = self.conf(f4.mean(dim=(2, 3)))       # (B, 21)
            return coords, conf_logits, logits

    return RoiLandmarker()


def heatmap_targets(uv, sigma=1.0):
    """Gaussian target distributions over the 24x24 grid, (N, 21, HEAT*HEAT)."""
    centers = (np.arange(HEAT, dtype=np.float32) + 0.5) / HEAT
    du = uv[..., 0:1] - centers[None, None, :]
    dv = uv[..., 1:2] - centers[None, None, :]
    cell_sigma = sigma / HEAT
    gu = np.exp(-0.5 * (du / cell_sigma) ** 2)
    gv = np.exp(-0.5 * (dv / cell_sigma) ** 2)
    grid = gv[..., :, None] * gu[..., None, :]
    flat = grid.reshape(*uv.shape[:-1], HEAT * HEAT)
    total = flat.sum(-1, keepdims=True)
    return (flat / np.maximum(total, 1e-12)).astype(np.float32)


def losses(model_out, target_uv, visible, joint_weight, heat_target):
    import torch
    import torch.nn.functional as F

    coords, conf_logits, logits = model_out
    mask = visible.float()
    weights = mask * joint_weight

    coord = F.smooth_l1_loss(coords, target_uv, beta=0.02, reduction="none").sum(-1)
    coord_loss = (coord * weights).sum() / weights.sum().clamp(min=1.0)

    log_prob = torch.log_softmax(logits.flatten(2), dim=2)
    heat_loss = -((heat_target * log_prob).sum(-1) * weights).sum() / weights.sum().clamp(min=1.0)

    with torch.no_grad():
        error = (coords - target_uv).norm(dim=-1)
        conf_target = (visible & (error < PCK_GOOD)).float()
    conf_loss = F.binary_cross_entropy_with_logits(conf_logits, conf_target)

    parent = torch.as_tensor(PARENT, device=coords.device)
    bone_pred = (coords - coords[:, parent]).norm(dim=-1)
    bone_true = (target_uv - target_uv[:, parent]).norm(dim=-1)
    bone_mask = mask * mask[:, parent]
    bone_loss = (F.smooth_l1_loss(bone_pred, bone_true, beta=0.02, reduction="none")
                 * bone_mask).sum() / bone_mask.sum().clamp(min=1.0)

    return 2.0 * coord_loss + heat_loss + 0.5 * conf_loss + 0.5 * bone_loss


def infer(model, crops, device, batch=512):
    import torch
    outputs, confidences = [], []
    model.eval()
    x_all = torch.from_numpy(
        crops.astype(np.float32) / 255.0 - 0.5).permute(0, 3, 1, 2).contiguous()
    with torch.no_grad():
        for rows in torch.split(torch.arange(len(x_all)), batch):
            coords, conf_logits, _ = model(x_all[rows].to(device))
            outputs.append(coords.cpu().numpy())
            confidences.append(torch.sigmoid(conf_logits).cpu().numpy())
    return np.concatenate(outputs), np.concatenate(confidences)


def roi_metrics(pred_uv, target_uv, visible):
    error = np.linalg.norm(pred_uv - target_uv, axis=-1)[visible]
    return {
        "joints": int(visible.sum()),
        "mean_roi": float(error.mean()),
        "median_roi": float(np.median(error)),
        "p95_roi": float(np.percentile(error, 95)),
        "pck_5pct_roi": float((error <= 0.05).mean()),
        "pck_10pct_roi": float((error <= 0.10).mean()),
    }


def source_metrics(pred_uv, rois, truth, keep, image_size=384.0):
    """Project ROI predictions back to source coordinates; px at 384 like the doc."""
    errors, per_joint = [], np.full((21,), np.nan)
    all_errors = np.full((len(keep), 21), np.nan, dtype=np.float64)
    for i, (uv, roi, row) in enumerate(zip(pred_uv, rois, keep)):
        source = project_to_source(uv, roi)
        target = truth[row]
        valid = np.isfinite(target).all(-1)
        error = np.linalg.norm(source - target, axis=-1) * image_size
        all_errors[i, valid] = error[valid]
    flat = all_errors[np.isfinite(all_errors)]
    return {
        "joints": int(len(flat)),
        "mean_px_384": float(flat.mean()),
        "median_px_384": float(np.median(flat)),
        "p95_px_384": float(np.percentile(flat, 95)),
        "pck_5px": float((flat <= 5.0).mean()),
        "pck_10px": float((flat <= 10.0).mean()),
        "per_joint_mean_px_384": [
            float(np.nanmean(all_errors[:, j])) if np.isfinite(all_errors[:, j]).any()
            else None for j in range(21)],
    }


def evaluate(model, images, truth, device, proposed_rois=None):
    crops, target_uv, visible, rois, keep, _ = make_crops(
        images, truth, rng=None, proposed_rois=proposed_rois, oracle_fraction=0.0)
    pred_uv, confidence = infer(model, crops, device)
    result = {
        "frames": int(len(keep)),
        "skipped_degenerate_roi": int(len(images) - len(keep)),
        "roi_local": roi_metrics(pred_uv, target_uv, visible),
        "source_frame": source_metrics(pred_uv, rois, truth, keep),
        "mean_confidence": float(confidence.mean()),
    }
    return result, pred_uv, target_uv, visible, confidence


def export_onnx(model, out: Path):
    import torch

    class Export(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, crops):
            coords, conf_logits, _ = self.inner(crops)
            return coords, torch.sigmoid(conf_logits)

    torch.onnx.export(Export(model.cpu().eval()),
                      (torch.zeros(1, 3, CROP, CROP),),
                      str(out), input_names=["crops"],
                      output_names=["landmarks_uv", "confidence"],
                      dynamic_axes={"crops": {0: "n"}, "landmarks_uv": {0: "n"},
                                    "confidence": {0: "n"}},
                      opset_version=17, dynamo=False)


def select_threshold(pred_uv, target_uv, visible, confidence):
    """Confidence threshold maximizing F1 of 'joint localized within 5% ROI'."""
    error = np.linalg.norm(pred_uv - target_uv, axis=-1)
    good = visible & (error < PCK_GOOD)
    best = (0.0, 0.25)
    for threshold in np.linspace(0.1, 0.9, 33):
        accepted = confidence >= threshold
        tp = float((accepted & good).sum())
        precision = tp / max(float(accepted.sum()), 1.0)
        recall = tp / max(float(good.sum()), 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        if f1 > best[0]:
            best = (f1, float(threshold))
    return best[1], best[0]


def main(args=None):
    import torch

    cfg = parser().parse_args(args)
    if not 0.0 <= cfg.oracle_mix <= 1.0:
        raise SystemExit("--oracle-mix must be between 0 and 1")
    if not 0.0 <= cfg.raw_proposer_mix <= 1.0:
        raise SystemExit("--raw-proposer-mix must be between 0 and 1")
    if not 0.0 < cfg.thermal_duty_cycle <= 1.0:
        raise SystemExit("--thermal-duty-cycle must be in (0, 1]")
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(max(1, cfg.cpu_threads))
    torch.set_num_interop_threads(1)
    device = "mps" if torch.backends.mps.is_available() and not cfg.cpu_train else "cpu"
    sources = [s for s in cfg.sources.split(",") if s]
    auxiliary_sources = {s for s in cfg.auxiliary_sources.split(",") if s}
    unknown_auxiliary = auxiliary_sources.difference(sources)
    if unknown_auxiliary:
        raise SystemExit(
            "--auxiliary-sources must be present in --sources: "
            + ",".join(sorted(unknown_auxiliary))
        )
    validation_sources = [s for s in sources if s not in auxiliary_sources]
    if not validation_sources:
        raise SystemExit("At least one non-auxiliary deployment source is required")

    train_images, train_truth, train_names, _ = load_split(
        sources, "train", cfg.train_cap
    )
    val_images, val_truth, val_names, _ = load_split(
        validation_sources, "val", cfg.val_cap
    )
    referee_images, referee_truth, _, _ = load_split([cfg.referee], "all", cfg.referee_cap)
    if not cfg.proposer.exists():
        raise SystemExit(f"frozen proposer missing: {cfg.proposer}")
    print(f"[proposer] freezing real training/validation crops from {cfg.proposer}")
    raw_train_rois, train_presence, raw_train_hypotheses = infer_proposer_rois(
        train_images, cfg.proposer, cpu_threads=cfg.cpu_threads,
        return_hypotheses=True)
    raw_val_rois, _ = infer_proposer_rois(
        val_images, cfg.proposer, cpu_threads=cfg.cpu_threads)
    raw_referee_rois, _ = infer_proposer_rois(
        referee_images, cfg.proposer, cpu_threads=cfg.cpu_threads)
    if cfg.acquisition_model_dir:
        print(f"[acquisition] tracing live crop selection with {cfg.acquisition_model_dir}")
        train_rois, _, train_trace_quality = infer_acquisition_rois(
            train_images, cfg.acquisition_model_dir, cfg.proposer,
            cfg.thermal_duty_cycle)
        val_rois, _, _ = infer_acquisition_rois(
            val_images, cfg.acquisition_model_dir, cfg.proposer,
            cfg.thermal_duty_cycle)
        referee_rois, _, _ = infer_acquisition_rois(
            referee_images, cfg.acquisition_model_dir, cfg.proposer,
            cfg.thermal_duty_cycle)
    else:
        train_rois, val_rois, referee_rois = (
            raw_train_rois, raw_val_rois, raw_referee_rois)
        train_trace_quality = None

    model = build_model().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[model] {params} parameters, device={device}")
    joint_weight = torch.ones(21, device=device)
    joint_weight[torch.as_tensor(FINGERTIPS, device=device)] = 1.5

    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=cfg.epochs)
    best_score, best_state = float("inf"), None
    started = time.perf_counter()
    for epoch in range(cfg.epochs):
        rng = np.random.default_rng(cfg.seed + epoch)
        crops, target_uv, visible, _, _, crop_kinds = make_crops(
            train_images, train_truth, rng=rng, proposed_rois=train_rois,
            oracle_fraction=cfg.oracle_mix,
            raw_proposed_rois=(raw_train_hypotheses
                               if cfg.acquisition_model_dir else None),
            raw_proposer_fraction=cfg.raw_proposer_mix)
        order = torch.randperm(len(crops))
        model.train()
        running = 0.0
        for rows in torch.split(order, cfg.batch):
            idx = rows.numpy()
            # uint8 crops and Gaussian heatmap targets are materialized per
            # batch: the full-epoch float tensors would exceed 2 GB.
            xb = torch.from_numpy(
                crops[idx].astype(np.float32) / 255.0
                - 0.5).permute(0, 3, 1, 2).contiguous().to(device)
            # photometric jitter: brightness/contrast, matching webcam variance
            xb = xb * (0.85 + 0.30 * torch.rand(len(xb), 1, 1, 1, device=device))
            xb = xb + 0.08 * (torch.rand(len(xb), 1, 1, 1, device=device) - 0.5)
            loss = losses(model(xb),
                          torch.from_numpy(target_uv[idx]).to(device),
                          torch.from_numpy(visible[idx]).to(device),
                          joint_weight,
                          torch.from_numpy(heatmap_targets(target_uv[idx])).to(device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running += float(loss.detach()) * len(rows)
        schedule.step()
        holdout, *_ = evaluate(model, val_images, val_truth, device,
                               proposed_rois=val_rois)
        # Source-frame error includes every labelled joint, including joints
        # that the proposer cropped out. Selecting only visible ROI-local
        # joints would reward a bad acquisition for hiding difficult targets.
        score = holdout["source_frame"]["mean_px_384"]
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[train] epoch={epoch + 1}/{cfg.epochs} loss={running / len(crops):.4f} "
              f"proposed_mean_px_384={score:.2f} "
              f"proposed_pck5={holdout['roi_local']['pck_5pct_roi']:.3f} "
              f"mix={dict(zip(*np.unique(crop_kinds, return_counts=True)))}",
              flush=True)
    model.load_state_dict(best_state)

    holdout_proposed, pred_uv, target_uv, visible, confidence = evaluate(
        model, val_images, val_truth, device, proposed_rois=val_rois)
    threshold, threshold_f1 = select_threshold(pred_uv, target_uv, visible, confidence)
    holdout_oracle, *_ = evaluate(model, val_images, val_truth, device)
    referee_proposed, *_ = evaluate(model, referee_images, referee_truth, device,
                                    proposed_rois=referee_rois)
    referee_oracle, *_ = evaluate(model, referee_images, referee_truth, device)

    cfg.out.mkdir(parents=True, exist_ok=True)
    export_onnx(model, cfg.out / "landmarker.onnx")
    meta = {
        "contract": "openpave.acquisition-matched-landmarker.v2",
        "benchmark": "permanent oracle-ROI ceiling + proposed-ROI deployment truth",
        "model": "soft-argmax heatmap landmarker",
        "crop_px": CROP, "heatmap_px": HEAT, "params": params,
        "sources": sources,
        "auxiliary_sources": sorted(auxiliary_sources),
        "deployment_validation_sources": validation_sources,
        "source_frame_counts": {
            "train": {
                name: int((train_names == name).sum())
                for name in sorted(set(train_names))
            },
            "deployment_validation": {
                name: int((val_names == name).sum()) for name in sorted(set(val_names))
            },
        },
        "referee": cfg.referee,
        "caps": {"train_per_source": cfg.train_cap, "val_per_source": cfg.val_cap,
                 "referee": cfg.referee_cap},
        "epochs": cfg.epochs, "seed": cfg.seed, "training_device": device,
        "cpu_threads": cfg.cpu_threads,
        "acquisition_trace_thermal_duty_cycle": cfg.thermal_duty_cycle,
        "confidence_threshold": threshold,
        "confidence_threshold_f1": threshold_f1,
        "confidence_semantics": f"P(joint within {PCK_GOOD:.0%} of ROI size)",
        "confidence_calibration_distribution": "frozen proposer validation crops",
        "training_distribution": {
            "proposed_roi_fraction": 1.0 - cfg.oracle_mix,
            "oracle_roi_fraction": cfg.oracle_mix,
            "runtime_selected_roi_fraction_within_non_oracle": (
                1.0 - cfg.raw_proposer_mix if cfg.acquisition_model_dir else 0.0),
            "raw_proposer_roi_fraction_within_non_oracle": (
                cfg.raw_proposer_mix if cfg.acquisition_model_dir else 1.0),
            "raw_proposer_hypotheses_per_frame": float(np.mean(
                [len(value) for value in raw_train_hypotheses])),
            "acquisition_model_dir": (str(cfg.acquisition_model_dir)
                                      if cfg.acquisition_model_dir else None),
            "acquisition_trace_quality_mean": (
                float(train_trace_quality.mean())
                if train_trace_quality is not None else None),
            "out_of_crop_joint_target": "visibility=0",
            "proposer_presence_summary": {
                "mean": float(train_presence.mean()),
                "below_0_5": float((train_presence < 0.5).mean()),
            },
            "proposer_onnx": str(cfg.proposer),
            "proposer_sha256": sha256(cfg.proposer),
        },
        "evaluation_columns": {
            "oracle_roi_capability_ceiling": {
                "exploration_holdout": holdout_oracle,
                "untouched_referee": referee_oracle,
            },
            "proposed_roi_deployment_truth": {
                "exploration_holdout": holdout_proposed,
                "untouched_referee": referee_proposed,
                "live_replay": None,
            },
        },
        # Compatibility aliases. New selection code must use evaluation_columns.
        "exploration_holdout": holdout_oracle,
        "untouched_referee": referee_oracle,
        "previous_student_reference": {
            "exploration_holdout_mean_px_384": 81.86,
            "untouched_referee_mean_px_384": 60.28,
            "source": "docs/training-with-monty.md results table",
        },
        "runtime_promotion": False,
        "runtime_policy": {
            "topk_roi_hypotheses": 3,
            "cold_start_scales": [0.75, 1.0, 1.4],
            "candidate_evidence": "75% palm-anchor confidence + 25% all-joint confidence",
            "partial_lock": "wrist + >=3 MCPs + valid palm proportions",
            "partial_emission": "partial lock plus >=6 accepted joints",
            "cold_history_frames": 5,
            "cold_min_history": 3,
            "rejected_joints": "NaN; never topology-filled",
        },
        "selection_metric": "proposed_roi_deployment_truth/live_replay",
        "selection_gate": {"passed": False,
                           "reason": "live replay has not yet updated meta.json"},
        "runtime_blocker": "live-replay acquisition/wrong-intent/time-to-lock gate pending",
        "training_seconds": time.perf_counter() - started,
    }
    meta["onnx_sha256"] = sha256(cfg.out / "landmarker.onnx")
    (cfg.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: meta[k] for k in (
        "params", "confidence_threshold", "evaluation_columns")},
        indent=2))
    print(f"[acquisition-matched] wrote {cfg.out / 'landmarker.onnx'}")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", default="crude,hagrid_shapes,jester,ipn")
    p.add_argument(
        "--auxiliary-sources",
        default="",
        help=(
            "training-only sources excluded from deployment validation and gate "
            "calibration"
        ),
    )
    p.add_argument("--referee", default="yolo26")
    p.add_argument("--train-cap", type=int, default=4000)
    p.add_argument("--val-cap", type=int, default=600)
    p.add_argument("--referee-cap", type=int, default=0)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--seed", type=int, default=37)
    p.add_argument("--cpu-train", action="store_true")
    p.add_argument("--cpu-threads", type=int, default=8)
    p.add_argument("--proposer", type=Path, default=DEFAULT_PROPOSER)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--oracle-mix", type=float, default=0.30)
    p.add_argument("--acquisition-model-dir", type=Path,
                   help="prior round landmarker used to trace live multi-scale crops")
    p.add_argument("--raw-proposer-mix", type=float, default=0.50,
                   help="fraction of non-oracle training crops kept at raw proposal")
    p.add_argument("--thermal-duty-cycle", type=float, default=0.90)
    return p


if __name__ == "__main__":
    main()
