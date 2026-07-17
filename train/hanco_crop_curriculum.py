#!/usr/bin/env python3
"""HanCo-only crop curriculum: Monty-graph label propagation + pose pretraining.

The reviewed manifest labels ~30 sequences; HanCo itself is unlabelled. This
curriculum turns the tbp.monty sparse checkpoint (all 1,502 reference-frame
graphs, exported to ``monty_reference_graphs.npz``) into supervision:

1. PROPAGATE — every unlabelled frame's canonical constellation is matched
   against the reviewed frames' constellations in the shared Monty hand frame.
   The acceptance gate is calibrated by leave-one-sequence-out over the
   reviewed sequences, so propagation precision is measured, never assumed.
2. PRETRAIN — the crop trunk first learns crop → canonical 3D pose on
   positives regardless of gesture label (the "foundational skeleton" stage;
   the target is view-invariant, so eight cameras of one frame agree).
3. FINE-TUNE — a 5-class head trains on reviewed labels plus
   confidence-weighted propagated pseudo-labels, with a multi-view
   consistency loss between synchronized cameras of the same frame.

Evaluation uses ONLY reviewed sequences (ground truth); pseudo-labels never
enter the evaluation or calibration splits. The exported ONNX keeps the
``crops``/``probabilities`` contract of train/hanco_crop_gesture.py, so the
GUI's HanCoCropGestureWorker runs it unchanged behind the frozen 71k acquirer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REFERENCE = ROOT / "train/datasets/hanco/monty_reference_graphs.npz"
MANIFEST = ROOT / "train/datasets/hanco/gesture_manifest.json"
INDEX = ROOT / "train/datasets/hanco_gestures/index.npz"
INDEX_META = INDEX.with_name("meta.json")
OUT = ROOT / "train/runs/hanco_crop_curriculum"
CROP = 64
GESTURES = ("fist", "like", "palm", "point")
CLASSES = ("fist", "like", "no_hand", "palm", "point")


# --------------------------------------------------------------------------
# Stage 1 data: pose descriptors from the Monty reference graphs
# --------------------------------------------------------------------------


_FINGER_CHAINS = np.asarray([
    [1, 2, 3, 4],      # thumb
    [5, 6, 7, 8],      # index
    [9, 10, 11, 12],   # middle
    [13, 14, 15, 16],  # ring
    [17, 18, 19, 20],  # pinky
])


def articulation_descriptor(canonical: np.ndarray) -> np.ndarray:
    """Shape-robust pose descriptor: curls, splay, and fingertip geometry.

    Raw canonical coordinates mix articulation with per-subject hand shape;
    joint angles and normalized fingertip relations transfer across subjects,
    which is what label propagation over 1,502 different recordings needs.
    """
    canonical = np.asarray(canonical, np.float32).reshape(-1, 21, 3)

    def angles(first, second):
        dot = (first * second).sum(-1)
        norms = np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
        return np.arccos(np.clip(dot / np.maximum(norms, 1e-9), -1.0, 1.0))

    chains = canonical[:, _FINGER_CHAINS]              # (n, 5, 4, 3)
    bones = np.diff(chains, axis=2)                    # (n, 5, 3, 3)
    curls = np.concatenate([
        angles(bones[:, :, 0], bones[:, :, 1]),
        angles(bones[:, :, 1], bones[:, :, 2]),
    ], axis=1)                                         # (n, 10)
    tips = canonical[:, [4, 8, 12, 16, 20]]
    reach = np.linalg.norm(tips - canonical[:, :1], axis=2)      # (n, 5)
    elevation = tips[:, :, 2]                                    # (n, 5)
    directions = tips - canonical[:, [1, 5, 9, 13, 17]]
    splay = angles(directions[:, :-1], directions[:, 1:])        # (n, 4)
    pinch = np.stack([
        np.linalg.norm(tips[:, 0] - tips[:, 1], axis=1),
        np.linalg.norm(tips[:, 0] - canonical[:, 5], axis=1),
    ], axis=1)                                                   # (n, 2)
    return np.concatenate([curls, reach, elevation, splay, pinch], axis=1)


def load_reference(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as stored:
        sequences = stored["sequences"].astype(str)
        offsets = stored["offsets"]
        canonical = stored["canonical"].astype(np.float32)
        frame_ids = stored["frame_ids"].astype(str)
    frame_sequences = np.repeat(sequences, np.diff(offsets))
    descriptors = articulation_descriptor(canonical)
    centre = descriptors.mean(0)
    spread = descriptors.std(0).clip(1e-6)
    return {
        "sequences": sequences,
        "canonical": canonical,
        "descriptors": ((descriptors - centre) / spread).astype(np.float32),
        "pose_targets_all": canonical.reshape(len(canonical), 63),
        "frame_sequences": frame_sequences,
        "frame_ids": frame_ids,
        "row_by_key": {
            (frame_sequences[row], frame_ids[row]): row
            for row in range(len(frame_ids))
        },
    }


def reviewed_frame_labels(index) -> dict[tuple[str, str], str]:
    """Per-frame gesture truth from the reviewed index (camera-independent)."""
    labels: dict[tuple[str, str], str] = {}
    for sequence, frame, label in zip(
            index["sequence_ids"].astype(str), index["frame_ids"].astype(str),
            index["labels"].astype(str), strict=True):
        key = (sequence, frame)
        if labels.setdefault(key, label) != label:
            raise ValueError(f"conflicting reviewed labels for {key}")
    return labels


def knn_vote(anchor_descriptors, anchor_labels, queries, k):
    """Majority label, vote fraction, and top-1 distance for each query row."""
    votes = np.empty(len(queries), dtype="<U16")
    fractions = np.empty(len(queries), np.float32)
    distances = np.empty(len(queries), np.float32)
    class_names = np.unique(anchor_labels)
    anchor_norms = (anchor_descriptors ** 2).sum(1)
    for start in range(0, len(queries), 4096):
        block = queries[start:start + 4096]
        squared = ((block ** 2).sum(1)[:, None] + anchor_norms[None, :]
                   - 2.0 * block @ anchor_descriptors.T)
        pairwise = np.sqrt(np.maximum(squared, 0.0))
        nearest = np.argpartition(pairwise, k - 1, axis=1)[:, :k]
        block_rows = np.arange(len(block))[:, None]
        near_labels = anchor_labels[nearest]
        near_distances = pairwise[block_rows, nearest]
        counts = np.stack(
            [(near_labels == name).sum(1) for name in class_names], axis=1)
        winner = counts.argmax(1)
        votes[start:start + 4096] = class_names[winner]
        fractions[start:start + 4096] = counts[np.arange(len(block)), winner] / k
        distances[start:start + 4096] = near_distances.min(1)
    return votes, fractions, distances


def calibrate_propagation(reference, frame_truth, k, precision_target, log):
    """Leave-one-sequence-out: measured precision picks the distance gate."""
    keys = list(frame_truth)
    rows = np.asarray([reference["row_by_key"][key] for key in keys])
    truths = np.asarray([frame_truth[key] for key in keys])
    owner = np.asarray([key[0] for key in keys])
    descriptors = reference["descriptors"][rows]
    predicted = np.empty(len(keys), dtype="<U16")
    fractions = np.empty(len(keys), np.float32)
    distances = np.empty(len(keys), np.float32)
    for sequence in np.unique(owner):
        held = owner == sequence
        predicted[held], fractions[held], distances[held] = knn_vote(
            descriptors[~held], truths[~held], descriptors[held], k)
    candidates = np.quantile(distances, np.linspace(0.05, 0.95, 19))
    table = []
    for vote_gate in (0.6, 0.8, 1.0):
        confident = fractions >= vote_gate
        for tau in candidates:
            accepted = confident & (distances <= tau)
            if not accepted.any():
                continue
            table.append({
                "vote_gate": vote_gate,
                "tau": float(tau),
                "precision": float((predicted[accepted] == truths[accepted]).mean()),
                "coverage": float(accepted.mean()),
            })
    meeting = [row for row in table if row["precision"] >= precision_target]
    if meeting:
        chosen = max(meeting, key=lambda row: row["coverage"])
    else:
        chosen = max(table, key=lambda row: row["precision"])
        log(f"[propagate] WARNING: no gate reached precision {precision_target};"
            f" using best measured {chosen['precision']:.3f}")
    per_class = {}
    accepted = (fractions >= chosen["vote_gate"]) & (distances <= chosen["tau"])
    for name in GESTURES:
        rows_class = truths == name
        hit = accepted & rows_class
        per_class[name] = {
            "coverage": float(hit.mean() / max(rows_class.mean(), 1e-9)),
            "precision": float((predicted[hit] == truths[hit]).mean()) if hit.any() else None,
        }
    log(f"[propagate] LOSO gate vote>={chosen['vote_gate']} tau={chosen['tau']:.4f} "
        f"precision={chosen['precision']:.3f} coverage={chosen['coverage']:.3f}")
    return chosen, {"sweep": table, "per_class": per_class, "k": k}


def propagate_labels(reference, frame_truth, excluded, gate, k, log):
    """Label every eligible unreviewed frame or leave it honestly unlabelled."""
    anchor_rows = np.asarray([reference["row_by_key"][key] for key in frame_truth])
    anchor_labels = np.asarray(list(frame_truth.values()))
    anchor_descriptors = reference["descriptors"][anchor_rows]
    reviewed = {key[0] for key in frame_truth}
    eligible = ~np.isin(reference["frame_sequences"], sorted(reviewed | excluded))
    rows = np.flatnonzero(eligible)
    votes, fractions, distances = knn_vote(
        anchor_descriptors, anchor_labels, reference["descriptors"][rows], k)
    accepted = (fractions >= gate["vote_gate"]) & (distances <= gate["tau"])
    confidence = np.where(accepted, fractions, 0.0).astype(np.float32)
    counts = {name: int((votes[accepted] == name).sum()) for name in GESTURES}
    log(f"[propagate] {accepted.sum():,}/{len(rows):,} unreviewed frames accepted "
        f"({json.dumps(counts)})")
    return {
        "rows": rows[accepted],
        "labels": votes[accepted],
        "confidence": confidence[accepted],
        "rejected_rows": rows[~accepted],
        "counts": counts,
    }


# --------------------------------------------------------------------------
# Stage 2 data: crops for reviewed + propagated + pose-only observations
# --------------------------------------------------------------------------


def project_frame(root: Path, sequence: str, frame: str) -> dict[int, np.ndarray]:
    from train.prepare_hanco_gestures import project

    xyz = np.asarray(
        json.loads((root / "xyz" / sequence / f"{frame}.json").read_text()),
        np.float32)
    calibration = json.loads(
        (root / "HanCo_calib_meta/calib" / sequence / f"{frame}.json").read_text())
    views = {}
    for camera in range(8):
        points = project(xyz, calibration, camera)
        if points is not None:
            views[camera] = points
    return views


def build_observation_table(reference, index, propagation, config, rng, log):
    """One row per (sequence, frame, camera) with label/split/weight columns."""
    root = Path(json.loads(config.index_meta.read_text())["root"]).expanduser()
    table = {key: [] for key in (
        "kind", "label", "split", "sequence", "frame", "camera",
        "landmarks", "pose_row", "weight")}

    def push(kind, label, split, sequence, frame, camera, landmarks, pose_row, weight):
        table["kind"].append(kind)
        table["label"].append(label)
        table["split"].append(split)
        table["sequence"].append(sequence)
        table["frame"].append(frame)
        table["camera"].append(camera)
        table["landmarks"].append(np.asarray(landmarks, np.float32).reshape(42))
        table["pose_row"].append(pose_row)
        table["weight"].append(weight)

    # NpzFile members decompress on every subscript; materialize once.
    reviewed = {key: np.asarray(index[key]) for key in (
        "labels", "splits", "sequence_ids", "frame_ids", "camera_ids", "landmarks")}
    for position in range(len(reviewed["labels"])):
        sequence = str(reviewed["sequence_ids"][position])
        frame = str(reviewed["frame_ids"][position])
        pose_row = reference["row_by_key"].get((sequence, frame), -1)
        push("reviewed", str(reviewed["labels"][position]),
             str(reviewed["splits"][position]), sequence, frame,
             int(reviewed["camera_ids"][position]),
             reviewed["landmarks"][position], pose_row, 1.0)

    pseudo_budget = {name: config.pseudo_per_class for name in GESTURES}
    order = rng.permutation(len(propagation["rows"]))
    projected_cache: dict[tuple[str, str], dict[int, np.ndarray]] = {}
    for position in order:
        row = int(propagation["rows"][position])
        label = str(propagation["labels"][position])
        if pseudo_budget[label] <= 0:
            continue
        sequence = str(reference["frame_sequences"][row])
        if int(sequence) % 5 in (0, 1):
            continue  # keep the sequence-modulo split contract for the future
        frame = str(reference["frame_ids"][row])
        views = projected_cache.setdefault(
            (sequence, frame), project_frame(root, sequence, frame))
        cameras = rng.permutation(sorted(views))[:config.pseudo_cameras]
        for camera in cameras:
            if pseudo_budget[label] <= 0:
                break
            push("pseudo", label, "train", sequence, frame, int(camera),
                 views[int(camera)], row,
                 float(propagation["confidence"][position]))
            pseudo_budget[label] -= 1

    extra = rng.permutation(propagation["rejected_rows"])[:config.pose_extra_frames]
    for row in extra:
        sequence = str(reference["frame_sequences"][int(row)])
        if int(sequence) % 5 in (0, 1):
            continue
        frame = str(reference["frame_ids"][int(row)])
        views = projected_cache.setdefault(
            (sequence, frame), project_frame(root, sequence, frame))
        if not views:
            continue
        camera = int(rng.permutation(sorted(views))[0])
        push("pose_only", "", "train", sequence, frame, camera,
             views[camera], int(row), 0.0)

    for key in table:
        table[key] = np.asarray(table[key])
    kinds, counts = np.unique(table["kind"], return_counts=True)
    log(f"[data] observations: {dict(zip(kinds.tolist(), counts.tolist()))}")
    return table, root


def build_crop_cache(table, reference, root: Path, config, rng, log):
    """Positives, canonical pose targets, and inpainted no_hand negatives."""
    import cv2

    from train.hanco_crop_gesture import take_crop

    digest = hashlib.sha256(json.dumps({
        "sequences": table["sequence"].tolist(), "frames": table["frame"].tolist(),
        "cameras": table["camera"].tolist(), "labels": table["label"].tolist(),
        "negative_ratio": config.negative_ratio, "crop": CROP, "seed": config.seed,
    }, sort_keys=True).encode()).hexdigest()
    cache = config.out / "crops.npz"
    if cache.is_file():
        stored = np.load(cache, allow_pickle=False)
        if str(stored["digest"]) == digest:
            log(f"[data] reusing crop cache {cache.name}")
            return (stored["crops"], stored["pose_targets"],
                    stored["negative_crops"], stored["negative_splits"])

    kernel = np.ones((7, 7), np.uint8)
    total = len(table["kind"])
    crops = np.empty((total, CROP, CROP, 3), np.uint8)
    pose_targets = np.empty((total, 63), np.float32)
    negative_crops, negative_splits = [], []
    negative_stride = max(1, config.negative_ratio)
    started = time.perf_counter()
    for position in range(total):
        sequence, frame = table["sequence"][position], table["frame"][position]
        camera = int(table["camera"][position])
        image = cv2.imread(
            str(root / "rgb" / sequence / f"cam{camera}" / f"{frame}.jpg"),
            cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"rgb missing for {sequence}/{frame}/cam{camera}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        landmarks = table["landmarks"][position].reshape(21, 2)
        crops[position] = take_crop(image, landmarks)
        pose_row = int(table["pose_row"][position])
        pose_targets[position] = (
            reference["pose_targets_all"][pose_row] if pose_row >= 0 else np.nan)
        if table["kind"][position] != "pose_only" and position % negative_stride == 0:
            mask = cv2.imread(
                str(root / "mask_hand" / sequence / f"cam{camera}" / f"{frame}.jpg"),
                cv2.IMREAD_GRAYSCALE)
            if mask is None or (mask >= 128).mean() < 0.002:
                continue
            binary = cv2.dilate((mask >= 128).astype(np.uint8) * 255, kernel)
            removed = cv2.inpaint(
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR), binary, 5, cv2.INPAINT_TELEA)
            negative_crops.append(
                take_crop(cv2.cvtColor(removed, cv2.COLOR_BGR2RGB), landmarks))
            negative_splits.append(table["split"][position])
        if position and position % 8000 == 0:
            log(f"[data] crops {position}/{total} "
                f"({time.perf_counter() - started:.0f}s)")
    negative_crops = np.asarray(negative_crops, np.uint8)
    negative_splits = np.asarray(negative_splits)
    config.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache, crops=crops, pose_targets=pose_targets,
        negative_crops=negative_crops, negative_splits=negative_splits,
        digest=np.asarray(digest))
    log(f"[data] {total:,} positive + {len(negative_crops):,} negative crops "
        f"cached in {time.perf_counter() - started:.0f}s")
    return crops, pose_targets, negative_crops, negative_splits


# --------------------------------------------------------------------------
# Model and training
# --------------------------------------------------------------------------


def build_model():
    import torch.nn as nn

    class DS(nn.Module):
        def __init__(self, channels_in, channels_out, stride=1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(channels_in, channels_in, 3, stride, 1,
                          groups=channels_in, bias=False),
                nn.BatchNorm2d(channels_in), nn.SiLU(),
                nn.Conv2d(channels_in, channels_out, 1, bias=False),
                nn.BatchNorm2d(channels_out), nn.SiLU())

        def forward(self, values):
            return self.net(values)

    class Curriculum(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Conv2d(3, 24, 3, 2, 1, bias=False), nn.BatchNorm2d(24), nn.SiLU(),
                DS(24, 40, 2), DS(40, 64, 2), DS(64, 96, 2), DS(96, 128),
                nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.pose_head = nn.Linear(128, 63)
            self.gesture_head = nn.Linear(128, len(CLASSES))

        def forward(self, values):
            return self.gesture_head(self.trunk(values))

    return Curriculum()


def run(config: argparse.Namespace) -> dict:
    import torch
    import torch.nn.functional as functional
    from sklearn.metrics import confusion_matrix, f1_score

    def log(message: str) -> None:
        print(message, flush=True)

    started = time.perf_counter()
    torch.set_num_threads(config.threads)
    reference = load_reference(config.reference)
    index = np.load(config.index, allow_pickle=True)
    rng = np.random.default_rng(config.seed)
    manifest = json.loads(config.manifest.read_text())
    excluded = {str(item) for item in manifest.get("unconfirmed", [])}

    frame_truth = reviewed_frame_labels(index)
    gate, propagation_report = calibrate_propagation(
        reference, frame_truth, config.knn, config.loso_precision, log)
    propagation = propagate_labels(
        reference, frame_truth, excluded, gate, config.knn, log)
    table, root = build_observation_table(
        reference, index, propagation, config, rng, log)
    crops, pose_targets, negative_crops, negative_splits = build_crop_cache(
        table, reference, root, config, rng, log)

    class_to_index = {name: position for position, name in enumerate(CLASSES)}
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = build_model().to(device)

    def tensorize(batch_crops):
        values = torch.tensor(batch_crops, device=device)
        # MPS autograd rejects the non-contiguous permute view in backward.
        return (values.permute(0, 3, 1, 2).float() / 255.0 - 0.5).contiguous()

    # ---- Stage 1: crop → canonical 3D pose (no gesture labels used) --------
    pose_rows = np.flatnonzero(
        np.isfinite(pose_targets).all(1) & (table["split"] == "train"))
    targets_tensor = torch.tensor(pose_targets, device=device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(config.pose_epochs):
        model.train()
        order = rng.permutation(pose_rows)
        running = 0.0
        for batch in np.array_split(order, max(1, len(order) // config.batch)):
            values = tensorize(crops[batch])
            values = values * (0.85 + 0.3 * torch.rand(len(batch), 1, 1, 1, device=device))
            prediction = model.pose_head(model.trunk(values))
            loss = functional.smooth_l1_loss(
                prediction, targets_tensor[batch], beta=0.2)
            optimiser.zero_grad(); loss.backward()
            optimiser.step(); running += float(loss.detach()) * len(batch)
        log(f"[pose] epoch={epoch + 1}/{config.pose_epochs} "
            f"loss={running / max(1, len(pose_rows)):.4f}")

    # ---- Stage 2: gesture head with pseudo-labels + view consistency -------
    labelled = table["kind"] != "pose_only"
    labels = table["label"].copy()
    splits = table["split"]
    weights_row = table["weight"].astype(np.float32)
    positive_train = np.flatnonzero(labelled & (splits == "train"))
    negative_train = np.flatnonzero(negative_splits == "train")
    group_ids = np.char.add(np.char.add(table["sequence"], "/"), table["frame"])
    frame_groups: dict[str, list[int]] = {}
    for position in positive_train:
        frame_groups.setdefault(group_ids[position], []).append(int(position))
    paired_groups = [rows for rows in frame_groups.values() if len(rows) >= 2]

    counts = np.bincount(
        [class_to_index[label] for label in labels[positive_train]],
        minlength=len(CLASSES)).astype(float)
    counts[class_to_index["no_hand"]] = max(1, len(negative_train))
    class_weights = torch.tensor(
        counts.sum() / (len(CLASSES) * counts.clip(1)), dtype=torch.float32,
        device=device)
    trunk_parameters = list(model.trunk.parameters())
    trunk_scale = config.trunk_lr_scale if config.pose_epochs > 0 else 1.0
    optimiser = torch.optim.AdamW([
        {"params": trunk_parameters, "lr": config.lr * trunk_scale},
        {"params": model.gesture_head.parameters(), "lr": config.lr},
    ], weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=config.epochs)

    def evaluate(split: str):
        model.eval()
        rows = np.flatnonzero(labelled & (splits == split) & (table["kind"] == "reviewed"))
        negatives = np.flatnonzero(negative_splits == split)
        with torch.no_grad():
            outputs, truths = [], []
            for batch in np.array_split(rows, max(1, len(rows) // 512)):
                outputs.append(model(tensorize(crops[batch])).softmax(1).cpu().numpy())
                truths.extend(class_to_index[label] for label in labels[batch])
            for batch in np.array_split(negatives, max(1, len(negatives) // 512)):
                outputs.append(model(tensorize(negative_crops[batch])).softmax(1).cpu().numpy())
                truths.extend([class_to_index["no_hand"]] * len(batch))
        return np.concatenate(outputs), np.asarray(truths)

    best_state, best_f1 = None, -1.0
    for epoch in range(config.epochs):
        model.train()
        order = rng.permutation(positive_train)
        negatives = rng.permutation(negative_train)
        negative_take = max(1, int(len(order) / max(1, len(negative_train)))) or 1
        running, batches = 0.0, np.array_split(order, max(1, len(order) // config.batch))
        for step, batch in enumerate(batches):
            take = negatives[(step * config.batch // 3) % max(1, len(negatives)):][
                : config.batch // 3]
            values = torch.cat([tensorize(crops[batch]), tensorize(negative_crops[take])])
            truth = torch.tensor(
                [class_to_index[label] for label in labels[batch]]
                + [class_to_index["no_hand"]] * len(take), device=device)
            sample_weight = torch.tensor(
                np.concatenate([
                    np.where(table["kind"][batch] == "pseudo",
                             config.pseudo_weight * weights_row[batch], 1.0),
                    np.ones(len(take), np.float32)]),
                device=device)
            values = values * (0.85 + 0.3 * torch.rand(len(values), 1, 1, 1, device=device))
            values = values + torch.randn_like(values) * 0.015
            flip = torch.rand(len(values), device=device) < 0.5
            values = torch.where(flip[:, None, None, None], values.flip(-1), values)
            logits = model(values.contiguous())
            loss_terms = functional.cross_entropy(
                logits, truth, weight=class_weights, reduction="none")
            loss = (loss_terms * sample_weight).mean()
            if paired_groups and config.consistency_weight > 0:
                chosen = [paired_groups[i] for i in
                          rng.integers(0, len(paired_groups), size=16)]
                pairs = np.asarray([rng.choice(rows, 2, replace=False)
                                    for rows in chosen])
                left = model(tensorize(crops[pairs[:, 0]]))
                right = model(tensorize(crops[pairs[:, 1]]))
                consistency = functional.kl_div(
                    left.log_softmax(1), right.softmax(1).detach(),
                    reduction="batchmean")
                loss = loss + config.consistency_weight * consistency
            optimiser.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0, foreach=False)
            optimiser.step(); running += float(loss.detach()) * len(batch)
        schedule.step()
        probabilities, truths = evaluate("calibration")
        score = float(f1_score(truths, probabilities.argmax(1),
                               average="macro", zero_division=0))
        log(f"[gesture] epoch={epoch + 1}/{config.epochs} "
            f"loss={running / max(1, len(order)):.4f} calibration_macro_f1={score:.3f}")
        if score > best_f1:
            best_f1 = score
            best_state = {name: tensor.detach().cpu().clone()
                          for name, tensor in model.state_dict().items()}

    model.load_state_dict(best_state)
    probabilities, truths = evaluate("evaluation")
    predicted = probabilities.argmax(1)
    matrix = confusion_matrix(truths, predicted, labels=np.arange(len(CLASSES)))
    positive = truths != class_to_index["no_hand"]
    metrics = {
        "macro_f1": float(f1_score(truths, predicted, average="macro", zero_division=0)),
        "overall_accuracy": float((truths == predicted).mean()),
        "correct_gesture_acquisition_rate": float(
            (truths[positive] == predicted[positive]).mean()),
        "wrong_gesture_rate": float(((truths[positive] != predicted[positive])
                                     & (predicted[positive] != class_to_index["no_hand"])).mean()),
        "no_hand_false_acquisition_rate": float(
            (predicted[~positive] != class_to_index["no_hand"]).mean()),
        "confusion_matrix_order": list(CLASSES),
        "confusion_matrix": matrix.tolist(),
    }
    for label in GESTURES:
        rows = truths == class_to_index[label]
        metrics[f"{label}_correct_rate"] = float(
            (predicted[rows] == truths[rows]).mean()) if rows.any() else None

    model = model.cpu().eval()

    class Export(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.trunk = inner.trunk
            self.head = inner.gesture_head

        def forward(self, values):
            return self.head(self.trunk(values)).softmax(1)

    model_path = config.out / "crop.onnx"
    torch.onnx.export(
        Export(model), torch.zeros(1, 3, CROP, CROP), model_path,
        input_names=["crops"], output_names=["probabilities"],
        dynamic_axes={"crops": {0: "n"}, "probabilities": {0: "n"}}, dynamo=False)
    exported_parameters = (
        sum(parameter.numel() for parameter in model.trunk.parameters())
        + sum(parameter.numel() for parameter in model.gesture_head.parameters()))
    report = {
        "contract": "openpave.hanco-crop-curriculum.v1",
        "front_end": "legacy 71k landmark_tower (frozen)",
        "training_sources": [
            "HanCo RGB", "HanCo mask_hand", "HanCo calibration/xyz/shape",
            "tbp.monty reference graphs (model.pt export)"],
        "external_training_sources": [],
        "classes": list(CLASSES),
        "reference_sha256": hashlib.sha256(config.reference.read_bytes()).hexdigest(),
        "index_sha256": hashlib.sha256(config.index.read_bytes()).hexdigest(),
        "crop_px": CROP,
        "params": exported_parameters,
        "propagation": {
            "gate": gate, **propagation_report,
            "accepted_frames": int(len(propagation["rows"])),
            "accepted_per_class": propagation["counts"],
            "excluded_sequences": sorted(excluded),
        },
        "curriculum": {
            "pose_epochs": config.pose_epochs,
            "trunk_lr_scale": trunk_scale,
            "gesture_epochs": config.epochs,
            "pseudo_per_class": config.pseudo_per_class,
            "pseudo_weight": config.pseudo_weight,
            "consistency_weight": config.consistency_weight,
            "pose_extra_frames": config.pose_extra_frames,
        },
        "evaluation_note": (
            "evaluation and calibration splits contain reviewed sequences only; "
            "pseudo-labels train, never grade"),
        "best_calibration_macro_f1": best_f1,
        "metrics": metrics,
        "model": str(model_path),
        "seconds": time.perf_counter() - started,
        "seed": config.seed,
    }
    (config.out / "meta.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--reference", type=Path, default=REFERENCE)
    result.add_argument("--manifest", type=Path, default=MANIFEST)
    result.add_argument("--index", type=Path, default=INDEX)
    result.add_argument("--index-meta", type=Path, default=INDEX_META)
    result.add_argument("--out", type=Path, default=OUT)
    result.add_argument("--knn", type=int, default=5)
    result.add_argument("--loso-precision", type=float, default=0.9)
    result.add_argument("--pseudo-per-class", type=int, default=3000)
    result.add_argument("--pseudo-cameras", type=int, default=3)
    result.add_argument("--pseudo-weight", type=float, default=0.6)
    result.add_argument("--pose-extra-frames", type=int, default=4000)
    result.add_argument("--consistency-weight", type=float, default=0.2)
    result.add_argument("--negative-ratio", type=int, default=3)
    result.add_argument("--pose-epochs", type=int, default=6)
    result.add_argument("--trunk-lr-scale", type=float, default=0.25)
    result.add_argument("--epochs", type=int, default=14)
    result.add_argument("--batch", type=int, default=192)
    result.add_argument("--lr", type=float, default=8e-4)
    result.add_argument("--threads", type=int, default=8)
    result.add_argument("--seed", type=int, default=37)
    return result


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
