"""Gesture lab: multi-source distillation trainer for the tiny gesture net.

The insect-poc idea applied to gestures: independent DATA SOURCES share one
contract (frames in, teacher-labelled shard out) but not a lifecycle —
re-preparing `hagrid` never touches `crude`'s shard, and `train` consumes
whichever prepared shards exist. See train/datasets/README.md for the source
catalogue and manual-fetch notes (InterHand2.6M / H2O are registration-gated).

Teacher = the shipping MediaPipe+SVM+geometry pipeline; student = the same
~99k-param TinyNet the viewer's "CPU · TinyNet (distilled)" runtime loads
(artifact is written to train/runs/tiny_gesture/, so the GUI picks a retrain
up automatically). The whole point vs train/tiny_gesture.py: PIXEL DIVERSITY.
A student that only ever saw six videos of one room fails in every other
room; hagrid alone adds hundreds of subjects/backgrounds.

Stages (via ./train/gesture-lab.sh):
  list     what exists, what is prepared, frame counts — check this FIRST
  fetch    download/locate raw data (hagrid reuses the HF-cached zip)
  prepare  teacher-label frames -> datasets/<source>/prepared.npz (cached)
  train    TinyNet on the union of prepared shards
  eval     outcome scoreboard on crude held-out tails (same as tiny_gesture)
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

TRAIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAIN_DIR))
from mediapipe_svm import (  # noqa: E402
    LANDMARKER, SOURCES as CRUDE_SOURCES, VAL_FRAC, _read_frames, _resolve_source,
    landmarks_to_features, point_direction,
)
from tiny_gesture import CLASSES, FLIP_SWAP, IMG, OUT_DIR, SVM_DIR, _build_net  # noqa: E402

DATASETS = TRAIN_DIR / "datasets"
HAGRID_REPO, HAGRID_FILE = "cj-mills/hagrid-sample-30k-384p", "hagrid-sample-30k-384p.zip"
ALL_SOURCES = ("crude", "custom", "hagrid", "hagrid_shapes", "swipe_phases", "jester", "yolo26", "ipn", "nvgesture", "interhand26m", "h2o")
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
VIDEO_EXTS = (".mp4", ".mov", ".gif")


# ── teacher ──────────────────────────────────────────────────────────────────

class Teacher:
    """MediaPipe landmarker + shape SVM + direction geometry, loaded once."""

    def __init__(self, conf: float) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        import onnxruntime as ort

        self.conf = conf
        self.mp = mp
        meta = json.loads((SVM_DIR / "meta.json").read_text())
        self.svm_classes = meta["classes"]
        self.sess = ort.InferenceSession(str(SVM_DIR / "model.onnx"),
                                         providers=["CPUExecutionProvider"])
        self.lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(LANDMARKER)),
            num_hands=1, min_hand_detection_confidence=0.5))

    def label(self, rgb: np.ndarray) -> str | None:
        return self.label_lm(rgb)[0]

    def label_lm(self, rgb: np.ndarray) -> tuple[str | None, np.ndarray | None]:
        """(outcome label, 21x(x,y) landmark vector) for one RGB frame.
        label None = ambiguous (do not teach); landmarks None = no hand.
        The landmark trajectories are the v2 student's auxiliary target: to
        predict them it MUST localise the hand, which a plain classifier
        never had to do — the fix for the yolo26 referee's verdict."""
        res = self.lm.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
        if not res.hand_landmarks:
            return "no_hand", None
        hand = res.hand_landmarks[0]
        lm42 = np.array([[q.x, q.y] for q in hand], dtype=np.float32).reshape(-1)
        feats = landmarks_to_features(hand).reshape(1, -1)
        proba = self.sess.run(["probabilities"], {"landmarks": feats})[0][0]
        if float(proba.max()) < self.conf:
            return None, lm42
        cls = self.svm_classes[int(proba.argmax())]
        if cls == "point":
            d = point_direction(hand)
            return {"LEFT": "point_left", "RIGHT": "point_right"}.get(d, "point_vertical"), lm42
        return cls, lm42


# ── frame generators per source ──────────────────────────────────────────────

def _frames_crude(cap: int):
    """(rgb, is_val) from the user's capture videos; temporal val tails."""
    for stem in CRUDE_SOURCES:
        src = _resolve_source(stem)
        if src is None:
            continue
        frames = _read_frames(src)
        if len(frames) > cap:
            frames = [frames[i] for i in np.linspace(0, len(frames) - 1, cap, dtype=int)]
        val_from = int(len(frames) * (1 - VAL_FRAC))
        for i, f in enumerate(frames):
            yield f, i >= val_from


def _frames_custom(cap: int):
    import cv2
    root = DATASETS / "custom"
    root.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS + VIDEO_EXTS)
    for p in files:
        if p.suffix.lower() in IMAGE_EXTS:
            img = cv2.imread(str(p))
            if img is not None:
                yield cv2.cvtColor(img, cv2.COLOR_BGR2RGB), random.random() < VAL_FRAC
        else:
            frames = _read_frames(p)
            if len(frames) > cap:
                frames = [frames[i] for i in np.linspace(0, len(frames) - 1, cap, dtype=int)]
            val_from = int(len(frames) * (1 - VAL_FRAC))
            for i, f in enumerate(frames):
                yield f, i >= val_from


def _frames_hagrid(cap: int):
    """Sample the HF-cached HaGRID zip in place (no extraction to disk)."""
    import cv2
    from huggingface_hub import hf_hub_download
    zf = zipfile.ZipFile(hf_hub_download(HAGRID_REPO, HAGRID_FILE, repo_type="dataset"))
    names = [m for m in zf.namelist() if m.lower().endswith(IMAGE_EXTS)]
    rng = random.Random(22)
    rng.shuffle(names)
    for m in names[:cap]:
        img = cv2.imdecode(np.frombuffer(zf.read(m), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        if max(h, w) > 384:
            s = 384 / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)))
        yield cv2.cvtColor(img, cv2.COLOR_BGR2RGB), rng.random() < 0.15


def _frames_local_videos(source: str, cap: int):
    """Walk datasets/<source>/raw for videos; sample frames uniformly per
    video; last 20% of the video LIST is val (grouped, no temporal leakage)."""
    import cv2
    root = DATASETS / source / "raw"
    vids = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".avi", ".mp4", ".mov", ".mkv"))
    per_video = max(1, cap // max(1, len(vids)))
    val_from = int(len(vids) * 0.8)
    for vi, vid in enumerate(vids):
        cap_v = cv2.VideoCapture(str(vid))
        n = int(cap_v.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        want = set(np.linspace(0, n - 1, per_video, dtype=int).tolist())
        i = 0
        while True:
            ok, bgr = cap_v.read()
            if not ok:
                break
            if i in want:
                h, w = bgr.shape[:2]
                if max(h, w) > 384:
                    sc = 384 / max(h, w)
                    bgr = cv2.resize(bgr, (int(w * sc), int(h * sc)))
                yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), vi >= val_from
            i += 1
        cap_v.release()


def _frames_local_images(source: str, cap: int):
    """interhand26m / h2o: consume whatever the user placed locally."""
    import cv2
    root = DATASETS / source / "images"
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    rng = random.Random(22)
    rng.shuffle(files)
    for p in files[:cap]:
        img = cv2.imread(str(p))
        if img is not None:
            yield cv2.cvtColor(img, cv2.COLOR_BGR2RGB), rng.random() < 0.15


IPN_DIR = TRAIN_DIR / "insect-poc" / "raw" / "ipn"

# hagrid_shapes: static intent shapes at USER-SPECIFIED quotas, extracted from
# the cached 30k zip by HaGRID's OWN folder labels (ground truth — the teacher
# only supplies landmarks). Non-target gestures (peace, ok, three, ...) become
# no_hand/noop pixels: a hand doing something that commands nothing.
HAGRID_QUOTAS = {          # our label -> (hagrid folder classes, quota)
    "like": (("like",), 1500),
    "fist": (("fist",), 1500),
    "stop": (("palm", "stop", "stop_inverted"), 750),
    "point_vertical": (("one",), 500),
    "no_hand": (("peace", "peace_inverted", "ok", "four", "three", "three2",
                 "call", "dislike", "mute", "rock", "two_up", "two_up_inverted"), 3000),
}


def _prepare_hagrid_shapes(force: bool, teacher_conf: float) -> None:
    import cv2
    from huggingface_hub import hf_hub_download
    shard = _shard_path("hagrid_shapes")
    if shard.exists() and not force:
        print("[prepare] hagrid_shapes: prepared.npz exists — skipping (--force to redo)")
        return
    teacher = Teacher(teacher_conf)     # landmarks only; labels are ground truth
    zf = zipfile.ZipFile(hf_hub_download(HAGRID_REPO, HAGRID_FILE, repo_type="dataset"))
    by_cls: dict[str, list[str]] = {}
    for m in zf.namelist():
        if m.lower().endswith(IMAGE_EXTS) and "train_val_" in m:
            by_cls.setdefault(m.split("train_val_")[1].split("/")[0], []).append(m)
    rng = random.Random(22)
    imgs, labels, is_val, lms, has_lm = [], [], [], [], []
    for our_label, (folders, quota) in HAGRID_QUOTAS.items():
        pool: list[str] = []
        for f in folders:
            pool.extend(by_cls.get(f, []))
        rng.shuffle(pool)
        kept = 0
        for m in pool:
            if kept >= quota:
                break
            img = cv2.imdecode(np.frombuffer(zf.read(m), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            if max(h, w) > 384:
                sc = 384 / max(h, w)
                img = cv2.resize(img, (int(w * sc), int(h * sc)))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            _, lm42 = teacher.label_lm(rgb)
            imgs.append(cv2.resize(rgb, (IMG, IMG)))
            labels.append(our_label)
            is_val.append(rng.random() < 0.15)
            lms.append(lm42 if lm42 is not None else np.zeros(42, np.float32))
            has_lm.append(lm42 is not None)
            kept += 1
        print(f"[prepare] hagrid_shapes: {our_label:14s} {kept}/{quota} "
              f"(from {', '.join(folders)})")
    shard.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(shard, imgs=np.array(imgs, dtype=np.uint8),
                        labels=np.array(labels), is_val=np.array(is_val),
                        landmarks=np.array(lms, dtype=np.float32),
                        has_lm=np.array(has_lm))
    print(f"[prepare] hagrid_shapes: {len(imgs)} ground-truth frames -> {shard}")


# jester: temporal direction (Qualcomm research license, account-gated — see
# datasets/README.md). Expects datasets/jester/raw/20bn-jester-v1/<id>/*.jpg
# frame dirs + jester-v1-*.csv labels. Only the five agreed classes are used.
JESTER_CLASSES = {
    "Thumb Up": "like",
    "Stop Sign": "stop",
    "Swiping Left": "point_left",
    "Swiping Right": "point_right",
}


def _prepare_jester(force: bool, teacher_conf: float, cap_per_class: int = 400) -> None:
    import csv
    import cv2
    shard = _shard_path("jester")
    legacy = DATASETS / "jester" / "raw"
    local = DATASETS / "20bn-jester"
    legacy_csvs = [(p, "validation" in p.name.lower(), legacy / "20bn-jester-v1")
                   for p in (sorted(legacy.glob("jester-v1-train*.csv"))
                             + sorted(legacy.glob("jester-v1-validation*.csv")))]
    local_csvs = [(local / "Train.csv", False, local / "Train"),
                  (local / "Validation.csv", True, local / "Validation")]
    sources = legacy_csvs or [(p, val, root) for p, val, root in local_csvs if p.exists()]
    if shard.exists() and not force:
        print("[prepare] jester: prepared.npz exists — skipping (--force to redo)")
        return
    if not sources:
        print("[prepare] jester: no legacy or datasets/20bn-jester data; skipping")
        return
    teacher = Teacher(teacher_conf)
    imgs, labels, is_val, lms, has_lm = [], [], [], [], []
    presence_known = []
    counts: dict[str, int] = {}
    for csv_p, val, frames_root in sources:
        with open(csv_p, newline="") as fh:
            if csv_p.name in ("Train.csv", "Validation.csv"):
                rows = ((r["video_id"], r["label"]) for r in csv.DictReader(fh))
            else:
                rows = ((r[0], r[1]) for r in csv.reader(fh, delimiter=";") if len(r) >= 2)
            for video_id, source_label in rows:
                # Every Jester gesture is valuable landmarker supervision. The
                # five command gestures retain their labels; unsupported hand
                # motions are safe no-op examples for downstream classifiers.
                our = JESTER_CLASSES.get(source_label, "no_hand")
                key = f"{'val' if val else 'train'}:{source_label}"
                limit = max(80, cap_per_class // 4) if val else cap_per_class
                if counts.get(key, 0) >= limit:
                    continue
                vid_dir = frames_root / str(video_id)
                frame_files = sorted(vid_dir.glob("*.jpg"))
                if len(frame_files) < 3:
                    continue
                # Three separated poses per clip maximize camera/person/gesture
                # variety without filling the shard with adjacent duplicates.
                take = [frame_files[int((len(frame_files) - 1) * q)] for q in (.25, .50, .75)]
                for fp in take:
                    img = cv2.imread(str(fp))
                    if img is None:
                        continue
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    _, lm42 = teacher.label_lm(rgb)
                    # A failed teacher on a densely labelled gesture clip is
                    # UNKNOWN, not a negative. Keep only genuine `No gesture`
                    # misses as explicit absence supervision.
                    if lm42 is None and source_label != "No gesture":
                        continue
                    imgs.append(cv2.resize(rgb, (IMG, IMG)))
                    labels.append(our)
                    is_val.append(val)
                    lms.append(lm42 if lm42 is not None else np.zeros(42, np.float32))
                    has_lm.append(lm42 is not None)
                    presence_known.append(True)
                counts[key] = counts.get(key, 0) + 1
    if not imgs:
        print("[prepare] jester: nothing matched the five classes; check the CSVs")
        return
    shard.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(shard, imgs=np.array(imgs, dtype=np.uint8),
                        labels=np.array(labels), is_val=np.array(is_val),
                        landmarks=np.array(lms, dtype=np.float32),
                        has_lm=np.array(has_lm), presence_known=np.array(presence_known))
    positives = int(np.asarray(has_lm).sum())
    print(f"[prepare] jester: {len(imgs)} frames, {positives} teacher landmarks "
          f"from {sum(counts.values())} clips / {len(counts)} split-classes -> {shard}")


def _frames_ipn(cap: int):
    """IPN Hand (CC BY 4.0): 40 continuous 640x480 recordings already on disk.
    Frames are teacher-labelled like any pixel source; the last 8 videos are
    the val split (grouped by recording, so no temporal leakage)."""
    import cv2
    vids = sorted((IPN_DIR / "videos" / "videos").glob("*.avi"))
    per_video = max(1, cap // max(1, len(vids)))
    for vi, vid in enumerate(vids):
        cap_v = cv2.VideoCapture(str(vid))
        n = int(cap_v.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        want = set(np.linspace(0, n - 1, per_video, dtype=int).tolist())
        i = 0
        while True:
            ok, bgr = cap_v.read()
            if not ok:
                break
            if i in want:
                h, w = bgr.shape[:2]
                if max(h, w) > 384:
                    sc = 384 / max(h, w)
                    bgr = cv2.resize(bgr, (int(w * sc), int(h * sc)))
                yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), vi >= len(vids) - 8
            i += 1
        cap_v.release()


FRAME_GENERATORS = {  # yolo26 is special-cased in stage_prepare (ground truth)
    "crude": _frames_crude,
    "custom": _frames_custom,
    "hagrid": _frames_hagrid,
    "ipn": _frames_ipn,
    "nvgesture": lambda cap: _frames_local_videos("nvgesture", cap),
    "interhand26m": lambda cap: _frames_local_images("interhand26m", cap),
    "h2o": lambda cap: _frames_local_images("h2o", cap),
}


# ── stages ───────────────────────────────────────────────────────────────────

def _shard_path(source: str) -> Path:
    return DATASETS / source / "prepared.npz"


def stage_list(_args) -> None:
    print(f"{'source':14s} {'raw data':>10s} {'prepared':>9s} {'frames':>7s}  labels")
    for src in ALL_SOURCES:
        shard = _shard_path(src)
        raw = {"crude": any(_resolve_source(s) for s in CRUDE_SOURCES),
               "yolo26": (TRAIN_DIR / "yolo26" / "data.yaml").exists(),
               "ipn": (TRAIN_DIR / "insect-poc" / "raw" / "ipn" / "videos").exists(),
               "hagrid_shapes": True,   # cached zip, quota extraction on demand
               "swipe_phases": (DATASETS / "dynamic_gestures" / "models").exists(),
               "jester": ((DATASETS / "jester" / "raw" / "20bn-jester-v1").exists()
                          or (DATASETS / "20bn-jester" / "Train").exists()),
               "nvgesture": (DATASETS / "nvgesture" / "raw").exists(),
               "custom": any((DATASETS / "custom").rglob("*")) if (DATASETS / "custom").exists() else False,
               "hagrid": True,  # HF cache fetches on demand
               "interhand26m": (DATASETS / "interhand26m" / "images").exists(),
               "h2o": (DATASETS / "h2o" / "images").exists()}[src]
        if shard.exists():
            d = np.load(shard, allow_pickle=True)
            labels, counts = np.unique(d["labels"], return_counts=True)
            info = ", ".join(f"{l}={c}" for l, c in zip(labels, counts))
            print(f"{src:14s} {'yes' if raw else 'MANUAL':>10s} {'yes':>9s} {len(d['labels']):>7d}  {info}")
        else:
            print(f"{src:14s} {'yes' if raw else 'MANUAL':>10s} {'no':>9s} {'-':>7s}  "
                  f"{'(see datasets/README.md)' if not raw else ''}")


def stage_fetch(args) -> None:
    for src in args.sources:
        if src == "hagrid":
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(HAGRID_REPO, HAGRID_FILE, repo_type="dataset")
            print(f"[fetch] hagrid: cached at {p} (reused, not re-downloaded)")
        elif src in ("interhand26m", "h2o"):
            print(f"[fetch] {src}: registration-gated — manual steps in train/datasets/README.md")
        else:
            print(f"[fetch] {src}: local source, nothing to fetch")


def stage_prepare(args) -> None:
    teacher = None
    for src in args.sources:
        shard = _shard_path(src)
        if shard.exists() and not args.force:
            print(f"[prepare] {src}: prepared.npz exists — skipping (--force to redo)")
            continue
        if src == "yolo26":
            _prepare_yolo26(args.force)
            continue
        if src == "hagrid_shapes":
            _prepare_hagrid_shapes(args.force, args.teacher_conf)
            continue
        if src == "jester":
            _prepare_jester(args.force, args.teacher_conf)
            continue
        if teacher is None:
            teacher = Teacher(args.teacher_conf)
        import cv2
        imgs, labels, is_val, lms, has_lm = [], [], [], [], []
        t0 = time.perf_counter()
        for rgb, val in FRAME_GENERATORS[src](args.per_source):
            label, lm42 = teacher.label_lm(rgb)
            if label is None:
                continue
            imgs.append(cv2.resize(rgb, (IMG, IMG)))
            labels.append(label)
            is_val.append(val)
            lms.append(lm42 if lm42 is not None else np.zeros(42, np.float32))
            has_lm.append(lm42 is not None)
        if not imgs:
            print(f"[prepare] {src}: no frames found (see datasets/README.md)")
            continue
        shard.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(shard, imgs=np.array(imgs, dtype=np.uint8),
                            labels=np.array(labels), is_val=np.array(is_val),
                            landmarks=np.array(lms, dtype=np.float32),
                            has_lm=np.array(has_lm))
        u, c = np.unique(labels, return_counts=True)
        print(f"[prepare] {src}: {len(imgs)} frames in {time.perf_counter() - t0:.0f}s -> {shard}"
              f"\n          " + ", ".join(f"{l}={n}" for l, n in zip(u, c)))


def stage_train(args) -> None:
    """TinyNet v2: shared conv trunk + class head + LANDMARK head. The
    landmark head exists only at training time (the exported "probabilities"
    output is unchanged, so the viewer worker needs no edits): to regress 21
    landmark positions the trunk must learn WHERE the hand is — the
    localisation structure the yolo26 referee showed a plain classifier
    never acquires. Translation augmentation (with landmarks shifted to
    match) stops the trunk from memorising hand positions."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    shards = [(s_, _shard_path(s_)) for s_ in args.sources if _shard_path(s_).exists()]
    if not shards:
        sys.exit("[train] no prepared shards — run prepare first (./train/gesture-lab.sh prepare)")
    imgs, labels, is_val, lms, has_lm, used = [], [], [], [], [], []
    for s_, p_ in shards:
        d = np.load(p_, allow_pickle=True)
        n = len(d["labels"])
        imgs.append(d["imgs"]); labels.append(d["labels"]); is_val.append(d["is_val"])
        if "landmarks" in d.files:
            lms.append(d["landmarks"]); has_lm.append(d["has_lm"])
        else:   # pre-v2 shard: usable, just without aux supervision
            lms.append(np.zeros((n, 42), np.float32)); has_lm.append(np.zeros(n, bool))
        used.append(f"{s_}({n})")
    x = np.concatenate(imgs).astype(np.float32) / 255.0 - 0.5
    x = x.transpose(0, 3, 1, 2)
    labels = np.concatenate(labels)
    y = np.array([CLASSES.index(c) for c in labels], dtype=np.int64)
    is_val = np.concatenate(is_val)
    lms = np.concatenate(lms)
    has_lm = np.concatenate(has_lm)
    print(f"[train] {len(y)} frames from {' + '.join(used)}; "
          f"{int(has_lm.sum())} with landmark supervision")

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    class Net(nn.Module):
        def __init__(self, n_classes: int) -> None:
            super().__init__()
            def block(cin, cout):
                return nn.Sequential(nn.Conv2d(cin, cout, 3, 2, 1),
                                     nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
            self.trunk = nn.Sequential(block(3, 16), block(16, 32),
                                       block(32, 64), block(64, 128),
                                       nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.cls = nn.Linear(128, n_classes)
            self.lm = nn.Linear(128, 42)

        def forward(self, xb):
            f = self.trunk(xb)
            return self.cls(f), self.lm(f)

    net = Net(len(CLASSES)).to(device)
    n_params = sum(p_.numel() for p_ in net.parameters())
    counts = np.bincount(y[~is_val], minlength=len(CLASSES)).astype(np.float32)
    weights = torch.tensor((counts.sum() / np.maximum(counts, 1)) ** 0.5, device=device)
    ce = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    xtr = torch.tensor(x[~is_val]); ytr = torch.tensor(y[~is_val])
    ltr = torch.tensor(lms[~is_val]); htr = torch.tensor(has_lm[~is_val])
    xva = torch.tensor(x[is_val]).to(device); yva = torch.tensor(y[is_val]).to(device)
    flip_map = torch.tensor([CLASSES.index(FLIP_SWAP.get(c, c)) for c in CLASSES])

    t0 = time.perf_counter()
    best_acc, best_state = 0.0, None
    for _ in range(args.epochs):
        net.train()
        perm = torch.randperm(len(xtr))
        for b in range(0, len(perm), 64):
            idx = perm[b:b + 64]
            xb, yb = xtr[idx].clone(), ytr[idx].clone()
            lb, hb = ltr[idx].clone(), htr[idx].clone()
            # horizontal flip: swap direction labels AND mirror landmark x
            flip = torch.rand(len(xb)) < 0.5
            xb[flip] = torch.flip(xb[flip], dims=[3])
            yb[flip] = flip_map[yb[flip]]
            lbv = lb.view(-1, 21, 2)
            lbv[flip, :, 0] = 1.0 - lbv[flip, :, 0]
            lb = lbv.view(-1, 42)
            # translation aug (localisation must generalise): shift up to ±12px,
            # landmarks shifted identically in normalised units
            dx = int(torch.randint(-12, 13, (1,))); dy = int(torch.randint(-12, 13, (1,)))
            xb = torch.roll(xb, shifts=(dy, dx), dims=(2, 3))
            lbv = lb.view(-1, 21, 2)
            lbv[:, :, 0] += dx / xb.shape[3]
            lbv[:, :, 1] += dy / xb.shape[2]
            lb = lbv.view(-1, 42)
            # photometric: brightness/contrast/noise
            xb = xb * (0.8 + 0.4 * torch.rand(len(xb), 1, 1, 1)) \
                 + (torch.rand(len(xb), 1, 1, 1) - 0.5) * 0.2
            xb = (xb + torch.randn_like(xb) * 0.02).contiguous()
            opt.zero_grad()
            logits, lm_pred = net(xb.to(device))
            loss = ce(logits, yb.to(device))
            if bool(hb.any()):
                m = hb.to(device)
                loss = loss + F.smooth_l1_loss(lm_pred[m], lb.to(device)[m])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            acc = float((net(xva)[0].argmax(1) == yva).float().mean())
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    print(f"[train] v2 (landmark-aux) {n_params / 1e3:.0f}k params, {args.epochs} epochs "
          f"in {time.perf_counter() - t0:.0f}s on {device}; best val acc {best_acc:.3f}")

    class Export(nn.Module):     # same single "probabilities" output as v1
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, xb):
            return torch.softmax(self.inner(xb)[0], dim=1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(Export(net.cpu().eval()), torch.zeros(1, 3, IMG, IMG),
                      str(OUT_DIR / "model.onnx"),
                      input_names=["frames"], output_names=["probabilities"],
                      dynamic_axes={"frames": {0: "n"}, "probabilities": {0: "n"}},
                      dynamo=False)
    meta = {
        "classes": CLASSES, "input_px": IMG, "params": n_params,
        "precision": "fp32 (model.onnx)", "val_acc": best_acc,
        "teacher": "mediapipe landmarker + shape SVM + direction geometry",
        "architecture": "v2: conv trunk + class head + landmark-aux head (train-time only)",
        "sources": used, "trained_by": "train/gesture_lab.py",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[train] saved {OUT_DIR / 'model.onnx'} "
          f"({(OUT_DIR / 'model.onnx').stat().st_size // 1024} KB)")


def stage_eval(args) -> None:
    import subprocess
    if "crude" in args.sources:
        subprocess.run([sys.executable, str(TRAIN_DIR / "tiny_gesture.py"), "eval"], check=False)
    if "yolo26" in args.sources and YOLO26_DIR.exists():
        stage_eval_yolo26(args)


# ── yolo26: the user's Roboflow ground-truth dataset (2026-02-11) ────────────
# 7 classes with REAL labels — no teacher needed for training, and its
# test/valid splits are the referee both variants are scored against.
# Left/Right convention verified to match ours empirically (2026-07-12):
# teacher geometry on its 'Left' images -> LEFT 31/40, 'Right' -> RIGHT 26/40.
YOLO26_DIR = TRAIN_DIR / "yolo26"
YOLO26_NAMES = ["Down", "Left", "Right", "Stop", "Thumbs Down", "Thumbs up", "Up"]
YOLO26_TO_CLASS = {  # 'Thumbs Down' is deliberately absent: unknown gesture,
    "Down": "point_vertical", "Up": "point_vertical",       # kept OUT of training
    "Left": "point_left", "Right": "point_right",           # so it stays an OOV probe
    "Stop": "stop", "Thumbs up": "like",
}
YOLO26_EXPECTED = {"Stop": "STOP", "Thumbs up": "TROT", "Left": "LEFT",
                   "Right": "RIGHT", "Up": "NOOP", "Down": "NOOP",
                   "Thumbs Down": "ABSTAIN"}   # unknown gesture must not command


def _yolo26_items(split: str):
    import cv2
    img_dir, lbl_dir = YOLO26_DIR / split / "images", YOLO26_DIR / split / "labels"
    for lbl in sorted(lbl_dir.glob("*.txt")):
        toks = lbl.read_text().split()
        if not toks:
            continue
        img = cv2.imread(str(img_dir / (lbl.stem + ".jpg")))
        if img is None:
            continue
        yield cv2.cvtColor(img, cv2.COLOR_BGR2RGB), YOLO26_NAMES[int(toks[0])]


def _prepare_yolo26(force: bool) -> None:
    """Ground-truth shard: train split -> training rows, valid split -> val
    rows. The test split is NEVER trained on — it stays the referee."""
    import cv2
    shard = _shard_path("yolo26")
    if shard.exists() and not force:
        print("[prepare] yolo26: prepared.npz exists — skipping (--force to redo)")
        return
    teacher = Teacher(0.85)   # labels stay ground truth; teacher supplies landmarks only
    imgs, labels, is_val, lms, has_lm = [], [], [], [], []
    for split, val in (("train", False), ("valid", True)):
        for rgb, name in _yolo26_items(split):
            cls = YOLO26_TO_CLASS.get(name)
            if cls is None:
                continue
            _, lm42 = teacher.label_lm(rgb)
            imgs.append(cv2.resize(rgb, (IMG, IMG)))
            labels.append(cls)
            is_val.append(val)
            lms.append(lm42 if lm42 is not None else np.zeros(42, np.float32))
            has_lm.append(lm42 is not None)
    shard.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(shard, imgs=np.array(imgs, dtype=np.uint8),
                        labels=np.array(labels), is_val=np.array(is_val),
                        landmarks=np.array(lms, dtype=np.float32),
                        has_lm=np.array(has_lm))
    u, c = np.unique(labels, return_counts=True)
    print(f"[prepare] yolo26: {len(imgs)} ground-truth frames -> {shard}\n          "
          + ", ".join(f"{l}={n}" for l, n in zip(u, c)))


def stage_eval_yolo26(args) -> None:
    """Score BOTH variants on yolo26 test+valid against ground truth.
    'Thumbs Down' is the out-of-vocabulary probe: anything but abstention/noop
    on it is a wrong action."""
    import cv2
    import onnxruntime as ort

    tiny_meta = json.loads((OUT_DIR / "meta.json").read_text())
    tiny = ort.InferenceSession(str(OUT_DIR / "model.onnx"), providers=["CPUExecutionProvider"])
    tiny_classes = tiny_meta["classes"]
    teacher = Teacher(args.teacher_conf)     # the baseline IS the teacher pipeline
    cls_to_outcome = {"stop": "STOP", "fist": "HOME", "like": "TROT",
                      "point_left": "LEFT", "point_right": "RIGHT",
                      "point_vertical": "NOOP", "no_hand": "ABSTAIN"}

    def tiny_outcome(rgb) -> str:
        x = (cv2.resize(rgb, (IMG, IMG)).astype(np.float32) / 255.0 - 0.5)
        proba = tiny.run(["probabilities"], {"frames": x.transpose(2, 0, 1)[None]})[0][0]
        if float(proba.max()) < args.teacher_conf:
            return "ABSTAIN"
        return cls_to_outcome[tiny_classes[int(proba.argmax())]]

    def base_outcome(rgb) -> str:
        label = teacher.label(rgb)
        return "ABSTAIN" if label is None else cls_to_outcome[label]

    items = [(rgb, name) for split in ("test", "valid") for rgb, name in _yolo26_items(split)]
    print(f"\n[eval-yolo26] {len(items)} ground-truth images (test+valid), both variants")
    for variant, fn in (("TinyNet", tiny_outcome), ("MediaPipe+SVM", base_outcome)):
        per: dict[str, list[int]] = {}      # gt -> [hits, wrong-actions, total]
        t0 = time.perf_counter()
        for rgb, name in items:
            out = fn(rgb)
            want = YOLO26_EXPECTED[name]
            hit = out == want or (want in ("NOOP", "ABSTAIN") and out in ("NOOP", "ABSTAIN"))
            wrong = out in ("STOP", "HOME", "TROT", "LEFT", "RIGHT") and out != want
            h, w_, t = per.get(name, [0, 0, 0])
            per[name] = [h + hit, w_ + wrong, t + 1]
        ms = (time.perf_counter() - t0) * 1000 / len(items)
        hits = sum(v[0] for v in per.values()); wrongs = sum(v[1] for v in per.values())
        total = sum(v[2] for v in per.values())
        print(f"\n  {variant}: {hits}/{total} correct ({hits / total:.0%}), "
              f"{wrongs / total:.1%} wrong-action, {ms:.1f}ms/frame")
        for name in YOLO26_NAMES:
            if name in per:
                h, w_, t = per[name]
                print(f"    {name:12s} want {YOLO26_EXPECTED[name]:7s} "
                      f"{h:>3}/{t:<3} hit  {w_:>2} wrong-action")


# ══ TinyNet v3: detect -> crop -> classify -> sequence (the MediaPipe recipe,
#    distilled 100x smaller). 6 towers: palm fist like point_right point_left
#    noop. Direction classes are safe here because the classifier sees a
#    CANONICAL hand crop (mirror-clean shapes), not a whole scene. ═══════════

CLASSES6 = ["palm", "fist", "like", "point_right", "point_left", "noop"]
FLIP6 = {"point_right": "point_left", "point_left": "point_right"}
LABEL_TO_6 = {  # v2 shard vocabulary -> the 6 towers
    "stop": "palm", "fist": "fist", "like": "like",
    "point_right": "point_right", "point_left": "point_left",
    "point_vertical": "noop", "no_hand": "noop",
}
CROP = 64          # canonical hand-crop size
SEQ_T = 8          # trajectory window (frames) for the sequence head
V3_INTENT = {"palm": "STOP", "fist": "HOME", "like": "TROT",
             "point_left": "LEFT", "point_right": "RIGHT", "noop": ""}


def _lm_crop_box(lm42: np.ndarray, margin: float = 0.35) -> tuple[float, float, float]:
    """Square crop (cx, cy, side) in normalised coords around the landmarks."""
    xs, ys = lm42[0::2], lm42[1::2]
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + margin)
    return float(cx), float(cy), float(max(side, 0.05))


def _take_crop(img: np.ndarray, cx: float, cy: float, side: float) -> np.ndarray:
    import cv2
    h, w = img.shape[:2]
    x1 = int(max(0, (cx - side / 2) * w)); x2 = int(min(w, (cx + side / 2) * w))
    y1 = int(max(0, (cy - side / 2) * h)); y2 = int(min(h, (cy + side / 2) * h))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return cv2.resize(img, (CROP, CROP))
    return cv2.resize(img[y1:y2, x1:x2], (CROP, CROP))


def stage_train_v3(args) -> None:
    """Three tiny graphs from the v2 shards:
      trunk.onnx  128² frame -> (landmarks 42, presence 1)     ~0.5ms
      crop.onnx   64² hand crop -> 6-tower probabilities        ~0.2ms
      seq.onnx    SEQ_T×42 landmark trajectory -> 6 towers      ~0.02ms
    The crop classifier is where accuracy comes from (canonical hands);
    the sequence head smooths/temporally votes over the student's OWN
    landmark stream — MediaPipe is never needed at runtime."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    shards = [(s_, _shard_path(s_)) for s_ in args.sources if _shard_path(s_).exists()]
    if not shards:
        sys.exit("[train-v3] no prepared shards")
    imgs, labels, is_val, lms, has_lm, seq_bounds, used = [], [], [], [], [], [], []
    offset = 0
    for s_, p_ in shards:
        d = np.load(p_, allow_pickle=True)
        if "landmarks" not in d.files:
            print(f"[train-v3] {s_}: pre-v2 shard (no landmarks) — re-prepare with --force; skipping")
            continue
        sel = np.arange(len(d["labels"]))
        if args.data_fraction < 1.0:
            # stride over TRAIN rows: keeps every source, every class mix and
            # the temporal spread — variety survives, volume halves. Val rows
            # stay whole so metrics remain comparable across fractions.
            tr = np.where(~d["is_val"])[0]
            va = np.where(d["is_val"])[0]
            stride = max(1, int(round(1.0 / args.data_fraction)))
            sel = np.sort(np.concatenate([tr[::stride], va]))
        n = len(sel)
        imgs.append(d["imgs"][sel]); labels.append(d["labels"][sel]); is_val.append(d["is_val"][sel])
        lms.append(d["landmarks"][sel]); has_lm.append(d["has_lm"][sel])
        if s_ in ("crude", "ipn", "custom", "nvgesture", "jester", "swipe_phases"):   # temporally ordered sources
            seq_bounds.append((offset, offset + n))   # strided rows still carry motion
        offset += n
        used.append(f"{s_}({n})")
    imgs = np.concatenate(imgs); labels = np.concatenate(labels)
    is_val = np.concatenate(is_val); lms = np.concatenate(lms); has_lm = np.concatenate(has_lm)
    y6 = np.array([CLASSES6.index(LABEL_TO_6[c]) for c in labels], dtype=np.int64)
    print(f"[train-v3] {len(y6)} frames from {' + '.join(used)}; "
          f"{int(has_lm.sum())} with landmarks")

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, 2, 1),
                             nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    class Trunk(nn.Module):     # 128² frame -> landmarks + presence
        def __init__(self, w: int = 1):
            super().__init__()
            self.body = nn.Sequential(block(3, 16 * w), block(16 * w, 32 * w),
                                      block(32 * w, 64 * w), block(64 * w, 128 * w),
                                      nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.lm = nn.Linear(128 * w, 42)
            self.presence = nn.Linear(128 * w, 1)

        def forward(self, xb):
            f = self.body(xb)
            return self.lm(f), self.presence(f)

    class CropCls(nn.Module):   # 64² canonical crop -> 6 towers
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(block(3, 16), block(16, 32), block(32, 64),
                                     nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                     nn.Linear(64, len(CLASSES6)))

        def forward(self, xb):
            return self.net(xb)

    class SeqHead(nn.Module):   # SEQ_T x 42 trajectory -> 6 towers
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(SEQ_T * 42, 128),
                                     nn.ReLU(inplace=True), nn.Linear(128, len(CLASSES6)))

        def forward(self, xb):
            return self.net(xb)

    x_full = torch.tensor(imgs.astype(np.float32) / 255.0 - 0.5).permute(0, 3, 1, 2)
    lm_t = torch.tensor(lms); hl_t = torch.tensor(has_lm); y_t = torch.tensor(y6)
    val_t = torch.tensor(is_val)

    # ── 1) trunk: landmarks + presence ──────────────────────────────────────
    trunk = Trunk(args.trunk_width).to(device)
    opt = torch.optim.Adam(trunk.parameters(), lr=3e-4)
    tr_idx = torch.nonzero(~val_t).squeeze(1)
    t0 = time.perf_counter()
    for _ in range(args.epochs):
        trunk.train()
        for b in torch.split(tr_idx[torch.randperm(len(tr_idx))], 64):
            xb = x_full[b].clone(); lb = lm_t[b].clone(); hb = hl_t[b]
            dx = int(torch.randint(-12, 13, (1,))); dy = int(torch.randint(-12, 13, (1,)))
            xb = torch.roll(xb, (dy, dx), dims=(2, 3))
            lb = lb.view(-1, 21, 2)
            lb[:, :, 0] += dx / 128; lb[:, :, 1] += dy / 128
            lb = lb.view(-1, 42)
            xb = (xb * (0.8 + 0.4 * torch.rand(len(xb), 1, 1, 1))
                  + torch.randn_like(xb) * 0.02).contiguous()
            opt.zero_grad()
            lm_p, pres = trunk(xb.to(device))
            loss = F.binary_cross_entropy_with_logits(pres.squeeze(1), hb.float().to(device))
            if bool(hb.any()):
                m = hb.to(device)
                loss = loss + F.smooth_l1_loss(lm_p[m], lb.to(device)[m]) * 4.0
            loss.backward(); opt.step()
    print(f"[train-v3] trunk done in {time.perf_counter() - t0:.0f}s")

    # ── 2) crop classifier on the TRUNK'S OWN predicted crops ───────────────
    # Self-consistency is the point: at inference the crop comes from the
    # trunk's landmarks, so training on teacher-landmark crops creates a
    # train/test mismatch (a slightly-off crop gets confidently misread).
    # Teacher landmarks still supervised the trunk; the crop tower learns to
    # read whatever the trunk actually serves it.
    import cv2
    trunk.eval()
    pred_lms = []
    with torch.no_grad():
        for b in torch.split(torch.arange(len(x_full)), 256):
            lm_p, _ = trunk(x_full[b].to(device))
            pred_lms.append(lm_p.cpu().numpy())
    pred_lms = np.concatenate(pred_lms)
    crops = np.zeros((len(imgs), CROP, CROP, 3), np.uint8)
    for i in range(len(imgs)):
        if has_lm[i]:
            cx, cy, side = _lm_crop_box(pred_lms[i])
            crops[i] = _take_crop(imgs[i], cx, cy, side)
        else:      # background crop -> teaches the tower to answer noop
            crops[i] = cv2.resize(imgs[i], (CROP, CROP))
    xc = torch.tensor(crops.astype(np.float32) / 255.0 - 0.5).permute(0, 3, 1, 2)
    flip6 = torch.tensor([CLASSES6.index(FLIP6.get(c, c)) for c in CLASSES6])
    cls = CropCls().to(device)
    counts = np.bincount(y6[~is_val], minlength=len(CLASSES6)).astype(np.float32)
    ce = nn.CrossEntropyLoss(weight=torch.tensor(
        (counts.sum() / np.maximum(counts, 1)) ** 0.5, device=device))
    opt = torch.optim.Adam(cls.parameters(), lr=3e-4)
    best_acc, best_state = 0.0, None
    xva = xc[val_t].to(device); yva = y_t[val_t].to(device)
    t0 = time.perf_counter()
    for _ in range(args.epochs):
        cls.train()
        for b in torch.split(tr_idx[torch.randperm(len(tr_idx))], 64):
            xb, yb = xc[b].clone(), y_t[b].clone()
            flip = torch.rand(len(xb)) < 0.5
            xb[flip] = torch.flip(xb[flip], dims=[3])
            yb[flip] = flip6[yb[flip]]
            xb = (xb * (0.8 + 0.4 * torch.rand(len(xb), 1, 1, 1))
                  + torch.randn_like(xb) * 0.02).contiguous()
            opt.zero_grad()
            ce(cls(xb.to(device)), yb.to(device)).backward()
            opt.step()
        cls.eval()
        with torch.no_grad():
            acc = float((cls(xva).argmax(1) == yva).float().mean())
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.detach().cpu().clone() for k, v in cls.state_dict().items()}
    cls.load_state_dict(best_state)
    print(f"[train-v3] crop classifier done in {time.perf_counter() - t0:.0f}s; "
          f"val acc on canonical crops {best_acc:.3f}")

    # ── 3) sequence head over landmark trajectories (ordered sources) ───────
    wins, wy = [], []
    for a, b in seq_bounds:
        for i in range(a, b - SEQ_T):
            sl = slice(i, i + SEQ_T)
            if has_lm[sl].all() and len(set(labels[sl].tolist())) == 1 and not is_val[sl].any():
                wins.append(lms[sl]); wy.append(y6[i])
    seq = SeqHead().to(device)
    if wins:
        xw = torch.tensor(np.stack(wins)); yw = torch.tensor(np.array(wy))
        opt = torch.optim.Adam(seq.parameters(), lr=3e-4)
        for _ in range(args.epochs):
            seq.train()
            perm = torch.randperm(len(xw))
            for b in torch.split(perm, 64):
                opt.zero_grad()
                ce(seq(xw[b].to(device)), yw[b].to(device)).backward()
                opt.step()
        print(f"[train-v3] sequence head done ({len(wins)} trajectory windows)")
    else:
        print("[train-v3] WARNING: no trajectory windows found; sequence head untrained")

    # ── export the three graphs ──────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, net, dummy, outs in (
            ("trunk", trunk, torch.zeros(1, 3, IMG, IMG), ["landmarks", "presence"]),
            ("crop", nn.Sequential(cls, nn.Softmax(dim=1)), torch.zeros(1, 3, CROP, CROP), ["probabilities"]),
            ("seq", nn.Sequential(seq, nn.Softmax(dim=1)), torch.zeros(1, SEQ_T, 42), ["probabilities"])):
        torch.onnx.export(net.cpu().eval(), dummy, str(OUT_DIR / f"{name}.onnx"),
                          input_names=["frames"], output_names=outs,
                          dynamic_axes={"frames": {0: "n"}}, dynamo=False)
    n_params = sum(p_.numel() for m in (trunk, cls, seq) for p_ in m.parameters())
    meta = {
        "version": 3, "classes": CLASSES6, "input_px": IMG, "crop_px": CROP,
        "seq_t": SEQ_T, "params": n_params, "precision": "fp32 (3 graphs)",
        "crop_val_acc": best_acc, "sources": used,
        "architecture": "v3 detect->crop->classify + landmark-trajectory sequence head",
        "trained_by": "train/gesture_lab.py train-v3",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[train-v3] saved trunk/crop/seq .onnx ({n_params / 1e3:.0f}k params total)")


class TinyV3Runtime:
    """Numpy/onnxruntime inference for the v3 stack — shared by the eval
    referee and the viewer worker so they can never diverge."""

    def __init__(self, run_dir: Path | None = None, conf: float = 0.6,
                 presence_gate: float = 0.5) -> None:
        import onnxruntime as ort
        d = run_dir or OUT_DIR
        opt = {"providers": ["CPUExecutionProvider"]}
        self.trunk = ort.InferenceSession(str(d / "trunk.onnx"), **opt)
        self.crop = ort.InferenceSession(str(d / "crop.onnx"), **opt)
        self.seq = ort.InferenceSession(str(d / "seq.onnx"), **opt)
        self.meta = json.loads((d / "meta.json").read_text())
        self.conf, self.presence_gate = conf, presence_gate
        self.ring: list[np.ndarray] = []

    def step(self, rgb: np.ndarray) -> tuple[str, float, np.ndarray | None]:
        """One frame -> (tower, confidence, predicted landmarks or None)."""
        import cv2
        x = (cv2.resize(rgb, (IMG, IMG)).astype(np.float32) / 255.0 - 0.5)
        lm42, pres = self.trunk.run(["landmarks", "presence"],
                                    {"frames": x.transpose(2, 0, 1)[None]})
        lm42 = lm42[0]
        if 1 / (1 + np.exp(-float(pres[0][0]))) < self.presence_gate:
            self.ring.clear()
            return "noop", 1.0, None
        cx, cy, side = _lm_crop_box(lm42)
        crop = _take_crop(rgb, cx, cy, side)
        xc = (cv2.resize(crop, (CROP, CROP)).astype(np.float32) / 255.0 - 0.5)
        probs = self.crop.run(["probabilities"], {"frames": xc.transpose(2, 0, 1)[None]})[0][0]
        self.ring.append(lm42)
        if len(self.ring) > SEQ_T:
            self.ring.pop(0)
        if len(self.ring) == SEQ_T:      # temporal vote once the window fills
            traj = np.stack(self.ring)[None].astype(np.float32)
            probs = (probs + self.seq.run(["probabilities"], {"frames": traj})[0][0]) / 2
        top = int(probs.argmax())
        tower, p = CLASSES6[top], float(probs[top])
        if p < self.conf:
            return "noop", p, lm42
        return tower, p, lm42


class HgDetRuntime:
    """Detector A/B front-end: dynamic_gestures' 1.2MB hand detector (2.35ms)
    localises; OUR v3 crop tower classifies. No MediaPipe anywhere — this is
    the ms-reduction experiment for scenes their detector can see."""

    STATUS_PREFIX = "HGDT"

    def __init__(self, conf: float = 0.6, det_thr: float = 0.4) -> None:
        import onnxruntime as ort
        mdir = DATASETS / "dynamic_gestures" / "models"
        self.det = ort.InferenceSession(str(mdir / "hand_detector.onnx"),
                                        providers=["CPUExecutionProvider"])
        self.crop = ort.InferenceSession(str(OUT_DIR / "crop.onnx"),
                                         providers=["CPUExecutionProvider"])
        self.conf, self.det_thr = conf, det_thr

    def step(self, rgb: np.ndarray):
        import cv2
        x = cv2.resize(rgb, (320, 240)).astype(np.float32)
        x = ((x - 127.0) / 128.0).transpose(2, 0, 1)[None]
        boxes, _labels, scores = self.det.run(None, {"input": x})
        keep = np.where(scores >= self.det_thr)[0]
        if not len(keep):
            return "noop", 1.0, None
        b = boxes[keep[int(scores[keep].argmax())]].astype(np.float32)
        if b.max() > 1.5:                      # pixel coords -> normalise
            b = b / np.array([320, 240, 320, 240], dtype=np.float32)
        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = (np.clip(b, 0, 1) * np.array([w, h, w, h])).astype(int)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return "noop", 1.0, None
        crop = cv2.resize(rgb[y1:y2, x1:x2], (CROP, CROP))
        xc = (crop.astype(np.float32) / 255.0 - 0.5).transpose(2, 0, 1)[None]
        probs = self.crop.run(["probabilities"], {"frames": xc})[0][0]
        top = int(probs.argmax())
        tower, p = CLASSES6[top], float(probs[top])
        if p < self.conf:
            return "noop", p, tuple(np.clip(b, 0, 1).tolist())
        return tower, p, tuple(np.clip(b, 0, 1).tolist())


def stage_eval_v3(args) -> None:
    """The same yolo26 referee, v3 runtime path (single stills: crop tower
    only — the sequence head needs a live stream to contribute)."""
    rt = TinyV3Runtime(conf=args.teacher_conf if args.teacher_conf < 0.84 else 0.6)
    expected = {"Stop": "STOP", "Thumbs up": "TROT", "Left": "LEFT", "Right": "RIGHT",
                "Up": "ABSTAIN", "Down": "ABSTAIN", "Thumbs Down": "ABSTAIN"}
    per: dict[str, list[int]] = {}
    times = []
    for split in ("test", "valid"):
        for rgb, name in _yolo26_items(split):
            rt.ring.clear()              # stills: no fake trajectories
            t0 = time.perf_counter()
            tower, p, _ = rt.step(rgb)
            times.append((time.perf_counter() - t0) * 1000)
            out = V3_INTENT.get(tower, "") or "ABSTAIN"
            want = expected[name]
            hit = out == want or (want == "ABSTAIN" and out == "ABSTAIN")
            wrong = out in ("STOP", "HOME", "TROT", "LEFT", "RIGHT") and out != want
            h, w_, t = per.get(name, [0, 0, 0])
            per[name] = [h + hit, w_ + wrong, t + 1]
    hits = sum(v[0] for v in per.values()); wrongs = sum(v[1] for v in per.values())
    total = sum(v[2] for v in per.values())
    times.sort()
    print(f"\n[eval-v3] yolo26 referee: {hits}/{total} correct ({hits / total:.0%}), "
          f"{wrongs / total:.1%} wrong-action, {times[len(times) // 2]:.2f}ms/frame median")
    for name, (h, w_, t) in per.items():
        print(f"    {name:12s} want {expected[name]:8s} {h:>3}/{t:<3} hit  {w_:>2} wrong-action")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["list", "fetch", "prepare", "train", "train-v3", "eval", "eval-v3", "all"])
    p.add_argument("--source", action="append", dest="sources", default=None,
                   help="restrict to specific source(s); default: all auto sources")
    p.add_argument("--per-source", type=int, default=4000,
                   help="max frames sampled per source before teacher labelling")
    p.add_argument("--teacher-conf", type=float, default=0.85)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--force", action="store_true", help="re-prepare even if a shard exists")
    p.add_argument("--trunk-width", type=int, default=1,
                   help="v3 trunk channel multiplier (2 = ~4x params; the overnight run)")
    p.add_argument("--data-fraction", type=float, default=1.0,
                   help="stride-subsample each shard's TRAIN rows to this fraction "
                        "(val rows untouched; every source kept -> variety preserved)")
    args = p.parse_args()
    if args.sources is None:
        args.sources = list(ALL_SOURCES)
    if args.stage in ("list",):
        stage_list(args)
    if args.stage in ("fetch", "all"):
        stage_fetch(args)
    if args.stage in ("prepare", "all"):
        stage_prepare(args)
    if args.stage in ("train", "all"):
        stage_train(args)
    if args.stage == "train-v3":
        stage_train_v3(args)
    if args.stage == "eval-v3":
        stage_eval_v3(args)
    if args.stage in ("eval", "all"):
        stage_eval(args)


if __name__ == "__main__":
    main()
