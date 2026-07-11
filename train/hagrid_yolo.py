"""Reusable HaGRID -> YOLO training pipeline for OpenPAVE gesture intents.

Trains ANY Ultralytics YOLO checkpoint (yolo26n, yolo11n, yolov8n, ...) on a
pruned slice of the HaGRID hand-gesture dataset: N gesture classes of your
choosing plus the implicit `no_gesture` distractor class.

The heavy source archive is fetched once via huggingface_hub and lives in the
same HF cache as the OpenPAVE VLM checkpoints (~/.cache/huggingface, or
$HF_HOME — see pave_mlx/downloads.py:hf_cache_dir). Only the images actually
selected for training are extracted, into train/HaGRID/.

Stages (run via train/HaGRID.sh or directly):

  prepare  download archive -> extract selected classes -> YOLO labels + data.yaml
  train    fine-tune a YOLO checkpoint on the prepared dataset
  export   export best.pt to ONNX (Mac benchmarking) and NCNN (Mali deployment)
  bench    time single-image inference on val images (the 20-40ms question)
  all      prepare + train + export

Slashing data / pruning complexity:
  --classes    fewer classes = smaller head + less data. Default is 5 proxies
               for the 5 OpenPAVE intents (pave_ui/perception.py intent map).
  --per-class  hard cap on images per class; 300 is plenty for a proof.
  --imgsz      320 (or below) quarters the compute vs the 640 default.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parent
DATASET_DIR = TRAIN_DIR / "HaGRID"
RUNS_DIR = TRAIN_DIR / "runs"
WEIGHTS_DIR = TRAIN_DIR / "weights"  # pretrained .pt downloads land here

HF_DATASET_REPO = "cj-mills/hagrid-sample-30k-384p"
HF_DATASET_FILE = "hagrid-sample-30k-384p.zip"

# HaGRID gesture classes for the OpenPAVE intents (perception.py
# _GESTURE_SYNONYMS). HaGRID has no directional point classes, but its "one"
# class IS an index finger pointing up — the other three directions are
# synthesized by rotating DISJOINT slices of "one" (image + bboxes together).
# Spec syntax: plain "stop" uses the HaGRID class as-is;
# "alias=source@deg" derives a new class by rotating `source` frames `deg`
# degrees CLOCKWISE. Swap in real captured data later just by changing specs.
DEFAULT_CLASSES = (
    "stop,fist,like,"
    "point_up=one@0,point_right=one@90,point_down=one@180,point_left=one@270"
)
INTENT_HINTS = {
    "stop": "STOP (open palm, raised)",
    "fist": "HOME",
    "like": "TROT (thumbs-up)",
    "point_up": "UP (index finger up — HaGRID 'one' as-is)",
    "point_right": "RIGHT (synthetic: 'one' rotated 90° CW)",
    "point_down": "DOWN (synthetic: 'one' rotated 180°)",
    "point_left": "LEFT (synthetic: 'one' rotated 90° CCW)",
}
NO_GESTURE = "no_gesture"


def _parse_class_specs(arg: str) -> list[tuple[str, str, int]]:
    """'stop,point_right=one@90' -> [(name, hagrid_source_class, cw_degrees)]."""
    specs = []
    for tok in (t.strip() for t in arg.split(",")):
        if not tok:
            continue
        name, src, rot = tok, tok, 0
        if "=" in tok:
            name, src = (s.strip() for s in tok.split("=", 1))
        if "@" in src:
            src, deg = src.split("@", 1)
            rot = int(deg) % 360
        if rot not in (0, 90, 180, 270):
            sys.exit(f"[prepare] rotation must be 0/90/180/270, got {rot} in {tok!r}")
        specs.append((name, src, rot))
    return specs


def _rot_bbox(bbox: list[float], rot: int) -> list[float]:
    """Rotate a normalized [top-left x, y, w, h] bbox with its frame, `rot` CW."""
    x, y, w, h = bbox
    if rot == 90:
        return [1 - y - h, x, h, w]
    if rot == 180:
        return [1 - x - w, 1 - y - h, w, h]
    if rot == 270:
        return [y, 1 - x - w, h, w]
    return bbox


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["prepare", "train", "export", "bench", "all"])
    p.add_argument("--classes", default=DEFAULT_CLASSES,
                   help="comma-separated HaGRID gesture classes (no_gesture is always appended)")
    p.add_argument("--per-class", type=int, default=300,
                   help="max images per class; slash this to shrink the dataset")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=22)
    p.add_argument("--model", default="yolo26n.pt",
                   help="any ultralytics checkpoint: yolo26n.pt, yolo11n.pt, yolov8n.pt, or a runs/*/best.pt")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="",
                   help="ultralytics device string; empty = auto (picks mps on Apple Silicon)")
    p.add_argument("--formats", default="onnx,ncnn", help="export formats, comma-separated")
    p.add_argument("--run-name", default="", help="training run name; default derives from model+imgsz")
    return p.parse_args(argv)


# ── prepare ──────────────────────────────────────────────────────────────────

def _fetch_archive() -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(HF_DATASET_REPO, HF_DATASET_FILE, repo_type="dataset"))


def _to_yolo_line(cls_idx: int, bbox: list[float]) -> str:
    # HaGRID bboxes are normalized [top-left x, top-left y, width, height];
    # YOLO wants normalized [center x, center y, width, height].
    x, y, w, h = bbox
    return f"{cls_idx} {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}"


def prepare(args: argparse.Namespace) -> Path:
    specs = _parse_class_specs(args.classes)
    names = [name for name, _, _ in specs] + [NO_GESTURE]
    cls_idx = {n: i for i, n in enumerate(names)}
    # rot==0 frames keep every selected upright class; rotated frames only keep
    # the class being synthesized (any other hand in a rotated frame is no
    # longer a valid example of its upright class -> distractor).
    upright_map = {src: name for name, src, rot in specs if rot == 0}
    rng = random.Random(args.seed)

    archive = _fetch_archive()
    print(f"[prepare] source archive: {archive}")

    zf = zipfile.ZipFile(archive)
    members = zf.namelist()
    ann_by_class = {Path(m).stem: m for m in members if m.endswith(".json") and "ann_train_val" in m}
    image_by_stem = {Path(m).stem: m for m in members if m.lower().endswith((".jpg", ".jpeg", ".png"))}
    missing = sorted({src for _, src, _ in specs if src not in ann_by_class})
    if missing:
        sys.exit(f"[prepare] classes not in archive: {missing}; available: {sorted(ann_by_class)}")

    for split in ("train", "val"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # classes sharing a HaGRID source (the four point_* from "one") get
    # DISJOINT image slices, so a val frame is never a rotation of a train one.
    by_source: dict[str, list[tuple[str, int]]] = {}
    for name, src, rot in specs:
        by_source.setdefault(src, []).append((name, rot))

    from io import BytesIO
    from PIL import Image
    pil_rot = {90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180,
               270: Image.Transpose.ROTATE_90}  # PIL rotates CCW; specs are CW

    counts: Counter[str] = Counter()
    box_counts: Counter[str] = Counter()
    for src, users in by_source.items():
        ann = json.loads(zf.read(ann_by_class[src]))
        stems = sorted(s for s in ann if s in image_by_stem)
        rng.shuffle(stems)
        per = args.per_class
        if len(stems) < per * len(users):
            per = len(stems) // len(users)
            print(f"[prepare] WARNING: only {len(stems)} '{src}' images for "
                  f"{len(users)} classes -> {per} per class")
        for u, (name, rot) in enumerate(users):
            chunk = stems[u * per: (u + 1) * per]
            n_val = max(1, int(len(chunk) * args.val_frac))
            for i, stem in enumerate(chunk):
                split = "val" if i < n_val else "train"
                lines = []
                for bbox, label in zip(ann[stem]["bboxes"], ann[stem]["labels"]):
                    if label == src:
                        cls = name
                    elif rot == 0:
                        cls = upright_map.get(label, NO_GESTURE)
                    else:
                        cls = NO_GESTURE
                    lines.append(_to_yolo_line(cls_idx[cls], _rot_bbox(bbox, rot)))
                    box_counts[cls] += 1
                img_out = DATASET_DIR / "images" / split / f"{name}_{stem}.jpg"
                if not img_out.exists():
                    raw = zf.read(image_by_stem[stem])
                    if rot == 0:
                        img_out.write_bytes(raw)
                    else:
                        Image.open(BytesIO(raw)).transpose(pil_rot[rot]).save(img_out, quality=92)
                (DATASET_DIR / "labels" / split / f"{name}_{stem}.txt").write_text("\n".join(lines) + "\n")
                counts[f"{name}/{split}"] += 1

    data_yaml = DATASET_DIR / "data.yaml"
    hints = "\n".join(f"#   {n}: {INTENT_HINTS.get(n, '(custom)')}" for n in names[:-1])
    data_yaml.write_text(
        "# Generated by train/hagrid_yolo.py prepare — do not hand-edit; re-run instead.\n"
        f"# OpenPAVE intent proxies (see pave_ui/perception.py):\n{hints}\n"
        f"path: {DATASET_DIR}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names: {json.dumps(names)}\n"
    )
    stats = {"images": dict(sorted(counts.items())), "boxes_per_class": dict(sorted(box_counts.items()))}
    (DATASET_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"[prepare] wrote {data_yaml}")
    print(json.dumps(stats, indent=2))
    return data_yaml


# ── train / export / bench ───────────────────────────────────────────────────

def _load_model(name: str):
    from ultralytics import YOLO
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(name)
    if not path.exists() and not path.is_absolute() and "/" not in name:
        path = WEIGHTS_DIR / name  # keep auto-downloaded pretrained weights in train/weights
    return YOLO(str(path) if path.exists() or path.parent == WEIGHTS_DIR else name)


def train(args: argparse.Namespace) -> Path:
    data_yaml = DATASET_DIR / "data.yaml"
    if not data_yaml.exists():
        prepare(args)
    model = _load_model(args.model)
    run_name = args.run_name or f"{Path(args.model).stem}_imgsz{args.imgsz}"
    results = model.train(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device or None,
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=True,
        patience=15,
        cache=True,  # dataset is tiny; keeping it in RAM removes the I/O tax per epoch
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"[train] best weights: {best}")
    return best


def export(args: argparse.Namespace, weights: Path | None = None) -> None:
    from ultralytics import YOLO
    weights = weights or _latest_best(args)
    print(f"[export] exporting {weights}")
    for fmt in [f.strip() for f in args.formats.split(",") if f.strip()]:
        try:
            kwargs = {"format": fmt, "imgsz": args.imgsz}
            if fmt == "onnx":
                kwargs["simplify"] = True
            if fmt == "ncnn":
                kwargs["half"] = True  # fp16 for the Mali/Vulkan path
            out = YOLO(str(weights)).export(**kwargs)
            print(f"[export] {fmt}: {out}")
        except Exception as exc:  # NCNN export is known-flaky on macOS arm64
            print(f"[export] {fmt} FAILED ({exc}); if this is ncnn on the Mac, "
                  f"convert the ONNX file with pnnx/onnx2ncnn on the Orion instead.")


def bench(args: argparse.Namespace, weights: Path | None = None) -> None:
    import time
    from ultralytics import YOLO
    weights = weights or _latest_best(args)
    imgs = sorted((DATASET_DIR / "images" / "val").glob("*.jpg"))[:50]
    if not imgs:
        sys.exit("[bench] no val images; run prepare first")
    model = YOLO(str(weights))
    model.predict(imgs[0], imgsz=args.imgsz, verbose=False)  # warmup
    times = []
    for img in imgs:
        t0 = time.perf_counter()
        model.predict(img, imgsz=args.imgsz, device="cpu", verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    print(f"[bench] {weights.name} @ {args.imgsz}px, CPU, {len(times)} images: "
          f"median {times[len(times) // 2]:.1f}ms, p90 {times[int(len(times) * 0.9)]:.1f}ms "
          f"(Mali proxy: aim for the low single digits here)")


def _latest_best(args: argparse.Namespace) -> Path:
    if args.model.endswith("best.pt") or args.model.endswith("last.pt"):
        return Path(args.model)
    candidates = sorted(RUNS_DIR.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        sys.exit(f"[error] no trained weights under {RUNS_DIR}; run the train stage first")
    return candidates[-1]


def main() -> None:
    args = parse_args()
    if args.stage in ("prepare", "all"):
        prepare(args)
    best = None
    if args.stage in ("train", "all"):
        best = train(args)
    if args.stage in ("export", "all"):
        export(args, best)
    if args.stage == "bench":
        bench(args)


if __name__ == "__main__":
    main()
