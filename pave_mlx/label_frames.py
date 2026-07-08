"""Shared, backend-agnostic frame labeler (one tool, not per-model — see §3.2).

Producing labeled `(frame -> intent)` data is the real bottleneck and it does not
depend on which encoder consumes it, so it lives here once. Two entry points:

    # ingest an existing folder of images under one label
    python -m pave_mlx.label_frames ingest --label TROT --src ~/clips/trot

    # capture from the Mac camera, press a key to label the current frame
    python -m pave_mlx.label_frames capture        # needs opencv-python

Output layout (consumed directly by train_intent_head.py):
    data/<LABEL>/<timestamp>.png

The only per-backend wrinkle is sampling: V-JEPA needs a short *window* of frames
per label, DINOv3 a single frame. Pass --window N for the windowed case.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from pave_mlx.heads.base import INTENT_LABELS

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


class LabelStore:
    def __init__(self, root: str | Path = "data"):
        self.root = Path(root)

    def dir_for(self, label: str) -> Path:
        d = self.root / label.upper()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def add_path(self, label: str, src: Path) -> Path:
        dst = self.dir_for(label) / f"{int(time.time()*1e6)}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        return dst

    def add_array(self, label: str, image_bgr) -> Path:
        import cv2  # only needed for camera capture

        dst = self.dir_for(label) / f"{int(time.time()*1e6)}.png"
        cv2.imwrite(str(dst), image_bgr)
        return dst


def _validate_label(label: str) -> str:
    up = label.upper()
    if up not in INTENT_LABELS:
        raise SystemExit(f"label '{label}' not in intent vocab {INTENT_LABELS}")
    return up


def cmd_ingest(args) -> None:
    label = _validate_label(args.label)
    store = LabelStore(args.out)
    src = Path(args.src)
    n = 0
    for p in sorted(src.iterdir()):
        if p.suffix.lower() in IMAGE_EXT:
            store.add_path(label, p)
            n += 1
    print(f"ingested {n} frames into {store.dir_for(label)}")


def cmd_capture(args) -> None:
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"capture needs opencv-python: {exc}")
    store = LabelStore(args.out)
    keymap = {str(i + 1): lbl for i, lbl in enumerate(INTENT_LABELS)}
    print("Keys:", ", ".join(f"{k}={v}" for k, v in keymap.items()), "| q=quit")
    cap = cv2.VideoCapture(args.camera)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow("label_frames (1..5 to save, q to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            ch = chr(key) if 32 <= key < 127 else ""
            if ch in keymap:
                dst = store.add_array(keymap[ch], frame)
                print(f"saved {keymap[ch]} -> {dst}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description="Label frames for intent-head training")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="copy an image folder under one label")
    p_ing.add_argument("--label", required=True)
    p_ing.add_argument("--src", required=True)
    p_ing.add_argument("--out", default="data")
    p_ing.add_argument("--window", type=int, default=1, help="frames per sample (V-JEPA)")
    p_ing.set_defaults(func=cmd_ingest)

    p_cap = sub.add_parser("capture", help="capture + key-label from the camera")
    p_cap.add_argument("--camera", type=int, default=0)
    p_cap.add_argument("--out", default="data")
    p_cap.add_argument("--window", type=int, default=1)
    p_cap.set_defaults(func=cmd_capture)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
