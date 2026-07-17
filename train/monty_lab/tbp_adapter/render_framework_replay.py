"""Render teacher/student overlays directly from Monty DetailedJSONHandler logs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_RUN = ROOT / "train" / "runs" / "monty_landmark_alignment" / "framework_verified" / "pretrained"

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def _episode(path):
    payload = json.loads(path.read_text())
    return next(iter(payload.values()))


def _points(raw, key):
    points = np.full((21, 2), np.nan, dtype=np.float32)
    for row in raw:
        points[int(row["joint_id"])] = row[key]
    return points


def _draw_skeleton(image, points, color, radius=4, thickness=2):
    xy = np.round(points * np.array([image.shape[1], image.shape[0]])).astype(int)
    for a, b in CONNECTIONS:
        if np.isfinite(points[[a, b]]).all():
            cv2.line(image, tuple(xy[a]), tuple(xy[b]), color, thickness, cv2.LINE_AA)
    for joint, point in enumerate(xy):
        if np.isfinite(points[joint]).all():
            cv2.circle(image, tuple(point), radius, color, -1, cv2.LINE_AA)
            cv2.putText(image, str(joint), tuple(point + (4, -4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA)


def render(detail, out, size):
    raw = detail["SM_0"]["raw_observations"]
    source, source_index = raw[0]["source"], int(raw[0]["source_index"])
    with np.load(ROOT / "train" / "datasets" / source / "prepared.npz",
                 allow_pickle=True) as data:
        image = cv2.cvtColor(data["imgs"][source_index], cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
    teacher, student = _points(raw, "teacher_uv"), _points(raw, "student_uv")
    error = np.linalg.norm((student - teacher) * size, axis=1)

    # Teacher is red, the candidate replacement is green. Residual vectors in
    # yellow make local failure direction visible without feeding it to Monty.
    teacher_xy = np.round(teacher * size).astype(int)
    student_xy = np.round(student * size).astype(int)
    for a, b in zip(teacher_xy, student_xy):
        cv2.line(image, tuple(a), tuple(b), (0, 220, 255), 1, cv2.LINE_AA)
    _draw_skeleton(image, teacher, (50, 50, 255), radius=4, thickness=2)
    _draw_skeleton(image, student, (30, 230, 30), radius=3, thickness=2)

    target = raw[0]["target"]
    caption = "%s  %s[%d]  mean %.1fpx  p95 %.1fpx" % (
        target, source, source_index, float(np.mean(error)),
        float(np.percentile(error, 95)))
    cv2.rectangle(image, (0, 0), (size, 30), (0, 0, 0), -1)
    cv2.putText(image, caption, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.46, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "teacher", (size - 145, size - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 50, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "student", (size - 75, size - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 230, 30), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), image)
    return image


def main(argv=None):
    args = parser().parse_args(argv)
    if args.out is None:
        args.out = args.run / "replay"
    # The consolidated DetailedJSONHandler file is the current format; the
    # per-episode directory can be a stale leftover from earlier runs and
    # must not shadow a fresh consolidated log.
    entries = []
    consolidated = args.run / "detailed_run_stats.json"
    if consolidated.exists():
        for line in consolidated.read_text().splitlines():
            payload = json.loads(line)
            key, detail = next(iter(payload.items()))
            entries.append(("episode_%06d" % int(key), detail))
    else:
        episode_dir = args.run / "detailed_run_stats"
        entries = [(path.stem, _episode(path))
                   for path in sorted(episode_dir.glob("episode_*.json"))]
    if not entries:
        raise SystemExit("no DetailedJSONHandler episodes found")
    args.out.mkdir(parents=True, exist_ok=True)
    images = []
    for stem, detail in entries[:args.limit]:
        destination = args.out / (stem + ".png")
        images.append(render(detail, destination, args.size))
        print(destination)
    if images:
        contact = np.concatenate(images, axis=1)
        cv2.imwrite(str(args.out / "contact_sheet.png"), contact)
        print(args.out / "contact_sheet.png")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=DEFAULT_RUN)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--limit", type=int, default=4)
    p.add_argument("--size", type=int, default=384)
    return p


if __name__ == "__main__":
    main()
