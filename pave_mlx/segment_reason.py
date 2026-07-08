"""Segment-then-reason pipeline: Falcon Perception (detect/segment) → Gemma 4 (reason).

This pairs the two MLX models that already run locally in OpenPAVE:

  1. **Falcon Perception** localises and segments the objects named in a natural
     -language query (open-vocabulary detection + instance masks).
  2. The frame is annotated with those masks/boxes and handed to the **already
     -working Gemma 4 E4B** VLM, which reasons over the highlighted scene.

The Gemma 4 checkpoint (`lmstudio-community/gemma-4-E4B-it-MLX-4bit`) is used
*as-is* — its redundant shared-KV tensors are dropped at load time by
`backends._patch_gemma4_shared_kv_sanitize()`, so the on-disk cache keeps all
126 tensors and nothing is re-quantised or re-exported.

CLI:

    python -m pave_mlx.segment_reason --image dogs.jpg --query dog
    python -m pave_mlx.segment_reason --image street.jpg --query "car" --task detection \
        --prompt "How many cars are highlighted? Reply with a number."
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from pave_mlx.backends import annotate_detections, make_backend


class SegmentReasonPipeline:
    """Falcon (segment) + Gemma 4 (reason). Backends load lazily and degrade
    gracefully: if Gemma is unavailable the pipeline still returns detections and
    the annotated frame; if Falcon is unavailable it returns no detections."""

    def __init__(self, falcon=None, gemma=None) -> None:
        self.falcon = falcon if falcon is not None else make_backend("falcon")
        self.gemma = gemma if gemma is not None else make_backend("gemma")

    def run(
        self,
        image_bgr: np.ndarray,
        query: str,
        prompt: str | None = None,
        task: str = "segmentation",
        max_tokens: int = 64,
    ) -> dict:
        """Returns `{"detections", "annotated_bgr", "gemma_text", "timings"}`."""
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        detections = self.falcon.detect(image_bgr, query, task=task)
        timings["falcon_detect_s"] = round(time.perf_counter() - t0, 2)

        annotated = annotate_detections(image_bgr, detections)

        gemma_text = None
        if getattr(self.gemma, "_model", None) is not None:
            reason_prompt = prompt or (
                f"{len(detections)} instance(s) of '{query}' have been segmented and "
                f"highlighted with coloured masks and boxes. Describe the scene concisely."
            )
            t0 = time.perf_counter()
            gemma_text = self.gemma.generate(annotated, reason_prompt, max_tokens=max_tokens)
            timings["gemma_reason_s"] = round(time.perf_counter() - t0, 2)

        return {
            "detections": detections,
            "annotated_bgr": annotated,
            "gemma_text": gemma_text,
            "timings": timings,
        }


def _load_bgr(path: str) -> np.ndarray:
    from PIL import Image

    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return rgb[:, :, ::-1].copy()  # RGB -> BGR


def main() -> None:
    ap = argparse.ArgumentParser(description="Falcon segment → Gemma reason")
    ap.add_argument("--image", required=True, help="path to an image")
    ap.add_argument("--query", required=True, help="what to detect/segment, e.g. 'dog'")
    ap.add_argument("--task", default="segmentation", choices=["segmentation", "detection"])
    ap.add_argument("--prompt", default=None, help="override the Gemma reasoning prompt")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", default=None, help="save the annotated frame to this path")
    ap.add_argument("--no-reason", action="store_true", help="skip Gemma; detect/segment only")
    args = ap.parse_args()

    image_bgr = _load_bgr(args.image)
    falcon = make_backend("falcon")
    print(f"[falcon] mode={falcon.mode} {falcon.load_status or falcon.load_error}")

    gemma = None
    if not args.no_reason:
        gemma = make_backend("gemma")
        note = getattr(gemma, "load_status", "") or getattr(gemma, "load_error", "")
        print(f"[gemma ] mode={gemma.mode} {note}")

    pipeline = SegmentReasonPipeline(falcon=falcon, gemma=gemma or _NullGemma())
    result = pipeline.run(image_bgr, args.query, prompt=args.prompt, task=args.task,
                          max_tokens=args.max_tokens)

    dets = result["detections"]
    print(f"\n[segment] query={args.query!r} task={args.task}: {len(dets)} detection(s)")
    for i, d in enumerate(dets):
        print(f"  [{i}] bbox={d['bbox']} mask_area_px={d['mask_area_px']}")
    if result["gemma_text"] is not None:
        print(f"\n[reason ] {result['gemma_text']}")
    print(f"\n[timings] {result['timings']}")

    if args.out:
        from PIL import Image

        Image.fromarray(result["annotated_bgr"][:, :, ::-1]).save(args.out)
        print(f"[saved  ] annotated frame -> {args.out}")


class _NullGemma:
    """Stands in for the Gemma backend when --no-reason is set."""

    _model = None
    mode = "disabled"


if __name__ == "__main__":
    main()
