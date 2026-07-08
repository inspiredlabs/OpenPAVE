---
url: https://www.youtube.com/watch?v=VFYnD1WREdU
---

# Detection & Segmentation — Falcon Perception (MLX) → Gemma 4

Open-vocabulary **object detection** and **instance segmentation** in OpenPAVE, running fully local on Apple Silicon via [Falcon Perception](https://github.com/tiiuae/Falcon-Perception), and feeding the segmented frame to the **already-working Gemma 4 E4B** VLM for
scene reasoning.

This document describes the pipeline **as actually built and verified** (macOS, Apple Silicon, Python 3.12.13, `mlx 0.31.2`) — not a proposal. Every command and output below was run against `pave_mlx` in this repo.

---

## 1. What was added

| Piece | Location | Role |
|---|---|---|
| `FalconPerceptionBackend` | `pave_mlx/backends.py` | Loads Falcon Perception (MLX); `detect(image_bgr, query, task)` → boxes + masks |
| `annotate_detections()` | `pave_mlx/backends.py` | Renders masks + boxes onto a BGR frame |
| `"falcon"` registry entry + `DETECTOR_NAMES` | `pave_mlx/backends.py` | `make_backend("falcon")`; marks detectors (not intent producers) |
| `SegmentReasonPipeline` + CLI | `pave_mlx/segment_reason.py` | Falcon (segment) → Gemma 4 (reason) |

Falcon runs on its **own MLX runtime** (shipped inside `falcon_perception`), so it
is completely independent of `mlx-vlm` (which serves Qwen3-VL / Gemma). The two
model stacks coexist in the same venv.

---

## 2. Install (done once, into this repo's `.venv`)

Falcon Perception is not on PyPI; it is installed editable from a local clone. On Apple Silicon the package is **MLX-only — it pulls no PyTorch**.

```bash
# clone alongside the repo
git clone https://github.com/tiiuae/Falcon-Perception.git \
    /Users/scottphillips/Documents/GitHub/Falcon-Perception

# install into OpenPAVE's venv WITHOUT churning the existing stack
cd /Users/scottphillips/Documents/GitHub/openpave
.venv/bin/python -m pip install -e /Users/scottphillips/Documents/GitHub/Falcon-Perception --no-deps
.venv/bin/python -m pip install "einops>=0.8.1" "pycocotools>=2.0.11" "tyro>=1.0.3"
```

**Why `--no-deps`:** OpenPAVE's venv already satisfied every Falcon minimum (`pillow 12.3`, `numpy 2.5`, `opencv-python 4.13`, `scipy 1.18`, `datasets 5.0`, `tokenizers 0.22`, `safetensors 0.8`, `mlx 0.31.2`, `huggingface-hub 1.21`). Only `einops`, `pycocotools`, and `tyro` were missing. Installing with full deps would risk re-resolving the shared VLM stack; `--no-deps` + the three leaves keeps it untouched.

> `hf-transfer` is listed by Falcon but is a no-op here — `huggingface-hub 1.21`
> uses Xet transfer. Do **not** set `HF_HUB_ENABLE_HF_TRANSFER` (deprecated); use
> `HF_XET_HIGH_PERFORMANCE=1` if you want faster downloads.

Weights (`tiiuae/Falcon-Perception`, ~1.2 GB) download to the HF cache on first load and are reused thereafter.

---

## 3. The Falcon MLX API (verified, current `falcon-perception==1.0.0`)

The API differs from older integrations — **verified against the installed package and `demo/perception_single_mlx.py`**, not assumed:

```python
from falcon_perception import build_prompt_for_task, load_from_hf_export_mlx from falcon_perception.mlx.batch_inference import BatchInferenceEngine, process_batch_and_generate

model, tokenizer, margs = load_from_hf_export_mlx(hf_model_id="tiiuae/Falcon-Perception", dtype="float16")
# or hf_local_dir=<snapshot> for offline load
engine = BatchInferenceEngine(model, tokenizer)

prompt = build_prompt_for_task("dog", "segmentation")   # task ∈ {"segmentation","detection"}
batch  = process_batch_and_generate(tokenizer, [(pil_rgb_image, prompt)],   # PIL image, NOT a path
             max_length=margs.max_seq_len, min_dimension=256, max_dimension=1024)
_, aux = engine.generate(tokens=batch["tokens"], pos_t=batch["pos_t"], pos_hw=batch["pos_hw"],
             pixel_values=batch["pixel_values"], pixel_mask=batch["pixel_mask"],
             max_new_tokens=200, temperature=0.0, task="segmentation")
```

Output on `aux_outputs[0]`:
- `aux.bboxes_raw` — a stream of `{x,y}` then `{h,w}` dicts; pair them into
  `{x, y, h, w}` where all values are **normalised** (`x,y` = box centre; `w,h` =
  box size, all in `[0,1]`).
- `aux.masks_rle` — one **COCO RLE** dict per detection (decode with
  `pycocotools.mask.decode`).

`FalconPerceptionBackend._parse_aux()` converts this into OpenPAVE's detection
dicts: `{"bbox":[x1,y1,x2,y2], "cx","cy","w","h", "mask": bool ndarray|None,
"mask_area_px": int}`.

---

## 4. Usage

### 4a. Detection / segmentation only

```python
import numpy as np
from PIL import Image
from pave_mlx.backends import make_backend, annotate_detections

frame_bgr = np.asarray(Image.open("dogs.jpg").convert("RGB"))[:, :, ::-1].copy()
falcon = make_backend("falcon")          # mode == "loaded" when ready
dets = falcon.detect(frame_bgr, "dog", task="segmentation")
overlay_bgr = annotate_detections(frame_bgr, dets)
```

`detect()` takes a **BGR uint8** array (OpenPAVE's frame convention, same as the shim and VLM backends) and converts to PIL RGB internally.

### 4b. Segment → reason (Falcon + Gemma 4), CLI

```bash
source .venv/bin/activate
python -m pave_mlx.segment_reason --image dogs.jpg --query dog \
    --out /tmp/annotated.jpg --max-tokens 64
# detection/segmentation only (skip the VLM):
python -m pave_mlx.segment_reason --image dogs.jpg --query dog --no-reason
```

Programmatic:

```python
from pave_mlx.segment_reason import SegmentReasonPipeline
result = SegmentReasonPipeline().run(frame_bgr, "dog", task="segmentation", max_tokens=64)
# -> {"detections": [...], "annotated_bgr": ndarray, "gemma_text": str|None, "timings": {...}}
```

---

## 5. Keeping the 126 shared-KV tensors (no re-export)

The Gemma 4 checkpoint used is **`lmstudio-community/gemma-4-E4B-it-MLX-4bit`, unmodified**. That checkpoint carries per-layer `k_proj`/`v_proj`/`k_norm` tensors for the 18 shared-KV layers (24–41) that mlx-vlm's `gemma4` architecture does not instantiate — 18 × 7 = **126 tensors** mlx-vlm otherwise rejects with `ValueError: Received 126 parameters not in model`.

OpenPAVE does **not** strip them from disk. Instead `backends._patch_gemma4_shared_kv_sanitize()` patches mlx-vlm's Gemma 4 loader to **drop those tensors in memory at load time** (the language-model sanitizer already knows how). At runtime you see the compat note:

```
[gemma ] mode=loaded Gemma 4 shared-KV load filter active; using local HF snapshot
```

So the on-disk cache stays byte-for-byte the lmstudio release (also usable by LM Studio and other tools); nothing is re-quantised or re-exported.

---

## 6. Verified run (2026-07-01, `test_data/dogs.jpg`, 800×534)

```
$ python -m pave_mlx.segment_reason --image dogs.jpg --query dog --task segmentation --max-tokens 64
[falcon] mode=loaded using local HF snapshot
[gemma ] mode=loaded Gemma 4 shared-KV load filter active; using local HF snapshot
[segment] query='dog' task=segmentation: 2 detection(s)
  [0] bbox=[163, 115, 349, 450] mask_area_px=42719
  [1] bbox=[398, 66, 643, 533] mask_area_px=72326
[reason ] A bright, outdoor scene features two depictions of a dog. On the left, a
          small, tan and white dog is shown in a blue bounding box. On the right, a
          fluffy ... dog ... within a green bounding box ... golden hour
[timings] {'falcon_detect_s': 4.61, 'gemma_reason_s': 17.23}
```

Two dogs correctly detected and segmented (boxes + instance masks), and Gemma
reasons over the **annotated** frame — it references the "blue" and "green"
bounding boxes, confirming the overlay is what it sees.

**Timings (M-series, warm cache):**
- Falcon load: ~5 s warm (first-ever load ≈ 9 min, dominated by the 1.2 GB download).
- Falcon detect+segment: ~4–16 s / frame (`max_new_tokens=200`, `min/max_dim=256/1024`).
- Gemma reason: ~17 s for 64 tokens.

Regression check: `python -m unittest tests.test_vlm_downloads` → **5 passed**.

---

## 7. Dependencies added to `.venv`

`falcon-perception 1.0.0` (editable), `einops 0.8.2`, `pycocotools 2.0.11`, `tyro 1.0.15` (+ `docstring-parser`, `typeguard`). No PyTorch. The existing MLX / mlx-vlm / transformers / numpy / pillow / opencv stack was unchanged.

---

## 8. Gotchas

- **Frame convention is BGR.** `detect()` / `annotate_detections()` expect BGR  uint8 (matches `openai_shim._decode_image`); Falcon wants RGB and is converted internally.
- **First load downloads ~1.2 GB.** Budget for it; it is one-time.
- **`falcon` is not an intent backend.** It returns structured detections, not a STOP/TROT/... word, so it is intentionally kept out of the shim's `--backend` choices (`DETECTOR_NAMES`). Use it via `make_backend("falcon")` or the segment-reason pipeline.
- **Model overrides:** `FALCON_PERCEPTION_MODEL` and `GEMMA_VLM_MODEL` env vars.
- **RAM:** running Falcon (float16, ~1.2 GB) and Gemma 4 (4-bit, ~4 GB) together needs ≥ ~8 GB free.
