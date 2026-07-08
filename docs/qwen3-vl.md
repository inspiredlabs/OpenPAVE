# Qwen3-VL — MLX VLMs for OpenPAVE

The PyQt console (`pave_ui`) defaults to **Qwen3-VL-4B-Instruct (MLX, 3-bit)** as the
vision-language model that turns live camera frames into robot intents. It runs
locally on Apple Silicon via [`mlx-vlm`](https://github.com/Blaizzy/mlx-vlm).
It can also use an external [`vllm-mlx`](https://github.com/waybarrios/vllm-mlx)
server through the same `generate(image, prompt) -> text` contract; the annotated
video overlay remains owned by `pave_ui/perception.py` and `pave_ui/viewer.py`.

## Install

```bash
# into the OpenPAVE venv (the same one mlx-runtime.sh creates)
.venv/bin/python -m pip install -U mlx-vlm
# optional external server adapter
.venv/bin/python -m pip install -U vllm-mlx
```

The weights download automatically from Hugging Face on first use and are cached
under `~/.cache/huggingface`. Models:

- `mlx-community/Qwen3-VL-4B-Instruct-3bit`
  (https://huggingface.co/mlx-community/Qwen3-VL-4B-Instruct-3bit)
- `mlx-community/Qwen3-VL-2B-Instruct-3bit`
  (https://huggingface.co/mlx-community/Qwen3-VL-2B-Instruct-3bit), exposed in the
  dropdown as `Qwen3-VL 2B` for latency A/B tests.

Override with a different MLX VLM repo via `QWEN_VLM_MODEL` for the 4B entry or
`QWEN_2B_VLM_MODEL` for the 2B entry:

```bash
export QWEN_VLM_MODEL="mlx-community/Qwen3-VL-4B-Instruct-3bit"
export QWEN_2B_VLM_MODEL="mlx-community/Qwen3-VL-2B-Instruct-3bit"
```

## Gemma 4 E4B

OpenPAVE's Gemma backend targets the already-downloaded LM Studio 4-bit MLX
conversion:

- `lmstudio-community/gemma-4-E4B-it-MLX-4bit`
  (https://huggingface.co/lmstudio-community/gemma-4-E4B-it-MLX-4bit)

The model card identifies this as a 4-bit MLX artifact around 6.83 GB. Its
safetensor shards are marked `format=mlx`, and the checkpoint index includes
shared-KV `k_proj` / `v_proj` / norm tensors for layers that the installed
`mlx-vlm` Gemma4 module intentionally does not instantiate. OpenPAVE preflights
that mismatch and filters those extra tensors at `load_weights()` so the model
cache can stay as-is.

Override with another Gemma MLX repo only when deliberately testing a different
conversion:

```bash
export GEMMA_VLM_MODEL="mlx-community/gemma-4-e4b-it-8bit"
```

## How it's wired

`pave_mlx` exposes the 4B model as the `qwen` backend and the 2B model as
`qwen_2b` behind the same OpenAI-compatible shim as the encoder backends
(DINOv3 / V-JEPA / LingBot):

```
camera frame → POST /v1/chat/completions (shim, --backend qwen)
   → QwenVLMBackend.generate(image, ROBOT_PROMPT)   # INTENT + FEATURE text
      → clamp_to_intent(text) → STOP|TROT|HOME|LEFT|RIGHT
         → POST /intent (intent_ingress) → control_daemon → ThreeJS robot
```

`ROBOT_PROMPT` / `ROBOT_PROMPT_QWEN` (override with `PAVE_ROBOT_PROMPT` or
`PAVE_ROBOT_PROMPT_QWEN`) asks the model for an `INTENT:` line plus short
`FEATURE:` lines for the overlay; `clamp_to_intent` pins free-form text to the
vocab and defaults to `STOP` if nothing matches.

## vLLM-MLX external server test

Start the server in one terminal:

```bash
.venv/bin/vllm-mlx serve lmstudio-community/gemma-4-E2B-it-MLX-4bit \
  --port 8000 \
  --served-model-name default
```

The upstream `vllm-mlx` docs describe continuous batching as a multi-user
throughput feature, but Qwen3-VL coherence must be checked before changing the
server scheduler. OpenPAVE defaults to the previously verified server mode. Try
single-request low-latency mode only as an explicit experiment:

```bash
export PAVE_VLLM_MLX_LOW_LATENCY=1
unset PAVE_VLLM_MLX_CONTINUOUS_BATCHING
unset PAVE_VLLM_MLX_PAGED_CACHE
unset PAVE_VLLM_MLX_PREFIX_CACHE
```

Then start OpenPAVE in another terminal:

```bash
export PAVE_VLM_RUNTIME=vllm-mlx
export PAVE_VLLM_MLX_URL=http://127.0.0.1:8000/v1
export PAVE_VLLM_MLX_MODEL=default
./mlx-runtime.sh
```

The server request keeps the prompt and image together in one multimodal user
message (`text` then `image_url`), matching the current `mlx-vlm` path. That is
what preserves the existing `INTENT:` / `FEATURE:` parsing and the annotated
boxes drawn on top of the webcam stream.

OpenPAVE sends the image as a JPEG data URL, because that is the portable
OpenAI-compatible path accepted by `vllm-mlx`.

The data URL path now avoids the slowest previous CPU packaging step:
`PAVE_VLLM_MLX_JPEG_ENCODER=auto` uses OpenCV JPEG encoding when available and
falls back to PIL without `optimize=True`. To expose per-frame packaging timing,
run:

```bash
export PAVE_VLLM_MLX_TIMINGS=1
export PAVE_VLM_TIMINGS=1
./mlx-runtime.sh
```

The attempted localhost frame URL transport is intentionally disabled by
default:

```bash
export PAVE_VLLM_MLX_IMAGE_TRANSPORT=http-url
./mlx-runtime.sh
```

Current `vllm-mlx` rejects that request with `Remote media URL is not allowed`,
so OpenPAVE ignores `http-url` unless
`PAVE_VLLM_MLX_ALLOW_REMOTE_IMAGE_URLS=1` is also set. If a server still rejects
remote media after that opt-in, OpenPAVE catches the response, falls back to the
data URL transport inside the same inference call, and keeps using data URLs for
the rest of the session. The next real base64-removal path needs a custom
binary/shared-memory endpoint or an in-process image handoff, not a remote
`image_url`.

> **Use `python -m pave_mlx.vllm_server serve …` (not bare `vllm-mlx serve`)**
> when starting an external server for OpenPAVE. The launcher applies required
> compatibility patches before the model loads — including the text-only
> prefix-cache guard described below. A stock `vllm-mlx serve` will corrupt
> the live camera path.

## Live-camera prefix-cache collision (fixed 2026-07-02)

**Symptom:** with Qwen3-VL on vllm-mlx, the first frame answered correctly,
then every later frame returned `content: null` + `finish_reason: length`.
The client stringified that to `"None"`, which `clamp_to_intent` pinned to a
permanent `STOP` with zero `FEATURE:` lines — no annotated boxes, no usable
command at the ingress. Gemma was unaffected only because its vision path is
pinned to direct in-process `mlx-vlm` (no prefix cache).

**Root cause:** vllm-mlx's MLLM KV prefix cache keys entries by *input token
ids alone*. Every camera frame tokenizes to the same ids (fixed prompt text +
a fixed-length run of image placeholder tokens — the pixels live in
`pixel_values`, which the cache never sees). Consecutive frames therefore
collide as full-prefix hits; the upstream "remaining tokens contain image
placeholders" guard never fires because nothing remains uncached, and the
server resumes generation from the previous frame's KV — a state that already
contains a finished answer, so it emits nothing extractable. The entries also
persist to disk (`~/.cache/vllm-mlx/prefix_cache`, plus the SSD tier), so a
fresh server could answer frame 0 from a *previous session's* webcam frame.

**Fix (three layers):**
1. `pave_mlx/vllm_server.py` patches the server so the KV prefix cache
   bypasses any token sequence containing image/video placeholder tokens —
   camera frames always prefill honestly; text-only prompts keep the cache
   (measured: 347 ms cold → 183 ms warm). vllm-mlx's separate vision cache is
   keyed by image content hash and remains enabled.
2. `VllmMlxBackend.generate()` now raises on null/empty `content` instead of
   returning the literal string `"None"` — a poisoned or failed response is an
   inference error, never a silent `STOP` command.
3. Poisoned persisted entries were purged; with the store guard in place no
   new image-keyed entries accumulate.

**Honest latency** (Qwen3-VL-4B-3bit, M4 Pro 24 GB, prefix cache correct):
the earlier "~0.3 s/frame" claim was the collision reading back cached
answers, not inference. Real cost is prefill-bound and scales with input
resolution: ~2.06 s @ 896 px, ~1.32 s @ 672 px, **~0.82 s @ 448 px**. The UI
now sends Qwen 448 px frames (`PAVE_VLM_INPUT_SIZE_QWEN`, default 448), which
beats Gemma 4 E2B's ~0.9–1.1 s with identical `INTENT`/`FEATURE` quality on
the coarse 3×3-grid task.

For the current low-resolution live path, `PAVE_VLLM_MLX_TIMINGS=1` showing
`jpeg_ms`/`base64_ms` near zero and `request_ms` around hundreds of milliseconds
means the remaining latency is server-side model prefill/decode, not client image
packaging. Meaningful request-time cuts come from lower image tokens, lower output
tokens, low-latency server mode, or a smaller/specialized model.

For a speed-first camera loop, disable model-reported `FEATURE` lines and cap
generation to one intent token:

```bash
export PAVE_VLM_FAST_INTENT_ONLY=1
export PAVE_VLM_MAX_TOKENS=4
export PAVE_VLM_INPUT_SIZE_QWEN=56
export PAVE_VLLM_MLX_TIMINGS=1
export PAVE_VLM_TIMINGS=1
./mlx-runtime.sh
```

This trades overlay quality for lower request time. The UI will still display a
fallback center marker when the model returns no `FEATURE:` lines.

## Smoke test (no GUI)

```bash
.venv/bin/python -m pave_mlx.openai_shim --backend qwen --port 8000
# the first log line prints backend_mode = qwen3-vl (loaded) or fallback (+ reason)
```

## Graceful fallback

If `mlx-vlm` or the weights are missing, the `qwen` backend loads in `fallback`
mode and the shim returns the safe-default `STOP` — the console, camera preview, and
control plane all keep working, so you can wire and demo the pipeline before the
5 GB download finishes.
