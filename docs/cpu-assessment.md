# CPU/VLM Frame Pipeline Assessment

Date: 2026-07-02

## Current Baseline

The live Qwen path is currently:

1. OpenCV captures a webcam frame as a CPU `numpy.ndarray` in BGR format.
2. `pave_ui/perception.py` resizes the frame before inference. For Qwen this is controlled by `VLM_INPUT_SIZE_QWEN`, currently defaulting to `224`.
3. `pave_mlx/backends.py` JPEG-encodes the frame and sends it to the local `vllm-mlx` server as an OpenAI-compatible `image_url`. The accepted transport is a base64 data URL; the attempted localhost frame URL transport is rejected by `vllm-mlx` as remote media. JPEG encoding is now OpenCV-first when available and otherwise uses non-optimized PIL instead of the previous slow PIL `optimize=True` path.
4. `vllm-mlx`/`mlx-vlm` decodes and preprocesses the image again before model prefill/generation.

Relevant code:

- `pave_ui/perception.py`: `VLM_INPUT_SIZE_QWEN` and `cv2.resize(...)`.
- `pave_mlx/backends.py`: `_image_data_url(...)` and the OpenAI-compatible chat request.
- `pave_mlx/backends.py`: `QwenVLMBackend` defaults to `mlx-community/Qwen3-VL-4B-Instruct-3bit`; `Qwen2BVLMBackend` uses `mlx-community/Qwen3-VL-2B-Instruct-3bit`.

Measured local baseline from the current run: about **600 ms per frame at 112 px**. The existing code comments also show that Qwen latency scales strongly with image size, which means image-token prefill remains a major cost.

## Expected Gains Summary

| Option | Expected speed gain | Confidence | Why |
| --- | ---: | --- | --- |
| Qwen3-VL-2B with `vllm-mlx` | **Implemented as dropdown option using the MLX 3-bit checkpoint** | Medium | Smaller language model should reduce prefill/generation work, but low-resolution live latency is also vision/preprocess/API-bound. |
| `mx.array(frame, copy=False)` zero-copy + MLX normalization | **Low in the current server path** | High | Current MLX Python docs do not expose `mx.array(..., copy=False)`, and the server path still JPEG-encodes before inference. |
| `mlx.data.Buffer.image_resize` and related ops | **Implemented as an opt-in resize backend; benchmark before keeping it on** | Medium | Useful image pipeline primitives exist, but wrapping individual live frames in a data buffer may or may not beat OpenCV. |
| Remove base64 and process frames closer to GPU | **Highest architectural upside; localhost `image_url` is blocked, so use a deeper API change** | High | The server rejects remote media URLs, so real gains require removing JPEG/base64 through custom binary/shared-memory or in-process handoff. |

## 1. Smaller Model: `mlx-community/Qwen3-VL-2B-Instruct-3bit` with `vllm-mlx`

Source model: <https://huggingface.co/mlx-community/Qwen3-VL-2B-Instruct-3bit>

This is now wired into OpenPAVE as a separate dropdown entry, `Qwen3-VL 2B`, backed by `pave_mlx.backends.Qwen2BVLMBackend`. The checkpoint is a 3-bit MLX conversion of `Qwen/Qwen3-VL-2B-Instruct`, with the Hugging Face model card reporting about 1.57 GB.

Expected outcomes:

- If `vllm-mlx` loads the 2B MLX checkpoint cleanly, expect a **real but not 2x** latency improvement.
- At very small image sizes such as 112 px, the model swap may only save **~10-30%**, because the fixed costs around request handling, image processing, vision encoding, and short text generation become a larger share of the total.
- At 224 px and above, the smaller language model may help more, roughly **~20-40%** in the best case, because prefill has more tokens to process.
- If the only available 2B path is BF16 while the current 4B path is 3-bit quantized, memory bandwidth and cache behavior may erase some of the expected gain.

How to test:

1. Select `Qwen3-VL 2B` in the model dropdown, or launch with:

   ```bash
   export PAVE_DEFAULT_MODEL="Qwen3-VL 2B"
   export PAVE_ALLOW_MODEL_DOWNLOADS=1
   export PAVE_VLLM_MLX_TIMINGS=1
   export PAVE_VLM_TIMINGS=1
   ./mlx-runtime.sh
   ```

2. Override the 2B checkpoint only when deliberately testing another conversion:

   ```bash
   export QWEN_2B_VLM_MODEL="mlx-community/Qwen3-VL-2B-Instruct-3bit"
   ```

3. Benchmark 112 px and 224 px separately. The 112 px result will show the latency floor; 224 px will show whether the smaller model buys enough quality-preserving speed.

Assessment: **worth testing immediately**, but judge it by measured `request_ms` and output validity. If it produces malformed text under `vllm-mlx`, keep the 4B model for control and use 2B only after prompt/server tuning.

## 2. Zero-Copy to MLX + Native MLX Normalization

The proposed line:

```python
mx.array(frame, copy=False)
```

does not appear to match the current documented MLX Python API. MLX arrays use unified memory, but converting an existing OpenCV/NumPy frame into an MLX array is still expected to allocate/copy unless a supported zero-copy bridge is used. The documented MLX array constructor does not list a `copy` parameter comparable to NumPy's `array(..., copy=False)`.

More importantly, the current Qwen server path does not perform model normalization in OpenPAVE. It resizes the frame, then converts the image into JPEG for the server request. The accepted OpenAI-compatible transport wraps that JPEG as a base64 data URL. Any MLX-side resize/normalization performed before the request would likely be converted back to CPU/JPEG, which gives up most of the benefit.

Expected gain in the current architecture:

- Resize/color conversion at 112-224 px is likely a small part of a 600 ms frame.
- Replacing `cv2.resize` with MLX ops while still returning to PIL/JPEG/base64 is likely **0-10%**, and can be slower due to extra transfers/synchronization.
- It will not remove the current JPEG/API overhead.

Recommendation:

- Do not prioritize `mx.array(frame, copy=False)` as written.
- Only revisit MLX-native preprocessing after choosing an inference path that can accept an MLX array, shared frame buffer, or at least raw binary image bytes without round-tripping through PIL/base64.

Assessment: **not the main speed lever yet**.

## 3. `mlx.data` / `Buffer.image_resize`

Relevant docs:

- <https://ml-explore.github.io/mlx-data/>
- <https://ml-explore.github.io/mlx-data/build/html/python/_autosummary/mlx.data.Buffer.image_resize.html>

`mlx.data` does include image pipeline primitives such as `Buffer.image_resize`, and OpenPAVE now has an opt-in live resize backend:

```bash
export PAVE_VLM_RESIZE_BACKEND=mlx-data
export PAVE_VLM_TIMINGS=1
./mlx-runtime.sh
```

However, the package is designed around data buffers and input pipelines. It is not automatically a live-camera, GPU-resident frame path.

Likely opportunities:

- Faster CPU-side resize/decode pipelines for batches or prerecorded data.
- Cleaner image transforms if the project later adds a dataset/capture benchmark harness.
- A possible replacement for ad hoc PIL/OpenCV transformations if it benchmarks better on this Mac.

Risks/limits:

- Wrapping every live OpenCV frame into an `mlx.data.Buffer` may add overhead.
- It may not eliminate the NumPy frame, PIL/JPEG, or base64 steps.
- It may not run the operation on the GPU in the way needed for end-to-end live inference speed.

Expected gain in the current architecture:

- For 112-224 px frames, likely **single-digit milliseconds to low tens of milliseconds**.
- As a percentage of 600 ms, likely **small** unless profiling shows `cv2.resize`/PIL encode is unexpectedly dominant.

Current implementation:

- Default remains `cv2.resize`.
- `PAVE_VLM_RESIZE_BACKEND=mlx-data` wraps the current live NumPy frame into an `mlx.data.Buffer`, calls `image_resize("image", size, size)`, and converts the result back to NumPy for the existing VLM request.
- If `mlx.data` fails or is unavailable, the runtime falls back to OpenCV once and does not keep retrying every frame.

Recommendation:

- Run a microbenchmark, but do not assume it will fix the current 600 ms floor.
- Compare:
  - `cv2.resize`
  - PIL resize
  - `mlx.data.Buffer.image_resize`
  - current `_image_data_url(...)` end-to-end
- Time each stage separately with `time.perf_counter()` around resize, JPEG encode, base64 encode, HTTP request, and server response.

Assessment: **useful research, probably not the primary unlock**.

## 4. Remove Base64 and Process Frames Closer to GPU

This is the highest-value architectural target. The current OpenAI-compatible `image_url` request forces a data-URL path:

```json
{
  "type": "image_url",
  "image_url": {
    "url": "data:image/jpeg;base64,..."
  }
}
```

That costs:

- BGR to RGB conversion.
- PIL image creation.
- JPEG compression.
- Base64 encoding, which expands payload size by roughly one third.
- JSON serialization.
- HTTP transfer to localhost.
- Server-side base64/JPEG decode and preprocessing.

Ways to remove or reduce it, from least to most invasive:

### A. Local HTTP image URLs

Serve the latest JPEG frame from a tiny local endpoint and pass `http://127.0.0.1:.../frame.jpg` instead of a data URL.

Status: **blocked by `vllm-mlx` server policy**.

```bash
export PAVE_VLLM_MLX_IMAGE_TRANSPORT=http-url
./mlx-runtime.sh
```

Observed result: `HTTP 400 from vllm-mlx: {"detail":"Remote media URL is not allowed"}`. OpenPAVE now ignores `http-url` unless `PAVE_VLLM_MLX_ALLOW_REMOTE_IMAGE_URLS=1` is also set. If a server still rejects remote media after that opt-in, OpenPAVE catches that response and falls back to the data URL path so the UI does not show a prompt error, but this is not a usable base64-removal strategy.

Pros:

- Removes base64 and JSON payload bloat.
- Keeps the OpenAI-compatible API.

Cons:

- Still JPEG-encodes and JPEG-decodes every frame.
- Adds another local HTTP hop.
- Does not process the webcam frame on GPU.

Expected gain: **none under the current server policy**.

### A.1 Faster JPEG/base64 packaging

The data URL path still exists, but its CPU packaging is now cheaper:

- `PAVE_VLLM_MLX_JPEG_ENCODER=auto` is the default.
- OpenCV JPEG encoding is used when `cv2` imports cleanly.
- PIL fallback no longer uses `optimize=True` by default.
- `PAVE_VLLM_MLX_TIMINGS=1` prints `jpeg_ms`, `base64_ms`, `request_ms`, and total backend wall time.

Local packaging benchmark in this Python environment, random frames, quality 80:

| Size | Old PIL optimize JPEG | New JPEG | New JPEG + base64 |
| ---: | ---: | ---: | ---: |
| 112 px | 0.089 ms | 0.073 ms | 0.082 ms |
| 224 px | 0.418 ms | 0.240 ms | 0.259 ms |
| 448 px | 2.238 ms | 1.009 ms | 1.108 ms |

This is a real cleanup of avoidable CPU work, but it also shows base64 packaging is not enough to explain a ~600 ms frame. The remaining latency is dominated by request/server/model work unless live timings say otherwise.

### B. Custom binary endpoint for `vllm-mlx`

Add an OpenPAVE-specific server endpoint that accepts raw JPEG/PNG bytes, raw RGB/BGR bytes, or a shared-memory frame ID instead of OpenAI chat JSON with a data URL.

Pros:

- Removes base64.
- Can remove repeated JSON body construction for image payloads.
- Shared memory can avoid copying full frames between UI and server process.

Cons:

- Requires maintaining a custom API beside the OpenAI-compatible API.
- If the payload is still JPEG, decode remains.
- If the payload is raw BGR/RGB, the server preprocessing path must accept it.

Expected gain: **modest for JPEG bytes; larger if raw/shared memory avoids encode/decode**.

### C. In-process VLM path with direct image objects

Run Qwen through an in-process `mlx-vlm`/`vllm-mlx` Python path and pass the frame as a PIL image, NumPy array, or eventually an MLX array.

Pros:

- Removes localhost HTTP, JSON, base64, and server serialization.
- Gives the best chance to own the image preprocessing path.

Cons:

- The previous direct `mlx-vlm` path may lose some `vllm-mlx` server benefits such as continuous batching, paged KV cache, and server-managed prefix cache.
- MLX stream/thread affinity matters in this repo; the existing `EngineWorker` design must be preserved carefully.

Expected gain: **moderate**, depending on how much of the 600 ms is API/image packaging versus model prefill.

### D. Real GPU capture/preprocess path

OpenCV `VideoCapture` gives CPU NumPy frames. To genuinely process webcam frames on the GPU, the capture path likely needs to move away from OpenCV toward AVFoundation/CoreVideo/Metal, for example:

- Capture camera frames as `CVPixelBuffer`.
- Keep frames in `IOSurface`/Metal texture form.
- Resize/color-convert with Metal or a direct MLX-compatible path.
- Feed the model without converting to JPEG/base64.

Pros:

- This is the cleanest route to true GPU-resident camera preprocessing.
- Avoids building optimizations around a CPU NumPy frame that arrived too late.

Cons:

- Highest implementation cost.
- Python bindings around `CVPixelBuffer`/Metal/MLX interop may require a Swift/Objective-C helper or a small native extension.
- Still only helps the non-model part unless paired with a model API that accepts the resulting tensor/image without server-side decode.

Expected gain: **best long-term architecture, but only worth doing after profiling confirms CPU packaging is a major share of latency**.

## Recommended Next Steps

1. **Benchmark the server-side request path first**. With `jpeg_ms=0.1`, `base64_ms=0.0`, and `request_ms=~581`, the bottleneck is no longer client packaging. It is `vllm-mlx` model prefill/decode plus scheduler/server overhead.
2. **Add stage timing before rewriting preprocessing**:

   - camera `cap.read`
   - `cv2.resize`
   - BGR to RGB/PIL conversion
   - JPEG encode
   - base64 encode
   - HTTP request/response wall time
   - model/server reported generation time, if available

3. **Remove base64 only after timing confirms packaging is material**. The first low-risk version is a custom binary or shared-memory path, not local image URLs.
4. **Treat MLX preprocessing as part of a larger API change**. MLX-native frame operations are valuable only if the frame stays in that representation through model preprocessing.

## Request-Time Research Update

The useful timing split is:

```text
jpeg_ms=0.1 base64_ms=0.0 request_ms=581 total_ms=581
```

This means resize/JPEG/base64 are not the live bottleneck. The request is spending time inside the `vllm-mlx` server and model.

Actionable levers, ranked:

1. **Low-latency server mode, experimental.** OpenPAVE previously launched `vllm-mlx` with continuous batching, paged cache, and prefix cache by default. Upstream `vllm-mlx` docs describe simple mode as the single-user/max-throughput mode and continuous batching as a multi-user throughput feature with per-request overhead. However, Qwen3-VL coherence must be checked when changing scheduler/cache mode, so OpenPAVE keeps the verified default and exposes low-latency as an explicit experiment:

   ```bash
   export PAVE_VLLM_MLX_LOW_LATENCY=1
   unset PAVE_VLLM_MLX_CONTINUOUS_BATCHING
   unset PAVE_VLLM_MLX_PAGED_CACHE
   unset PAVE_VLLM_MLX_PREFIX_CACHE
   ./mlx-runtime.sh
   ```

   For an external server, start it without `--continuous-batching` / `--use-paged-cache` for the live webcam benchmark.

2. **Reduce image tokens further.** Qwen3-VL latency scales with image size. Test below 112 px if the task tolerates it:

   ```bash
   export PAVE_VLM_INPUT_SIZE_QWEN=56
   export PAVE_VLM_TIMINGS=1
   export PAVE_VLLM_MLX_TIMINGS=1
   ./mlx-runtime.sh
   ```

   This is the most direct way to reduce prefill while keeping the same model.

3. **Reduce output tokens / task complexity.** The live prompt asks for `INTENT` plus up to three `FEATURE` lines. If speed matters more than overlay quality, an intent-only prompt plus `PAVE_VLM_MAX_TOKENS=4` can cut decode and some prompt prefill. This trades away model-reported boxes.

   ```bash
   export PAVE_VLM_FAST_INTENT_ONLY=1
   export PAVE_VLM_MAX_TOKENS=4
   export PAVE_VLM_TIMINGS=1
   export PAVE_VLLM_MLX_TIMINGS=1
   ./mlx-runtime.sh
   ```

4. **MLLM cache only helps repeated identical images.** `vllm-mlx` documents an MLLM cache that hashes image content and reuses vision embeddings/KV state. This does not help normal webcam frames because every frame is different. It can help static-scene tests or a deliberate frame-dedup strategy:

   ```bash
   export PAVE_VLLM_MLX_MLLM_CACHE=1
   export PAVE_VLLM_MLX_MLLM_CACHE_MB=1024
   ./mlx-runtime.sh
   ```

5. **Use a smaller or specialized model.** If 56-112 px plus low-latency server mode is still too slow, the remaining speedup must come from less model. That can mean Qwen3-VL-2B if compatible, or a non-VLM detector/gesture model for the hot path with Qwen reserved for slower semantic checks.

## Bottom Line

The most likely near-term request-time win is **low-latency server mode plus more aggressive Qwen input-size/output-token tuning**. Base64 and JPEG are already sub-millisecond at the current size, so removing them will not turn a 581 ms request into a real-time one. If the request still needs a large cut after those server/model knobs, the architecture needs a smaller/specialized hot-path model rather than more CPU preprocessing work.
