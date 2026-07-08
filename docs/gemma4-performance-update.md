# Gemma 4 Performance Update — MLX runtime tuning (macOS / Apple Silicon)

Tracking the work to make the live Gemma 4 E4B path feel responsive on
macOS, after profiling showed the ~3 s/frame the operator saw was **memory
pressure**, not a compute or "missing MLX flag" problem.

Machine of record: **Apple M4 Pro, 24 GB**, macOS 15.5. Metal's recommended GPU
working set on this box is only **~17.2 GB** of the 24 GB.

---

## 1. Diagnosis (measured)

Profiled `mlx-vlm` on `Gemma 4 E4B` (4-bit) with the live prompt:

| Phase | Measurement | Note |
|---|---|---|
| prompt tokens | **445** (280 image + ~165 text) | image-token count is fixed by the model |
| **prefill** | **~1.6 s @ ~280 tok/s** | **~90 % of per-frame cost** |
| decode | ~11 tokens @ ~46 tok/s ≈ 0.24 s | model hits EOS early |
| **total (Gemma only)** | **~1.8 s** | isolated bench, no other model resident |
| Gemma resident memory | **6.8 GB active / 7.66 GB peak** | measured via `mx.get_*_memory()` |

**What does NOT help** (all measured): image resize (fixed 280 image tokens),
lower `max_tokens` (EOS hits early), prompt caching (image-first template → no
reusable prefix), shorter prompt (broke format → model rambled → *slower*).

**Why the app felt like ~3 s (worse than the 1.8 s bench):** Gemma (7.7 GB) +
a Falcon model (1.2–3 GB) + the embedded QtWebEngine viewer (Chromium, 1–3 GB) +
OS can exceed the ~17 GB Metal working set. Past that, Metal evicts/re-stages
weight buffers every inference and **prefill collapses**. The runtime is already
using MLX's fast paths (quantized matmul, `mx.fast.scaled_dot_product_attention`);
the lever on a 24 GB box is **memory discipline**, not compute.

---

## 2. Changes

| # | Change | Status |
|---|---|---|
| 1 | Keep only one big model resident (`mx.clear_cache()` on teardown; free Falcon when a VLM is active) | ✅ |
| 2 | Free the embedded Chromium viewer, gated behind two buttons: `Visualiser: [Internal (QtWebEngine)] [Browser]` | ✅ |
| 3 | Pin weights with a wired-memory limit, with a checkbox to turn it off | ✅ |
| 4 | Warmup + compile on model load (first live frame doesn't eat the `mx.compile` cost) | ✅ |
| 5 | Upgrade the MLX runtime | ✅ (verified already latest) |
| 6 | Zero-click startup (Gemma default + auto camera inference) | ✅ (prior pass) |

### 5. Runtime upgrade — already current
`mlx 0.31.2`, `mlx-metal 0.31.2`, `mlx-lm 0.31.3`, `mlx-vlm 0.6.3` are the
**newest versions published on PyPI** as of 2026-07-01 (`pip index versions`
confirms LATEST == INSTALLED). There is nothing newer to install; the runtime is
already at its optimized ceiling. No branch needed — nothing changed.

### 4. Warmup + compile
`perception.build_engine` runs one tiny `generate()` right after a VLM loads, on
the same Metal-bound thread, so the one-time `mx.compile` / kernel build happens
during load instead of stalling the first live frame. (Helps the ~0.24 s decode
phase; prefill is unaffected — it's a single pass.)

### 1. One big model resident
- `mx.clear_cache()` is called whenever an engine is torn down (VLM model switch)
  and whenever the Falcon detector is stopped — the previous model's buffers are
  returned to the OS immediately instead of lingering in MLX's cache.
- When a VLM becomes the active model, the Falcon detector (if loaded) is freed,
  so Gemma runs without a second big model co-resident.

### 2. Visualiser buttons (free the embedded Chromium)
Row 1 now shows `Visualiser: [ Internal (QtWebEngine) ] [ Browser ]`. Selecting
**Browser** blanks the embedded `QWebEngineView` (releasing the ThreeJS page and
its GPU surface) and opens the viewer in the system browser — freeing ~1–3 GB
back to Metal for Gemma. **Internal** re-embeds it. Layout is otherwise unchanged.

### 3. Wired-limit pin (with off switch)
A `Pin weights (wired)` checkbox (Runtime row, default ON). When on, after a VLM
loads the app calls `mx.set_wired_limit(N)` sized to the loaded model (≈ peak×1.25,
capped at ~60 % of the Metal working set ≈ 10 GB on this box), pinning weights so
Metal cannot evict them. Unchecking calls `mx.set_wired_limit(0)` so you can
directly A/B whether pinning helps on your workload. Env override:
`PAVE_WIRED_LIMIT_GB`.

---

## 3. Results

Isolated bench (Gemma only, no memory pressure) — as expected, the memory work
does **not** change latency when nothing is competing for the working set; it
removes the *pressure-induced* slowdown that shows up in the full app:

| Scenario | Median / frame | Notes |
|---|---|---|
| Gemma, no pin | ~1819 ms | prefill-bound floor |
| Gemma, **wired pin ON** (9.5 GB) | **1824 ms** | no regression; active 6.8 GB, peak 7.6 GB |
| warmup | first frame no longer pays `mx.compile` | folded into model load |

The wired pin and cache-clear are **preventative**: their benefit appears when
Gemma would otherwise be evicted (Falcon + QtWebEngine + OS pushing past the
~17 GB working set). Use the **`Pin weights (wired)`** checkbox and the
**Visualiser: Browser** button to A/B the difference on a loaded machine — the
console logs `[mem]` lines (wired size, active/peak) on each change.

### Files touched
- `pave_mlx/mlx_mem.py` (new) — `clear_cache`, `set_wired_limit`, `suggested_wired_bytes`, `snapshot`.
- `pave_ui/viewer.py` — Visualiser buttons + `_set_visualiser`/`_update_viz_buttons`; `Pin weights` checkbox + `_apply_wired_limit`/`_toggle_wired`; `_teardown_falcon`/`_free_falcon_for_vlm`; `clear_cache` in `_teardown_engine`; wired pin + Falcon-free in `_on_engine_ready`.
- `pave_ui/perception.py` — VLM warmup in `build_engine` (prior pass), retained.

### Honest bottom line
On an M4 Pro the model is not compute-starved — ~1.8 s is a prefill floor for
Gemma 4 E4B (280 image tokens through an ~8B model), and the runtime is already
the newest published MLX. These changes make the app **hold** that ~1.8 s under
real multi-component memory load instead of degrading to ~3 s. To go materially
below ~1.8 s still requires a smaller model (E2B ~2×) or a single-pass detector
(Falcon-300M ~0.4 s, boxes only).

---

## 4. Second pass — Gemma 4 E2B default + KV-cache investigation

### E2B added and made the default — ✅ (the real speed win)
`Gemma 4 E2B` (`lmstudio-community/gemma-4-E2B-it-MLX-4bit`, 4.1 GB) is now a
model option and the **default on startup**. Measured on the M4 Pro vs E4B:

| Model | Median / frame | Peak memory | Intent + FEATURE |
|---|---|---|---|
| Gemma 4 E4B | ~1820 ms | 7.66 GB | ✅ |
| **Gemma 4 E2B** | **~1136 ms** | **5.2 GB** | ✅ |

~1.6× faster **and** ~2.5 GB less memory — so it also eases the working-set
pressure from §1. Loads cleanly with the existing shared-KV load filter. Wiring:
new `GemmaE2BVLMBackend` + `"gemma_e2b"` registry key; added to `MODELS`,
`MODEL_BACKEND`, `VLM_BACKENDS`, `VLM_MODELS`, `MODEL_SIZE_GB`, `VLM_NAMES`.

### Startup: no embedded ThreeJS, Browser highlighted — ✅
`_rendering_embedded` now defaults to **False** and the `QWebEngineView` is
created **lazily** (only when *Internal* is chosen), so **no Chromium process
spins up at startup** — the ~1–3 GB stays available to the VLM. The **Browser**
button is highlighted by default; the viewer content area shows the browser URL
placeholder instead of an embedded scene.

### KV cache / "don't re-encode the prompt every frame" — ❌ not feasible for this path
Investigated thoroughly; it cannot help a live camera VLM, for two independent
reasons (both evidence-backed):

1. **The prompt is image-first and the image changes every frame.** The actual
   formatted Gemma prompt is:
   ```
   <bos><|turn>user\n<|image|>{instruction}<turn|>\n<|turn>model\n
   ```
   The `<|image|>` block expands to **280 tokens (≈63 % of the 445-token prompt)**
   and sits at the very front. A live feed sends a **new image every frame**, so
   that KV is different every frame and cannot be reused. The instruction text
   comes *after* the image, so its KV depends on the changing image → also not
   reusable. Only `<bos><|turn>user\n` (~3 tokens) precedes the image; caching it
   saves nothing. "Only append minimal updates for new frames" is impossible here
   because **the new frame's image *is* the update — 280 fresh tokens that must be
   encoded and prefilled every time.**
2. **mlx-vlm 0.6.3 exposes no cache API.** `generate(model, processor, prompt,
   image, ...)` takes no `prompt_cache`; there is no `make_prompt_cache`. Reusing
   a KV cache would require forking the library — for a best-case saving of only
   the ~165 non-image tokens, which are unreachable anyway because of (1).

**Verdict:** KV-cache reuse is the wrong lever for a changing-image VLM. The
lever that *did* work is the smaller model (E2B, above). Left unimplemented on
purpose rather than shipping a no-op cache.

### Second-pass bottom line
Default is now E2B (~1.1 s, ~5 GB) with no Chromium at startup — a real,
compounding win on both latency and the memory pressure from §1. Sub-second would
need E2B + reduced image tokens (not exposed by the model) or a single-pass
detector; KV reuse across frames is not applicable.
