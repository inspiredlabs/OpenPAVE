"""vLLM-MLX server launcher — OpenPAVE's serving tier for the VLM hot path.

Runs ``vllm-mlx serve`` (an OpenAI-compatible vLLM-style engine compiled for
MLX: paged KV cache, automatic prefix caching, continuous batching, SSD-tiered
KV) with compatibility patches applied *before* the model loads:

1. Gemma 4 shared-KV sanitize — the same checkpoint quirk backends.py patches
   for in-process loads (`_patch_gemma4_shared_kv_sanitize`). vllm-mlx uses
   mlx-vlm's gemma4 loader internally, so it hits the identical
   "parameters not in model" failure on otherwise complete caches.

2. Text-only KV prefix cache — vllm-mlx's MLLM prefix cache keys entries by
   input token ids alone. A live camera frame tokenizes to the SAME ids every
   time (fixed prompt text + a fixed-length run of image placeholder tokens;
   the pixels live in ``pixel_values``, which the cache never sees), so
   consecutive frames collide: fetch returns the previous frame's KV as a
   *full-prefix* hit, the upstream "remaining tokens contain image
   placeholders" guard never fires (nothing remains), and the server answers
   from stale state — on the live path that surfaced as ``content: null`` +
   ``finish_reason: length`` for every frame after the first, i.e. permanent
   STOP and no FEATURE overlay. The entries also persist to disk, so a fresh
   server can answer frame 0 from a previous session's webcam frame. The patch
   guards fetch AND store so any token sequence containing image/video
   placeholder tokens bypasses the cache; text-only requests keep the full
   prefix-cache/warm-prompt benefit. (vllm-mlx's separate vision cache is
   keyed by image content hash and is safe — untouched.)

3. Stream rebind coverage — mlx-vlm 0.6.3 split its generate machinery into
   ``mlx_vlm.generate.{ar,common,dispatch,diffusion}``, each holding its own
   module-level ``generation_stream``. vllm-mlx 0.4.0's
   ``bind_generation_streams`` only rebinds ``mlx_vlm.generate`` (the package
   ``__init__``), so generation on a worker thread dies with
   ``RuntimeError: There is no Stream(gpu, 1) in current thread.`` Extending
   the function's default module list covers the submodules the token loop
   actually uses; mutating ``__defaults__`` reaches every ``from … import``
   call site because they all share the one function object.

Usage (any ``vllm-mlx serve`` flags pass through):

    python -m pave_mlx.vllm_server serve <model-id> --port 8977 --offline

The VllmMlxBackend in backends.py spawns this module as a subprocess; it can
also be run standalone to share one server between the GUI and other clients.
"""

from __future__ import annotations

import sys

# Prefix-cache note: the live-camera win depends on the *message ordering* the
# client sends — the invariant instruction must arrive as a system message so
# its KV prefix is cacheable across frames while only the per-frame image
# tokens re-prefill. See VllmMlxBackend.generate() in backends.py.

_MLX_VLM_STREAM_MODULES = (
    "mlx_lm.generate",
    "mlx_vlm.generate",
    "mlx_vlm.generate.ar",
    "mlx_vlm.generate.common",
    "mlx_vlm.generate.dispatch",
    "mlx_vlm.generate.diffusion",
)


def _patch_vllm_stream_rebind() -> bool:
    """Extend vllm-mlx's worker-thread stream rebind to mlx-vlm's submodules."""
    try:
        from vllm_mlx import mlx_streams
    except Exception:
        return False
    fn = getattr(mlx_streams, "bind_generation_streams", None)
    if fn is None or not getattr(fn, "__defaults__", None):
        return False
    current = fn.__defaults__[0]
    merged = tuple(dict.fromkeys(tuple(current) + _MLX_VLM_STREAM_MODULES))
    fn.__defaults__ = (merged,)
    return True


def _patch_mllm_prefix_cache_text_only() -> bool:
    """Bypass the MLLM KV prefix cache for token sequences containing media tokens.

    See module docstring §2. The guard wraps the cache *instance* created in
    ``MLLMBatchGenerator.__init__`` so all four upstream call sites (prefill
    fetch/store, chunked-prefill fetch/store) go through it, and resolves the
    media placeholder token ids lazily from the loaded model's config.
    """
    try:
        from vllm_mlx import mllm_batch_generator as mbg
    except Exception:
        return False
    cls = getattr(mbg, "MLLMBatchGenerator", None)
    if cls is None:
        return False
    if getattr(cls, "_openpave_text_only_prefix_cache", False):
        return True

    original_init = cls.__init__

    def init_with_guarded_cache(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _guard_prefix_cache(self)

    cls.__init__ = init_with_guarded_cache
    cls._openpave_text_only_prefix_cache = True
    return True


def _guard_prefix_cache(generator) -> bool:
    """Wrap `generator.prefix_cache` fetch/store to bypass media token sequences.

    The media placeholder token ids are resolved lazily from the loaded model's
    config on every call (the model may not be attached yet at generator
    construction time). Returns True when a cache was wrapped.
    """
    cache = getattr(generator, "prefix_cache", None)
    if cache is None:
        return False

    def media_token_ids() -> set[int]:
        config = getattr(getattr(generator, "model", None), "config", None)
        ids = set()
        for attr in ("image_token_index", "image_token_id",
                     "video_token_index", "video_token_id"):
            token = getattr(config, attr, None)
            if isinstance(token, int):
                ids.add(token)
        return ids

    original_fetch, original_store = cache.fetch, cache.store

    def fetch_text_only(tokens, *fargs, **fkwargs):
        media = media_token_ids()
        if media and not media.isdisjoint(tokens):
            return None, tokens  # multimodal -> always prefill honestly
        return original_fetch(tokens, *fargs, **fkwargs)

    def store_text_only(tokens, kv_cache, *sargs, **skwargs):
        media = media_token_ids()
        if media and not media.isdisjoint(tokens):
            return False  # never persist KV keyed by pixel-blind ids
        return original_store(tokens, kv_cache, *sargs, **skwargs)

    cache.fetch = fetch_text_only
    cache.store = store_text_only
    return True


def apply_compat_patches() -> list[str]:
    """Apply both patches; returns human-readable notes for logging."""
    from pave_mlx.backends import _patch_gemma4_shared_kv_sanitize

    notes = []
    if _patch_gemma4_shared_kv_sanitize():
        notes.append("gemma4 shared-KV load filter active")
    if _patch_mllm_prefix_cache_text_only():
        notes.append("KV prefix cache restricted to text-only requests")
    if _patch_vllm_stream_rebind():
        notes.append("mlx-vlm 0.6.x generation-stream rebind extended")
    return notes


def main() -> None:
    for note in apply_compat_patches():
        print(f"[pave_mlx.vllm_server] {note}", flush=True)
    from vllm_mlx.cli import main as vllm_main

    sys.argv[0] = "vllm-mlx"
    vllm_main()


if __name__ == "__main__":
    main()
