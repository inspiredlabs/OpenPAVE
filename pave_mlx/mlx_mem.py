"""MLX (Metal) memory helpers for keeping one big model resident on Apple Silicon.

On a memory-constrained Mac — e.g. the M4 Pro / 24 GB of record, where Metal's
recommended working set is only ~17 GB — the dominant cost of VLM inference is
memory pressure, not compute: once resident buffers exceed the working set, Metal
evicts and re-stages weights on every inference and prefill collapses. These
helpers let the UI (a) release the MLX buffer cache when a model is torn down and
(b) pin the loaded model's weights with a wired-memory limit so they cannot be
evicted, with an off switch so the effect can be A/B'd.

Every function degrades to a safe no-op when MLX is unavailable (tests / CI /
non-Apple hosts), so importing this never fails and callers need no guards.
"""

from __future__ import annotations

import os

_GB = 1024 ** 3


def _mx():
    try:
        import mlx.core as mx

        return mx
    except Exception:  # noqa: BLE001 - MLX absent -> caller no-ops
        return None


def available() -> bool:
    return _mx() is not None


def clear_cache() -> None:
    """Return MLX's buffer cache to the OS (frees a just-torn-down model)."""
    mx = _mx()
    if mx is not None:
        try:
            mx.clear_cache()
        except Exception:  # noqa: BLE001
            pass


def recommended_working_set_bytes() -> int:
    mx = _mx()
    if mx is None:
        return 0
    try:
        return int(mx.device_info().get("max_recommended_working_set_size", 0))
    except Exception:  # noqa: BLE001
        return 0


def peak_bytes() -> int:
    mx = _mx()
    if mx is None:
        return 0
    try:
        return int(mx.get_peak_memory())
    except Exception:  # noqa: BLE001
        return 0


def snapshot() -> dict:
    """{active, peak, cache} bytes — for logging so the effect is visible."""
    mx = _mx()
    if mx is None:
        return {}
    try:
        return {
            "active": int(mx.get_active_memory()),
            "peak": int(mx.get_peak_memory()),
            "cache": int(mx.get_cache_memory()),
        }
    except Exception:  # noqa: BLE001
        return {}


def under_pressure(reserve_bytes: int = 3 * _GB) -> bool:
    """True only when resident MLX memory is close enough to the Metal working set
    that holding a SECOND big model (e.g. Falcon alongside the VLM) risks eviction.

    Conservative by design: returns False whenever there's comfortable headroom, so
    the Falcon overlay coexists with the VLM on roomy Macs (e.g. E2B ~5GB + 300M
    ~1.2GB on a 24GB / ~17GB-working-set box never trips this). It only trips on
    genuinely tight machines, where freeing the detector protects VLM speed."""
    mx = _mx()
    if mx is None:
        return False
    ws = recommended_working_set_bytes()
    if not ws:
        return False
    try:
        active = int(mx.get_active_memory())
    except Exception:  # noqa: BLE001
        return False
    return active + reserve_bytes > ws


def suggested_wired_bytes(model_peak_bytes: int | None = None) -> int:
    """A safe wired limit: big enough to pin the loaded model, capped well under
    the Metal working set so the OS + UI (incl. QtWebEngine) keep headroom.

    Override with PAVE_WIRED_LIMIT_GB (a hard value, still floored at 2 GB)."""
    env = os.environ.get("PAVE_WIRED_LIMIT_GB")
    if env:
        try:
            return max(2 * _GB, int(float(env) * _GB))
        except Exception:  # noqa: BLE001
            pass
    ws = recommended_working_set_bytes()
    cap = int(ws * 0.6) if ws else 10 * _GB  # ~10 GB on a 24 GB / 17 GB-working-set box
    if model_peak_bytes:
        want = int(model_peak_bytes * 1.25)
        return min(max(want, 2 * _GB), cap)
    return cap


def set_wired_limit(nbytes: int) -> int | None:
    """Pin up to `nbytes` of weights as wired (non-evictable). 0 disables/relaxes.
    Returns the previous limit in bytes, or None if MLX is unavailable."""
    mx = _mx()
    if mx is None:
        return None
    try:
        return int(mx.set_wired_limit(int(max(0, nbytes))))
    except Exception:  # noqa: BLE001
        return None
