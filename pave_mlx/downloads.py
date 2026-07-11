"""Deterministic Hugging Face cache preflight for OpenPAVE VLMs."""

from __future__ import annotations

import json
import os
import re
import struct
import sys
from pathlib import Path

from pave_mlx.backends import backend_model_id, checkpoint_label

# Preflight titles are DERIVED from each backend's checkpoint id (basename) so
# the CLI line, the dropdown, and the HF cache dirs on disk always agree.
_PREFLIGHT_SIZE_BY_KEY = {
    "qwen": 2.5,
    "qwen_2b": 1.78,
    "qwen_8b": 5.4,
    "qwen35_2b": 4.5,
    "moondream3": 5.5,
    "smolvlm_256m": 0.52,
    "fourier_qwen2vl_2b": 4.42,
    "fourier_4bit": 1.1,
    "fourier_3bit": 1.9,
    "gemma": 6.83,
}
VLM_MODELS = {checkpoint_label(key): key for key in _PREFLIGHT_SIZE_BY_KEY}
MODEL_SIZE_GB = {checkpoint_label(key): size for key, size in _PREFLIGHT_SIZE_BY_KEY.items()}
_MODEL_CARD_NOTES_BY_KEY = {
    "qwen_2b": {
        "precision": "4-bit MLX (3-bit collapses to token repetition on this 2B)",
        "reported_size": "1.78 GB",
        "source": "Hugging Face model card",
    },
    "qwen_8b": {
        "precision": "4-bit MLX",
        "reported_size": "5.4 GB",
        "source": "LM Studio model card",
    },
    "qwen35_2b": {
        "precision": "bf16 safetensors (quantized MLX exports of this arch are broken; "
                     "the Rishu11277 mlx-lm export has no vision tower)",
        "reported_size": "4.5 GB",
        "source": "Hugging Face model card",
    },
    "fourier_qwen2vl_2b": {
        "precision": "bf16 safetensors (GGUF quants unusable on MLX; serving the source repo)",
        "reported_size": "4.42 GB",
        "source": "Hugging Face model card",
    },
    "gemma": {
        "precision": "4-bit MLX",
        "reported_size": "6.83 GB",
        "source": "LM Studio model card",
    },
}
MODEL_CARD_NOTES = {checkpoint_label(key): notes for key, notes in _MODEL_CARD_NOTES_BY_KEY.items()}
WEIGHT_EXTENSIONS = (".safetensors", ".npz", ".bin")
RUNTIME_EXTENSIONS = (".json", ".jinja", ".txt", ".safetensors", ".npz", ".bin")

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def color(text: str, code: str) -> str:
    if os.environ.get("NO_COLOR"):
        return text
    return f"{code}{text}{RESET}"


def fmt_bytes(size: int | float | None) -> str:
    if size is None:
        return "unknown"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1000.0 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1000.0
    return f"{value:.2f} TB"


def hf_cache_dir() -> Path:
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return home / "hub"


def model_root(name: str) -> Path:
    repo = backend_model_id(VLM_MODELS[name])
    return hf_cache_dir() / ("models--" + repo.replace("/", "--"))


def model_blobs_dir(name: str) -> Path:
    return model_root(name) / "blobs"


def model_snapshot_dir(name: str) -> Path | None:
    root = model_root(name)
    refs_main = root / "refs" / "main"
    snapshots = root / "snapshots"
    if refs_main.is_file():
        snapshot = snapshots / refs_main.read_text(encoding="utf-8").strip()
    else:
        dirs = [p for p in snapshots.glob("*") if p.is_dir()] if snapshots.exists() else []
        snapshot = max(dirs, key=lambda p: p.stat().st_mtime) if dirs else None
    if not snapshot or not snapshot.exists():
        return None
    return snapshot


def indexed_weight_files(name: str) -> tuple[list[str], int | None]:
    snapshot = model_snapshot_dir(name)
    if snapshot is None:
        return [], None
    index = snapshot / "model.safetensors.index.json"
    if not index.exists():
        return [], None
    try:
        data = json.loads(index.read_text(encoding="utf-8"))
    except Exception:
        return [], None
    return sorted(set(data.get("weight_map", {}).values())), data.get("metadata", {}).get("total_size")


def indexed_weight_keys(name: str) -> list[str]:
    snapshot = model_snapshot_dir(name)
    if snapshot is None:
        return []
    index = snapshot / "model.safetensors.index.json"
    if not index.exists():
        return []
    try:
        data = json.loads(index.read_text(encoding="utf-8"))
    except Exception:
        return []
    return sorted(data.get("weight_map", {}))


def safetensors_metadata(name: str) -> dict[str, object]:
    snapshot = model_snapshot_dir(name)
    if snapshot is None:
        return {}
    metadata = {}
    for path in sorted(snapshot.glob("*.safetensors")):
        try:
            with path.open("rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_len))
            metadata[path.name] = header.get("__metadata__", {})
        except Exception:
            metadata[path.name] = {}
    return metadata


def snapshot_runtime_files(name: str) -> list[str]:
    snapshot = model_snapshot_dir(name)
    if snapshot is None:
        return []
    try:
        return sorted(path.name for path in snapshot.iterdir() if path.name.endswith(RUNTIME_EXTENSIONS))
    except OSError:
        return []


def snapshot_config(name: str) -> dict[str, object]:
    snapshot = model_snapshot_dir(name)
    if snapshot is None:
        return {}
    path = snapshot / "config.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def model_config_summary(name: str) -> dict[str, object]:
    data = snapshot_config(name)
    if not data:
        return {}
    text_config = data.get("text_config") if isinstance(data.get("text_config"), dict) else {}
    audio_config = data.get("audio_config") if isinstance(data.get("audio_config"), dict) else {}
    vision_config = data.get("vision_config") if isinstance(data.get("vision_config"), dict) else {}
    quantization = data.get("quantization") if isinstance(data.get("quantization"), dict) else {}
    arch = data.get("architectures")
    if isinstance(arch, list):
        arch = ", ".join(str(item) for item in arch)
    return {
        "model_type": data.get("model_type"),
        "architectures": arch,
        "text_model_type": text_config.get("model_type"),
        "text_layers": text_config.get("num_hidden_layers"),
        "text_shared_kv_layers": text_config.get("num_kv_shared_layers"),
        "vision_model_type": vision_config.get("model_type"),
        "audio_model_type": audio_config.get("model_type"),
        "image_token_id": data.get("image_token_id"),
        "audio_token_id": data.get("audio_token_id"),
        "quantization_bits": quantization.get("bits"),
        "quantization_group_size": quantization.get("group_size"),
    }


def shared_kv_extra_weights(name: str) -> list[str]:
    data = snapshot_config(name)
    text_config = data.get("text_config") if isinstance(data.get("text_config"), dict) else {}
    layers = text_config.get("num_hidden_layers")
    shared = text_config.get("num_kv_shared_layers")
    if not isinstance(layers, int) or not isinstance(shared, int) or shared <= 0:
        return []

    first_shared_layer = layers - shared
    extra = []
    pattern = re.compile(r"language_model\.model\.layers\.(\d+)\.self_attn\.([^.]+)\.")
    for key in indexed_weight_keys(name):
        match = pattern.match(key)
        if not match:
            continue
        layer_idx = int(match.group(1))
        part = match.group(2)
        if layer_idx >= first_shared_layer and part in {"k_proj", "v_proj", "k_norm", "v_norm"}:
            extra.append(key)
    return extra


def loader_compat_report(name: str) -> dict[str, object]:
    metadata = safetensors_metadata(name)
    mlx_format = any(item.get("format") == "mlx" for item in metadata.values() if isinstance(item, dict))
    shared_kv = shared_kv_extra_weights(name)
    return {
        "safetensors_metadata": metadata,
        "mlx_format": mlx_format,
        "shared_kv_extra_count": len(shared_kv),
        "shared_kv_extra_examples": shared_kv[:6],
        "openpave_filter": VLM_MODELS.get(name) == "gemma" and bool(shared_kv),
    }


def incomplete_cache_files(name: str | None = None) -> list[Path]:
    names = [name] if name else list(VLM_MODELS)
    files: list[Path] = []
    for model_name in names:
        if model_name not in VLM_MODELS:
            continue
        blobs = model_blobs_dir(model_name)
        if blobs.exists():
            files.extend(sorted(blobs.glob("*.incomplete")))
    return files


def snapshot_weight_size(name: str) -> int:
    snapshot = model_snapshot_dir(name)
    if snapshot is None:
        return 0
    total = 0
    for path in snapshot.iterdir():
        if path.name.endswith(WEIGHT_EXTENSIONS):
            try:
                total += path.stat().st_size
            except OSError:
                pass
    return total


def missing_snapshot_files(name: str) -> list[str]:
    snapshot = model_snapshot_dir(name)
    if snapshot is None:
        return ["snapshot"]
    try:
        files = [p for p in snapshot.iterdir() if p.is_file()]
    except OSError:
        return ["snapshot"]
    missing: list[str] = []
    if not any((snapshot / f).is_file() for f in ("config.json", "tokenizer_config.json")):
        missing.append("config/tokenizer")
    missing.extend(path.name for path in files if not path.exists())
    weight_files, _ = indexed_weight_files(name)
    if weight_files:
        for weight in weight_files:
            path = snapshot / weight
            if not path.is_file() or path.stat().st_size <= 0:
                missing.append(weight)
    elif not any(p.name.endswith(WEIGHT_EXTENSIONS) and p.stat().st_size > 0 for p in files):
        missing.append("weights")
    return sorted(set(missing))


def model_download_report(name: str) -> dict[str, object]:
    if name not in VLM_MODELS:
        return {}
    repo = backend_model_id(VLM_MODELS[name])
    snapshot = model_snapshot_dir(name)
    runtime_files = snapshot_runtime_files(name)
    weight_files, indexed_total = indexed_weight_files(name)
    config_summary = model_config_summary(name)
    expected_files = sorted(set(runtime_files) | set(weight_files))
    present: list[dict[str, object]] = []
    missing: list[str] = []
    present_weight_bytes = 0
    indexed_weight_set = set(weight_files)

    if snapshot is None:
        missing.append("snapshot")

    for filename in expected_files:
        path = snapshot / filename if snapshot is not None else None
        if path is not None and path.is_file():
            size = path.stat().st_size
            present.append({"name": filename, "size": size})
            if filename in indexed_weight_set or (not indexed_weight_set and filename.endswith(WEIGHT_EXTENSIONS)):
                present_weight_bytes += size
        else:
            missing.append(filename)

    incomplete = []
    for path in incomplete_cache_files(name):
        try:
            incomplete.append({"name": path.name, "size": path.stat().st_size})
        except OSError:
            pass

    expected_weight_bytes = int(indexed_total or snapshot_weight_size(name) or MODEL_SIZE_GB.get(name, 0) * 1e9)
    missing_weight_bytes = max(0, expected_weight_bytes - present_weight_bytes)
    if missing_weight_bytes > 0 and not any(item in missing for item in ("snapshot", "weights")):
        missing.append("weights")
    return {
        "name": name,
        "repo": repo,
        "snapshot": str(snapshot) if snapshot else None,
        "expected_files": expected_files,
        "present": present,
        "missing": missing,
        "incomplete": incomplete,
        "expected_weight_bytes": expected_weight_bytes,
        "present_weight_bytes": present_weight_bytes,
        "missing_weight_bytes": missing_weight_bytes,
        "allow_downloads": os.environ.get("PAVE_ALLOW_MODEL_DOWNLOADS", "0") == "1",
        "config": config_summary,
        "model_card": MODEL_CARD_NOTES.get(name, {}),
        "loader_compat": loader_compat_report(name),
    }


def print_model_download_report(name: str, stream=None) -> None:
    stream = stream or sys.stderr
    report = model_download_report(name)
    if not report:
        return
    expected_files = report["expected_files"]
    present = report["present"]
    missing = report["missing"]
    incomplete = report["incomplete"]
    print("", file=stream)
    print(f"[openpave] VLM cache preflight: {report['name']} ({report['repo']})", file=stream)
    print(f"[openpave]   runtime files : {len(present)}/{len(expected_files)} present", file=stream)
    print(
        f"[openpave]   weights       : {fmt_bytes(report['present_weight_bytes'])} present / "
        f"{fmt_bytes(report['expected_weight_bytes'])} expected / "
        f"{fmt_bytes(report['missing_weight_bytes'])} missing",
        file=stream,
    )
    config = report.get("config") or {}
    if config:
        arch = config.get("architectures") or "unknown architecture"
        model_type = config.get("model_type") or "unknown model_type"
        text_type = config.get("text_model_type") or "unknown text model"
        layers = config.get("text_layers")
        shared = config.get("text_shared_kv_layers")
        shape = f"{layers} text layers" if layers is not None else "unknown text layer count"
        if shared:
            shape = f"{shape}, {shared} shared-KV layers"
        print(f"[openpave]   model shape  : {model_type} / {text_type} / {arch}", file=stream)
        print(f"[openpave]   runtime note : {shape}", file=stream)
        if config.get("quantization_bits"):
            print(
                f"[openpave]   quantization : {config['quantization_bits']}-bit, group_size={config.get('quantization_group_size')}",
                file=stream,
            )
        io_bits = []
        if config.get("image_token_id") is not None:
            io_bits.append(f"image_token_id={config['image_token_id']}")
        if config.get("audio_model_type"):
            io_bits.append(f"audio tower={config['audio_model_type']}")
        if config.get("audio_token_id") is not None:
            io_bits.append(f"audio_token_id={config['audio_token_id']}")
        if io_bits:
            print(f"[openpave]   model I/O    : {', '.join(io_bits)}", file=stream)
            print("[openpave]   app I/O      : OpenPAVE currently wires camera image + text prompt; microphone/audio input is not wired", file=stream)
    model_card = report.get("model_card") or {}
    if model_card:
        print(
            f"[openpave]   model card   : {model_card['precision']} · {model_card['reported_size']} · {model_card['source']}",
            file=stream,
        )
    compat = report.get("loader_compat") or {}
    if compat.get("mlx_format"):
        print("[openpave]   shard format : safetensors metadata format=mlx", file=stream)
    if compat.get("shared_kv_extra_count"):
        print(
            color(
                f"[openpave]   loader gap  : {compat['shared_kv_extra_count']} shared-KV tensors exist in the checkpoint "
                "but not in the instantiated mlx-vlm Gemma4 module",
                YELLOW,
            ),
            file=stream,
        )
        for key in compat.get("shared_kv_extra_examples", [])[:4]:
            print(color(f"[openpave]     extra {key}", YELLOW), file=stream)
        if compat.get("openpave_filter"):
            print(color("[openpave]   compat      : OpenPAVE filters these extras at load_weights()", GREEN), file=stream)
    if incomplete:
        total = sum(item["size"] for item in incomplete)
        print(color(f"[openpave]   partials      : {len(incomplete)} incomplete files ({fmt_bytes(total)})", YELLOW), file=stream)
        for item in incomplete[:4]:
            print(color(f"[openpave]     partial {item['name']} {fmt_bytes(item['size'])}", YELLOW), file=stream)
    cache_complete = not missing and report["missing_weight_bytes"] == 0 and bool(expected_files)
    if not cache_complete:
        print(color(f"[openpave]   missing      : {len(missing)} file(s)", RED), file=stream)
        for filename in missing[:8]:
            print(color(f"[openpave]     MISSING {filename}", RED), file=stream)
        if report["allow_downloads"]:
            print(color("[openpave]   action       : downloads enabled; missing files may be fetched now", YELLOW), file=stream)
        else:
            print(color("[openpave]   action       : downloads disabled; model load will be blocked", RED), file=stream)
    else:
        print(color("[openpave]   cache        : complete; no model bytes expected", GREEN), file=stream)
    print("", file=stream)


def main() -> None:
    names = sys.argv[1:] or list(VLM_MODELS)
    for name in names:
        print_model_download_report(name, stream=sys.stdout)


if __name__ == "__main__":
    main()
