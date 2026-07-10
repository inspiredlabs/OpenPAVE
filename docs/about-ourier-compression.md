---
how do I use this in my own work?:
```
.venv/bin/python -m mlx_vlm.convert --hf-path whyisverysmart/Fourier-Qwen2-VL-2B-0.67 -q --mlx-path ./models/fourier-qwen2vl-2b-4bit
FOURIER_QWEN2VL_MODEL=./models/fourier-qwen2vl-2b-4bit ./mlx-runtime.sh # Launch MLX based OpenPAVE
```

be aware of the risk of model collapse. Try:
```
.venv/bin/python -m mlx_vlm.convert --hf-path Rishu11277/Qwen3.5-2B-mlx-fp16 -q --mlx-path ./models/rishu-qwen35-2b-4bit
RISHU_QWEN35_VLM_MODEL=./models/rishu-qwen35-2b-4bit ./mlx-runtime.sh
```

---

# What does Fourier compression do for Qwen?

`Fourier-Qwen2-VL-2B-0.67` uses a **frequency‑domain visual token compression scheme**: the “Fourier 0.67” part means that only about **67% of the original visual information (tokens) is kept** after a learned Fourier‑space compression, while the rest is discarded or down‑weighted to shrink the image token sequence. [huggingface](https://huggingface.co/whyisverysmart/Fourier-Qwen2.5-VL-3B-0.67)

More concretely, based on the **Fourier Compressor: Frequency‑Domain Visual Token Compression for Vision‑Language Models** work: [arxiv](https://arxiv.org/html/2508.06038v3)

## What “Fourier 0.67” means

- The HF card for `Fourier-Qwen2.5-VL-3B-0.67` lists **Fourier-Qwen2-VL-2B-0.67** as an official checkpoint of the Fourier Compressor method, with:

  > `Dynamic` mode and `55.6%` compression. [huggingface](https://huggingface.co/whyisverysmart/Fourier-Qwen2.5-VL-3B-0.67)

- “0.67” refers to the **retained amplitude fraction** in the Fourier domain (roughly two‑thirds of the energy), which translates into keeping **around 55–67% of visual tokens** vs the original sequence, depending on the exact layer and configuration. [arxiv](https://arxiv.org/html/2508.06038v3)

In other words:

- The visual stream (image patches/features) is transformed into a frequency representation (2D Fourier domain).
- Tokens (or channels) are **ranked by importance** in frequency space—low‑frequency components (broad structure) are more important, high‑frequency components (fine detail) can be partially dropped.
- The compressor **keeps only the most informative subset** whose cumulative Fourier energy is about 0.67, leading to **~55.6% token count** relative to the full image token sequence for Qwen2/Qwen2.5. [huggingface](https://huggingface.co/whyisverysmart/Fourier-Qwen2.5-VL-3B-0.67)

So “Fourier 0.67” encodes the compression *target*: retain ~67% of the spectral energy, which yields a compressed token sequence with roughly 55–60% of the original length.

## How the compression actually works (in the VLM pipeline)

From the Fourier Compressor paper: [arxiv](https://arxiv.org/html/2508.06038v3)

1. **Visual features extraction**
   - The Qwen2‑VL vision encoder (ViT‑style) produces a sequence of patch embeddings (visual tokens) for the input image. [arxiv](https://arxiv.org/html/2409.12191v1)

2. **Frequency‑domain transformation**
   - These patch embeddings are arranged spatially and passed through a **Fourier transform** (e.g. 2D FFT on the spatial grid of features).
   - This moves the representation from pixel/patch space into frequency space, where low‑frequency components capture global structure and high‑frequency components capture local details. [arxiv](https://arxiv.org/html/2508.06038v3)

3. **Energy‑based or learned selection at 0.67 level**
   - The compressor computes **spectral energy** or uses a learned mask to rank frequency components.
   - It then **selects the subset of Fourier coefficients** whose cumulative energy reaches the 0.67 threshold—this is the “0.67” setting.
   - Remaining coefficients (less informative high frequencies) are either dropped or heavily attenuated, effectively reducing the number of visual tokens passed to the language model. [huggingface](https://huggingface.co/whyisverysmart/Fourier-Qwen2.5-VL-3B-0.67)

4. **Inverse mapping to compressed token sequence**
   - The selected Fourier coefficients are mapped back to a **compressed spatial/token representation** (e.g. via inverse Fourier transform restricted to the kept frequencies, or via a projection).
   - This yields a shorter sequence of visual tokens with similar semantic content but fewer elements, so the VLM’s transformer layers see a **compressed image representation**. [arxiv](https://arxiv.org/html/2508.06038v3)

5. **Joint training / fine‑tuning**
   - The whole VLM (Qwen2‑VL‑2B‑Instruct) is fine‑tuned with this compressor in the loop, so the language side learns to work with compressed visual sequences without losing too much performance on vision–language tasks. [huggingface](https://huggingface.co/whyisverysmart/Fourier-Qwen2.5-VL-3B-0.67)

The result: **visual token count drops by about 44–45% (down to 55.6% of original)** while preserving most of the task‑relevant semantics, which reduces memory and compute for the VLM’s vision + cross‑attention layers. [huggingface](https://huggingface.co/whyisverysmart/Fourier-Qwen2.5-VL-3B-0.67)

## Why this matters for you (performance & scaling)

For a robotics / real‑time setup:

- **Less visual tokens** → shorter prefill and cheaper attention → lower latency and memory use per frame.
- **Fourier 0.67** gives you a **fixed, predictable compression ratio** (roughly halving tokens) while still keeping enough information for typical VLM tasks (captioning, spatial Q&A). [arxiv](https://arxiv.org/html/2508.06038v3)

So when you see `Fourier-Qwen2-VL-2B-0.67`:

- Think: “Qwen2‑VL‑2B‑Instruct where the **image stream** is compressed in Fourier space so that ~0.67 of spectral energy (≈ 55–60% tokens) is retained.”
- It’s not weight quantization (like 4‑bit); it’s **visual token compression** using Fourier transforms, targeting better throughput on vision tasks without retraining the entire model from scratch. [huggingface](https://huggingface.co/whyisverysmart/Fourier-Qwen2.5-VL-3B-0.67)
