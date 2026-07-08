"""pave_mlx — local MLX perception backends + intent heads for OpenPAVE Tier A.

See docs/dgx-spark-mlx-port.md §3.2. The split is:

  A  openai_shim.py     one universal OpenAI-compatible server
  B  heads/configs/*    one manifest schema, per-backend serialised artifacts
  C  backends.py        per-model featurizer adapters (the only per-model code);
                        heads collapse to 2 types, the trainer/labeler to 1 each
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
