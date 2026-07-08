"""pave_ui — PyQt6 operator console for the OpenPAVE MLX runtime.

Ties together the streaming robot-state visualiser (mlx-runtime/state_server.py +
visualiser/index.html), the control plane (intent_ingress + control_daemon), and
the pave_mlx Tier A perception backends, with all the flooded process logs routed
into an in-app console panel instead of the terminal. See docs/dgx-spark-mlx-port.md.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
