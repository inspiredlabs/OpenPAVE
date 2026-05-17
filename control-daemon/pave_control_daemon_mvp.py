"""Legacy script entry point for the OpenPAVE control daemon.

Prefer:
    python3 -m control_daemon.daemon
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control_daemon.daemon import main

if __name__ == "__main__":
    main()
