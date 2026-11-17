"""Backward-compatible CLI entrypoint for MET Flow (v2.8 MULTI-SOURCE).

Delegates execution to scripts/run_pipeline.py.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.run_pipeline import main


if __name__ == "__main__":
    sys.exit(main())
