"""Generic entry point for WP-DPO checkpoint-curve sample generation."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate.generate_case1_checkpoint_curve_samples import main


if __name__ == "__main__":
    main()
