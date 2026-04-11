"""Shared constants for docking backends."""

from pathlib import Path


INVALID_SCORE = 0.0
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEPTOR = str(PROJECT_ROOT / "data" / "docking" / "6dn5_receptor.pdbqt")
DEFAULT_REF_SDF = str(PROJECT_ROOT / "data" / "docking" / "raw_cyclic_pep.sdf")
