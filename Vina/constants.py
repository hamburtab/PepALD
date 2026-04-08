"""Shared constants for docking backends."""

import os.path as osp


INVALID_SCORE = 0.0
DEFAULT_RECEPTOR = osp.join(osp.dirname(__file__), "6dn5_receptor.pdbqt")
DEFAULT_REF_SDF = osp.join(osp.dirname(__file__), "raw_cyclic_pep.sdf")
