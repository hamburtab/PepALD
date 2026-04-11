"""
PepAR-Diff package.

The top-level package uses lazy imports so light-weight submodules such as
`pepar_diff.evaluation` remain usable inside `perm_env` without requiring
training dependencies like PyTorch.
"""

from importlib import import_module

__version__ = "1.0.0"
__all__ = [
    "AutoregressiveLatentDiffusion",
    "CausalContextEncoder",
    "TokenMapper",
    "RingBondPredictor",
    "DiffusionEngine",
    "HELMDataset",
]


def __getattr__(name: str):
    if name == "AutoregressiveLatentDiffusion":
        return import_module("pepar_diff.models.ald_model").AutoregressiveLatentDiffusion
    if name == "CausalContextEncoder":
        return import_module("pepar_diff.models.context_encoder").CausalContextEncoder
    if name == "TokenMapper":
        return import_module("pepar_diff.models.token_mapper").TokenMapper
    if name == "RingBondPredictor":
        return import_module("pepar_diff.models.ring_predictor").RingBondPredictor
    if name == "DiffusionEngine":
        return import_module("pepar_diff.diffusion.engine").DiffusionEngine
    if name == "HELMDataset":
        return import_module("pepar_diff.data.datasets").HELMDataset
    raise AttributeError(f"module 'pepar_diff' has no attribute {name!r}")
