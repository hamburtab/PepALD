"""Evaluation modules for PepAR-Diff."""

from importlib import import_module

__all__ = ["Permeability", "PepTuneSolubility", "Solubility"]


def __getattr__(name: str):
    if name == "Permeability":
        return import_module("pepar_diff.evaluation.permeability").Permeability
    if name == "PepTuneSolubility":
        return import_module("pepar_diff.evaluation.solubility").PepTuneSolubility
    if name == "Solubility":
        return import_module("pepar_diff.evaluation.solubility").Solubility
    raise AttributeError(f"module 'pepar_diff.evaluation' has no attribute {name!r}")
