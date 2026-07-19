"""Morgan fingerprint extraction used by the ChemEmb ablation."""

from collections.abc import Iterable
from operator import index

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def _bitvect_to_numpy(fp) -> np.ndarray:
    arr = np.zeros((fp.GetNumBits(),), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _normalize_atom_indices(input_idxs: Iterable[int], num_atoms: int) -> list[int]:
    atom_indices = []
    for atom_idx in input_idxs:
        try:
            normalized_idx = index(atom_idx)
        except TypeError as exc:
            raise TypeError(f"Atom index must be an integer, got {atom_idx!r}") from exc

        if normalized_idx < 0 or normalized_idx >= num_atoms:
            raise IndexError(
                f"Atom index {normalized_idx} is out of range for molecule with {num_atoms} atoms"
            )
        atom_indices.append(normalized_idx)
    return atom_indices


def get_morgan_fingerprints(
    smiles: str,
    input_idxs: Iterable[int],
    radius: int = 2,
    n_bits: int = 1024,
    include_chirality: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a molecule Morgan fingerprint and atom-centered Morgan fingerprints.

    Args:
        smiles: Input molecule SMILES.
        input_idxs: Atom indices in RDKit atom-index order.
        radius: Morgan fingerprint radius. Defaults to ECFP4-style radius 2.
        n_bits: Fingerprint length.
        include_chirality: Whether to encode chirality in the Morgan fingerprint.

    Returns:
        A tuple of:
        - molecule_fp: uint8 numpy array with shape ``(n_bits,)``.
        - atom_fps: uint8 numpy array with shape ``(len(input_idxs), n_bits)``.
          Each row is the Morgan fingerprint generated from the corresponding atom.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    atom_indices = _normalize_atom_indices(input_idxs, mol.GetNumAtoms())
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        includeChirality=include_chirality,
    )

    molecule_fp = _bitvect_to_numpy(generator.GetFingerprint(mol))
    if atom_indices:
        atom_fps = np.vstack(
            [
                _bitvect_to_numpy(generator.GetFingerprint(mol, fromAtoms=[atom_idx]))
                for atom_idx in atom_indices
            ]
        )
    else:
        atom_fps = np.zeros((0, n_bits), dtype=np.uint8)

    return molecule_fp, atom_fps
