from rdkit import Chem
from rdkit.Chem import AllChem
import os.path as osp
import os, sys
import time
from typing import Optional, Union
from meeko import MoleculePreparation
from meeko import PDBQTWriterLegacy
from meeko import PDBQTMolecule
from meeko import RDKitMolCreate
from vina import Vina
#from collections import defaultdict


INVALID_SCORE = 0.0
DEFAULT_RECEPTOR = osp.join(osp.dirname(__file__), "6dn5_receptor.pdbqt")
DEFAULT_REF_SDF = osp.join(osp.dirname(__file__), "raw_cyclic_pep.sdf")


def _build_vina_instance(device: str = "cuda", cpu: int = 1) -> Vina:
    device = str(device).lower()
    if device == "cuda":
        try:
            return Vina(sf_name='Vina', cpu=cpu, verbosity=0, gpu=True)
        except TypeError as e:
            raise RuntimeError(
                "Current vina Python package does not expose CUDA mode. "
                "Install a CUDA-enabled Vina build or set vina_device='cpu'."
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Vina with CUDA: {e}") from e
    return Vina(sf_name='Vina', cpu=cpu, verbosity=0)


def vina_score(
    ligand_mol_smi: str,
    protein_pdbqt_path: str,
    reference_mol: Chem.rdchem.Mol,
    device: str = "cuda",
    cpu: int = 1,
    exhaustiveness: int = 8,
    n_poses: int = 2,
    ):
    mmff_sec = 0.0
    dock_sec = 0.0

    ligand_mol = Chem.MolFromSmiles(ligand_mol_smi)
    if ligand_mol is None:
        return INVALID_SCORE, mmff_sec, dock_sec

    ligand_mol = Chem.AddHs(ligand_mol)
    embed_status = AllChem.EmbedMolecule(ligand_mol, randomSeed=42)
    if embed_status == -1:
        embed_status = AllChem.EmbedMolecule(ligand_mol, useRandomCoords=True, randomSeed=42)
        if embed_status == -1:
            return INVALID_SCORE, mmff_sec, dock_sec

    mmff_t0 = time.perf_counter()
    try:
        AllChem.MMFFOptimizeMolecule(ligand_mol, maxIters=500)
    except Exception:
        pass
    mmff_sec = time.perf_counter() - mmff_t0

    try:
        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(ligand_mol)
        pdbqt_string, is_ok, _ = PDBQTWriterLegacy.write_string(mol_setups[0])
        if not is_ok:
            return INVALID_SCORE, mmff_sec, dock_sec
    except Exception:
        return INVALID_SCORE, mmff_sec, dock_sec

    # Optional debug hook from your mentor's suggestion:
    # export VINA_ASSERT_BEFORE_DOCK=1 to stop right before docking and inspect MMFF timing.
    if os.getenv("VINA_ASSERT_BEFORE_DOCK", "0") == "1":
        assert 1 == 0, f"pre-dock timing: mmff={mmff_sec:.3f}s"

    dock_t0 = time.perf_counter()
    try:
        center = reference_mol.GetConformers()[0].GetPositions().mean(axis=0)
        v = _build_vina_instance(device=device, cpu=cpu)
        v.set_receptor(protein_pdbqt_path)
        v.set_ligand_from_string(pdbqt_string)
        v.compute_vina_maps(center=center.tolist(), box_size=[30, 30, 30])
        v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
        docking_score = v.energies()[0][0]
    except RuntimeError:
        raise
    except Exception:
        dock_sec = time.perf_counter() - dock_t0
        return INVALID_SCORE, mmff_sec, dock_sec

    dock_sec = time.perf_counter() - dock_t0

    return docking_score, mmff_sec, dock_sec