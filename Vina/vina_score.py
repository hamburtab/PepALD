from rdkit import Chem
from rdkit.Chem import AllChem
import os.path as osp
import os, sys
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


def vina_score(
    ligand_mol_smi: str,
    protein_pdbqt_path: str,
    reference_mol: Chem.rdchem.Mol,
    ):

    ligand_mol = Chem.MolFromSmiles(ligand_mol_smi)
    if ligand_mol is None:
        return INVALID_SCORE

    ligand_mol = Chem.AddHs(ligand_mol)
    embed_status = AllChem.EmbedMolecule(ligand_mol, randomSeed=42)
    if embed_status == -1:
        embed_status = AllChem.EmbedMolecule(ligand_mol, useRandomCoords=True, randomSeed=42)
        if embed_status == -1:
            return INVALID_SCORE

    try:
        AllChem.MMFFOptimizeMolecule(ligand_mol, maxIters=500)
    except Exception:
        pass

    try:
        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(ligand_mol)
        pdbqt_string, is_ok, _ = PDBQTWriterLegacy.write_string(mol_setups[0])
        if not is_ok:
            return INVALID_SCORE
    except Exception:
        return INVALID_SCORE

    try:
        center = reference_mol.GetConformers()[0].GetPositions().mean(axis=0)
        v = Vina(sf_name='Vina', cpu=1, verbosity=0)
        v.set_receptor(protein_pdbqt_path)
        v.set_ligand_from_string(pdbqt_string)
        v.compute_vina_maps(center=center.tolist(), box_size=[30, 30, 30])
        v.dock(exhaustiveness=32, n_poses=8)
        docking_score = v.energies()[0][0]
    except Exception:
        return INVALID_SCORE

    return docking_score