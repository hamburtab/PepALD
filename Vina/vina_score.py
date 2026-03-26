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


def vina_score(
    ligand_mol_smi: str, # generated mol
    protein_pdbqt_path: str,     # pdbqt file
    reference_mol: Chem.rdchem.Mol,    # sdf file
    ):
    """
    ligand_mol_smi: ligand_mol_smi 环肽的smiles
    protein_pdbqt_path: 输入蛋白质口袋的pdbqt文件地址
    reference_mol: 待优化的参考分子，带有三维坐标
    """
    ligand_mol= Chem.MolFromSmiles(ligand_mol_smi) 
    ligand_mol=Chem.AddHs(ligand_mol)
    AllChem.EmbedMolecule(ligand_mol)

    AllChem.MMFFOptimizeMolecule(ligand_mol)
    preparator = MoleculePreparation()
    mol_setups = preparator.prepare(ligand_mol)
    pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(mol_setups[0])
    v = Vina(sf_name='Vina', cpu=1)
    v.set_receptor(protein_pdbqt_path)
    v.set_ligand_from_string(pdbqt_string)
    v.compute_vina_maps(center=reference_mol.GetConformers()[0].GetPositions().mean(0), box_size=[30, 30, 30])
    energy_minimized = v.optimize()

    v.dock(exhaustiveness=32, n_poses=8)

    docking_score = v.score()[0]

    #ligand_mol.SetProp('docking_score', str(docking_score))
    
    return docking_score