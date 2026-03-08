import os
import sys
import warnings
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from rdkit.Chem import Descriptors, rdMolDescriptors
import joblib
from rdkit import Chem, rdBase, DataStructs
from rdkit.Chem import AllChem
from typing import List

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from utils.helm2smiles import get_cycpep_smi_from_helm

rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def fingerprints_from_mol(molecule, radius=3, size=2048, hashed=False):
    """
        Create ECFP fingerprint of a molecule
    """
    if hashed:
        fp_bits = AllChem.GetHashedMorganFingerprint(molecule, radius, nBits=size)
    else:
        fp_bits = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=size)
    fp_np = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp_bits, fp_np)
    return fp_np.reshape(1, -1)


def fingerprints_from_smiles(smiles: List, size=2048):
    """ Create ECFP fingerprints of smiles, with validity check """
    fps = []
    valid_mask = []
    for i, smile in enumerate(smiles):
        mol = Chem.MolFromSmiles(smile)
        valid_mask.append(int(mol is not None))
        fp = fingerprints_from_mol(mol, size=size) if mol else np.zeros((1, size))
        fps.append(fp)
    
    fps = np.concatenate(fps, axis=0) if len(fps) > 0 else np.zeros((0, size))
    return fps, valid_mask


def getMolDescriptors(mol, missingVal=0):
    """ calculate the full list of descriptors for a molecule """

    values, names = [], []
    for nm, fn in Descriptors._descList:
        try:
            val = fn(mol)
        except:
            val = missingVal
        values.append(val)
        names.append(nm)

    custom_descriptors = {'hydrogen-bond donors': rdMolDescriptors.CalcNumLipinskiHBD,
                          'hydrogen-bond acceptors': rdMolDescriptors.CalcNumLipinskiHBA,
                          'rotatable bonds': rdMolDescriptors.CalcNumRotatableBonds,}
    
    for nm, fn in custom_descriptors.items():
        try:
            val = fn(mol)
        except:
            val = missingVal
        values.append(val)
        names.append(nm)
    return values, names


def get_pep_dps_from_smi(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
    except:
        print(f"convert smi {smi} to molecule failed!")
        mol = None
    
    dps, _ = getMolDescriptors(mol)
    return np.array(dps)


def get_pep_dps(smi_list):
    if len(smi_list) == 0:
        return np.zeros((0, 211))
    return np.array([get_pep_dps_from_smi(smi) for smi in smi_list])


def get_smi_from_helms(helm_seqs: list):
    valid_idxes = []
    valid_smiles = []

    for idx, helm in enumerate(helm_seqs):
        # Ignore helm which cannot converted into molecules
        try:
            smi = get_cycpep_smi_from_helm(helm)
            if smi:
                valid_idxes.append(idx)
                valid_smiles.append(smi)
        except Exception as e:
            # logger.debug(f'Error: {e} in helm {helm}')
            pass
    return valid_smiles, valid_idxes


def check_smi_validity(smiles: list):
    valid_smi, valid_idx = [], []
    for idx, smi in enumerate(smiles):
        try:
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol:
                valid_smi.append(smi)
                valid_idx.append(idx)
        except Exception as e:
            # logger.debug(f'Error: {e} in smiles {smi}')
            pass 
    return valid_smi, valid_idx

class Permeability:
    """
        Predict permeability of peptides using helms as inputs
    """
    def __init__(self, model_path=None, batch_size=1000, input_type='helm'):
        if model_path is None:
            model_path = os.path.join(PROJECT_ROOT, 'data', 'cpp', 'regression_rf.pkl')
        self.predictor = self.load_predictor(model_path)
        self.batch_size = batch_size
        self.input_type = input_type if input_type in ['helm', 'smiles'] else 'helm'

    @staticmethod
    def load_predictor(model_path):
        try:
            predictor = joblib.load(model_path)
        except ValueError:
            # Handle sklearn version mismatch: old model lacks 'missing_go_to_left' field
            import sklearn.tree._tree as _tree
            original_check = _tree._check_node_ndarray
            def _patched_check(node_ndarray, expected_dtype):
                if node_ndarray.dtype != expected_dtype:
                    new_dtype = expected_dtype
                    new_array = np.zeros(node_ndarray.shape, dtype=new_dtype)
                    for field in node_ndarray.dtype.names:
                        new_array[field] = node_ndarray[field]
                    return new_array
                return node_ndarray
            _tree._check_node_ndarray = _patched_check
            predictor = joblib.load(model_path)
            _tree._check_node_ndarray = original_check
        return predictor

    def get_features(self, input_seqs: list):
        if self.input_type == 'helm':
            valid_smiles, valid_idxes = get_smi_from_helms(input_seqs)
        else:
            valid_smiles, valid_idxes = check_smi_validity(input_seqs)

        X_fps = fingerprints_from_smiles(valid_smiles)[0]
        X_dps = get_pep_dps(valid_smiles)
        # logger.debug(f'X_fps.shape: {X_fps.shape}, X_dps.shape: {X_dps.shape}')

        if len(X_fps) == 0:
            valid_features = np.zeros((0, X_fps.shape[1] + X_dps.shape[1]))
        else:
            valid_features = np.concatenate([X_fps, X_dps], axis=1)
            
        return valid_features, valid_idxes

    def __call__(self, input_seqs: list):
        scores = self.get_scores(input_seqs)
        return scores
    
    def get_scores(self, input_seqs: list):
        scores = -10 * np.ones(len(input_seqs))
        valid_features, valid_idxes = self.get_features(input_seqs)
        if len(valid_features) == 0:
            return scores
        
        valid_features = np.nan_to_num(valid_features, nan=0.)
        valid_features = np.clip(valid_features, np.finfo(np.float32).min, np.finfo(np.float32).max)
        scores[valid_idxes] = self.predictor.predict(valid_features)
        return scores


if __name__ == '__main__':
    input_file = os.path.join(PROJECT_ROOT, 'chembl32_samples', 'cyclic_samples.txt')
    with open(input_file, 'r') as f:
        helm_seqs = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(helm_seqs)} HELM sequences from {input_file}")

    permeability = Permeability()
    scores = permeability(helm_seqs)

    valid_scores = scores[scores > -10]
    print(f"\nResults:")
    print(f"  Total sequences: {len(helm_seqs)}")
    print(f"  Valid predictions: {len(valid_scores)}")
    print(f"  Mean permeability: {valid_scores.mean():.4f}")
    print(f"  Std permeability:  {valid_scores.std():.4f}")
    print(f"  Min permeability:  {valid_scores.min():.4f}")
    print(f"  Max permeability:  {valid_scores.max():.4f}")
    print(f"  High permeability (> -6): {(valid_scores > -6).sum()} ({(valid_scores > -6).sum()/len(valid_scores)*100:.1f}%)")
