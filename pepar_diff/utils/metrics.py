"""Metrics for generated HELM samples."""
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, DataStructs

rdBase.DisableLog('rdApp.error')

from .helm import get_cycpep_smi_from_helm
from .sascore import sascorer


def fingerprint(mol, radius=3, size=2048):
    """Compute a Morgan fingerprint."""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=size)
    arr = np.zeros((size,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def batch_tanimoto(ref_fps, gen_fps, agg='max'):
    """
    Compute batched Tanimoto similarity.

    agg='max' returns the average nearest-reference similarity for generated
    molecules; agg='mean' returns the global mean.
    """
    # Tanimoto = (A·B) / (|A| + |B| - A·B)
    dot = np.dot(gen_fps, ref_fps.T)  # (M, N)
    gen_sum = gen_fps.sum(axis=1, keepdims=True)  # (M, 1)
    ref_sum = ref_fps.sum(axis=1, keepdims=True).T  # (1, N)
    tanimoto = dot / (gen_sum + ref_sum - dot + 1e-10)
    
    if agg == 'max':
        return tanimoto.max(axis=1).mean()
    else:
        return tanimoto.mean()


def _first_existing_column(df, candidates):
    """Return the first column from candidates, case-sensitive then case-insensitive."""
    for name in candidates:
        if name in df.columns:
            return name

    lower_to_original = {str(col).lower(): col for col in df.columns}
    for name in candidates:
        col = lower_to_original.get(name.lower())
        if col is not None:
            return col
    return None


def load_reference_smiles(prior_path):
    """
    Load reference molecules from a CSV prior file.

    Supported schemas:
      - processed prior: cano_smi
      - ChEMBL raw:      cano_smi + HELM
      - CycPeptMPDB raw: SMILES + HELM
      - HELM-only CSV:   HELM / helm
    """
    df = pd.read_csv(prior_path)

    smiles_col = _first_existing_column(
        df,
        ["cano_smi", "canonical_smiles", "SMILES", "smiles", "Smiles"],
    )
    if smiles_col is not None:
        return df[smiles_col].dropna().astype(str).str.strip().tolist(), smiles_col

    helm_col = _first_existing_column(df, ["HELM", "helm"])
    if helm_col is not None:
        helms = df[helm_col].dropna().astype(str).str.strip()
        return [get_cycpep_smi_from_helm(h) for h in helms if h], helm_col

    fallback_col = df.columns[0]
    values = df[fallback_col].dropna().astype(str).str.strip()
    return [get_cycpep_smi_from_helm(h) for h in values if h], fallback_col


class Metrics:
    """Evaluator for generated HELM samples."""
    
    def __init__(self, prior_path, n_jobs=1, input_type='helm'):
        """
        prior_path: training/reference CSV with cano_smi, SMILES, or HELM columns
        """
        train_smiles, ref_col = load_reference_smiles(prior_path)
        print(f"[Metrics] Reference column: {ref_col} ({len(train_smiles)} entries)")
        
        # Build reference fingerprints.
        self.train_smiles = set()
        ref_fps = []
        for smi in train_smiles:
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol:
                self.train_smiles.add(Chem.MolToSmiles(mol))
                ref_fps.append(fingerprint(mol))
        self.ref_fps = np.vstack(ref_fps) if ref_fps else np.zeros((0, 2048))
        self.input_type = input_type
    
    def get_metrics(self, inputs):
        """
        Compute validity, uniqueness, diversity, SNN, novelty, and SA score.
        """
        # Convert to SMILES and parse molecules.
        if self.input_type == 'helm':
            smiles = [get_cycpep_smi_from_helm(h) for h in inputs]
        else:
            smiles = inputs
        
        mols = [Chem.MolFromSmiles(s) if s else None for s in smiles]
        valid_mols = [m for m in mols if m is not None]
        
        # Validity.
        validity = len(valid_mols) / len(mols)
        
        # Canonical SMILES for valid molecules.
        valid_smiles = [Chem.MolToSmiles(m) for m in valid_mols]
        
        # Uniqueness.
        unique_smiles = list(set(valid_smiles))
        uniqueness = len(unique_smiles) / len(valid_smiles) if valid_smiles else 0
        
        # Fingerprints for unique molecules.
        unique_mols = [Chem.MolFromSmiles(s) for s in unique_smiles]
        gen_fps = np.vstack([fingerprint(m) for m in unique_mols]) if unique_mols else np.zeros((0, 2048))
        
        # Diversity: 1 - mean internal similarity.
        if len(gen_fps) > 1:
            diversity = 1 - batch_tanimoto(gen_fps, gen_fps, agg='mean')
        else:
            diversity = 0.0
        
        # SNN: nearest-reference structural similarity.
        if len(gen_fps) > 0 and len(self.ref_fps) > 0:
            snn = batch_tanimoto(self.ref_fps, gen_fps, agg='max')
        else:
            snn = 0.0
        
        # Novelty.
        novel_count = sum(1 for s in unique_smiles if s not in self.train_smiles)
        novelty = novel_count / len(unique_smiles) if unique_smiles else 0
        
        # SA score; lower is better.
        sa_scores = [sascorer.calculateScore(m) for m in unique_mols]
        mean_sa = np.mean(sa_scores) if sa_scores else 0

        print(f"validity\tuniqueness\tdiversity\tsnn\tnovelty\tSA")
        print(f"{validity:.3f}\t{uniqueness:.3f}\t{diversity:.3f}\t{snn:.3f}\t{novelty:.3f}\t{mean_sa:.3f}")
        
        return {
            "validity": validity,
            "uniqueness": uniqueness, 
            "diversity": diversity,
            "snn": snn,
            "novelty": novelty,
            "SA": mean_sa
        }
