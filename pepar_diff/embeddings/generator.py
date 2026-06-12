"""
Generate Uni-Mol embeddings for monomer-library CXSMILES entries.

Reference: https://github.com/dptech-corp/Uni-Mol
"""

import pandas as pd
import numpy as np
import torch
import pickle
import json
import os
import re
import traceback
from typing import List, Dict, Optional
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from rdkit import Chem
from rdkit.Chem import AllChem

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SMILESProcessor:
    """Clean and standardize CXSMILES strings."""
    
    def __init__(self):
        self.connection_pattern = re.compile(r'\|\$.*?\$\|')
        self.asterisk_pattern = re.compile(r'\[\*\]')
    
    def extract_smiles_from_cxsmiles(self, cxsmiles: str) -> str:
        """Extract the standard SMILES part from a CXSMILES string."""
        if pd.isna(cxsmiles) or not isinstance(cxsmiles, str):
            return ""
        
        # Drop CXSMILES extension annotations.
        smiles = self.connection_pattern.sub('', cxsmiles).strip()
        
        # Replace connection placeholders with carbon atoms.
        smiles = self.asterisk_pattern.sub('C', smiles)
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
        except:
            pass
        
        return smiles
    
    def validate_smiles(self, smiles: str) -> bool:
        """Return whether a SMILES string is valid."""
        if not smiles or len(smiles) < 1:
            return False
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def extract_r_group_info(self, cxsmiles: str) -> tuple:
        """Return canonical SMILES and mapped R1/R2/R3 attachment-site indices."""
        # Split SMILES and position annotations.
        smi = cxsmiles.split(' |')[0]
        mol_raw = Chem.MolFromSmiles(smi)
        
        if mol_raw is None:
            raise ValueError("Failed to parse SMILES string")
        
        # Read CXSMILES position annotations.
        pos_info = cxsmiles.split('$')[1]
        pos_list = pos_info.split(';')
        
        # Locate R1/R2/R3 atoms and their attachment sites.
        r1_atomidx = pos_list.index('_R1') if '_R1' in pos_list else None
        r2_atomidx = pos_list.index('_R2') if '_R2' in pos_list else None
        r3_atomidx = pos_list.index('_R3') if '_R3' in pos_list else None
        
        r1_site_idx = None
        r2_site_idx = None
        r3_site_idx = None
        
        if r1_atomidx is not None:
            atom = mol_raw.GetAtomWithIdx(r1_atomidx)
            neighbors = atom.GetNeighbors()
            if neighbors:
                r1_site_idx = neighbors[0].GetIdx()
                mol_raw.GetAtomWithIdx(r1_site_idx).SetProp('R_site', 'R1')
        
        if r2_atomidx is not None:
            atom = mol_raw.GetAtomWithIdx(r2_atomidx)
            neighbors = atom.GetNeighbors()
            if neighbors:
                r2_site_idx = neighbors[0].GetIdx()
                mol_raw.GetAtomWithIdx(r2_site_idx).SetProp('R_site', 'R2')
        
        if r3_atomidx is not None:
            atom = mol_raw.GetAtomWithIdx(r3_atomidx)
            neighbors = atom.GetNeighbors()
            if neighbors:
                r3_site_idx = neighbors[0].GetIdx()
                mol_raw.GetAtomWithIdx(r3_site_idx).SetProp('R_site', 'R3')
        
        # Remove dummy atoms ([*]).
        mol_edit = Chem.EditableMol(mol_raw)
        atoms_to_remove = []
        for atom in mol_raw.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atoms_to_remove.append(atom.GetIdx())
        
        # Delete in reverse order to keep indices stable.
        for idx in sorted(atoms_to_remove, reverse=True):
            mol_edit.RemoveAtom(idx)
        
        mol_raw = mol_edit.GetMol()
        
        # Canonicalize and map old atom indices to canonical indices.
        cano_smiles = Chem.MolToSmiles(mol_raw)
        cano_mol = Chem.MolFromSmiles(cano_smiles)
        
        match = cano_mol.GetSubstructMatch(mol_raw)
        
        if not match:
            match = mol_raw.GetSubstructMatch(cano_mol)
            if match:
                reverse_map = {v: k for k, v in enumerate(match)}
            else:
                raise ValueError("Failed to map atom indices")
        else:
            reverse_map = {k: v for k, v in enumerate(match)}
        
        cano_r1_site_idx = None
        cano_r2_site_idx = None
        cano_r3_site_idx = None
        
        for atom in mol_raw.GetAtoms():
            if atom.HasProp('R_site'):
                r_type = atom.GetProp('R_site')
                old_idx = atom.GetIdx()
                if old_idx in reverse_map:
                    new_idx = reverse_map[old_idx]
                    if r_type == 'R1':
                        cano_r1_site_idx = new_idx
                    elif r_type == 'R2':
                        cano_r2_site_idx = new_idx
                    elif r_type == 'R3':
                        cano_r3_site_idx = new_idx
        
        return cano_smiles, cano_r1_site_idx, cano_r2_site_idx, cano_r3_site_idx


class UniMolEmbeddingGenerator:
    """Generate Uni-Mol embeddings."""
    
    def __init__(self, device: str = "auto"):
        self.device = self._get_device(device)
        self.model = None
        self.smiles_processor = SMILESProcessor()
        
        logger.info("Initializing Uni-Mol model")
        logger.info(f"Device: {self.device}")
    
    def _get_device(self, device: str) -> str:
        """Resolve the compute device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def load_model(self):
        """Load the Uni-Mol model."""
        try:
            logger.info("Loading Uni-Mol model...")
            
            try:
                from unimol_tools import UniMolRepr
            except ImportError:
                logger.error("unimol_tools is not installed: pip install unimol_tools")
                raise ImportError("unimol_tools not installed. Please run: pip install unimol_tools")
            
            # data_type can be 'molecule' (2D), 'oled', etc.
            self.model = UniMolRepr(data_type='molecule', remove_hs=False)
            
            logger.info("Uni-Mol model loaded")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_embedding(self, smiles: str) -> Optional[Dict]:
        """Generate CLS and atom-level embeddings for one SMILES string."""
        if self.model is None:
            raise ValueError("Model is not loaded. Call load_model() first.")
        
        if not smiles or not self.smiles_processor.validate_smiles(smiles):
            logger.warning(f"Invalid SMILES: {smiles}")
            return None
        
        try:
            reprs = self.model.get_repr([smiles])
            
            cls_repr = None
            atomic_reprs = None
            
            if isinstance(reprs, dict):
                # Dict output: {'cls_repr': ..., 'atomic_reprs': ...}
                if 'cls_repr' in reprs:
                    cls_data = reprs['cls_repr']
                    if isinstance(cls_data, list) and len(cls_data) > 0:
                        cls_repr = np.array(cls_data[0])
                    elif isinstance(cls_data, np.ndarray) and cls_data.size > 0:
                        cls_repr = cls_data[0] if len(cls_data.shape) == 2 else cls_data
                
                if 'atomic_reprs' in reprs:
                    atomic_data = reprs['atomic_reprs']
                    if isinstance(atomic_data, list) and len(atomic_data) > 0:
                        atomic_reprs = np.array(atomic_data[0])
                    elif isinstance(atomic_data, np.ndarray) and atomic_data.size > 0:
                        atomic_reprs = atomic_data[0] if len(atomic_data.shape) == 3 else atomic_data
                        
            elif isinstance(reprs, list) and len(reprs) > 0:
                # List output may only include molecule-level embeddings.
                item = reprs[0]
                if isinstance(item, np.ndarray) and item.size > 0:
                    cls_repr = item
                    atomic_reprs = None
            
            # Fallback to the atom-embedding mean when CLS is absent.
            if cls_repr is None and atomic_reprs is not None and atomic_reprs.size > 0:
                cls_repr = np.mean(atomic_reprs, axis=0)
                logger.debug(f"Using mean atomic embedding as CLS: {smiles}")
            
            if cls_repr is None:
                logger.warning(f"Failed to get embedding: {smiles}")
                return None
            
            return {'cls_repr': cls_repr, 'atomic_reprs': atomic_reprs}
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for SMILES '{smiles}': {e}")
            traceback.print_exc()
            return None
    
    def process_monomer_library(self, csv_path: str, batch_size: int = 32) -> Dict:
        """Process the monomer-library CSV file."""
        logger.info(f"Processing monomer library: {csv_path}")
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} monomers")
        
        results = {
            'symbols': [],
            'smiles': [],
            'cxsmiles': [],
            'full_embeddings': [],  # (N, 4, 512): [CLS, R1, R2, R3]
            'monomer_types': [],
            'failed_indices': [],
            'metadata': {
                'model_name': 'Uni-Mol',
                'total_monomers': len(df),
                'processed_date': pd.Timestamp.now().isoformat(),
                'embedding_dim': None
            }
        }
        
        if self.model is None:
            self.load_model()
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating embeddings"):
            symbol = row['Symbol']
            cxsmiles = row['CXSMILES']
            monomer_type = row.get('Monomer_Type', '')
            
            smiles, r1_site_idx, r2_site_idx, r3_site_idx = self.smiles_processor.extract_r_group_info(cxsmiles)
            
            emb_result = self.generate_embedding(smiles)
            
            if emb_result is not None:
                cls_repr = emb_result['cls_repr']
                atomic_reprs = emb_result['atomic_reprs']
                hidden_dim = len(cls_repr)
                
                # Use zero vectors for missing R-site embeddings.
                r1_vec = np.zeros(hidden_dim)
                r2_vec = np.zeros(hidden_dim)
                r3_vec = np.zeros(hidden_dim)
                
                if atomic_reprs is not None:
                    n_atoms = atomic_reprs.shape[0]
                    if r1_site_idx is not None and r1_site_idx < n_atoms:
                        r1_vec = atomic_reprs[r1_site_idx]
                    if r2_site_idx is not None and r2_site_idx < n_atoms:
                        r2_vec = atomic_reprs[r2_site_idx]
                    if r3_site_idx is not None and r3_site_idx < n_atoms:
                        r3_vec = atomic_reprs[r3_site_idx]
                
                # [CLS, R1, R2, R3] -> (4, hidden_dim)
                full_emb = np.stack([cls_repr, r1_vec, r2_vec, r3_vec], axis=0)
                
                results['symbols'].append(symbol)
                results['smiles'].append(smiles)
                results['cxsmiles'].append(cxsmiles)
                results['full_embeddings'].append(full_emb)
                results['monomer_types'].append(monomer_type)
                
                if results['metadata']['embedding_dim'] is None:
                    results['metadata']['embedding_dim'] = hidden_dim
            else:
                results['failed_indices'].append(idx)
                logger.warning(f"Failed to process: {symbol} - {cxsmiles}")
        
        if results['full_embeddings']:
            results['full_embeddings'] = np.array(results['full_embeddings'])  # (N, 4, 512)
        
        results['metadata']['successful_count'] = len(results['symbols'])
        results['metadata']['failed_count'] = len(results['failed_indices'])
        
        logger.info(f"Done. Success: {results['metadata']['successful_count']}, "
                   f"failed: {results['metadata']['failed_count']}")
        
        return results
    
    def save_embeddings(self, results: Dict, output_dir: str):
        """Save embedding outputs."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Full result bundle.
        pickle_path = os.path.join(output_dir, 'monomer_embeddings.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"Saved full data to: {pickle_path}")
        
        # Full embedding matrix: (N, 4, 512).
        npy_path = None
        cls_npy_path = None
        if len(results['full_embeddings']) > 0:
            npy_path = os.path.join(output_dir, 'full_embeddings.npy')
            np.save(npy_path, results['full_embeddings'])
            logger.info(f"Saved full embedding matrix (N, 4, {results['metadata']['embedding_dim']}) to: {npy_path}")
            
            # CLS-only matrix for TokenMapper.
            cls_npy_path = os.path.join(output_dir, 'embeddings_matrix.npy')
            np.save(cls_npy_path, results['full_embeddings'][:, 0, :])
            logger.info(f"Saved CLS embedding matrix (N, {results['metadata']['embedding_dim']}) to: {cls_npy_path}")
        
        # Monomer mapping table.
        mapping_df = pd.DataFrame({
            'symbol': results['symbols'],
            'smiles': results['smiles'],
            'cxsmiles': results['cxsmiles'],
            'monomer_type': results['monomer_types']
        })
        csv_path = os.path.join(output_dir, 'monomer_mapping.csv')
        mapping_df.to_csv(csv_path, index=False)
        logger.info(f"Saved monomer mapping to: {csv_path}")
        
        metadata_path = os.path.join(output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(results['metadata'], f, indent=2)
        logger.info(f"Saved metadata to: {metadata_path}")
        
        return {
            'pickle_path': pickle_path,
            'numpy_path': npy_path,
            'csv_path': csv_path,
            'metadata_path': metadata_path
        }


def main():
    """CLI entry point."""
    csv_path = "./data/processed/monomer_library.csv"
    output_dir = "./data/processed/unimol_embeddings"
    
    if not os.path.exists(csv_path):
        alt_path = "./monomer_library.csv"
        if os.path.exists(alt_path):
            csv_path = alt_path
        else:
            logger.error(f"Input file does not exist: {csv_path}")
            return
    
    generator = UniMolEmbeddingGenerator()
    
    try:
        results = generator.process_monomer_library(csv_path)
        
        output_paths = generator.save_embeddings(results, output_dir)
        
        print("\n" + "="*50)
        print("Uni-Mol embedding generation complete")
        print("="*50)
        print(f"Total monomers: {results['metadata']['total_monomers']}")
        print(f"Successful: {results['metadata']['successful_count']}")
        print(f"Failed: {results['metadata']['failed_count']}")
        print(f"Embedding dim: {results['metadata']['embedding_dim']}")
        print(f"Output dir: {output_dir}")
        print("\nOutput files:")
        for key, path in output_paths.items():
            if path:
                print(f"  {key}: {path}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
