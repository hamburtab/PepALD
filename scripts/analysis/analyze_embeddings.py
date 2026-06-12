"""Inspect numeric properties of CLS and atom embeddings."""
import numpy as np
from unimol_tools import UniMolRepr

print("Loading Uni-Mol model...")
model = UniMolRepr(data_type='molecule', remove_hs=False)

test_smiles = [
    'CC(N)C',                    # Small: alanine-like backbone
    'CC(C)CC(N)C(=O)O',          # Medium: leucine
    'c1ccc(CC(N)C(=O)O)cc1',     # Aromatic: phenylalanine
]

print('='*70)
print('CLS vs atom embedding statistics')
print('='*70)

all_cls_norms = []
all_atom_norms = []

for smi in test_smiles:
    reprs = model.get_repr([smi])
    cls_repr = np.array(reprs['cls_repr'][0])
    atomic_reprs = np.array(reprs['atomic_reprs'][0])
    
    print(f'\nSMILES: {smi}')
    print(f'Atoms: {atomic_reprs.shape[0]}')
    print('-'*50)
    
    cls_norm = np.linalg.norm(cls_repr)
    all_cls_norms.append(cls_norm)
    print('CLS embedding:')
    print(f'  L2 norm: {cls_norm:.4f}')
    print(f'  Mean: {cls_repr.mean():.6f}')
    print(f'  Std: {cls_repr.std():.4f}')
    print(f'  Max: {cls_repr.max():.4f}')
    print(f'  Min: {cls_repr.min():.4f}')
    
    atom_norms = [np.linalg.norm(atomic_reprs[i]) for i in range(len(atomic_reprs))]
    all_atom_norms.extend(atom_norms)
    atom_means = [atomic_reprs[i].mean() for i in range(len(atomic_reprs))]
    atom_stds = [atomic_reprs[i].std() for i in range(len(atomic_reprs))]
    
    print('Atom embeddings:')
    print(f'  L2 norm: mean={np.mean(atom_norms):.4f}, range=[{min(atom_norms):.4f}, {max(atom_norms):.4f}]')
    print(f'  Mean: {np.mean(atom_means):.6f}')
    print(f'  Std: {np.mean(atom_stds):.4f}')
    
    ratio = cls_norm / np.mean(atom_norms)
    print(f'Ratio (CLS norm / mean atom norm): {ratio:.2f}')

print('\n' + '='*70)
print('Summary')
print('='*70)
print(f'Mean CLS L2 norm: {np.mean(all_cls_norms):.4f}')
print(f'Mean atom L2 norm: {np.mean(all_atom_norms):.4f}')
print(f'Overall ratio: {np.mean(all_cls_norms) / np.mean(all_atom_norms):.2f}')
print()
print('If summing CLS + R1 + R2 + R3 directly:')
print('  With all 3 R sites present, fused norm is roughly CLS + 3*atom.')
print(f'  Approx CLS share: 1 / (1 + 3*{np.mean(all_atom_norms)/np.mean(all_cls_norms):.2f}) = {1/(1+3*np.mean(all_atom_norms)/np.mean(all_cls_norms)):.1%}')
