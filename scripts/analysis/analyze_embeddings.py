"""分析 CLS 向量和原子向量的数值特性"""
import numpy as np
from unimol_tools import UniMolRepr

# 加载模型
print("正在加载 Uni-Mol 模型...")
model = UniMolRepr(data_type='molecule', remove_hs=False)

# 测试几个不同大小的分子
test_smiles = [
    'CC(N)C',                    # 小分子 (丙氨酸骨架)
    'CC(C)CC(N)C(=O)O',          # 中等 (亮氨酸)
    'c1ccc(CC(N)C(=O)O)cc1',     # 含苯环 (苯丙氨酸)
]

print('='*70)
print('CLS向量 vs 原子向量 数值特性分析')
print('='*70)

all_cls_norms = []
all_atom_norms = []

for smi in test_smiles:
    reprs = model.get_repr([smi])
    cls_repr = np.array(reprs['cls_repr'][0])
    atomic_reprs = np.array(reprs['atomic_reprs'][0])
    
    print(f'\nSMILES: {smi}')
    print(f'原子数: {atomic_reprs.shape[0]}')
    print('-'*50)
    
    # CLS 统计
    cls_norm = np.linalg.norm(cls_repr)
    all_cls_norms.append(cls_norm)
    print(f'CLS向量:')
    print(f'  L2范数: {cls_norm:.4f}')
    print(f'  均值: {cls_repr.mean():.6f}')
    print(f'  标准差: {cls_repr.std():.4f}')
    print(f'  最大值: {cls_repr.max():.4f}')
    print(f'  最小值: {cls_repr.min():.4f}')
    
    # 单个原子统计
    atom_norms = [np.linalg.norm(atomic_reprs[i]) for i in range(len(atomic_reprs))]
    all_atom_norms.extend(atom_norms)
    atom_means = [atomic_reprs[i].mean() for i in range(len(atomic_reprs))]
    atom_stds = [atomic_reprs[i].std() for i in range(len(atomic_reprs))]
    
    print(f'原子向量 (各原子统计):')
    print(f'  L2范数: 平均={np.mean(atom_norms):.4f}, 范围=[{min(atom_norms):.4f}, {max(atom_norms):.4f}]')
    print(f'  均值: {np.mean(atom_means):.6f}')
    print(f'  标准差: {np.mean(atom_stds):.4f}')
    
    # 比较
    ratio = cls_norm / np.mean(atom_norms)
    print(f'比值 (CLS范数 / 平均原子范数): {ratio:.2f}')

print('\n' + '='*70)
print('总结')
print('='*70)
print(f'CLS 向量 L2 范数: 平均={np.mean(all_cls_norms):.4f}')
print(f'原子向量 L2 范数: 平均={np.mean(all_atom_norms):.4f}')
print(f'整体比值: {np.mean(all_cls_norms) / np.mean(all_atom_norms):.2f}')
print()
print('如果直接相加 CLS + R1 + R2 + R3:')
print(f'  假设 3 个 R 位点都有值，融合后范数 ≈ CLS + 3*原子')
print(f'  CLS 占比约: 1 / (1 + 3*{np.mean(all_atom_norms)/np.mean(all_cls_norms):.2f}) = {1/(1+3*np.mean(all_atom_norms)/np.mean(all_cls_norms)):.1%}')
