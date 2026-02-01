"""
使用Uni-Mol模型将单体库中的CXSMILES结构转换为高维embedding向量
支持批量处理和结果保存

参考: https://github.com/dptech-corp/Uni-Mol
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

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SMILESProcessor:
    """用于清理和标准化CXSMILES"""
    
    def __init__(self):
        self.connection_pattern = re.compile(r'\|\$.*?\$\|')
        self.asterisk_pattern = re.compile(r'\[\*\]')
    
    def extract_smiles_from_cxsmiles(self, cxsmiles: str) -> str:
        """从CXSMILES中提取标准SMILES部分"""
        if pd.isna(cxsmiles) or not isinstance(cxsmiles, str):
            return ""
        
        # 移除CXSMILES的扩展标注部分
        smiles = self.connection_pattern.sub('', cxsmiles).strip()
        
        # 替换连接点占位符为碳原子
        smiles = self.asterisk_pattern.sub('C', smiles)
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
        except:
            pass
        
        return smiles
    
    def validate_smiles(self, smiles: str) -> bool:
        """验证SMILES是否有效"""
        if not smiles or len(smiles) < 1:
            return False
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def extract_r_group_info(self, cxsmiles: str) -> tuple:
        """
        处理带有R基团标记的SMILES字符串
        
        参数:
            cxsmiles: 带有位置标记的SMILES字符串，格式如：
                    'COC(=O)[C@@H](...)[*] |$...;_R2;...;_R1$|'
        
        返回:
            tuple: (cano_smiles, r1_site_idx, r2_site_idx, r3_site_idx)
                其中不存在的R基团返回None
        """
        # 1. 分割SMILES和位置信息
        smi = cxsmiles.split(' |')[0]
        mol_raw = Chem.MolFromSmiles(smi)
        
        if mol_raw is None:
            raise ValueError("无法解析SMILES字符串")
        
        # 2. 获取位置信息
        pos_info = cxsmiles.split('$')[1]
        pos_list = pos_info.split(';')
        
        # 3. 查找R1, R2, R3的原子索引及其连接位点
        r1_atomidx = pos_list.index('_R1') if '_R1' in pos_list else None
        r2_atomidx = pos_list.index('_R2') if '_R2' in pos_list else None
        r3_atomidx = pos_list.index('_R3') if '_R3' in pos_list else None
        
        # 4. 获取R基团连接位点的原子索引，并设置属性标记
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
        
        # 5. 创建可编辑的分子并删除所有原子序数为0的原子（即[*]）
        mol_edit = Chem.EditableMol(mol_raw)
        atoms_to_remove = []
        for atom in mol_raw.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atoms_to_remove.append(atom.GetIdx())
        
        # 从后往前删除，避免索引变化
        for idx in sorted(atoms_to_remove, reverse=True):
            mol_edit.RemoveAtom(idx)
        
        mol_raw = mol_edit.GetMol()
        
        # 6. 生成规范SMILES
        cano_smiles = Chem.MolToSmiles(mol_raw)
        cano_mol = Chem.MolFromSmiles(cano_smiles)
        
        # 7. 使用子结构匹配获取原子映射关系
        match = cano_mol.GetSubstructMatch(mol_raw)
        
        if not match:
            # 尝试反向匹配
            match = mol_raw.GetSubstructMatch(cano_mol)
            if match:
                # 创建反向映射
                reverse_map = {v: k for k, v in enumerate(match)}
            else:
                raise ValueError("无法建立原子映射关系")
        else:
            # match[i] 表示 mol_raw 的原子 i 对应 cano_mol 的原子 match[i]
            reverse_map = {k: v for k, v in enumerate(match)}
        
        # 8. 找到规范分子中对应的R基团连接位点
        cano_r1_site_idx = None
        cano_r2_site_idx = None
        cano_r3_site_idx = None
        
        # 通过属性标记找到对应的原子
        for atom in mol_raw.GetAtoms():
            if atom.HasProp('R_site'):
                r_type = atom.GetProp('R_site')
                old_idx = atom.GetIdx()
                # 在映射中找到对应的新索引
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
    """Uni-Mol模型的embedding生成器"""
    
    def __init__(self, device: str = "auto"):
        self.device = self._get_device(device)
        self.model = None
        self.smiles_processor = SMILESProcessor()
        
        logger.info(f"初始化Uni-Mol模型")
        logger.info(f"使用设备: {self.device}")
    
    def _get_device(self, device: str) -> str:
        """确定计算设备"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def load_model(self):
        """加载Uni-Mol模型"""
        try:
            logger.info("正在加载Uni-Mol模型...")
            
            # 尝试导入unimol_tools
            try:
                from unimol_tools import UniMolRepr
            except ImportError:
                logger.error("未安装unimol_tools: pip install unimol_tools")
                raise ImportError("unimol_tools not installed. Please run: pip install unimol_tools")
            
            # 初始化Uni-Mol模型
            # data_type可以是 'molecule' (2D) 或 'oled' 等
            self.model = UniMolRepr(data_type='molecule', remove_hs=False)
            
            logger.info("Uni-Mol模型加载成功!")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def generate_embedding(self, smiles: str) -> Optional[Dict]:
        """
        为单个SMILES生成embedding，返回CLS向量和原子级特征
        
        Returns:
            dict: {'cls_repr': (512,), 'atomic_reprs': (N_atoms, 512)} 或 None
        """
        if self.model is None:
            raise ValueError("模型未加载，请先调用load_model()")
        
        if not smiles or not self.smiles_processor.validate_smiles(smiles):
            logger.warning(f"无效的SMILES: {smiles}")
            return None
        
        try:
            reprs = self.model.get_repr([smiles])
            
            cls_repr = None
            atomic_reprs = None
            
            if isinstance(reprs, dict):
                if 'cls_repr' in reprs:
                    cls_data = reprs['cls_repr']
                    if isinstance(cls_data, list) and len(cls_data) > 0:
                        cls_repr = np.array(cls_data[0])
                    elif isinstance(cls_data, np.ndarray):
                        cls_repr = cls_data[0] if len(cls_data.shape) == 2 else cls_data
                
                if 'atomic_reprs' in reprs:
                    atomic_data = reprs['atomic_reprs']
                    if isinstance(atomic_data, list) and len(atomic_data) > 0:
                        atomic_reprs = np.array(atomic_data[0])
                    elif isinstance(atomic_data, np.ndarray):
                        atomic_reprs = atomic_data[0] if len(atomic_data.shape) == 3 else atomic_data
            
            # 如果没有 cls_repr，用 atomic_reprs 的均值代替
            if cls_repr is None and atomic_reprs is not None:
                cls_repr = np.mean(atomic_reprs, axis=0)
            
            if cls_repr is None:
                logger.warning(f"无法获取embedding: {smiles}")
                return None
            
            return {'cls_repr': cls_repr, 'atomic_reprs': atomic_reprs}
            
        except Exception as e:
            logger.error(f"生成embedding失败 for SMILES '{smiles}': {e}")
            traceback.print_exc()
            return None
    
    def process_monomer_library(self, csv_path: str, batch_size: int = 32) -> Dict:
        """处理整个单体库文件"""
        logger.info(f"开始处理单体库文件: {csv_path}")
        
        # 读取数据
        df = pd.read_csv(csv_path)
        logger.info(f"读取到 {len(df)} 个单体")
        
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
        
        # 加载模型
        if self.model is None:
            self.load_model()
        
        # 逐个处理（也可以改为批处理）
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="生成embeddings"):
            symbol = row['Symbol']
            cxsmiles = row['CXSMILES']
            monomer_type = row.get('Monomer_Type', '')
            
            # 提取SMILES
            smiles, r1_site_idx, r2_site_idx, r3_site_idx = self.smiles_processor.extract_r_group_info(cxsmiles)
            
            # 生成embedding（包含cls和atomic_reprs）
            emb_result = self.generate_embedding(smiles)
            
            if emb_result is not None:
                cls_repr = emb_result['cls_repr']
                atomic_reprs = emb_result['atomic_reprs']
                hidden_dim = len(cls_repr)
                
                # 提取 R1, R2, R3 原子的特征向量，不存在则填零
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
                
                # 组装 full_embedding: [CLS, R1, R2, R3] -> (4, hidden_dim)
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
                logger.warning(f"处理失败: {symbol} - {cxsmiles}")
        
        # 转换为numpy数组
        if results['full_embeddings']:
            results['full_embeddings'] = np.array(results['full_embeddings'])  # (N, 4, 512)
        
        # 更新统计信息
        results['metadata']['successful_count'] = len(results['symbols'])
        results['metadata']['failed_count'] = len(results['failed_indices'])
        
        logger.info(f"处理完成! 成功: {results['metadata']['successful_count']}, "
                   f"失败: {results['metadata']['failed_count']}")
        
        return results
    
    def save_embeddings(self, results: Dict, output_dir: str):
        """保存embedding结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为pickle格式（包含所有数据）
        pickle_path = os.path.join(output_dir, 'monomer_embeddings.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"完整数据已保存到: {pickle_path}")
        
        # 保存full_embeddings矩阵为numpy格式 (N, 4, 512)
        npy_path = None
        if len(results['full_embeddings']) > 0:
            npy_path = os.path.join(output_dir, 'full_embeddings.npy')
            np.save(npy_path, results['full_embeddings'])
            logger.info(f"Full Embedding矩阵 (N, 4, {results['metadata']['embedding_dim']}) 已保存到: {npy_path}")
        
        # 保存映射信息为CSV
        mapping_df = pd.DataFrame({
            'symbol': results['symbols'],
            'smiles': results['smiles'],
            'cxsmiles': results['cxsmiles'],
            'monomer_type': results['monomer_types']
        })
        csv_path = os.path.join(output_dir, 'monomer_mapping.csv')
        mapping_df.to_csv(csv_path, index=False)
        logger.info(f"单体映射已保存到: {csv_path}")
        
        # 保存元数据
        metadata_path = os.path.join(output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(results['metadata'], f, indent=2)
        logger.info(f"元数据已保存到: {metadata_path}")
        
        return {
            'pickle_path': pickle_path,
            'numpy_path': npy_path,
            'csv_path': csv_path,
            'metadata_path': metadata_path
        }


def main():
    """主函数"""
    # 输入文件路径
    csv_path = "./data/monomer_library.csv"
    # 输出目录（与molformer_embeddings区分）
    output_dir = "./unimol_embeddings"
    
    # 检查输入文件
    if not os.path.exists(csv_path):
        # 尝试其他可能的路径
        alt_path = "./monomer_library.csv"
        if os.path.exists(alt_path):
            csv_path = alt_path
        else:
            logger.error(f"输入文件不存在: {csv_path}")
            return
    
    # 创建embedding生成器
    generator = UniMolEmbeddingGenerator()
    
    try:
        # 处理单体库
        results = generator.process_monomer_library(csv_path)
        
        # 保存结果
        output_paths = generator.save_embeddings(results, output_dir)
        
        # 打印总结
        print("\n" + "="*50)
        print("Uni-Mol Embedding生成完成")
        print("="*50)
        print(f"总单体数: {results['metadata']['total_monomers']}")
        print(f"成功处理: {results['metadata']['successful_count']}")
        print(f"失败数量: {results['metadata']['failed_count']}")
        print(f"Embedding维度: {results['metadata']['embedding_dim']}")
        print(f"输出目录: {output_dir}")
        print("\n输出文件:")
        for key, path in output_paths.items():
            if path:
                print(f"  {key}: {path}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"处理过程中出现错误: {e}")
        raise


if __name__ == "__main__":
    main()
