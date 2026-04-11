"""
使用Uni-Mol模型将单体库中的CXSMILES结构转换为高维embedding向量
支持批量处理和结果保存

Uni-Mol是一个用于分子表示学习的预训练模型，支持2D/3D分子结构
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


class UniMolEmbeddingGenerator:
    """Uni-Mol模型的embedding生成器"""
    
    def __init__(self, device: str = "auto"):
        """
        初始化Uni-Mol模型
        Args:
            device: 计算设备 ('cpu', 'cuda', 'auto')
        """
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
                logger.error("请先安装unimol_tools: pip install unimol_tools")
                raise ImportError("unimol_tools not installed. Please run: pip install unimol_tools")
            
            # 初始化Uni-Mol模型
            # data_type可以是 'molecule' (2D) 或 'oled' 等
            self.model = UniMolRepr(data_type='molecule', remove_hs=False)
            
            logger.info("Uni-Mol模型加载成功!")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def generate_embedding(self, smiles: str) -> Optional[np.ndarray]:
        """为单个SMILES生成embedding"""
        if self.model is None:
            raise ValueError("模型未加载，请先调用load_model()")
        
        if not smiles or not self.smiles_processor.validate_smiles(smiles):
            logger.warning(f"无效的SMILES: {smiles}")
            return None
        
        try:
            # Uni-Mol接受SMILES列表作为输入
            reprs = self.model.get_repr([smiles])
            
            embedding = None
            
            # Case 1: reprs is a dictionary (e.g. {'cls_repr': ..., 'atomic_reprs': ...})
            if isinstance(reprs, dict):
                if 'cls_repr' in reprs:
                    cls_repr = reprs['cls_repr']
                    if isinstance(cls_repr, list) and len(cls_repr) > 0:
                        embedding = cls_repr[0]
                    elif isinstance(cls_repr, np.ndarray):
                        if len(cls_repr.shape) == 2:
                            embedding = cls_repr[0]
                        else:
                            embedding = cls_repr
                elif 'atomic_reprs' in reprs:
                    atomic_reprs = reprs['atomic_reprs']
                    if isinstance(atomic_reprs, list) and len(atomic_reprs) > 0:
                        embedding = np.mean(atomic_reprs[0], axis=0)
                    elif isinstance(atomic_reprs, np.ndarray):
                        embedding = np.mean(atomic_reprs, axis=0)

            # Case 2: reprs is a list (batch output)
            elif isinstance(reprs, list):
                if len(reprs) > 0:
                    item = reprs[0]
                    # Subcase 2a: List of numpy arrays (embeddings directly)
                    if isinstance(item, np.ndarray):
                        embedding = item
                    # Subcase 2b: List of dictionaries
                    elif isinstance(item, dict):
                        if 'cls_repr' in item:
                            embedding = item['cls_repr']
                        elif 'atomic_reprs' in item:
                            embedding = np.mean(item['atomic_reprs'], axis=0)
            
            if embedding is None:
                logger.warning(f"无法获取embedding: {smiles}")
                return None
            
            return embedding
            
        except Exception as e:
            logger.error(f"生成embedding失败 for SMILES '{smiles}': {e}")
            traceback.print_exc()
            return None
    
    def generate_embeddings_batch(self, smiles_list: List[str]) -> Dict[str, np.ndarray]:
        """批量生成embeddings（更高效）"""
        if self.model is None:
            raise ValueError("模型未加载，请先调用load_model()")
        
        # 过滤无效的SMILES
        valid_indices = []
        valid_smiles = []
        for i, smi in enumerate(smiles_list):
            if smi and self.smiles_processor.validate_smiles(smi):
                valid_indices.append(i)
                valid_smiles.append(smi)
        
        if not valid_smiles:
            return {'embeddings': None, 'valid_indices': []}
        
        try:
            # 批量处理
            reprs = self.model.get_repr(valid_smiles)
            
            if 'cls_repr' in reprs:
                embeddings = reprs['cls_repr']  # (n_samples, hidden_dim)
            else:
                # 使用atomic_reprs的均值
                embeddings = []
                for atomic_repr in reprs['atomic_reprs']:
                    embeddings.append(np.mean(atomic_repr, axis=0))
                embeddings = np.array(embeddings)
            
            return {
                'embeddings': embeddings,
                'valid_indices': valid_indices
            }
            
        except Exception as e:
            logger.error(f"批量生成embedding失败: {e}")
            return {'embeddings': None, 'valid_indices': []}
    
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
            'embeddings': [],
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
            smiles = self.smiles_processor.extract_smiles_from_cxsmiles(cxsmiles)
            
            # 生成embedding
            embedding = self.generate_embedding(smiles)
            
            # 使用isinstance检查，避免numpy array的布尔值歧义
            if isinstance(embedding, np.ndarray) and embedding.size > 0:
                results['symbols'].append(symbol)
                results['smiles'].append(smiles)
                results['cxsmiles'].append(cxsmiles)
                results['embeddings'].append(embedding)
                results['monomer_types'].append(monomer_type)
                
                # 记录embedding维度
                if results['metadata']['embedding_dim'] is None:
                    results['metadata']['embedding_dim'] = len(embedding)
            else:
                results['failed_indices'].append(idx)
                logger.warning(f"处理失败: {symbol} - {cxsmiles}")
        
        # 转换为numpy数组
        if results['embeddings']:
            results['embeddings'] = np.array(results['embeddings'])
        
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
        
        # 保存embedding矩阵为numpy格式
        npy_path = None
        if len(results['embeddings']) > 0:
            npy_path = os.path.join(output_dir, 'embeddings_matrix.npy')
            np.save(npy_path, results['embeddings'])
            logger.info(f"Embedding矩阵已保存到: {npy_path}")
        
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
