"""
使用MolFormer模型将单体库中的CXSMILES结构转换为高维embedding向量
支持批量处理和结果保存
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pickle
import json
import os
import re
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from rdkit import Chem

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
        
        # 替换连接点占位符为氢原子
        smiles = self.asterisk_pattern.sub('C', smiles)
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        
        return smiles
    
    def validate_smiles(self, smiles: str) -> bool:
        """简单的SMILES格式验证"""
        if not smiles or len(smiles) < 2:
            return False
        
        # 检查基本的化学元素
        valid_chars = set('CHNOPSFBrClI()[]{}=+-#@/\\0123456789cnos')
        return all(c in valid_chars for c in smiles)

class MolFormerEmbeddingGenerator:
    """MolFormer模型的embedding生成器"""
    
    def __init__(self, model_name: str = "ibm/MoLFormer-XL-both-10pct", device: str = "auto"):
        """
        初始化MolFormer模型
        Args:
            model_name: HuggingFace模型名称
            device: 计算设备 ('cpu', 'cuda', 'auto')
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.tokenizer = None
        self.model = None
        self.smiles_processor = SMILESProcessor()
        
        logger.info(f"初始化MolFormer模型: {model_name}")
        logger.info(f"使用设备: {self.device}")
    
    def _get_device(self, device: str) -> str:
        """确定计算设备"""
        if device == "auto":
            # 暂时使用CPU避免MPS兼容性问题
            return "cpu"
        return device
    
    def load_model(self):
        """加载MolFormer模型和分词器"""
        try:
            logger.info("正在加载MolFormer模型...")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(self.model_name, deterministic_eval=True, trust_remote_code=True)
            self.model.to(self.device)
            self.model.eval()
            logger.info("模型加载成功!")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def generate_embedding(self, smiles: str) -> Optional[np.ndarray]:
        """为单个SMILES生成embedding"""
        if not self.model or not self.tokenizer:
            raise ValueError("模型未加载，得先调用load_model()")
        
        # if not smiles or not self.smiles_processor.validate_smiles(smiles):
        #     logger.warning(f"无效的SMILES: {smiles}")
        #     return None
        
        #try:
            # 分词和编码
        inputs = self.tokenizer([smiles], padding=True, return_tensors="pt")
        
        # 生成embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用[CLS] token的embedding或者平均池化
            embedding = outputs.last_hidden_state[0].sum(0).cpu().numpy()
        
        return embedding#.flatten()
        
        # except Exception as e:
        #     logger.error(f"生成embedding失败 for SMILES '{smiles}': {e}")
        #     return None
    
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
                'model_name': self.model_name,
                'total_monomers': len(df),
                'processed_date': pd.Timestamp.now().isoformat(),
                'embedding_dim': None
            }
        }
        
        # 加载模型
        if not self.model:
            self.load_model()
        
        # 批量处理
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="生成embeddings"):
            symbol = row['Symbol']
            cxsmiles = row['CXSMILES']
            monomer_type = row.get('Monomer_Type', '')
            
            # 提取SMILES
            smiles = self.smiles_processor.extract_smiles_from_cxsmiles(cxsmiles)
            
            # 生成embedding
            embedding = self.generate_embedding(smiles)
            
            if embedding is not None:
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
            'numpy_path': npy_path if len(results['embeddings']) > 0 else None,
            'csv_path': csv_path,
            'metadata_path': metadata_path
        }

def main():
    """主函数"""
    csv_path = "./monomer_library.csv"
    output_dir = "./molformer_embeddings"
    
    # 检查输入文件
    if not os.path.exists(csv_path):
        logger.error(f"输入文件不存在: {csv_path}")
        return
    
    # 创建embedding生成器
    generator = MolFormerEmbeddingGenerator()
    
    try:
        # 处理单体库
        results = generator.process_monomer_library(csv_path)
        
        # 保存结果
        output_paths = generator.save_embeddings(results, output_dir)
        
        # 打印总结
        print("\n" + "="*50)
        print("Embedding生成完成")
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
