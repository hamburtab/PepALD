import pandas as pd
import numpy as np
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from collections import Counter
import argparse

class ChEMBL32DataProcessor:
    def __init__(self, input_file: str = None, output_dir: str = "./data"):
        self.input_file = input_file or "./data/chembl32/biotherapeutics_dict_prot_flt.csv"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.valid_sequences = []
        self.invalid_sequences = []
        self.stats = {
            'total_rows': 0,
            'valid_helm': 0,
            'invalid_helm': 0,
            'too_short': 0,
            'too_long': 0,
            'length_distribution': Counter(),
            'monomer_usage': Counter(),
        }
        self.ring_type_set = ['R3R3', 'R1R2', 'R1R3', 'R3R2']
    
    def load_data(self) -> pd.DataFrame:
        print(f" 正在加载数据: {self.input_file}")
        
        if not Path(self.input_file).exists():
            raise FileNotFoundError(f"数据文件不存在: {self.input_file}")
        
        df = pd.read_csv(self.input_file, low_memory=False)
        print(f" 成功读取 {len(df)} 行数据")
        print(f" 列名: {list(df.columns)}")
        
        self.stats['total_rows'] = len(df)
        return df
    
    def validate_helm_sequence(self, helm_seq: str, min_len: int = 3, max_len: int = 128) -> bool:
        if not isinstance(helm_seq, str) or not helm_seq.strip():
            return False
        
        helm_seq = helm_seq.strip()
        
        if not (helm_seq.startswith('PEPTIDE1{') and helm_seq.endswith('}$$$$')):
            return False
        
        try:
            content = helm_seq[len('PEPTIDE1{'):-len('}$$$$')]
            if not content:
                return False
            
            # 分割单体
            monomers = content.split('.')
            
            # 检查长度
            if len(monomers) < min_len:
                self.stats['too_short'] += 1
                return False
            
            if len(monomers) > max_len:
                self.stats['too_long'] += 1
                return False
            
            # 检查单体格式（基本验证）
            for monomer in monomers:
                if not monomer or len(monomer.strip()) == 0:
                    return False
                # 检查是否包含无效字符
                if any(char in monomer for char in ['|', '$']):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def extract_monomers(self, helm_seq: str) -> List[str]:
        """从HELM序列中提取单体列表"""
        try:
            content = helm_seq[len('PEPTIDE1{'):-len('}$$$$')]
            return content.split('.')
        except:
            return []
    
    def process_data(self, min_seq_len: int = 3, max_seq_len: int = 128) -> str:
        """处理ChEMBL32数据"""
        print(" 开始处理ChEMBL32数据...")
        
        # 1. 加载数据
        df = self.load_data()
        
        # 2. 提取HELM列
        if 'HELM' not in df.columns:
            raise ValueError("数据中未找到HELM列")
        
        print(" 正在处理HELM序列...")
        
        # 3. 处理每个HELM序列
        processed_count = 0
        for idx, row in df.iterrows():
            helm_seq = row['HELM']
            
            # 验证HELM序列
            if self.validate_helm_sequence(helm_seq, min_seq_len, max_seq_len):
                self.valid_sequences.append(helm_seq)
                
                # 统计信息
                monomers = self.extract_monomers(helm_seq)
                self.stats['length_distribution'][len(monomers)] += 1
                for monomer in monomers:
                    self.stats['monomer_usage'][monomer] += 1
                
                self.stats['valid_helm'] += 1
            else:
                self.invalid_sequences.append(str(helm_seq))
                self.stats['invalid_helm'] += 1
            
            processed_count += 1
            if processed_count % 1000 == 0:
                print(f"处理进度: {processed_count}/{len(df)} ({processed_count/len(df)*100:.1f}%)")
        
        # 4. 计算统计信息
        if self.valid_sequences:
            total_length = sum(self.stats['length_distribution'][k] * k 
                             for k in self.stats['length_distribution'])
            self.stats['avg_length'] = total_length / len(self.valid_sequences)
        
        # 5. 保存处理结果
        output_file = self.output_dir / "helm_sequences_chembl32.txt"
        print(f" 保存HELM序列到: {output_file}")
        
        with open(output_file, 'w') as f:
            for helm_seq in self.valid_sequences:
                f.write(helm_seq + '\n')
        
        # 6. 保存统计信息
        stats_file = self.output_dir / "chembl32_processing_stats.json"
        stats_to_save = dict(self.stats)
        stats_to_save['length_distribution'] = dict(self.stats['length_distribution'])
        stats_to_save['monomer_usage'] = dict(self.stats['monomer_usage'])
        
        with open(stats_file, 'w') as f:
            json.dump(stats_to_save, f, indent=2)
        
        # 7. 打印处理结果
        print(f"\n 数据处理完成!")
        print(f" 处理统计:")
        print(f"   总数据行数: {self.stats['total_rows']:,}")
        print(f"   有效HELM序列: {self.stats['valid_helm']:,}")
        print(f"   无效序列: {self.stats['invalid_helm']:,}")
        print(f"   过短序列 (<{min_seq_len}): {self.stats['too_short']:,}")
        print(f"   过长序列 (>{max_seq_len}): {self.stats['too_long']:,}")
        print(f"   平均序列长度: {self.stats['avg_length']:.1f}")
        print(f"   有效率: {self.stats['valid_helm']/self.stats['total_rows']*100:.1f}%")
        
        # 8. 显示长度分布
        print(f"\n 序列长度分布 (Top 10):")
        for length, count in self.stats['length_distribution'].most_common(10):
            print(f"   长度 {length}: {count} 个序列")
        
        # 9. 显示常用单体
        print(f"\n 常用单体 (Top 15):")
        for monomer, count in self.stats['monomer_usage'].most_common(15):
            print(f"   {monomer}: {count}")
        
        return str(output_file)
    
    def extract_ring_info(self, helm_seq: str) -> List[str]:
        ring_type = []
        ring_head_tail_idx = []
        ring_head_tail_type = []
        res_num = len(helm_seq.split('{')[1].split('}')[0].split('.'))
        bond_matrix = np.triu(np.ones((res_num, res_num)), 1)

        if helm_seq.split('}')[-1] != '$$$$':
            ring_info = helm_seq.split('}')[-1]
            for ring in ring_info.split('|'):
                r_st = ring.split(':')[1].split('-')[0]
                res_st = int(ring.split(':')[0].split(',')[-1]) - 1
                r_ed = ring.split(':')[2].split('$')[0]
                res_ed = int(ring.split(':')[1].split('-')[1]) - 1

                #ith_bond = (res_ed + 1) * (res_st + 1) - 1
                if res_st > res_ed:
                    r_link = r_ed + r_st
                    ring_head_tail_idx.append([res_ed, res_st])
                    bond_matrix[res_ed, res_st] = 2 + self.ring_type_set.index(r_link)
                    

                else:
                    r_link = r_st + r_ed
                    ring_head_tail_idx.append([res_st, res_ed])
                    bond_matrix[res_st, res_ed] = 2 + self.ring_type_set.index(r_link)
                #if r_link not in  ring_type:
                    # print(r_link)
                    # print(helm)
                ring_head_tail_type.append(r_link)
                ring_type.append(self.ring_type_set.index(r_link))
        
        # 获得上半角的键连矩阵 0表示没有成环键连 往后表示 ['R3R3', 'R1R2', 'R1R3', 'R3R2']
        bond_array = bond_matrix[np.where(bond_matrix > 0)] - 1
        
        return bond_array


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="处理ChEMBL32数据为HELM预训练格式")
    parser.add_argument("--input_file", 
                       default="./data/chembl32/biotherapeutics_dict_prot_flt.csv",
                       help="输入的ChEMBL32 CSV文件")
    parser.add_argument("--output_dir", 
                       default="./data",
                       help="输出目录")
    parser.add_argument("--min_seq_len", type=int, default=3,
                       help="最小序列长度")
    parser.add_argument("--max_seq_len", type=int, default=128,
                       help="最大序列长度")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 创建处理器
        processor = ChEMBL32DataProcessor(
            input_file=args.input_file,
            output_dir=args.output_dir
        )
        
        # 处理数据
        output_file = processor.process_data(
            min_seq_len=args.min_seq_len,
            max_seq_len=args.max_seq_len
        )
        
        print(f"\n 处理成功!")
        print(f" 输出文件: {output_file}")
        print(f" 现在可以使用此文件进行预训练")
        
    except Exception as e:
        print(f" 处理失败: {e}")
        logging.exception("处理过程中发生错误")


if __name__ == "__main__":
    main()
