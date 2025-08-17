"""
创建HELM词汇表文件
基于ChEMBL32数据中的单体生成词汇表
"""

import json
import re
from pathlib import Path
from collections import Counter
from typing import Dict, Set, List

def extract_monomers_from_chembl32() -> Set[str]:
    """从ChEMBL32数据中提取所有单体"""
    monomers = set()
    
    # 读取ChEMBL32处理后的序列
    data_file = "./data/helm_sequences_chembl32.txt"
    
    if not Path(data_file).exists():
        print(f"错误: 数据文件不存在 {data_file}")
        return monomers
    
    with open(data_file, 'r') as f:
        for line in f:
            helm_seq = line.strip()
            if helm_seq.startswith('PEPTIDE1{') and helm_seq.endswith('}$$$$'):
                # 提取单体部分
                content = helm_seq[len('PEPTIDE1{'):-len('}$$$$')]
                if content:
                    # 分割单体
                    sequence_monomers = content.split('.')
                    monomers.update(sequence_monomers)
    
    return monomers

def extract_monomers_from_original() -> Set[str]:
    """从原始数据中提取单体"""
    monomers = set()
    
    # 读取原始预训练序列（如果存在）
    original_file = "./data/helm_sequences_pretrain.txt"
    
    if Path(original_file).exists():
        with open(original_file, 'r') as f:
            for line in f:
                helm_seq = line.strip()
                if helm_seq.startswith('PEPTIDE1{') and helm_seq.endswith('}$$$$'):
                    content = helm_seq[len('PEPTIDE1{'):-len('}$$$$')]
                    if content:
                        sequence_monomers = content.split('.')
                        monomers.update(sequence_monomers)
    
    return monomers

def create_helm_vocab() -> Dict[str, int]:
    """创建HELM词汇表"""
    print(" 创建HELM词汇表...")
    
    # 合并所有数据源的单体
    all_monomers = set()
    
    # ChEMBL32数据
    chembl32_monomers = extract_monomers_from_chembl32()
    print(f"   ChEMBL32单体数: {len(chembl32_monomers)}")
    all_monomers.update(chembl32_monomers)
    
    # 原始数据（如果存在）
    original_monomers = extract_monomers_from_original()
    if original_monomers:
        print(f"   原始数据单体数: {len(original_monomers)}")
        all_monomers.update(original_monomers)
    
    # 添加特殊token
    special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
    
    # 创建词汇表
    vocab = {}
    idx = 0
    
    # 先添加特殊token
    for token in special_tokens:
        vocab[token] = idx
        idx += 1
    
    # 添加所有单体（按字母顺序排序以保证一致性）
    sorted_monomers = sorted(list(all_monomers))
    for monomer in sorted_monomers:
        if monomer and monomer not in vocab:  # 排除空字符串
            vocab[monomer] = idx
            idx += 1
    
    print(f"   总词汇量: {len(vocab)}")
    print(f"   特殊token: {special_tokens}")
    print(f"   单体数量: {len(sorted_monomers)}")
    
    # 显示一些常见单体
    common_amino_acids = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    found_aa = [aa for aa in common_amino_acids if aa in vocab]
    print(f"   标准氨基酸: {found_aa}")
    
    return vocab

def save_vocab(vocab: Dict[str, int], output_file: str = "./data/helm_vocab.json"):
    """保存词汇表到JSON文件"""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    
    print(f" 词汇表已保存: {output_file}")

def create_reverse_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    """创建反向词汇表（索引->token）"""
    return {v: k for k, v in vocab.items()}

def analyze_monomer_frequency() -> Dict[str, int]:
    """分析单体频率"""
    print(" 分析单体使用频率...")
    
    monomer_counts = Counter()
    
    # 读取ChEMBL32数据
    data_file = "./data/helm_sequences_chembl32.txt"
    
    if Path(data_file).exists():
        with open(data_file, 'r') as f:
            for line in f:
                helm_seq = line.strip()
                if helm_seq.startswith('PEPTIDE1{') and helm_seq.endswith('}$$$$'):
                    content = helm_seq[len('PEPTIDE1{'):-len('}$$$$')]
                    if content:
                        monomers = content.split('.')
                        for monomer in monomers:
                            monomer_counts[monomer] += 1
    
    # 显示最常用的单体
    print("   最常用单体 (Top 20):")
    for monomer, count in monomer_counts.most_common(20):
        print(f"     {monomer}: {count}")
    
    return dict(monomer_counts)

def main():
    """主函数"""
    print(" HELM词汇表生成器")
    print("=" * 50)
    
    try:
        # 创建词汇表
        vocab = create_helm_vocab()
        
        # 保存词汇表
        save_vocab(vocab)
        
        # 创建反向词汇表
        reverse_vocab = create_reverse_vocab(vocab)
        reverse_vocab_file = "./data/helm_vocab_reverse.json"
        with open(reverse_vocab_file, 'w') as f:
            json.dump(reverse_vocab, f, indent=2, ensure_ascii=False)
        print(f" 反向词汇表已保存: {reverse_vocab_file}")
        
        # 分析单体频率
        monomer_freq = analyze_monomer_frequency()
        freq_file = "./data/monomer_frequency.json"
        with open(freq_file, 'w') as f:
            json.dump(monomer_freq, f, indent=2, ensure_ascii=False)
        print(f" 单体频率已保存: {freq_file}")
        
        print("\n 词汇表生成完成!")
        print(f" 词汇表统计:")
        print(f"   总词汇量: {len(vocab)}")
        print(f"   特殊token: 4")
        print(f"   单体类型: {len(vocab) - 4}")
        
    except Exception as e:
        print(f" 词汇表生成失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
