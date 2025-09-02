import json
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import Dict, Set, List

def extract_monomers_from_library() -> Set[str]:
    """从monomer_library.csv中提取所有单体符号"""
    monomers = set()
    
    library_file = "./data/monomer_library.csv"
    
    if not Path(library_file).exists():
        print(f"错误: 单体库文件不存在 {library_file}")
        return monomers
    
    try:
        df = pd.read_csv(library_file)
        if 'Symbol' in df.columns:
            # 提取所有单体符号，去除空值
            symbols = df['Symbol'].dropna().unique()
            monomers.update(symbols)
            print(f"   从单体库加载了 {len(monomers)} 个单体")
        else:
            print(f"错误: CSV文件中未找到'Symbol'列")
    except Exception as e:
        print(f"错误: 读取单体库文件失败 - {e}")
    
    return monomers

def create_helm_vocab() -> Dict[str, int]:
    print(" 创建HELM词汇表...")
    
    # 从单体库加载所有单体
    library_monomers = extract_monomers_from_library()
    print(f"   单体库中的单体数: {len(library_monomers)}")
    
    # 使用单体库中的所有单体
    all_monomers = library_monomers
    
    # 定义特殊token
    special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
    
    vocab = {}
    idx = 0
    
    # 首先添加特殊token
    for token in special_tokens:
        vocab[token] = idx
        idx += 1
    
    # 然后添加所有单体（按字母顺序排序以保证一致性）
    sorted_monomers = sorted(list(all_monomers))
    for monomer in sorted_monomers:
        if monomer and monomer not in vocab:
            vocab[monomer] = idx
            idx += 1
    
    print(f"   总词汇量: {len(vocab)}")
    print(f"   特殊token: {special_tokens}")
    print(f"   单体数量: {len(sorted_monomers)}")
    
    # 显示一些常见氨基酸单体
    common_amino_acids = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    found_aa = [aa for aa in common_amino_acids if aa in vocab]
    print(f"   标准氨基酸: {found_aa}")
    
    # 显示前10个单体作为示例
    example_monomers = sorted_monomers[:10]
    print(f"   示例单体: {example_monomers}")
    
    return vocab

def save_vocab_to_json(vocab: Dict[str, int], filepath: str):
    """保存词汇表到JSON文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"   词汇表已保存到: {filepath}")

def create_reverse_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    """创建反向词汇表（索引->单体）"""
    return {v: k for k, v in vocab.items()}

def analyze_monomer_frequency() -> Dict[str, int]:
    """分析ChEMBL32数据中的单体使用频率"""
    print(" 分析单体使用频率...")
    
    monomer_counts = Counter()
    
    data_file = "./data/helm_sequences_chembl32.txt"
    
    if not Path(data_file).exists():
        print(f"   警告: 数据文件不存在 {data_file}")
        return {}
    
    total_sequences = 0
    with open(data_file, 'r') as f:
        for line in f:
            helm_seq = line.strip()
            if helm_seq.startswith('PEPTIDE1{') and helm_seq.endswith('}$$$$'):
                total_sequences += 1
                content = helm_seq[len('PEPTIDE1{'):-len('}$$$$')]
                if content:
                    monomers = content.split('.')
                    for monomer in monomers:
                        monomer_counts[monomer] += 1
    
    print(f"   分析了 {total_sequences} 个序列")
    print("   最常用单体 (Top 20):")
    for monomer, count in monomer_counts.most_common(20):
        print(f"     {monomer}: {count}")
    
    return dict(monomer_counts)

def main():
    print(" HELM词汇表生成器")
    print("=" * 50)
    
    try:
        vocab = create_helm_vocab()
        
        save_vocab_to_json(vocab, "./data/helm_vocab.json")
        
        reverse_vocab = create_reverse_vocab(vocab)
        reverse_vocab_file = "./data/helm_vocab_reverse.json"
        with open(reverse_vocab_file, 'w') as f:
            json.dump(reverse_vocab, f, indent=2, ensure_ascii=False)
        print(f" 反向词汇表已保存: {reverse_vocab_file}")
        
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
