import pandas as pd
from pathlib import Path
import argparse
def extract_helm_sequences(input_csv, output_txt):
    """从CSV文件直接提取HELM列的所有数据到txt文件"""
    print(f"正在加载数据: {input_csv}")
    
    if not Path(input_csv).exists():
        raise FileNotFoundError(f"输入文件不存在: {input_csv}")
    
    # 读取CSV文件
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"成功读取 {len(df)} 行数据")
    
    if 'HELM' not in df.columns:
        raise ValueError("CSV文件中未找到HELM列")
    
    # 直接提取HELM列的所有非空数据
    helm_sequences = df['HELM'].dropna().astype(str).str.strip()
    
    # 过滤掉空字符串
    helm_sequences = helm_sequences[helm_sequences != '']
    
    # 统计线性肽和环肽
    # 线性肽：以}$$$$结尾
    linear_count = helm_sequences.str.endswith('}$$$$').sum()
    # 环肽：包含连接信息，以$$$结尾（但不是}$$$$）
    cyclic_count = (helm_sequences.str.endswith('$$$') & ~helm_sequences.str.endswith('}$$$$')).sum()
    
    # 保存到txt文件
    output_path = Path(output_txt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for seq in helm_sequences:
            f.write(seq + '\n')
    
    # 打印统计信息
    print(f"\n提取完成!")
    print(f"总数据行数: {len(df):,}")
    print(f"有效HELM序列: {len(helm_sequences):,}")
    print(f"  - 线性肽 (}}$$$$): {linear_count:,}")
    print(f"  - 环肽 (含连接信息，以$$$结尾): {cyclic_count:,}")
    print(f"输出文件: {output_path}")
    
    return str(output_path)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="从CSV文件提取HELM序列到txt文件")
    parser.add_argument("--input_csv", 
                       default="./data/chembl32/biotherapeutics_dict_prot_flt.csv",
                       help="输入的CSV文件路径")
    parser.add_argument("--output_txt", 
                       default="./data/helm_sequences_chembl32.txt",
                       help="输出的txt文件路径")
    
    args = parser.parse_args()
    
    try:
        # 提取HELM序列
        output_file = extract_helm_sequences(args.input_csv, args.output_txt)
        print(f"\n提取成功！输出文件: {output_file}")
        
    except Exception as e:
        print(f"提取失败: {e}")


if __name__ == "__main__":
    main()
