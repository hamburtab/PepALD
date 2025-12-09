"""
评估生成样本的 validity, uniqueness, diversity, snn, novelty 指标
"""
import sys
sys.path.insert(0, '..')

import pandas as pd
from utils.metrics_utils import Metrics

def main():
    # 读取生成的样本
    sample_file = "../chembl32_samples/helm_chembl32only_samples.txt"
    with open(sample_file, 'r') as f:
        helms = [line.strip() for line in f if line.strip()]
    
    print(f"加载了 {len(helms)} 个生成样本")
    
    # 创建 Metrics 对象，使用训练集的 canonical SMILES 作为参考
    # prior_data.csv 第二列是 cano_smi
    prior_df = pd.read_csv("../data/prior_data.csv")
    prior_smiles = prior_df['cano_smi'].dropna().tolist()
    
    # 临时创建一个只有 smiles 的文件供 Metrics 使用
    tmp_prior = "/tmp/prior_smiles.csv"
    pd.DataFrame(prior_smiles).to_csv(tmp_prior, index=False, header=False)
    
    metrics = Metrics(prior_path=tmp_prior, n_jobs=1, input_type='helm')
    
    print("\n计算指标中...")
    results = metrics.get_metrics(helms)
    
    print("\n===== 评估结果 =====")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
