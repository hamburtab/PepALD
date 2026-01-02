"""
评估生成样本的完整指标：validity, uniqueness, diversity, snn, novelty
"""
import sys
import os

# 将项目根目录加入 path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.metrics_utils import Metrics

def main():
    # 读取生成的 HELM 序列
    samples_file = os.path.join(project_root, 'chembl32_samples/helm_chembl32only_samples.txt')
    # samples_file = os.path.join(project_root, 'chembl32_samples/random_generated_samples.txt')
    with open(samples_file, 'r') as f:
        helms = [line.strip() for line in f if line.strip()]
    
    print(f"加载了 {len(helms)} 个生成样本")
    
    # 初始化 Metrics（使用训练集 prior_data.csv 中的 cano_smi 列作为参考）
    prior_path = os.path.join(project_root, 'data/prior_data.csv')
    metrics = Metrics(
        prior_path=prior_path,
        n_jobs=1,  # 单线程避免多进程问题
        input_type='helm'
    )
    
    # 计算所有指标
    print("\n计算指标中...")
    results = metrics.get_metrics(helms)
    
    print("\n=== 评估结果 ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
