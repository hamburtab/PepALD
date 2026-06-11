"""
评估生成样本的完整指标：validity, uniqueness, diversity, snn, novelty
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pepar_diff.utils.metrics import Metrics

# DEFAULT_INPUT = "outputs/samples/case1/generated/helm_dpo_samples.txt"
DEFAULT_INPUT = "outputs/samples/pretrain/helm_chembl32only_samples.txt"
# DEFAULT_INPUT = "scripts/eval_reference_model/helmgpt_samples/perm_samples.txt"
# DEFAULT_INPUT = "outputs/samples/finetune/cycpeptMPDB_samples.txt"
DEFAULT_PRIOR = "data/processed/prior_data.csv"

def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate full HELM generation metrics")
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help="Input HELM sample file, one sequence per line.",
    )
    parser.add_argument(
        "--prior_path",
        type=str,
        default=DEFAULT_PRIOR,
        help="Prior CSV used as reference for SNN/novelty metrics.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 读取生成的 HELM 序列
    samples_file = resolve_path(args.input)
    prior_path = resolve_path(args.prior_path)

    if not samples_file.exists():
        raise FileNotFoundError(f"Input sample file not found: {samples_file}")
    if not prior_path.exists():
        raise FileNotFoundError(f"Prior CSV not found: {prior_path}")

    with open(samples_file, 'r') as f:
        helms = [line.strip() for line in f if line.strip()]
    
    print(f"输入文件: {samples_file}")
    print(f"参考集:   {prior_path}")
    print(f"加载了 {len(helms)} 个生成样本")
    
    # 初始化 Metrics（使用训练集 prior_data.csv 中的 cano_smi 列作为参考）
    metrics = Metrics(
        prior_path=str(prior_path),
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
