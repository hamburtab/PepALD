"""
快速测试脚本：使用较少的扩散步数和样本来测试归一化效果
"""
import torch
from pathlib import Path
import argparse
import sys

from helm_transformer import create_helm_transformer_for_chembl32
from helm_diffusion import HELMDiffusion, HELMSequenceDataset

def quick_test(num_samples=20, max_seq_len=10, diffusion_steps=100):
    """
    快速测试生成
    Args:
        num_samples: 生成样本数量
        max_seq_len: 最大序列长度
        diffusion_steps: 扩散步数（越少越快）
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    checkpoint_path = "chembl32_checkpoints/chembl32_latest_model.pth"
    
    print(f"正在加载模型...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = argparse.Namespace(**checkpoint['config'])
    vocab = checkpoint['vocab']
    
    # 创建dataset用于解码
    dataset = HELMSequenceDataset(
        data_file=str(config.chembl32_data_file),
        max_seq_len=config.max_seq_len,
        vocab_file=config.vocab_file
    )
    dataset.vocab = vocab
    dataset.idx_to_token = {v: k for k, v in vocab.items()}
    
    # 构建模型
    transformer = create_helm_transformer_for_chembl32(
        vocab_size=len(vocab),
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout
    )
    
    # 使用较少的扩散步数来加速
    model = HELMDiffusion(
        transformer=transformer,
        vocab_size=len(vocab),
        T=diffusion_steps,  # 减少步数
        beta_schedule=config.beta_schedule,
        vocab=vocab,
        use_unimol=True,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        d_ff=config.dim_feedforward
    ).to(device)
    
    # 只加载transformer权重（扩散参数会重新计算）
    model_state = checkpoint['model_state_dict']
    # 过滤掉扩散相关的buffer
    filtered_state = {k: v for k, v in model_state.items() 
                      if not any(x in k for x in ['alphas', 'betas', 'sigmas', 'sqrt'])}
    model.load_state_dict(filtered_state, strict=False)
    model.eval()
    print("模型加载成功！")
    
    # 生成样本
    print(f"\n正在生成 {num_samples} 个样本（步数={diffusion_steps}）...")
    with torch.no_grad():
        samples = model.sample(
            num_samples=num_samples,
            max_seq_len=max_seq_len,
            device=device,
            predict_ring_bonds=False,
            min_seq_len=5
        )
    
    # 解码并统计
    print("\n生成的HELM序列:")
    from collections import Counter
    mono_counts = Counter()
    
    for i, sample in enumerate(samples):
        helm_seq = dataset.decode_sequence(sample)
        print(f"{i+1}. {helm_seq}")
        
        # 统计单体
        try:
            seq_part = helm_seq.split('{')[1].split('}')[0]
            monomers = seq_part.split('.')
            mono_counts.update(monomers)
        except:
            pass
    
    print("\n生成样本中的单体频率:")
    total = sum(mono_counts.values())
    for mono, count in mono_counts.most_common(15):
        print(f"  {mono}: {count} ({count/total*100:.1f}%)")
    
    # 验证有效性
    print("\n验证有效性...")
    sys.path.append('.')
    from utils.helm2smiles import is_helm_valid
    
    valid_count = 0
    for sample in samples:
        helm_seq = dataset.decode_sequence(sample)
        if is_helm_valid(helm_seq):
            valid_count += 1
    
    print(f"有效序列: {valid_count}/{num_samples} ({valid_count/num_samples*100:.1f}%)")


if __name__ == "__main__":
    # 快速测试：20个样本，10个单体长度，100步扩散
    quick_test(num_samples=20, max_seq_len=10, diffusion_steps=100)
