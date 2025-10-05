
import torch
from pathlib import Path
import argparse

from helm_transformer import create_helm_transformer_for_chembl32
from helm_diffusion import HELMDiffusion, HELMSequenceDataset

def generate_samples(checkpoint_path: str, num_samples: int, max_seq_len: int):
    """
    使用指定的检查点生成HELM样本。

    Args:
        checkpoint_path: 最佳模型检查点文件的路径。
        num_samples: 要生成的样本数量。
        max_seq_len: 生成样本的最大序列长度。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 加载检查点
    if not Path(checkpoint_path).exists():
        print(f"错误: 检查点文件不存在 '{checkpoint_path}'")
        return

    print(f"正在从 '{checkpoint_path}' 加载模型...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 从检查点恢复配置和词汇表
    config_dict = checkpoint['config']
    # 将字典转换为类似对象的结构以便访问
    config = argparse.Namespace(**config_dict)
    vocab = checkpoint['vocab']
    
    # 1. 构建一个临时的Dataset实例，主要为了使用它的解码功能
    # 这里提供一个虚拟的数据文件路径，因为我们实际上不会加载数据
    dummy_data_file = Path(config.chembl32_data_file)
    if not dummy_data_file.parent.exists():
        dummy_data_file.parent.mkdir(parents=True, exist_ok=True)
    if not dummy_data_file.exists():
        dummy_data_file.touch()

    dataset = HELMSequenceDataset(
        data_file=str(dummy_data_file),
        max_seq_len=config.max_seq_len,
        vocab_file=config.vocab_file
    )
    # 确保dataset的词汇表与模型训练时一致
    dataset.vocab = vocab
    dataset.idx_to_token = {v: k for k, v in vocab.items()}


    # 2. 构建模型
    print("正在构建模型结构...")
    transformer = create_helm_transformer_for_chembl32(
        vocab_size=len(vocab),
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout
    )
    
    model = HELMDiffusion(
        transformer=transformer,
        vocab_size=len(vocab),
        T=config.T,
        beta_schedule=config.beta_schedule,
        vocab=vocab,
        use_molformer=True,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        d_ff=config.dim_feedforward
    ).to(device)

    # 3. 加载模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("模型加载成功！")

    # 4. 生成样本
    print(f"\n正在生成 {num_samples} 个样本...")
    with torch.no_grad():
        generated_samples = model.sample(
            num_samples=num_samples,
            max_seq_len=max_seq_len,
            device=device,
            predict_ring_bonds=True  # 启用环键预测
        )

    # 5. 解码并打印样本
    print("\n--- 生成的HELM序列 ---")
    with open('../chembl32_samples/helm_chembl32only.txt', 'w') as f:
        for i, sample in enumerate(generated_samples):
            if isinstance(sample, dict) and 'tokens' in sample:
                ring_connections = sample.get('ring_connections', [])
                # 使用dataset的解码方法
                helm_seq = dataset.decode_sequence(sample['tokens'], ring_connections)
                ring_info = f"[环键数: {len(ring_connections)}]" if ring_connections else "[线性]"
                f.write(f"{helm_seq}\n")
                print(f"样本 {i+1} {ring_info}: {helm_seq}")
            else:
                # 不返回字典的情况
                helm_seq = dataset.decode_sequence(sample)
                print(f"样本 {i+1}: {helm_seq}")
    print("--- 生成并写入完毕 ---\n")


if __name__ == "__main__":
    # 使用训练好的最佳模型
    best_model_path = "chembl32_checkpoints/chembl32_best_model.pth"
    
    # 定义生成参数
    num_to_generate = 10  # 生成10个样本
    sequence_length = 46  # 与训练时保持一致的最大长度

    generate_samples(
        checkpoint_path=best_model_path,
        num_samples=num_to_generate,
        max_seq_len=sequence_length
    )
