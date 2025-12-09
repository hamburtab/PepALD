
import torch
from pathlib import Path
import argparse

from helm_transformer import create_helm_transformer_for_chembl32
from helm_diffusion import HELMDiffusion, HELMSequenceDataset


# ==================== 生成控制开关 ====================
# 在这里统一控制所有生成相关的功能开关

# 基础生成参数
NUM_SAMPLES = 100                # 生成样本数量
MAX_SEQ_LEN = 20                 # 最大序列长度
MIN_SEQ_LEN = 5                  # 最小序列长度（随机长度生成）
CHECKPOINT_PATH = "chembl32_checkpoints/chembl32_latest_model.pth"  # 模型检查点路径
OUTPUT_FILE = "chembl32_samples/helm_chembl32only_samples.txt"      # 输出文件路径

# 功能开关
USE_EMBEDDING_NORM = True        # 是否对embedding做L2归一化（使用余弦相似度匹配）
USE_FREQ_WEIGHT = True           # 是否启用频率加权（惩罚低频单体）
USE_BLACKLIST = True             # 是否启用黑名单过滤（排除问题单体）
USE_TEMPERATURE_SAMPLING = False # 是否启用温度采样（概率性选择token）

# 参数值
FREQ_WEIGHT_SCALE = 0.1          # 频率惩罚强度 (0.0-1.0)
TOKEN_SAMPLING_TEMPERATURE = 0.5 # 温度采样温度值 (仅当USE_TEMPERATURE_SAMPLING=True时生效)

# 环键预测
PREDICT_RING_BONDS = True        # 是否预测环键连接

# ======================================================


def generate_samples(
    checkpoint_path: str, 
    num_samples: int, 
    max_seq_len: int,
    min_seq_len: int = 5,
    output_file: str = "chembl32_samples/helm_chembl32only_samples.txt",
    use_embedding_norm: bool = True,
    use_freq_weight: bool = True,
    use_blacklist: bool = True,
    use_temperature_sampling: bool = False,
    freq_weight_scale: float = 0.1,
    token_sampling_temperature: float = 0.5,
    predict_ring_bonds: bool = True
):
    """
    使用指定的检查点生成HELM样本。

    Args:
        checkpoint_path: 最佳模型检查点文件的路径。
        num_samples: 要生成的样本数量。
        max_seq_len: 生成样本的最大序列长度。
        min_seq_len: 生成样本的最小序列长度。
        output_file: 输出文件路径。
        use_embedding_norm: 是否对embedding做L2归一化（余弦相似度匹配）。
        use_freq_weight: 是否启用频率加权（惩罚低频单体）。
        use_blacklist: 是否启用黑名单过滤（排除问题单体）。
        use_temperature_sampling: 是否启用温度采样（概率性选择token）。
        freq_weight_scale: 频率惩罚强度。
        token_sampling_temperature: 温度采样的温度值。
        predict_ring_bonds: 是否预测环键连接。
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
        use_unimol=True,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        d_ff=config.dim_feedforward,
        # 功能开关
        use_embedding_norm=use_embedding_norm,
        use_freq_weight=use_freq_weight,
        use_blacklist=use_blacklist,
        use_temperature_sampling=use_temperature_sampling,
        # 参数值
        freq_weight_scale=freq_weight_scale,
        token_sampling_temperature=token_sampling_temperature
    ).to(device)

    # 3. 加载模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 打印当前配置
    print("模型加载成功！")
    print(f"\n当前生成配置:")
    print(f"  - use_embedding_norm (L2归一化): {use_embedding_norm}")
    print(f"  - use_freq_weight (频率加权): {use_freq_weight}")
    print(f"  - use_blacklist (黑名单过滤): {use_blacklist}")
    print(f"  - use_temperature_sampling (温度采样): {use_temperature_sampling}")
    if use_freq_weight:
        print(f"  - freq_weight_scale: {freq_weight_scale}")
    if use_temperature_sampling:
        print(f"  - token_sampling_temperature: {token_sampling_temperature}")

    # 4. 生成样本
    print(f"\n正在生成 {num_samples} 个样本...")
    with torch.no_grad():
        generated_samples = model.sample(
            num_samples=num_samples,
            max_seq_len=max_seq_len,
            device=device,
            predict_ring_bonds=predict_ring_bonds,
            min_seq_len=min_seq_len
        )

    # 5. 解码并打印样本
    print("\n--- 生成的HELM序列 ---")
    
    # 确保输出目录存在
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for i, sample in enumerate(generated_samples):
            if isinstance(sample, dict) and 'tokens' in sample:
                ring_connections = sample.get('ring_connections', [])
                # 使用dataset的解码方法
                helm_seq = dataset.decode_sequence(sample['tokens'], ring_connections)
                
                # 计算序列长度（排除PAD）
                pad_id = vocab.get('<PAD>', 0)
                seq_len = (sample['tokens'] != pad_id).sum().item()
                
                ring_info = f"[环键数: {len(ring_connections)}]" if ring_connections else "[线性]"
                f.write(f"{helm_seq}\n")
                print(f"样本 {i+1} {ring_info}（长度：{seq_len}）: {helm_seq}")
            else:
                # 不返回字典的情况
                helm_seq = dataset.decode_sequence(sample)
                # 计算序列长度
                pad_id = vocab.get('<PAD>', 0)
                seq_len = (sample != pad_id).sum().item()
                print(f"样本 {i+1}（长度：{seq_len}）: {helm_seq}")
    print("--- 生成并写入完毕 ---\n")


if __name__ == "__main__":
    # 使用文件顶部定义的配置参数
    generate_samples(
        checkpoint_path=CHECKPOINT_PATH,
        num_samples=NUM_SAMPLES,
        max_seq_len=MAX_SEQ_LEN,
        min_seq_len=MIN_SEQ_LEN,
        output_file=OUTPUT_FILE,
        # 功能开关
        use_embedding_norm=USE_EMBEDDING_NORM,
        use_freq_weight=USE_FREQ_WEIGHT,
        use_blacklist=USE_BLACKLIST,
        use_temperature_sampling=USE_TEMPERATURE_SAMPLING,
        # 参数值
        freq_weight_scale=FREQ_WEIGHT_SCALE,
        token_sampling_temperature=TOKEN_SAMPLING_TEMPERATURE,
        predict_ring_bonds=PREDICT_RING_BONDS
    )

