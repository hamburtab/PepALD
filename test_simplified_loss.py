"""测试简化后的损失计算"""
import torch
import json
from chembl32_config import ChEMBL32Config
from helm_diffusion import HELMDiffusion
from helm_transformer import HELMTransformer

def test_simplified_loss():
    """测试简化后的损失计算"""
    print(" 测试简化的MolFormer损失计算...")
    
    # 加载配置
    config = ChEMBL32Config()
    
    # 加载词汇表
    with open('data/helm_vocab.json', 'r') as f:
        vocab = json.load(f)
    
    print(f" 词汇表大小: {len(vocab)}")
    
    # 创建transformer
    transformer = HELMTransformer(
        embedding_dim=768,
        d_model=config.d_model,
        n_heads=config.nhead,
        n_layers=config.num_layers,
        d_ff=config.dim_feedforward,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout
    )
    
    # 创建MolFormer模型
    print("\n 创建MolFormer模型...")
    model = HELMDiffusion(
        transformer=transformer,
        vocab_size=len(vocab),
        T=config.T,
        vocab=vocab,
        use_molformer=True
    )
    
    print(f"   模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建测试数据
    print("\n 创建测试数据...")
    batch_size = 4
    seq_len = 20
    
    # 随机生成一些有效的token ids
    input_ids = torch.randint(4, len(vocab), (batch_size, seq_len))  # 避开特殊tokens
    attention_mask = torch.ones_like(input_ids)
    
    print(f"   输入形状: {input_ids.shape}")
    
    # 测试前向传播
    print("\n 测试前向传播...")
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        
        print(f"   主损失: {outputs['loss'].item():.4f}")
        
        if 'reconstruction_loss' in outputs:
            print(f"   重构损失: {outputs['reconstruction_loss'].item():.4f}")
        
        if 'target_embeddings' in outputs:
            target = outputs['target_embeddings']
            print(f"   Target embeddings形状: {target.shape}")
            print(f"   Target embeddings范数: {target.norm():.3f}")
            
        if 'reconstructed' in outputs:
            recon = outputs['reconstructed']
            print(f"   重构embeddings形状: {recon.shape}")
            print(f"   重构embeddings范数: {recon.norm():.3f}")
    
    # 测试多个batch的损失范围
    print("\n 测试损失范围...")
    losses = []
    model.eval()
    
    for i in range(10):
        with torch.no_grad():
            # 每次生成不同的输入
            test_input = torch.randint(4, len(vocab), (2, 15))
            test_mask = torch.ones_like(test_input)
            
            outputs = model(test_input, test_mask)
            losses.append(outputs['loss'].item())
    
    print(f"   损失范围: {min(losses):.4f} - {max(losses):.4f}")
    print(f"   平均损失: {sum(losses)/len(losses):.4f}")
    print(f"   损失标准差: {torch.tensor(losses).std().item():.4f}")
    
    # 测试训练模式
    print("\n 测试训练模式...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    train_losses = []
    for step in range(5):
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = outputs['loss']
        
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        print(f"   步骤 {step+1}: 损失 = {loss.item():.4f}")
    
    print(f"\n 训练效果:")
    print(f"   初始损失: {train_losses[0]:.4f}")
    print(f"   最终损失: {train_losses[-1]:.4f}")
    print(f"   损失变化: {train_losses[-1] - train_losses[0]:.4f}")
    
    print("\n 简化损失测试完成!")
    print(f"\n 预期收敛目标:")
    print(f"   良好收敛: 0.5 - 2.0")
    print(f"   优秀收敛: 0.1 - 0.8")
    print(f"   当前测试范围: {min(losses):.4f} - {max(losses):.4f}")

if __name__ == "__main__":
    test_simplified_loss()
