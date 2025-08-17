"""测试MolFormer集成的小规模预训练"""
import torch
import json
from chembl32_config import ChEMBL32Config
from helm_diffusion import HELMDiffusion

def test_molformer_integration():
    """测试MolFormer集成功能"""
    print("🧪 测试MolFormer集成...")
    
    # 加载配置
    config = ChEMBL32Config()
    
    # 加载词汇表
    with open('data/helm_vocab.json', 'r') as f:
        vocab = json.load(f)
    
    print(f" 词汇表大小: {len(vocab)}")
    
    # 创建模型 - 不使用MolFormer
    print("\n 创建标准模型 (不使用MolFormer)...")
    model_standard = HELMDiffusion(
        vocab_size=len(vocab),
        d_model=config.d_model,
        num_heads=config.nhead,
        num_layers=config.num_layers,
        vocab=vocab,
        use_molformer=False
    )
    
    # 创建模型 - 使用MolFormer
    print(" 创建MolFormer模型...")
    model_molformer = HELMDiffusion(
        vocab_size=len(vocab),
        d_model=config.d_model,
        num_heads=config.nhead,
        num_layers=config.num_layers,
        vocab=vocab,
        use_molformer=True
    )
    
    print(f"   标准模型参数: {sum(p.numel() for p in model_standard.parameters()):,}")
    print(f"   MolFormer模型参数: {sum(p.numel() for p in model_molformer.parameters()):,}")
    
    # 创建测试数据
    print("\n 创建测试数据...")
    batch_size = 2
    seq_len = 10
    
    # 随机生成一些有效的token ids
    input_ids = torch.randint(4, len(vocab), (batch_size, seq_len))  # 避开特殊tokens
    attention_mask = torch.ones_like(input_ids)
    
    print(f"   输入形状: {input_ids.shape}")
    print(f"   输入样例: {input_ids[0][:5].tolist()}")
    
    # 测试标准模型
    print("\n 测试标准模型前向传播...")
    model_standard.eval()
    with torch.no_grad():
        outputs_standard = model_standard(input_ids, attention_mask)
        print(f"   标准模型损失: {outputs_standard['loss'].item():.4f}")
        print(f"   输出键: {list(outputs_standard.keys())}")
    
    # 测试MolFormer模型
    print("\n 测试MolFormer模型前向传播...")
    model_molformer.eval()
    with torch.no_grad():
        outputs_molformer = model_molformer(input_ids, attention_mask)
        print(f"   MolFormer模型损失: {outputs_molformer['loss'].item():.4f}")
        print(f"   输出键: {list(outputs_molformer.keys())}")
        
        # 检查ground truth embeddings
        if 'ground_truth_embeddings' in outputs_molformer:
            gt_emb = outputs_molformer['ground_truth_embeddings']
            print(f"   Ground truth embeddings形状: {gt_emb.shape}")
            print(f"   Ground truth embeddings范数: {gt_emb.norm():.3f}")
        
        # 检查重构损失
        if 'reconstruction_loss' in outputs_molformer:
            recon_loss = outputs_molformer['reconstruction_loss']
            print(f"   重构损失: {recon_loss.item():.4f}")
    
    # 测试训练步骤
    print("\n 测试训练步骤...")
    model_molformer.train()
    optimizer = torch.optim.AdamW(model_molformer.parameters(), lr=1e-4)
    
    initial_loss = None
    for step in range(3):
        optimizer.zero_grad()
        outputs = model_molformer(input_ids, attention_mask)
        loss = outputs['loss']
        
        if initial_loss is None:
            initial_loss = loss.item()
        
        loss.backward()
        optimizer.step()
        
        print(f"   步骤 {step+1}: 损失 = {loss.item():.4f}")
    
    print(f"\n 训练效果:")
    print(f"   初始损失: {initial_loss:.4f}")
    print(f"   最终损失: {loss.item():.4f}")
    print(f"   损失变化: {loss.item() - initial_loss:.4f}")
    
    print("\n MolFormer集成测试完成!")

if __name__ == "__main__":
    test_molformer_integration()
