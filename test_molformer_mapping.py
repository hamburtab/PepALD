"""测试MolFormer embedding映射修复"""
import json
import os
import torch
from molformer_embedding import MolFormerEmbedding

def test_vocab_mapping():
    """测试词汇表映射效果"""
    # 加载词汇表
    with open('data/helm_vocab.json', 'r') as f:
        vocab = json.load(f)
    
    # 创建MolFormer embedding
    molformer = MolFormerEmbedding("molformer_embeddings", vocab)
    
    print(" 映射结果分析:")
    
    # 分析特殊tokens
    special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
    print("\n  特殊Token映射:")
    for token in special_tokens:
        if token in vocab:
            vocab_idx = vocab[token]
            embedding_idx = molformer.vocab_to_embedding[vocab_idx].item()
            print(f"   {token} -> vocab[{vocab_idx}] -> embedding[{embedding_idx}]")
    
    # 分析常规单体映射
    regular_tokens = [token for token in vocab.keys() if not token.startswith('<')]
    matched_regular = 0
    
    print(f"\n 常规单体映射 (前20个):")
    for i, token in enumerate(regular_tokens[:20]):
        vocab_idx = vocab[token]
        embedding_idx = molformer.vocab_to_embedding[vocab_idx].item()
        
        # 检查是否是有效映射（不是UNK）
        unk_embedding_idx = molformer.num_monomers + special_tokens.index('<UNK>')
        is_matched = embedding_idx != unk_embedding_idx and embedding_idx < molformer.num_monomers
        
        if is_matched:
            matched_regular += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"   {status} {token} -> vocab[{vocab_idx}] -> embedding[{embedding_idx}]")
    
    # 总体统计
    total_regular = len(regular_tokens)
    total_matched = 0
    
    for token in regular_tokens:
        vocab_idx = vocab[token]
        embedding_idx = molformer.vocab_to_embedding[vocab_idx].item()
        unk_embedding_idx = molformer.num_monomers + special_tokens.index('<UNK>')
        
        if embedding_idx != unk_embedding_idx and embedding_idx < molformer.num_monomers:
            total_matched += 1
    
    print(f"\n 映射效果总结:")
    print(f"   总词汇数: {len(vocab)}")
    print(f"   常规单体数: {total_regular}")
    print(f"   成功映射: {total_matched}")
    print(f"   映射成功率: {total_matched/total_regular*100:.1f}%")
    
    # 测试embedding获取
    print(f"\n Embedding测试:")
    sample_tokens = list(vocab.keys())[:5]
    for token in sample_tokens:
        vocab_idx = vocab[token]
        try:
            embedding = molformer.get_embedding(vocab_idx)
            print(f"   {token}: 形状={embedding.shape}, 范数={embedding.norm():.3f}")
        except Exception as e:
            print(f"   {token}: 错误 - {e}")

if __name__ == "__main__":
    test_vocab_mapping()
