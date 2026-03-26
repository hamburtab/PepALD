# Project: New-HELM-Diffusion

## 项目概述
基于 Autoregressive Latent Diffusion (ALD) 的环肽生成模型。模型逐位置自回归生成，每个位置内部用 diffusion 过程生成 Uni-Mol embedding，再映射到离散 monomer token。

## 当前任务：Diffusion-DPO 微调
目标：用 DPO (Direct Preference Optimization) 微调 pretrained 模型，使生成的环肽在 **Vina docking score**（与 6DN5 蛋白结合）和 **预测透膜性** 两个指标上更优。

### DPO 核心公式
```
loss = -log σ( β · [(mse_ref_w - mse_θ_w) - (mse_ref_l - mse_θ_l)] )
```
- ref_model: pretrained 模型的冻结副本，提供 baseline
- model: 被 DPO 更新的模型，当前只解冻 denoiser
- 好/差样本共享同一个 t 和 noise（配对方差缩减）

### DPO 训练流程
1. 用 pretrained model 生成 2000 条环肽
2. 对每条算 reward = w1 * (-vina_score) + w2 * perm_score（vina 越负越好，取负号统一为越大越好）
3. Top-25% = winner, Bottom-25% = loser, 中间丢弃
4. 配对后送入 DPO 训练

### 关键实现细节
- denoiser 输入是 2D [batch, dm]，不是 3D，所以训练时需要将 [Bz, L] 的 valid 位置 flatten 成大 batch
- per-position MSE 通过 scatter_mean 按样本 index 聚合回 per-sample MSE [Bz]
- scatter_mean 手动实现 sum/count，避免 PyTorch scatter_reduce_ 的 include_self=True 陷阱
- context_encoder 当前被冻结，model 和 ref_model 的 context 输出相同，但代码保留分别计算的写法以兼容未来解冻

### 文件结构
```
train_dpo.py                    ← DPO 训练主入口
configs/dpo.json                ← DPO 配置（beta=0.1, lr=1e-5, denoiser_only）
ald/dpo/
├── __init__.py
├── loss.py                     ← compute_diffusion_dpo_loss + scatter_mean
├── dataset.py                  ← PreferencePairDataset + build_preference_pairs
└── trainer.py                  ← DPOTrainer（train_step 中实现完整 DPO 流程）
Vina/
├── __init__.py
├── vina_score.py               ← 单条 SMILES → Vina docking score（已加异常处理）
├── dock.py                     ← HELM 列表 → batch docking（HELM→SMILES→vina_score）
├── 6dn5_receptor.pdbqt         ← 受体蛋白（锁）
└── raw_cyclic_pep.sdf          ← 参考配体（定位 docking box 中心）
```

### 核心模型文件
```
ald/models/ald_model.py         ← AutoregressiveLatentDiffusion 主模型
ald/models/context_encoder.py   ← CausalContextEncoder
ald/diffusion/engine.py         ← DiffusionEngine (add_noise, predict_noise, sample)
ald/diffusion/denoiser.py       ← DiffusionDenoiser (实际的 noise prediction 网络)
ald/models/token_mapper.py      ← embedding → discrete token
ald/utils/data.py               ← HELMDataset, CyclicHELMDataset
utils/helm2smiles.py            ← HELM ↔ SMILES 转换
eval/eval_permeability.py       ← 随机森林透膜性预测
```

### 当前进度
- [x] DPO loss 函数 (ald/dpo/loss.py)
- [x] Preference pair 数据集 (ald/dpo/dataset.py)
- [x] DPO trainer (ald/dpo/trainer.py)
- [x] 训练主脚本 (train_dpo.py)
- [x] Vina docking 评分 (Vina/vina_score.py)
- [x] HELM→Vina 衔接 (Vina/dock.py)
- [x] DPO 配置 (configs/dpo.json)
- [ ] 端到端测试运行
- [ ] 验证生成质量（validity, reward 分布, diversity）

### 分支
当前在 `ALD-DPO` 分支上开发。
