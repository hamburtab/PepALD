"""
Diffusion-DPO Loss (Wallace et al. 2023).

Core formula:
    loss = -log σ( β · [(mse_ref_w - mse_θ_w) - (mse_ref_l - mse_θ_l)] )

Where:
    mse_ref_w - mse_θ_w = progress_w  (新模型在好样本上比老模型进步了多少)
    mse_ref_l - mse_θ_l = progress_l  (新模型在差样本上比老模型进步了多少)
    margin = progress_w - progress_l   (我们希望好样本上的进步远大于差样本)

Properties:
    - logsigmoid 提供自稳定梯度: margin 大时梯度自动消失, 防止过拟合
    - ref_model 提供 baseline: 消除样本本身难度差异
    - beta 控制偏离 ref 的程度: 越大越激进, 越小越保守
"""

import torch
import torch.nn.functional as F


def compute_diffusion_dpo_loss(
    mse_theta_w: torch.Tensor,
    mse_ref_w: torch.Tensor,
    mse_theta_l: torch.Tensor,
    mse_ref_l: torch.Tensor,
    beta: float = 0.1
):
    """
    Compute standard Diffusion-DPO loss.

    Args:
        mse_theta_w: [Bz] 新模型在好样本上的 per-sample MSE
        mse_ref_w:   [Bz] 老模型在好样本上的 per-sample MSE
        mse_theta_l: [Bz] 新模型在差样本上的 per-sample MSE
        mse_ref_l:   [Bz] 老模型在差样本上的 per-sample MSE
        beta:        DPO 温度系数

    Returns:
        loss:   标量, batch 平均后的 DPO loss
        margin: 标量, 监控用 (正值说明模型更偏好好样本)
    """
    progress_w = mse_ref_w - mse_theta_w       # [Bz]
    progress_l = mse_ref_l - mse_theta_l       # [Bz]
    margin = progress_w - progress_l           # [Bz]
    loss = -F.logsigmoid(beta * margin)        # [Bz]

    return loss.mean(), margin.mean()


def scatter_mean(
    values: torch.Tensor,
    index: torch.Tensor,
    num_segments: int,
    device: torch.device
) -> torch.Tensor:
    """
    Safe scatter mean: 手动算 sum/count, 避免 scatter_reduce_ 的 include_self 陷阱.

    用途: 将 flatten 后的 per-position MSE 按样本 index 聚合为 per-sample MSE.

    示例:
        values = [0.5, 0.3, 0.4, 0.8, 0.6]   # 5 个 valid 位置的 MSE
        index  = [  0,   0,   0,   1,   1]     # 前3个属于样本0, 后2个属于样本1
        num_segments = 2
        → result = [0.4, 0.7]                  # 样本0: (0.5+0.3+0.4)/3, 样本1: (0.8+0.6)/2

    Args:
        values:       [N]   要聚合的值
        index:        [N]   每个值所属的 segment (0 ~ num_segments-1)
        num_segments: int   输出长度 (= Bz)
        device:       torch.device

    Returns:
        result: [num_segments] 每个 segment 的均值
    """
    sum_buf = torch.zeros(num_segments, device=device)
    cnt_buf = torch.zeros(num_segments, device=device)
    sum_buf.scatter_add_(0, index, values)
    cnt_buf.scatter_add_(0, index, torch.ones_like(values))
    return sum_buf / cnt_buf.clamp(min=1)
