"""
Diffusion-DPO Loss (Wallace et al. 2023).

Core formula:
    loss = -log σ( β · [(mse_ref_w - mse_θ_w) - (mse_ref_l - mse_θ_l)] )

Where:
    mse_ref_w - mse_theta_w = progress_w
    mse_ref_l - mse_theta_l = progress_l
    margin = progress_w - progress_l

Properties:
    - logsigmoid gives a stable, saturating preference loss.
    - the reference model removes sample-difficulty bias.
    - beta controls how strongly the policy can move from the reference.
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
        mse_theta_w: [Bz] policy MSE on winners
        mse_ref_w:   [Bz] reference MSE on winners
        mse_theta_l: [Bz] policy MSE on losers
        mse_ref_l:   [Bz] reference MSE on losers
        beta:        DPO temperature

    Returns:
        loss: batch-averaged DPO loss
        margin: mean preference margin
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
    Safe scatter mean using explicit sum/count buffers.

    Aggregates flattened per-position MSE values into per-sample MSE.

    Example:
        values = [0.5, 0.3, 0.4, 0.8, 0.6]
        index  = [  0,   0,   0,   1,   1]
        num_segments = 2
        -> result = [0.4, 0.7]

    Args:
        values: values to aggregate
        index: segment id for each value
        num_segments: output length
        device:       torch.device

    Returns:
        Mean value for each segment.
    """
    sum_buf = torch.zeros(num_segments, device=device)
    cnt_buf = torch.zeros(num_segments, device=device)
    sum_buf.scatter_add_(0, index, values)
    cnt_buf.scatter_add_(0, index, torch.ones_like(values))
    return sum_buf / cnt_buf.clamp(min=1)
