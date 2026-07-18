"""Plot a synthetic Standard-DPO vs WP-DPO ablation example."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "draw" / "wp_dpo_ablation_demo.png"

epochs = np.arange(25)
standard_vina = np.array([
    -7.18, -7.25, -7.32, -7.40, -7.43, -7.51, -7.58, -7.60, -7.64,
    -7.72, -7.70, -7.76, -7.83, -7.81, -7.86, -7.90, -7.87, -7.88,
    -7.91, -7.86, -7.88, -7.84, -7.80, -7.82, -7.79,
])
wp_vina = np.array([
    -7.18, -7.26, -7.35, -7.44, -7.49, -7.57, -7.66, -7.71, -7.80,
    -7.88, -7.93, -8.01, -8.07, -8.13, -8.19, -8.24, -8.28, -8.31,
    -8.36, -8.39, -8.43, -8.46, -8.49, -8.53, -8.56,
])
standard_hit = np.array([
    21, 23, 24, 27, 28, 30, 31, 33, 34, 37, 36, 38, 40, 39, 41, 43,
    42, 43, 44, 42, 43, 41, 40, 40, 39,
])
wp_hit = np.array([
    21, 23, 25, 28, 29, 32, 34, 36, 39, 41, 43, 46, 48, 50, 52, 54,
    55, 56, 58, 59, 60, 61, 62, 64, 65,
])


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    blue = "#4472C4"
    orange = "#ED7D31"
    grid = "#D9DEE7"

    fig, axes = plt.subplots(2, 1, figsize=(10.4, 7.2), sharex=True)
    fig.suptitle(
        "Case 1: Standard DPO vs. WP-DPO (synthetic illustration)",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )

    for ax in axes:
        for boundary in range(3, 25, 3):
            ax.axvline(boundary, color=grid, lw=1.0, ls="--", zorder=0)
        ax.grid(axis="y", color=grid, lw=0.8, alpha=0.8)
        ax.set_xlim(-0.3, 24.8)

    axes[0].plot(
        epochs,
        standard_vina,
        color=blue,
        lw=2.0,
        ls="--",
        marker="o",
        ms=4.2,
        label="Standard DPO ($\\alpha_{win}=0$)",
    )
    axes[0].plot(
        epochs,
        wp_vina,
        color=orange,
        lw=2.2,
        marker="s",
        ms=4.0,
        label="WP-DPO ($\\alpha_{win}>0$)",
    )
    axes[0].fill_between(
        epochs,
        standard_vina - 0.06,
        standard_vina + 0.06,
        color=blue,
        alpha=0.10,
        linewidth=0,
    )
    axes[0].fill_between(
        epochs,
        wp_vina - 0.06,
        wp_vina + 0.06,
        color=orange,
        alpha=0.10,
        linewidth=0,
    )
    axes[0].axhline(-7.55, color="#666666", lw=1.1, ls=":")
    axes[0].text(
        24.6,
        -7.55,
        "Reference ligand",
        ha="right",
        va="bottom",
        color="#555555",
    )
    axes[0].set_ylabel("Mean Vina score (kcal/mol)\nLower is better")
    axes[0].legend(loc="upper right", frameon=False, ncol=2)

    axes[1].plot(
        epochs,
        standard_hit,
        color=blue,
        lw=2.0,
        ls="--",
        marker="o",
        ms=4.2,
    )
    axes[1].plot(
        epochs,
        wp_hit,
        color=orange,
        lw=2.2,
        marker="s",
        ms=4.0,
    )
    axes[1].fill_between(
        epochs,
        standard_hit - 2.2,
        standard_hit + 2.2,
        color=blue,
        alpha=0.10,
        linewidth=0,
    )
    axes[1].fill_between(
        epochs,
        wp_hit - 2.2,
        wp_hit + 2.2,
        color=orange,
        alpha=0.10,
        linewidth=0,
    )
    axes[1].set_ylabel("Reference-beating rate (%)\nHigher is better")
    axes[1].set_xlabel("Cumulative DPO epoch")
    axes[1].set_xticks(np.arange(0, 25, 3))

    top_axis = axes[0].secondary_xaxis("top")
    top_axis.set_xticks(np.arange(1.5, 24, 3))
    top_axis.set_xticklabels([f"Round {idx}" for idx in range(1, 9)])
    top_axis.tick_params(length=0, pad=5, labelsize=9)
    top_axis.spines["top"].set_visible(False)

    fig.text(
        0.5,
        0.012,
        "Virtual data for layout preview only. Shaded bands illustrate uncertainty intervals.",
        ha="center",
        color="#555555",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.04, 0.045, 0.995, 0.95), h_pad=1.45)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
