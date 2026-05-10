"""Draw the PepAR-Diff framework figure for the manuscript.

The figure is generated as vector-first artwork so labels and formulas remain
sharp in Overleaf. It intentionally mirrors the paper's method flow:
HELM/Uni-Mol vocabulary, autoregressive latent diffusion, R-group-aware ring
prediction, and diffusion-DPO reward alignment.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import (
    Arc,
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    PathPatch,
    Polygon,
    Rectangle,
)
from matplotlib.path import Path as MplPath


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "overleaf-upload-clean" / "paper" / "figures"
OUT_BASENAME = OUT_DIR / "pepar_diff_framework"


COLORS = {
    "ink": "#25313b",
    "muted": "#6c7884",
    "line": "#a8b1bb",
    "panel": "#f8fafc",
    "blue": "#4f8fd8",
    "blue_light": "#dbeafe",
    "cyan": "#58b6c7",
    "cyan_light": "#dff5f7",
    "green": "#4f9d69",
    "green_light": "#e2f4e8",
    "amber": "#e5a94b",
    "amber_light": "#fff1d6",
    "rose": "#d96a6a",
    "rose_light": "#fde2e2",
    "violet": "#8d78bd",
    "violet_light": "#eee8fb",
    "slate_light": "#eef2f6",
}


def setup_axis() -> tuple[plt.Figure, plt.Axes]:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )
    fig = plt.figure(figsize=(17.5, 10.0), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    w: float,
    h: float,
    label: str = "",
    fc: str = "white",
    ec: str = COLORS["line"],
    lw: float = 1.2,
    radius: float = 0.015,
    text_kwargs: dict | None = None,
    zorder: int = 1,
) -> FancyBboxPatch:
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=zorder,
    )
    ax.add_patch(box)
    if label:
        kwargs = {
            "ha": "center",
            "va": "center",
            "color": COLORS["ink"],
            "fontsize": 8,
            "zorder": zorder + 1,
        }
        if text_kwargs:
            kwargs.update(text_kwargs)
        ax.text(xy[0] + w / 2, xy[1] + h / 2, label, **kwargs)
    return box


def add_panel(
    ax: plt.Axes,
    xy: tuple[float, float],
    w: float,
    h: float,
    letter: str,
    title: str,
    accent: str,
) -> None:
    rounded_box(ax, xy, w, h, fc=COLORS["panel"], ec="#ccd3da", lw=1.4, radius=0.018)
    ax.text(
        xy[0] + 0.012,
        xy[1] + h - 0.028,
        f"({letter})",
        fontsize=12,
        fontweight="bold",
        color=COLORS["ink"],
        va="top",
        ha="left",
    )
    ax.text(
        xy[0] + 0.045,
        xy[1] + h - 0.028,
        title,
        fontsize=11,
        fontweight="bold",
        color=COLORS["ink"],
        va="top",
        ha="left",
    )
    ax.plot(
        [xy[0] + 0.045, xy[0] + min(w - 0.018, 0.27)],
        [xy[1] + h - 0.044, xy[1] + h - 0.044],
        color=accent,
        lw=2.2,
        solid_capstyle="round",
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = COLORS["ink"],
    lw: float = 1.4,
    ms: float = 12,
    style: str = "-|>",
    dashed: bool = False,
    rad: float = 0.0,
    alpha: float = 1.0,
    zorder: int = 10,
) -> FancyArrowPatch:
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        linestyle=(0, (3, 3)) if dashed else "solid",
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(arr)
    return arr


def mini_matrix(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    rows: int,
    cols: int,
    palette: list[str],
    labels: list[str] | None = None,
    hatch_cols: set[int] | None = None,
) -> None:
    hatch_cols = hatch_cols or set()
    cw = w / cols
    rh = h / rows
    for r in range(rows):
        for c in range(cols):
            rect = Rectangle(
                (x + c * cw, y + (rows - 1 - r) * rh),
                cw,
                rh,
                facecolor=palette[(r + 2 * c) % len(palette)],
                edgecolor="#ffffff",
                linewidth=0.6,
                hatch="////" if c in hatch_cols else None,
                zorder=3,
            )
            ax.add_patch(rect)
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor="#6b7280", lw=0.8, zorder=4))
    if labels:
        for r, label in enumerate(labels):
            ax.text(
                x - 0.006,
                y + h - (r + 0.5) * rh,
                label,
                ha="right",
                va="center",
                fontsize=6.8,
                color=COLORS["muted"],
            )


def draw_lock(ax: plt.Axes, x: float, y: float, s: float, color: str = COLORS["muted"]) -> None:
    ax.add_patch(Rectangle((x, y), s, s * 0.62, facecolor="white", edgecolor=color, lw=1.0, zorder=5))
    ax.add_patch(Arc((x + s / 2, y + s * 0.63), s * 0.62, s * 0.65, theta1=0, theta2=180, color=color, lw=1.0, zorder=5))
    ax.add_patch(Circle((x + s / 2, y + s * 0.33), s * 0.055, color=color, zorder=5))


def draw_peptide_ring(ax: plt.Axes, cx: float, cy: float, r: float, n: int = 7, accent: str = COLORS["green"]) -> None:
    pts = []
    for i in range(n):
        ang = 2.0 * 3.14159265 * i / n + 0.2
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        ax.plot([x1, x2], [y1, y2], color="#52606d", lw=1.0, zorder=3)
    for i, (x, y) in enumerate(pts):
        fc = [COLORS["blue"], COLORS["amber"], COLORS["green"], COLORS["rose"], COLORS["violet"]][i % 5]
        ax.add_patch(Circle((x, y), r * 0.16, facecolor=fc, edgecolor="white", lw=0.7, zorder=4))
    ax.add_patch(Arc((cx, cy), 2.2 * r, 2.0 * r, theta1=205, theta2=337, color=accent, lw=2.0, zorder=2))


def flow_chip(ax: plt.Axes, x: float, y: float, label: str, fc: str, ec: str, w: float = 0.052) -> None:
    rounded_box(
        ax,
        (x, y),
        w,
        0.026,
        label,
        fc=fc,
        ec=ec,
        lw=0.9,
        radius=0.009,
        text_kwargs={"fontsize": 7.0, "fontweight": "bold"},
        zorder=4,
    )


def draw_panel_a(ax: plt.Axes) -> None:
    add_panel(ax, (0.035, 0.555), 0.25, 0.37, "A", "HELM grammar and chemical codebook", COLORS["blue"])
    px, py, pw, ph = 0.035, 0.555, 0.25, 0.37

    rounded_box(ax, (px + 0.018, py + 0.252), 0.093, 0.075, fc="white", ec=COLORS["blue"], lw=1.1)
    ax.text(px + 0.064, py + 0.309, "HELM corpora", ha="center", va="center", fontsize=8.4, fontweight="bold", color=COLORS["ink"])
    ax.text(px + 0.064, py + 0.284, "ChEMBL32\nCycPeptMPDB", ha="center", va="center", fontsize=7.1, color=COLORS["muted"])
    for i in range(3):
        ax.add_patch(Rectangle((px + 0.026 + i * 0.022, py + 0.262), 0.016, 0.011, facecolor=[COLORS["blue_light"], COLORS["green_light"], COLORS["amber_light"]][i], edgecolor="#90a4b7", lw=0.5))

    rounded_box(ax, (px + 0.137, py + 0.252), 0.105, 0.075, fc="white", ec=COLORS["cyan"], lw=1.1)
    ax.text(px + 0.189, py + 0.309, "Monomer library", ha="center", va="center", fontsize=8.4, fontweight="bold", color=COLORS["ink"])
    ax.text(px + 0.189, py + 0.284, "3105 tokens\nCXSMILES + R sites", ha="center", va="center", fontsize=7.0, color=COLORS["muted"])
    arrow(ax, (px + 0.114, py + 0.289), (px + 0.134, py + 0.289), color=COLORS["muted"], lw=1.1, ms=9)

    rounded_box(ax, (px + 0.018, py + 0.118), 0.224, 0.103, fc="white", ec=COLORS["green"], lw=1.15)
    ax.text(px + 0.032, py + 0.205, "Uni-Mol structured embedding", ha="left", va="center", fontsize=8.6, fontweight="bold", color=COLORS["ink"])
    ax.text(px + 0.032, py + 0.184, r"$e_m=[e^{CLS}, e^{R1}, e^{R2}, e^{R3}]$", ha="left", va="center", fontsize=8.2, color=COLORS["ink"])
    mini_matrix(
        ax,
        px + 0.150,
        py + 0.135,
        0.068,
        0.058,
        4,
        8,
        [COLORS["blue_light"], COLORS["green_light"], COLORS["amber_light"], COLORS["rose_light"], COLORS["violet_light"]],
        labels=["CLS", "R1", "R2", "R3"],
    )
    draw_lock(ax, px + 0.222, py + 0.151, 0.013)
    ax.text(px + 0.223, py + 0.131, "frozen", fontsize=6.8, ha="center", color=COLORS["muted"])
    arrow(ax, (px + 0.130, py + 0.249), (px + 0.130, py + 0.224), color=COLORS["muted"], lw=1.1, ms=9)

    rounded_box(ax, (px + 0.025, py + 0.030), 0.204, 0.058, fc=COLORS["slate_light"], ec="#b9c4cf", lw=0.9)
    ax.text(px + 0.127, py + 0.064, "Native HELM output keeps typed connections", ha="center", va="center", fontsize=7.1, color=COLORS["ink"], fontweight="bold")
    ax.text(px + 0.127, py + 0.043, r"PEPTIDE1{...}\$1:R1-N:R2\$\$\$", ha="center", va="center", fontsize=6.8, color=COLORS["muted"])


def draw_panel_b(ax: plt.Axes) -> None:
    add_panel(ax, (0.305, 0.555), 0.415, 0.37, "B", "Autoregressive latent diffusion step", COLORS["amber"])
    px, py = 0.305, 0.555

    ax.text(px + 0.025, py + 0.305, "generated prefix", ha="left", va="center", fontsize=7.4, color=COLORS["muted"])
    chip_x = px + 0.020
    for i, lab in enumerate([r"$x_1$", r"$x_2$", r"$\cdots$", r"$x_{t-1}$"]):
        flow_chip(ax, chip_x + i * 0.039, py + 0.265, lab, "white", COLORS["line"], w=0.034)
    ax.text(chip_x + 0.153, py + 0.278, "+ start token", fontsize=6.6, color=COLORS["muted"], ha="left")

    rounded_box(ax, (px + 0.022, py + 0.145), 0.105, 0.090, fc=COLORS["blue_light"], ec=COLORS["blue"], lw=1.1)
    ax.text(px + 0.0745, py + 0.212, "Causal Context\nEncoder", ha="center", va="center", fontsize=8.2, fontweight="bold", color=COLORS["ink"])
    mini_matrix(ax, px + 0.043, py + 0.157, 0.041, 0.034, 4, 4, ["#cfe0f5", "#ffffff"], hatch_cols=set())
    tri = Polygon(
        [(px + 0.092, py + 0.157), (px + 0.120, py + 0.157), (px + 0.120, py + 0.185)],
        closed=True,
        facecolor="#ffffff",
        edgecolor=COLORS["blue"],
        lw=0.8,
    )
    ax.add_patch(tri)
    ax.text(px + 0.106, py + 0.191, "mask", fontsize=5.7, ha="center", color=COLORS["muted"])
    arrow(ax, (chip_x + 0.070, py + 0.262), (px + 0.075, py + 0.238), color=COLORS["muted"], lw=1.0, ms=8)

    rounded_box(ax, (px + 0.162, py + 0.124), 0.130, 0.132, fc=COLORS["amber_light"], ec=COLORS["amber"], lw=1.2)
    ax.text(px + 0.227, py + 0.236, "Diffusion Engine", ha="center", va="center", fontsize=8.5, fontweight="bold", color=COLORS["ink"])
    ax.text(px + 0.227, py + 0.213, r"$z_{t,K}\rightarrow z_{t,k}\rightarrow z_{t,0}$", ha="center", va="center", fontsize=8.2, color=COLORS["ink"])
    for i, fc in enumerate(["#d5d9df", "#c7d8ef", "#b9e6ed", "#fff6cf"]):
        ax.add_patch(Circle((px + 0.183 + i * 0.028, py + 0.188), 0.010, facecolor=fc, edgecolor="#8291a3", lw=0.6))
        if i < 3:
            arrow(ax, (px + 0.194 + i * 0.028, py + 0.188), (px + 0.204 + i * 0.028, py + 0.188), color=COLORS["muted"], lw=0.9, ms=7)
    rounded_box(ax, (px + 0.180, py + 0.138), 0.094, 0.035, fc="white", ec="#c9a762", lw=0.8)
    ax.text(px + 0.227, py + 0.157, "AdaLN(k) +\nCross-attn($h_t$)", ha="center", va="center", fontsize=6.5, color=COLORS["muted"])

    arrow(ax, (px + 0.129, py + 0.190), (px + 0.159, py + 0.190), color=COLORS["ink"], lw=1.3, ms=11)
    ax.text(px + 0.139, py + 0.205, r"$h_t$", fontsize=8.4, color=COLORS["ink"], ha="center")

    rounded_box(ax, (px + 0.324, py + 0.128), 0.075, 0.125, fc=COLORS["green_light"], ec=COLORS["green"], lw=1.1)
    ax.text(px + 0.3615, py + 0.235, "Hybrid Token\nMapper", ha="center", va="center", fontsize=7.7, fontweight="bold", color=COLORS["ink"])
    mini_matrix(ax, px + 0.336, py + 0.169, 0.035, 0.045, 5, 5, [COLORS["blue_light"], COLORS["green_light"], COLORS["amber_light"], COLORS["violet_light"]])
    ax.text(px + 0.3615, py + 0.153, "cosine codebook\n+ LM prior\n+ R constraints", ha="center", va="center", fontsize=6.1, color=COLORS["muted"])
    arrow(ax, (px + 0.294, py + 0.190), (px + 0.321, py + 0.190), color=COLORS["ink"], lw=1.3, ms=11)
    ax.text(px + 0.306, py + 0.207, r"$z_t$", fontsize=8.4, color=COLORS["ink"], ha="center")

    flow_chip(ax, px + 0.358, py + 0.283, r"$x_t$", COLORS["rose_light"], COLORS["rose"], w=0.042)
    arrow(ax, (px + 0.361, py + 0.254), (px + 0.375, py + 0.281), color=COLORS["ink"], lw=1.2, ms=10)
    arrow(ax, (px + 0.379, py + 0.283), (chip_x + 0.149, py + 0.283), color=COLORS["rose"], lw=1.2, ms=9, dashed=True, rad=0.20)
    ax.text(
        px + 0.251,
        py + 0.316,
        "append and repeat until target length T",
        fontsize=7.0,
        color=COLORS["rose"],
        ha="center",
        bbox={"facecolor": COLORS["panel"], "edgecolor": "none", "pad": 0.7},
        zorder=20,
    )

    rounded_box(ax, (px + 0.036, py + 0.040), 0.344, 0.050, fc="white", ec="#cfd8e3", lw=0.85)
    ax.text(
        px + 0.208,
        py + 0.066,
        r"Training: $L_{diff}$ predicts noise, $L_{CE}$ regularizes the LM head",
        ha="center",
        va="center",
        fontsize=7.4,
        color=COLORS["ink"],
    )


def draw_panel_c(ax: plt.Axes) -> None:
    add_panel(ax, (0.740, 0.555), 0.225, 0.37, "C", "R-group-aware ring predictor", COLORS["rose"])
    px, py = 0.740, 0.555
    draw_peptide_ring(ax, px + 0.060, py + 0.235, 0.043, n=6, accent=COLORS["rose"])
    ax.text(px + 0.060, py + 0.300, "history $x_j$, current $x_t$", ha="center", va="center", fontsize=7.0, color=COLORS["muted"])

    rounded_box(ax, (px + 0.122, py + 0.222), 0.078, 0.072, fc="white", ec=COLORS["rose"], lw=1.0)
    ax.text(px + 0.161, py + 0.279, "3 x 3 R-map", ha="center", va="center", fontsize=7.2, fontweight="bold", color=COLORS["ink"])
    mini_matrix(ax, px + 0.139, py + 0.235, 0.044, 0.031, 3, 3, [COLORS["rose_light"], COLORS["amber_light"], COLORS["violet_light"], COLORS["blue_light"]])
    ax.text(px + 0.161, py + 0.223, r"$G=(E_t^R E_j^{R\top})/\sqrt{d_R}$", ha="center", va="center", fontsize=5.9, color=COLORS["muted"])
    arrow(ax, (px + 0.095, py + 0.241), (px + 0.119, py + 0.253), color=COLORS["muted"], lw=1.0, ms=8)

    rounded_box(ax, (px + 0.027, py + 0.125), 0.071, 0.061, fc=COLORS["blue_light"], ec=COLORS["blue"], lw=0.95)
    ax.text(px + 0.0625, py + 0.158, r"$MLP_{ctx}$", ha="center", va="center", fontsize=8.0, fontweight="bold", color=COLORS["ink"])
    ax.text(px + 0.0625, py + 0.139, r"$[h_t || h_j]$", ha="center", va="center", fontsize=6.8, color=COLORS["muted"])

    rounded_box(ax, (px + 0.122, py + 0.125), 0.071, 0.061, fc=COLORS["amber_light"], ec=COLORS["amber"], lw=0.95)
    ax.text(px + 0.1575, py + 0.158, r"$MLP_R$", ha="center", va="center", fontsize=8.0, fontweight="bold", color=COLORS["ink"])
    ax.text(px + 0.1575, py + 0.139, r"$vec(G)$", ha="center", va="center", fontsize=6.8, color=COLORS["muted"])
    arrow(ax, (px + 0.161, py + 0.220), (px + 0.1575, py + 0.189), color=COLORS["muted"], lw=1.0, ms=8)

    rounded_box(ax, (px + 0.048, py + 0.050), 0.130, 0.045, fc=COLORS["slate_light"], ec="#b9c4cf", lw=0.9)
    ax.text(px + 0.113, py + 0.081, "position head + type head", ha="center", va="center", fontsize=7.1, fontweight="bold", color=COLORS["ink"])
    ax.text(px + 0.113, py + 0.063, r"$\sigma(s^{pos})$ ; softmax {R3R3,R1R2,R1R3,R3R2}", ha="center", va="center", fontsize=5.9, color=COLORS["muted"])
    arrow(ax, (px + 0.063, py + 0.123), (px + 0.094, py + 0.098), color=COLORS["muted"], lw=1.0, ms=8)
    arrow(ax, (px + 0.158, py + 0.123), (px + 0.132, py + 0.098), color=COLORS["muted"], lw=1.0, ms=8)

    rounded_box(ax, (px + 0.037, py + 0.010), 0.152, 0.026, fc="white", ec=COLORS["rose"], lw=0.85)
    ax.text(px + 0.113, py + 0.023, "chemical gate: distance + free R-groups", ha="center", va="center", fontsize=6.2, color=COLORS["rose"])


def draw_stage(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, body: str, fc: str, ec: str) -> None:
    rounded_box(ax, (x, y), w, h, fc=fc, ec=ec, lw=1.05, radius=0.013)
    ax.text(x + w / 2, y + h - 0.021, title, ha="center", va="top", fontsize=8.2, fontweight="bold", color=COLORS["ink"])
    ax.text(x + w / 2, y + h / 2 - 0.010, body, ha="center", va="center", fontsize=6.6, color=COLORS["muted"], linespacing=1.18)


def draw_panel_d(ax: plt.Axes) -> None:
    add_panel(ax, (0.035, 0.075), 0.930, 0.425, "D", "Training and reward-alignment workflow", COLORS["violet"])
    px, py = 0.035, 0.075

    y = py + 0.245
    h = 0.122
    stage_w = 0.145
    xs = [px + 0.028, px + 0.205, px + 0.382, px + 0.560, px + 0.740]
    draw_stage(
        ax,
        xs[0],
        y,
        stage_w,
        h,
        "Supervised pretrain",
        "ChEMBL32 HELM\nteacher forcing\n$L_{diff} + \\lambda_{CE}L_{CE}$",
        COLORS["blue_light"],
        COLORS["blue"],
    )
    draw_stage(
        ax,
        xs[1],
        y,
        stage_w,
        h,
        "Macrocycle fine-tune",
        "CycPeptMPDB cycles\nring supervision\n$L_{pos}+L_{type}$",
        COLORS["green_light"],
        COLORS["green"],
    )
    draw_stage(
        ax,
        xs[2],
        y,
        stage_w,
        h,
        "Permeability prior",
        "top 1000 CPP-like\ncyclic-only stage\npolicy initialization",
        COLORS["amber_light"],
        COLORS["amber"],
    )
    draw_stage(
        ax,
        xs[3],
        y,
        stage_w,
        h,
        "Candidate scoring",
        "Uni-Dock/Vina\npermeability RF\nchemistry prior",
        COLORS["rose_light"],
        COLORS["rose"],
    )
    draw_stage(
        ax,
        xs[4],
        y,
        stage_w,
        h,
        "Diffusion-DPO",
        "MMR diverse winners\nnearest hard negatives\nupdate denoiser",
        COLORS["violet_light"],
        COLORS["violet"],
    )

    for i in range(4):
        arrow(ax, (xs[i] + stage_w + 0.006, y + h / 2), (xs[i + 1] - 0.008, y + h / 2), color=COLORS["ink"], lw=1.25, ms=10)

    rounded_box(ax, (px + 0.050, py + 0.060), 0.225, 0.100, fc="white", ec="#cfd8e3", lw=0.95)
    ax.text(px + 0.162, py + 0.139, "Composite reward", ha="center", va="center", fontsize=8.5, fontweight="bold", color=COLORS["ink"])
    ax.text(
        px + 0.162,
        py + 0.103,
        r"$R=w_{vina}\phi(-S_{vina})+w_{perm}\phi(\hat{P})+w_{chem}\phi(C)$",
        ha="center",
        va="center",
        fontsize=7.5,
        color=COLORS["ink"],
    )
    ax.text(px + 0.162, py + 0.078, "robust median/MAD normalization; invalid docking removed", ha="center", va="center", fontsize=6.4, color=COLORS["muted"])

    rounded_box(ax, (px + 0.336, py + 0.060), 0.225, 0.100, fc="white", ec="#cfd8e3", lw=0.95)
    ax.text(px + 0.448, py + 0.139, "Preference pairs", ha="center", va="center", fontsize=8.5, fontweight="bold", color=COLORS["ink"])
    ax.text(px + 0.448, py + 0.111, "top 30% / bottom 30% pools", ha="center", va="center", fontsize=6.8, color=COLORS["muted"])
    ax.text(px + 0.448, py + 0.090, "select 10% diverse winners/losers", ha="center", va="center", fontsize=6.8, color=COLORS["muted"])
    ax.text(px + 0.448, py + 0.069, r"pair by min $d_J$ with reward gap $\delta_R$", ha="center", va="center", fontsize=6.8, color=COLORS["muted"])

    rounded_box(ax, (px + 0.623, py + 0.060), 0.250, 0.100, fc="white", ec="#cfd8e3", lw=0.95)
    ax.text(px + 0.748, py + 0.139, "DPO loss on denoising progress", ha="center", va="center", fontsize=8.5, fontweight="bold", color=COLORS["ink"])
    ax.text(
        px + 0.748,
        py + 0.104,
        r"$-\log\sigma\{\beta[(MSE_{ref}-MSE_\theta)_w-(MSE_{ref}-MSE_\theta)_l]\}$",
        ha="center",
        va="center",
        fontsize=7.0,
        color=COLORS["ink"],
    )
    ax.text(px + 0.748, py + 0.078, "winner and loser share timestep k and Gaussian noise", ha="center", va="center", fontsize=6.4, color=COLORS["muted"])

    arrow(ax, (px + 0.672, py + 0.245), (px + 0.692, py + 0.164), color=COLORS["rose"], lw=1.1, ms=9, dashed=True)
    arrow(ax, (px + 0.275, py + 0.110), (px + 0.333, py + 0.110), color=COLORS["muted"], lw=1.0, ms=9)
    arrow(ax, (px + 0.561, py + 0.110), (px + 0.620, py + 0.110), color=COLORS["muted"], lw=1.0, ms=9)

    draw_peptide_ring(ax, px + 0.897, py + 0.105, 0.034, n=7, accent=COLORS["violet"])
    ax.text(px + 0.897, py + 0.048, "optimized macrocyclic HELM", ha="center", va="center", fontsize=7.0, color=COLORS["ink"], fontweight="bold")
    arrow(ax, (xs[4] + stage_w / 2, y - 0.006), (px + 0.878, py + 0.144), color=COLORS["violet"], lw=1.15, ms=9)


def draw_cross_panel_links(ax: plt.Axes) -> None:
    arrow(ax, (0.258, 0.675), (0.303, 0.675), color=COLORS["ink"], lw=1.3, ms=12)
    ax.text(0.280, 0.693, "frozen reference embeddings", ha="center", va="center", fontsize=6.8, color=COLORS["muted"])
    arrow(ax, (0.720, 0.675), (0.740, 0.675), color=COLORS["ink"], lw=1.3, ms=12)
    ax.text(0.731, 0.696, r"$x_t,h_t,E^R$", ha="center", va="center", fontsize=6.8, color=COLORS["muted"])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = setup_axis()

    ax.text(
        0.035,
        0.965,
        "PepAR-Diff: AutoRegressive Latent Diffusion for Macrocyclic Peptide Generation",
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.035,
        0.938,
        "Residue-level HELM generation in a chemically grounded Uni-Mol latent space with R-group-aware cyclization and diffusion-DPO reward alignment.",
        ha="left",
        va="top",
        fontsize=8.4,
        color=COLORS["muted"],
    )

    draw_panel_a(ax)
    draw_panel_b(ax)
    draw_panel_c(ax)
    draw_panel_d(ax)
    draw_cross_panel_links(ax)

    for ext in ("pdf", "svg", "png"):
        path = OUT_BASENAME.with_suffix(f".{ext}")
        if ext == "png":
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
        else:
            fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
        print(path)


if __name__ == "__main__":
    main()
