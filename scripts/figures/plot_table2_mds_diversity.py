"""Draw an MDS diversity map for peptide sample sets.

The figure is built from valid molecules only:
1. convert HELM inputs to SMILES when requested,
2. canonicalize molecules with RDKit,
3. compute Morgan fingerprints,
4. embed pairwise Tanimoto distances with MDS,
5. save the scatter plot and per-sample audit tables.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pepar_diff.utils.helm import get_cycpep_smi_from_helm  # noqa: E402


rdBase.DisableLog("rdApp.*")


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "overleaf-upload-clean" / "result_data" / "table2_data"


@dataclass(frozen=True)
class SampleSet:
    label: str
    path: Path
    input_type: str
    color: str


@dataclass
class ValidRecord:
    label: str
    source_path: Path
    source_index: int
    raw: str
    smiles: str
    canonical_smiles: str
    mol: Chem.Mol


@dataclass
class InvalidRecord:
    label: str
    source_path: Path
    source_index: int
    raw: str
    reason: str


def parse_sample_arg(value: str) -> SampleSet:
    """Parse label:path:input_type:color."""
    parts = value.split(":")
    if len(parts) < 3:
        raise argparse.ArgumentTypeError(
            "--sample must be 'label:path:input_type[:color]', where input_type is helm, smiles, or auto"
        )

    label = parts[0].strip()
    if len(parts) == 3:
        path_text = parts[1].strip()
        input_type = parts[2].strip().lower()
        color = ""
    else:
        path_text = ":".join(parts[1:-2]).strip()
        input_type = parts[-2].strip().lower()
        color = parts[-1].strip()

    if input_type not in {"helm", "smiles", "auto"}:
        raise argparse.ArgumentTypeError("input_type must be helm, smiles, or auto")
    if not label:
        raise argparse.ArgumentTypeError("sample label cannot be empty")
    if not path_text:
        raise argparse.ArgumentTypeError("sample path cannot be empty")

    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return SampleSet(label=label, path=path, input_type=input_type, color=color)


def default_samples() -> list[SampleSet]:
    data_dir = PROJECT_ROOT / "overleaf-upload-clean" / "result_data" / "table2_data"
    return [
        SampleSet(
            label="Our pretrained model",
            path=data_dir / "prior_chembl_1000samples.txt",
            input_type="helm",
            color="#7a1f66",
        ),
        SampleSet(
            label="pepMDLM",
            path=data_dir / "pepMDLM.txt",
            input_type="smiles",
            color="#1f77b4",
        ),
        SampleSet(
            label="HELM-GPT",
            path=data_dir / "HELM-GPT.txt",
            input_type="helm",
            color="#4daf4a",
        ),
        SampleSet(
            label="Top1000",
            path=data_dir / "top1000.txt",
            input_type="helm",
            color="#e6550d",
        ),
    ]


def ensure_sample_colors(samples: list[SampleSet]) -> list[SampleSet]:
    palette = ["#7a1f66", "#1f77b4", "#4daf4a", "#e6550d", "#6a51a3", "#525252"]
    colored = []
    for idx, sample in enumerate(samples):
        color = sample.color or palette[idx % len(palette)]
        colored.append(
            SampleSet(
                label=sample.label,
                path=sample.path,
                input_type=sample.input_type,
                color=color,
            )
        )
    return colored


def infer_input_type(raw: str) -> str:
    if "{" in raw and "$" in raw:
        return "helm"
    return "smiles"


def read_lines(path: Path) -> list[tuple[int, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return [(idx, line.strip()) for idx, line in enumerate(handle, start=1) if line.strip()]


def smiles_from_raw(raw: str, input_type: str) -> tuple[str | None, str | None]:
    resolved_type = infer_input_type(raw) if input_type == "auto" else input_type
    if resolved_type == "helm":
        smiles = get_cycpep_smi_from_helm(raw)
        if not smiles:
            return None, "helm_to_smiles_failed"
        return smiles, None
    if resolved_type == "smiles":
        return raw, None
    return None, f"unsupported_input_type:{resolved_type}"


def load_records(samples: list[SampleSet]) -> tuple[list[ValidRecord], list[InvalidRecord]]:
    valid_records: list[ValidRecord] = []
    invalid_records: list[InvalidRecord] = []

    for sample in samples:
        if not sample.path.exists():
            raise FileNotFoundError(sample.path)

        for source_index, raw in read_lines(sample.path):
            smiles, error = smiles_from_raw(raw, sample.input_type)
            if error is not None:
                invalid_records.append(
                    InvalidRecord(sample.label, sample.path, source_index, raw, error)
                )
                continue

            mol = Chem.MolFromSmiles(smiles) if smiles else None
            if mol is None:
                invalid_records.append(
                    InvalidRecord(sample.label, sample.path, source_index, raw, "invalid_smiles")
                )
                continue

            valid_records.append(
                ValidRecord(
                    label=sample.label,
                    source_path=sample.path,
                    source_index=source_index,
                    raw=raw,
                    smiles=smiles,
                    canonical_smiles=Chem.MolToSmiles(mol),
                    mol=mol,
                )
            )

    return valid_records, invalid_records


def morgan_fingerprints(mols: list[Chem.Mol], radius: int, n_bits: int) -> list[DataStructs.ExplicitBitVect]:
    return [AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits) for mol in mols]


def tanimoto_distance_matrix(fps: list[DataStructs.ExplicitBitVect]) -> np.ndarray:
    n_items = len(fps)
    distances = np.zeros((n_items, n_items), dtype=np.float32)

    for i, fp in enumerate(fps):
        similarities = DataStructs.BulkTanimotoSimilarity(fp, fps[:i])
        if similarities:
            row = 1.0 - np.asarray(similarities, dtype=np.float32)
            distances[i, :i] = row
            distances[:i, i] = row

    return distances


def select_records(
    records: list[ValidRecord],
    samples: list[SampleSet],
    max_per_label: int | None,
    seed: int,
) -> list[ValidRecord]:
    if max_per_label is None or max_per_label <= 0:
        return records

    rng = np.random.default_rng(seed)
    selected: list[ValidRecord] = []

    for sample in samples:
        label_records = [record for record in records if record.label == sample.label]
        if len(label_records) <= max_per_label:
            selected.extend(label_records)
            continue

        chosen = np.sort(rng.choice(len(label_records), size=max_per_label, replace=False))
        selected.extend(label_records[int(idx)] for idx in chosen)

    return selected


def shuffle_records(records: list[ValidRecord], seed: int) -> list[ValidRecord]:
    order = np.arange(len(records))
    np.random.default_rng(seed).shuffle(order)
    return [records[int(idx)] for idx in order]


def min_valid_count(records: list[ValidRecord], samples: list[SampleSet]) -> int:
    counts = [sum(1 for record in records if record.label == sample.label) for sample in samples]
    return min(counts) if counts else 0


def embed_mds(
    distances: np.ndarray,
    seed: int,
    max_iter: int,
    n_init: int,
    n_jobs: int,
) -> np.ndarray:
    from sklearn.manifold import MDS

    try:
        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=seed,
            normalized_stress="auto",
            max_iter=max_iter,
            n_init=n_init,
            n_jobs=n_jobs,
        )
    except TypeError:
        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=seed,
            max_iter=max_iter,
            n_init=n_init,
            n_jobs=n_jobs,
        )

    return mds.fit_transform(distances)


def write_invalid_csv(path: Path, invalid_records: list[InvalidRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "source_path", "source_index", "reason", "raw"],
        )
        writer.writeheader()
        for record in invalid_records:
            writer.writerow(
                {
                    "label": record.label,
                    "source_path": str(record.source_path.relative_to(PROJECT_ROOT)),
                    "source_index": record.source_index,
                    "reason": record.reason,
                    "raw": record.raw,
                }
            )


def write_points_csv(path: Path, records: list[ValidRecord], coords: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "source_path",
                "source_index",
                "mds_1",
                "mds_2",
                "canonical_smiles",
                "smiles",
                "raw",
            ],
        )
        writer.writeheader()
        for record, xy in zip(records, coords, strict=True):
            writer.writerow(
                {
                    "label": record.label,
                    "source_path": str(record.source_path.relative_to(PROJECT_ROOT)),
                    "source_index": record.source_index,
                    "mds_1": f"{xy[0]:.8f}",
                    "mds_2": f"{xy[1]:.8f}",
                    "canonical_smiles": record.canonical_smiles,
                    "smiles": record.smiles,
                    "raw": record.raw,
                }
            )


def summarize_counts(samples: list[SampleSet], valid: list[ValidRecord], invalid: list[InvalidRecord]) -> dict[str, dict[str, int]]:
    summary = {
        sample.label: {"total": 0, "valid": 0, "invalid": 0, "selected": 0}
        for sample in samples
    }
    for sample in samples:
        summary[sample.label]["total"] = len(read_lines(sample.path))
    for record in valid:
        summary[record.label]["valid"] += 1
    for record in invalid:
        summary[record.label]["invalid"] += 1
    return summary


def add_selected_counts(summary: dict[str, dict[str, int]], selected: list[ValidRecord]) -> None:
    for record in selected:
        summary[record.label]["selected"] += 1


def write_summary_csv(path: Path, summary: dict[str, dict[str, int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "total", "valid", "invalid", "selected", "valid_fraction"],
        )
        writer.writeheader()
        for label, counts in summary.items():
            total = counts["total"]
            valid = counts["valid"]
            writer.writerow(
                {
                    "label": label,
                    "total": total,
                    "valid": valid,
                    "invalid": counts["invalid"],
                    "selected": counts["selected"],
                    "valid_fraction": f"{valid / total:.4f}" if total else "0.0000",
                }
            )


def plot_embedding(
    output_path: Path,
    samples: list[SampleSet],
    records: list[ValidRecord],
    coords: np.ndarray,
    title: str,
    point_size: float,
    alpha: float,
    dpi: int,
    seed: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(5.2, 5.8), dpi=dpi)

    color_by_label = {sample.label: sample.color for sample in samples}
    order = np.arange(len(records))
    np.random.default_rng(seed).shuffle(order)
    colors = [color_by_label[records[int(idx)].label] for idx in order]

    ax.scatter(
        coords[order, 0],
        coords[order, 1],
        s=point_size,
        c=colors,
        alpha=alpha,
        linewidths=0,
    )

    legend_handles = []
    for sample in samples:
        indices = [idx for idx, record in enumerate(records) if record.label == sample.label]
        if not indices:
            continue
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=sample.color,
                markeredgewidth=0,
                markersize=5,
                label=sample.label,
            )
        )

    ax.set_xlabel("axis-1", loc="right", fontsize=8, color="#555555", labelpad=2)
    ax.set_ylabel("")
    ax.text(
        0.0,
        1.015,
        "axis-2",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        -0.04,
        1.055,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
        fontsize=11,
        color="#111111",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8a8a8a")
    ax.spines["bottom"].set_color("#8a8a8a")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.legend(handles=legend_handles, frameon=False, markerscale=1.0, fontsize=7, loc="upper right")
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout(pad=0.9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot an MDS map from peptide fingerprints for Table 2 diversity visualization."
    )
    parser.add_argument(
        "--sample",
        action="append",
        type=parse_sample_arg,
        default=None,
        help=(
            "Sample definition as label:path:input_type[:color]. "
            "Can be passed multiple times. input_type is helm, smiles, or auto."
        ),
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure_name", default="table2_mds_diversity.png")
    parser.add_argument("--pdf_name", default="table2_mds_diversity.pdf")
    parser.add_argument("--points_name", default="table2_mds_points.csv")
    parser.add_argument("--invalid_name", default="table2_mds_invalid.csv")
    parser.add_argument("--summary_name", default="table2_mds_summary.csv")
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--n_bits", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_iter", type=int, default=300)
    parser.add_argument("--n_init", type=int, default=2)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument(
        "--max_per_label",
        type=int,
        default=300,
        help="Maximum valid molecules sampled per label for MDS. Use 0 to keep all valid molecules.",
    )
    parser.add_argument(
        "--balance_to_min_valid",
        action="store_true",
        help="Use the minimum valid count across labels as --max_per_label.",
    )
    parser.add_argument("--point_size", type=float, default=3.2)
    parser.add_argument("--alpha", type=float, default=0.82)
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--title", default="a")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = ensure_sample_colors(args.sample if args.sample is not None else default_samples())
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    mpl_config_dir = output_dir / ".mplconfig"
    mpl_config_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    valid_records, invalid_records = load_records(samples)
    if len(valid_records) < 3:
        raise ValueError(f"Need at least 3 valid molecules for MDS, got {len(valid_records)}")

    max_per_label = args.max_per_label
    if args.balance_to_min_valid:
        max_per_label = min_valid_count(valid_records, samples)

    selected_records = select_records(valid_records, samples, max_per_label, args.seed)
    if len(selected_records) < 3:
        raise ValueError(
            f"Need at least 3 selected valid molecules for MDS, got {len(selected_records)}"
        )
    selected_records = shuffle_records(selected_records, args.seed)

    fps = morgan_fingerprints([record.mol for record in selected_records], args.radius, args.n_bits)
    distances = tanimoto_distance_matrix(fps)
    coords = embed_mds(
        distances,
        seed=args.seed,
        max_iter=args.max_iter,
        n_init=args.n_init,
        n_jobs=args.n_jobs,
    )

    summary = summarize_counts(samples, valid_records, invalid_records)
    add_selected_counts(summary, selected_records)
    write_summary_csv(output_dir / args.summary_name, summary)
    write_invalid_csv(output_dir / args.invalid_name, invalid_records)
    write_points_csv(output_dir / args.points_name, selected_records, coords)

    figure_path = output_dir / args.figure_name
    pdf_path = output_dir / args.pdf_name
    plot_embedding(
        figure_path,
        samples,
        selected_records,
        coords,
        args.title,
        args.point_size,
        args.alpha,
        args.dpi,
        args.seed,
    )
    plot_embedding(
        pdf_path,
        samples,
        selected_records,
        coords,
        args.title,
        args.point_size,
        args.alpha,
        args.dpi,
        args.seed,
    )

    print("MDS diversity figure complete.")
    print(f"Valid molecules: {len(valid_records)}")
    print(f"Selected molecules for MDS: {len(selected_records)}")
    print(f"Max per label used: {max_per_label}")
    print(f"Invalid/failed records: {len(invalid_records)}")
    for label, counts in summary.items():
        print(
            f"{label}: {counts['valid']}/{counts['total']} valid "
            f"({counts['invalid']} invalid)"
        )
    print(f"Figure: {figure_path}")
    print(f"PDF: {pdf_path}")
    print(f"Points CSV: {output_dir / args.points_name}")
    print(f"Invalid CSV: {output_dir / args.invalid_name}")
    print(f"Summary CSV: {output_dir / args.summary_name}")


if __name__ == "__main__":
    main()
