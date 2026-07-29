#!/usr/bin/env python3
"""Recreate the paper's six-modality curve figure and summary table at 128 trials/task."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BLUE = "#2563EB"
RED = "#DC2626"
GRID = "#D1D5DB"


@dataclass(frozen=True)
class Panel:
    run_key: str
    title: str


PANELS = (
    Panel("imagegpt", "Image (ImageGPT-large)"),
    Panel("timesfm", "Time Series (TimesFM-2.5)"),
    Panel("qwen3", "Language (Qwen3-14B)"),
    Panel("nextterm", "Integer Sequence (NextTerm-440M)"),
    Panel("progen2", "Protein (ProGen2-base)"),
    Panel("evo2", "Genome (Evo2-40B)"),
)

TABLE_ROWS = (
    ("qwen3", "Language", "Qwen3-14B"),
    ("evo2", "Genome", "Evo2-40B"),
    ("nextterm", "Integer seq.", "NextTerm-440M"),
    ("imagegpt", "Image", "ImageGPT-large"),
    ("timesfm", "Time-series", "TimesFM-2.5"),
    ("progen2", "Protein", "ProGen2-base"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-summary", type=Path, required=True)
    parser.add_argument("--best-shot-summary", type=Path, required=True)
    parser.add_argument("--pvalues", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_and_index(
    cell_rows: list[dict[str, str]],
) -> dict[tuple[str, str], list[dict[str, str]]]:
    expected = {panel.run_key for panel in PANELS}
    selected = [
        row
        for row in cell_rows
        if row["run_key"] in expected and row["canonical"] == "True"
    ]
    lookup: dict[tuple[str, str], list[dict[str, str]]] = {}
    for run_key in expected:
        for condition in ("clean", "deranged"):
            rows = [
                row
                for row in selected
                if row["run_key"] == run_key and row["condition"] == condition
            ]
            rows.sort(key=lambda row: int(row["shot"]))
            if not rows:
                raise ValueError(f"Missing canonical {run_key}/{condition} rows")
            for row in rows:
                if (
                    int(row["tasks"]) != 100
                    or int(row["trials_per_task"]) != 128
                    or int(row["total_predictions"]) != 12_800
                ):
                    raise ValueError(f"Unexpected evaluation size in {row}")
            lookup[(run_key, condition)] = rows

        clean_shots = [row["shot"] for row in lookup[(run_key, "clean")]]
        deranged_shots = [row["shot"] for row in lookup[(run_key, "deranged")]]
        if clean_shots != deranged_shots:
            raise ValueError(f"Clean/deranged shot grids differ for {run_key}")
    return lookup


def style_axis(ax: plt.Axes, *, left: bool, bottom: bool) -> None:
    ax.set_ylim(0.0, 0.55)
    ax.set_yticks(np.arange(0.0, 0.51, 0.1))
    ax.grid(True, color=GRID, alpha=0.55, linewidth=0.7)
    ax.set_axisbelow(True)
    if left:
        ax.set_ylabel("Exact Accuracy", fontsize=11)
    else:
        ax.tick_params(labelleft=False)
    if bottom:
        ax.set_xlabel("In-Context Examples", fontsize=11)
    ax.tick_params(labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#111827")


def draw_panels(
    axes: list[plt.Axes] | np.ndarray,
    lookup: dict[tuple[str, str], list[dict[str, str]]],
) -> None:
    flat_axes = list(np.asarray(axes, dtype=object).flat)
    for index, (panel, ax) in enumerate(zip(PANELS, flat_axes, strict=True)):
        clean_rows = lookup[(panel.run_key, "clean")]
        deranged_rows = lookup[(panel.run_key, "deranged")]
        shots = [int(row["shot"]) for row in clean_rows]
        x = np.arange(len(shots), dtype=float)

        clean_mean = np.asarray(
            [float(row["mean_accuracy"]) for row in clean_rows], dtype=float
        )
        clean_se = np.asarray(
            [float(row["accuracy_task_cluster_se"]) for row in clean_rows],
            dtype=float,
        )
        deranged_mean = np.asarray(
            [float(row["mean_accuracy"]) for row in deranged_rows], dtype=float
        )
        deranged_se = np.asarray(
            [float(row["accuracy_task_cluster_se"]) for row in deranged_rows],
            dtype=float,
        )

        ax.errorbar(
            x,
            clean_mean,
            yerr=clean_se,
            color=BLUE,
            marker="o",
            markersize=4.2,
            linewidth=1.8,
            elinewidth=1.1,
            capsize=2.5,
            capthick=1.0,
            label="clean",
            zorder=3,
        )
        ax.errorbar(
            x,
            deranged_mean,
            yerr=deranged_se,
            color=RED,
            marker="s",
            markersize=3.9,
            linewidth=1.45,
            linestyle="--",
            elinewidth=1.0,
            capsize=2.5,
            capthick=1.0,
            label="deranged",
            zorder=4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([str(shot) for shot in shots])
        ax.set_title(panel.title, fontsize=10.5, pad=5)
        style_axis(ax, left=index in (0, 3), bottom=index >= 3)
        if index == 0:
            ax.legend(
                loc="upper left",
                fontsize=8.5,
                frameon=True,
                framealpha=0.95,
                borderpad=0.35,
                handlelength=1.7,
                labelspacing=0.25,
            )


def format_p(value: float) -> str:
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / (10.0**exponent)
    return rf"${mantissa:.2g}\times10^{{{exponent}}}$"


def summary_rows(
    best_rows: list[dict[str, str]],
    pvalue_rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[list[str]]]:
    best_by_key = {
        row["run_key"]: row
        for row in best_rows
        if row["canonical"] == "True"
    }
    pvalue_by_key = {row["run_key"]: row for row in pvalue_rows}
    machine_rows: list[dict[str, object]] = []
    display_rows: list[list[str]] = []
    for run_key, modality, model in TABLE_ROWS:
        if run_key not in best_by_key:
            raise ValueError(f"Missing canonical best-shot row for {run_key}")
        if run_key not in pvalue_by_key:
            raise ValueError(f"Missing highest-shot p-value row for {run_key}")
        row = best_by_key[run_key]
        pvalue_row = pvalue_by_key[run_key]
        clean = float(row["best_clean_accuracy"])
        deranged = float(row["best_deranged_accuracy"])
        gap = float(row["best_accuracy_gap"])
        z_gap = float(row["z_gap_unpaired"])
        clean_shot = int(row["best_clean_shot"])
        deranged_shot = int(row["best_deranged_shot"])
        highest_shot = int(pvalue_row["highest_shot"])
        raw_p = float(pvalue_row["one_sided_exact_task_swap_p"])
        holm_p = float(pvalue_row["holm_p_6_tests"])
        machine_rows.append(
            {
                "run_key": run_key,
                "modality": modality,
                "model": model,
                "best_clean_accuracy": clean,
                "best_clean_shot": clean_shot,
                "best_deranged_accuracy": deranged,
                "best_deranged_shot": deranged_shot,
                "gap": gap,
                "z_gap_unpaired": z_gap,
                "p_test_highest_shot": highest_shot,
                "one_sided_exact_paired_task_p": raw_p,
                "holm_p_6_modalities": holm_p,
            }
        )
        display_rows.append(
            [
                modality,
                model,
                f"{100 * clean:.1f}% ({clean_shot})",
                f"{100 * deranged:.1f}% ({deranged_shot})",
                f"{100 * gap:.1f}%",
                f"{z_gap:.2f}",
                format_p(raw_p),
            ]
        )
    return machine_rows, display_rows


def draw_table(ax: plt.Axes, display_rows: list[list[str]]) -> None:
    ax.axis("off")
    headers = [
        "Modality",
        "Model",
        "Best clean\nacc. % ($n$)",
        "Best deranged\nacc. % ($n$)",
        "Gap",
        "z-gap",
        "$p_{\\rm task}$\nat $n_{\\max}$",
    ]
    table = ax.table(
        cellText=display_rows,
        colLabels=headers,
        cellLoc="left",
        colLoc="left",
        colWidths=[0.13, 0.18, 0.17, 0.18, 0.09, 0.07, 0.18],
        loc="center",
        edges="horizontal",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.48)

    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor("#111827")
        cell.set_linewidth(0.45)
        cell.set_facecolor("white")
        if row_index == 0:
            cell.set_text_props(weight="bold")
            cell.set_height(cell.get_height() * 1.35)
            if col_index == 2:
                cell.get_text().set_color(BLUE)
            elif col_index == 3:
                cell.get_text().set_color(RED)
        elif col_index >= 2:
            cell.get_text().set_ha("center")

    ax.text(
        0.5,
        -0.04,
        "Best clean and deranged shots are selected independently; z-gap uses "
        "the paper's unpaired task-cluster SE. Final column: exact one-sided "
        "within-task condition-swap p at the maximum shot count. Each p-value "
        "tests a pre-specified modality-specific hypothesis and is unadjusted.",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#374151",
        transform=ax.transAxes,
    )


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tex_escape(value: str) -> str:
    return value.replace("%", r"\%")


def write_latex_table(path: Path, display_rows: list[list[str]]) -> None:
    lines = [
        "% Generated from the validated 128-trial-per-task replication.",
        r"\begin{table}[!htbp]",
        r"    \centering",
        r"    \small",
        r"    \setlength{\tabcolsep}{5pt}",
        r"    \renewcommand{\arraystretch}{1.03}",
        r"    \begin{tabular}{@{}llccccc@{}}",
        r"        \toprule",
        r"        \textbf{Modality} & \textbf{Model} & "
        r"\textcolor[HTML]{2563EB}{\textbf{Best clean}} & "
        r"\textcolor[HTML]{DC2626}{\textbf{Best deranged}} & "
        r"\textbf{Gap} & \textbf{z-gap} & "
        r"\textbf{$p_{\rm task}$} \\",
        r"        & & \textbf{acc.\%\ ($n$)} & "
        r"\textbf{acc.\%\ ($n$)} & & & \textbf{at $n_{\max}$} \\",
        r"        \midrule",
    ]
    for modality, model, clean, deranged, gap, z_gap, p_values in display_rows:
        lines.append(
            "        "
            + " & ".join(
                tex_escape(value)
                for value in (
                    modality,
                    model,
                    clean,
                    deranged,
                    gap,
                    z_gap,
                    p_values,
                )
            )
            + r" \\"
        )
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"    \caption{Per-modality summary from the 128-trial-per-task "
            r"replication. Best clean and deranged shot counts are selected "
            r"independently. Error estimates and z-gap use tasks as the "
            r"sampling unit. The final column reports the unadjusted "
            r"$p$-value from an exact one-sided within-task condition-swap "
            r"randomization test of clean $>$ deranged at each modality's "
            r"maximum shot count; each value tests a pre-specified, "
            r"modality-specific hypothesis.}",
            r"    \label{tab:modality-summary-128}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cell_rows = read_csv(args.cell_summary)
    best_rows = read_csv(args.best_shot_summary)
    pvalue_rows = read_csv(args.pvalues)
    lookup = validate_and_index(cell_rows)
    machine_rows, display_rows = summary_rows(best_rows, pvalue_rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    plot_figure, plot_axes = plt.subplots(
        2,
        3,
        figsize=(15, 8.1),
        sharey=True,
        gridspec_kw={"hspace": 0.30, "wspace": 0.12},
    )
    draw_panels(plot_axes, lookup)
    plot_figure.savefig(
        args.output_dir / "modalities_clean_vs_deranged_128.pdf",
        bbox_inches="tight",
    )
    plot_figure.savefig(
        args.output_dir / "modalities_clean_vs_deranged_128.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(plot_figure)

    combined = plt.figure(figsize=(15, 10.8))
    grid = combined.add_gridspec(
        3,
        3,
        height_ratios=(1.0, 1.0, 0.82),
        hspace=0.31,
        wspace=0.12,
    )
    combined_axes = [
        combined.add_subplot(grid[row, column])
        for row in range(2)
        for column in range(3)
    ]
    draw_panels(combined_axes, lookup)
    table_axis = combined.add_subplot(grid[2, :])
    draw_table(table_axis, display_rows)
    combined.suptitle(
        "Clean versus deranged few-shot ICL — 128 trials per task",
        fontsize=15,
        y=0.985,
    )
    combined.subplots_adjust(top=0.925, bottom=0.055, left=0.055, right=0.99)
    combined.savefig(
        args.output_dir / "modalities_clean_vs_deranged_with_table_128.pdf",
        bbox_inches="tight",
    )
    combined.savefig(
        args.output_dir / "modalities_clean_vs_deranged_with_table_128.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(combined)

    write_summary_csv(
        args.output_dir / "modality_summary_128.csv",
        machine_rows,
    )
    write_latex_table(
        args.output_dir / "modality_summary_128.tex",
        display_rows,
    )

    print("Generated 128-trial graph, combined preview, CSV, and LaTeX table.")
    for name in (
        "modalities_clean_vs_deranged_128.pdf",
        "modalities_clean_vs_deranged_128.png",
        "modalities_clean_vs_deranged_with_table_128.pdf",
        "modalities_clean_vs_deranged_with_table_128.png",
        "modality_summary_128.csv",
        "modality_summary_128.tex",
    ):
        print(args.output_dir / name)


if __name__ == "__main__":
    main()
