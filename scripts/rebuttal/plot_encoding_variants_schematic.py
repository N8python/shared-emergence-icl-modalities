#!/usr/bin/env python3
"""Rebuttal figure: schematic of the language and image encoding variants.

One demonstration (input 10110100, output 00101101) is drawn under each of the
five language encodings and five image encodings of the encoding-robustness
experiment. Bits are dark (0) or light (1) cells; separator cells are filled
with a separator colour; coloured rules under each strip mark input and output.
Language cells carry an example symbol assignment (symbols are resampled on
every trial). Purely illustrative: no experimental data is read.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT, OUTPUT = "10110100", "00101101"
ZERO, ONE = "#161616", "#F4F4F4"
SEP_A, SEP_B = "#FFD166", "#8FD3B0"
IN_RULE, OUT_RULE, INK, EDGE = "#2B5FD6", "#D64545", "#111827", "#6B7280"

# (row label, layout, symbols). Layout items: "I" input, "O" output, "A"/"B" separators.
LANGUAGE = [
    ("Reference", "IOA", {"0": "5", "1": "6", "A": "3"}),
    ("Two separators", "IBOA", {"0": "2", "1": "9", "A": "4", "B": "7"}),
    ("No separator", "IO", {"0": "8", "1": "1"}),
    ("Non-digit tokens", "IOA", {"0": "of", "1": "to", "A": ","}),
    ("Non-digit,\ntwo separators", "IBOA", {"0": "in", "1": "is", "A": ".", "B": "a"}),
]
# Image separators are spans: (kind, width in pixels).
IMAGE = [
    ("Reference\n(left separator)", [("A", 16), ("I", 8), ("O", 8)]),
    ("Right separator", [("I", 8), ("O", 8), ("A", 16)]),
    ("Middle separator", [("I", 8), ("A", 16), ("O", 8)]),
    ("Left,\ntwo separators", [("A", 8), ("I", 8), ("B", 8), ("O", 8)]),
    ("Right,\ntwo separators", [("I", 8), ("B", 8), ("O", 8), ("A", 8)]),
]


def draw_strip(axis, y, cells, cell_w, labels=None):
    """cells: list of (kind, bit-or-None). Returns nothing; draws at row y."""
    x = 0.0
    runs = {"I": [], "O": []}
    for index, (kind, bit) in enumerate(cells):
        if kind in "IO":
            face = ONE if bit == "1" else ZERO
            runs[kind].append(x)
        else:
            face = SEP_A if kind == "A" else SEP_B
        axis.add_patch(Rectangle((x, y), cell_w, 1, facecolor=face, edgecolor=EDGE, linewidth=0.5))
        if labels is not None:
            text = labels[bit] if kind in "IO" else labels[kind]
            colour = "white" if (kind in "IO" and bit == "0") else INK
            axis.text(x + cell_w / 2, y + 0.5, text, ha="center", va="center", fontsize=6.3, color=colour)
        x += cell_w
    for kind, colour in (("I", IN_RULE), ("O", OUT_RULE)):
        if runs[kind]:
            axis.plot([runs[kind][0] + 0.08, runs[kind][-1] + cell_w - 0.08], [y - 0.2, y - 0.2],
                      color=colour, linewidth=2.2, solid_capstyle="butt")
    return x


def expand(layout):
    cells = []
    for item in layout:
        kind, width = (item, None) if isinstance(item, str) else item
        if kind == "I":
            cells += [("I", b) for b in INPUT]
        elif kind == "O":
            cells += [("O", b) for b in OUTPUT]
        else:
            cells += [(kind, None)] * (width or 1)
    return cells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-stem", type=Path, default=REPO_ROOT / "results_rebuttal/figures/encoding_variants_schematic")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, (ax_lang, ax_img) = plt.subplots(2, 1, figsize=(7.0, 3.7), constrained_layout=True)
    pitch = 2.1

    lang_w = 32 / 18
    for row, (label, layout, symbols) in enumerate(LANGUAGE):
        y = -row * pitch
        end = draw_strip(ax_lang, y, expand(layout), lang_w, labels=symbols)
        if "A" in layout:
            ax_lang.text(end + 0.35, y + 0.5, "next demonstration …", ha="left", va="center", fontsize=6.3, color=EDGE, style="italic")
        else:
            ax_lang.text(end + 0.35, y + 0.5, "next demonstration follows directly …", ha="left", va="center", fontsize=6.3, color=EDGE, style="italic")
        ax_lang.text(-0.6, y + 0.5, label, ha="right", va="center", fontsize=8, color=INK, linespacing=0.95)
    ax_lang.set_title("Language (Qwen3-14B): one demonstration, one token per cell", fontsize=9, loc="left", color=INK)

    for row, (label, layout) in enumerate(IMAGE):
        y = -row * pitch
        draw_strip(ax_img, y, expand(layout), 1.0)
        ax_img.text(-0.6, y + 0.5, label, ha="right", va="center", fontsize=8, color=INK, linespacing=0.95)
    ax_img.set_title("Image (ImageGPT-large): one demonstration, one 32-pixel row", fontsize=9, loc="left", color=INK)

    for axis, rows in ((ax_lang, len(LANGUAGE)), (ax_img, len(IMAGE))):
        axis.set_xlim(-8.2, 44)
        axis.set_ylim(-(rows - 1) * pitch - 0.55, 1.25)
        axis.set_aspect("equal")
        axis.axis("off")

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=ZERO, edgecolor=EDGE, linewidth=0.5, label="bit 0"),
        Rectangle((0, 0), 1, 1, facecolor=ONE, edgecolor=EDGE, linewidth=0.5, label="bit 1"),
        Rectangle((0, 0), 1, 1, facecolor=SEP_A, edgecolor=EDGE, linewidth=0.5, label="separator"),
        Rectangle((0, 0), 1, 1, facecolor=SEP_B, edgecolor=EDGE, linewidth=0.5, label="second separator"),
        plt.Line2D([], [], color=IN_RULE, linewidth=2.2, label="input"),
        plt.Line2D([], [], color=OUT_RULE, linewidth=2.2, label="output"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=6, frameon=False, fontsize=7.5, handlelength=1.3, columnspacing=1.4)
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output_stem.with_suffix(f".{suffix}"), dpi=300)
    print("wrote", args.output_stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
