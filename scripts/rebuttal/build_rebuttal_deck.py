#!/usr/bin/env python3
"""Build the lab-meeting deck for the rebuttal encoding experiments (.pptx).

Needs python-pptx. Figures come from --figures-dir; control numbers from
results_rebuttal/analysis/lm_controls_summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

REPO_ROOT = Path(__file__).resolve().parents[2]
BG, INK, MUTED, ACCENT, TINT = "FAFAFA", "1F2937", "6B7280", "2563EB", "E8EEF7"
HEAD, BODY = "Merriweather", "Lato"
W, H, M = 13.333, 7.5, 0.6


def rgb(hex_color: str) -> RGBColor:
    return RGBColor.from_string(hex_color)


def text(slide, x, y, w, h, content, *, size=18, bold=False, color=INK, font=BODY, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, bullets=False, space_after=0):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    for attr in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(frame, attr, 0)
    lines = content if isinstance(content, list) else [content]
    for i, line in enumerate(lines):
        paragraph = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        paragraph.alignment = align
        paragraph.space_after = Pt(space_after)
        segments = line if isinstance(line, list) else [(line, bold)]
        if bullets:
            segments = [("•  ", False)] + segments
        for segment, is_bold in segments:
            run = paragraph.add_run()
            run.text = segment
            run.font.size = Pt(size)
            run.font.bold = is_bold
            run.font.name = font
            run.font.color.rgb = rgb(color)
    return box


def new_slide(prs, background=BG):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = rgb(background)
    return slide


def title(slide, content):
    text(slide, M, 0.45, W - 2 * M, 1.25, content, size=30, bold=True, font=HEAD, anchor=MSO_ANCHOR.TOP)


def figure(slide, path: Path, top: float, bottom: float, max_tile_px: int = 2000):
    """Place a figure as side-by-side tiles cut at blank columns.

    Google Slides downsamples any image wider than about 2046 px, so a wide
    multi-panel figure is cut into tiles that each stay under that width.
    """
    import numpy as np

    with Image.open(path) as image:
        image = image.convert("RGB")
        pixels = np.asarray(image)
        blank = (pixels.min(axis=(0, 2)) >= 250)
        n = -(-image.width // max_tile_px)
        cuts = [0]
        for k in range(1, n):
            target = k * image.width // n
            candidates = np.flatnonzero(blank)
            cuts.append(int(candidates[np.argmin(np.abs(candidates - target))]) if candidates.size else target)
        cuts.append(image.width)
        aspect = image.width / image.height
        max_w, max_h = W - 2 * M, bottom - top
        width = min(max_w, max_h * aspect)
        height = width / aspect
        left, y = (W - width) / 2, top + (max_h - height) / 2
        for i, (x0, x1) in enumerate(zip(cuts, cuts[1:])):
            tile = path.with_name(f"{path.stem}.tile{i}.png")
            image.crop((x0, 0, x1, image.height)).save(tile)
            slide.shapes.add_picture(str(tile), Inches(left + width * x0 / image.width), Inches(y),
                                     Inches(width * (x1 - x0) / image.width), Inches(height))
            tile.unlink()


def footnote(slide, content):
    text(slide, M, H - 0.62, W - 2 * M, 0.3, content, size=12, color=MUTED)


def table(slide, x, y, w, rows, col_widths, row_h=0.62, size=16, highlight_rows=()):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows)))
    tbl = shape._element.graphic.graphicData.tbl
    tbl[0][-1].text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"  # No Style, No Grid
    grid = shape.table
    grid.first_row = True
    grid.horz_banding = False
    for c, width in enumerate(col_widths):
        grid.columns[c].width = Inches(width)
    for r, row in enumerate(rows):
        grid.rows[r].height = Inches(row_h)
        for c, value in enumerate(row):
            cell = grid.cell(r, c)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = Inches(0.12)
            if r in highlight_rows:
                cell.fill.solid()
                cell.fill.fore_color.rgb = rgb(TINT)
            else:
                cell.fill.background()
            paragraph = cell.text_frame.paragraphs[0]
            paragraph.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT
            run = paragraph.add_run()
            run.text = value
            run.font.size = Pt(size)
            run.font.name = BODY
            run.font.bold = r == 0 or c == 0
            run.font.color.rgb = rgb(INK)
    return shape


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures-dir", type=Path, required=True)
    parser.add_argument("--controls", type=Path, default=REPO_ROOT / "results_rebuttal/analysis/lm_controls_summary.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", default="Controls: 64-shot results in; lower shot counts still running")
    args = parser.parse_args()
    controls = json.loads(args.controls.read_text())

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)

    # 1. Title
    slide = new_slide(prs, background=INK)
    text(slide, M + 0.4, 2.3, W - 2 * M - 0.8, 1.9, "Does the encoding matter?", size=44, bold=True, font=HEAD, color="FFFFFF")
    text(slide, M + 0.4, 3.75, W - 2 * M - 0.8, 0.9,
         "Encoding-resampling experiments for the ICLR 2027 rebuttal", size=22, color="E5E7EB")
    text(slide, M + 0.4, 5.9, W - 2 * M - 0.8, 0.5,
         "Convergent Emergence of In-Context Learning Across Modalities  ·  12 November 2026", size=14, color="9CA3AF")

    # 2. Setup
    slide = new_slide(prs)
    title(slide, "We re-ran language and image under four alternative encodings each")
    card_w = (W - 2 * M - 0.5) / 2
    cards = [
        ("Language  ·  Qwen3-14B, 64 shots", ["Reference: random digit symbols, one separator", "Two separators", "No separator",
                                             "Non-digit tokens (frequent-token pool)", "Non-digit tokens, two separators"]),
        ("Image  ·  ImageGPT-large, 31 shots", ["Reference: separator on the left", "Separator on the right", "Separator in the middle",
                                                "Two separators, left", "Two separators, right"]),
    ]
    for i, (head, items) in enumerate(cards):
        x = M + i * (card_w + 0.5)
        card = slide.shapes.add_shape(5, Inches(x), Inches(1.95), Inches(card_w), Inches(3.5))
        card.adjustments[0] = 0.04
        card.fill.solid()
        card.fill.fore_color.rgb = rgb(TINT)
        card.line.fill.background()
        card.shadow.inherit = False
        text(slide, x + 0.3, 2.2, card_w - 0.6, 0.45, head, size=20, bold=True, font=HEAD)
        text(slide, x + 0.3, 2.85, card_w - 0.6, 2.5, items, size=17, bullets=True, space_after=7)
    text(slide, M, 5.8, W - 2 * M, 0.9,
         [[("Each encoding: ", True), ("100 tasks × 128 trials, clean and label-shuffled (deranged), with its own random seed. "
                                       "A re-seeded reference run gives the noise ceiling.", False)]], size=17)

    # 3-5. Figures
    figures = [
        ("Under every encoding, language stays in the five-modality block and image stays out",
         "encoding_cross_modality.png",
         "Rows: encodings. Columns: other modalities’ published profiles. Hatched = not significant (one-sided permutation p ≥ 0.05, 1M permutations)."),
        ("Image task profiles sit at the seed-noise ceiling; language profiles drift further",
         "encoding_self_correlation.png",
         "Spearman ρ of the per-task clean − deranged effect between encodings. Noise ceiling = reference vs. new seed (0.86 language, 0.84 image)."),
        ("Layout leaves the image effect at 24–25 points; language ranges from 4 to 22",
         "encoding_effect_sizes.png",
         "Mean accuracy over 100 tasks. Brackets: 95% task-bootstrap interval of the clean − deranged effect."),
    ]
    for heading, name, note in figures:
        slide = new_slide(prs)
        title(slide, heading)
        figure(slide, args.figures_dir / name, 1.75, H - 0.8)
        footnote(slide, note)

    # 6-7. Controls, one slide each
    last = controls["64"]["runs"]
    literal, mistral = last["qwen3_14b_literal"], last["mistral7b_digits"]
    control_slides = [
        (f"Control 1: literal bitstrings lift Qwen3-14B from 30% to {100 * literal['clean']:.0f}% at 64 shots",
         "lm_control_literal.png",
         "Same model, same tasks; prompts are “00001111->00001111” lines instead of random digit symbols. "
         "Deranged baseline is unchanged; the task profile differs from the digit encoding’s."),
        (f"Control 2: Mistral 7B also learns in context, with a smaller effect (+{100 * mistral['gap']:.1f} pp) and a Qwen-like profile",
         "lm_control_mistral.png",
         "A second language model on the paper’s digit encoding, against published Qwen3-8B and Qwen3-14B. "
         "Llama 3.1 8B was tried first and dropped: its tokenizer merges digit runs."),
    ]
    for heading, name, note in control_slides:
        slide = new_slide(prs)
        title(slide, heading)
        figure(slide, args.figures_dir / name, 1.75, H - 0.8)
        footnote(slide, note)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    prs.save(args.output)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
