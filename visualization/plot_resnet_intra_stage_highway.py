#!/usr/bin/env python3
"""
Generate publication-quality schematic contrasting:
(a) Plain convolutional sequence without residual connections:
    Multiplicative gradient backpropagation leading to exponential decay.
(b) Intra-stage residual highway (e.g., Stage 3 of ResNet-34: 6 identical residual blocks):
    Direct identity shortcut carrying unattenuated backpropagated gradient.

Enhanced with larger, publication-grade font sizes (headers ~16pt, section titles ~13.5pt,
text and formulas ~11-12pt).
"""

import os

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import argparse
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "figure.autolayout": False,
    }
)

DEFAULT_OUTPUT_DIR = Path("ctf2_article/figures/schematics")


def main(output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    """Render the schematic as PDF and PNG into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Title (17.5 pt bold)
    fig = plt.figure(figsize=(19.0, 9.2), dpi=300)
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 1.18], hspace=0.28, left=0.02, right=0.98, top=0.93, bottom=0.04
    )
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    fig.suptitle(
        r"$\mathbf{Intra-Stage\ Gradient\ Highway:\ Exact\ Residual\ Formulation\ (e.g.,\ Stage\ 3:\ 6\ ResBlocks)}$",
        fontsize=17.5,
        weight="bold",
        color="#111827",
        y=0.982,
    )

    C_PLAIN = "#DC2626"
    C_RES = "#D97706"

    # ==============================================================================
    # PANEL (A): PLAIN DEEP SEQUENCE (Without Residual Shortcuts)
    # ==============================================================================
    ax1.set_xlim(0, 168)
    ax1.set_ylim(-0.8, 14.6)
    ax1.axis("off")

    ax1.text(
        0,
        13.4,
        r"$\mathbf{(a)\ Plain\ Convolutional\ Sequence\ (Gradient\ Backpropagation)}$",
        fontsize=15.0,
        weight="bold",
        color="#991B1B",
    )
    ax1.text(
        0,
        11.2,
        r"Gradient decays through chained weight Jacobians: $\mathbf{\frac{\partial \mathcal{E}}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{E}}{\partial \mathbf{x}_L} \prod_{i=l}^{L-1} \frac{\partial \mathbf{x}_{i+1}}{\partial \mathbf{x}_i}}$ (Earlier units in deep sequences starve of signal)",
        fontsize=14.0,
        color="#7F1D1D",
    )

    # Context banner at right showing incoming gradient from deeper layers
    box_in1 = FancyBboxPatch(
        (143.0, 1.3),
        23.0,
        7.6,
        boxstyle="round,pad=0.2",
        facecolor="#F3F4F6",
        edgecolor="#9CA3AF",
        lw=1.3,
    )
    ax1.add_patch(box_in1)
    ax1.text(
        154.5,
        7.3,
        "Incoming Gradient",
        ha="center",
        va="center",
        fontsize=13.5,
        color="#374151",
        weight="bold",
    )
    ax1.text(
        154.5,
        5.3,
        r"$\mathbf{\frac{\partial \mathcal{E}}{\partial \mathbf{x}_L}}$",
        ha="center",
        va="center",
        fontsize=20.0,
        color="#1F2937",
    )
    ax1.text(
        154.5,
        3.6,
        r"(from Decoder / $L$)",
        ha="center",
        va="center",
        fontsize=12.0,
        color="#4B5563",
    )
    ax1.text(
        154.5,
        2.2,
        r"Magnitude: $\mathbf{1.0}$",
        ha="center",
        va="center",
        fontsize=12.5,
        color="#4B5563",
    )

    # 6 sequential convolutional blocks: width 14.5, pitch 23.0 -> gap 8.5 units
    # Box x-coords: 5.0, 28.0, 51.0, 74.0, 97.0, 120.0
    plain_blocks = [
        {
            "name": r"$\mathbf{Block\ 1}$",
            "dim": r"$\mathbf{32\times32\times256}$",
            "x": 5.0,
            "grad": r"$\mathbf{\approx 0.04}$",
        },
        {
            "name": r"$\mathbf{Block\ 2}$",
            "dim": r"$\mathbf{32\times32\times256}$",
            "x": 28.0,
            "grad": r"$\mathbf{\approx 0.08}$",
        },
        {
            "name": r"$\mathbf{Block\ 3}$",
            "dim": r"$\mathbf{32\times32\times256}$",
            "x": 51.0,
            "grad": r"$\mathbf{\approx 0.17}$",
        },
        {
            "name": r"$\mathbf{Block\ 4}$",
            "dim": r"$\mathbf{32\times32\times256}$",
            "x": 74.0,
            "grad": r"$\mathbf{\approx 0.35}$",
        },
        {
            "name": r"$\mathbf{Block\ 5}$",
            "dim": r"$\mathbf{32\times32\times256}$",
            "x": 97.0,
            "grad": r"$\mathbf{\approx 0.58}$",
        },
        {
            "name": r"$\mathbf{Block\ 6}$",
            "dim": r"$\mathbf{32\times32\times256}$",
            "x": 120.0,
            "grad": r"$\mathbf{\approx 0.82}$",
        },
    ]

    for b in plain_blocks:
        box = FancyBboxPatch(
            (b["x"], 1.3),
            14.5,
            7.6,
            boxstyle="round,pad=0.2",
            facecolor="#FAFAFA",
            edgecolor="#9CA3AF",
            lw=1.3,
        )
        ax1.add_patch(box)
        ax1.text(
            b["x"] + 7.25, 7.4, b["name"], ha="center", va="center", fontsize=13.5, color="#1F2937"
        )
        ax1.text(
            b["x"] + 7.25, 5.8, b["dim"], ha="center", va="center", fontsize=12.5, color="#4B5563"
        )
        ax1.text(
            b["x"] + 7.25,
            4.3,
            "Gradient:",
            ha="center",
            va="center",
            fontsize=12.5,
            color="#DC2626",
            weight="bold",
        )
        ax1.text(
            b["x"] + 7.25,
            2.6,
            b["grad"],
            ha="center",
            va="center",
            fontsize=17.5,
            color="#DC2626",
            weight="bold",
        )

    # Backward gradient arrows (right-to-left) at y=5.1
    decay_lws = [1.0, 1.5, 2.2, 3.0, 4.0, 5.0]

    # Incoming to Block 6
    ax1.annotate(
        "",
        xy=(134.75, 5.1),
        xytext=(142.75, 5.1),
        arrowprops={"arrowstyle": "->", "lw": decay_lws[5], "color": C_PLAIN},
    )

    # Between blocks
    for i in reversed(range(len(plain_blocks) - 1)):
        x_from = plain_blocks[i + 1]["x"] - 0.25
        x_to = plain_blocks[i]["x"] + 14.75
        ax1.annotate(
            "",
            xy=(x_to, 5.1),
            xytext=(x_from, 5.1),
            arrowprops={"arrowstyle": "->", "lw": decay_lws[i], "color": C_PLAIN},
        )

    # Out of Block 1
    ax1.annotate(
        "",
        xy=(0.0, 5.1),
        xytext=(4.75, 5.1),
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": C_PLAIN},
    )

    ax1.text(
        70.0,
        0.2,
        r"$\longleftarrow\ \mathbf{Backward\ gradient\ attenuates\ multiplicatively\ across\ successive\ weight\ matrices}$",
        ha="center",
        va="center",
        fontsize=12.5,
        color="#991B1B",
        weight="bold",
    )

    # ==============================================================================
    # PANEL (B): INTRA-STAGE RESIDUAL HIGHWAY (Exact He et al. Formulation)
    # ==============================================================================
    ax2.set_xlim(0, 168)
    ax2.set_ylim(-0.9, 16.5)
    ax2.axis("off")

    # Elevated headers
    ax2.text(
        0,
        15.3,
        r"$\mathbf{(b)\ Intra-Stage\ Residual\ Highway\ (Gradient\ Backpropagation)}$",
        fontsize=15.0,
        weight="bold",
        color="#1E40AF",
    )
    ax2.text(
        0,
        13.3,
        r"Additive unrolling across units: $\mathbf{x}_L = \mathbf{x}_l + \sum_{i=l}^{L-1} \mathcal{F}_i(\mathbf{x}_i) \longrightarrow \mathbf{\frac{\partial \mathcal{E}}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{E}}{\partial \mathbf{x}_L} \cdot \left( \mathbf{I} + \frac{\partial}{\partial \mathbf{x}_l}\sum_{i=l}^{L-1}\mathcal{F}_i \right) = \frac{\partial \mathcal{E}}{\partial \mathbf{x}_L} + \mathbf{Residual}}$",
        fontsize=14.0,
        color="#1E3A8A",
    )

    # Context banner at right
    box_in2 = FancyBboxPatch(
        (143.0, 1.3),
        23.0,
        7.6,
        boxstyle="round,pad=0.2",
        facecolor="#EFF6FF",
        edgecolor="#2563EB",
        lw=1.4,
    )
    ax2.add_patch(box_in2)
    ax2.text(
        154.5,
        7.3,
        "Incoming Gradient",
        ha="center",
        va="center",
        fontsize=13.5,
        color="#1E40AF",
        weight="bold",
    )
    ax2.text(
        154.5,
        5.3,
        r"$\mathbf{\frac{\partial \mathcal{E}}{\partial \mathbf{x}_L}}$",
        ha="center",
        va="center",
        fontsize=20.0,
        color="#1E3A8A",
    )
    ax2.text(
        154.5,
        3.6,
        r"(from Decoder / $L$)",
        ha="center",
        va="center",
        fontsize=12.0,
        color="#1E40AF",
    )
    ax2.text(
        154.5,
        2.2,
        r"Unit gradient: $\mathbf{1.0}$",
        ha="center",
        va="center",
        fontsize=12.5,
        color="#2563EB",
    )

    res_blocks = [
        {"name": r"$\mathbf{ResBlock\ 1}$", "dim": r"$\mathbf{32\times32\times256}$", "x": 5.0},
        {"name": r"$\mathbf{ResBlock\ 2}$", "dim": r"$\mathbf{32\times32\times256}$", "x": 28.0},
        {"name": r"$\mathbf{ResBlock\ 3}$", "dim": r"$\mathbf{32\times32\times256}$", "x": 51.0},
        {"name": r"$\mathbf{ResBlock\ 4}$", "dim": r"$\mathbf{32\times32\times256}$", "x": 74.0},
        {"name": r"$\mathbf{ResBlock\ 5}$", "dim": r"$\mathbf{32\times32\times256}$", "x": 97.0},
        {"name": r"$\mathbf{ResBlock\ 6}$", "dim": r"$\mathbf{32\times32\times256}$", "x": 120.0},
    ]

    for _i, b in enumerate(res_blocks):
        box = FancyBboxPatch(
            (b["x"], 1.3),
            14.5,
            7.6,
            boxstyle="round,pad=0.2",
            facecolor="#DBEAFE",
            edgecolor="#2563EB",
            lw=1.4,
        )
        ax2.add_patch(box)
        ax2.text(
            b["x"] + 7.25, 7.4, b["name"], ha="center", va="center", fontsize=13.5, color="#1E3A8A"
        )
        ax2.text(
            b["x"] + 7.25, 5.8, b["dim"], ha="center", va="center", fontsize=12.5, color="#1E40AF"
        )
        ax2.text(
            b["x"] + 7.25,
            4.3,
            "Receives gradient:",
            ha="center",
            va="center",
            fontsize=10.5,
            color="#047857",
            weight="bold",
        )
        ax2.text(
            b["x"] + 7.25,
            2.5,
            r"$\mathbf{\geq 1.0 \times \frac{\partial \mathcal{E}}{\partial \mathbf{x}_L}}$",
            ha="center",
            va="center",
            fontsize=15.5,
            color="#047857",
            weight="bold",
        )

    # Split node between ResBlock 6 and Incoming Gradient
    split_x = 138.5
    ax2.plot([142.8, split_x], [5.1, 5.1], color="#2563EB", lw=2.4)
    ax2.annotate(
        "",
        xy=(134.8, 5.1),
        xytext=(split_x, 5.1),
        arrowprops={"arrowstyle": "->", "lw": 2.4, "color": "#2563EB"},
    )
    circ_split = patches.Circle((split_x, 5.1), 0.5, facecolor="#1E40AF", zorder=5)
    ax2.add_patch(circ_split)

    # Addition nodes in the 8.5-unit gaps between ResBlocks:
    # For ResBlock 6 (x=120.0): gap before it is between 111.5 and 120.0 -> mid cx = 115.75
    # For ResBlock 5 (x=97.0): mid cx = 92.75
    # For ResBlock 4 (x=74.0): mid cx = 69.75
    # For ResBlock 3 (x=51.0): mid cx = 46.75
    # For ResBlock 2 (x=28.0): mid cx = 23.75
    # For ResBlock 1 (x=5.0): exit Stage 3 after ResBlock 1 -> cx = 2.5
    cx_gaps = [115.75, 92.75, 69.75, 46.75, 23.75, 2.5]

    # Orange highway up
    ax2.plot([split_x, split_x], [5.1, 10.3], color=C_RES, lw=2.4)
    # Orange highway across: begins exactly at leftmost drop line cx = 2.5
    ax2.plot([cx_gaps[-1], split_x], [10.3, 10.3], color=C_RES, lw=2.4)

    for idx, cx in enumerate(cx_gaps):
        # Addition circle at y=5.1
        r_circ = 1.1
        circ = patches.Circle(
            (cx, 5.1), r_circ, facecolor="#DBEAFE", edgecolor="#2563EB", lw=1.4, zorder=5
        )
        ax2.add_patch(circ)
        ax2.text(
            cx,
            5.1,
            "+",
            ha="center",
            va="center",
            color="#1E40AF",
            weight="bold",
            fontsize=15.0,
            zorder=6,
        )

        # Orange drop terminating at top of circle
        ax2.annotate(
            "",
            xy=(cx, 5.1 + r_circ),
            xytext=(cx, 10.3),
            arrowprops={"arrowstyle": "->", "lw": 2.4, "color": C_RES},
            zorder=4,
        )
        # +I label placed midway between drop line (cx) and the left block edge (cx - 4.25)
        # cx - 1.8 gives plenty of breathing room on both sides (1.8 from drop line, 2.45 from box edge)
        ax2.text(
            cx - 1.8,
            8.0,
            r"$+\mathbf{I}$",
            fontsize=14.0,
            weight="bold",
            color="#B45309",
            ha="center",
            va="center",
        )

        # Blue arrow from right (coming out of ResBlock [5 - idx])
        b_curr = res_blocks[5 - idx]
        x_from = b_curr["x"] - 0.25
        ax2.annotate(
            "",
            xy=(cx + r_circ, 5.1),
            xytext=(x_from, 5.1),
            arrowprops={
                "arrowstyle": "->",
                "lw": 2.4,
                "color": "#2563EB",
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=4,
        )

        # Blue arrow to left (entering ResBlock [5 - idx - 1] or exiting stage)
        if idx == 5:
            # Exiting Stage 3 after ResBlock 1
            x_to = 0.0
        else:
            # Entering block to the left
            b_prev = res_blocks[5 - idx - 1]
            x_to = b_prev["x"] + 14.75
        ax2.annotate(
            "",
            xy=(x_to, 5.1),
            xytext=(cx - r_circ, 5.1),
            arrowprops={
                "arrowstyle": "->",
                "lw": 2.4,
                "color": "#2563EB",
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=4,
        )

    # Central label card on the identity highway
    ax2.text(
        69.75,
        11.25,
        r"$\mathbf{Intra-Stage\ Identity\ Highway:\ Exact\ Dimension\ Match\ (\mathbf{I}\in\mathbb{R}^{D\times D},\ D=32\times32\times256)}$",
        ha="center",
        va="center",
        fontsize=13.0,
        weight="bold",
        color="#B45309",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#FFFBEB",
            "edgecolor": "#D97706",
            "lw": 1.3,
        },
    )

    # Bottom note clarifying boundary condition and decoder origin
    ax2.text(
        84.0,
        -0.4,
        r"$\mathbf{Architectural\ Reality:}$ Inside any stage, channels & spatial resolution are constant; the identity shortcut preserves the full gradient without decay. Across different stages (e.g. Stage 1 to 2), stride-2 projections $W_s$ adapt tensor shapes.",
        ha="center",
        va="center",
        fontsize=11.5,
        color="#1E3A8A",
    )

    pdf_path = output_dir / "resnet_intra_stage_highway.pdf"
    png_path = output_dir / "resnet_intra_stage_highway.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved resnet_intra_stage_highway figure to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the generated PDF/PNG schematic",
    )
    main(parser.parse_args().output_dir)
