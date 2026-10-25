#!/usr/bin/env python3
"""
Generate publication-quality schematic of the exact smp.Unet(encoder_name="resnet34") architecture:
Accurately grounded in PyTorch torchvision.models.resnet34 and segmentation_models_pytorch:
  - Input Tile: 512x512x1
  - Stem Conv: Conv 7x7 (stride 2) -> 256x256x64, followed by MaxPool 3x3 (stride 2) -> 128x128x64.
  - Stage 1 (layer1, 3 ResBlocks): 128x128x64
  - Stage 2 (layer2, 4 ResBlocks): 64x64x128
  - Stage 3 (layer3, 6 ResBlocks): 32x32x256
  - Stage 4 (layer4, 3 ResBlocks, Bottleneck): 16x16x512
  - Decoder 5 (upsample 16x16 to 32x32 + concat Stage 3): out 256 ch | 32x32  <-- Exact 32x32 Skip from Stage 3!
  - Decoder 4 (upsample 32x32 to 64x64 + concat Stage 2): out 128 ch | 64x64  <-- Exact 64x64 Skip from Stage 2!
  - Decoder 3 (upsample 64x64 to 128x128 + concat Stage 1): out 64 ch | 128x128 <-- Exact 128x128 Skip from Stage 1!
  - Decoder 2 (upsample 128x128 to 256x256 + concat Stem Conv): out 32 ch | 256x256 <-- Exact 256x256 Skip from Stem Conv!
  - Decoder 1 (upsample 256x256 to 512x512): out 16 ch | 512x512
  - Segmentation Head (Conv 1x1): 512x512x1 output mask.

Mathematical Grounding:
  Residual block gradient vector calculus:
  Let x in R^D be the activation vector and E be the scalar objective function.
  Under the residual formulation y = F(x, W) + x, the chain rule yields:
    dE/dx = dE/dy * (dF/dx + I)
  where I in R^(D x D) is the identity matrix (or 1 in scalar notation), ensuring non-decaying gradient flow.
"""

import os

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import shutil
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "cm",
        "figure.dpi": 300,
    }
)

OUTPUT_DIR = Path("ctf2_article/figures/schematics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

fig = plt.figure(figsize=(10.8, 5.6), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 108)
ax.set_ylim(0, 56)
ax.axis("off")

# Title / Header
ax.text(
    54,
    53.8,
    "ResNet-34 U-Net Architecture: Exact smp.Unet Multi-Scale Topology",
    ha="center",
    va="top",
    fontsize=11.5,
    weight="bold",
    color="#111827",
)
ax.text(
    54,
    51.5,
    "Accurate PyTorch ResNet-34 stem, dimension-matched horizontal skips (256, 128, 64, 32), and 5 decoder stages",
    ha="center",
    va="top",
    fontsize=8.2,
    color="#4B5563",
    style="italic",
)

# Colors
C_ENC = "#2563EB"  # Encoder blue
C_ENC_BG = "#EFF6FF"
C_DEC = "#059669"  # Decoder emerald
C_DEC_BG = "#ECFDF5"
C_BOT = "#7C3AED"  # Stage 4 / Bottleneck purple
C_BOT_BG = "#F5F3FF"
C_RES_SKIP = "#D97706"  # Local residual skip (amber)
C_UNET_SKIP = "#1D4ED8"  # U-Net long skip (blue)
C_INPUT = "#475569"
C_CARD_BORDER = "#0284C7"
C_CARD_BG = "#F0F9FF"

# Stages Definition with EXACT Dimension-Matched Horizontal Heights:
# Level 256x256 (y=36.5): Stem Conv (256x256) <---> Decoder 2 (256x256)
# Level 128x128 (y=29.0): Stage 1   (128x128) <---> Decoder 3 (128x128)
# Level 64x64   (y=21.5): Stage 2   (64x64)   <---> Decoder 4 (64x64)
# Level 32x32   (y=14.0): Stage 3   (32x32)   <---> Decoder 5 (32x32)
# Bottleneck 16x16 (y=5.8): Stage 4 (Bottleneck) (16x16)
# Top 512x512 (y=44.0): Input Tile (512x512) and Decoder 1 (512x512) -> Output Mask (512x512)

matched_levels = [
    {
        "y": 36.5,
        "w": 7.6,
        "h": 5.2,
        "enc_x": 12.0,
        "enc_title": "Stem Conv",
        "enc_sub": "Conv 7x7 (s2)",
        "enc_desc": r"$64\text{ ch} \mid 256 \times 256$",
        "dec_x": 77.5,
        "dec_title": "Decoder 2",
        "dec_sub": "UpConv + Conv",
        "dec_desc": r"$32\text{ ch} \mid 256 \times 256$",
        "dim_label": r"$256 \times 256$",
    },
    {
        "y": 29.0,
        "w": 7.6,
        "h": 5.2,
        "enc_x": 19.5,
        "enc_title": "Stage 1",
        "enc_sub": "3 ResBlocks",
        "enc_desc": r"$64\text{ ch} \mid 128 \times 128$",
        "dec_x": 69.5,
        "dec_title": "Decoder 3",
        "dec_sub": "UpConv + Conv",
        "dec_desc": r"$64\text{ ch} \mid 128 \times 128$",
        "dim_label": r"$128 \times 128$",
    },
    {
        "y": 21.5,
        "w": 7.6,
        "h": 5.2,
        "enc_x": 27.0,
        "enc_title": "Stage 2",
        "enc_sub": "4 ResBlocks",
        "enc_desc": r"$128\text{ ch} \mid 64 \times 64$",
        "dec_x": 61.5,
        "dec_title": "Decoder 4",
        "dec_sub": "UpConv + Conv",
        "dec_desc": r"$128\text{ ch} \mid 64 \times 64$",
        "dim_label": r"$64 \times 64$",
    },
    {
        "y": 14.0,
        "w": 7.6,
        "h": 5.2,
        "enc_x": 34.5,
        "enc_title": "Stage 3",
        "enc_sub": "6 ResBlocks",
        "enc_desc": r"$256\text{ ch} \mid 32 \times 32$",
        "dec_x": 53.5,
        "dec_title": "Decoder 5",
        "dec_sub": "UpConv + Conv",
        "dec_desc": r"$256\text{ ch} \mid 32 \times 32$",
        "dim_label": r"$32 \times 32$",
    },
]

b_stage4 = {"x": 42.0, "y": 5.8, "w": 9.6, "h": 5.4}

# 1. Input Node (Top Left, 512x512)
box_in = FancyBboxPatch(
    (1.8, 43.7), 7.8, 5.8, boxstyle="round,pad=0.25", facecolor="#F8FAFC", edgecolor=C_INPUT, lw=1.4
)
ax.add_patch(box_in)
ax.text(
    5.7, 47.7, "Input Tile", ha="center", va="center", fontsize=8.5, weight="bold", color="#1E293B"
)
ax.text(5.7, 46.0, "Raw Frame", ha="center", va="center", fontsize=7.5, color="#64748B")
ax.text(
    5.7, 44.4, r"$512 \times 512 \times 1$", ha="center", va="center", fontsize=7.0, color="#64748B"
)

# Arrow from Input Tile down-right to Stem Conv (Conv 7x7 stride 2)
ax.annotate(
    "",
    xy=(11.7, 40.5),
    xytext=(9.9, 44.5),
    arrowprops={"arrowstyle": "->", "lw": 1.3, "color": "#64748B"},
)
ax.text(9.2, 42.0, "Conv 7x7 (s2)", ha="right", va="center", fontsize=6.3, color="#475569")

# 2. Decoder 1 (Top Right, 512x512)
box_d1 = FancyBboxPatch(
    (84.5, 44.0),
    7.6,
    5.2,
    boxstyle="round,pad=0.2",
    facecolor=C_DEC_BG,
    edgecolor=C_DEC,
    lw=1.4,
    zorder=4,
)
ax.add_patch(box_d1)
ax.text(
    84.5 + 3.8,
    44.0 + 5.2 * 0.75,
    "Decoder 1",
    ha="center",
    va="center",
    fontsize=7.8,
    weight="bold",
    color="#065F46",
    zorder=5,
)
ax.text(
    84.5 + 3.8,
    44.0 + 5.2 * 0.48,
    "UpConv + Conv",
    ha="center",
    va="center",
    fontsize=7.0,
    color="#047857",
    zorder=5,
)
ax.text(
    84.5 + 3.8,
    44.0 + 5.2 * 0.22,
    r"$16\text{ ch} \mid 512 \times 512$",
    ha="center",
    va="center",
    fontsize=6.3,
    color="#475569",
    zorder=5,
)

# Output Node (Right of Decoder 1, 512x512)
box_out = FancyBboxPatch(
    (96.5, 43.7),
    10.0,
    5.8,
    boxstyle="round,pad=0.25",
    facecolor="#F8FAFC",
    edgecolor="#059669",
    lw=1.4,
)
ax.add_patch(box_out)
ax.text(
    101.5,
    47.7,
    "Output Mask",
    ha="center",
    va="center",
    fontsize=8.5,
    weight="bold",
    color="#065F46",
)
ax.text(101.5, 46.0, "Vapor Probability", ha="center", va="center", fontsize=7.5, color="#059669")
ax.text(
    101.5,
    44.4,
    r"$512 \times 512 \times 1$",
    ha="center",
    va="center",
    fontsize=7.0,
    color="#059669",
)

# Arrow from Decoder 1 to Output Mask (Conv 1x1 head)
ax.annotate(
    "",
    xy=(96.1, 46.6),
    xytext=(92.4, 46.6),
    arrowprops={"arrowstyle": "->", "lw": 1.3, "color": "#059669"},
)
ax.text(
    94.25, 47.6, "Conv 1x1", ha="center", va="bottom", fontsize=6.5, color="#047857", weight="bold"
)

# Arrow from Decoder 2 up-right to Decoder 1
dec2 = matched_levels[0]
ax.annotate(
    "",
    xy=(84.5 + 3.8, 43.7),
    xytext=(dec2["dec_x"] + dec2["w"] / 2, dec2["y"] + dec2["h"] + 0.3),
    arrowprops={"arrowstyle": "->", "lw": 1.3, "color": C_DEC, "ls": "-"},
    zorder=4,
)

# 3. Draw Matched Encoder Nodes (Stem Conv + Stages 1..3)
for i, lvl in enumerate(matched_levels):
    x, y, w, h = lvl["enc_x"], lvl["y"], lvl["w"], lvl["h"]
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.2",
        facecolor=C_ENC_BG,
        edgecolor=C_ENC,
        lw=1.4,
        zorder=4,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h * 0.75,
        lvl["enc_title"],
        ha="center",
        va="center",
        fontsize=7.8,
        weight="bold",
        color="#1E3A8A",
        zorder=5,
    )
    ax.text(
        x + w / 2,
        y + h * 0.48,
        lvl["enc_sub"],
        ha="center",
        va="center",
        fontsize=7.0,
        color="#1D4ED8",
        zorder=5,
    )
    ax.text(
        x + w / 2,
        y + h * 0.22,
        lvl["enc_desc"],
        ha="center",
        va="center",
        fontsize=6.3,
        color="#475569",
        zorder=5,
    )

    if i == 0:
        # Stem Conv -> MaxPool 3x3 (s2) -> Stage 1
        next_lvl = matched_levels[i + 1]
        ax.annotate(
            "",
            xy=(next_lvl["enc_x"] + next_lvl["w"] / 2, next_lvl["y"] + next_lvl["h"] + 0.3),
            xytext=(x + w / 2, y - 0.3),
            arrowprops={"arrowstyle": "->", "lw": 1.3, "color": C_ENC, "ls": "-"},
            zorder=4,
        )
        # Position label clearly to the left and below the arrow with zero collision
        ax.text(
            17.2,
            33.2,
            "MaxPool s2",
            fontsize=6.6,
            color="#1E40AF",
            ha="right",
            va="center",
            weight="bold",
        )
    elif i < len(matched_levels) - 1:
        next_lvl = matched_levels[i + 1]
        ax.annotate(
            "",
            xy=(next_lvl["enc_x"] + next_lvl["w"] / 2, next_lvl["y"] + next_lvl["h"] + 0.3),
            xytext=(x + w / 2, y - 0.3),
            arrowprops={"arrowstyle": "->", "lw": 1.3, "color": C_ENC, "ls": "-"},
            zorder=4,
        )
    else:
        # Stage 3 -> Stage 4 (Bottleneck)
        ax.annotate(
            "",
            xy=(b_stage4["x"] + b_stage4["w"] / 2, b_stage4["y"] + b_stage4["h"] + 0.3),
            xytext=(x + w / 2, y - 0.3),
            arrowprops={"arrowstyle": "->", "lw": 1.3, "color": C_ENC, "ls": "-"},
            zorder=4,
        )

# 4. Draw Stage 4 (Bottleneck Base, 16x16)
sx, sy, sw, sh = b_stage4["x"], b_stage4["y"], b_stage4["w"], b_stage4["h"]
box_s4 = FancyBboxPatch(
    (sx, sy),
    sw,
    sh,
    boxstyle="round,pad=0.2",
    facecolor=C_BOT_BG,
    edgecolor=C_BOT,
    lw=1.5,
    zorder=4,
)
ax.add_patch(box_s4)
ax.text(
    sx + sw / 2,
    sy + sh * 0.75,
    "Stage 4 (Bottleneck)",
    ha="center",
    va="center",
    fontsize=7.8,
    weight="bold",
    color="#5B21B6",
    zorder=5,
)
ax.text(
    sx + sw / 2,
    sy + sh * 0.48,
    "3 ResBlocks (layer4)",
    ha="center",
    va="center",
    fontsize=7.0,
    color="#6D28D9",
    zorder=5,
)
ax.text(
    sx + sw / 2,
    sy + sh * 0.22,
    r"$512\text{ ch} \mid 16 \times 16$",
    ha="center",
    va="center",
    fontsize=6.5,
    color="#4C1D95",
    zorder=5,
)

# Stage 4 -> Decoder 5 upsampling arrow
dec5 = matched_levels[3]
ax.annotate(
    "",
    xy=(dec5["dec_x"] + dec5["w"] / 2, dec5["y"] - 0.3),
    xytext=(sx + sw - 0.3, sy + sh * 0.7),
    arrowprops={"arrowstyle": "->", "lw": 1.3, "color": C_DEC},
    zorder=4,
)

# 5. Draw Matched Decoders 5..2
for _i, lvl in enumerate(matched_levels):
    x, y, w, h = lvl["dec_x"], lvl["y"], lvl["w"], lvl["h"]
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.2",
        facecolor=C_DEC_BG,
        edgecolor=C_DEC,
        lw=1.4,
        zorder=4,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h * 0.75,
        lvl["dec_title"],
        ha="center",
        va="center",
        fontsize=7.8,
        weight="bold",
        color="#065F46",
        zorder=5,
    )
    ax.text(
        x + w / 2,
        y + h * 0.48,
        lvl["dec_sub"],
        ha="center",
        va="center",
        fontsize=7.0,
        color="#047857",
        zorder=5,
    )
    ax.text(
        x + w / 2,
        y + h * 0.22,
        lvl["dec_desc"],
        ha="center",
        va="center",
        fontsize=6.3,
        color="#475569",
        zorder=5,
    )

# Upsampling arrows between decoders: Dec 5 -> Dec 4 -> Dec 3 -> Dec 2
for i in range(len(matched_levels) - 1, 0, -1):
    curr_dec = matched_levels[i]
    next_dec = matched_levels[i - 1]
    ax.annotate(
        "",
        xy=(next_dec["dec_x"] + next_dec["w"] / 2, next_dec["y"] - 0.3),
        xytext=(curr_dec["dec_x"] + curr_dec["w"] / 2, curr_dec["y"] + curr_dec["h"] + 0.3),
        arrowprops={"arrowstyle": "->", "lw": 1.3, "color": C_DEC, "ls": "-"},
        zorder=4,
    )

# 6. U-Net Long Horizontal Skip Connections (Exact Same Resolutions!)
for lvl in matched_levels:
    y_line = lvl["y"] + lvl["h"] / 2
    x_start = lvl["enc_x"] + lvl["w"] + 0.3
    x_end = lvl["dec_x"] - 0.3
    ax.annotate(
        "",
        xy=(x_end, y_line),
        xytext=(x_start, y_line),
        arrowprops={"arrowstyle": "->", "lw": 1.5, "color": C_UNET_SKIP, "ls": "--"},
        zorder=3,
    )

# Center explanatory text card for Long Skip Connections (Cyan-blue theme)
# Sits cleanly in the clearance gap between the 64x64 skip (y=24.1) and 128x128 skip (y=31.6)
card_long = FancyBboxPatch(
    (41.5, 25.5),
    18.0,
    4.8,
    boxstyle="round,pad=0.25",
    facecolor=C_CARD_BG,
    edgecolor=C_CARD_BORDER,
    lw=1.3,
    zorder=6,
)
ax.add_patch(card_long)
ax.text(
    50.5,
    28.8,
    "U-Net Long Skip Connections",
    ha="center",
    va="center",
    fontsize=7.8,
    weight="bold",
    color="#0369A1",
    zorder=7,
)
ax.text(
    50.5,
    27.4,
    "Dimension-matched spatial transfer",
    ha="center",
    va="center",
    fontsize=6.8,
    color="#0284C7",
    zorder=7,
)
ax.text(
    50.5,
    26.2,
    "Enables sub-pixel boundary localization",
    ha="center",
    va="center",
    fontsize=6.4,
    color="#475569",
    zorder=7,
)

# 7. RESIDUAL UNIT CALLOUT (Exact Form as reference image)
callout_bg = FancyBboxPatch(
    (1.5, 1.2),
    31.5,
    17.2,
    boxstyle="round,pad=0.3",
    facecolor="#FFFBEB",
    edgecolor=C_RES_SKIP,
    lw=1.5,
    ls="-",
    zorder=3,
)
ax.add_patch(callout_bg)

# Header
ax.text(
    17.25,
    17.1,
    "ResNet-34 Residual Unit (Local Identity Shortcut)",
    ha="center",
    va="center",
    fontsize=8.0,
    weight="bold",
    color="#92400E",
)

# Micro-flow inside unit
ux = 2.4
ax.text(
    ux + 1.2,
    10.8,
    r"$\mathbf{x}$",
    fontsize=9.2,
    weight="bold",
    color="#78350F",
    va="center",
    ha="center",
    zorder=5,
)
ax.annotate(
    "",
    xy=(ux + 4.6, 10.8),
    xytext=(ux + 2.2, 10.8),
    arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#92400E"},
    zorder=5,
)

# Weight layer 1
b_c1 = FancyBboxPatch(
    (ux + 4.6, 9.0),
    6.2,
    3.6,
    boxstyle="round,pad=0.15",
    facecolor="#FEF3C7",
    edgecolor="#D97706",
    lw=1.2,
    zorder=4,
)
ax.add_patch(b_c1)
ax.text(
    ux + 7.7,
    10.8,
    r"$\mathrm{Conv\ 3\times3}$",
    fontsize=7.2,
    ha="center",
    va="center",
    color="#92400E",
    zorder=5,
)

# Arrow to Conv 2
ax.annotate(
    "",
    xy=(ux + 12.8, 10.8),
    xytext=(ux + 10.8, 10.8),
    arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#92400E"},
    zorder=5,
)

# Weight layer 2
b_c2 = FancyBboxPatch(
    (ux + 12.8, 9.0),
    6.2,
    3.6,
    boxstyle="round,pad=0.15",
    facecolor="#FEF3C7",
    edgecolor="#D97706",
    lw=1.2,
    zorder=4,
)
ax.add_patch(b_c2)
ax.text(
    ux + 15.9,
    10.8,
    r"$\mathrm{Conv\ 3\times3}$",
    fontsize=7.2,
    ha="center",
    va="center",
    color="#92400E",
    zorder=5,
)

# Arrow to Sum
ax.annotate(
    "",
    xy=(ux + 21.0, 10.8),
    xytext=(ux + 19.0, 10.8),
    arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#92400E"},
    zorder=5,
)

# Addition circle (+)
circ_add = patches.Circle(
    (ux + 22.1, 10.8), 1.1, facecolor="#FEF3C7", edgecolor="#D97706", lw=1.2, zorder=4
)
ax.add_patch(circ_add)
ax.text(
    ux + 22.1,
    10.8,
    "+",
    fontsize=9.2,
    weight="bold",
    color="#92400E",
    ha="center",
    va="center",
    zorder=5,
)

# Arrow out to F(x) + x
ax.annotate(
    "",
    xy=(ux + 25.0, 10.8),
    xytext=(ux + 23.2, 10.8),
    arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#92400E"},
    zorder=5,
)
ax.text(
    ux + 25.6,
    10.8,
    r"$\mathcal{F}(\mathbf{x}) + \mathbf{x}$",
    fontsize=8.0,
    weight="bold",
    color="#78350F",
    va="center",
    ha="left",
    zorder=5,
)

# Residual Skip Curve (Identity Shortcut)
ax.plot([ux + 1.2, ux + 1.2], [11.8, 14.3], color=C_RES_SKIP, lw=1.6, zorder=5)
ax.plot([ux + 1.2, ux + 22.1], [14.3, 14.3], color=C_RES_SKIP, lw=1.6, zorder=5)
ax.annotate(
    "",
    xy=(ux + 22.1, 12.0),
    xytext=(ux + 22.1, 14.3),
    arrowprops={"arrowstyle": "->", "lw": 1.6, "color": C_RES_SKIP},
    zorder=5,
)

ax.text(
    ux + 11.65,
    14.8,
    "Identity Shortcut (Gradient Highway)",
    fontsize=7.2,
    weight="bold",
    color="#B45309",
    ha="center",
    va="bottom",
    zorder=5,
)

# Mathematical Rigor Clarification (Vector / Jacobian formulation):
ax.text(
    17.25,
    6.8,
    r"$\frac{d\mathcal{E}}{d\mathbf{x}} = \frac{d\mathcal{E}}{d(\mathcal{F}+\mathbf{x})} \cdot \left( \frac{d\mathcal{F}}{d\mathbf{x}} + \mathbf{I} \right) \quad \left[\text{Scalar: } \frac{d\mathcal{E}}{dx} = \frac{d\mathcal{E}}{dy}\left(\frac{d\mathcal{F}}{dx} + 1\right)\right]$",
    ha="center",
    va="center",
    fontsize=7.2,
    color="#78350F",
)
ax.text(
    17.25,
    5.0,
    r"$\text{Vector identity matrix } \mathbf{I} \in \mathbb{R}^{D \times D} \text{ guarantees non-vanishing backpropagation}$",
    ha="center",
    va="center",
    fontsize=6.8,
    style="italic",
    color="#B45309",
)

ax.text(
    17.25,
    2.7,
    "• Direct backpropagation without exponential decay across all 34 layers\n• Preserves interfacial gradients under optical glare and defocus blur",
    ha="center",
    va="center",
    fontsize=6.6,
    color="#92400E",
    linespacing=1.35,
)

# 8. Legend / Key at bottom right
leg_bg = FancyBboxPatch(
    (69.0, 1.2),
    36.5,
    11.5,
    boxstyle="round,pad=0.3",
    facecolor="#F9FAFB",
    edgecolor="#E5E7EB",
    lw=1.2,
)
ax.add_patch(leg_bg)
ax.text(
    87.25,
    11.2,
    "Architecture Key & Information Flow",
    ha="center",
    va="center",
    fontsize=7.8,
    weight="bold",
    color="#111827",
)

# Legend entries
ax.plot([70.5, 75.0], [9.1, 9.1], color=C_RES_SKIP, lw=1.8)
ax.text(
    76.2,
    9.1,
    "Local Residual Skip: Direct gradient flow (I / 1)",
    fontsize=6.8,
    color="#374151",
    va="center",
)

ax.plot([70.5, 75.0], [6.9, 6.9], color=C_UNET_SKIP, lw=1.8, ls="--")
ax.text(
    76.2,
    6.9,
    "Long U-Net Skip: Same-resolution boundary transfer",
    fontsize=6.8,
    color="#374151",
    va="center",
)

ax.annotate(
    "",
    xy=(74.5, 4.7),
    xytext=(70.5, 4.7),
    arrowprops={"arrowstyle": "->", "lw": 1.4, "color": C_ENC},
)
ax.text(
    76.2,
    4.7,
    "Downsampling: Stride-2 Conv / MaxPool (Context)",
    fontsize=6.8,
    color="#374151",
    va="center",
)

ax.annotate(
    "",
    xy=(74.5, 2.5),
    xytext=(70.5, 2.5),
    arrowprops={"arrowstyle": "->", "lw": 1.4, "color": C_DEC},
)
ax.text(
    76.2,
    2.5,
    "Upsampling: Transposed Conv / Bilinear (Resolution)",
    fontsize=6.8,
    color="#374151",
    va="center",
)

plt.savefig(OUTPUT_DIR / "resnet_unet_architecture.pdf", bbox_inches="tight")
plt.savefig(OUTPUT_DIR / "resnet_unet_architecture.png", dpi=300, bbox_inches="tight")
plt.close()

art_path = Path(
    "/Users/gabrielribeiro/.gemini/antigravity/brain/eb073713-902e-472a-a008-04f529db4fa1/resnet_unet_architecture.png"
)
shutil.copy(OUTPUT_DIR / "resnet_unet_architecture.png", art_path)
print(
    "Successfully regenerated exact smp.Unet architecture schematic with aligned horizontal skips!"
)
