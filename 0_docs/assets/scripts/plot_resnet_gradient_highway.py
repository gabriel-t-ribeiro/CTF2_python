"""
Generate a clean standalone schematic illustrating gradient backpropagation
comparison between a plain deep network vs. a residual network (ResNet-34).

Shows how the identity gradient term (I / 1) acts as a direct gradient highway,
delivering unattenuated loss gradients back to ANY earlier layer (Stages 1-4).
"""

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "figure.autolayout": False,
    }
)

OUTPUT_DIR = Path("ctf2_article/figures/schematics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 6.8), dpi=300)
fig.subplots_adjust(hspace=0.46, left=0.04, right=0.96, top=0.91, bottom=0.06)

# Title
fig.suptitle(
    r"$\mathbf{Gradient\ Propagation\ Comparison:\ Plain\ Deep\ Network\ vs.\ ResNet\ Identity\ Highway}$",
    fontsize=12.0,
    weight="bold",
    color="#111827",
    y=0.97,
)

# Common styling
C_PLAIN_VANISH = "#DC2626"
C_RES_HIGHWAY = "#D97706"

# -------------------------------------------------------------
# Panel 1: Plain Deep Network (Exponential Gradient Decay)
# -------------------------------------------------------------
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 11)
ax1.axis("off")

# Section Label
ax1.text(
    0,
    9.8,
    r"$\mathbf{(a)\ Plain\ Deep\ Network\ (Standard\ Chain\ Rule\ Backpropagation)}$",
    fontsize=9.8,
    weight="bold",
    color="#991B1B",
)
ax1.text(
    0,
    8.4,
    r"Gradient decays exponentially: $\frac{\partial \mathcal{E}}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{E}}{\partial \mathbf{x}_L} \prod_{i=l}^{L-1} \frac{\partial \mathbf{x}_{i+1}}{\partial \mathbf{x}_i} \approx 1.0 \times (0.8)^{30} \to \mathbf{0.001}$ (Layers starve of learning signal)",
    fontsize=7.8,
    color="#7F1D1D",
)

# Plain stages from Input (left) to Loss (right)
plain_stages = [
    {
        "name": "Stage 1\n(Low-level edges)",
        "x": 6,
        "grad": "Grad: 0.001\n(Vanished!)",
        "color": "#FCA5A5",
    },
    {"name": "Stage 2\n(Texture / Curves)", "x": 26, "grad": "Grad: 0.018", "color": "#FED7AA"},
    {"name": "Stage 3\n(Part semantics)", "x": 46, "grad": "Grad: 0.115", "color": "#FEF08A"},
    {"name": "Stage 4\n(Deep context)", "x": 66, "grad": "Grad: 0.420", "color": "#BBF7D0"},
    {
        "name": "Loss Layer\n" + r"$\mathcal{E}$ (Dice + BCE)",
        "x": 84,
        "grad": "Grad: 1.000\n(Full signal)",
        "color": "#93C5FD",
    },
]

for s in plain_stages:
    box = FancyBboxPatch(
        (s["x"], 2.2),
        12.0,
        4.6,
        boxstyle="round,pad=0.2",
        facecolor="#FAFAFA",
        edgecolor="#9CA3AF",
        lw=1.2,
    )
    ax1.add_patch(box)
    ax1.text(
        s["x"] + 6.0,
        4.8,
        s["name"],
        ha="center",
        va="center",
        fontsize=7.2,
        color="#1F2937",
        weight="bold",
    )
    ax1.text(
        s["x"] + 6.0,
        3.0,
        s["grad"],
        ha="center",
        va="center",
        fontsize=6.5,
        color="#DC2626" if "Vanished" in s["grad"] else "#4B5563",
    )

# Forward arrows (top, thin)
for i in range(len(plain_stages) - 1):
    x1_f = plain_stages[i]["x"] + 12.0
    x2_f = plain_stages[i + 1]["x"]
    ax1.annotate(
        "",
        xy=(x2_f, 5.0),
        xytext=(x1_f, 5.0),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#9CA3AF", "ls": "-"},
    )

# Backward gradient arrows (bottom, decaying thickness and red tone)
decay_widths = [1.0, 1.8, 2.8, 4.0]
for i in range(len(plain_stages) - 1):
    x_from = plain_stages[i + 1]["x"]
    x_to = plain_stages[i]["x"] + 12.0
    lw_val = decay_widths[i]
    ax1.annotate(
        "",
        xy=(x_to, 1.2),
        xytext=(x_from, 1.2),
        arrowprops={"arrowstyle": "->", "lw": lw_val, "color": C_PLAIN_VANISH, "ls": "-"},
    )

ax1.text(
    50,
    0.2,
    r"$\longleftarrow\ \mathrm{Backpropagation\ path\ shrinks\ exponentially\ through\ consecutive\ weight\ matrices}$",
    ha="center",
    va="center",
    fontsize=7.2,
    color="#991B1B",
)


# -------------------------------------------------------------
# Panel 2: ResNet-34 Architecture (Direct Additive Identity Highway)
# -------------------------------------------------------------
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 12.2)
ax2.axis("off")

ax2.text(
    0,
    11.2,
    r"$\mathbf{(b)\ ResNet\ Encoder\ (Additive\ Identity\ Highway\ to\ Every\ Layer)}$",
    fontsize=9.8,
    weight="bold",
    color="#1E40AF",
)
ax2.text(
    0,
    9.8,
    r"Gradient splits additively: $\frac{\partial \mathcal{E}}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{E}}{\partial \mathbf{x}_L} \cdot \left(\mathbf{I} + \frac{\partial}{\partial \mathbf{x}_l}\sum \mathcal{F}_i\right) = \frac{\partial \mathcal{E}}{\partial \mathbf{x}_L} + \mathrm{Residual}$ (Guaranteed delivery to ANY layer $l$)",
    fontsize=7.8,
    color="#1E3A8A",
)

res_stages = [
    {
        "name": "Stage 1 (64 ch)\nInitial Edges & Glare",
        "x": 6,
        "desc": "Receives direct loss gradient\nRefines sharp bubble boundaries",
        "color": "#DBEAFE",
    },
    {
        "name": "Stage 2 (128 ch)\nCurvature & Blurs",
        "x": 26,
        "desc": "Receives direct loss gradient\nLearns defocus contours",
        "color": "#DBEAFE",
    },
    {
        "name": "Stage 3 (256 ch)\nSlug Bodies & Walls",
        "x": 46,
        "desc": "Receives direct loss gradient\nResolves slug noses / tails",
        "color": "#DBEAFE",
    },
    {
        "name": "Stage 4 (512 ch)\nGlobal Void Context",
        "x": 66,
        "desc": "Bottleneck representation\nGlobal two-phase topology",
        "color": "#EDE9FE",
    },
    {
        "name": "Loss Layer\n" + r"$\mathcal{E}$ (Dice + BCE)",
        "x": 84,
        "desc": "Loss evaluation\n" + r"$\frac{\partial \mathcal{E}}{\partial \mathbf{x}_L} = 1.0$",
        "color": "#FEF3C7",
    },
]

for s in res_stages:
    box = FancyBboxPatch(
        (s["x"], 1.8),
        12.0,
        4.6,
        boxstyle="round,pad=0.2",
        facecolor=s["color"],
        edgecolor="#2563EB" if "Stage" in s["name"] else "#D97706",
        lw=1.2,
    )
    ax2.add_patch(box)
    ax2.text(
        s["x"] + 6.0,
        4.4,
        s["name"],
        ha="center",
        va="center",
        fontsize=7.2,
        color="#1E3A8A" if "Stage" in s["name"] else "#92400E",
        weight="bold",
    )
    ax2.text(s["x"] + 6.0, 2.7, s["desc"], ha="center", va="center", fontsize=5.8, color="#1F2937")

# Standard forward arrows between adjacent stages
for i in range(len(res_stages) - 1):
    x1_f = res_stages[i]["x"] + 12.0
    x2_f = res_stages[i + 1]["x"]
    ax2.annotate(
        "",
        xy=(x2_f, 4.1),
        xytext=(x1_f, 4.1),
        arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "#2563EB", "ls": "-"},
    )

# The Identity Highway (top loop spanning back to ALL layers)
highway_y = 7.7
ax2.plot([90.0, 90.0], [6.4, highway_y], color=C_RES_HIGHWAY, lw=2.0, ls="-")
ax2.plot([12.0, 90.0], [highway_y, highway_y], color=C_RES_HIGHWAY, lw=2.0, ls="-")

# Label card above the highway line (centered with ample clearance)
ax2.text(
    50.0,
    highway_y + 0.65,
    r"$\mathbf{Residual\ Identity\ Highway\ (+\mathbf{I}\ Term\ Transmits\ Gradient\ Direct\ to\ Any\ Layer)}$",
    ha="center",
    va="center",
    fontsize=7.6,
    weight="bold",
    color="#B45309",
    bbox={"boxstyle": "round,pad=0.25", "facecolor": "#FFFBEB", "edgecolor": "#D97706", "lw": 1.0},
)

# Taps dropping unattenuated gradient down into Stage 1, Stage 2, Stage 3, and Stage 4
targets = [
    (12.0, "Stage 1"),
    (32.0, "Stage 2"),
    (52.0, "Stage 3"),
    (72.0, "Stage 4"),
]

for tx, _stg in targets:
    ax2.annotate(
        "",
        xy=(tx, 6.4),
        xytext=(tx, highway_y),
        arrowprops={"arrowstyle": "->", "lw": 1.8, "color": C_RES_HIGHWAY},
    )
    ax2.text(
        tx,
        highway_y - 0.65,
        r"$+\mathbf{I}$",
        fontsize=7.8,
        weight="bold",
        color="#B45309",
        ha="center",
        va="center",
    )

# Bottom summary note
ax2.text(
    50,
    0.4,
    "Key benefit: Gradients reach low-level edge-detecting layers directly, delivering learning updates where boundary precision matters most.",
    ha="center",
    va="center",
    fontsize=7.2,
    color="#1E3A8A",
)

pdf_path = OUTPUT_DIR / "resnet_gradient_highway.pdf"
png_path = OUTPUT_DIR / "resnet_gradient_highway.png"
plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close()

# Copy to artifact folder for user viewing
art_path = Path(
    "/Users/gabrielribeiro/.gemini/antigravity/brain/eb073713-902e-472a-a008-04f529db4fa1/resnet_gradient_highway.png"
)
shutil.copy(png_path, art_path)
print("Saved resnet_gradient_highway figure successfully!")
