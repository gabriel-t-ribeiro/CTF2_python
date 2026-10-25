#!/usr/bin/env python3
"""
Generate publication-quality vector schematics for the scientific manuscript (ctf2_article).
Grounds:
  1. Tube cross-section, chord depth profile, and dual-regime 3D volume reconstruction (Regime 1 & 2).
  2. Two-stage Kuhn-Munkres kinematic tracking, search envelopes, and breakup quarantine.
  3. Single-camera Depth from Defocus (DFD) universal inversion curves and smoothstep near-wall regularization.
"""

import os

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Patch, Polygon, Rectangle

# Publication style configuration (standard mathtext, no unescaped LaTeX outside math)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "figure.titlesize": 11.5,
        "figure.dpi": 300,
    }
)

OUTPUT_DIR = Path("ctf2_article/figures/schematics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_tube_chord_geometry():
    """Generates Figure grounding Section 2.4: Tube chord profile and 3D reconstruction."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4), dpi=300)

    # ----------------------------------------------------
    # Subplot 1: Circular Tube Cross-Section & Regime 1
    # ----------------------------------------------------
    R = 10.0  # mm
    ax1.set_aspect("equal")

    # Tube outer & inner wall
    tube_wall = Circle(
        (0, 0),
        R,
        edgecolor="#1B365D",
        facecolor="#EBF3FA",
        lw=2.2,
        label=r"Inner tube wall ($R_{\mathrm{tube}}=10\,\mathrm{mm}$)",
    )
    ax1.add_patch(tube_wall)

    # Coordinates & Centerline
    ax1.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax1.axvline(0, color="gray", ls=":", lw=0.8, alpha=0.7)
    ax1.plot(0, 0, "k+", ms=7, mew=1.4)
    # Tube center callout from open Quadrant 3
    ax1.annotate(
        r"$\mathrm{Tube\ axis\ }(y=0)$",
        xy=(0, 0),
        xytext=(-6.8, -2.5),
        arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.9},
        fontsize=7.8,
        color="#333333",
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "edgecolor": "#AAAAAA",
            "lw": 0.6,
            "alpha": 0.92,
        },
    )

    # Vertical chord position y_c = 4.5 mm
    y_c = 4.5
    half_chord = np.sqrt(R**2 - y_c**2)

    # Red chord line across the tube
    ax1.plot([-half_chord, half_chord], [y_c, y_c], color="#D9534F", lw=2.0, ls="-")

    # Clean engineering dimension line above the bubble (at y = 7.8) with extension lines
    ax1.plot([-half_chord, -half_chord], [y_c, 8.2], color="#D9534F", ls=":", lw=0.8)
    ax1.plot([half_chord, half_chord], [y_c, 8.2], color="#D9534F", ls=":", lw=0.8)
    ax1.annotate(
        "",
        xy=(-half_chord, 7.8),
        xytext=(half_chord, 7.8),
        arrowprops={"arrowstyle": "<->", "color": "#D9534F", "lw": 1.2},
    )
    ax1.text(
        0,
        8.3,
        r"$L_{\mathrm{chord}}(y) = 2\sqrt{R_{\mathrm{tube}}^2 - \Delta y^2}$",
        ha="center",
        va="bottom",
        fontsize=7.8,
        color="#C9302C",
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#D9534F",
            "lw": 0.8,
            "alpha": 0.95,
        },
    )

    # Delta y dimension line along vertical axis
    ax1.annotate(
        "",
        xy=(0, 0),
        xytext=(0, y_c),
        arrowprops={"arrowstyle": "<->", "color": "#333333", "lw": 1.0},
    )
    ax1.text(
        -0.6,
        y_c / 2 - 0.1,
        r"$\Delta y$",
        fontsize=8.5,
        fontweight="bold",
        color="#333333",
        ha="right",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.15",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
        },
    )

    # Regime 1: Small bubble ellipsoid in cross-section
    b_semi = 2.3
    c_semi = min(b_semi, half_chord * 0.7)
    bubble_ell = Ellipse(
        (1.5, y_c),
        2 * c_semi,
        2 * b_semi,
        edgecolor="#0275D8",
        facecolor="#5BC0DE",
        alpha=0.6,
        lw=1.5,
    )
    ax1.add_patch(bubble_ell)

    # Semi-axis c dimension line with side-pull leader line (ISO/ASME drafting standard for constrained regions)
    # Highlight semi-axis c on the chord line with endpoints
    ax1.plot([1.5, 1.5 + c_semi], [y_c, y_c], color="#0275D8", ls="-", lw=2.2)
    ax1.plot([1.5, 1.5 + c_semi], [y_c, y_c], "o", color="#0275D8", ms=3.2)

    # Extension lines dropping down to y = 2.8
    y_dim = 3.0
    ax1.plot([1.5, 1.5], [y_c, y_dim - 0.3], color="#0275D8", ls=":", lw=0.85)
    ax1.plot([1.5 + c_semi, 1.5 + c_semi], [y_c, y_dim - 0.3], color="#0275D8", ls=":", lw=0.85)

    # Dimension arrow between extension lines
    ax1.annotate(
        "",
        xy=(1.5, y_dim),
        xytext=(1.5 + c_semi, y_dim),
        arrowprops={"arrowstyle": "<->", "color": "#0275D8", "lw": 1.1},
    )

    # Extended horizontal leader line pulling out to the side into clear open space
    ax1.plot([1.5 + c_semi, 5.0], [y_dim, y_dim], color="#0275D8", ls="-", lw=1.1)

    # Dimension label card placed to the side, aligned with the leader line
    ax1.text(
        5.1,
        y_dim,
        r"$c \leq \frac{1}{2}L_{\mathrm{chord}}(y)$",
        color="#0275D8",
        fontsize=7.6,
        fontweight="bold",
        ha="left",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "#0275D8",
            "lw": 0.7,
            "alpha": 0.95,
        },
    )

    # Radial dimension R_tube
    angle = np.deg2rad(-42)
    ax1.annotate(
        "",
        xy=(0, 0),
        xytext=(R * np.cos(angle), R * np.sin(angle)),
        arrowprops={"arrowstyle": "<->", "color": "#1B365D", "lw": 1.2},
    )
    ax1.text(
        R * 0.5 * np.cos(angle) + 0.3,
        R * 0.5 * np.sin(angle) - 0.8,
        r"$R_{\mathrm{tube}}$",
        color="#1B365D",
        fontsize=8.5,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.15",
            "facecolor": "white",
            "edgecolor": "#1B365D",
            "lw": 0.6,
            "alpha": 0.9,
        },
    )

    ax1.set_xlim(-12.5, 12.5)
    ax1.set_ylim(-12.5, 12.5)
    ax1.set_xlabel(r"Transverse depth $z$ [mm]")
    ax1.set_ylabel(r"Vertical coordinate $y$ [mm]")
    ax1.set_title(
        r"$\mathbf{(a)}$ Tube Cross-Section & Regime 1 (Ellipsoid)", fontsize=9.5, color="#1B365D"
    )
    ax1.grid(True, ls=":", alpha=0.4)

    # ----------------------------------------------------
    # Subplot 2: Regime 2 Elongated Slug & Boundary Attenuation
    # ----------------------------------------------------
    y_vals = np.linspace(-R, R, 200)
    L_vals = 2 * np.sqrt(np.maximum(0, R**2 - y_vals**2))

    ax2.plot(L_vals, y_vals, color="#1B365D", lw=2.0, label=r"Max. chord $L_{\mathrm{chord}}(y)$")
    ax2.fill_betweenx(y_vals, 0, L_vals, color="#EBF3FA", alpha=0.5)

    # Slug hugging upper wall (y from 2.5 mm to 9.5 mm)
    y_slug = np.linspace(2.5, 9.5, 150)
    L_slug = 2 * np.sqrt(np.maximum(0, R**2 - y_slug**2))
    d_phys = np.sin((y_slug - 2.5) / (9.5 - 2.5) * np.pi) * 3.5  # mm
    lambda_phys = 1.7  # 1/mm
    D_slug = L_slug * np.exp(-1.0 / (lambda_phys * d_phys + 1e-3))
    D_slug_bounded = np.minimum(D_slug, L_slug)

    ax2.plot(
        D_slug_bounded,
        y_slug,
        color="#D9534F",
        lw=2.2,
        label=r"Slug profile $D_{\mathrm{bounded}}(y)$",
    )
    ax2.fill_betweenx(y_slug, 0, D_slug_bounded, color="#F2DEDE", alpha=0.7)

    # Buoyancy arrow in clear upper area
    ax2.annotate(
        "",
        xy=(18.5, 8.8),
        xytext=(18.5, 5.5),
        arrowprops={"facecolor": "#5CB85C", "edgecolor": "#4CAE4C", "width": 2.2, "headwidth": 6.5},
    )
    ax2.text(
        18.5,
        9.4,
        "Buoyancy",
        fontsize=8.0,
        fontweight="bold",
        color="#3C763D",
        ha="center",
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "edgecolor": "#5CB85C",
            "lw": 0.7,
            "alpha": 0.9,
        },
    )

    # Annotation for Laplacian smoothing placed in lower clear quadrant
    ax2.annotate(
        r"$D'(p) = L_{\mathrm{chord}}(y)\,\exp\left(-\frac{1}{\lambda_{\mathrm{phys}} d_{\mathrm{phys}}}\right)$",
        xy=(D_slug_bounded[len(D_slug_bounded) // 2], y_slug[len(y_slug) // 2]),
        xytext=(2.2, -2.8),
        arrowprops={
            "arrowstyle": "->",
            "connectionstyle": "arc3,rad=0.15",
            "color": "#D9534F",
            "lw": 1.2,
        },
        fontsize=7.8,
        color="#A94442",
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "#D9534F",
            "lw": 0.9,
            "alpha": 0.95,
        },
    )

    ax2.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax2.set_xlabel(r"Transverse thickness / chord [mm]")
    ax2.set_ylabel(r"Vertical coordinate $y$ [mm]")
    ax2.set_title(
        r"$\mathbf{(b)}$ Regime 2: Slug Confinement & Attenuation", fontsize=9.5, color="#1B365D"
    )
    ax2.set_xlim(0, 22.5)
    ax2.set_ylim(-11.5, 11.5)
    ax2.legend(
        loc="lower left",
        bbox_to_anchor=(0.04, 0.04),
        framealpha=0.95,
        facecolor="white",
        edgecolor="#D0D0D0",
        fontsize=7.8,
    )
    ax2.grid(True, ls=":", alpha=0.4)

    plt.tight_layout()
    pdf_path = OUTPUT_DIR / "tube_chord_geometry.pdf"
    png_path = OUTPUT_DIR / "tube_chord_geometry.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DONE] Saved {pdf_path} and {png_path}")


def generate_tracking_breakup_schematic():
    """Generates Figure grounding Section 2.5: Kinematics, search window, and breakup decoupling."""
    fig, ax = plt.subplots(figsize=(7.5, 3.5), dpi=300)

    # Background frames boundary with wide central gap to prevent any overlap with connection cards
    frame_w = 3.2
    frame_x1 = 0.35
    gap_x2 = 7.45
    frame_t = Rectangle(
        (frame_x1, 0.5), frame_w, 5.8, facecolor="#F8F9FA", edgecolor="#495057", lw=1.5
    )
    frame_t1 = Rectangle(
        (gap_x2, 0.5), frame_w, 5.8, facecolor="#F8F9FA", edgecolor="#495057", lw=1.5
    )
    ax.add_patch(frame_t)
    ax.add_patch(frame_t1)

    ax.text(
        frame_x1 + frame_w / 2,
        6.6,
        r"$\mathbf{Frame\ t}$ ($N_t$ entities)",
        ha="center",
        fontsize=11.5,
        color="#1B365D",
    )
    ax.text(
        gap_x2 + frame_w / 2,
        6.6,
        r"$\mathbf{Frame\ t+1}$ ($N_{t+1}$ entities)",
        ha="center",
        fontsize=11.5,
        color="#1B365D",
    )

    # Bulk advection vector across frames in central gap
    bulk_arrow = FancyArrowPatch(
        (4.35, 3.6), (6.65, 3.6), arrowstyle="-|>", mutation_scale=14, color="#5CB85C", lw=2.4
    )
    ax.add_patch(bulk_arrow)
    ax.text(
        5.5,
        3.95,
        r"$\mathbf{v}_{\mathrm{bulk}}$",
        ha="center",
        fontsize=11.0,
        fontweight="bold",
        color="#3C763D",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
        },
    )

    # ----------------------------------------------------
    # Case A: Steady 1-to-1 Translation (Top)
    # ----------------------------------------------------
    # Bubble 1 at frame t
    b1_t = Circle((1.8, 4.9), 0.52, facecolor="#64B5F6", edgecolor="#0D47A1", lw=1.6, alpha=0.85)
    ax.add_patch(b1_t)
    ax.text(
        1.8,
        4.9,
        r"$B_1$",
        ha="center",
        va="center",
        color="#FFFFFF",
        fontsize=10.0,
        fontweight="bold",
        path_effects=[pe.withStroke(linewidth=2.2, foreground="#0D47A1")],
    )

    # Bubble 1 translated at frame t+1
    b1_t1 = Circle((8.5, 4.9), 0.52, facecolor="#64B5F6", edgecolor="#0D47A1", lw=1.6, alpha=0.85)
    ax.add_patch(b1_t1)
    ax.text(
        8.5,
        4.9,
        r"$B_1'$",
        ha="center",
        va="center",
        color="#FFFFFF",
        fontsize=10.0,
        fontweight="bold",
        path_effects=[pe.withStroke(linewidth=2.2, foreground="#0D47A1")],
    )

    # Translation link (Kuhn-Munkres 1-to-1 match) - arc reaches peak at y=5.4
    match_arrow = FancyArrowPatch(
        (2.4, 5.0),
        (7.9, 5.0),
        arrowstyle="->",
        connectionstyle="arc3,rad=-0.10",
        ls="--",
        color="#0275D8",
        lw=1.8,
    )
    ax.add_patch(match_arrow)
    # Match card placed cleanly ABOVE the arc in the central gap (y=6.05), perfectly clear of frames and curve
    ax.text(
        5.5,
        6.05,
        r"$\mathbf{Optimal\ 1\mathrm{-}to\mathrm{-}1\ Match}\ (C_{ij} \leq d_{\max})$",
        ha="center",
        fontsize=9.0,
        color="#0275D8",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "#0275D8",
            "lw": 0.8,
            "alpha": 0.95,
        },
    )

    # ----------------------------------------------------
    # Case B: Hydrodynamic Breakup & Quarantine (Bottom)
    # ----------------------------------------------------
    # Parent bubble P at frame t
    parent_p = Ellipse(
        (1.9, 1.8), 1.6, 0.95, facecolor="#E57373", edgecolor="#B71C1C", lw=1.6, alpha=0.85
    )
    ax.add_patch(parent_p)
    ax.text(
        1.9,
        1.8,
        r"$\mathbf{Parent\ } P$",
        ha="center",
        va="center",
        color="#FFFFFF",
        fontsize=9.0,
        fontweight="bold",
        path_effects=[pe.withStroke(linewidth=2.2, foreground="#B71C1C")],
    )

    # Search envelope W_i in frame t+1
    env_w = Rectangle(
        (7.55, 0.9),
        2.55,
        1.8,
        facecolor="#FBE9E7",
        edgecolor="#D9534F",
        ls="--",
        lw=1.4,
        alpha=0.85,
    )
    ax.add_patch(env_w)
    ax.text(
        9.95,
        2.95,
        r"$\mathcal{W}_i$",
        ha="center",
        va="center",
        color="#D9534F",
        fontsize=10.0,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#D9534F",
            "lw": 0.7,
            "alpha": 0.95,
        },
    )

    # Two daughter bubbles inside W_i
    d1 = Circle((8.15, 1.9), 0.36, facecolor="#FFB74D", edgecolor="#E65100", lw=1.4, alpha=0.9)
    d2 = Circle((9.25, 1.45), 0.38, facecolor="#FFB74D", edgecolor="#E65100", lw=1.4, alpha=0.9)
    ax.add_patch(d1)
    ax.add_patch(d2)
    ax.text(
        8.15,
        1.9,
        r"$D_1$",
        ha="center",
        va="center",
        color="#FFFFFF",
        fontsize=8.5,
        fontweight="bold",
        path_effects=[pe.withStroke(linewidth=2.0, foreground="#E65100")],
    )
    ax.text(
        9.25,
        1.45,
        r"$D_2$",
        ha="center",
        va="center",
        color="#FFFFFF",
        fontsize=8.5,
        fontweight="bold",
        path_effects=[pe.withStroke(linewidth=2.0, foreground="#E65100")],
    )

    # Breakup arrow connecting parent to daughter envelope
    breakup_arrow = FancyArrowPatch(
        (2.7, 2.1),
        (7.4, 2.1),
        arrowstyle="->",
        connectionstyle="arc3,rad=-0.10",
        color="#D9534F",
        lw=1.8,
    )
    ax.add_patch(breakup_arrow)

    # Dedicated central card for breakup detection and quarantine rule (fits inside [3.8, 7.2] gap, below the red arc)
    breakup_card = (
        r"$\mathbf{Breakup\ Detected:}$"
        "\n"
        r"$\frac{\left|\sum A_k - A_P\right|}{A_P} \leq 0.35$"
        "\n"
        r"$\widetilde{C}_{P,:},\ \widetilde{C}_{:,D_k} \leftarrow C_{\mathrm{large}}\ \mathrm{(Quarantine)}$"
    )
    ax.text(
        5.5,
        1.35,
        breakup_card,
        ha="center",
        va="center",
        fontsize=9.0,
        color="#A94442",
        bbox={
            "boxstyle": "round,pad=0.30",
            "facecolor": "#FFF8F8",
            "edgecolor": "#D9534F",
            "lw": 0.9,
            "alpha": 0.95,
        },
    )

    ax.set_xlim(0, 11)
    ax.set_ylim(0.2, 7.1)
    ax.axis("off")

    plt.tight_layout()
    pdf_path = OUTPUT_DIR / "tracking_breakup_schematic.pdf"
    png_path = OUTPUT_DIR / "tracking_breakup_schematic.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DONE] Saved {pdf_path} and {png_path}")


def generate_dfd_curves_regularization():
    """Generates Figure grounding Section 2.6: DFD universal inversion curves and smoothstep regularization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=300)

    # ----------------------------------------------------
    # Subplot 1: Rao et al. Universal Curves f1 and f2
    # ----------------------------------------------------
    sigma_tilde = np.linspace(0.005, 0.45, 300)
    G_tilde = 0.35 / (sigma_tilde**0.85 + 0.01)
    rho_tilde = 0.5 + 0.18 * sigma_tilde**0.65

    color_g = "#D9534F"
    color_r = "#0275D8"

    ax1.plot(
        sigma_tilde,
        G_tilde,
        color=color_g,
        lw=2.0,
        label=r"$\widetilde{G} = f_2(\widetilde{\sigma})$",
    )
    ax1.set_xlabel(r"Dimensionless blur $\tilde{\sigma} = \sigma / d_0$")
    ax1.set_ylabel(r"Dimensionless boundary gradient $\tilde{G}$", color=color_g)
    ax1.tick_params(axis="y", labelcolor=color_g)
    ax1.set_ylim(0, 18)

    # Twin axis for rho_tilde
    ax1_twin = ax1.twinx()
    ax1_twin.plot(
        sigma_tilde,
        rho_tilde,
        color=color_r,
        lw=1.8,
        ls="--",
        label=r"$\tilde{\rho} = f_1(\tilde{\sigma})$",
    )
    ax1_twin.set_ylabel(r"Inflection ratio $\tilde{\rho} = r_t / d_0$", color=color_r)
    ax1_twin.tick_params(axis="y", labelcolor=color_r)
    ax1_twin.set_ylim(0.45, 0.70)

    # Highlight blind zone sigma < 0.05
    ax1.axvspan(0, 0.05, color="#F0AD4E", alpha=0.25)
    ax1.axvline(0.05, color="#EEA236", ls=":", lw=1.2)
    # Move square to the right of the yellow area with pointer arrow to eliminate any overlap with the steep red curve
    ax1.annotate(
        "Near-wall\nBlind zone\n" r"$\tilde{\sigma} < 0.05$",
        xy=(0.04, 14.8),
        xytext=(0.14, 14.8),
        arrowprops={"arrowstyle": "->", "color": "#EEA236", "lw": 1.2},
        ha="center",
        va="center",
        fontsize=7.2,
        color="#8A6D3B",
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "#FFFBF0",
            "edgecolor": "#EEA236",
            "lw": 0.7,
            "alpha": 0.95,
        },
    )

    # Highlight upper saturation region sigma > 0.35 (Rao et al. 2024 critical threshold)
    ax1.axvspan(0.35, 0.45, color="#D9534F", alpha=0.20)
    ax1.axvline(0.35, color="#D9534F", ls="--", lw=1.2)
    ax1.annotate(
        "Critical Limit\n" r"$\tilde{\sigma} > 0.35$" "\n(Rear wall / Saturated)",
        xy=(0.35, 4.2),
        xytext=(0.20, 4.2),
        arrowprops={"arrowstyle": "->", "color": "#D9534F", "lw": 1.1},
        ha="center",
        va="center",
        fontsize=6.5,
        color="#A94442",
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.20",
            "facecolor": "#FFF8F8",
            "edgecolor": "#D9534F",
            "lw": 0.7,
            "alpha": 0.95,
        },
    )

    ax1.set_title(
        r"$\mathbf{(a)}$ Universal Analytical DFD Inversion", fontsize=9.5, color="#1B365D"
    )
    ax1.grid(True, ls=":", alpha=0.4)

    # ----------------------------------------------------
    # Subplot 2: Smoothstep Near-Wall Regularization S(u)
    # ----------------------------------------------------
    u = np.linspace(0, 1.0, 200)
    S_u = 3 * u**2 - 2 * u**3  # C1 continuous smoothstep

    ax2.plot(u, S_u, color="#1B365D", lw=2.2, label=r"$S(u) = 3u^2 - 2u^3$ (Smoothstep)")
    ax2.plot(u, u, color="gray", lw=1.0, ls=":", label=r"Linear ramp (discontinuous $C^0$)")

    # Physical boundary points
    ax2.plot(0, 0, "o", color="#D9534F", ms=6)
    # Place Front Wall card cleanly in the open lower region (u=0.48, y=0.07) well below the curve S(0.48)=0.48
    ax2.annotate(
        r"$\mathbf{Front\ wall\ (z \to 0):}$" "\n" r"$S(0)=0 \to d_0 = 2r_t$",
        xy=(0.02, 0.01),
        xytext=(0.48, 0.07),
        arrowprops={
            "arrowstyle": "->",
            "color": "#D9534F",
            "lw": 1.1,
            "connectionstyle": "arc3,rad=-0.04",
        },
        fontsize=7.5,
        color="#C9302C",
        ha="center",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "#FFF8F8",
            "edgecolor": "#D9534F",
            "lw": 0.8,
            "alpha": 0.95,
        },
    )

    ax2.plot(1.0, 1.0, "o", color="#5CB85C", ms=6)
    ax2.annotate(
        r"$\mathbf{Core\ flow\ (\tilde{\sigma} \geq 0.05):}$"
        "\n"
        r"$S(1)=1 \to \mathrm{Full\ DFD}$",
        xy=(1.0, 1.0),
        xytext=(0.80, 0.42),
        arrowprops={
            "arrowstyle": "->",
            "color": "#3C763D",
            "lw": 1.1,
            "connectionstyle": "arc3,rad=0.05",
        },
        fontsize=7.5,
        color="#2E6B30",
        ha="center",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "#F4FAF4",
            "edgecolor": "#4CAE4C",
            "lw": 0.8,
            "alpha": 0.95,
        },
    )

    ax2.set_xlabel(r"Normalized blur parameter $u = \min(1, \, \tilde{\sigma} / 0.05)$")
    ax2.set_ylabel(r"Regularization weight $S(u)$")
    ax2.set_title(
        r"$\mathbf{(b)}$ Near-Wall Regularizer ($C^1$ Smoothstep)", fontsize=9.5, color="#1B365D"
    )
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.05)
    # Legend in completely free upper-left space
    ax2.legend(
        loc="upper left",
        bbox_to_anchor=(0.04, 0.96),
        fontsize=7.8,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#D0D0D0",
    )
    ax2.grid(True, ls=":", alpha=0.4)

    plt.tight_layout()
    pdf_path = OUTPUT_DIR / "dfd_curves_regularization.pdf"
    png_path = OUTPUT_DIR / "dfd_curves_regularization.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DONE] Saved {pdf_path} and {png_path}")


def generate_dfd_tube_saturation_cross_section():
    """Generates Figure grounding Section 2.6.3 and Section 3:
    2D circular tube cross-section (y-z plane) showing the 3.0 mm borosilicate wall,
    optical focal plane (z_f = 0), and critical saturation zones for d_0 = 3 mm, 5 mm, and 6 mm bubbles.
    Strict zero-collision layout with bottom legend placed outside geometric canvas.
    """
    fig, ax = plt.subplots(figsize=(8.0, 5.6), dpi=300)

    r_inner = 10.0  # mm
    wall_th = 3.0  # mm
    r_outer = r_inner + wall_th  # 13.0 mm
    z_center = wall_th + r_inner  # 13.0 mm

    theta_full = np.linspace(0, 2 * np.pi, 500)
    z_in = z_center + r_inner * np.cos(theta_full)
    y_in = r_inner * np.sin(theta_full)

    z_out = z_center + r_outer * np.cos(theta_full)
    y_out = r_outer * np.sin(theta_full)

    # 1. Glass tube wall
    ax.fill(
        z_out,
        y_out,
        color="#ECEFF1",
        alpha=0.9,
        label=r"Borosilicate Wall ($e = 3.0\ \mathrm{mm}$)",
    )
    ax.fill(
        z_in,
        y_in,
        color="#FFFFFF",
        alpha=1.0,
        edgecolor="#1B365D",
        lw=2.4,
        label=r"$\mathbf{d_0 \geq 7.66\ mm:}\ \mathrm{Resolvable}\ (100\%\ \mathrm{vol})$",
    )
    ax.plot(z_out, y_out, color="#78909C", lw=1.5, ls="--")

    # Helper function to compute saturated crescent polygons
    def get_sat_patch(z_cut):
        ys = np.linspace(-r_inner, r_inner, 500)
        valid_y = ys[np.sqrt(np.maximum(0, r_inner**2 - ys**2)) >= (z_cut - z_center)]
        if len(valid_y) == 0:
            return None
        y_min, y_max = valid_y[0], valid_y[-1]
        y_arr = np.linspace(y_min, y_max, 250)
        z_rear = z_center + np.sqrt(np.maximum(0, r_inner**2 - y_arr**2))

        verts = [(z_cut, y_min)]
        for y_val, z_val in zip(y_arr, z_rear):
            verts.append((z_val, y_val))
        verts.append((z_cut, y_max))
        verts.append((z_cut, y_min))
        return np.array(verts)

    # Saturated crescent zones
    # 3 mm bubble: z_crit = 7.83 mm -> z_cut = 3.0 + 7.83 = 10.83 mm (51.5% unresolvable)
    v_3mm = get_sat_patch(wall_th + 7.83)
    if v_3mm is not None:
        poly_3 = Polygon(
            v_3mm,
            closed=True,
            facecolor="#90CAF9",
            alpha=0.45,
            hatch="//",
            edgecolor="#1565C0",
            lw=1.3,
            ls="--",
            label=r"$\mathbf{d_0 = 3.0\ mm:}\ \mathrm{Saturated}\ (51.5\%\ \mathrm{vol})$",
        )
        ax.add_patch(poly_3)

    # 5 mm bubble: z_crit = 13.05 mm -> z_cut = 3.0 + 13.05 = 16.05 mm (23.3% unresolvable)
    v_5mm = get_sat_patch(wall_th + 13.05)
    if v_5mm is not None:
        poly_5 = Polygon(
            v_5mm,
            closed=True,
            facecolor="#FFB74D",
            alpha=0.60,
            hatch="\\\\",
            edgecolor="#E65100",
            lw=1.4,
            ls="-.",
            label=r"$\mathbf{d_0 = 5.0\ mm:}\ \mathrm{Saturated}\ (23.3\%\ \mathrm{vol})$",
        )
        ax.add_patch(poly_5)

    # 6 mm bubble: z_crit = 15.66 mm -> z_cut = 3.0 + 15.66 = 18.66 mm (11.7% unresolvable)
    v_6mm = get_sat_patch(wall_th + 15.66)
    if v_6mm is not None:
        poly_6 = Polygon(
            v_6mm,
            closed=True,
            facecolor="#EF9A9A",
            alpha=0.75,
            hatch="xx",
            edgecolor="#C62828",
            lw=1.5,
            ls=":",
            label=r"$\mathbf{d_0 = 6.0\ mm:}\ \mathrm{Saturated}\ (11.7\%\ \mathrm{vol})$",
        )
        ax.add_patch(poly_6)

    # Internal vertical cut lines
    ax.plot(
        [wall_th + 7.83, wall_th + 7.83], [-9.7, 9.7], color="#1565C0", ls="--", lw=1.1, alpha=0.7
    )
    ax.plot(
        [wall_th + 13.05, wall_th + 13.05], [-9.5, 9.5], color="#E65100", ls="-.", lw=1.1, alpha=0.7
    )
    ax.plot(
        [wall_th + 15.66, wall_th + 15.66], [-8.9, 8.9], color="#C62828", ls=":", lw=1.2, alpha=0.7
    )

    # 2. Optical Focal Plane (z = 0)
    ax.plot([0.0, 0.0], [-13.5, 14.0], color="#D9534F", lw=2.2, ls="-.")
    ax.text(
        -0.6,
        5.0,
        r"$\mathbf{Optical\ Focal\ Plane\ }(z_f = 0)$",
        rotation=90,
        va="center",
        ha="right",
        fontsize=9.2,
        color="#C9302C",
        fontweight="bold",
    )

    # 3. Fluid front wall interface (z = 3.0 mm)
    ax.plot([wall_th, wall_th], [-13.5, 14.0], color="#1B365D", lw=1.2, ls=":")

    # 4. TOP DIMENSIONING ZONE (y >= 14.5)
    y_dim_top = 15.0
    # Extension lines
    ax.plot([0, 0], [0.0, y_dim_top + 1.2], color="#78909C", ls=":", lw=0.9)
    ax.plot([wall_th, wall_th], [10.0, y_dim_top + 1.2], color="#78909C", ls=":", lw=0.9)
    ax.plot(
        [wall_th + 2 * r_inner, wall_th + 2 * r_inner],
        [10.0, y_dim_top + 1.2],
        color="#1B365D",
        ls=":",
        lw=0.9,
    )

    # Wall thickness e = 3.0 mm
    ax.annotate(
        "",
        xy=(0, y_dim_top),
        xytext=(wall_th, y_dim_top),
        arrowprops={"arrowstyle": "<->", "color": "#37474F", "lw": 1.2},
    )
    ax.text(
        wall_th / 2.0,
        y_dim_top + 0.6,
        r"$e = 3.0\ \mathrm{mm}$",
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="#37474F",
        fontweight="bold",
    )

    # Inner tube diameter dimension D_i = 20.0 mm (two compact lines)
    ax.annotate(
        "",
        xy=(wall_th, y_dim_top),
        xytext=(wall_th + 2 * r_inner, y_dim_top),
        arrowprops={"arrowstyle": "<->", "color": "#1B365D", "lw": 1.4},
    )
    ax.text(
        wall_th + r_inner,
        y_dim_top + 0.6,
        r"$\mathbf{Inner\ Diameter}$" "\n" r"$(D_i = 20.0\ \mathrm{mm})$",
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="#1B365D",
        fontweight="bold",
    )

    # 5. BOTTOM DIMENSIONING ZONE (y in [-20.5, -14.5])
    # Extension line from front inner wall z = 3.0 mm
    ax.plot([wall_th, wall_th], [-10.0, -19.5], color="#1B365D", ls=":", lw=0.9)
    # Extension lines from cut positions
    ax.plot([wall_th + 7.83, wall_th + 7.83], [-9.8, -16.2], color="#1565C0", ls=":", lw=0.9)
    ax.plot([wall_th + 13.05, wall_th + 13.05], [-9.5, -19.2], color="#E65100", ls=":", lw=0.9)

    # Dimension 1: z_crit = 7.83 mm at y = -15.5
    y_dim1 = -15.5
    ax.annotate(
        "",
        xy=(wall_th, y_dim1),
        xytext=(wall_th + 7.83, y_dim1),
        arrowprops={"arrowstyle": "<->", "color": "#1565C0", "lw": 1.2},
    )
    ax.text(
        7.3,
        y_dim1 - 0.4,
        r"$z_{\mathrm{crit}} = 7.83\ \mathrm{mm}$",
        ha="center",
        va="top",
        fontsize=8.4,
        color="#0D47A1",
        fontweight="bold",
    )

    # Dimension 2: z_crit = 13.05 mm at y = -18.5
    y_dim2 = -18.5
    ax.annotate(
        "",
        xy=(wall_th, y_dim2),
        xytext=(wall_th + 13.05, y_dim2),
        arrowprops={"arrowstyle": "<->", "color": "#E65100", "lw": 1.2},
    )
    ax.text(
        (wall_th + wall_th + 13.05) / 2.0,
        y_dim2 - 0.4,
        r"$z_{\mathrm{crit}} = 13.05\ \mathrm{mm}$",
        ha="center",
        va="top",
        fontsize=8.8,
        color="#BF360C",
        fontweight="bold",
    )

    # 6. LEFT ZONE: Near-Wall Regularization Callout
    ax.annotate(
        r"$\mathbf{Near\mathrm{-}Wall\ Regularization}$"
        "\n"
        r"$(\widetilde{\sigma} < 0.05 \to \mathrm{Smoothstep})$",
        xy=(wall_th, -2.5),
        xytext=(-7.5, -7.5),
        arrowprops={
            "arrowstyle": "->",
            "color": "#2E7D32",
            "lw": 1.3,
            "connectionstyle": "arc3,rad=-0.14",
        },
        fontsize=8.6,
        color="#1B5E20",
        ha="center",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "#E8F5E9",
            "edgecolor": "#4CAF50",
            "lw": 0.8,
            "alpha": 0.95,
        },
    )

    # 7. Axes limits and styling
    ax.set_aspect("equal")
    ax.set_xlabel(
        r"Optical depth coordinate $z$ from outer focal plane [$\mathrm{mm}$]", fontsize=9.5
    )
    ax.set_ylabel(r"Transverse vertical elevation $y$ [$\mathrm{mm}$]", fontsize=9.5)
    ax.set_title(
        r"$\mathbf{Cross\mathrm{-}Sectional\ DFD\ Saturation\ Map\ (y\mathrm{-}z\ Plane)}$",
        fontsize=11.0,
        color="#1B365D",
        pad=12,
    )

    # Balanced xlim: centered tube from z = -15.5 to z = 30.5
    ax.set_xlim(-15.5, 30.5)
    ax.set_ylim(-21.5, 18.5)
    ax.grid(True, ls=":", alpha=0.35)

    # 8. LEGEND PLACED AT THE BOTTOM (below the figure, 0% overlap with geometry)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        fontsize=8.5,
        framealpha=0.98,
        facecolor="white",
        edgecolor="#90A4AE",
    )

    pdf_path = OUTPUT_DIR / "dfd_tube_saturation_cross_section.pdf"
    png_path = OUTPUT_DIR / "dfd_tube_saturation_cross_section.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DONE] Saved {pdf_path} and {png_path}")


def generate_dfd_dual_camera_cross_section():
    """Generates Figure for Dual-Camera DFD Coverage:
    Two-panel schematic:
      Panel (a): Small Bubbles (d0 = 3.0 mm, z_crit = 7.83 mm) showing planar reach from both walls,
                 unreached central core gap of 4.34 mm (27.4% blind band), and 72.6% total coverage.
      Panel (b): Intermediate Bubbles (d0 = 5.0 mm, z_crit = 13.05 mm) showing 100% total coverage,
                 0% blind volume, and 38.2% dual-camera stereoscopic overlap.
    Cameras:
      - Camera 1: primary camera at theta_1 = 180 deg (optical axis along 0 deg).
      - Camera 2: secondary camera at theta_2 = 350 deg (optical axis along 170 deg,
                  10 deg angular offset from Camera 1 backlight axis to eliminate optical flare).
    Strict zero visual collisions adhering to ASME/ISO drafting standards and planar focal physics.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.6, 7.2), dpi=300)

    r_inner = 10.0  # mm
    wall_th = 3.0  # mm
    r_outer = r_inner + wall_th  # 13.0 mm
    z_center = wall_th + r_inner  # 13.0 mm

    theta_full = np.linspace(0, 2 * np.pi, 500)
    z_in = z_center + r_inner * np.cos(theta_full)
    y_in = r_inner * np.sin(theta_full)
    z_out = z_center + r_outer * np.cos(theta_full)
    y_out = r_outer * np.sin(theta_full)

    theta2_deg = 350.0
    theta2 = np.deg2rad(theta2_deg)
    r2_hat = np.array([np.cos(theta2), np.sin(theta2)])
    u2_hat = -r2_hat
    perp2 = np.array([-r2_hat[1], r2_hat[0]])

    N = 1000
    y_grid = np.linspace(-r_inner, r_inner, N)
    z_grid = np.linspace(z_center - r_inner, z_center + r_inner, N)
    Z, Y = np.meshgrid(z_grid, y_grid)
    in_tube = (Z - z_center) ** 2 + Y**2 <= r_inner**2
    tube_pts = np.sum(in_tube)

    panels = [
        (
            ax1,
            3.0,
            r"$\mathbf{(a)\ Small\ Bubbles\ (d_0 = 3.0\ mm,\ z_{\mathrm{crit}} = 7.83\ mm)}$",
        ),
        (
            ax2,
            5.0,
            r"$\mathbf{(b)\ Intermediate\ Bubbles\ (d_0 = 5.0\ mm,\ z_{\mathrm{crit}} = 13.05\ mm)}$",
        ),
    ]

    for ax, d0_val, title_text in panels:
        z_crit = 2.61 * d0_val

        # Wall
        ax.fill(z_out, y_out, color="#ECEFF1", alpha=0.9)
        ax.fill(z_in, y_in, color="#FAFAFA", alpha=1.0)
        ax.plot(z_out, y_out, color="#78909C", lw=1.3, ls="--")

        # Planar depth calculation (collimated planar wavefronts):
        # Cam 1 planar reach: Z <= wall_th + z_crit
        res1 = in_tube & (wall_th + z_crit >= Z)

        # Cam 2 planar reach: projection along r2_hat >= r_inner - z_crit
        proj2 = (Z - z_center) * r2_hat[0] + Y * r2_hat[1]
        res2 = in_tube & (proj2 >= r_inner - z_crit)

        both = res1 & res2
        c1_only = res1 & (~res2)
        c2_only = res2 & (~res1)
        neither = in_tube & (~res1) & (~res2)

        pct_c1 = np.sum(c1_only) / tube_pts * 100
        np.sum(c2_only) / tube_pts * 100
        pct_ov = np.sum(both) / tube_pts * 100
        pct_blind = np.sum(neither) / tube_pts * 100
        pct_tot = np.sum(res1 | res2) / tube_pts * 100

        # Shading regions
        if np.sum(c1_only) > 0:
            ax.contourf(
                Z, Y, c1_only.astype(float), levels=[0.5, 1.5], colors=["#BBDEFB"], alpha=0.75
            )
        if np.sum(c2_only) > 0:
            ax.contourf(
                Z, Y, c2_only.astype(float), levels=[0.5, 1.5], colors=["#FFE0B2"], alpha=0.75
            )
        if np.sum(both) > 0:
            ax.contourf(Z, Y, both.astype(float), levels=[0.5, 1.5], colors=["#C8E6C9"], alpha=0.85)
            ax.contour(Z, Y, both.astype(float), levels=[0.5], colors=["#2E7D32"], linewidths=1.3)
        if np.sum(neither) > 0:
            ax.contourf(
                Z,
                Y,
                neither.astype(float),
                levels=[0.5, 1.5],
                colors=["#CFD8DC"],
                alpha=0.6,
                hatches=["xx"],
            )
            ax.contour(
                Z,
                Y,
                neither.astype(float),
                levels=[0.5],
                colors=["#546E7A"],
                linewidths=1.2,
                linestyles=":",
            )

        # Internal planar boundary lines
        z1_cut = wall_th + z_crit
        if z1_cut < z_center + r_inner:
            y_cut1 = np.sqrt(np.maximum(0.0, r_inner**2 - (z1_cut - z_center) ** 2))
            ax.plot([z1_cut, z1_cut], [-y_cut1, y_cut1], color="#1565C0", lw=1.6, ls="--")

        s_cut2 = r_inner - z_crit
        if abs(s_cut2) < r_inner:
            t_max = np.sqrt(r_inner**2 - s_cut2**2)
            p_mid2 = np.array([z_center, 0]) + s_cut2 * r2_hat
            pt_a = p_mid2 - t_max * perp2
            pt_b = p_mid2 + t_max * perp2
            ax.plot([pt_a[0], pt_b[0]], [pt_a[1], pt_b[1]], color="#E65100", lw=1.6, ls="-.")

        # Re-stroke inner wall
        ax.plot(z_in, y_in, color="#1B365D", lw=2.3)
        ax.plot(z_center, 0, "+", color="#78909C", ms=7, mew=1.1)

        # Optical axes - starting directly from camera cards
        ax.annotate(
            "",
            xy=(z_center + 1.5, 0),
            xytext=(-5.0, 0),
            arrowprops={"arrowstyle": "-|>", "color": "#1565C0", "lw": 1.4, "mutation_scale": 11},
        )
        p2_start = np.array([z_center, 0]) + 21.0 * r2_hat
        p2_end = np.array([z_center, 0]) + 1.5 * u2_hat
        ax.annotate(
            "",
            xy=(p2_end[0], p2_end[1]),
            xytext=(p2_start[0], p2_start[1]),
            arrowprops={"arrowstyle": "-|>", "color": "#E65100", "lw": 1.4, "mutation_scale": 11},
        )

        # Focal planes
        ax.plot([0.0, 0.0], [-13.5, 13.5], color="#1565C0", lw=1.8, ls="-.")
        ax.text(
            -0.6,
            6.0,
            r"$\mathbf{z_{f,1} = 0}$",
            rotation=90,
            va="center",
            ha="right",
            fontsize=8.5,
            color="#0D47A1",
            fontweight="bold",
        )

        p2_outer = np.array([z_center, 0]) + r_outer * r2_hat
        fp2_start = p2_outer - 13.5 * perp2
        fp2_end = p2_outer + 13.5 * perp2
        ax.plot(
            [fp2_start[0], fp2_end[0]], [fp2_start[1], fp2_end[1]], color="#E65100", lw=1.8, ls="-."
        )
        p2_txt = p2_outer + 9.0 * perp2 + 2.0 * r2_hat
        ax.text(
            p2_txt[0],
            p2_txt[1],
            r"$\mathbf{z_{f,2} = 0\ (350^\circ)}$",
            ha="left",
            va="center",
            fontsize=8.5,
            color="#BF360C",
            fontweight="bold",
        )

        # Top dimensioning: e = 3 mm, Di = 20 mm
        y_top = 15.2
        ax.plot([0, 0], [0, y_top + 1.0], color="#78909C", ls=":", lw=0.8)
        ax.plot([wall_th, wall_th], [10, y_top + 1.0], color="#78909C", ls=":", lw=0.8)
        ax.plot(
            [wall_th + 2 * r_inner, wall_th + 2 * r_inner],
            [10, y_top + 1.0],
            color="#1B365D",
            ls=":",
            lw=0.8,
        )

        ax.annotate(
            "",
            xy=(0, y_top),
            xytext=(wall_th, y_top),
            arrowprops={"arrowstyle": "<->", "color": "#37474F", "lw": 1.1},
        )
        ax.text(
            wall_th / 2.0,
            y_top + 0.5,
            r"$e = 3.0\ \mathrm{mm}$",
            ha="center",
            va="bottom",
            fontsize=8.4,
            color="#37474F",
            fontweight="bold",
        )

        ax.annotate(
            "",
            xy=(wall_th, y_top),
            xytext=(wall_th + 2 * r_inner, y_top),
            arrowprops={"arrowstyle": "<->", "color": "#1B365D", "lw": 1.2},
        )
        ax.text(
            wall_th + r_inner,
            y_top + 0.5,
            r"$D_i = 20.0\ \mathrm{mm}$",
            ha="center",
            va="bottom",
            fontsize=8.4,
            color="#1B365D",
            fontweight="bold",
        )

        # Camera callouts
        ax.text(
            -5.5,
            -3.5,
            r"$\mathbf{Cam\ 1}\ (180^\circ)$" "\n" r"$\mathrm{Axis:\ } 0^\circ$",
            ha="center",
            va="top",
            fontsize=8.2,
            color="#0D47A1",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "#E3F2FD",
                "edgecolor": "#1565C0",
                "lw": 0.7,
            },
        )

        ax.text(
            35.5,
            -3.5,
            r"$\mathbf{Cam\ 2}\ (350^\circ)$" "\n" r"$\mathrm{Axis:\ } 170^\circ$",
            ha="center",
            va="top",
            fontsize=8.2,
            color="#BF360C",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "#FFF3E0",
                "edgecolor": "#E65100",
                "lw": 0.7,
            },
        )

        # Bilateral bottom dimensions (ASME/ISO standards)
        y_dim_c1 = -15.5
        y_dim_c2 = -18.2

        z_front_in = wall_th  # 3.0 mm
        z_rear_in = wall_th + 2 * r_inner  # 23.0 mm
        z_pen_c1 = z_front_in + z_crit
        z_pen_c2 = z_rear_in - z_crit

        # Cam 1 extension & dimension
        ax.plot([z_front_in, z_front_in], [-10.0, y_dim_c1 - 0.7], color="#1B365D", ls=":", lw=0.8)
        ax.plot([z_pen_c1, z_pen_c1], [-9.5, y_dim_c1 - 0.7], color="#1565C0", ls=":", lw=0.8)
        ax.annotate(
            "",
            xy=(z_front_in, y_dim_c1),
            xytext=(z_pen_c1, y_dim_c1),
            arrowprops={"arrowstyle": "<->", "color": "#1565C0", "lw": 1.1},
        )
        ax.text(
            (z_front_in + z_pen_c1) / 2.0,
            y_dim_c1 - 0.4,
            rf"$z_{{\mathrm{{crit}},1}} = {z_crit:.2f}\ \mathrm{{mm}}$",
            ha="center",
            va="top",
            fontsize=8.2,
            color="#0D47A1",
            fontweight="bold",
        )

        # Cam 2 extension & dimension
        ax.plot([z_rear_in, z_rear_in], [-10.0, y_dim_c2 - 0.7], color="#1B365D", ls=":", lw=0.8)
        ax.plot([z_pen_c2, z_pen_c2], [-9.5, y_dim_c2 - 0.7], color="#E65100", ls=":", lw=0.8)
        ax.annotate(
            "",
            xy=(z_rear_in, y_dim_c2),
            xytext=(z_pen_c2, y_dim_c2),
            arrowprops={"arrowstyle": "<->", "color": "#E65100", "lw": 1.1},
        )
        ax.text(
            (z_rear_in + z_pen_c2) / 2.0,
            y_dim_c2 - 0.4,
            rf"$z_{{\mathrm{{crit}},2}} = {z_crit:.2f}\ \mathrm{{mm}}$",
            ha="center",
            va="top",
            fontsize=8.2,
            color="#BF360C",
            fontweight="bold",
        )

        # Coverage header card placed in open space (y = 18.8)
        if pct_blind == 0.0:
            header_str = (
                r"$\mathbf{Total\ Coverage:\ 100.0\%\ (0.0\%\ Blind)}$"
                "\n"
                rf"$\mathrm{{Dual\ Overlap\ (Stereo):}}\ \mathbf{{{pct_ov:.1f}\%}}\ |\ \mathrm{{Single\ Cam:}}\ {pct_c1:.1f}\%\ \mathrm{{each}}$"
            )
            h_edge, h_bg, h_fg = "#2E7D32", "#E8F5E9", "#1B5E20"
        else:
            unreached_gap = z_pen_c2 - z_pen_c1
            header_str = (
                rf"$\mathbf{{Total\ Coverage:\ {pct_tot:.1f}\%\ (Blind\ Band:\ {pct_blind:.1f}\%)}}$"
                "\n"
                rf"$\mathrm{{Single\ Cam:\ {pct_c1:.1f}\%\ each}}\ |\ \mathrm{{Unreached\ Gap:}}\ \mathbf{{{unreached_gap:.2f}\ mm}}$"
            )
            h_edge, h_bg, h_fg = "#1565C0", "#E3F2FD", "#0D47A1"

        ax.text(
            z_center,
            18.8,
            header_str,
            ha="center",
            va="center",
            fontsize=8.5,
            color=h_fg,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": h_bg, "edgecolor": h_edge, "lw": 0.8},
        )

        ax.set_aspect("equal")
        ax.set_xlabel(r"Optical depth coordinate $z$ [$\mathrm{mm}$]", fontsize=9.5)
        ax.set_ylabel(r"Transverse elevation $y$ [$\mathrm{mm}$]", fontsize=9.5)
        ax.set_title(title_text, fontsize=10.5, color="#1B365D", pad=12)
        ax.set_xlim(-12.0, 39.0)
        ax.set_ylim(-21.5, 21.5)
        ax.grid(True, ls=":", alpha=0.35)

    # Global Bottom Legend
    legend_elements = [
        Patch(
            facecolor="#ECEFF1",
            edgecolor="#78909C",
            label=r"Borosilicate Wall ($e = 3.0\ \mathrm{mm}$)",
        ),
        Patch(
            facecolor="#BBDEFB",
            edgecolor="#1976D2",
            label=r"$\mathbf{Cam\ 1\ Only}\ (\theta_1 = 180^\circ)$",
        ),
        Patch(
            facecolor="#C8E6C9",
            edgecolor="#2E7D32",
            label=r"$\mathbf{Dual\ Overlap\ (Stereo\ Depth)}$",
        ),
        Patch(
            facecolor="#FFE0B2",
            edgecolor="#F57C00",
            label=r"$\mathbf{Cam\ 2\ Only\ (Recovered\ 350^\circ)}$",
        ),
        Patch(
            facecolor="#CFD8DC",
            edgecolor="#546E7A",
            hatch="xx",
            label=r"$\mathbf{Blind\ Core\ (Unreached\ Gap)}$",
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=5,
        fontsize=8.5,
        framealpha=0.98,
        facecolor="white",
        edgecolor="#90A4AE",
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10)
    pdf_path = OUTPUT_DIR / "dfd_dual_camera_cross_section.pdf"
    png_path = OUTPUT_DIR / "dfd_dual_camera_cross_section.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DONE] Saved {pdf_path} and {png_path}")


def generate_dfd_dual_camera_single_cross_section():
    """Generates Single-Panel Figure for Dual-Camera DFD Coverage (d_0 = 5.0 mm):
    Shows 100% total coverage, 0% blind volume, and 38.2% dual-camera stereoscopic overlap.
    """
    fig, ax = plt.subplots(figsize=(8.8, 5.8), dpi=300)

    r_inner = 10.0  # mm
    wall_th = 3.0  # mm
    r_outer = r_inner + wall_th  # 13.0 mm
    z_center = wall_th + r_inner  # 13.0 mm

    theta_full = np.linspace(0, 2 * np.pi, 500)
    z_in = z_center + r_inner * np.cos(theta_full)
    y_in = r_inner * np.sin(theta_full)
    z_out = z_center + r_outer * np.cos(theta_full)
    y_out = r_outer * np.sin(theta_full)

    theta2_deg = 350.0
    theta2 = np.deg2rad(theta2_deg)
    r2_hat = np.array([np.cos(theta2), np.sin(theta2)])
    u2_hat = -r2_hat
    perp2 = np.array([-r2_hat[1], r2_hat[0]])

    N = 1000
    y_grid = np.linspace(-r_inner, r_inner, N)
    z_grid = np.linspace(z_center - r_inner, z_center + r_inner, N)
    Z, Y = np.meshgrid(z_grid, y_grid)
    in_tube = (Z - z_center) ** 2 + Y**2 <= r_inner**2
    tube_pts = np.sum(in_tube)

    d0_val = 5.0
    z_crit = 2.61 * d0_val

    # Wall
    ax.fill(z_out, y_out, color="#ECEFF1", alpha=0.9)
    ax.fill(z_in, y_in, color="#FAFAFA", alpha=1.0)
    ax.plot(z_out, y_out, color="#78909C", lw=1.3, ls="--")

    # Planar depth calculation:
    res1 = in_tube & (wall_th + z_crit >= Z)
    proj2 = (Z - z_center) * r2_hat[0] + Y * r2_hat[1]
    res2 = in_tube & (proj2 >= r_inner - z_crit)

    cam1_only = res1 & (~res2)
    cam2_only = res2 & (~res1)
    overlap = res1 & res2

    pct_c1 = np.sum(cam1_only) / tube_pts * 100
    pct_c2 = np.sum(cam2_only) / tube_pts * 100
    pct_ov = np.sum(overlap) / tube_pts * 100

    # Shading regions
    ax.contourf(Z, Y, cam1_only.astype(float), levels=[0.5, 1.5], colors=["#BBDEFB"], alpha=0.75)
    ax.contourf(Z, Y, cam2_only.astype(float), levels=[0.5, 1.5], colors=["#FFE0B2"], alpha=0.75)
    ax.contourf(Z, Y, overlap.astype(float), levels=[0.5, 1.5], colors=["#C8E6C9"], alpha=0.85)
    ax.contour(Z, Y, overlap.astype(float), levels=[0.5], colors=["#2E7D32"], linewidths=1.3)

    # Internal planar cut lines
    z1_cut = wall_th + z_crit
    if z1_cut < z_center + r_inner:
        y_cut1 = np.sqrt(np.maximum(0.0, r_inner**2 - (z1_cut - z_center) ** 2))
        ax.plot([z1_cut, z1_cut], [-y_cut1, y_cut1], color="#1565C0", lw=1.6, ls="--")

    s_cut2 = r_inner - z_crit
    if abs(s_cut2) < r_inner:
        t_max = np.sqrt(r_inner**2 - s_cut2**2)
        p_mid2 = np.array([z_center, 0]) + s_cut2 * r2_hat
        pt_a = p_mid2 - t_max * perp2
        pt_b = p_mid2 + t_max * perp2
        ax.plot([pt_a[0], pt_b[0]], [pt_a[1], pt_b[1]], color="#E65100", lw=1.6, ls="-.")

    # Re-stroke inner wall
    ax.plot(z_in, y_in, color="#1B365D", lw=2.3)
    ax.plot(z_center, 0, "+", color="#78909C", ms=7, mew=1.1)

    # Optical axes
    ax.annotate(
        "",
        xy=(z_center + 1.5, 0),
        xytext=(-5.0, 0),
        arrowprops={"arrowstyle": "-|>", "color": "#1565C0", "lw": 1.4, "mutation_scale": 11},
    )
    p2_start = np.array([z_center, 0]) + 21.0 * r2_hat
    p2_end = np.array([z_center, 0]) + 1.5 * u2_hat
    ax.annotate(
        "",
        xy=(p2_end[0], p2_end[1]),
        xytext=(p2_start[0], p2_start[1]),
        arrowprops={"arrowstyle": "-|>", "color": "#E65100", "lw": 1.4, "mutation_scale": 11},
    )

    # Focal planes
    ax.plot([0.0, 0.0], [-13.5, 13.5], color="#1565C0", lw=1.8, ls="-.")
    ax.text(
        -0.6,
        6.0,
        r"$\mathbf{z_{f,1} = 0}$",
        rotation=90,
        va="center",
        ha="right",
        fontsize=8.5,
        color="#0D47A1",
        fontweight="bold",
    )

    p2_outer = np.array([z_center, 0]) + r_outer * r2_hat
    fp2_start = p2_outer - 13.5 * perp2
    fp2_end = p2_outer + 13.5 * perp2
    ax.plot(
        [fp2_start[0], fp2_end[0]], [fp2_start[1], fp2_end[1]], color="#E65100", lw=1.8, ls="-."
    )
    p2_txt = p2_outer + 9.0 * perp2 + 2.0 * r2_hat
    ax.text(
        p2_txt[0],
        p2_txt[1],
        r"$\mathbf{z_{f,2} = 0\ (350^\circ)}$",
        ha="left",
        va="center",
        fontsize=8.5,
        color="#BF360C",
        fontweight="bold",
    )

    # Top dimensioning: e = 3 mm, Di = 20 mm
    y_top = 15.2
    ax.plot([0, 0], [0, y_top + 1.0], color="#78909C", ls=":", lw=0.8)
    ax.plot([wall_th, wall_th], [10, y_top + 1.0], color="#78909C", ls=":", lw=0.8)
    ax.plot(
        [wall_th + 2 * r_inner, wall_th + 2 * r_inner],
        [10, y_top + 1.0],
        color="#1B365D",
        ls=":",
        lw=0.8,
    )

    ax.annotate(
        "",
        xy=(0, y_top),
        xytext=(wall_th, y_top),
        arrowprops={"arrowstyle": "<->", "color": "#37474F", "lw": 1.1},
    )
    ax.text(
        wall_th / 2.0,
        y_top + 0.5,
        r"$e = 3.0\ \mathrm{mm}$",
        ha="center",
        va="bottom",
        fontsize=8.4,
        color="#37474F",
        fontweight="bold",
    )

    ax.annotate(
        "",
        xy=(wall_th, y_top),
        xytext=(wall_th + 2 * r_inner, y_top),
        arrowprops={"arrowstyle": "<->", "color": "#1B365D", "lw": 1.2},
    )
    ax.text(
        wall_th + r_inner,
        y_top + 0.5,
        r"$D_i = 20.0\ \mathrm{mm}$",
        ha="center",
        va="bottom",
        fontsize=8.4,
        color="#1B365D",
        fontweight="bold",
    )

    # Camera callouts
    ax.text(
        -5.5,
        -3.5,
        r"$\mathbf{Cam\ 1}\ (180^\circ)$" "\n" r"$\mathrm{Axis:\ } 0^\circ$",
        ha="center",
        va="top",
        fontsize=8.2,
        color="#0D47A1",
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "#E3F2FD",
            "edgecolor": "#1565C0",
            "lw": 0.7,
        },
    )

    ax.text(
        35.5,
        -3.5,
        r"$\mathbf{Cam\ 2}\ (350^\circ)$" "\n" r"$\mathrm{Axis:\ } 170^\circ$",
        ha="center",
        va="top",
        fontsize=8.2,
        color="#BF360C",
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "#FFF3E0",
            "edgecolor": "#E65100",
            "lw": 0.7,
        },
    )

    # Bilateral bottom dimensions
    y_dim_c1 = -15.5
    y_dim_c2 = -18.2

    z_front_in = wall_th  # 3.0 mm
    z_rear_in = wall_th + 2 * r_inner  # 23.0 mm
    z_pen_c1 = z_front_in + z_crit
    z_pen_c2 = z_rear_in - z_crit

    ax.plot([z_front_in, z_front_in], [-10.0, y_dim_c1 - 0.7], color="#1B365D", ls=":", lw=0.8)
    ax.plot([z_pen_c1, z_pen_c1], [-9.5, y_dim_c1 - 0.7], color="#1565C0", ls=":", lw=0.8)
    ax.annotate(
        "",
        xy=(z_front_in, y_dim_c1),
        xytext=(z_pen_c1, y_dim_c1),
        arrowprops={"arrowstyle": "<->", "color": "#1565C0", "lw": 1.1},
    )
    ax.text(
        (z_front_in + z_pen_c1) / 2.0,
        y_dim_c1 - 0.4,
        rf"$z_{{\mathrm{{crit}},1}} = {z_crit:.2f}\ \mathrm{{mm}}$",
        ha="center",
        va="top",
        fontsize=8.2,
        color="#0D47A1",
        fontweight="bold",
    )

    ax.plot([z_rear_in, z_rear_in], [-10.0, y_dim_c2 - 0.7], color="#1B365D", ls=":", lw=0.8)
    ax.plot([z_pen_c2, z_pen_c2], [-9.5, y_dim_c2 - 0.7], color="#E65100", ls=":", lw=0.8)
    ax.annotate(
        "",
        xy=(z_rear_in, y_dim_c2),
        xytext=(z_pen_c2, y_dim_c2),
        arrowprops={"arrowstyle": "<->", "color": "#E65100", "lw": 1.1},
    )
    ax.text(
        (z_rear_in + z_pen_c2) / 2.0,
        y_dim_c2 - 0.4,
        rf"$z_{{\mathrm{{crit}},2}} = {z_crit:.2f}\ \mathrm{{mm}}$",
        ha="center",
        va="top",
        fontsize=8.2,
        color="#BF360C",
        fontweight="bold",
    )

    # Header banner
    header_str = (
        r"$\mathbf{Total\ Coverage:\ 100.0\%\ (0.0\%\ Blind)}$"
        "\n"
        rf"$\mathrm{{Dual\ Overlap\ (Stereo):}}\ \mathbf{{{pct_ov:.1f}\%}}\ |\ \mathrm{{Single\ Cam:}}\ {pct_c1:.1f}\%\ \mathrm{{each}}$"
    )
    ax.text(
        z_center,
        18.8,
        header_str,
        ha="center",
        va="center",
        fontsize=8.6,
        color="#1B5E20",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "#E8F5E9",
            "edgecolor": "#2E7D32",
            "lw": 0.8,
        },
    )

    ax.set_aspect("equal")
    ax.set_xlabel(
        r"Optical depth coordinate $z$ from primary focal plane [$\mathrm{mm}$]", fontsize=9.5
    )
    ax.set_ylabel(r"Transverse vertical elevation $y$ [$\mathrm{mm}$]", fontsize=9.5)
    ax.set_title(
        r"$\mathbf{Dual\mathrm{-}Camera\ DFD\ Cross\mathrm{-}Sectional\ Coverage\ (d_0 = 5.0\ mm)}$",
        fontsize=10.8,
        color="#1B365D",
        pad=12,
    )
    ax.set_xlim(-12.0, 39.0)
    ax.set_ylim(-21.5, 21.5)
    ax.grid(True, ls=":", alpha=0.35)

    legend_elements = [
        Patch(
            facecolor="#ECEFF1",
            edgecolor="#78909C",
            label=r"Borosilicate Wall ($e = 3.0\ \mathrm{mm}$)",
        ),
        Patch(
            facecolor="#BBDEFB",
            edgecolor="#1976D2",
            label=rf"$\mathbf{{Camera\ 1\ Only:}}\ {pct_c1:.1f}\%\ \mathrm{{vol}}$",
        ),
        Patch(
            facecolor="#C8E6C9",
            edgecolor="#2E7D32",
            label=rf"$\mathbf{{Dual\ Overlap\ (Stereo):}}\ {pct_ov:.1f}\%\ \mathrm{{vol}}$",
        ),
        Patch(
            facecolor="#FFE0B2",
            edgecolor="#F57C00",
            label=rf"$\mathbf{{Camera\ 2\ Only\ (Recovered):}}\ {pct_c2:.1f}\%\ \mathrm{{vol}}$",
        ),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        fontsize=8.5,
        framealpha=0.98,
        facecolor="white",
        edgecolor="#90A4AE",
    )

    pdf_path = OUTPUT_DIR / "dfd_dual_camera_single_cross_section.pdf"
    png_path = OUTPUT_DIR / "dfd_dual_camera_single_cross_section.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DONE] Saved {pdf_path} and {png_path}")


if __name__ == "__main__":
    generate_tube_chord_geometry()
    generate_tracking_breakup_schematic()
    generate_dfd_curves_regularization()
    generate_dfd_tube_saturation_cross_section()
    generate_dfd_dual_camera_cross_section()
    generate_dfd_dual_camera_single_cross_section()
