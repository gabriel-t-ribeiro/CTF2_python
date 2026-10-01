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
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Rectangle

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
    frame_w = 3.3
    0.5 + frame_w  # 3.8
    gap_x2 = 7.2
    frame_t = Rectangle((0.5, 0.5), frame_w, 5.8, facecolor="#F8F9FA", edgecolor="#495057", lw=1.5)
    frame_t1 = Rectangle(
        (gap_x2, 0.5), frame_w, 5.8, facecolor="#F8F9FA", edgecolor="#495057", lw=1.5
    )
    ax.add_patch(frame_t)
    ax.add_patch(frame_t1)

    ax.text(
        0.5 + frame_w / 2,
        6.6,
        r"$\mathbf{Frame\ t}$ ($N_t$ entities)",
        ha="center",
        fontsize=9.5,
        color="#1B365D",
    )
    ax.text(
        gap_x2 + frame_w / 2,
        6.6,
        r"$\mathbf{Frame\ t+1}$ ($N_{t+1}$ entities)",
        ha="center",
        fontsize=9.5,
        color="#1B365D",
    )

    # Bulk advection vector across frames in central gap
    bulk_arrow = FancyArrowPatch(
        (4.3, 3.6), (6.7, 3.6), arrowstyle="-|>", mutation_scale=14, color="#5CB85C", lw=2.4
    )
    ax.add_patch(bulk_arrow)
    ax.text(
        5.5,
        3.95,
        r"$\mathbf{v}_{\mathrm{bulk}}$",
        ha="center",
        fontsize=9.5,
        fontweight="bold",
        color="#3C763D",
        bbox={
            "boxstyle": "round,pad=0.15",
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
        fontsize=8.5,
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
        fontsize=8.5,
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
        fontsize=8.0,
        color="#0275D8",
        bbox={
            "boxstyle": "round,pad=0.25",
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
        "Parent P",
        ha="center",
        va="center",
        color="#FFFFFF",
        fontsize=8.0,
        fontweight="bold",
        path_effects=[pe.withStroke(linewidth=2.2, foreground="#B71C1C")],
    )

    # Search envelope W_i in frame t+1
    env_w = Rectangle(
        (7.4, 0.9), 2.7, 1.8, facecolor="#FBE9E7", edgecolor="#D9534F", ls="--", lw=1.4, alpha=0.85
    )
    ax.add_patch(env_w)
    ax.text(
        10.2,
        2.8,
        r"$\mathcal{W}_i$",
        color="#D9534F",
        fontsize=9.0,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.15",
            "facecolor": "white",
            "edgecolor": "#D9534F",
            "lw": 0.6,
            "alpha": 0.9,
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
        fontsize=7.5,
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
        fontsize=7.5,
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
        fontsize=7.5,
        color="#A94442",
        bbox={
            "boxstyle": "round,pad=0.32",
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


if __name__ == "__main__":
    generate_tube_chord_geometry()
    generate_tracking_breakup_schematic()
    generate_dfd_curves_regularization()
