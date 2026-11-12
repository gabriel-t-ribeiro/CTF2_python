#!/usr/bin/env python3
"""
Plot Void Fraction Timeline.

Generates publication-quality static figures (PNG/PDF/SVG) and standalone
interactive HTML dashboards showing void fraction evolution throughout time
from frame-by-frame flow reconstruction CSVs.

Supports:
  - Single run analysis or multi-run comparative overlays
  - Automatic frame index parsing and high-speed temporal mapping (FPS -> ms / s)
  - Dual axis representation: Time (bottom) and Frame Index (top)
  - Rolling moving average with configurable temporal window
  - Time-averaged mean void fraction with +/- 1-sigma standard deviation confidence band
  - Theoretical drift-flux reference benchmarks (Rouhani & Axelsson correlation)
  - Multi-panel mode (Void fraction, Gas volume evolution, Probability distribution)
  - Standalone interactive HTML dashboard (Chart.js via CDN, zoom/pan/tooltips)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default SI Constants for Steam-Water at 100 °C (1 atm saturation)
# ---------------------------------------------------------------------------
RHO_L_DEFAULT = 958.4  # Liquid density [kg/m^3]
RHO_V_DEFAULT = 0.597  # Vapor density [kg/m^3]
SIGMA_DEFAULT = 0.0589  # Surface tension [N/m]
G_ACCEL = 9.81  # Gravitational acceleration [m/s^2]


def calculate_rouhani_axelsson(
    x: float,
    mass_flux_G: float,
    rho_l: float = RHO_L_DEFAULT,
    rho_v: float = RHO_V_DEFAULT,
    sigma: float = SIGMA_DEFAULT,
    g: float = G_ACCEL,
) -> float:
    """
    Computes theoretical volumetric void fraction using Rouhani & Axelsson (1970).

    Strict SI Units:
      x: Vapor quality [-] (0.0 to 1.0)
      mass_flux_G: Total mass flux G [kg/(m^2 s)]
      rho_l: Liquid density [kg/m^3]
      rho_v: Vapor density [kg/m^3]
      sigma: Surface tension [N/m]
      g: Gravitational acceleration [m/s^2]
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if mass_flux_G <= 0.0:
        raise ValueError(f"Mass flux G must be positive, got {mass_flux_G}")

    c0 = 1.0 + 0.12 * (1.0 - x)
    kinematic_term = c0 * ((x / rho_v) + ((1.0 - x) / rho_l))
    delta_rho = max(0.0, rho_l - rho_v)
    buoyancy = (g * sigma * delta_rho) ** 0.25
    drift_term = (1.18 * (1.0 - x) * buoyancy) / (mass_flux_G * np.sqrt(rho_l))

    epsilon = (x / rho_v) / (kinematic_term + drift_term)
    return float(np.clip(epsilon, 0.0, 1.0))


def extract_frame_index(val: Any) -> int | None:
    """Extract integer frame number from string or number (e.g. 'frame_0016' -> 16)."""
    if pd.isna(val):
        return None
    if isinstance(val, int | np.integer):
        return int(val)
    if isinstance(val, float):
        return int(val)
    s = str(val)
    matches = re.findall(r"\d+", s)
    if matches:
        return int(matches[-1])
    return None


def detect_columns(df: pd.DataFrame, alpha_col: str | None = None) -> tuple[str, str, str | None]:
    """
    Detect the frame, void fraction, and optional gas volume columns in a DataFrame.

    Returns:
        (frame_col, void_fraction_col, gas_volume_col_or_none)
    """
    cols = list(df.columns)

    # 1. Frame column candidates
    frame_col = None
    for cand in ["frame", "Frame", "frame_id", "filename", "image", "Image", "idx", "Index"]:
        if cand in cols:
            frame_col = cand
            break
    if frame_col is None:
        # Fallback to first column
        frame_col = cols[0]

    # 2. Void fraction column candidates
    vf_col = None
    if alpha_col and alpha_col in cols:
        vf_col = alpha_col
    else:
        for cand in [
            "void_fraction_alpha",
            "Alpha_Voxel",
            "Alpha_Analytical",
            "alpha",
            "void_fraction",
            "VoidFraction",
            "Void_Fraction",
            "void_fraction_pct",
            "alpha_exp",
            "epsilon",
        ]:
            if cand in cols:
                vf_col = cand
                break
    if vf_col is None:
        raise ValueError(
            f"Could not automatically detect void fraction column. Available columns: {cols}. "
            f"Please specify using --alpha-col."
        )

    # 3. Gas volume column candidates
    vol_col = None
    for cand in [
        "gas_volume_m3",
        "GasVolume_Voxel_px3",
        "gas_volume_px3",
        "GasVolume_Analytical_px3",
        "gas_volume",
        "gas_vol",
    ]:
        if cand in cols:
            vol_col = cand
            break

    return frame_col, vf_col, vol_col


def load_void_fraction_series(
    csv_path: str | Path,
    fps: float = 2000.0,
    time_unit: str = "ms",
    relative_time: bool = True,
    alpha_col: str | None = None,
    series_label: str | None = None,
) -> dict[str, Any]:
    """
    Loads and processes a single void fraction CSV file into structured timeline data.

    Returns a dict containing:
      - 'df': pd.DataFrame with normalized columns
      - 'time': np.ndarray of timestamps
      - 'time_unit': 'ms' or 's'
      - 'frame_indices': np.ndarray of integer frame numbers
      - 'alpha': np.ndarray of instantaneous void fraction values
      - 'gas_volume': Optional np.ndarray of gas volume
      - 'label': str display label
      - 'source_path': Path
      - 'stats': summary statistics dict
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV file is empty: {csv_path}")

    frame_col, vf_col, vol_col = detect_columns(df, alpha_col)

    # Extract numerical frame indices
    frame_indices = []
    for val in df[frame_col]:
        f_idx = extract_frame_index(val)
        frame_indices.append(f_idx)

    # If all frame indices could not be parsed, fallback to row range
    if any(idx is None for idx in frame_indices):
        frame_indices = list(range(len(df)))

    frame_indices = np.array(frame_indices, dtype=int)

    # Sort chronologically if frames were out of order
    sort_idx = np.argsort(frame_indices)
    df_sorted = df.iloc[sort_idx].copy().reset_index(drop=True)
    frame_indices = frame_indices[sort_idx]

    # Void fraction array
    alpha_vals = df_sorted[vf_col].to_numpy(dtype=float)

    # Handle percentage values (> 1.0 but <= 100.0)
    if np.nanmax(alpha_vals) > 1.0 and np.nanmax(alpha_vals) <= 100.0:
        alpha_vals = alpha_vals / 100.0

    # Physical time mapping based on camera FPS
    # dt = 1.0 / fps (in seconds)
    time_sec = (frame_indices - frame_indices[0] if relative_time else frame_indices) / float(fps)
    if time_unit == "ms":
        time_vals = time_sec * 1000.0
    else:
        time_vals = time_sec

    # Gas volume extraction if present
    gas_vol_vals = None
    if vol_col is not None:
        gas_vol_vals = df_sorted[vol_col].to_numpy(dtype=float)

    # Label
    label = series_label or csv_path.stem.replace("_", " ").title()

    # Calculate core statistics
    valid_alpha = alpha_vals[~np.isnan(alpha_vals)]
    mean_val = float(np.mean(valid_alpha)) if len(valid_alpha) > 0 else 0.0
    std_val = float(np.std(valid_alpha, ddof=1)) if len(valid_alpha) > 1 else 0.0
    min_val = float(np.min(valid_alpha)) if len(valid_alpha) > 0 else 0.0
    max_val = float(np.max(valid_alpha)) if len(valid_alpha) > 0 else 0.0
    median_val = float(np.median(valid_alpha)) if len(valid_alpha) > 0 else 0.0
    p25 = float(np.percentile(valid_alpha, 25)) if len(valid_alpha) > 0 else 0.0
    p75 = float(np.percentile(valid_alpha, 75)) if len(valid_alpha) > 0 else 0.0

    stats = {
        "count": len(valid_alpha),
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "median": median_val,
        "iqr": p75 - p25,
        "duration": float(time_vals[-1] - time_vals[0]) if len(time_vals) > 1 else 0.0,
    }

    return {
        "df": df_sorted,
        "time": time_vals,
        "time_unit": time_unit,
        "fps": fps,
        "frame_col": frame_col,
        "vf_col": vf_col,
        "vol_col": vol_col,
        "frame_indices": frame_indices,
        "alpha": alpha_vals,
        "gas_volume": gas_vol_vals,
        "label": label,
        "source_path": csv_path,
        "stats": stats,
    }


def compute_rolling_mean(
    time_vals: np.ndarray,
    alpha_vals: np.ndarray,
    window: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes symmetric or centered moving average on timeline."""
    if len(alpha_vals) < window:
        window = max(1, len(alpha_vals))
    s = pd.Series(alpha_vals)
    roll = s.rolling(window=window, min_periods=1, center=True).mean().to_numpy()
    return time_vals, roll


def plot_void_fraction_timeline(
    series_list: Sequence[dict[str, Any]],
    output_path: str | Path,
    rolling_window: int = 5,
    vapor_quality: float | None = None,
    mass_flux: float | None = None,
    multi_panel: bool = False,
    title: str | None = None,
    dpi: int = 300,
) -> Path:
    """
    Generates a publication-quality figure of void fraction evolution throughout time.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Elsevier / Nature clean styling
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "axes.edgecolor": "#2c3e50",
            "axes.linewidth": 1.2,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "grid.color": "#e2e8f0",
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
        }
    )

    # Color palette
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    time_unit = series_list[0]["time_unit"]
    primary_series = series_list[0]
    time_label = f"Time $t$ [{time_unit}]"

    # Compute Rouhani-Axelsson if requested
    epsilon_th = None
    if vapor_quality is not None and mass_flux is not None:
        try:
            epsilon_th = calculate_rouhani_axelsson(vapor_quality, mass_flux)
        except Exception as e:
            print(f"[WARNING] Could not compute Rouhani-Axelsson benchmark: {e}")

    if multi_panel and primary_series.get("gas_volume") is not None:
        # 3-panel publication layout:
        # (Top) Void Fraction timeline
        # (Middle) Gas volume timeline
        # (Bottom) Void fraction histogram & KDE
        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1.4, 1.2], hspace=0.35)
        ax_main = fig.add_subplot(gs[0])
        ax_vol = fig.add_subplot(gs[1], sharex=ax_main)
        ax_hist = fig.add_subplot(gs[2])
    else:
        fig, ax_main = plt.subplots(figsize=(9.5, 5.5))
        ax_vol = None
        ax_hist = None

    # Top frame axis linked to ax_main
    ax_top = ax_main.twiny()

    # Plot each series
    for idx, s in enumerate(series_list):
        c = colors[idx % len(colors)]
        t = s["time"]
        a = s["alpha"]
        lbl = s["label"]
        stats = s["stats"]

        # Instantaneous points + connecting thin trace
        ax_main.plot(
            t,
            a,
            color=c,
            alpha=0.35,
            linewidth=1.2,
            linestyle="-",
            marker="o",
            markersize=3.5,
            markerfacecolor=c,
            markeredgewidth=0.5,
            label=f"{lbl} (raw)",
        )

        # Rolling average
        if len(a) >= 2:
            t_roll, a_roll = compute_rolling_mean(t, a, window=rolling_window)
            ax_main.plot(
                t_roll,
                a_roll,
                color=c,
                linewidth=2.4,
                linestyle="-",
                label=f"{lbl} (rolling avg, w={rolling_window})",
            )

        # Mean and +/- 1-sigma band for primary series
        if idx == 0:
            mean_a = stats["mean"]
            std_a = stats["std"]
            ax_main.axhline(
                mean_a,
                color="#0f172a",
                linestyle="--",
                linewidth=1.5,
                alpha=0.85,
                label=f"Time-Avg $\\bar{{\\alpha}} = {mean_a:.3f} \\pm {std_a:.3f}$",
            )
            ax_main.fill_between(
                [t[0], t[-1]],
                max(0.0, mean_a - std_a),
                min(1.0, mean_a + std_a),
                color=c,
                alpha=0.12,
                label="$\\pm 1\\sigma$ Confidence Band",
            )

            # Secondary plot: gas volume if available
            if ax_vol is not None and s.get("gas_volume") is not None:
                vol_data = s["gas_volume"]
                # Auto-scale units (m^3 or mm^3 or px^3)
                if np.max(vol_data) < 1e-2:
                    # In m^3 -> convert to mm^3 for human-readable y-axis (1 m^3 = 1e9 mm^3)
                    vol_plot = vol_data * 1e9
                    vol_unit_lbl = "Gas Volume $V_{\\mathrm{gas}}$ [$\\mathrm{mm}^3$]"
                else:
                    vol_plot = vol_data
                    vol_unit_lbl = "Gas Volume $V_{\\mathrm{gas}}$ [$\\mathrm{px}^3$]"

                ax_vol.plot(t, vol_plot, color="#d97706", linewidth=1.8, marker="s", markersize=3)
                ax_vol.set_ylabel(vol_unit_lbl, fontweight="bold", fontsize=10)
                ax_vol.grid(True, linestyle="--", alpha=0.5)

            # Tertiary plot: Void fraction distribution histogram
            if ax_hist is not None:
                ax_hist.hist(
                    a,
                    bins=min(15, max(5, len(a) // 2)),
                    color="#3b82f6",
                    edgecolor="#1d4ed8",
                    alpha=0.75,
                    density=True,
                )
                ax_hist.set_xlabel("Void Fraction $\\alpha$ [-]", fontweight="bold", fontsize=10)
                ax_hist.set_ylabel("Probability Density", fontweight="bold", fontsize=10)
                ax_hist.grid(True, linestyle="--", alpha=0.5)

    # Theoretical Rouhani-Axelsson benchmark line
    if epsilon_th is not None:
        ax_main.axhline(
            epsilon_th,
            color="#dc2626",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.9,
            label=f"Rouhani-Axelsson $\\varepsilon_{{\\mathrm{{th}}}} = {epsilon_th:.3f}$ ($x={vapor_quality}$, $G={mass_flux}$)",
        )

    # Coordinate top axis to show frame index
    p_frames = primary_series["frame_indices"]
    p_time = primary_series["time"]
    ax_main.set_xlim(p_time[0], p_time[-1])

    # Synchronize top axis limits and ticks
    ax_top.set_xlim(p_frames[0], p_frames[-1])
    ax_top.set_xlabel(
        "High-Speed Camera Frame Index ($f = 2000\\text{ FPS}$)",
        fontweight="bold",
        fontsize=10,
        labelpad=8,
    )

    # Format main axes
    ax_main.set_xlabel(time_label, fontweight="bold", fontsize=11)
    ax_main.set_ylabel("Volumetric Void Fraction $\\alpha(t)$ [-]", fontweight="bold", fontsize=11)
    ax_main.set_ylim(
        0.0,
        min(1.0, max(0.6, float(np.nanmax([np.nanmax(s["alpha"]) for s in series_list])) * 1.25)),
    )
    ax_main.grid(True, linestyle="--", alpha=0.6)

    # Title
    main_title = title or "Two-Phase Flow Boiling: Void Fraction Evolution vs. Time"
    fig.suptitle(main_title, fontsize=13, fontweight="bold", y=0.98 if ax_vol is not None else 1.01)

    # Stats box annotation
    p_stats = primary_series["stats"]
    stats_lines = [
        f"Frames: {p_stats['count']} ($\\Delta t = {p_stats['duration']:.1f}\\,\\mathrm{{{time_unit}}}$)",
        f"Mean $\\bar{{\\alpha}}$: {p_stats['mean']:.4f} $\\pm$ {p_stats['std']:.4f}",
        f"Range: [{p_stats['min']:.4f}, {p_stats['max']:.4f}]",
        f"Median: {p_stats['median']:.4f} (IQR: {p_stats['iqr']:.4f})",
    ]
    if epsilon_th is not None:
        diff_pct = abs(p_stats["mean"] - epsilon_th) / max(epsilon_th, 1e-5) * 100.0
        stats_lines.append(f"Diff vs R-A: {diff_pct:.1f}\\%")

    stats_text = "\n".join(stats_lines)
    ax_main.text(
        0.02,
        0.96,
        stats_text,
        transform=ax_main.transAxes,
        verticalalignment="top",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#f8fafc",
            "edgecolor": "#cbd5e1",
            "alpha": 0.9,
        },
        fontsize=8.5,
        fontfamily="monospace",
    )

    ax_main.legend(loc="upper right", framealpha=0.92, facecolor="#f8fafc", edgecolor="#cbd5e1")

    import warnings

    import matplotlib.ticker as ticker

    ax_top.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))

    # Clean layout handling for multi-panel twiny subplots
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with contextlib.suppress(Exception):
            plt.tight_layout()

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] Saved void fraction timeline plot: {output_path}")
    return output_path


def generate_interactive_html(
    series_list: Sequence[dict[str, Any]],
    output_html_path: str | Path,
    rolling_window: int = 5,
    vapor_quality: float | None = None,
    mass_flux: float | None = None,
    title: str | None = None,
) -> Path:
    """
    Generates a rich, standalone, interactive HTML dashboard visualizing void fraction over time
    using Chart.js (via CDN). No external Python graphics packages required.
    """
    output_html_path = Path(output_html_path)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    primary_series = series_list[0]
    time_unit = primary_series["time_unit"]
    fps = primary_series["fps"]

    # Compute theoretical Rouhani-Axelsson benchmark if available
    epsilon_th = None
    if vapor_quality is not None and mass_flux is not None:
        with contextlib.suppress(Exception):
            epsilon_th = calculate_rouhani_axelsson(vapor_quality, mass_flux)

    # Build datasets for Chart.js
    chart_datasets = []
    colors = [
        {"raw": "#3b82f6", "roll": "#1d4ed8", "band": "rgba(59, 130, 246, 0.15)"},
        {"raw": "#f97316", "roll": "#c2410c", "band": "rgba(249, 115, 22, 0.15)"},
        {"raw": "#10b981", "roll": "#047857", "band": "rgba(16, 185, 129, 0.15)"},
    ]

    all_times = primary_series["time"].tolist()

    for s_idx, s in enumerate(series_list):
        c = colors[s_idx % len(colors)]
        t = s["time"]
        a = s["alpha"]
        lbl = s["label"]
        stats = s["stats"]

        # Raw points
        raw_data = [{"x": float(t[i]), "y": round(float(a[i]), 5)} for i in range(len(t))]
        chart_datasets.append(
            {
                "type": "scatter",
                "label": f"{lbl} (Raw Points)",
                "data": raw_data,
                "backgroundColor": c["raw"],
                "borderColor": c["raw"],
                "pointRadius": 4,
                "pointHoverRadius": 6,
                "showLine": False,
                "yAxisID": "y",
            }
        )

        # Rolling mean
        if len(a) >= 2:
            t_roll, a_roll = compute_rolling_mean(t, a, window=rolling_window)
            roll_data = [
                {"x": float(t_roll[i]), "y": round(float(a_roll[i]), 5)} for i in range(len(t_roll))
            ]
            chart_datasets.append(
                {
                    "type": "line",
                    "label": f"{lbl} (Rolling Avg, w={rolling_window})",
                    "data": roll_data,
                    "borderColor": c["roll"],
                    "backgroundColor": c["roll"],
                    "borderWidth": 2.5,
                    "tension": 0.2,
                    "pointRadius": 0,
                    "pointHoverRadius": 4,
                    "yAxisID": "y",
                }
            )

        # Mean line for primary series
        if s_idx == 0:
            mean_val = stats["mean"]
            std_val = stats["std"]
            chart_datasets.append(
                {
                    "type": "line",
                    "label": f"Time-Avg Mean ({mean_val:.4f})",
                    "data": [
                        {"x": float(t[0]), "y": round(mean_val, 5)},
                        {"x": float(t[-1]), "y": round(mean_val, 5)},
                    ],
                    "borderColor": "#0f172a",
                    "borderWidth": 2,
                    "borderDash": [6, 4],
                    "pointRadius": 0,
                    "fill": False,
                    "yAxisID": "y",
                }
            )

            # Upper sigma bound
            chart_datasets.append(
                {
                    "type": "line",
                    "label": "+1σ Upper Bound",
                    "data": [
                        {"x": float(t[0]), "y": min(1.0, round(mean_val + std_val, 5))},
                        {"x": float(t[-1]), "y": min(1.0, round(mean_val + std_val, 5))},
                    ],
                    "borderColor": "rgba(100, 116, 139, 0.4)",
                    "borderDash": [3, 3],
                    "pointRadius": 0,
                    "fill": False,
                    "hidden": True,
                    "yAxisID": "y",
                }
            )

            # Lower sigma bound
            chart_datasets.append(
                {
                    "type": "line",
                    "label": "-1σ Lower Bound",
                    "data": [
                        {"x": float(t[0]), "y": max(0.0, round(mean_val - std_val, 5))},
                        {"x": float(t[-1]), "y": max(0.0, round(mean_val - std_val, 5))},
                    ],
                    "borderColor": "rgba(100, 116, 139, 0.4)",
                    "borderDash": [3, 3],
                    "pointRadius": 0,
                    "fill": False,
                    "hidden": True,
                    "yAxisID": "y",
                }
            )

    # Theoretical Rouhani-Axelsson line
    if epsilon_th is not None:
        chart_datasets.append(
            {
                "type": "line",
                "label": f"Rouhani-Axelsson ({epsilon_th:.4f})",
                "data": [
                    {"x": float(all_times[0]), "y": round(epsilon_th, 5)},
                    {"x": float(all_times[-1]), "y": round(epsilon_th, 5)},
                ],
                "borderColor": "#dc2626",
                "borderWidth": 2,
                "borderDash": [8, 4],
                "pointRadius": 0,
                "fill": False,
                "yAxisID": "y",
            }
        )

    # Prepare table rows for primary series
    p_df = primary_series["df"]
    p_frames = primary_series["frame_indices"]
    p_time = primary_series["time"]
    p_alpha = primary_series["alpha"]
    p_fcol = primary_series["frame_col"]

    table_rows = []
    for i in range(len(p_df)):
        frame_name = str(p_df[p_fcol].iloc[i])
        f_num = int(p_frames[i])
        t_val = float(p_time[i])
        a_val = float(p_alpha[i])
        vol_str = "N/A"
        if primary_series.get("gas_volume") is not None:
            vol_raw = float(primary_series["gas_volume"][i])
            vol_str = f"{vol_raw * 1e9:.2f} mm³" if vol_raw < 1e-2 else f"{vol_raw:.0f} px³"

        table_rows.append(
            {
                "frame": frame_name,
                "index": f_num,
                "time": f"{t_val:.2f}",
                "alpha": f"{a_val:.4f}",
                "alpha_pct": f"{a_val * 100.0:.2f}%",
                "volume": vol_str,
            }
        )

    p_stats = primary_series["stats"]
    ra_diff_badge = ""
    if epsilon_th is not None:
        diff_pct = abs(p_stats["mean"] - epsilon_th) / max(epsilon_th, 1e-5) * 100.0
        ra_diff_badge = f"""
        <div class="stat-card border-red">
          <div class="stat-title">Rouhani-Axelsson Ref</div>
          <div class="stat-value text-red">{epsilon_th:.4f}</div>
          <div class="stat-sub">Δ = {diff_pct:.2f}% (x={vapor_quality}, G={mass_flux})</div>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title or "CTF2 Void Fraction Timeline"}</title>
  <!-- Chart.js and Plugins -->
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/hammerjs@2.0.8"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1/dist/chartjs-plugin-zoom.min.js"></script>
  <style>
    :root {{
      --bg-primary: #f8fafc;
      --bg-card: #ffffff;
      --text-main: #0f172a;
      --text-muted: #64748b;
      --border-color: #e2e8f0;
      --accent-blue: #2563eb;
      --accent-amber: #d97706;
      --accent-red: #dc2626;
      --shadow-sm: 0 1px 3px rgba(0,0,0,0.08);
      --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -2px rgba(0,0,0,0.1);
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background: var(--bg-primary);
      color: var(--text-main);
      margin: 0;
      padding: 24px;
      line-height: 1.5;
    }}
    .header-container {{
      max-width: 1200px;
      margin: 0 auto 20px auto;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .title-group h1 {{
      margin: 0;
      font-size: 1.5rem;
      font-weight: 700;
      color: var(--text-main);
    }}
    .title-group p {{
      margin: 4px 0 0 0;
      font-size: 0.875rem;
      color: var(--text-muted);
    }}
    .stats-grid {{
      max-width: 1200px;
      margin: 0 auto 24px auto;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
    }}
    .stat-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 10px;
      padding: 16px;
      box-shadow: var(--shadow-sm);
    }}
    .stat-card.border-red {{
      border-left: 4px solid var(--accent-red);
    }}
    .stat-card.border-blue {{
      border-left: 4px solid var(--accent-blue);
    }}
    .stat-title {{
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
    }}
    .stat-value {{
      font-size: 1.75rem;
      font-weight: 700;
      margin: 6px 0;
      color: var(--text-main);
    }}
    .stat-value.text-red {{ color: var(--accent-red); }}
    .stat-value.text-blue {{ color: var(--accent-blue); }}
    .stat-sub {{
      font-size: 0.8125rem;
      color: var(--text-muted);
    }}
    .chart-container-card {{
      max-width: 1200px;
      margin: 0 auto 24px auto;
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 24px;
      box-shadow: var(--shadow-md);
      position: relative;
    }}
    .chart-toolbar {{
      display: flex;
      justify-content: flex-end;
      gap: 8px;
      margin-bottom: 12px;
    }}
    .btn {{
      background: #f1f5f9;
      border: 1px solid var(--border-color);
      color: var(--text-main);
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 0.8125rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s ease;
    }}
    .btn:hover {{
      background: #e2e8f0;
    }}
    .canvas-wrapper {{
      position: relative;
      height: 480px;
      width: 100%;
    }}
    .table-container-card {{
      max-width: 1200px;
      margin: 0 auto;
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 24px;
      box-shadow: var(--shadow-md);
    }}
    .table-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }}
    .table-header h2 {{
      margin: 0;
      font-size: 1.125rem;
      font-weight: 600;
    }}
    .search-input {{
      padding: 6px 12px;
      border: 1px solid var(--border-color);
      border-radius: 6px;
      font-size: 0.875rem;
      outline: none;
    }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.875rem;
      text-align: left;
    }}
    .data-table th {{
      background: #f8fafc;
      padding: 10px 14px;
      border-bottom: 2px solid var(--border-color);
      font-weight: 600;
      color: var(--text-muted);
    }}
    .data-table td {{
      padding: 8px 14px;
      border-bottom: 1px solid var(--border-color);
      font-variant-numeric: tabular-nums;
    }}
    .data-table tr:hover td {{
      background: #f1f5f9;
    }}
  </style>
</head>
<body>

  <div class="header-container">
    <div class="title-group">
      <h1>{title or "Two-Phase Flow Boiling: Void Fraction Timeline"}</h1>
      <p>Source: <code>{primary_series["source_path"].name}</code> &bull; Capture Rate: <strong>{fps:.1f} FPS</strong> &bull; dt = {1000.0 / fps:.3f} ms</p>
    </div>
  </div>

  <div class="stats-grid">
    <div class="stat-card border-blue">
      <div class="stat-title">Mean Void Fraction (α)</div>
      <div class="stat-value text-blue">{p_stats["mean"]:.4f}</div>
      <div class="stat-sub">± {p_stats["std"]:.4f} (1σ) &bull; {p_stats["mean"] * 100:.2f}%</div>
    </div>

    <div class="stat-card">
      <div class="stat-title">Dynamic Range</div>
      <div class="stat-value">{p_stats["min"]:.3f} – {p_stats["max"]:.3f}</div>
      <div class="stat-sub">Span Δα = {(p_stats["max"] - p_stats["min"]):.4f}</div>
    </div>

    <div class="stat-card">
      <div class="stat-title">Analyzed Interval</div>
      <div class="stat-value">{p_stats["count"]} <span style="font-size:1rem; font-weight: normal;">frames</span></div>
      <div class="stat-sub">Duration: {p_stats["duration"]:.2f} {time_unit}</div>
    </div>

    {ra_diff_badge}
  </div>

  <div class="chart-container-card">
    <div class="chart-toolbar">
      <button class="btn" onclick="chart.resetZoom()">Reset Zoom</button>
      <button class="btn" onclick="toggleDataset(0)">Toggle Points</button>
      <button class="btn" onclick="toggleDataset(1)">Toggle Rolling Avg</button>
    </div>
    <div class="canvas-wrapper">
      <canvas id="vfChart"></canvas>
    </div>
  </div>

  <div class="table-container-card">
    <div class="table-header">
      <h2>Frame Measurement Log</h2>
      <input type="text" id="searchInput" class="search-input" placeholder="Filter frames..." onkeyup="filterTable()">
    </div>
    <div style="max-height: 360px; overflow-y: auto;">
      <table class="data-table" id="measurementTable">
        <thead>
          <tr>
            <th>Frame Name</th>
            <th>Frame #</th>
            <th>Time [{time_unit}]</th>
            <th>Void Fraction α [-]</th>
            <th>Void Fraction [%]</th>
            <th>Gas Volume</th>
          </tr>
        </thead>
        <tbody>
          {"".join(f"<tr><td>{row['frame']}</td><td>{row['index']}</td><td>{row['time']}</td><td><strong>{row['alpha']}</strong></td><td>{row['alpha_pct']}</td><td>{row['volume']}</td></tr>" for row in table_rows)}
        </tbody>
      </table>
    </div>
  </div>

  <script>
    const datasets = {json.dumps(chart_datasets)};

    const ctx = document.getElementById('vfChart').getContext('2d');
    const chart = new Chart(ctx, {{
      data: {{
        datasets: datasets
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        interaction: {{
          mode: 'nearest',
          intersect: false,
        }},
        plugins: {{
          legend: {{
            position: 'top',
            labels: {{
              usePointStyle: true,
              boxWidth: 8,
              font: {{ family: "-apple-system, BlinkMacSystemFont, sans-serif", size: 12 }}
            }}
          }},
          tooltip: {{
            backgroundColor: 'rgba(15, 23, 42, 0.9)',
            titleFont: {{ size: 13, weight: 'bold' }},
            bodyFont: {{ size: 12 }},
            padding: 10,
            cornerRadius: 8,
            callbacks: {{
              label: function(context) {{
                let label = context.dataset.label || '';
                if (label) {{
                  label += ': ';
                }}
                if (context.parsed.y !== null) {{
                  label += context.parsed.y.toFixed(4) + ' (' + (context.parsed.y * 100).toFixed(2) + '%)';
                }}
                return label;
              }},
              afterBody: function(context) {{
                const item = context[0];
                return ['Time: ' + item.parsed.x.toFixed(2) + ' {time_unit}'];
              }}
            }}
          }},
          zoom: {{
            pan: {{
              enabled: true,
              mode: 'x',
            }},
            zoom: {{
              wheel: {{
                enabled: true,
              }},
              pinch: {{
                enabled: true
              }},
              mode: 'x',
            }}
          }}
        }},
        scales: {{
          x: {{
            type: 'linear',
            title: {{
              display: true,
              text: 'Physical Time t [{time_unit}]',
              font: {{ size: 13, weight: 'bold' }}
            }},
            grid: {{ color: '#f1f5f9' }}
          }},
          y: {{
            min: 0.0,
            max: {min(1.0, max(0.5, float(np.nanmax(p_alpha)) * 1.25)):.2f},
            title: {{
              display: true,
              text: 'Volumetric Void Fraction α [-]',
              font: {{ size: 13, weight: 'bold' }}
            }},
            grid: {{ color: '#e2e8f0' }}
          }}
        }}
      }}
    }});

    function toggleDataset(index) {{
      const meta = chart.getDatasetMeta(index);
      meta.hidden = meta.hidden === null ? !chart.data.datasets[index].hidden : null;
      chart.update();
    }}

    function filterTable() {{
      const input = document.getElementById("searchInput");
      const filter = input.value.toUpperCase();
      const table = document.getElementById("measurementTable");
      const tr = table.getElementsByTagName("tr");
      for (let i = 1; i < tr.length; i++) {{
        const td0 = tr[i].getElementsByTagName("td")[0];
        const td1 = tr[i].getElementsByTagName("td")[1];
        if (td0 || td1) {{
          const txt0 = td0.textContent || td0.innerText;
          const txt1 = td1.textContent || td1.innerText;
          if (txt0.toUpperCase().indexOf(filter) > -1 || txt1.toUpperCase().indexOf(filter) > -1) {{
            tr[i].style.display = "";
          }} else {{
            tr[i].style.display = "none";
          }}
        }}
      }}
    }}
  </script>
</body>
</html>
"""
    output_html_path.write_text(html_content, encoding="utf-8")
    print(f"[DONE] Saved interactive void fraction dashboard: {output_html_path}")
    return output_html_path


def run_timeline_pipeline(
    input_paths: Sequence[str | Path],
    output_plot: str | Path = Path("results/void_fraction_timeline.png"),
    output_html: str | Path | None = Path("results/void_fraction_timeline.html"),
    fps: float = 2000.0,
    time_unit: str = "ms",
    relative_time: bool = True,
    rolling_window: int = 5,
    vapor_quality: float | None = None,
    mass_flux: float | None = None,
    multi_panel: bool = False,
    alpha_col: str | None = None,
    labels: Sequence[str] | None = None,
    title: str | None = None,
    dpi: int = 300,
) -> tuple[Path, Path | None]:
    """
    Main orchestration routine for loading CSVs, running mathematical processing,
    and generating high-resolution figures and interactive dashboards.
    """
    # Expand globs if any input is a glob string
    expanded_paths: list[Path] = []
    for p in input_paths:
        p_obj = Path(p)
        if "*" in str(p):
            matches = sorted(p_obj.parent.glob(p_obj.name))
            expanded_paths.extend(matches)
        elif p_obj.is_dir():
            csvs = sorted(p_obj.glob("*.csv"))
            expanded_paths.extend(csvs)
        else:
            expanded_paths.append(p_obj)

    if not expanded_paths:
        raise FileNotFoundError(f"No CSV files found matching inputs: {input_paths}")

    series_list = []
    for idx, p in enumerate(expanded_paths):
        lbl = labels[idx] if (labels and idx < len(labels)) else None
        s = load_void_fraction_series(
            p,
            fps=fps,
            time_unit=time_unit,
            relative_time=relative_time,
            alpha_col=alpha_col,
            series_label=lbl,
        )
        series_list.append(s)

    # 1. Generate static publication plot
    plot_path = plot_void_fraction_timeline(
        series_list=series_list,
        output_path=output_plot,
        rolling_window=rolling_window,
        vapor_quality=vapor_quality,
        mass_flux=mass_flux,
        multi_panel=multi_panel,
        title=title,
        dpi=dpi,
    )

    # 2. Generate interactive HTML if requested
    html_path = None
    if output_html is not None:
        html_path = generate_interactive_html(
            series_list=series_list,
            output_html_path=output_html,
            rolling_window=rolling_window,
            vapor_quality=vapor_quality,
            mass_flux=mass_flux,
            title=title,
        )

    return plot_path, html_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality void fraction timeline graphs and interactive dashboards."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to void fraction CSV file(s) or directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("results/void_fraction_timeline.png"),
        help="Output path for publication figure (PNG, SVG, or PDF)",
    )
    parser.add_argument(
        "--html",
        "--interactive-html",
        dest="interactive_html",
        type=Path,
        default=Path("results/void_fraction_timeline.html"),
        help="Output path for standalone interactive HTML viewer",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Disable interactive HTML dashboard generation",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2000.0,
        help="High-speed camera framerate in 1/s (Hz, default: 2000.0 FPS, dt = 0.5 ms)",
    )
    parser.add_argument(
        "--time-unit",
        choices=["ms", "s"],
        default="ms",
        help="Physical time unit for timeline axis (default: ms)",
    )
    parser.add_argument(
        "--absolute-time",
        action="store_true",
        help="Use absolute camera time instead of zero-relative start time",
    )
    parser.add_argument(
        "--rolling-window",
        "-w",
        type=int,
        default=5,
        help="Moving average smoothing window in frames (default: 5)",
    )
    parser.add_argument(
        "--vapor-quality",
        "-x",
        type=float,
        default=None,
        help="Thermodynamic vapor quality x [-] for Rouhani-Axelsson benchmark",
    )
    parser.add_argument(
        "--mass-flux",
        "-G",
        type=float,
        default=None,
        help="Total mass flux G [kg/(m^2 s)] for Rouhani-Axelsson benchmark",
    )
    parser.add_argument(
        "--multi-panel",
        action="store_true",
        help="Generate 3-panel layout: Void fraction, Gas volume, and Distribution histogram",
    )
    parser.add_argument(
        "--alpha-col",
        type=str,
        default=None,
        help="Explicit column name for void fraction values",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Custom labels for series when comparing multiple CSVs",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom figure title",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution in DPI for raster figure output (default: 300)",
    )

    args = parser.parse_args()

    html_out = None if args.no_html else args.interactive_html

    run_timeline_pipeline(
        input_paths=args.input,
        output_plot=args.output,
        output_html=html_out,
        fps=args.fps,
        time_unit=args.time_unit,
        relative_time=not args.absolute_time,
        rolling_window=args.rolling_window,
        vapor_quality=args.vapor_quality,
        mass_flux=args.mass_flux,
        multi_panel=args.multi_panel,
        alpha_col=args.alpha_col,
        labels=args.labels,
        title=args.title,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
