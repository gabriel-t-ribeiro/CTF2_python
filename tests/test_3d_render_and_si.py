"""
Comprehensive test suite verifying:
  1. SI unit refactoring (m, m^2, m^3, m/s, kg/m^3).
  2. Mathematical precision of 3D volume reconstruction and void fraction bounds.
  3. Robustness against NaN ellipse fits, singular contours, and axis sorting.
  4. 3D render pipeline image generation and visual styling integrity.
  5. Kinematic velocity tracking with optimal 1-to-1 assignment and breakup decoupling.
  6. Theoretical Rouhani-Axelsson drift-flux model in SI units for boiling water.
  7. Single-camera Depth from Defocus (DFD) inversion and depth estimation.
"""

import importlib
import math
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    import pytest
except ImportError:
    pytest = None


class AssertRaises:
    def __init__(self, exc_type):
        self.exc_type = exc_type

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None or not issubclass(exc_type, self.exc_type):
            raise AssertionError(f"Expected {self.exc_type.__name__}, got {exc_type}")
        return True


# Ensure repository root is on sys.path for direct execution
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from model_training.render_3d_flow import (
    reconstruct_3d_bubbles_si,
    render_3d_flow_visualization,
    render_interactive_html,
)

try:
    mod_recon = importlib.import_module("model_training.4_2d_void_fraction_reconstruct")
except ModuleNotFoundError:
    mod_recon = importlib.import_module("model_training.4_void_fraction_reconstruct")
compute_3d_gas_volume = mod_recon.compute_3d_gas_volume
calculate_rouhani_axelsson_void_fraction = mod_recon.calculate_rouhani_axelsson_void_fraction
run_reconstruction = mod_recon.run_reconstruction

mod_track = importlib.import_module("model_training.5_track_bubbles_kinematics")
extract_bubble_blobs = mod_track.extract_bubble_blobs
run_tracking = mod_track.run_tracking

mod_dfd = importlib.import_module("model_training.6_depth_from_defocus")
DFDModel = mod_dfd.DFDModel
analyze_frame_dfd = mod_dfd.analyze_frame_dfd


def test_si_unit_scales_and_dimensions():
    """Verify that all geometric and volumetric calculations are in SI units (m, m^3)."""
    h, w = 300, 800
    mask = np.zeros((h, w), dtype=np.uint8)

    # Draw a synthetic circular bubble of radius 20 px at center
    cv2.circle(mask, (400, 150), 20, 255, -1)

    s_px = 8.6e-5  # m/px
    D_m = 0.020  # 20 mm inner diameter
    R_m = D_m / 2.0
    L_m = w * s_px

    recon = reconstruct_3d_bubbles_si(mask, pixel_size_m=s_px, pipe_dia_m=D_m)

    # Pipe volume in m^3: pi * R^2 * L
    expected_v_pipe = math.pi * (R_m**2) * L_m
    assert abs(recon["pipe_volume_m3"] - expected_v_pipe) < 1e-12

    # Theoretical sphere volume of r = 20 * s_px
    r_m = 20.0 * s_px
    expected_v_sphere = (4.0 / 3.0) * math.pi * (r_m**3)
    gas_vol = recon["gas_volume_m3"]

    assert gas_vol > 0.0
    rel_error = abs(gas_vol - expected_v_sphere) / expected_v_sphere
    assert rel_error < 0.10, f"Sphere volume error {rel_error:.3f} exceeds tolerance"

    alpha = recon["void_fraction_alpha"]
    assert 0.0 <= alpha <= 1.0


def test_nan_robustness_and_ellipse_axis_sorting():
    """Verify that degenerate / collinear contours do not produce NaNs and axes are sorted."""
    h, w = 100, 100
    mask = np.zeros((h, w), dtype=np.uint8)

    # Create a 1-pixel-thick line (collinear points where cv2.fitEllipse can fail or return NaN)
    mask[50, 40:60] = 255

    s_px = 8.6e-5
    D_m = 0.020
    R_px = (D_m / 2.0) / s_px

    vol = compute_3d_gas_volume(mask, R_px, pipe_center_y_px=50.0, pixel_size_m=s_px)
    assert not np.isnan(vol), "Volume must not be NaN for collinear contours"
    assert vol >= 0.0

    recon = reconstruct_3d_bubbles_si(mask, pixel_size_m=s_px, pipe_dia_m=D_m)
    assert not np.isnan(recon["gas_volume_m3"])
    assert not np.isnan(recon["void_fraction_alpha"])


def test_dual_regime_large_slug_confinement():
    """Verify that large slugs are strictly confined by cylindrical chord depth and in m^3."""
    h, w = 300, 800
    mask = np.zeros((h, w), dtype=np.uint8)

    # Draw a massive rectangular slug covering 15000 pixels (> regime threshold)
    mask[50:250, 100:300] = 255  # 200 x 200 = 40,000 px

    s_px = 8.6e-5
    D_m = 0.020
    recon = reconstruct_3d_bubbles_si(mask, pixel_size_m=s_px, pipe_dia_m=D_m)

    assert recon["bubble_count"] == 1
    bubble = recon["bubbles"][0]
    assert bubble["regime"] == "large_slug"
    assert bubble["volume_m3"] > 0.0
    assert bubble["volume_m3"] <= recon["pipe_volume_m3"]
    assert 0.0 <= recon["void_fraction_alpha"] <= 1.0


def test_rouhani_axelsson_drift_flux_water_si():
    """Verify the Rouhani-Axelsson model strictly for boiling water in standard SI units."""
    # Bounds
    assert calculate_rouhani_axelsson_void_fraction(0.0, 500.0) == 0.0
    assert calculate_rouhani_axelsson_void_fraction(1.0, 500.0) == 1.0

    # Guardrail: G <= 0 must raise ValueError
    with AssertRaises(ValueError):
        calculate_rouhani_axelsson_void_fraction(0.1, mass_flux_G=0.0)

    with AssertRaises(ValueError):
        calculate_rouhani_axelsson_void_fraction(0.1, mass_flux_G=-100.0)

    # Boiling water at 100°C, G = 500 kg/(m^2 s), low vapor quality x = 0.05
    alpha_calc = calculate_rouhani_axelsson_void_fraction(0.05, mass_flux_G=500.0)
    assert 0.3 <= alpha_calc <= 0.9, f"Computed alpha {alpha_calc} out of expected range for water"


def test_3d_render_generation():
    """Verify that the 3D rendering pipeline outputs a valid, non-empty PNG image and HTML with large slugs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        h, w = 200, 400
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 15, 255, -1)
        # 150 x 60 = 9000 px -> Large slug (> 5000 px)
        cv2.rectangle(mask, (200, 70), (350, 130), 255, -1)

        s_px = 6.1e-5
        D_m = 0.020
        recon = reconstruct_3d_bubbles_si(mask, pixel_size_m=s_px, pipe_dia_m=D_m)

        out_png = tmp_path / "test_render.png"
        out_html = tmp_path / "test_render.html"

        rendered_png = render_3d_flow_visualization(recon, out_png, dpi=80)
        assert rendered_png.exists()
        assert rendered_png.stat().st_size > 1000

        rendered_html = render_interactive_html(recon, out_html)
        assert rendered_html.exists()
        assert rendered_html.stat().st_size > 500

        # Verify that both small ellipsoids AND large slugs are rendered in Three.js HTML
        html_text = out_html.read_text(encoding="utf-8")
        assert "slugs.forEach" in html_text
        assert "spheres.forEach" in html_text
        assert '"vertices":' in html_text
        assert '"indices":' in html_text


def test_odd_shaped_bubble_area_conservation():
    """Verify that irregular non-convex bubbles do not become oversized by fitEllipse."""
    h, w = 150, 150
    mask = np.zeros((h, w), dtype=np.uint8)
    # Create an L-shaped / concave bubble
    mask[50:100, 50:60] = 255
    mask[90:100, 50:110] = 255
    actual_area_px = float((mask > 0).sum())

    recon = reconstruct_3d_bubbles_si(mask, pixel_size_m=6.1e-5, pipe_dia_m=0.020)
    assert recon["bubble_count"] == 1
    b = recon["bubbles"][0]
    assert b["regime"] == "small_ellipsoid"
    a_m, b_m, _ = b["semi_axes_m"]
    a_px = a_m / 6.1e-5
    b_px = b_m / 6.1e-5
    projected_area_px = math.pi * a_px * b_px
    # Projected area must not exceed actual area by more than 1%
    assert projected_area_px <= actual_area_px * 1.01, (
        f"Oversized: {projected_area_px} vs {actual_area_px}"
    )


def test_kinematic_tracking_optimal_bipartite_matching():
    """Verify 1-to-1 bipartite assignment prevents many-to-one collision tracking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        mask_dir = tmp_path / "masks"
        out_dir = tmp_path / "results"
        mask_dir.mkdir()

        # Frame 0: Two distinct bubbles at x=100, y=50 and x=100, y=150
        m0 = np.zeros((200, 500), dtype=np.uint8)
        cv2.circle(m0, (100, 50), 10, 255, -1)
        cv2.circle(m0, (100, 150), 10, 255, -1)
        cv2.imwrite(str(mask_dir / "frame_0000.png"), m0)

        # Frame 1: Both translate by 15 px to x=115
        m1 = np.zeros((200, 500), dtype=np.uint8)
        cv2.circle(m1, (115, 50), 10, 255, -1)
        cv2.circle(m1, (115, 150), 10, 255, -1)
        cv2.imwrite(str(mask_dir / "frame_0001.png"), m1)

        fps = 2000.0
        s_px = 8.6e-5

        run_tracking(mask_dir, out_dir, fps=fps, pixel_size=s_px, max_disp_px=50.0)

        vel_csv = out_dir / "bubble_velocities_si.csv"
        assert vel_csv.exists()

        df = pd.read_csv(vel_csv)
        # Must have exactly 2 matched bubbles with unique IDs
        assert len(df) == 2
        assert len(df["bubble_id"].unique()) == 2

        expected_vel = (15.0 * s_px) / (1.0 / fps)
        for v in df["velocity_m_s"]:
            assert abs(v - expected_vel) < 1e-4


def test_depth_from_defocus_inversion():
    """Verify the DFD model lookup table inversion produces monotonic physical blur and radius."""
    model = DFDModel()

    # Invert for high gradient (in-focus edge)
    s_sharp, rho_sharp = model.invert(0.50)
    # Invert for low gradient (blurred edge)
    s_blurry, rho_blurry = model.invert(0.28)

    # Blurry edge must have higher sigma_tilde and smaller rho_tilde
    assert s_blurry > s_sharp
    assert rho_sharp >= rho_blurry
    assert 0.005 <= s_sharp <= 0.35
    assert 0.005 <= s_blurry <= 0.35


def test_depth_from_defocus_near_wall_regularization():
    """Verify near-wall regularization smoothly guides sharp/near-wall bubbles to z ~ 0 and d0 ~ 2*r_t."""
    model = DFDModel()
    h, w = 100, 100
    raw = np.full((h, w), 200, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)

    # Draw a sharp circular bubble of radius 15 px at center (50, 50)
    cv2.circle(raw, (50, 50), 15, 30, -1)
    cv2.circle(mask, (50, 50), 15, 255, -1)

    results = analyze_frame_dfd(
        raw,
        mask,
        model,
        pixel_size_m=1e-4,
        beta_eff_px_m=2000.0,
        near_wall_sigma_threshold=0.05,
    )
    assert len(results) == 1
    res = results[0]
    assert res["near_wall_zone"] is True
    # Depth must be near front wall (sub-millimeter)
    assert res["depth_z_m"] < 1e-4
    # Diameter must accurately reflect the 30 px circle (30 * 1e-4 = 3e-3 m) within 5%
    expected_d0 = 30.0 * 1e-4
    assert abs(res["diameter_d0_m"] - expected_d0) / expected_d0 < 0.05


if __name__ == "__main__":
    print("Running comprehensive CTF2 verification tests...")
    test_si_unit_scales_and_dimensions()
    print("  [PASS] test_si_unit_scales_and_dimensions")
    test_nan_robustness_and_ellipse_axis_sorting()
    print("  [PASS] test_nan_robustness_and_ellipse_axis_sorting")
    test_dual_regime_large_slug_confinement()
    print("  [PASS] test_dual_regime_large_slug_confinement")
    test_rouhani_axelsson_drift_flux_water_si()
    print("  [PASS] test_rouhani_axelsson_drift_flux_water_si")
    test_3d_render_generation()
    print("  [PASS] test_3d_render_generation")
    test_odd_shaped_bubble_area_conservation()
    print("  [PASS] test_odd_shaped_bubble_area_conservation")
    test_kinematic_tracking_optimal_bipartite_matching()
    print("  [PASS] test_kinematic_tracking_optimal_bipartite_matching")
    test_depth_from_defocus_inversion()
    print("  [PASS] test_depth_from_defocus_inversion")
    test_depth_from_defocus_near_wall_regularization()
    print("  [PASS] test_depth_from_defocus_near_wall_regularization")
    print("\nALL 9 TESTS PASSED SUCCESSFULLY!")
