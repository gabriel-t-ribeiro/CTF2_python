"""
Comprehensive test suite for the void fraction timeline visualization pipeline.
Tests CSV parsing, temporal time mapping, rolling average calculations,
Rouhani-Axelsson theoretical benchmarks, publication PNG plotting, and interactive HTML dashboards.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_training.plot_void_fraction_timeline import (
    calculate_rouhani_axelsson,
    compute_rolling_mean,
    detect_columns,
    extract_frame_index,
    generate_interactive_html,
    load_void_fraction_series,
    plot_void_fraction_timeline,
    run_timeline_pipeline,
)


class TestVoidFractionMathAndParsing(unittest.TestCase):
    def test_extract_frame_index(self):
        self.assertEqual(extract_frame_index("frame_0016"), 16)
        self.assertEqual(extract_frame_index("frame_0028.png"), 28)
        self.assertEqual(extract_frame_index("Test_run_19_2_50kHz_001234.png"), 1234)
        self.assertEqual(extract_frame_index(42), 42)
        self.assertEqual(extract_frame_index(42.0), 42)
        self.assertIsNone(extract_frame_index("no_digits_here"))
        self.assertIsNone(extract_frame_index(np.nan))

    def test_detect_columns_standard(self):
        df = pd.DataFrame(
            {
                "frame": ["f1", "f2"],
                "gas_volume_m3": [1e-6, 2e-6],
                "void_fraction_alpha": [0.25, 0.35],
            }
        )
        f_col, vf_col, vol_col = detect_columns(df)
        self.assertEqual(f_col, "frame")
        self.assertEqual(vf_col, "void_fraction_alpha")
        self.assertEqual(vol_col, "gas_volume_m3")

    def test_detect_columns_matlab_style(self):
        df = pd.DataFrame(
            {
                "Frame": ["frame_0016.png", "frame_0017.png"],
                "Alpha_Voxel": [0.377, 0.396],
                "GasVolume_Voxel_px3": [2.84e7, 2.98e7],
            }
        )
        f_col, vf_col, vol_col = detect_columns(df)
        self.assertEqual(f_col, "Frame")
        self.assertEqual(vf_col, "Alpha_Voxel")
        self.assertEqual(vol_col, "GasVolume_Voxel_px3")

    def test_detect_columns_custom_override(self):
        df = pd.DataFrame(
            {
                "img": [1, 2],
                "my_alpha": [0.1, 0.2],
                "other_col": [5, 6],
            }
        )
        f_col, vf_col, vol_col = detect_columns(df, alpha_col="my_alpha")
        self.assertEqual(vf_col, "my_alpha")

    def test_calculate_rouhani_axelsson(self):
        # Boundary cases
        self.assertEqual(calculate_rouhani_axelsson(0.0, 400.0), 0.0)
        self.assertEqual(calculate_rouhani_axelsson(1.0, 400.0), 1.0)

        # Standard flow boiling case (x = 0.05, G = 400 kg/(m^2 s))
        eps = calculate_rouhani_axelsson(0.05, 400.0)
        self.assertGreater(eps, 0.70)
        self.assertLess(eps, 0.95)

        # Negative / invalid mass flux
        with self.assertRaises(ValueError):
            calculate_rouhani_axelsson(0.05, -10.0)

    def test_compute_rolling_mean(self):
        time_vals = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        alpha_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        t_out, a_roll = compute_rolling_mean(time_vals, alpha_vals, window=3)
        self.assertEqual(len(t_out), len(time_vals))
        self.assertEqual(len(a_roll), len(alpha_vals))
        # Center element for window 3 around index 2 is (0.2 + 0.3 + 0.4) / 3 = 0.3
        self.assertAlmostEqual(a_roll[2], 0.3, places=5)


class TestVoidFractionPipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

        # Create synthetic CSV 1 (Python style)
        self.csv1 = self.test_dir / "run1.csv"
        df1 = pd.DataFrame(
            {
                "frame": [f"frame_{i:04d}" for i in range(10, 20)],
                "gas_volume_m3": [5e-6 + i * 1e-7 for i in range(10)],
                "pipe_volume_m3": [1.7e-5] * 10,
                "void_fraction_alpha": [0.20 + 0.01 * (i % 4) for i in range(10)],
            }
        )
        df1.to_csv(self.csv1, index=False)

        # Create synthetic CSV 2 (MATLAB style)
        self.csv2 = self.test_dir / "run2.csv"
        df2 = pd.DataFrame(
            {
                "Frame": [f"frame_{i:04d}.png" for i in range(10, 20)],
                "Alpha_Voxel": [0.22 + 0.012 * (i % 4) for i in range(10)],
                "GasVolume_Voxel_px3": [2.5e7] * 10,
            }
        )
        df2.to_csv(self.csv2, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_void_fraction_series(self):
        series = load_void_fraction_series(
            self.csv1, fps=2000.0, time_unit="ms", relative_time=True
        )
        self.assertEqual(series["label"], "Run1")
        self.assertEqual(len(series["alpha"]), 10)
        self.assertEqual(series["time_unit"], "ms")
        self.assertEqual(series["time"][0], 0.0)
        # At 2000 FPS, dt = 0.5 ms. 10 frames span from frame 10 to 19 -> 9 steps = 4.5 ms
        self.assertAlmostEqual(series["time"][-1], 4.5, places=3)
        self.assertAlmostEqual(series["stats"]["mean"], float(np.mean(series["alpha"])), places=4)

    def test_load_void_fraction_series_absolute_time(self):
        series = load_void_fraction_series(
            self.csv1, fps=2000.0, time_unit="s", relative_time=False
        )
        self.assertEqual(series["time_unit"], "s")
        # Frame 10 at 2000 FPS is 10 / 2000 = 0.005 s
        self.assertAlmostEqual(series["time"][0], 0.005, places=5)

    def test_plot_void_fraction_timeline_single_panel(self):
        series = load_void_fraction_series(self.csv1, fps=2000.0, time_unit="ms")
        out_png = self.test_dir / "timeline_single.png"
        plot_void_fraction_timeline(
            series_list=[series],
            output_path=out_png,
            rolling_window=3,
            vapor_quality=0.02,
            mass_flux=300.0,
            multi_panel=False,
        )
        self.assertTrue(out_png.exists())
        self.assertGreater(out_png.stat().st_size, 5000)

    def test_plot_void_fraction_timeline_multi_panel(self):
        series = load_void_fraction_series(self.csv1, fps=2000.0, time_unit="ms")
        out_png = self.test_dir / "timeline_multi.png"
        plot_void_fraction_timeline(
            series_list=[series],
            output_path=out_png,
            rolling_window=3,
            multi_panel=True,
        )
        self.assertTrue(out_png.exists())
        self.assertGreater(out_png.stat().st_size, 10000)

    def test_generate_interactive_html(self):
        series = load_void_fraction_series(self.csv1, fps=2000.0, time_unit="ms")
        out_html = self.test_dir / "timeline.html"
        generate_interactive_html(
            series_list=[series],
            output_html_path=out_html,
            rolling_window=3,
            vapor_quality=0.05,
            mass_flux=400.0,
        )
        self.assertTrue(out_html.exists())
        content = out_html.read_text(encoding="utf-8")
        self.assertIn("Chart.js", content)
        self.assertIn("vfChart", content)
        self.assertIn("Rouhani-Axelsson", content)
        self.assertIn("frame_0010", content)

    def test_run_timeline_pipeline_multi_csv(self):
        out_png = self.test_dir / "multi_comparison.png"
        out_html = self.test_dir / "multi_comparison.html"
        plot_path, html_path = run_timeline_pipeline(
            input_paths=[self.csv1, self.csv2],
            output_plot=out_png,
            output_html=out_html,
            labels=["Run A", "Run B"],
            rolling_window=3,
        )
        self.assertTrue(plot_path.exists())
        self.assertTrue(html_path.exists())
        self.assertGreater(plot_path.stat().st_size, 5000)
        self.assertGreater(html_path.stat().st_size, 2000)


if __name__ == "__main__":
    unittest.main()
