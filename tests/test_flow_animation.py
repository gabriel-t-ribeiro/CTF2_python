#!/usr/bin/env python3
"""
Comprehensive test suite for high-speed bubble velocity vector field visualization
and flow animation pipeline (MP4, PNG figures, animated GIF, interactive HTML, and SI CSV).
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_training.visualize_flow_animation import (
    compute_boundary_vectors,
    extract_bubble_blobs,
    get_colormap_color,
    render_flow_animation,
    sample_contour_evenly,
)


class TestFlowAnimationMath(unittest.TestCase):
    def test_sample_contour_evenly(self):
        # Create a square contour of 100x100
        # Perimeter is ~400 px
        pts = []
        for x in range(0, 100):
            pts.append([x, 0])
        for y in range(0, 100):
            pts.append([100, y])
        for x in range(100, 0, -1):
            pts.append([x, 100])
        for y in range(100, 0, -1):
            pts.append([0, y])
        cnt = np.array(pts, dtype=np.int32)

        sampled = sample_contour_evenly(cnt, step_px=20.0)
        # Expected around 400 / 20 = 20 points
        self.assertGreaterEqual(len(sampled), 15)
        self.assertLessEqual(len(sampled), 30)

    def test_extract_bubble_blobs(self):
        # Create synthetic mask with 2 distinct circular bubbles
        mask = np.zeros((150, 150), dtype=np.uint8)
        cv2.circle(mask, (40, 40), 15, 255, -1)
        cv2.circle(mask, (100, 100), 20, 255, -1)

        blobs = extract_bubble_blobs(mask, pixel_size_m=1e-4, min_area_px=15)
        self.assertEqual(len(blobs), 2)
        for b in blobs:
            self.assertIn("contour_px", b)
            self.assertIn("centroid_px", b)
            self.assertIn("area_m2", b)
            self.assertGreater(b["area_px"], 50)

    def test_boundary_velocity_si_units(self):
        # Create two vertical line contours translated by 10 px in x at 2000 FPS
        # dt = 1/2000 = 0.0005 s
        # pixel_size = 1e-4 m/px
        # dx_m = 10 * 1e-4 = 1e-3 m
        # vx = 1e-3 / 0.0005 = 2.0 m/s
        y_vals = np.linspace(20, 80, 50)
        c1 = np.column_stack((np.full_like(y_vals, 50.0), y_vals))
        c2 = np.column_stack((np.full_like(y_vals, 60.0), y_vals))

        dt_s = 1.0 / 2000.0
        pixel_size_m = 1e-4

        vecs = compute_boundary_vectors(
            c1, c2, step_px=10.0, dt_s=dt_s, pixel_size_m=pixel_size_m, max_disp_px=50.0
        )
        self.assertGreater(len(vecs), 0)

        for v in vecs:
            self.assertAlmostEqual(v["vx_m_s"], 2.0, places=4)
            self.assertAlmostEqual(v["vy_m_s"], 0.0, places=4)
            self.assertAlmostEqual(v["speed_m_s"], 2.0, places=4)

    def test_colormap_color_boundaries(self):
        # Speed 0.0 should map to low end
        c_low = get_colormap_color(0.0, 5.0, cv2.COLORMAP_TURBO)
        self.assertEqual(len(c_low), 3)

        # Speed at max should map to high end
        c_high = get_colormap_color(5.0, 5.0, cv2.COLORMAP_TURBO)
        self.assertEqual(len(c_high), 3)
        self.assertNotEqual(c_low, c_high)


class TestFlowAnimationPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)
        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "masks"
        self.out_dir = self.root / "results"
        self.images_dir.mkdir(parents=True)
        self.masks_dir.mkdir(parents=True)

        # Generate 3 synthetic frames of translating circle bubble
        for t in range(3):
            # 200x300 canvas
            img = np.full((200, 300, 3), 200, dtype=np.uint8)
            mask = np.zeros((200, 300), dtype=np.uint8)

            # Bubble moves right by 8 px per frame
            cx = 60 + t * 8
            cy = 80
            cv2.circle(img, (cx, cy), 25, (40, 40, 40), -1)
            cv2.circle(mask, (cx, cy), 25, 255, -1)

            cv2.imwrite(str(self.images_dir / f"frame_{t:04d}.png"), img)
            cv2.imwrite(str(self.masks_dir / f"frame_{t:04d}.png"), mask)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_full_pipeline_generation(self):
        res = render_flow_animation(
            image_dir=self.images_dir,
            mask_dir=self.masks_dir,
            output_dir=self.out_dir,
            fps=2000.0,
            pixel_size_m=6.1e-5,
            max_disp_px=50.0,
            vector_step_px=10.0,
            playback_fps=5.0,
        )

        self.assertTrue(res["video_path"].exists())
        self.assertGreater(res["video_path"].stat().st_size, 0)

        self.assertTrue(res["html_path"].exists())
        self.assertGreater(res["html_path"].stat().st_size, 0)

        self.assertTrue(res["csv_path"].exists())
        self.assertGreater(res["csv_path"].stat().st_size, 0)

        # GIF should also be created
        gif_file = self.out_dir / "bubble_velocity_flow_animation.gif"
        self.assertTrue(gif_file.exists())
        self.assertGreater(gif_file.stat().st_size, 0)

        # Frame PNGs should exist (2 transitions for 3 frames)
        frame_pngs = list(res["frames_dir"].glob("*.png"))
        self.assertEqual(len(frame_pngs), 2)

    def test_cli_animate_flow_command(self):
        import subprocess

        cmd = [
            sys.executable,
            str(REPO_ROOT / "main.py"),
            "animate-flow",
            "--masks",
            str(self.masks_dir),
            "--images",
            str(self.images_dir),
            "-o",
            str(self.out_dir / "cli_anim"),
            "--fps",
            "2000",
            "--pixel-size",
            "6.1e-5",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"CLI animate-flow failed: {res.stderr}")
        self.assertTrue((self.out_dir / "cli_anim" / "bubble_velocity_flow_animation.mp4").exists())
        self.assertTrue((self.out_dir / "cli_anim" / "interactive_flow_viewer.html").exists())

    def test_cli_track_with_render_animation(self):
        import subprocess

        cmd = [
            sys.executable,
            str(REPO_ROOT / "main.py"),
            "track",
            "--masks",
            str(self.masks_dir),
            "--images",
            str(self.images_dir),
            "--render-animation",
            "-o",
            str(self.out_dir / "cli_track"),
            "--fps",
            "2000",
            "--pixel-size",
            "6.1e-5",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"CLI track failed: {res.stderr}")
        self.assertTrue((self.out_dir / "cli_track" / "bubble_velocities_si.csv").exists())
        self.assertTrue(
            (
                self.out_dir / "cli_track" / "animation" / "bubble_velocity_flow_animation.mp4"
            ).exists()
        )
        self.assertTrue(
            (self.out_dir / "cli_track" / "animation" / "interactive_flow_viewer.html").exists()
        )

    def test_trail_pruning_on_bubble_exit(self):
        # Create sequence where bubble disappears in frame 2
        p_masks = self.root / "masks_disappear"
        p_masks.mkdir(parents=True)
        for t in range(4):
            mask = np.zeros((100, 100), dtype=np.uint8)
            if t < 2:  # Bubble only present in frames 0 and 1
                cv2.circle(mask, (30 + t * 5, 50), 10, 255, -1)
            cv2.imwrite(str(p_masks / f"mask_{t:04d}.png"), mask)

        out_disappear = self.root / "out_disappear"
        res = render_flow_animation(
            image_dir=None,
            mask_dir=p_masks,
            output_dir=out_disappear,
            fps=2000.0,
            pixel_size_m=1e-4,
            save_individual_frames=True,
        )
        self.assertTrue(res["video_path"].exists())

    def test_stride_and_duration_enforcement(self):
        # Create 50 synthetic frames
        p_many = self.root / "masks_many"
        p_many.mkdir(parents=True)
        for t in range(50):
            mask = np.zeros((80, 80), dtype=np.uint8)
            cv2.circle(mask, (20 + (t % 30), 40), 8, 255, -1)
            cv2.imwrite(str(p_many / f"mask_{t:04d}.png"), mask)

        out_many = self.root / "out_many"
        res = render_flow_animation(
            image_dir=None,
            mask_dir=p_many,
            output_dir=out_many,
            fps=2000.0,
            pixel_size_m=1e-4,
            playback_fps=60.0,
            max_duration_s=0.5,  # Force short max duration: at 60 fps, max 30 frames -> stride ceil(50/30) = 2
            max_html_frames=10,
        )
        self.assertTrue(res["video_path"].exists())
        self.assertLessEqual(res["duration_s"], 0.55)
        # Verify HTML file is generated and compact
        self.assertLess(res["html_path"].stat().st_size, 5 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
