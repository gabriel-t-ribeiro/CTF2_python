"""
Comprehensive verification test suite for CTF2 optical pixel size calibration,
report parsing, directory search heuristics, and pipeline auto-detection.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from image_scripts.calibrate_pixel_size import (
    calculate_pixel_scale,
    find_and_load_calibration,
    parse_calibration_file,
    write_calibration_file,
)
from main import _resolve_auto_calibration


class TestCalibrationMath(unittest.TestCase):
    def test_vertical_scale_calculation(self):
        pt1 = (100.0, 50.0)
        pt2 = (100.0, 282.5)
        real_size_m = 0.020  # 20 mm

        metrics = calculate_pixel_scale(pt1, pt2, real_size_m)

        self.assertAlmostEqual(metrics["distance_px"], 232.5, places=4)
        expected_s_px_m = 0.020 / 232.5
        self.assertAlmostEqual(metrics["pixel_size_m"], expected_s_px_m, places=8)
        self.assertAlmostEqual(metrics["pixel_size_mm"], expected_s_px_m * 1000, places=5)
        self.assertAlmostEqual(metrics["pixels_per_mm"], 1.0 / (expected_s_px_m * 1000), places=4)

    def test_diagonal_scale_calculation(self):
        pt1 = (0.0, 0.0)
        pt2 = (300.0, 400.0)
        real_size_m = 0.050

        metrics = calculate_pixel_scale(pt1, pt2, real_size_m)
        self.assertAlmostEqual(metrics["distance_px"], 500.0, places=4)
        self.assertAlmostEqual(metrics["pixel_size_m"], 0.050 / 500.0, places=8)
        self.assertAlmostEqual(metrics["pixel_size_mm"], 0.1, places=5)

    def test_angle_and_orientation_vertical(self):
        pt1 = (100.0, 50.0)
        pt2 = (100.0, 250.0)
        metrics = calculate_pixel_scale(pt1, pt2, 0.020)
        self.assertAlmostEqual(metrics["dx_px"], 0.0, places=4)
        self.assertAlmostEqual(metrics["dy_px"], 200.0, places=4)
        self.assertAlmostEqual(metrics["angle_from_horizontal_deg"], 90.0, places=4)

    def test_angle_and_orientation_horizontal(self):
        pt1 = (50.0, 100.0)
        pt2 = (250.0, 100.0)
        metrics = calculate_pixel_scale(pt1, pt2, 0.020)
        self.assertAlmostEqual(metrics["dx_px"], 200.0, places=4)
        self.assertAlmostEqual(metrics["dy_px"], 0.0, places=4)
        self.assertAlmostEqual(metrics["angle_from_horizontal_deg"], 0.0, places=4)

    def test_angle_and_orientation_diagonal(self):
        pt1 = (0.0, 0.0)
        pt2 = (100.0, 100.0)
        metrics = calculate_pixel_scale(pt1, pt2, 0.020)
        self.assertAlmostEqual(metrics["angle_from_horizontal_deg"], 45.0, places=4)

    def test_zero_distance_raises_error(self):
        pt1 = (50.0, 50.0)
        pt2 = (50.0, 50.0)
        with self.assertRaises(ValueError):
            calculate_pixel_scale(pt1, pt2, 0.020)


class TestCalibrationFileIO(unittest.TestCase):
    def test_write_and_parse_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "images_calibration.txt"
            pt1 = (50.0, 100.0)
            pt2 = (50.0, 350.0)
            real_size = 20.0
            real_unit = "mm"
            real_size_m = 0.020
            metrics = calculate_pixel_scale(pt1, pt2, real_size_m)

            saved = write_calibration_file(
                output_path=out_file,
                folder_name="images",
                ref_image_name="frame_0001.png",
                img_width=1024,
                img_height=1024,
                pt1=pt1,
                pt2=pt2,
                metrics=metrics,
                real_size=real_size,
                real_unit=real_unit,
                real_size_m=real_size_m,
            )

            self.assertTrue(saved.is_file())
            content = saved.read_text(encoding="utf-8")
            self.assertIn("CTF2 Two-Phase Flow Pixel Scale Calibration Report", content)
            self.assertIn("diameter_m=2.000000e-02", content)
            self.assertIn("pixels_per_mm=", content)

            parsed = parse_calibration_file(saved)
            self.assertEqual(parsed["folder_name"], "images")
            self.assertEqual(parsed["reference_image"], "frame_0001.png")
            self.assertEqual(parsed["image_width_px"], 1024)
            self.assertAlmostEqual(parsed["point1_x"], 50.0)
            self.assertAlmostEqual(parsed["distance_px"], 250.0)
            self.assertAlmostEqual(parsed["pixel_size_m"], 0.020 / 250.0)
            self.assertAlmostEqual(parsed["diameter_m"], 0.020)


class TestCalibrationDiscoveryAndPipeline(unittest.TestCase):
    def test_find_and_load_direct_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cal_file = Path(tmpdir) / "test_calibration.txt"
            write_calibration_file(
                output_path=cal_file,
                folder_name="test",
                ref_image_name="frame.png",
                img_width=512,
                img_height=512,
                pt1=(0, 0),
                pt2=(0, 200),
                metrics=calculate_pixel_scale((0, 0), (0, 200), 0.020),
                real_size=20.0,
                real_unit="mm",
                real_size_m=0.020,
            )

            loaded = find_and_load_calibration(cal_file)
            self.assertIsNotNone(loaded)
            self.assertAlmostEqual(loaded["pixel_size_m"], 0.020 / 200.0)

    def test_find_and_load_from_masks_sibling_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            images_dir = dataset_dir / "images"
            masks_dir = dataset_dir / "masks"
            images_dir.mkdir(parents=True)
            masks_dir.mkdir(parents=True)

            cal_file = images_dir / "images_calibration.txt"
            write_calibration_file(
                output_path=cal_file,
                folder_name="images",
                ref_image_name="frame_0016.png",
                img_width=1024,
                img_height=1024,
                pt1=(10, 20),
                pt2=(10, 220),
                metrics=calculate_pixel_scale((10, 20), (10, 220), 0.020),
                real_size=20.0,
                real_unit="mm",
                real_size_m=0.020,
            )

            # Searching from masks_dir should find images_dir/images_calibration.txt
            loaded = find_and_load_calibration(masks_dir)
            self.assertIsNotNone(loaded)
            self.assertAlmostEqual(loaded["distance_px"], 200.0)

    def test_auto_calibration_resolver(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "run_01"
            data_dir.mkdir()
            cal_file = data_dir / "run_01_calibration.txt"
            write_calibration_file(
                output_path=cal_file,
                folder_name="run_01",
                ref_image_name="ref.png",
                img_width=800,
                img_height=600,
                pt1=(0, 0),
                pt2=(0, 250),
                metrics=calculate_pixel_scale((0, 0), (0, 250), 0.025),
                real_size=25.0,
                real_unit="mm",
                real_size_m=0.025,
            )

            class MockArgs:
                pixel_size = None
                diameter = None

            args = MockArgs()
            _resolve_auto_calibration(args, data_dir)
            self.assertAlmostEqual(args.pixel_size, 0.025 / 250.0)
            self.assertAlmostEqual(args.diameter, 0.025)

            # CLI override should be preserved if explicitly provided
            override_args = MockArgs()
            override_args.pixel_size = 9.99e-5
            override_args.diameter = 0.050
            _resolve_auto_calibration(override_args, data_dir)
            self.assertEqual(override_args.pixel_size, 9.99e-5)
            self.assertEqual(override_args.diameter, 0.050)

    def test_multi_folder_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            images_dir = dataset_dir / "images"
            masks_dir = dataset_dir / "masks"
            extra_dir = dataset_dir / "inferred_masks"
            images_dir.mkdir(parents=True)
            masks_dir.mkdir(parents=True)

            # Create a dummy image
            import cv2
            import numpy as np

            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(images_dir / "frame_0001.png"), dummy_img)

            # Test writing calibration directly to images and copying to masks and extra
            from unittest.mock import patch

            from image_scripts.calibrate_pixel_size import run_calibration

            with patch(
                "image_scripts.calibrate_pixel_size.select_calibration_points"
            ) as mock_select:
                mock_metrics = calculate_pixel_scale((0, 0), (0, 100), 0.020)
                mock_select.return_value = ((0.0, 0.0), (0.0, 100.0), mock_metrics)

                run_calibration(
                    input_path=images_dir,
                    size_mm=20.0,
                    copy_to=[masks_dir, extra_dir],
                )

            # Check that 0_images_calibration.txt exists (prefixed for top sort in PyCharm)
            self.assertTrue((images_dir / "0_images_calibration.txt").is_file())
            # Check that 0_masks_calibration.txt exists in masks_dir
            self.assertTrue((masks_dir / "0_masks_calibration.txt").is_file())
            # Check that 0_inferred_masks_calibration.txt was created in extra_dir
            self.assertTrue((extra_dir / "0_inferred_masks_calibration.txt").is_file())

            # Verify contents
            cal_masks = parse_calibration_file(masks_dir / "0_masks_calibration.txt")
            self.assertEqual(cal_masks["folder_name"], "masks")
            self.assertAlmostEqual(cal_masks["pixel_size_m"], 0.020 / 100.0)
            self.assertAlmostEqual(cal_masks["diameter_m"], 0.020)

            # Test auto-loading priority with 0_ prefix
            loaded = find_and_load_calibration(masks_dir)
            self.assertIsNotNone(loaded)
            self.assertIn("0_masks_calibration.txt", loaded["_file_path"])


class TestCropCalibrationHandling(unittest.TestCase):
    def setUp(self):
        import importlib

        self.roi_crop = importlib.import_module("image_scripts.0_roi_crop")

    def test_find_calibration_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            cal1 = d / "0_images_calibration.txt"
            cal2 = d / "old_calibration.txt"
            img = d / "frame_0001.png"
            cal1.write_text("test")
            cal2.write_text("test")
            img.write_text("dummy")

            found = self.roi_crop.find_calibration_files(d)
            self.assertEqual(len(found), 2)
            self.assertIn(cal1, found)
            self.assertIn(cal2, found)

    def test_check_and_handle_keep(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            in_dir = Path(tmpdir) / "in"
            out_dir = Path(tmpdir) / "out"
            in_dir.mkdir()
            out_dir.mkdir()
            cal_file = in_dir / "0_in_calibration.txt"
            cal_file.write_text("pixel_size_m: 0.0001")

            action, found, kwargs = self.roi_crop.check_and_handle_calibration(
                in_dir, out_dir, action="keep"
            )
            self.assertEqual(action, "keep")
            self.assertEqual(len(found), 1)
            self.assertTrue(cal_file.exists())

    def test_check_and_handle_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            in_dir = Path(tmpdir) / "in"
            out_dir = Path(tmpdir) / "out"
            in_dir.mkdir()
            out_dir.mkdir()
            cal_in = in_dir / "0_in_calibration.txt"
            cal_out = out_dir / "0_out_calibration.txt"
            cal_in.write_text("test")
            cal_out.write_text("test")

            action, found, kwargs = self.roi_crop.check_and_handle_calibration(
                in_dir, out_dir, action="delete"
            )
            self.assertEqual(action, "delete")
            self.assertEqual(len(found), 2)
            self.assertFalse(cal_in.exists())
            self.assertFalse(cal_out.exists())

    def test_check_and_handle_redo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            in_dir = Path(tmpdir) / "in"
            out_dir = Path(tmpdir) / "out"
            in_dir.mkdir()
            out_dir.mkdir()
            cal_in = in_dir / "0_in_calibration.txt"
            cal_out = out_dir / "0_out_calibration.txt"
            cal_in.write_text("test")
            cal_out.write_text("test")

            action, found, kwargs = self.roi_crop.check_and_handle_calibration(
                in_dir, out_dir, action="redo"
            )
            self.assertEqual(action, "redo")
            # Old calibration in out_dir should be purged prior to recalibration
            self.assertFalse(cal_out.exists())
            # In_dir original calibration kept
            self.assertTrue(cal_in.exists())

    def test_prompt_calibration_action_variations(self):
        from unittest.mock import patch

        dummy_files = [Path("test_calibration.txt")]
        dummy_recalled = {
            "_file_path": "test_calibration.txt",
            "diameter_real": 20.0,
            "diameter_unit": "mm",
            "pixel_size_mm": 0.086,
            "pixels_per_mm": 11.625,
            "distance_px": 232.5,
        }

        with patch("builtins.input", side_effect=[""]):
            act, kwargs = self.roi_crop.prompt_calibration_action(dummy_files)
            self.assertEqual(act, "delete")
        with patch("builtins.input", side_effect=["1"]):
            act, kwargs = self.roi_crop.prompt_calibration_action(dummy_files)
            self.assertEqual(act, "delete")
        with patch("builtins.input", side_effect=["delete"]):
            act, kwargs = self.roi_crop.prompt_calibration_action(dummy_files)
            self.assertEqual(act, "delete")
        with patch("builtins.input", side_effect=["2"]):
            act, kwargs = self.roi_crop.prompt_calibration_action(dummy_files)
            self.assertEqual(act, "keep")
        with patch("builtins.input", side_effect=["keep"]):
            act, kwargs = self.roi_crop.prompt_calibration_action(dummy_files)
            self.assertEqual(act, "keep")

        # Test selecting '3' (redo) with default Enter
        with patch("builtins.input", side_effect=["3", ""]):
            act, kwargs = self.roi_crop.prompt_calibration_action(
                dummy_files, recalled_cal=dummy_recalled
            )
            self.assertEqual(act, "redo")
            self.assertEqual(kwargs.get("size_mm"), 20.0)

        # Test selecting 'redo' with custom CLI flags
        with patch("builtins.input", side_effect=["redo", "--size-mm 25 -c dataset/masks"]):
            act, kwargs = self.roi_crop.prompt_calibration_action(
                dummy_files, recalled_cal=dummy_recalled
            )
            self.assertEqual(act, "redo")
            self.assertEqual(kwargs.get("size_mm"), 25.0)
            self.assertEqual(kwargs.get("copy_to"), [Path("dataset/masks")])

    def test_parse_custom_calibration_call_and_formatting(self):
        cal = {
            "_file_path": "0_images_calibration.txt",
            "diameter_real": 20.0,
            "diameter_unit": "mm",
            "diameter_m": 0.020,
            "pixel_size_mm": 0.086022,
            "pixel_size_um": 86.02,
            "pixels_per_mm": 11.625,
            "distance_px": 232.5,
        }
        text = self.roi_crop.format_recalled_calibration(cal)
        self.assertIn("20.0 mm", text)
        self.assertIn("11.6250 px/mm", text)
        self.assertIn("0.086022 mm/px", text)

        # Parse custom call string
        parsed = self.roi_crop.parse_custom_calibration_call(
            "--size-mm 20 --copy-to dataset/masks -ref frame_0005.png -o results/cal.txt",
            Path("output"),
            recalled_cal=cal,
        )
        self.assertEqual(parsed["size_mm"], 20.0)
        self.assertEqual(parsed["copy_to"], [Path("dataset/masks")])
        self.assertEqual(parsed["ref_arg"], "frame_0005.png")
        self.assertEqual(parsed["output_path"], Path("results/cal.txt"))

    def test_non_interactive_defaults_to_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            in_dir = Path(tmpdir) / "in"
            out_dir = Path(tmpdir) / "out"
            in_dir.mkdir()
            out_dir.mkdir()
            cal_in = in_dir / "0_in_calibration.txt"
            cal_in.write_text("test")

            from unittest.mock import patch

            with patch("sys.stdin.isatty", return_value=False):
                action, found, kwargs = self.roi_crop.check_and_handle_calibration(
                    in_dir, out_dir, action=None
                )
                self.assertEqual(action, "delete")
                self.assertFalse(cal_in.exists())

    def test_run_crop_pipeline_integration_with_calibration(self):
        from unittest.mock import patch

        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            in_dir = Path(tmpdir) / "raw"
            out_dir = Path(tmpdir) / "cropped"
            in_dir.mkdir()

            # Create dummy 50x50 image
            img = np.ones((50, 50, 3), dtype=np.uint8) * 128
            cv2.imwrite(str(in_dir / "frame_0001.png"), img)

            # Create calibration file in in_dir
            cal_file = in_dir / "0_raw_calibration.txt"
            cal_file.write_text(
                "diameter_real=20.0\ndiameter_unit=mm\ndiameter_m=0.020\npixel_size_m=0.0001\n"
            )

            # 1. Test crop with action="delete"
            self.roi_crop.run_crop_pipeline(
                input_dir=in_dir,
                output_dir=out_dir,
                roi_arg="5,5,20,20",
                on_calibration="delete",
            )
            cropped_img = cv2.imread(str(out_dir / "frame_0001.png"))
            self.assertIsNotNone(cropped_img)
            self.assertEqual(cropped_img.shape[:2], (20, 20))
            self.assertFalse(cal_file.exists())

            # 2. Test crop with action="redo" and custom redo_args
            cal_file.write_text(
                "diameter_real=20.0\ndiameter_unit=mm\ndiameter_m=0.020\npixel_size_m=0.0001\n"
            )
            with patch("image_scripts.calibrate_pixel_size.run_calibration") as mock_cal:
                self.roi_crop.run_crop_pipeline(
                    input_dir=in_dir,
                    output_dir=out_dir,
                    roi_arg="5,5,15,15",
                    on_calibration="redo",
                    redo_args="--size-mm 20 -c dataset/masks",
                )
                mock_cal.assert_called_once_with(
                    input_path=out_dir.resolve(),
                    size_mm=20.0,
                    size_m=None,
                    ref_arg=None,
                    output_path=None,
                    copy_to=[Path("dataset/masks")],
                )


def run_all():
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == "__main__":
    run_all()
