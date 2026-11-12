#!/usr/bin/env python3
"""
Single-Camera Depth from Defocus (DFD): sizing, depth and ground-truth evaluation (SI units).

Implements the calibration-free single-camera DFD technique of:

  Rao, D. C. K., Sharma, S., Basu, S. & Tropea, C. (2024). "Depth from defocus technique:
  a simple calibration-free approach for dispersion size measurement."
  Experiments in Fluids 65, 55.

Equation numbers below refer to that article.

Theory (Rao et al., 2024)
-------------------------
  Blur kernel (Eq. 3)  : Gaussian PSF with standard deviation sigma.
  Blur vs. depth (Eq. 5): sigma = beta * |Delta z|, Delta z measured from the object (focal) plane.
  Scaling (Eq. 6)      : rho~ = r / d_0,  rho~_t = r_t / d_0,  sigma~ = sigma / d_0.
  Blurred disc (Eq. 7) : g_t(rho~_t) = (1 / sigma~^2) * Int_0^{1/2} exp(-(rho~^2 + rho~_t^2) / (2 sigma~^2))
                                        * I_0(rho~ rho~_t / sigma~^2) * rho~ d rho~
  Gradient (Eq. 9)     : G~ = r_t * |dg_t / dr_t| at g_t = 0.5  ->  G~ = f2(sigma~)
  Radius (Eq. 10)      : rho~_t = f1(sigma~) at g_t = 0.5
  Size                 : sigma~ = f2^-1(G~),  d_0 = r_t / f1(sigma~),  sigma = sigma~ * d_0
  Depth (Sect. 3.2)    : |Delta z| = sigma / beta   (linear experimental calibration)
  Validity (Eq. 13)    : results are disregarded beyond sigma~_c = 0.35
                         (there rho~_t = 0.3211 and G~ = 0.2501); for sigma~ < 0.05 the depth
                         estimate is unreliable while the diameter estimate stays robust.

Units
-----
  All lengths are metres, except image-plane quantities which are pixels (suffix `_px`).
  `beta_eff` is expressed in px/m: blur sigma [px] produced per metre of defocus distance, so
  |Delta z| [m] = sigma [px] / beta_eff [px/m]. An in-situ estimate from the tube itself is
  beta_eff = sigma_rear [px] / D_int [m] (blur of an object sitting on the rear inner wall).

Adaptations for flow boiling in a circular tube (this project, not part of Rao et al.)
-------------------------------------------------------------------------------------
  1. r_t is the equivalent radius sqrt(A / pi) of the segmentation mask (U-Net or Label Studio
     ground truth) instead of the area enclosed by the g_t = 0.5 iso-contour.
  2. G~ uses a 15-85 % trimmed mean of the Sobel gradient magnitude along the mask contour
     instead of the mean inside the thin strip g_t = 0.5 +/- 0.005.
  3. Near focus (sigma~ < 0.05) a smoothstep regulariser blends the estimate towards the in-focus
     limit (rho~_t = 0.5, sigma = 0) so that noisy steep gradients do not produce spurious depth.
  4. Depth is confined to the tube chord L_chord(y) = 2 * sqrt(R^2 - (y - y_c)^2); slugs and
     strongly non-circular bodies are excluded and handled by the EDT reconstruction instead.

Input modes
-----------
  --images DIR --masks DIR    paired raw frames and binary masks (U-Net output or ground truth)
  --json FILE [--images DIR]  Label Studio export (brush, polygon, ellipse, rectangle labels)
  --prepare-tasks --images DIR  sample evenly spaced frames for Label Studio annotation
"""

import argparse
import base64
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import i0e, i1e

# Repository root on sys.path so `visualization.*` resolves when this file runs as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Physical and optical constants (SI)
# ---------------------------------------------------------------------------

# The optical scale has no default: it comes from a calibration file or from --pixel-size.
# Effective defocus constant beta in px/m (placeholder until an in-situ calibration is done).
DEFAULT_BETA_EFF_PX_PER_M = 2000.0
DEFAULT_INNER_DIAMETER_M = 0.020
DEFAULT_WALL_THICKNESS_M = 0.003

# Rao et al. (2024): reference intensity, near-focus bound and critical blur (Eq. 13).
REFERENCE_INTENSITY = 0.5
SIGMA_TILDE_NEAR_FOCUS = 0.05
SIGMA_TILDE_CRITICAL = 0.35

# Dual-regime area threshold shared with visualization/render_3d_flow.py.
SLUG_AREA_THRESHOLD_PX = 5000.0
MIN_BUBBLE_AREA_PX = 15
MAX_ASPECT_RATIO = 2.5

STATUS_COLORS = {
    "valid": "#00E676",
    "near_wall_regularized": "#00B0FF",
    "saturated_rear_wall": "#FF5252",
    "unreliable_defocus": "#FF9100",
    "slug_excluded": "#9E9E9E",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Column order of the per-entity CSV written by the pipeline.
DFD_COLUMNS = [
    "bubble_id",
    "centroid_x_m",
    "centroid_y_m",
    "depth_z_m",
    "depth_fluid_z_m",
    "diameter_d0_m",
    "blur_sigma_px",
    "sigma_tilde",
    "g_tilde",
    "near_wall_zone",
    "is_reliable_dfd",
    "dfd_status",
    "area_px",
    "circularity",
    "aspect_ratio",
    "l_chord_m",
    "z_rear_wall_m",
    "centroid_x_px",
    "centroid_y_px",
    "frame",
]

# ---------------------------------------------------------------------------
# 1. Label Studio RLE & Geometry Decoders
# ---------------------------------------------------------------------------


class _InputStream:
    """Helper stream for bitwise parsing of Label Studio RLE brush data."""

    def __init__(self, data: str):
        self.data = data
        self.i = 0

    def read(self, size: int) -> int:
        out = self.data[self.i : self.i + size]
        self.i += size
        return int(out, 2)


def _access_bit(data: bytes, num: int) -> int:
    base = int(num // 8)
    shift = 7 - int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def _bytes2bit(data: bytes) -> str:
    return "".join([str(_access_bit(data, i)) for i in range(len(data) * 8)])


def decode_rle_brush(rle: list[int], height: int, width: int) -> np.ndarray:
    """
    Decodes Label Studio 4-channel RGBA RLE brush bitstream into a 2D uint8 binary mask.
    """
    inp = _InputStream(_bytes2bit(bytes(rle)))
    num = inp.read(32)
    word_size = inp.read(5) + 1
    rle_sizes = [inp.read(4) + 1 for _ in range(4)]
    i = 0
    out = np.zeros(num, dtype=np.uint8)
    while i < num:
        x = inp.read(1)
        j = i + 1 + inp.read(rle_sizes[inp.read(2)])
        if x:
            val = inp.read(word_size)
            out[i:j] = val
            i = j
        else:
            while i < j:
                out[i] = inp.read(word_size)
                i += 1

    if num == height * width * 4:
        rgba = out.reshape((height, width, 4))
        mask = (rgba[:, :, 3] > 0).astype(np.uint8) * 255
    elif num == height * width:
        mask = (out.reshape((height, width)) > 0).astype(np.uint8) * 255
    else:
        mask = (out[: height * width].reshape((height, width)) > 0).astype(np.uint8) * 255
    return mask


def rasterize_label_studio_results(
    results: list[dict[str, Any]], height: int, width: int
) -> np.ndarray:
    """
    Renders polygon, brush, ellipse, and rectangle annotations into a single binary mask.
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    for item in results:
        item_type = item.get("type")
        val = item.get("value", {})

        if item_type == "polygonlabels":
            points = val.get("points", [])
            if len(points) >= 3:
                pts_px = np.array(
                    [
                        [round(pt[0] * width / 100.0), round(pt[1] * height / 100.0)]
                        for pt in points
                    ],
                    dtype=np.int32,
                )
                cv2.fillPoly(mask, [pts_px], 255)

        elif item_type == "brushlabels":
            rle = val.get("rle")
            if rle:
                brush_mask = decode_rle_brush(rle, height, width)
                mask = np.maximum(mask, brush_mask)

        elif item_type == "ellipselabels":
            # Label Studio stores x, y as top-left percentage of unrotated bounding box
            x_pct = val.get("x", 0.0)
            y_pct = val.get("y", 0.0)
            rx_pct = val.get("radiusX", 0.0)
            ry_pct = val.get("radiusY", 0.0)
            rot = val.get("rotation", 0.0)

            cx = (x_pct + rx_pct) * width / 100.0
            cy = (y_pct + ry_pct) * height / 100.0
            rx = rx_pct * width / 100.0
            ry = ry_pct * height / 100.0

            if rx > 0 and ry > 0:
                cv2.ellipse(
                    mask,
                    (int(round(cx)), int(round(cy))),
                    (int(round(rx)), int(round(ry))),
                    rot,
                    0,
                    360,
                    255,
                    -1,
                )

        elif item_type == "rectanglelabels":
            x_pct = val.get("x", 0.0)
            y_pct = val.get("y", 0.0)
            w_pct = val.get("width", 0.0)
            h_pct = val.get("height", 0.0)
            rot = val.get("rotation", 0.0)

            x1 = round(x_pct * width / 100.0)
            y1 = round(y_pct * height / 100.0)
            w_box = round(w_pct * width / 100.0)
            h_box = round(h_pct * height / 100.0)

            if rot == 0.0:
                cv2.rectangle(mask, (int(x1), int(y1)), (int(x1 + w_box), int(y1 + h_box)), 255, -1)
            else:
                rect = ((x1 + w_box / 2.0, y1 + h_box / 2.0), (w_box, h_box), rot)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                cv2.fillPoly(mask, [box], 255)

    return mask


# ---------------------------------------------------------------------------
# 2. Calibration Auto-Detection Helper
# ---------------------------------------------------------------------------


def auto_detect_calibration(
    search_dirs: list[Path], target_shape: tuple[int, int] | None = None
) -> dict[str, Any]:
    """
    Searches candidate directories and parent trees for machine-readable '0_*calibration.txt' files.
    If target_shape (H, W) is provided, prioritizes calibration files with matching image dimensions.
    """
    candidates_all: list[Path] = []
    for d in search_dirs:
        curr = d if d.is_dir() else d.parent
        for _ in range(4):
            if curr.exists():
                candidates_all.extend(list(curr.glob("*calibration.txt")))
                candidates_all.extend(list(curr.glob("**/*calibration.txt")))
            curr = curr.parent

    # Remove duplicates preserving order
    seen: set[Path] = set()
    candidate_files: list[Path] = []
    for c in candidates_all:
        if c not in seen:
            seen.add(c)
            candidate_files.append(c)

    best_match = None
    first_valid = None

    for cal_file in candidate_files:
        try:
            params = {}
            with open(cal_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip()
                        try:
                            params[k] = float(v)
                        except ValueError:
                            params[k] = v
            if "pixel_size_m" in params:
                params["source_file"] = cal_file
                if first_valid is None:
                    first_valid = params
                if target_shape is not None:
                    h_target, w_target = target_shape
                    cal_w = params.get("image_width_px")
                    cal_h = params.get("image_height_px")
                    if cal_w == w_target and cal_h == h_target:
                        best_match = params
                        break
        except Exception:
            continue

    if best_match is not None:
        return best_match
    # A calibration recorded for a different image size belongs to another optical setup.
    if first_valid is not None and target_shape is None:
        return first_valid
    return {}


# ---------------------------------------------------------------------------
# 3. DFD analytical engine (Rao et al., 2024)
# ---------------------------------------------------------------------------


def to_si_pixel_size(pixel_size: float) -> float:
    """Return the pixel size in m/px. Values above 1e-3 are taken as mm/px and converted."""
    return pixel_size * 1e-3 if pixel_size > 0.001 else pixel_size


def _compute_disc_convolved_profile(sigma_tilde: float, rho_tilde: float) -> tuple[float, float]:
    """
    Intensity g_t and radial derivative dg_t/d(rho~_t) of a unit-diameter disc convolved with a
    Gaussian PSF of standard deviation sigma~ (Rao et al., Eq. 7).

    The Bessel term is evaluated with the exponentially scaled functions I0e / I1e so that the
    integrand stays finite as sigma~ -> 0:
      exp(-(r^2 + rho^2) / (2 s^2)) * I0(r rho / s^2) = exp(-(r - rho)^2 / (2 s^2)) * I0e(r rho / s^2)
    """
    if sigma_tilde < 1e-4:
        g = 1.0 if rho_tilde < 0.5 else (0.5 if abs(rho_tilde - 0.5) < 1e-4 else 0.0)
        return g, 0.0

    s2 = sigma_tilde**2

    def integrand(r: float) -> float:
        arg = (r * rho_tilde) / s2
        exp_factor = np.exp(-((r - rho_tilde) ** 2) / (2.0 * s2))
        return (r / s2) * exp_factor * i0e(arg)

    def integrand_deriv(r: float) -> float:
        arg = (r * rho_tilde) / s2
        exp_factor = np.exp(-((r - rho_tilde) ** 2) / (2.0 * s2))
        d_term = (-rho_tilde / s2) * i0e(arg) + (r / s2) * i1e(arg)
        return (r / s2) * exp_factor * d_term

    val, _ = quad(integrand, 0.0, 0.5, limit=50)
    deriv, _ = quad(integrand_deriv, 0.0, 0.5, limit=50)
    return float(val), float(deriv)


class DFDModel:
    """
    Lookup table for the analytical functions of Rao et al. (2024) at g_t = 0.5:
      f2: sigma~ -> G~      (Eq. 9, dimensionless gradient)
      f1: sigma~ -> rho~_t  (Eq. 10, dimensionless threshold radius)

    The table keeps the original 45 nodes on [0.005, 0.35] and continues with the same spacing
    slightly past the critical blur sigma~_c = 0.35, so that measurements beyond the validity
    limit invert to sigma~ > 0.35 and can be rejected instead of being clamped onto the limit.
    """

    _SIGMA_MIN = 0.005
    _SIGMA_STEP = (SIGMA_TILDE_CRITICAL - 0.005) / 44.0
    _N_NODES = 51  # 45 nodes up to sigma~_c plus 6 nodes of margin (up to ~0.397)

    def __init__(self) -> None:
        sigmas = self._SIGMA_MIN + self._SIGMA_STEP * np.arange(self._N_NODES)
        rhos = []
        g_tildes = []

        for s in sigmas:
            # Bisection for the radius where the blurred profile crosses the reference intensity.
            low, high = 0.0, 0.52
            for _ in range(20):
                mid = 0.5 * (low + high)
                g_val, _ = _compute_disc_convolved_profile(s, mid)
                if g_val > REFERENCE_INTENSITY:
                    low = mid
                else:
                    high = mid
            rho_ref = 0.5 * (low + high)
            _, dg_ref = _compute_disc_convolved_profile(s, rho_ref)

            rhos.append(rho_ref)
            g_tildes.append(abs(rho_ref * dg_ref))

        self._sigmas = sigmas
        self._rhos = np.array(rhos)
        self._g_tildes = np.array(g_tildes)

        sort_idx = np.argsort(self._g_tildes)
        self._interp_sigma = interp1d(
            self._g_tildes[sort_idx],
            self._sigmas[sort_idx],
            kind="linear",
            bounds_error=False,
            fill_value=(self._sigmas[sort_idx[0]], self._sigmas[sort_idx[-1]]),
        )
        self._interp_rho = interp1d(
            self._sigmas,
            self._rhos,
            kind="linear",
            bounds_error=False,
            fill_value=(self._rhos[0], self._rhos[-1]),
        )
        self._interp_g = interp1d(self._sigmas, self._g_tildes, kind="linear")

    def f1(self, sigma_tilde: float) -> float:
        """Dimensionless threshold radius rho~_t = f1(sigma~) (Eq. 10)."""
        return float(self._interp_rho(sigma_tilde))

    def f2(self, sigma_tilde: float) -> float:
        """Dimensionless gradient G~ = f2(sigma~) (Eq. 9)."""
        return float(self._interp_g(sigma_tilde))

    def invert(self, g_tilde: float) -> tuple[float, float]:
        """Return (sigma~, rho~_t) = (f2^-1(G~), f1(f2^-1(G~)))."""
        s_tilde = float(self._interp_sigma(g_tilde))
        return s_tilde, self.f1(s_tilde)


def _normalise_intensity(raw_gray: np.ndarray, background_img: np.ndarray | None) -> np.ndarray:
    """
    Shadow-image normalisation to [0, 1] (Rao et al., Eq. 15): background -> 0, shadow -> 1.

    With a background frame I_bi the darkest shadow in the frame stands in for the black-shade
    level I_bs. Without one, the 8-bit range is used (I_bi = 255, I_bs = 0).
    """
    raw_f = raw_gray.astype(np.float32)
    if background_img is None:
        return (255.0 - raw_f) / 255.0
    diff = np.maximum(0.0, background_img.astype(np.float32) - raw_f)
    max_diff = float(np.max(diff))
    return diff / (max_diff if max_diff > 0 else 255.0)


def analyze_frame_dfd(
    raw_img: np.ndarray,
    binary_mask: np.ndarray,
    dfd_model: DFDModel,
    pixel_size_m: float,
    beta_eff_px_m: float = DEFAULT_BETA_EFF_PX_PER_M,
    background_img: np.ndarray | None = None,
    near_wall_sigma_threshold: float = SIGMA_TILDE_NEAR_FOCUS,
    max_sigma_threshold: float = SIGMA_TILDE_CRITICAL,
    wall_thickness_m: float = DEFAULT_WALL_THICKNESS_M,
    inner_diameter_m: float = DEFAULT_INNER_DIAMETER_M,
    pipe_center_y_px: float | None = None,
    max_bubble_area_px: float | None = None,
    min_circularity: float = 0.55,
    focal_plane_z_m: float | None = None,
) -> list[dict[str, Any]]:
    """
    Estimate (x, y, z) and the in-focus diameter d_0 of every bubble in one frame.

    Coordinates: z = 0 at the outer glass surface facing the camera; the fluid at elevation y
    spans z in [z_front(y), z_front(y) + L_chord(y)] with
    z_front(y) = wall_thickness + R - L_chord(y) / 2.

    focal_plane_z_m selects how |Delta z| = sigma / beta is anchored:
      None  -> the focus is taken at the local front fluid boundary, so the fluid depth is
               |Delta z| itself (convention used for all results produced so far).
      float -> planar object plane at that absolute z (Rao et al., Eq. 5); the bubble is placed
               behind it at z = focal_plane_z_m + |Delta z|.

    Status values written to `dfd_status`:
      slug_excluded          area >= max_bubble_area_px, aspect ratio > 2.5 or low circularity
      valid                  near_wall_sigma_threshold <= sigma~ <= max_sigma_threshold
      near_wall_regularized  sigma~ < near_wall_sigma_threshold (smoothstep regulariser applied)
      unreliable_defocus     sigma~ > max_sigma_threshold (beyond sigma~_c, Eq. 13)
      saturated_rear_wall    estimated depth reaches the rear inner wall
    """
    h, w = raw_img.shape[:2]
    raw_gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY) if raw_img.ndim == 3 else raw_img

    if pipe_center_y_px is None:
        pipe_center_y_px = float(h) / 2.0
    if max_bubble_area_px is None:
        max_bubble_area_px = SLUG_AREA_THRESHOLD_PX

    r_tube_m = inner_diameter_m / 2.0
    norm_img = _normalise_intensity(raw_gray, background_img)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    # 3x3 Sobel scaled by 1/8 returns the intensity derivative per pixel.
    sobel_x = cv2.Sobel(norm_img, cv2.CV_64F, 1, 0, ksize=3, scale=1.0 / 8.0)
    sobel_y = cv2.Sobel(norm_img, cv2.CV_64F, 0, 1, ksize=3, scale=1.0 / 8.0)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    results: list[dict[str, Any]] = []

    for i in range(1, num_labels):
        area_px = int(stats[i, cv2.CC_STAT_AREA])
        if area_px < MIN_BUBBLE_AREA_PX:
            continue

        c_x_px, c_y_px = float(centroids[i][0]), float(centroids[i][1])
        w_blob = stats[i, cv2.CC_STAT_WIDTH]
        h_blob = stats[i, cv2.CC_STAT_HEIGHT]
        aspect_ratio = max(w_blob, h_blob) / max(1, min(w_blob, h_blob))
        # Equivalent threshold radius r_t = sqrt(A_t / pi) (Rao et al., Sect. 3.3)
        r_t_px = float(np.sqrt(area_px / np.pi))

        blob_mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue

        perimeter_px = cv2.arcLength(contours[0], True)
        circularity = (4.0 * np.pi * area_px) / (perimeter_px**2) if perimeter_px > 0 else 0.0

        # Chord of the circular tube available along the line of sight at this elevation.
        delta_y_m = abs(c_y_px - pipe_center_y_px) * pixel_size_m
        l_chord_m = 2.0 * np.sqrt(r_tube_m**2 - delta_y_m**2) if delta_y_m < r_tube_m else 0.0
        z_front_inner_m = wall_thickness_m + (r_tube_m - 0.5 * l_chord_m)
        z_rear_inner_m = wall_thickness_m + (r_tube_m + 0.5 * l_chord_m)

        record: dict[str, Any] = {
            "bubble_id": i,
            "centroid_x_m": c_x_px * pixel_size_m,
            "centroid_y_m": c_y_px * pixel_size_m,
            "area_px": area_px,
            "circularity": float(circularity),
            "aspect_ratio": float(aspect_ratio),
            "l_chord_m": float(l_chord_m),
            "z_rear_wall_m": float(z_rear_inner_m),
            "centroid_x_px": c_x_px,
            "centroid_y_px": c_y_px,
        }

        is_slug = (
            (area_px >= max_bubble_area_px)
            or (aspect_ratio > MAX_ASPECT_RATIO)
            or (circularity < min_circularity)
        )
        if is_slug:
            record.update(
                {
                    "depth_z_m": float(wall_thickness_m + r_tube_m),
                    "depth_fluid_z_m": float(0.5 * l_chord_m),
                    "diameter_d0_m": float(2.0 * r_t_px * pixel_size_m),
                    "blur_sigma_px": 0.0,
                    "sigma_tilde": 0.0,
                    "g_tilde": 0.0,
                    "near_wall_zone": False,
                    "is_reliable_dfd": False,
                    "dfd_status": "slug_excluded",
                }
            )
            results.append(record)
            continue

        edge_pixels = contours[0].squeeze()
        if edge_pixels.ndim != 2:
            continue

        ys = np.clip(edge_pixels[:, 1], 0, h - 1)
        xs = np.clip(edge_pixels[:, 0], 0, w - 1)
        edge_grads = grad_mag[ys, xs]

        # Trimmed mean rejects gradient spikes from glare points and caustics.
        if len(edge_grads) >= 8:
            p_low, p_high = np.percentile(edge_grads, [15, 85])
            valid_grads = edge_grads[(edge_grads >= p_low) & (edge_grads <= p_high)]
            mean_edge_grad = float(np.mean(valid_grads if len(valid_grads) > 0 else edge_grads))
        else:
            mean_edge_grad = float(np.mean(edge_grads))

        # Eq. 9: G~ = r_t * |dg_t / dr_t|  (px * 1/px -> dimensionless)
        g_tilde = r_t_px * mean_edge_grad
        # sigma~ = f2^-1(G~), rho~_t = f1(sigma~)
        s_tilde, rho_tilde = dfd_model.invert(g_tilde)

        is_near_wall = s_tilde < near_wall_sigma_threshold
        if is_near_wall and near_wall_sigma_threshold > 0:
            # Project regulariser (not in Rao et al.): smoothstep S(u) = 3u^2 - 2u^3 blends towards
            # the in-focus limit rho~_t = 0.5, sigma = 0 where the gradient estimate is noisy.
            u = max(0.0, min(1.0, s_tilde / near_wall_sigma_threshold))
            reg_factor = (3.0 * u**2) - (2.0 * u**3)
            rho_reg = 0.5 * (1.0 - reg_factor) + rho_tilde * reg_factor
            d0_px = r_t_px / rho_reg
            sigma_px = s_tilde * d0_px * reg_factor
        else:
            # d_0 = r_t / rho~_t and sigma = sigma~ * d_0
            d0_px = r_t_px / rho_tilde
            sigma_px = s_tilde * d0_px

        d0_m = d0_px * pixel_size_m
        # Eq. 5 inverted: |Delta z| [m] = sigma [px] / beta [px/m]
        delta_z_m = sigma_px / beta_eff_px_m

        if focal_plane_z_m is None:
            z_raw_m = delta_z_m
        else:
            z_raw_m = focal_plane_z_m + delta_z_m - z_front_inner_m

        depth_fluid_z_m = float(np.clip(z_raw_m, 0.0, l_chord_m))
        depth_z_m = float(z_front_inner_m + depth_fluid_z_m)

        if z_raw_m >= l_chord_m:
            dfd_status, is_reliable = "saturated_rear_wall", False
        elif s_tilde > max_sigma_threshold:
            dfd_status, is_reliable = "unreliable_defocus", False
        elif is_near_wall:
            dfd_status, is_reliable = "near_wall_regularized", True
        else:
            dfd_status, is_reliable = "valid", True

        record.update(
            {
                "depth_z_m": depth_z_m,
                "depth_fluid_z_m": depth_fluid_z_m,
                "diameter_d0_m": d0_m,
                "blur_sigma_px": sigma_px,
                "sigma_tilde": s_tilde,
                "g_tilde": g_tilde,
                "near_wall_zone": bool(is_near_wall),
                "is_reliable_dfd": is_reliable,
                "dfd_status": dfd_status,
            }
        )
        results.append(record)

    return results


# ---------------------------------------------------------------------------
# 4. Two-Panel Diagnostic Visualization
# ---------------------------------------------------------------------------


def render_dfd_diagnostic(
    raw_img: np.ndarray,
    binary_mask: np.ndarray,
    results: list[dict[str, Any]],
    output_path: Path,
    pipe_center_y_px: float,
    r_tube_px: float,
    pixel_size_m: float,
    wall_thickness_m: float = DEFAULT_WALL_THICKNESS_M,
    inner_diameter_m: float = DEFAULT_INNER_DIAMETER_M,
    min_label_area_px: float = 50.0,
    show_labels: bool = False,
):
    """
    Renders publication-grade 2-panel diagnostic figure:
      Panel (a): Front View (x-y) overlaying raw image with segmented bubbles color-coded by DFD status.
      Panel (b): Transverse Cross-Section (y-z) showing the circular tube profile, glass wall,
                 and the reconstructed out-of-plane depth of each bubble.
    """
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    fig, (ax_front, ax_side) = plt.subplots(
        1, 2, figsize=(11.5, 4.2), dpi=300, gridspec_kw={"width_ratios": [2.25, 1.0]}
    )

    status_colors = STATUS_COLORS

    # Soft translucent color tint on bubble silhouettes in front view
    def hex_to_rgb(hex_str: str) -> tuple[float, float, float]:
        hex_clean = hex_str.lstrip("#")
        return tuple(int(hex_clean[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

    status_colors_rgb = {k: hex_to_rgb(v) for k, v in status_colors.items()}

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    res_by_id = {r["bubble_id"]: r for r in results}

    if len(raw_img.shape) == 2:
        base_rgb = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2RGB).astype(float) / 255.0
    else:
        base_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB).astype(float) / 255.0

    overlay = base_rgb.copy()
    for b_id, res in res_by_id.items():
        st = res["dfd_status"]
        c_rgb = np.array(status_colors_rgb.get(st, (1.0, 1.0, 1.0)))
        b_mask = labels == b_id
        overlay[b_mask] = 0.72 * overlay[b_mask] + 0.28 * c_rgb

    ax_front.imshow(overlay, aspect="equal")

    h_img, w_img = raw_img.shape[:2]
    y_top_wall = pipe_center_y_px - r_tube_px
    y_bot_wall = pipe_center_y_px + r_tube_px
    ax_front.axhline(
        y_top_wall, color="#FFC107", ls="--", lw=1.2, alpha=0.85, label="Upper tube wall"
    )
    ax_front.axhline(
        y_bot_wall, color="#FFC107", ls="--", lw=1.2, alpha=0.85, label="Lower tube wall"
    )
    ax_front.axhline(
        pipe_center_y_px, color="#17A2B8", ls=":", lw=1.0, alpha=0.6, label="Tube centerline"
    )

    for b_id, res in res_by_id.items():
        st = res["dfd_status"]
        color = status_colors.get(st, "#FFFFFF")
        cx = res["centroid_x_px"]
        cy = res["centroid_y_px"]
        d0_mm = res["diameter_d0_m"] * 1e3
        z_mm = res["depth_fluid_z_m"] * 1e3

        b_mask = (labels == b_id).astype(np.uint8)
        cnts, _ = cv2.findContours(b_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            for cnt in cnts:
                pts = cnt.squeeze()
                if len(pts.shape) == 2:
                    ax_front.plot(pts[:, 0], pts[:, 1], color=color, lw=1.4)

        if show_labels:
            if st == "slug_excluded":
                lbl_txt = f"ID {b_id} (Slug)"
            elif st == "saturated_rear_wall":
                lbl_txt = f"ID {b_id}\nSAT: z={z_mm:.1f}mm"
            elif st == "near_wall_regularized":
                lbl_txt = f"ID {b_id}\nNear: z={z_mm:.2f}mm"
            else:
                lbl_txt = f"ID {b_id}\nz={z_mm:.1f}mm | d={d0_mm:.2f}mm"

            # Declutter text labels for small specks to avoid visual crowding
            if res["area_px"] >= min_label_area_px or st == "slug_excluded":
                ax_front.text(
                    cx,
                    cy,
                    lbl_txt,
                    ha="center",
                    va="center",
                    fontsize=6.8,
                    fontweight="bold",
                    color=color,
                    path_effects=[pe.withStroke(linewidth=2.0, foreground="#000000")],
                )

    ax_front.set_title(
        r"$\mathbf{(a)}$ Front Projection ($x-y$): DFD Reliability Classification",
        fontsize=9.5,
        color="#1B365D",
    )
    ax_front.set_xlabel("Axial coordinate $x$ [px]")
    ax_front.set_ylabel("Transverse coordinate $y$ [px]")
    ax_front.set_xlim(0, w_img)
    ax_front.set_ylim(h_img, 0)

    # Panel (b): Transverse Cross-Section (y-z)
    r_tube_mm = (inner_diameter_m / 2.0) * 1e3
    wall_mm = wall_thickness_m * 1e3
    d_tube_mm = inner_diameter_m * 1e3

    theta = np.linspace(0, 2 * np.pi, 200)
    z_center_mm = wall_mm + r_tube_mm
    y_center_mm = 0.0

    z_inner = z_center_mm + r_tube_mm * np.cos(theta)
    y_inner = y_center_mm + r_tube_mm * np.sin(theta)
    z_outer = z_center_mm + (r_tube_mm + wall_mm) * np.cos(theta)
    y_outer = y_center_mm + (r_tube_mm + wall_mm) * np.sin(theta)

    ax_side.plot(
        z_outer, y_outer, color="#888888", lw=1.2, ls="--", label="Outer wall (focus $z=0$)"
    )
    ax_side.plot(z_inner, y_inner, color="#1B365D", lw=1.8, label="Inner tube wall")
    ax_side.fill(
        np.append(z_outer, z_inner[::-1]),
        np.append(y_outer, y_inner[::-1]),
        color="#E0E0E0",
        alpha=0.5,
    )

    ax_side.axvline(0.0, color="#E91E63", ls=":", lw=1.3, label="Focal plane ($z_f=0$)")
    ax_side.axvline(wall_mm, color="#00BCD4", ls=":", lw=1.1, alpha=0.7)

    for b_id, res in res_by_id.items():
        st = res["dfd_status"]
        color = status_colors.get(st, "#FFFFFF")
        cy_px = res["centroid_y_px"]
        delta_y_mm = (cy_px - pipe_center_y_px) * (pixel_size_m * 1e3)
        y_elev_mm = -delta_y_mm

        # Bound elevation within circular tube profile
        y_elev_clamped_mm = float(np.clip(y_elev_mm, -r_tube_mm * 0.995, r_tube_mm * 0.995))
        z_chord_half_mm = float(np.sqrt(max(0.0, r_tube_mm**2 - y_elev_clamped_mm**2)))
        z_front_inner_mm = z_center_mm - z_chord_half_mm
        z_rear_inner_mm = z_center_mm + z_chord_half_mm

        # Fluid depth measured from front inner wall into fluid
        z_fluid_mm = float(res.get("depth_fluid_z_m", 0.0)) * 1e3
        d0_mm = float(res.get("diameter_d0_m", 0.0)) * 1e3
        r0_mm = max(0.35, d0_mm / 2.0)

        # Optical depth coordinate
        if st == "slug_excluded":
            z_bubble_mm = z_center_mm
        else:
            z_bubble_raw_mm = z_front_inner_mm + z_fluid_mm
            z_bubble_mm = float(
                np.clip(z_bubble_raw_mm, z_front_inner_mm + 0.15, z_rear_inner_mm - 0.15)
            )

        # Check distance from tube center (z_center_mm, 0.0)
        dist_from_tube_center = np.sqrt((z_bubble_mm - z_center_mm) ** 2 + y_elev_clamped_mm**2)
        max_allowed_center_dist = r_tube_mm - 0.25
        if dist_from_tube_center > max_allowed_center_dist:
            scale_center = max_allowed_center_dist / dist_from_tube_center
            z_bubble_mm = z_center_mm + (z_bubble_mm - z_center_mm) * scale_center
            y_elev_clamped_mm = y_elev_clamped_mm * scale_center
            dist_from_tube_center = max_allowed_center_dist

        # Compute exact radial clearance to the circular tube wall
        radial_clearance_mm = max(0.18, r_tube_mm - dist_from_tube_center)

        # Scale visual patch radius so it remains strictly inside tube bounds in all directions
        r0_plot_mm = min(r0_mm, max(0.18, radial_clearance_mm * 0.92))

        c_patch = Circle(
            (z_bubble_mm, y_elev_clamped_mm),
            r0_plot_mm,
            facecolor=color,
            edgecolor="#000000",
            lw=0.8,
            alpha=0.85,
        )
        ax_side.add_patch(c_patch)

        if show_labels and (res["area_px"] >= min_label_area_px or st == "slug_excluded"):
            ax_side.text(
                z_bubble_mm,
                y_elev_clamped_mm,
                str(b_id),
                ha="center",
                va="center",
                fontsize=6.5,
                fontweight="bold",
                color="#FFFFFF",
                path_effects=[pe.withStroke(linewidth=1.8, foreground="#000000")],
            )

    ax_side.set_title(r"$\mathbf{(b)}$ Transverse Profile ($z-y$)", fontsize=9.5, color="#1B365D")
    ax_side.set_xlabel("Optical depth $z$ [mm]")
    ax_side.set_ylabel("Transverse elevation $y$ [mm]")
    ax_side.set_xlim(-1.0, wall_mm * 2 + d_tube_mm + 1.0)
    ax_side.set_ylim(-r_tube_mm - wall_mm - 1.0, r_tube_mm + wall_mm + 1.0)
    ax_side.grid(True, ls=":", alpha=0.4)
    ax_side.set_box_aspect(1)
    ax_side.set_aspect("equal", adjustable="box")

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#00E676",
            markeredgecolor="k",
            markersize=7,
            label=r"Valid ($\tilde{\sigma} \leq 0.35$)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#00B0FF",
            markeredgecolor="k",
            markersize=7,
            label=r"Near-wall ($\tilde{\sigma} < 0.05$)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#FF5252",
            markeredgecolor="k",
            markersize=7,
            label=r"Saturated Rear Wall",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#FF9100",
            markeredgecolor="k",
            markersize=7,
            label=r"Unreliable Blur ($\tilde{\sigma} > 0.35$)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#9E9E9E",
            markeredgecolor="k",
            markersize=7,
            label=r"Slug (Excluded)",
        ),
    ]
    ax_front.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=7.2,
        framealpha=0.95,
        facecolor="#FFFFFF",
        edgecolor="#B0BEC5",
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def render_3d_dfd_spatial(
    results: list[dict[str, Any]],
    output_png: Path,
    pipe_radius_m: float,
    pipe_length_m: float,
    pixel_size_m: float,
    pipe_center_y_px: float,
    title: str = "3D Depth from Defocus (DFD) Spatial Reconstruction",
):
    """
    Renders 3D perspective visualization of the cylindrical tube conduit showing
    all bubbles positioned at their actual 3D physical coordinates (X, Y, Z_depth)
    measured by single-camera Depth from Defocus.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(12.0, 7.0), facecolor="#ECECEC", dpi=200)
    ax = fig.add_subplot(111, projection="3d", facecolor="white")
    ax.set_box_aspect((2.8, 1.0, 1.0))

    # White panes with soft gray grid
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.xaxis._axinfo["grid"]["color"] = (0.88, 0.88, 0.88, 1.0)
    ax.yaxis._axinfo["grid"]["color"] = (0.88, 0.88, 0.88, 1.0)
    ax.zaxis._axinfo["grid"]["color"] = (0.88, 0.88, 0.88, 1.0)

    ls = LightSource(azdeg=315, altdeg=45)

    # 1. Wireframe cylindrical tube conduit (Deep Navy Blue)
    N_rings = 45
    N_lines = 16
    theta = np.linspace(0, 2 * np.pi, 60)
    for x_pos in np.linspace(0, pipe_length_m, N_rings):
        y_ring = pipe_radius_m * np.sin(theta)  # optical depth (transverse)
        z_ring = pipe_radius_m * np.cos(theta)  # vertical elevation (buoyancy)
        ax.plot(
            x_pos * np.ones_like(theta),
            y_ring,
            z_ring,
            color="#000088",
            linewidth=0.65,
            alpha=0.6,
        )

    for th in np.linspace(0, 2 * np.pi, N_lines, endpoint=False):
        y_line = pipe_radius_m * np.sin(th)
        z_line = pipe_radius_m * np.cos(th)
        ax.plot(
            [0.0, pipe_length_m],
            [y_line, y_line],
            [z_line, z_line],
            color="#000088",
            linewidth=0.65,
            alpha=0.6,
        )

    status_colors = STATUS_COLORS

    # Sphere mesh template for bubbles
    u = np.linspace(0, 2 * np.pi, 16)
    v = np.linspace(0, np.pi, 10)
    xs_unit = np.outer(np.cos(u), np.sin(v))
    ys_unit = np.outer(np.sin(u), np.sin(v))
    zs_unit = np.outer(np.ones_like(u), np.cos(v))

    for res in results:
        st = res["dfd_status"]
        if st == "slug_excluded":
            # Slugs are continuous elongated vapor structures, rendered separately in 3D volumetric model
            continue

        cx_m = res["centroid_x_m"]
        cy_px = res["centroid_y_px"]
        # Vertical elevation (buoyancy axis: top > 0, bottom < 0)
        cz_elev_m = -(cy_px - pipe_center_y_px) * pixel_size_m
        z_chord_half_m = np.sqrt(max(0.0, pipe_radius_m**2 - cz_elev_m**2))
        # Optical depth (transverse axis: front wall at local chord elevation is -z_chord_half_m)
        z_fluid_m = res["depth_fluid_z_m"]

        d0_m = res["diameter_d0_m"]
        r0_m = max(0.00035, d0_m / 2.0)
        if z_chord_half_m > 0.00005:
            r0_plot_m = min(r0_m, max(0.00015, z_chord_half_m * 0.85))
            cy_depth_raw_m = -z_chord_half_m + z_fluid_m
            cy_depth_m = np.clip(
                cy_depth_raw_m, -z_chord_half_m + r0_plot_m, z_chord_half_m - r0_plot_m
            )
        else:
            cy_depth_m = 0.0
            r0_plot_m = 0.00015

        color = status_colors.get(st, "#00E676")

        ax.plot_surface(
            cx_m + r0_plot_m * xs_unit,
            cy_depth_m + r0_plot_m * ys_unit,
            cz_elev_m + r0_plot_m * zs_unit,
            color=color,
            shade=True,
            lightsource=ls,
            alpha=0.92,
        )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=14, color="#1B365D")
    ax.set_xlabel("Axial coordinate X (flow) [m]", fontsize=10, labelpad=8)
    ax.set_ylabel("Optical depth Y (line-of-sight) [m]", fontsize=10, labelpad=8)
    ax.set_zlabel("Transverse elevation Z (vertical) [m]", fontsize=10, labelpad=8)

    ax.set_xlim(0, pipe_length_m)
    ax.set_ylim(-pipe_radius_m * 1.15, pipe_radius_m * 1.15)
    ax.set_zlim(-pipe_radius_m * 1.15, pipe_radius_m * 1.15)
    ax.view_init(elev=22, azim=-55)

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#00E676",
            markeredgecolor="k",
            markersize=7,
            label=r"Valid ($\tilde{\sigma} \leq 0.35$)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#00B0FF",
            markeredgecolor="k",
            markersize=7,
            label=r"Near-wall ($\tilde{\sigma} < 0.05$)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#FF5252",
            markeredgecolor="k",
            markersize=7,
            label=r"Saturated Rear Wall",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#9E9E9E",
            markeredgecolor="k",
            markersize=7,
            label=r"Slug (Excluded)",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7.5, framealpha=0.9)

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close()


def render_dfd_interactive_html(
    results: list[dict[str, Any]],
    output_html_path: Path,
    pipe_radius_m: float,
    pipe_length_m: float,
    pixel_size_m: float,
    pipe_center_y_px: float,
    vol_recon: dict[str, Any] | None = None,
    beta_eff: float = DEFAULT_BETA_EFF_PX_PER_M,
    frame_name: str = "",
) -> Path:
    """
    Renders an interactive 3D WebGL HTML visualization of single-camera Depth from Defocus (DFD)
    reconstruction using Three.js and OrbitControls. Allows dynamic 3D rotation, depth slicing,
    camera view presets (front, cross-section, top, perspective), and color-coding by depth/status.
    """
    output_html_path = Path(output_html_path)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Extract large deformed slugs from dual-regime reconstruction
    slugs = []
    if vol_recon and "bubbles" in vol_recon:
        for b in vol_recon["bubbles"]:
            if b.get("regime") == "large_slug":
                try:
                    from visualization.render_3d_flow import extract_slug_mesh_data

                    bw_px = b["bbox_px"][2]
                    bh_px = b["bbox_px"][3]
                    step = max(2, int(round(max(bw_px, bh_px) / 100.0)))
                    s_mesh = extract_slug_mesh_data(
                        b["blob_mask"],
                        b["depth_map_m"],
                        pixel_size_m=pixel_size_m,
                        y_center_px=pipe_center_y_px,
                        step=step,
                    )
                    if s_mesh["vertices"] and s_mesh["indices"]:
                        slugs.append(s_mesh)
                except Exception as e:
                    print(f"  [WARN] Failed to extract slug mesh: {e}")

    # 2. Extract small bubble entities with DFD 3D coordinates
    bubbles = []
    for r in results:
        if r.get("dfd_status") == "slug_excluded":
            continue

        cx_m = float(r["centroid_x_m"])
        cy_px = float(r["centroid_y_px"])
        # Vertical elevation (buoyancy axis: top > 0, bottom < 0)
        cy_m = float(-(cy_px - pipe_center_y_px) * pixel_size_m)
        z_chord_half_m = float(np.sqrt(max(0.0, pipe_radius_m**2 - cy_m**2)))
        # Fluid depth measured from front inner wall into fluid
        z_fluid_m = float(r["depth_fluid_z_m"])
        d0_m = float(r["diameter_d0_m"])
        r0_m = float(max(0.00035, d0_m / 2.0))

        # Three.js depth axis: front inner wall at local chord elevation is +z_chord_half_m
        raw_z = z_chord_half_m - z_fluid_m
        cz_m = float(np.clip(raw_z, -z_chord_half_m + r0_m, z_chord_half_m - r0_m))

        bubbles.append(
            {
                "id": int(r["bubble_id"]),
                "x": round(cx_m, 6),
                "y": round(cy_m, 6),
                "z": round(cz_m, 6),
                "z_chord_half_m": round(z_chord_half_m, 6),
                "depth_fluid_z_m": round(z_fluid_m, 6),
                "depth_fluid_z_mm": round(z_fluid_m * 1e3, 3),
                "depth_fluid_z_um": round(z_fluid_m * 1e6, 1),
                "diameter_d0_m": round(d0_m, 6),
                "diameter_d0_mm": round(d0_m * 1e3, 3),
                "radius_m": round(r0_m, 6),
                "blur_sigma_px": round(float(r.get("blur_sigma_px", 0.0)), 2),
                "sigma_tilde": round(float(r.get("sigma_tilde", 0.0)), 4),
                "dfd_status": str(r.get("dfd_status", "valid")),
                "area_px": int(r.get("area_px", 0)),
                "circularity": round(float(r.get("circularity", 0.0)), 3),
            }
        )

    # Summary statistics
    n_valid = sum(1 for b in bubbles if b["dfd_status"] == "valid")
    n_near = sum(1 for b in bubbles if b["dfd_status"] == "near_wall_regularized")
    depths_mm = [b["depth_fluid_z_mm"] for b in bubbles]
    mean_depth_mm = round(float(np.mean(depths_mm)), 3) if depths_mm else 0.0
    min_depth_mm = round(float(np.min(depths_mm)), 3) if depths_mm else 0.0
    max_depth_mm = round(float(np.max(depths_mm)), 3) if depths_mm else 0.0
    d0_vals_mm = [b["diameter_d0_mm"] for b in bubbles]
    mean_d0_mm = round(float(np.mean(d0_vals_mm)), 3) if d0_vals_mm else 0.0
    void_fraction_pct = (
        round(float(vol_recon.get("void_fraction_alpha", 0.0) * 100.0), 3) if vol_recon else 0.0
    )

    bubbles_json = json.dumps(bubbles)
    slugs_json = json.dumps(slugs)

    # Physical pipe dimensions in mm for HTML display
    d_pipe_mm = round(pipe_radius_m * 2.0 * 1e3, 2)
    l_pipe_mm = round(pipe_length_m * 1e3, 2)
    r_pipe_mm = round(pipe_radius_m * 1e3, 2)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D DFD Reconstruction - {frame_name or "Frame"}</title>
    <style>
        :root {{
            --bg-color: #0b0f19;
            --panel-bg: rgba(15, 23, 42, 0.85);
            --panel-border: rgba(56, 189, 248, 0.25);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #38bdf8;
            --accent-glow: rgba(56, 189, 248, 0.4);
            --btn-bg: rgba(255, 255, 255, 0.06);
            --btn-border: rgba(255, 255, 255, 0.12);
            --btn-hover: rgba(56, 189, 248, 0.15);
            --divider: rgba(255, 255, 255, 0.08);
            --shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
        }}
        body.theme-light {{
            --bg-color: #f8fafc;
            --panel-bg: rgba(255, 255, 255, 0.92);
            --panel-border: rgba(203, 213, 225, 0.8);
            --text-main: #0f172a;
            --text-muted: #64748b;
            --accent: #0284c7;
            --accent-glow: rgba(2, 132, 199, 0.25);
            --btn-bg: rgba(241, 245, 249, 0.9);
            --btn-border: rgba(203, 213, 225, 0.9);
            --btn-hover: rgba(2, 132, 199, 0.12);
            --divider: rgba(0, 0, 0, 0.08);
            --shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        }}
        body {{
            margin: 0;
            padding: 0;
            overflow: hidden;
            background-color: var(--bg-color);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            color: var(--text-main);
            user-select: none;
            transition: background-color 0.2s ease, color 0.2s ease;
        }}
        #canvas-container {{
            width: 100vw;
            height: 100vh;
            position: absolute;
            top: 0;
            left: 0;
            z-index: 1;
        }}
        /* Header Card */
        #header-bar {{
            position: absolute;
            top: 14px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 10;
            background: var(--panel-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--panel-border);
            padding: 8px 24px;
            border-radius: 9999px;
            display: flex;
            align-items: center;
            gap: 16px;
            box-shadow: var(--shadow);
            pointer-events: auto;
            transition: background 0.2s ease, border-color 0.2s ease;
        }}
        #header-bar h1 {{
            margin: 0;
            font-size: 15px;
            font-weight: 700;
            letter-spacing: -0.01em;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .badge {{
            background: rgba(56, 189, 248, 0.15);
            color: var(--accent);
            border: 1px solid rgba(56, 189, 248, 0.35);
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
        }}
        .badge-alpha {{
            background: rgba(16, 185, 129, 0.15);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.35);
        }}

        /* Control Panel (Top Right) */
        #control-panel {{
            position: absolute;
            top: 16px;
            right: 16px;
            z-index: 10;
            width: 290px;
            background: var(--panel-bg);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            padding: 14px;
            box-shadow: var(--shadow);
            transition: background 0.2s ease, border-color 0.2s ease;
        }}
        .panel-section {{
            margin-bottom: 12px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--divider);
        }}
        .panel-section:last-child {{
            margin-bottom: 0;
            padding-bottom: 0;
            border-bottom: none;
        }}
        .section-title {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-muted);
            margin-bottom: 8px;
            font-weight: 700;
        }}
        .btn-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 6px;
        }}
        .btn-mode {{
            background: var(--btn-bg);
            border: 1px solid var(--btn-border);
            color: var(--text-main);
            padding: 6px 8px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s ease;
            text-align: center;
        }}
        .btn-mode:hover {{
            background: var(--btn-hover);
            border-color: var(--accent);
            color: var(--accent);
        }}
        .btn-mode.active {{
            background: var(--accent);
            border-color: var(--accent);
            color: #ffffff;
            font-weight: 700;
            box-shadow: 0 0 12px var(--accent-glow);
        }}
        .slider-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: 6px;
            font-size: 12px;
        }}
        .slider-row input[type="range"] {{
            width: 140px;
            accent-color: var(--accent);
            cursor: pointer;
        }}
        .toggle-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 12px;
            margin-top: 6px;
        }}

        /* HUD Info Card (Bottom Left) */
        #hud-card {{
            position: absolute;
            bottom: 16px;
            left: 16px;
            z-index: 10;
            background: var(--panel-bg);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            padding: 14px 18px;
            font-size: 12px;
            line-height: 1.55;
            box-shadow: var(--shadow);
            max-width: 320px;
            transition: background 0.2s ease, border-color 0.2s ease;
        }}
        #hud-card .hud-header {{
            font-size: 13px;
            font-weight: 700;
            color: var(--accent);
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .metric-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 2px;
        }}
        .metric-row .val {{
            font-weight: 600;
            color: var(--text-main);
        }}

        /* Depth Colorbar (Bottom Right) */
        #colorbar-card {{
            position: absolute;
            bottom: 16px;
            right: 16px;
            z-index: 10;
            background: var(--panel-bg);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            padding: 12px 16px;
            box-shadow: var(--shadow);
            display: flex;
            flex-direction: column;
            align-items: center;
            transition: background 0.2s ease, border-color 0.2s ease;
        }}
        #colorbar-title {{
            font-size: 11px;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-bottom: 6px;
        }}
        #colorbar-grad {{
            width: 180px;
            height: 12px;
            border-radius: 4px;
            background: linear-gradient(to right, #00e5ff, #00e676, #ffeb3b, #ff5722);
            border: 1px solid rgba(125, 125, 125, 0.3);
        }}
        .colorbar-labels {{
            width: 180px;
            display: flex;
            justify-content: space-between;
            font-size: 10px;
            color: var(--text-muted);
            margin-top: 4px;
            font-weight: 600;
        }}

        /* Hover Tooltip / Inspector */
        #tooltip-card {{
            position: absolute;
            z-index: 50;
            background: var(--panel-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--accent);
            border-radius: 10px;
            padding: 10px 14px;
            font-size: 12px;
            line-height: 1.5;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.25);
            pointer-events: none;
            display: none;
            min-width: 180px;
            color: var(--text-main);
        }}
        #tooltip-card .bubble-id {{
            font-size: 13px;
            font-weight: 700;
            color: var(--accent);
            margin-bottom: 4px;
            border-bottom: 1px solid var(--divider);
            padding-bottom: 2px;
        }}

        /* Orientation Helper Pill */
        #view-pill {{
            position: absolute;
            bottom: 16px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 10;
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            padding: 4px 14px;
            border-radius: 9999px;
            font-size: 11px;
            color: var(--text-muted);
            letter-spacing: 0.02em;
            box-shadow: var(--shadow);
        }}
    </style>
    <!-- Three.js + OrbitControls (Local offline first, CDN fallback) -->
    <script src="lib/three.min.js"></script>
    <script src="lib/OrbitControls.js"></script>
    <script>
        if (typeof THREE === 'undefined') {{
            document.write('<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"><\\/script>');
            document.write('<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"><\\/script>');
        }}
    </script>
</head>
<body>
    <div id="canvas-container"></div>

    <!-- Header -->
    <div id="header-bar">
        <h1>3D Depth from Defocus (DFD) Reconstruction</h1>
        <span class="badge">{frame_name or "frame_0071"}</span>
        <span class="badge badge-alpha">Void Fraction α: {void_fraction_pct:.3f}%</span>
    </div>

    <!-- Top Right Controls -->
    <div id="control-panel">
        <div class="panel-section">
            <div class="section-title">Color Theme</div>
            <div class="btn-grid">
                <button class="btn-mode active" id="btn-theme-dark">🌙 Dark Mode</button>
                <button class="btn-mode" id="btn-theme-light">☀️ White Mode</button>
            </div>
        </div>

        <div class="panel-section">
            <div class="section-title">Camera Views</div>
            <div class="btn-grid">
                <button class="btn-mode active" id="btn-view-persp">🎥 3D Orbit</button>
                <button class="btn-mode" id="btn-view-front">📐 Front (X-Y)</button>
                <button class="btn-mode" id="btn-view-trans">🔍 Cross-Section</button>
                <button class="btn-mode" id="btn-view-top">⬆️ Top (X-Z)</button>
            </div>
        </div>

        <div class="panel-section">
            <div class="section-title">Color Mode</div>
            <div class="btn-grid">
                <button class="btn-mode active" id="btn-col-depth">🌈 Depth Colormap</button>
                <button class="btn-mode" id="btn-col-status">🟢 DFD Status</button>
            </div>
        </div>

        <div class="panel-section">
            <div class="section-title">Depth Separation Scale</div>
            <div class="slider-row">
                <span>Depth Z Scale:</span>
                <input type="range" id="slider-depth-scale" min="1.0" max="4.0" step="0.1" value="1.0">
                <span id="depth-scale-val" style="font-weight:700; width:30px; text-align:right;">1.0×</span>
            </div>
            <small style="color:var(--text-muted); font-size:10px; display:block; margin-top:3px;">1.0× = True SI physical dimensions</small>
        </div>

        <div class="panel-section">
            <div class="section-title">Conduit Elements</div>
            <div class="toggle-row">
                <span>Tube Wireframe:</span>
                <input type="checkbox" id="toggle-wireframe" checked style="accent-color:var(--accent);">
            </div>
            <div class="toggle-row">
                <span>Wall Reference Rings:</span>
                <input type="checkbox" id="toggle-rings" checked style="accent-color:var(--accent);">
            </div>
            <div class="toggle-row">
                <span>Vapor Slug Mesh:</span>
                <input type="checkbox" id="toggle-slug" checked style="accent-color:var(--accent);">
            </div>
        </div>
    </div>

    <!-- Bottom Left HUD -->
    <div id="hud-card">
        <div class="hud-header">
            <span>Experimental Setup</span>
            <span style="font-size:11px; color:var(--accent);">SI Metrology</span>
        </div>
        <div class="metric-row">
            <span style="color:var(--text-muted);">Tube Diameter:</span>
            <span class="val">{d_pipe_mm} mm (Radius {r_pipe_mm} mm)</span>
        </div>
        <div class="metric-row">
            <span style="color:var(--text-muted);">Conduit Length:</span>
            <span class="val">{l_pipe_mm} mm</span>
        </div>
        <div class="metric-row">
            <span style="color:var(--text-muted);">Optical Scale:</span>
            <span class="val">{pixel_size_m * 1e6:.2f} µm/px ({pixel_size_m:.4e} m/px)</span>
        </div>
        <div class="metric-row">
            <span style="color:var(--text-muted);">DFD Calibration β:</span>
            <span class="val">{beta_eff:.1f} px/m</span>
        </div>
        <div class="metric-row" style="margin-top:6px; padding-top:4px; border-top:1px solid var(--divider);">
            <span style="color:var(--text-muted);">Analyzed Bubbles:</span>
            <span class="val">{len(bubbles)} ({n_valid} Valid, {n_near} Near-wall)</span>
        </div>
        <div class="metric-row">
            <span style="color:var(--text-muted);">Mean Fluid Depth:</span>
            <span class="val" style="color:var(--accent);">{mean_depth_mm} mm ({mean_depth_mm * 1e3:.0f} µm)</span>
        </div>
        <div class="metric-row">
            <span style="color:var(--text-muted);">Depth Range:</span>
            <span class="val">{min_depth_mm} – {max_depth_mm} mm</span>
        </div>
        <div class="metric-row">
            <span style="color:var(--text-muted);">Mean Diameter d₀:</span>
            <span class="val">{mean_d0_mm} mm</span>
        </div>
    </div>

    <!-- Bottom Right Colorbar -->
    <div id="colorbar-card">
        <div id="colorbar-title">Fluid Depth z<sub>fluid</sub> [mm]</div>
        <div id="colorbar-grad"></div>
        <div class="colorbar-labels">
            <span>0.0 (Wall)</span>
            <span>0.3</span>
            <span>0.6</span>
            <span>0.9</span>
            <span>1.2+ mm</span>
        </div>
    </div>

    <!-- Tooltip -->
    <div id="tooltip-card">
        <div class="bubble-id" id="tt-id">Bubble #1</div>
        <div class="metric-row"><span style="color:var(--text-muted);">Fluid Depth:</span><span class="val" id="tt-depth">--</span></div>
        <div class="metric-row"><span style="color:var(--text-muted);">Diameter d₀:</span><span class="val" id="tt-d0">--</span></div>
        <div class="metric-row"><span style="color:var(--text-muted);">Blur σ̃:</span><span class="val" id="tt-blur">--</span></div>
        <div class="metric-row"><span style="color:var(--text-muted);">DFD Status:</span><span class="val" id="tt-status">--</span></div>
        <div class="metric-row"><span style="color:var(--text-muted);">Pos (X, Y, Z):</span><span class="val" id="tt-pos" style="font-size:11px;">--</span></div>
    </div>

    <!-- Instructions Pill -->
    <div id="view-pill">Left-click: Orbit | Right-click: Pan | Scroll: Zoom | Hover bubble: Inspect</div>

    <script>
        // Data Payload
        const pipeRadius = {pipe_radius_m};
        const pipeLength = {pipe_length_m};
        const bubblesData = {bubbles_json};
        const slugsData = {slugs_json};

        const container = document.getElementById('canvas-container');
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0b0f19);

        const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.0005, 50);
        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true, powerPreference: "high-performance" }});
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.15;
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.06;
        controls.target.set(pipeLength * 0.5, 0, 0);

        // Initial Camera Position (Perspective)
        function setPerspectiveView() {{
            camera.position.set(pipeLength * 0.5, pipeRadius * 1.5, pipeRadius * 3.6);
            controls.target.set(pipeLength * 0.5, 0, 0);
            controls.update();
        }}
        setPerspectiveView();

        // 3D Lighting Setup
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.65);
        scene.add(ambientLight);

        const lightFront = new THREE.DirectionalLight(0xffffff, 0.85);
        lightFront.position.set(pipeLength * 0.5, pipeRadius * 2, pipeRadius * 4);
        scene.add(lightFront);

        const lightTop = new THREE.DirectionalLight(0xffffff, 0.45);
        lightTop.position.set(pipeLength * 0.5, pipeRadius * 4, 0);
        scene.add(lightTop);

        const lightBack = new THREE.DirectionalLight(0x38bdf8, 0.4);
        lightBack.position.set(pipeLength * 0.5, -pipeRadius * 2, -pipeRadius * 4);
        scene.add(lightBack);

        // 1. Cylindrical Tube Conduit
        const tubeGroup = new THREE.Group();
        const cylGeom = new THREE.CylinderGeometry(pipeRadius, pipeRadius, pipeLength, 64, 32, true);
        cylGeom.rotateZ(Math.PI / 2);
        cylGeom.translate(pipeLength / 2, 0, 0);

        // Wireframe
        const wireframeGeom = new THREE.WireframeGeometry(cylGeom);
        const wireframeMat = new THREE.LineBasicMaterial({{
            color: 0x1e3a8a,
            transparent: true,
            opacity: 0.55
        }});
        const wireframeMesh = new THREE.LineSegments(wireframeGeom, wireframeMat);
        tubeGroup.add(wireframeMesh);

        // Semi-transparent glass inner shell
        const glassMat = new THREE.MeshPhysicalMaterial({{
            color: 0x0f172a,
            roughness: 0.1,
            metalness: 0.1,
            transmission: 0.75,
            transparent: true,
            opacity: 0.18,
            side: THREE.DoubleSide
        }});
        const glassMesh = new THREE.Mesh(cylGeom, glassMat);
        tubeGroup.add(glassMesh);

        // 2. Reference Circular Rings along the tube
        const ringsGroup = new THREE.Group();
        const nRings = 7;
        const ringLines = [];
        for (let i = 0; i <= nRings; i++) {{
            const xRing = (pipeLength / nRings) * i;
            const ringGeom = new THREE.BufferGeometry();
            const ringPts = [];
            for (let th = 0; th <= Math.PI * 2 + 0.1; th += 0.1) {{
                ringPts.push(xRing, Math.cos(th) * pipeRadius, Math.sin(th) * pipeRadius);
            }}
            ringGeom.setAttribute('position', new THREE.Float32BufferAttribute(ringPts, 3));
            const isCap = (i === 0 || i === nRings);
            const ringLine = new THREE.Line(ringGeom, new THREE.LineBasicMaterial({{
                color: isCap ? 0x38bdf8 : 0x1e293b,
                transparent: true,
                opacity: 0.75
            }}));
            ringLine.userData = {{ isCap: isCap }};
            ringsGroup.add(ringLine);
            ringLines.push(ringLine);
        }}
        tubeGroup.add(ringsGroup);
        scene.add(tubeGroup);

        // Theme switching logic (Dark vs White Mode)
        let currentTheme = 'dark';
        const btnThemeDark = document.getElementById('btn-theme-dark');
        const btnThemeLight = document.getElementById('btn-theme-light');

        function setTheme(theme) {{
            currentTheme = theme;
            if (theme === 'light') {{
                document.body.classList.add('theme-light');
                btnThemeLight.classList.add('active');
                btnThemeDark.classList.remove('active');

                // White / Light grey background
                scene.background = new THREE.Color(0xf8fafc);

                // Adjust lighting for clean white mode
                ambientLight.intensity = 0.85;
                lightFront.intensity = 0.95;
                lightBack.color.setHex(0x94a3b8);
                lightBack.intensity = 0.45;

                // Tube materials adjusted for white contrast
                wireframeMat.color.setHex(0x94a3b8);
                wireframeMat.opacity = 0.65;
                glassMat.color.setHex(0xe2e8f0);
                glassMat.opacity = 0.15;

                // Rings in light theme
                ringLines.forEach(l => {{
                    if (l.userData.isCap) {{
                        l.material.color.setHex(0x0284c7);
                        l.material.opacity = 0.9;
                    }} else {{
                        l.material.color.setHex(0xcbd5e1);
                        l.material.opacity = 0.85;
                    }}
                }});
            }} else {{
                document.body.classList.remove('theme-light');
                btnThemeDark.classList.add('active');
                btnThemeLight.classList.remove('active');

                // Dark scene background
                scene.background = new THREE.Color(0x0b0f19);

                // Dark mode lighting
                ambientLight.intensity = 0.65;
                lightFront.intensity = 0.85;
                lightBack.color.setHex(0x38bdf8);
                lightBack.intensity = 0.4;

                // Tube materials for dark contrast
                wireframeMat.color.setHex(0x1e3a8a);
                wireframeMat.opacity = 0.55;
                glassMat.color.setHex(0x0f172a);
                glassMat.opacity = 0.18;

                // Rings in dark theme
                ringLines.forEach(l => {{
                    if (l.userData.isCap) {{
                        l.material.color.setHex(0x38bdf8);
                        l.material.opacity = 0.75;
                    }} else {{
                        l.material.color.setHex(0x1e293b);
                        l.material.opacity = 0.75;
                    }}
                }});
            }}
        }}

        btnThemeDark.addEventListener('click', () => setTheme('dark'));
        btnThemeLight.addEventListener('click', () => setTheme('light'));

        // 3. Colormap Helper (Turbo polynomial approximation)
        function getDepthColor(z_mm) {{
            // Scale from 0.0 mm (front wall) to 1.2 mm
            let t = Math.max(0, Math.min(1, z_mm / 1.2));
            const r = Math.max(0, Math.min(1, 0.1357 + t * (4.5153 + t * (-8.5627 + t * 4.9086))));
            const g = Math.max(0, Math.min(1, 0.0914 + t * (2.1942 + t * (4.8429 + t * (-14.1850 + t * 8.0519)))));
            const b = Math.max(0, Math.min(1, 0.1067 + t * (12.5919 + t * (-60.1802 + t * (109.0745 + t * (-88.5022 + t * 27.0581))))));
            return new THREE.Color(r, g, b);
        }}

        function getStatusColor(status) {{
            if (status === 'valid') return new THREE.Color(0x00E676);            // Emerald Green
            if (status === 'near_wall_regularized') return new THREE.Color(0x00B0FF); // Cyan
            if (status === 'saturated_rear_wall') return new THREE.Color(0xFF5252);   // Coral
            return new THREE.Color(0x9E9E9E);                                    // Grey
        }}

        // 4. Render Small Bubbles (Ellipsoid Spheres)
        const bubblesGroup = new THREE.Group();
        const bubbleMeshes = [];
        let currentMode = 'depth'; // 'depth' or 'status'
        let depthExaggeration = 1.0;

        bubblesData.forEach(b => {{
            const geom = new THREE.SphereGeometry(b.radius_m, 22, 16);
            const depthCol = getDepthColor(b.depth_fluid_z_mm);
            const statusCol = getStatusColor(b.dfd_status);

            const mat = new THREE.MeshStandardMaterial({{
                color: depthCol,
                roughness: 0.25,
                metalness: 0.15,
                emissive: 0x000000,
                emissiveIntensity: 0.2
            }});

            const mesh = new THREE.Mesh(geom, mat);
            mesh.position.set(b.x, b.y, b.z);
            mesh.userData = {{
                ...b,
                depthColor: depthCol,
                statusColor: statusCol,
                originalZ: b.z,
                material: mat
            }};

            bubbleMeshes.push(mesh);
            bubblesGroup.add(mesh);
        }});
        scene.add(bubblesGroup);

        // 5. Render Large Slugs (Watertight Meshes)
        const slugGroup = new THREE.Group();
        const slugMat = new THREE.MeshPhongMaterial({{
            color: 0x881337,
            emissive: 0x1f030a,
            specular: 0x4c0519,
            shininess: 65,
            side: THREE.DoubleSide
        }});

        slugsData.forEach(slug => {{
            if (slug.vertices && slug.vertices.length > 0 && slug.indices && slug.indices.length > 0) {{
                const geom = new THREE.BufferGeometry();
                geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(slug.vertices), 3));
                geom.setIndex(slug.indices);
                geom.computeVertexNormals();
                const sMesh = new THREE.Mesh(geom, slugMat);
                slugGroup.add(sMesh);
            }}
        }});
        scene.add(slugGroup);

        // 6. Raycasting & Tooltip Inspector
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2(-999, -999);
        const tooltip = document.getElementById('tooltip-card');
        let hoveredMesh = null;

        function updateHover() {{
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObjects(bubbleMeshes);

            if (intersects.length > 0) {{
                const hit = intersects[0].object;
                if (hoveredMesh !== hit) {{
                    if (hoveredMesh) {{
                        hoveredMesh.material.emissive.setHex(0x000000);
                        hoveredMesh.material.emissiveIntensity = 0.2;
                    }}
                    hoveredMesh = hit;
                    hoveredMesh.material.emissive.setHex(0xffffff);
                    hoveredMesh.material.emissiveIntensity = 0.55;
                }}

                const d = hit.userData;
                document.getElementById('tt-id').innerText = `Bubble #${{d.id}} (${{d.dfd_status}})`;
                document.getElementById('tt-depth').innerText = `${{d.depth_fluid_z_mm}} mm (${{d.depth_fluid_z_um}} µm)`;
                document.getElementById('tt-d0').innerText = `${{d.diameter_d0_mm}} mm`;
                document.getElementById('tt-blur').innerText = `${{d.sigma_tilde}} (${{d.blur_sigma_px}} px)`;
                document.getElementById('tt-status').innerText = d.dfd_status;
                document.getElementById('tt-pos').innerText = `${{(d.x * 1e3).toFixed(1)}}, ${{(d.y * 1e3).toFixed(1)}}, ${{(d.z * 1e3).toFixed(1)}} mm`;

                tooltip.style.display = 'block';
            }} else {{
                if (hoveredMesh) {{
                    hoveredMesh.material.emissive.setHex(0x000000);
                    hoveredMesh.material.emissiveIntensity = 0.2;
                    hoveredMesh = null;
                }}
                tooltip.style.display = 'none';
            }}
        }}

        window.addEventListener('pointermove', (e) => {{
            mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
            tooltip.style.left = `${{e.clientX + 14}}px`;
            tooltip.style.top = `${{e.clientY + 14}}px`;
            updateHover();
        }});

        // 7. Interactive Controls & View Presets
        const btnPersp = document.getElementById('btn-view-persp');
        const btnFront = document.getElementById('btn-view-front');
        const btnTrans = document.getElementById('btn-view-trans');
        const btnTop = document.getElementById('btn-view-top');
        const viewBtns = [btnPersp, btnFront, btnTrans, btnTop];

        function setActiveViewBtn(activeBtn) {{
            viewBtns.forEach(b => b.classList.remove('active'));
            activeBtn.classList.add('active');
        }}

        btnPersp.addEventListener('click', () => {{
            setActiveViewBtn(btnPersp);
            setPerspectiveView();
        }});

        btnFront.addEventListener('click', () => {{
            setActiveViewBtn(btnFront);
            // Looking directly at tube from front (+Z to -Z)
            camera.position.set(pipeLength * 0.5, 0, pipeRadius * 3.4);
            controls.target.set(pipeLength * 0.5, 0, 0);
            controls.update();
        }});

        btnTrans.addEventListener('click', () => {{
            setActiveViewBtn(btnTrans);
            // Looking down the tube axis along X from inlet to observe circular depth cross-section
            camera.position.set(-pipeLength * 0.5, 0, 0);
            controls.target.set(pipeLength * 0.5, 0, 0);
            controls.update();
        }});

        btnTop.addEventListener('click', () => {{
            setActiveViewBtn(btnTop);
            // Looking from top (+Y to -Y)
            camera.position.set(pipeLength * 0.5, pipeRadius * 3.4, 0);
            controls.target.set(pipeLength * 0.5, 0, 0);
            controls.update();
        }});

        // Color Mode Buttons
        const btnColDepth = document.getElementById('btn-col-depth');
        const btnColStatus = document.getElementById('btn-col-status');
        const colorbarCard = document.getElementById('colorbar-card');

        btnColDepth.addEventListener('click', () => {{
            btnColDepth.classList.add('active');
            btnColStatus.classList.remove('active');
            colorbarCard.style.display = 'flex';
            currentMode = 'depth';
            bubbleMeshes.forEach(m => {{
                m.material.color.copy(m.userData.depthColor);
            }});
        }});

        btnColStatus.addEventListener('click', () => {{
            btnColStatus.classList.add('active');
            btnColDepth.classList.remove('active');
            colorbarCard.style.display = 'none';
            currentMode = 'status';
            bubbleMeshes.forEach(m => {{
                m.material.color.copy(m.userData.statusColor);
            }});
        }});

        // Depth Exaggeration Slider
        const sliderDepth = document.getElementById('slider-depth-scale');
        const sliderVal = document.getElementById('depth-scale-val');

        sliderDepth.addEventListener('input', (e) => {{
            depthExaggeration = parseFloat(e.target.value);
            sliderVal.innerText = `${{depthExaggeration.toFixed(1)}}×`;

            // Adjust Z position: front wall at local chord is +z_chord_half_m, depth moves inward
            bubbleMeshes.forEach(m => {{
                const d = m.userData;
                const newZ = d.z_chord_half_m - (d.depth_fluid_z_m * depthExaggeration);
                m.position.z = Math.max(-d.z_chord_half_m + d.radius_m, Math.min(d.z_chord_half_m - d.radius_m, newZ));
            }});
        }});

        // Visibility Toggles
        document.getElementById('toggle-wireframe').addEventListener('change', (e) => {{
            wireframeMesh.visible = e.target.checked;
            glassMesh.visible = e.target.checked;
        }});

        document.getElementById('toggle-rings').addEventListener('change', (e) => {{
            ringsGroup.visible = e.target.checked;
        }});

        document.getElementById('toggle-slug').addEventListener('change', (e) => {{
            slugGroup.visible = e.target.checked;
        }});

        // Animation Loop
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();

        // Responsive Resize
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_html_path


# ---------------------------------------------------------------------------
# 5. Label Studio ground-truth extraction and sampling
# ---------------------------------------------------------------------------


def extract_from_label_studio_json(
    json_path: Path,
    output_images_dir: Path,
    output_masks_dir: Path,
    source_images_dir: Path | None = None,
    use_predictions: bool = False,
) -> list[tuple[Path, Path]]:
    """
    Parses Label Studio JSON export, decoding base64 images or copying from source_images_dir,
    and rendering corresponding ground-truth binary masks.
    Returns list of (image_path, mask_path) pairs.
    """
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_masks_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, encoding="utf-8") as f:
        tasks = json.load(f)

    if not isinstance(tasks, list):
        if isinstance(tasks, dict) and "tasks" in tasks:
            tasks = tasks["tasks"]
        else:
            tasks = [tasks]

    pairs = []
    print(f"[LABEL STUDIO] Parsing {len(tasks)} tasks from: {json_path}")

    for idx, task in enumerate(tasks):
        annotations = task.get("annotations", [])
        results = []

        if annotations and len(annotations) > 0:
            results = annotations[0].get("result", [])
        elif use_predictions and task.get("predictions") and len(task["predictions"]) > 0:
            results = task["predictions"][0].get("result", [])

        if not results:
            continue

        # Extract dimension
        orig_w = next((r.get("original_width") for r in results if "original_width" in r), None)
        orig_h = next((r.get("original_height") for r in results if "original_height" in r), None)

        # Determine target image filename
        task_id = task.get("id") or idx + 1
        file_upload = task.get("file_upload") or ""
        clean_name = re.sub(r"^[a-f0-9]{8,}-", "", Path(file_upload).name) if file_upload else ""
        valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

        image_entry = task.get("data", {}).get("image", "")

        if clean_name and Path(clean_name).suffix.lower() in valid_exts:
            target_stem_name = Path(clean_name).stem
        else:
            entry_name = Path(image_entry).name
            clean_entry = re.sub(r"^[a-f0-9]{8,}-", "", entry_name)
            if clean_entry and Path(clean_entry).suffix.lower() in valid_exts:
                target_stem_name = Path(clean_entry).stem
            else:
                target_stem_name = f"frame_{int(task_id):04d}"

        target_img_path = output_images_dir / f"{target_stem_name}.png"
        target_mask_path = output_masks_dir / f"{target_stem_name}.png"
        raw_img = None

        # Try base64 decoding
        if "base64," in image_entry:
            b64_str = image_entry.split("base64,")[1]
            img_bytes = base64.b64decode(b64_str)
            raw_img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if raw_img is not None:
                cv2.imwrite(str(target_img_path), raw_img)

        # If not base64, look up in source_images_dir
        if raw_img is None and source_images_dir and source_images_dir.exists():
            candidates = [
                source_images_dir / target_img_path.name,
                source_images_dir / f"{target_stem_name}.png",
                source_images_dir / f"{target_stem_name}.jpg",
                source_images_dir / f"{target_stem_name}.tif",
                source_images_dir / Path(image_entry).name,
            ]
            if clean_name:
                candidates.append(source_images_dir / clean_name)
            for c in candidates:
                if c.exists():
                    raw_img = cv2.imread(str(c))
                    if raw_img is not None:
                        shutil.copy2(c, target_img_path)
                        break

        # Fallback to dimensions from raw_img if original_width was missing
        if raw_img is not None and (orig_h is None or orig_w is None):
            orig_h, orig_w = raw_img.shape[:2]

        if orig_h is None or orig_w is None:
            continue

        # Rasterize mask
        mask = rasterize_label_studio_results(results, int(orig_h), int(orig_w))
        cv2.imwrite(str(target_mask_path), mask)

        if target_img_path.exists():
            pairs.append((target_img_path, target_mask_path))

    print(f"[LABEL STUDIO] Extracted {len(pairs)} image/mask ground truth pairs.")
    return pairs


def prepare_sample_tasks(
    images_dir: Path,
    count: int = 10,
    output_dir: Path = Path("sample_dfd_tasks"),
):
    """
    Utility helper to pick N evenly-spaced frames from an experimental run
    and prepare them into a clean folder ready for Label Studio import.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images_out = output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    all_files = sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif"} and not p.name.startswith(".")
        ]
    )

    if not all_files:
        print(f"[ERROR] No image files found in: {images_dir}")
        return

    n = len(all_files)
    step = max(1, n // count)
    selected = [all_files[i * step] for i in range(min(count, n))]

    tasks_json = []
    print(f"[SAMPLER] Selected {len(selected)} frames from {n} total frames in {images_dir}:")
    for idx, f in enumerate(selected):
        dest = images_out / f.name
        shutil.copy2(f, dest)

        # Also create embedded base64 task for single-file Label Studio upload
        with open(f, "rb") as img_f:
            b64_val = base64.b64encode(img_f.read()).decode("utf-8")
        tasks_json.append(
            {
                "id": idx + 1,
                "file_upload": f.name,
                "data": {"image": f"data:image/png;base64,{b64_val}"},
            }
        )
        print(f"  -> {dest.name}")

    json_import_file = output_dir / "label_studio_import.json"
    with open(json_import_file, "w", encoding="utf-8") as jf:
        json.dump(tasks_json, jf, indent=2)

    print(f"\n[DONE] Prepared sample image set in: {images_out}")
    print(f"[DONE] Created single-file Label Studio import at: {json_import_file}")
    print("       In Label Studio, you can either:")
    print(f"         1. Upload '{json_import_file.name}' via the Import button, OR")
    print(f"         2. Drag & drop the {len(selected)} PNG files from '{images_out.name}/'.")


# ---------------------------------------------------------------------------
# 6. Pipeline
# ---------------------------------------------------------------------------


def collect_image_mask_pairs(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    """Pair every mask PNG with its raw frame (same stem, optional `_mask` suffix)."""
    pairs: list[tuple[Path, Path]] = []
    for mask_file in sorted(masks_dir.glob("*.png")):
        stem = mask_file.stem.replace("_mask", "")
        candidates = [
            images_dir / mask_file.name,
            images_dir / f"{stem}.png",
            images_dir / f"{stem}.jpg",
            images_dir / f"{stem}.tif",
        ]
        image_file = next((c for c in candidates if c.exists()), None)
        if image_file is not None:
            pairs.append((image_file, mask_file))
    return pairs


def _print_summary(df: pd.DataFrame, n_frames: int, elapsed_s: float) -> None:
    """Print the per-status breakdown and the reliable-bubble statistics."""
    print("\n" + "=" * 65)
    print("          DEPTH FROM DEFOCUS (DFD) SUMMARY (SI)")
    print("=" * 65)
    print(
        f"Execution Time               : {elapsed_s:.2f} s ({elapsed_s / max(1, n_frames):.3f} s/frame)"
    )
    print(f"Total Entities Analyzed      : {len(df)}")
    if df.empty:
        print("=" * 65)
        return

    labels = [
        ("valid", "Valid Bubbles (Core DFD)"),
        ("near_wall_regularized", "Near-Wall Regularized"),
        ("saturated_rear_wall", "Saturated Rear Wall Bound"),
        ("unreliable_defocus", "Unreliable Defocus (>0.35)"),
        ("slug_excluded", "Slugs Excluded (Regime 2)"),
    ]
    for status, label in labels:
        count = int((df["dfd_status"] == status).sum())
        print(f"  {label:<27}: {count:4d} ({count / len(df) * 100:.1f}%)")

    valid_df = df[df["is_reliable_dfd"]]
    if not valid_df.empty:
        print("-" * 65)
        print(f"Reliable Bubble Metrics (N = {len(valid_df)}):")
        print(f"  Mean Physical Diameter (d0): {valid_df['diameter_d0_m'].mean() * 1e3:.3f} mm")
        print(f"  Mean Fluid Depth (z_fluid) : {valid_df['depth_fluid_z_m'].mean() * 1e3:.3f} mm")
        print(f"  Min Fluid Depth (z_fluid)  : {valid_df['depth_fluid_z_m'].min() * 1e3:.3f} mm")
        print(f"  Max Fluid Depth (z_fluid)  : {valid_df['depth_fluid_z_m'].max() * 1e3:.3f} mm")
        print(f"  Mean Dimensionless Blur    : {valid_df['sigma_tilde'].mean():.4f}")
    print("=" * 65)


def _render_frame_3d_outputs(
    raw_img: np.ndarray,
    mask: np.ndarray,
    frame_results: list[dict[str, Any]],
    frame_name: str,
    diagnostic_dir: Path,
    output_root: Path,
    pixel_size_m: float,
    inner_diameter_m: float,
    pipe_center_y_px: float,
    beta_eff: float,
) -> None:
    """Render the 3D spatial figure, the dual-regime volumetric figure and the WebGL viewer."""
    pipe_length_m = raw_img.shape[1] * pixel_size_m
    pipe_radius_m = inner_diameter_m / 2.0

    render_3d_dfd_spatial(
        frame_results,
        diagnostic_dir / f"dfd_3d_spatial_{frame_name}.png",
        pipe_radius_m=pipe_radius_m,
        pipe_length_m=pipe_length_m,
        pixel_size_m=pixel_size_m,
        pipe_center_y_px=pipe_center_y_px,
        title=r"$\mathbf{3D\ DFD\ Spatial\ Reconstruction}$" + f" ({frame_name})",
    )

    vol_recon = None
    try:
        from visualization.render_3d_flow import (
            reconstruct_3d_bubbles_si,
            render_3d_flow_visualization,
        )

        vol_recon = reconstruct_3d_bubbles_si(
            mask,
            pixel_size_m=pixel_size_m,
            pipe_dia_m=inner_diameter_m,
            dfd_records=frame_results,
        )
        render_3d_flow_visualization(
            vol_recon,
            diagnostic_dir / f"dfd_3d_volumetric_{frame_name}.png",
            title_suffix=f"Ground Truth ({frame_name})",
        )
    except Exception as e:
        print(f"  [WARN] Volumetric 3D render skipped: {e}")

    try:
        interactive_html = diagnostic_dir / f"dfd_3d_interactive_{frame_name}.html"
        render_dfd_interactive_html(
            frame_results,
            output_html_path=interactive_html,
            pipe_radius_m=pipe_radius_m,
            pipe_length_m=pipe_length_m,
            pixel_size_m=pixel_size_m,
            pipe_center_y_px=pipe_center_y_px,
            vol_recon=vol_recon,
            beta_eff=beta_eff,
            frame_name=frame_name,
        )
        shutil.copy2(interactive_html, output_root / "dfd_3d_interactive.html")
        print(f"  [HTML] Generated interactive 3D WebGL: {interactive_html.name}")
    except Exception as e:
        print(f"  [WARN] Interactive 3D HTML render skipped: {e}")


def run_dfd_on_pairs(
    image_mask_pairs: list[tuple[Path, Path]],
    output_csv: Path,
    pixel_size_m: float,
    beta_eff: float = DEFAULT_BETA_EFF_PX_PER_M,
    background_path: Path | None = None,
    near_wall_threshold: float = SIGMA_TILDE_NEAR_FOCUS,
    max_sigma_threshold: float = SIGMA_TILDE_CRITICAL,
    wall_thickness_m: float = DEFAULT_WALL_THICKNESS_M,
    inner_diameter_m: float = DEFAULT_INNER_DIAMETER_M,
    pipe_center_y_px: float | None = None,
    focal_plane_z_m: float | None = None,
    diagnostic_dir: Path | None = None,
    max_diagnostic_frames: int | None = None,
    render_3d: bool = False,
    show_labels: bool = False,
    min_label_area_px: float = 50.0,
) -> pd.DataFrame:
    """
    Run DFD on a list of (image, mask) pairs and write one CSV row per detected entity.

    Diagnostics are rendered when diagnostic_dir is given: the 2-panel figure for every frame
    (or the first max_diagnostic_frames - 1 frames plus the last one), and with render_3d also
    the 3D spatial figure, the volumetric figure and the interactive WebGL viewer.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if diagnostic_dir is not None:
        diagnostic_dir.mkdir(parents=True, exist_ok=True)

    r_tube_px = (inner_diameter_m / 2.0) / pixel_size_m
    dfd_model = DFDModel()

    bg_img = None
    if background_path is not None and background_path.exists():
        bg_img = cv2.imread(str(background_path), cv2.IMREAD_GRAYSCALE)

    n_frames = len(image_mask_pairs)
    print("\n" + "=" * 65)
    print("   CTF2 DEPTH FROM DEFOCUS (DFD) ANALYSIS")
    print("=" * 65)
    print(f"  Total Frames           : {n_frames}")
    print(f"  Pixel size (SI)        : {pixel_size_m:.6e} m/px ({pixel_size_m * 1e6:.2f} um/px)")
    print(f"  Beta calibration       : {beta_eff:.2f} px/m")
    print(f"  Wall thickness (SI)    : {wall_thickness_m * 1e3:.1f} mm")
    print(f"  Tube inner diameter    : {inner_diameter_m * 1e3:.1f} mm")
    print(
        f"  Reliability bounds     : sigma_tilde in [{near_wall_threshold:.2f}, {max_sigma_threshold:.2f}]"
    )
    focus_txt = (
        "local front fluid boundary" if focal_plane_z_m is None else f"z = {focal_plane_z_m:.4f} m"
    )
    print(f"  Focal plane            : {focus_txt}")
    bg_txt = f"subtracted ({background_path})" if bg_img is not None else "none (8-bit inversion)"
    print(f"  Background             : {bg_txt}")
    print("-" * 65)

    all_records: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    for idx, (img_path, mask_path) in enumerate(image_mask_pairs):
        raw_img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw_img is None or mask is None:
            print(f"[SKIP] Could not load pair: {img_path.name}")
            continue

        frame_center_y = (
            pipe_center_y_px if pipe_center_y_px is not None else float(raw_img.shape[0]) / 2.0
        )
        frame_results = analyze_frame_dfd(
            raw_img,
            mask,
            dfd_model,
            pixel_size_m=pixel_size_m,
            beta_eff_px_m=beta_eff,
            background_img=bg_img,
            near_wall_sigma_threshold=near_wall_threshold,
            max_sigma_threshold=max_sigma_threshold,
            wall_thickness_m=wall_thickness_m,
            inner_diameter_m=inner_diameter_m,
            pipe_center_y_px=frame_center_y,
            focal_plane_z_m=focal_plane_z_m,
        )
        for r in frame_results:
            r["frame"] = img_path.stem
            all_records.append(r)

        render_this_frame = diagnostic_dir is not None and (
            max_diagnostic_frames is None or idx < max_diagnostic_frames - 1 or idx == n_frames - 1
        )
        if render_this_frame:
            render_dfd_diagnostic(
                raw_img,
                mask,
                frame_results,
                diagnostic_dir / f"dfd_diag_{img_path.stem}.png",
                pipe_center_y_px=frame_center_y,
                r_tube_px=r_tube_px,
                pixel_size_m=pixel_size_m,
                wall_thickness_m=wall_thickness_m,
                inner_diameter_m=inner_diameter_m,
                show_labels=show_labels,
                min_label_area_px=min_label_area_px,
            )
            if render_3d:
                _render_frame_3d_outputs(
                    raw_img,
                    mask,
                    frame_results,
                    frame_name=img_path.stem,
                    diagnostic_dir=diagnostic_dir,
                    output_root=output_csv.parent,
                    pixel_size_m=pixel_size_m,
                    inner_diameter_m=inner_diameter_m,
                    pipe_center_y_px=frame_center_y,
                    beta_eff=beta_eff,
                )

        print(
            f"  Frame {idx + 1:02d}/{n_frames:02d} [{img_path.name}]: {len(frame_results)} entities analyzed."
        )

    elapsed_s = time.perf_counter() - start_time
    df = pd.DataFrame(all_records, columns=DFD_COLUMNS)
    df.to_csv(output_csv, index=False)

    _print_summary(df, n_frames, elapsed_s)
    print(f"[DONE] Saved DFD results to: {output_csv}")
    if diagnostic_dir is not None:
        print(f"[DONE] Saved diagnostic figures to: {diagnostic_dir}")
    return df


def run_dfd_pipeline(
    raw_images_dir: Path,
    masks_dir: Path,
    output_csv: Path,
    pixel_size: float,
    beta_eff: float = DEFAULT_BETA_EFF_PX_PER_M,
    background_path: Path | None = None,
    near_wall_threshold: float = SIGMA_TILDE_NEAR_FOCUS,
    max_sigma_threshold: float = SIGMA_TILDE_CRITICAL,
    wall_thickness_m: float = DEFAULT_WALL_THICKNESS_M,
    inner_diameter_m: float = DEFAULT_INNER_DIAMETER_M,
    pipe_center_y_px: float | None = None,
    render_diagnostic: bool = False,
    diagnostic_output_dir: Path | None = None,
    focal_plane_z_m: float | None = None,
    render_3d: bool = False,
    max_diagnostic_frames: int | None = 6,
) -> pd.DataFrame:
    """Run DFD on a folder of raw frames and a folder of binary masks (entry point of `main.py dfd`)."""
    pairs = collect_image_mask_pairs(raw_images_dir, masks_dir)
    if not pairs:
        raise FileNotFoundError(
            f"No mask/image pairs found (masks: {masks_dir}, images: {raw_images_dir})"
        )
    return run_dfd_on_pairs(
        pairs,
        output_csv,
        pixel_size_m=to_si_pixel_size(pixel_size),
        beta_eff=beta_eff,
        background_path=background_path,
        near_wall_threshold=near_wall_threshold,
        max_sigma_threshold=max_sigma_threshold,
        wall_thickness_m=wall_thickness_m,
        inner_diameter_m=inner_diameter_m,
        pipe_center_y_px=pipe_center_y_px,
        focal_plane_z_m=focal_plane_z_m,
        diagnostic_dir=diagnostic_output_dir if render_diagnostic else None,
        max_diagnostic_frames=max_diagnostic_frames,
        render_3d=render_3d,
    )


# ---------------------------------------------------------------------------
# 7. Command line interface
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-camera Depth from Defocus (DFD) sizing and depth estimation in SI units",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    inputs = parser.add_argument_group("inputs")
    inputs.add_argument(
        "--images",
        "-i",
        type=Path,
        default=None,
        help="Raw image directory (also the image pool for --json and --prepare-tasks)",
    )
    inputs.add_argument("--masks", "-m", type=Path, default=None, help="Binary mask directory")
    inputs.add_argument("--json", "-j", type=Path, default=None, help="Label Studio JSON export")
    inputs.add_argument(
        "--use-predictions",
        action="store_true",
        help="With --json: fall back to task predictions when a task has no annotations",
    )
    inputs.add_argument(
        "--background", "-bg", type=Path, default=None, help="Background image for normalisation"
    )

    sampler = parser.add_argument_group("Label Studio sampling")
    sampler.add_argument(
        "--prepare-tasks",
        action="store_true",
        help="Sample frames from --images into a folder ready for Label Studio, then exit",
    )
    sampler.add_argument("--sample-count", type=int, default=10, help="Frames to sample")
    sampler.add_argument(
        "--sample-output", type=Path, default=Path("sample_dfd_tasks"), help="Sample destination"
    )

    physics = parser.add_argument_group("calibration (SI)")
    physics.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="Pixel size in m/px. Required unless a 0_*calibration.txt matching the image "
        "size is found near the inputs (there is no default)",
    )
    physics.add_argument(
        "--beta-eff",
        type=float,
        default=DEFAULT_BETA_EFF_PX_PER_M,
        help="Defocus constant beta in px/m (sigma [px] per metre of defocus)",
    )
    physics.add_argument(
        "--diameter",
        "-d",
        type=float,
        default=None,
        help=f"Tube inner diameter in m; auto-read from the calibration file when omitted, "
        f"else {DEFAULT_INNER_DIAMETER_M}",
    )
    physics.add_argument(
        "--wall-thickness",
        type=float,
        default=DEFAULT_WALL_THICKNESS_M,
        help="Glass wall thickness in m (offset of the fluid from the outer surface z = 0)",
    )
    physics.add_argument(
        "--pipe-center-y", type=float, default=None, help="Tube centreline row in px"
    )
    physics.add_argument(
        "--focal-plane-z",
        type=float,
        default=None,
        help="Absolute z of a planar object plane in m (z = 0 at the outer glass surface). "
        "Omit to anchor the focus at the local front fluid boundary",
    )
    physics.add_argument(
        "--near-wall-threshold",
        type=float,
        default=SIGMA_TILDE_NEAR_FOCUS,
        help="sigma_tilde below which the near-focus regulariser is applied",
    )
    physics.add_argument(
        "--max-sigma-threshold",
        type=float,
        default=SIGMA_TILDE_CRITICAL,
        help="Critical sigma_tilde beyond which results are rejected",
    )

    outputs = parser.add_argument_group("outputs")
    outputs.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Output directory: writes dfd_results.csv, diagnostics/ and (with --json) "
        "extracted_data/. A path ending in .csv is taken as the CSV file itself. "
        "Defaults to results/dfd_groundtruth with --json",
    )
    outputs.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="CSV path (overrides --output-dir; default results/dfd_results.csv)",
    )
    outputs.add_argument(
        "--render-diagnostic",
        action="store_true",
        help="Render the 2-panel diagnostic figures (always on with --json unless --no-diagnostics)",
    )
    outputs.add_argument("--no-diagnostics", action="store_true", help="Disable all figures")
    outputs.add_argument(
        "--diagnostic-dir", type=Path, default=None, help="Diagnostic figure directory"
    )
    outputs.add_argument(
        "--max-diagnostic-frames",
        type=int,
        default=None,
        help="Limit the number of frames with figures (default: 6 for folders, all with --json)",
    )
    outputs.add_argument(
        "--render-3d",
        action="store_true",
        help="Also render the 3D spatial figure, volumetric figure and interactive WebGL viewer "
        "(always on with --json)",
    )
    outputs.add_argument("--show-labels", action="store_true", help="Annotate bubbles with ID/z/d")
    outputs.add_argument(
        "--min-label-area",
        type=float,
        default=60.0,
        help="Minimum bubble area in px for a text label",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.prepare_tasks:
        if not args.images:
            print("[ERROR] --prepare-tasks requires --images <path_to_images>.")
            return 1
        prepare_sample_tasks(args.images, count=args.sample_count, output_dir=args.sample_output)
        return 0

    ground_truth_mode = args.json is not None
    output_dir = args.output_dir
    if output_dir is not None and output_dir.suffix.lower() == ".csv":
        # `-o results/dfd_results.csv` (the previous CLI form) names the CSV directly.
        args.output_csv = args.output_csv or output_dir
        output_dir = output_dir.parent
    if output_dir is None and ground_truth_mode:
        output_dir = Path("results/dfd_groundtruth")
    output_csv = args.output_csv or (
        output_dir / "dfd_results.csv" if output_dir else Path("results/dfd_results.csv")
    )

    if ground_truth_mode:
        if not args.json.exists():
            print(f"[ERROR] Label Studio JSON file not found: {args.json}")
            return 1
        dataset_dir = output_csv.parent / "extracted_data"
        pairs = extract_from_label_studio_json(
            args.json,
            output_images_dir=dataset_dir / "images",
            output_masks_dir=dataset_dir / "masks",
            source_images_dir=args.images,
            use_predictions=args.use_predictions,
        )
    elif args.images and args.masks:
        if not args.images.exists() or not args.masks.exists():
            print(f"[ERROR] Images ({args.images}) or masks ({args.masks}) path does not exist.")
            return 1
        pairs = collect_image_mask_pairs(args.images, args.masks)
        print(f"[DATASET] Found {len(pairs)} matched image and mask pairs.")
    else:
        print("[ERROR] Provide one of:")
        print("  1. --images <images_dir> --masks <masks_dir>")
        print("  2. --json <label_studio_export.json> [--images <image_pool>]")
        print("  3. --prepare-tasks --images <images_dir>")
        return 1

    if not pairs:
        print("[ERROR] No valid image-mask pairs found to process.")
        return 1

    # Calibration: explicit flags win, then the nearest 0_*calibration.txt, then defaults.
    pixel_size_m = args.pixel_size
    pipe_center_y_px = args.pipe_center_y
    inner_diameter_m = args.diameter

    search_dirs = [pairs[0][0].parent]
    if args.json:
        search_dirs.append(args.json.parent)
    if args.images:
        search_dirs.append(args.images)
    sample_im = cv2.imread(str(pairs[0][0]))
    target_shape = sample_im.shape[:2] if sample_im is not None else None

    calib = auto_detect_calibration(search_dirs, target_shape=target_shape)
    if calib:
        src = calib.get("source_file", "calibration file")
        if pixel_size_m is None and "pixel_size_m" in calib:
            pixel_size_m = float(calib["pixel_size_m"])
            print(f"[CALIBRATION] Auto-detected pixel size from {src}: {pixel_size_m:.6e} m/px")
        if pipe_center_y_px is None and "point1_y" in calib and "point2_y" in calib:
            pipe_center_y_px = (float(calib["point1_y"]) + float(calib["point2_y"])) / 2.0
            print(f"[CALIBRATION] Auto-detected pipe centerline: {pipe_center_y_px:.1f} px")
        if inner_diameter_m is None and "diameter_m" in calib:
            inner_diameter_m = float(calib["diameter_m"])

    if pixel_size_m is None:
        shape_txt = f"{target_shape[1]}x{target_shape[0]} px" if target_shape else "unknown size"
        print("[ERROR] No optical calibration available. There is no default pixel size.")
        print(f"        No 0_*calibration.txt matching the frames ({shape_txt}) was found near:")
        for d in search_dirs:
            print(f"          {d}")
        print("        Either pass --pixel-size <m/px>, or create the calibration file with:")
        print("          python main.py calibrate -i <image_folder> --size-mm <known_length_mm>")
        return 1
    if inner_diameter_m is None:
        inner_diameter_m = DEFAULT_INNER_DIAMETER_M

    render_figures = (ground_truth_mode or args.render_diagnostic) and not args.no_diagnostics
    diagnostic_dir = None
    if render_figures:
        default_diag = output_dir / "diagnostics" if output_dir else Path("results/dfd_diagnostics")
        diagnostic_dir = args.diagnostic_dir or default_diag
    max_diag = args.max_diagnostic_frames
    if max_diag is None and not ground_truth_mode:
        max_diag = 6

    run_dfd_on_pairs(
        pairs,
        output_csv,
        pixel_size_m=to_si_pixel_size(pixel_size_m),
        beta_eff=args.beta_eff,
        background_path=args.background,
        near_wall_threshold=args.near_wall_threshold,
        max_sigma_threshold=args.max_sigma_threshold,
        wall_thickness_m=args.wall_thickness,
        inner_diameter_m=inner_diameter_m,
        pipe_center_y_px=pipe_center_y_px,
        focal_plane_z_m=args.focal_plane_z,
        diagnostic_dir=diagnostic_dir,
        max_diagnostic_frames=max_diag,
        render_3d=ground_truth_mode or args.render_3d,
        show_labels=args.show_labels,
        min_label_area_px=args.min_label_area,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
