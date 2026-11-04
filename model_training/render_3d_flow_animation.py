#!/usr/bin/env python3
"""
3D Animated Flow Reconstruction Engine in Standard SI Units.

Reconstructs 3D multi-phase flow structures across continuous frame sequences:
  - Small bubbles: Tri-axial ellipsoids computed via moments/eigenvalues.
  - Large slugs: Watertight 3D meshes reconstructed via distance-transform chord extrusion
    and Laplacian boundary smoothing.
  - Cylindrical confinement: Local chord depth constraint z_chord(y) = sqrt(max(0, R^2 - delta_y^2)).
  - Interactive WebGL Flow Animation Player:
      * Interactive playback scrubber (Play / Pause / Loop / Step +/-).
      * Playback speed control (0.25x, 0.5x, 1x, 2x, 5x, 10x).
      * OrbitControls (full 3D rotation, pan, zoom).
      * Live HUD telemetry (Frame index, physical timestamp ms, void fraction %, bubble count, gas volume mm³).
      * Theme toggle (White Mode / Dark Mode) for presentation readability.
      * Fast client-side decoding via zlib / pako.js decompression.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
import zlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from image_scripts.calibrate_pixel_size import parse_calibration_file
from model_training.render_3d_flow import (
    extract_slug_mesh_data,
    reconstruct_3d_bubbles_si,
)


def format_frame_label(stem: str) -> str:
    clean = re.sub(r"(_mask|_pred|_seg)$", "", stem)
    match = re.search(r"(\d+)$", clean)
    if match:
        return f"#{int(match.group(1)):04d}"
    return clean[-10:] if len(clean) > 10 else clean


def process_sequence_3d(
    mask_dir: Path,
    output_html_path: Path,
    pixel_size_m: float = 5.936412e-05,
    pipe_dia_m: float = 0.020,
    fps: float = 2000.0,
    start_frame: int = 0,
    max_frames: int = 120,
    stride: int = 1,
    slug_mesh_step: int = 6,
    area_thresh_px: float = 5000.0,
    title: str = "3D Flow Animation Reconstruction",
    calib_file: Path | str | None = None,
    y_center_px: float | None = None,
) -> Path:
    output_html_path = Path(output_html_path)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    # Automatically ingest calibration file if provided
    calib_dict: dict[str, Any] = {}
    if calib_file is not None:
        c_path = Path(calib_file)
        if c_path.is_file():
            calib_dict = parse_calibration_file(c_path)
            if "pixel_size_m" in calib_dict:
                pixel_size_m = float(calib_dict["pixel_size_m"])
            if "diameter_m" in calib_dict:
                pipe_dia_m = float(calib_dict["diameter_m"])
            if y_center_px is None:
                if "point1_y" in calib_dict and "point2_y" in calib_dict:
                    y_center_px = (
                        float(calib_dict["point1_y"]) + float(calib_dict["point2_y"])
                    ) / 2.0
                elif "image_height_px" in calib_dict:
                    y_center_px = float(calib_dict["image_height_px"]) / 2.0

    mask_files = sorted(list(mask_dir.glob("*.png")) + list(mask_dir.glob("*.tif*")))
    if not mask_files:
        raise FileNotFoundError(f"No mask files found in: {mask_dir}")

    total_available = len(mask_files)
    safe_start = max(0, min(start_frame, total_available - 1))
    selected_files = mask_files[safe_start : safe_start + max_frames * stride : stride]

    if not selected_files:
        raise ValueError(
            f"No frames selected with start={safe_start}, max={max_frames}, stride={stride}"
        )

    first_mask = cv2.imread(str(selected_files[0]), cv2.IMREAD_GRAYSCALE)
    if first_mask is None:
        raise FileNotFoundError(f"Could not read first mask: {selected_files[0]}")

    h_px, w_px = first_mask.shape
    if y_center_px is None:
        y_center_px = h_px / 2.0
    R_m = pipe_dia_m / 2.0
    L_m = w_px * pixel_size_m
    pipe_volume_m3 = np.pi * (R_m**2) * L_m

    print("=" * 64)
    print("      3D DYNAMIC FLOW ANIMATION RECONSTRUCTION (SI UNITS)      ")
    print("=" * 64)
    print(f"  Mask Directory    : {mask_dir}")
    print(f"  Frames Available  : {total_available}")
    print(f"  Frames Selected   : {len(selected_files)} (stride = {stride})")
    print(f"  Capture Rate (FPS): {fps:.1f} Hz")
    print(f"  Pipe Diameter (D) : {pipe_dia_m * 1e3:.2f} mm ({pipe_dia_m:.4e} m)")
    print(f"  Pixel Scale (SI)  : {pixel_size_m * 1e3:.4f} mm/px ({pixel_size_m:.6e} m/px)")
    print(f"  Centerline Y      : {y_center_px:.2f} px")
    print(f"  Slug Mesh Step    : {slug_mesh_step} px")
    if calib_dict:
        print(f"  Calibration File  : {calib_file}")
    print("=" * 64)

    frames_data = []
    t_start = time.perf_counter()

    for idx, fpath in enumerate(selected_files):
        mask = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        recon = reconstruct_3d_bubbles_si(
            mask,
            pixel_size_m=pixel_size_m,
            pipe_dia_m=pipe_dia_m,
            area_thresh_px=area_thresh_px,
            y_center_px=y_center_px,
        )

        spheres = []
        slugs = []

        for b in recon["bubbles"]:
            if b["regime"] == "small_ellipsoid":
                cx, cy, _ = b["centroid_m"]
                cz = (y_center_px - cy / pixel_size_m) * pixel_size_m
                a, b_axis, c = b["semi_axes_m"]
                cz = float(np.clip(cz, -R_m + b_axis, R_m - b_axis)) if R_m > b_axis else cz
                rot = float(b.get("orientation_rad", 0.0))
                # Store compact array: [cx, cz, 0, rx, ry, rz, rot]
                spheres.append(
                    [
                        round(cx, 5),
                        round(cz, 5),
                        0.0,
                        round(a, 5),
                        round(b_axis, 5),
                        round(c, 5),
                        round(rot, 3),
                    ]
                )
            elif b["regime"] == "large_slug":
                slug_mesh = extract_slug_mesh_data(
                    b["blob_mask"],
                    b["depth_map_m"],
                    pixel_size_m=pixel_size_m,
                    y_center_px=y_center_px,
                    step=slug_mesh_step,
                    pipe_radius_m=R_m,
                )
                if slug_mesh["vertices"] and slug_mesh["indices"]:
                    # Round vertices to 5 decimals
                    v_rounded = [round(x, 5) for x in slug_mesh["vertices"]]
                    slugs.append(
                        {
                            "v": v_rounded,
                            "i": slug_mesh["indices"],
                        }
                    )

        frame_time_ms = (idx * stride / fps) * 1e3
        frames_data.append(
            {
                "idx": idx,
                "label": format_frame_label(fpath.stem),
                "time_ms": round(frame_time_ms, 2),
                "vf_pct": round(recon["void_fraction_alpha"] * 100.0, 3),
                "gas_vol_mm3": round(recon["gas_volume_m3"] * 1e9, 2),
                "bubble_count": recon["bubble_count"],
                "spheres": spheres,
                "slugs": slugs,
            }
        )

        if (idx + 1) % 20 == 0 or (idx + 1) == len(selected_files):
            el = time.perf_counter() - t_start
            print(
                f"  Reconstructed [{idx + 1}/{len(selected_files)}] frames ({((idx + 1) / len(selected_files)) * 100:.1f}%) - Speed: {(idx + 1) / max(1e-5, el):.1f} FPS"
            )

    # Serialize & Compress Frame Data using zlib + Base64
    raw_json_bytes = json.dumps(frames_data).encode("utf-8")
    compressed_bytes = zlib.compress(raw_json_bytes, level=9)
    b64_payload = base64.b64encode(compressed_bytes).decode("ascii")

    print(
        f"\n[DATA COMPRESSION] Raw JSON: {len(raw_json_bytes) / (1024 * 1024):.2f} MB -> Compressed B64: {len(b64_payload) / (1024 * 1024):.2f} MB ({len(raw_json_bytes) / len(b64_payload):.2f}x compression)"
    )

    # Build HTML Interactive Viewer
    html_content = generate_html_viewer(
        b64_payload=b64_payload,
        total_frames=len(frames_data),
        fps=fps,
        stride=stride,
        R_m=R_m,
        L_m=L_m,
        pipe_volume_m3=pipe_volume_m3,
        pixel_size_m=pixel_size_m,
        title=title,
    )

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    file_size_mb = os.path.getsize(output_html_path) / (1024 * 1024)
    print(
        f"[DONE] Interactive 3D Flow Animation Viewer saved to: {output_html_path} ({file_size_mb:.2f} MB)"
    )
    return output_html_path


def generate_html_viewer(
    b64_payload: str,
    total_frames: int,
    fps: float,
    stride: int,
    R_m: float,
    L_m: float,
    pipe_volume_m3: float,
    pixel_size_m: float,
    title: str,
) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title} (SI Units)</title>
  <style>
    :root {{
      --bg-dark: #0a0d14;
      --card-bg: rgba(16, 21, 33, 0.88);
      --border-color: rgba(56, 189, 248, 0.22);
      --accent-cyan: #00d2ff;
      --accent-blue: #3a7bd5;
      --text-main: #f0f3f8;
      --text-muted: #8b9bb4;
      --slug-color: #8b0000;
      --pipe-color: #1a4f94;
    }}
    body.theme-light {{
      --bg-dark: #f1f5f9;
      --card-bg: rgba(255, 255, 255, 0.94);
      --border-color: rgba(203, 213, 225, 0.9);
      --accent-cyan: #0284c7;
      --accent-blue: #2563eb;
      --text-main: #0f172a;
      --text-muted: #64748b;
      --slug-color: #990000;
      --pipe-color: #002277;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
    body {{
      background-color: var(--bg-dark);
      color: var(--text-main);
      overflow: hidden;
      width: 100vw;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }}
    #webgl-canvas {{
      width: 100%;
      height: 100%;
      position: absolute;
      top: 0;
      left: 0;
      z-index: 1;
    }}

    /* Top Floating Header */
    #header {{
      position: absolute;
      top: 14px;
      left: 50%;
      transform: translateX(-50%);
      z-index: 10;
      background: var(--card-bg);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 10px 22px;
      backdrop-filter: blur(12px);
      box-shadow: 0 8px 32px rgba(0,0,0,0.4);
      text-align: center;
      pointer-events: auto;
    }}
    #header h1 {{
      font-size: 1.15rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      background: linear-gradient(135deg, #00d2ff 0%, #ffffff 80%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    body.theme-light #header h1 {{
      background: linear-gradient(135deg, #0284c7 0%, #0f172a 80%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    #header p {{
      font-size: 0.78rem;
      color: var(--text-muted);
      margin-top: 2px;
    }}

    /* Top-Right Legend & View Presets */
    #top-right {{
      position: absolute;
      top: 14px;
      right: 16px;
      z-index: 10;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .panel {{
      background: var(--card-bg);
      border: 1px solid var(--border-color);
      border-radius: 10px;
      padding: 12px 14px;
      backdrop-filter: blur(12px);
      box-shadow: 0 6px 24px rgba(0,0,0,0.3);
      font-size: 0.78rem;
    }}
    .legend-row {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 5px;
    }}
    .legend-box {{
      width: 14px;
      height: 14px;
      border-radius: 3px;
    }}
    .view-btn-group {{
      display: flex;
      gap: 4px;
      margin-top: 6px;
    }}
    .mini-btn {{
      background: rgba(255,255,255,0.06);
      border: 1px solid var(--border-color);
      color: var(--text-main);
      border-radius: 4px;
      padding: 3px 8px;
      font-size: 0.72rem;
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .mini-btn:hover {{
      background: rgba(56, 189, 248, 0.2);
      border-color: var(--accent-cyan);
    }}

    /* Bottom Control Bar */
    #controls-container {{
      position: absolute;
      bottom: 16px;
      left: 50%;
      transform: translateX(-50%);
      width: min(94vw, 1080px);
      z-index: 10;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .control-card {{
      background: var(--card-bg);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 12px 18px;
      backdrop-filter: blur(14px);
      box-shadow: 0 10px 36px rgba(0,0,0,0.45);
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .scrubber-row {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    input[type=range] {{
      flex: 1;
      height: 6px;
      border-radius: 3px;
      background: rgba(255,255,255,0.12);
      outline: none;
      -webkit-appearance: none;
      cursor: pointer;
    }}
    body.theme-light input[type=range] {{
      background: #cbd5e1;
    }}
    input[type=range]::-webkit-slider-thumb {{
      -webkit-appearance: none;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: var(--accent-cyan);
      box-shadow: 0 0 10px rgba(0,210,255,0.7);
      cursor: pointer;
    }}
    .badge {{
      font-variant-numeric: tabular-nums;
      font-size: 0.82rem;
      font-weight: 600;
      color: var(--accent-cyan);
      min-width: 90px;
      text-align: right;
    }}

    .btn-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .button-group {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    button.btn {{
      background: rgba(255,255,255,0.08);
      border: 1px solid var(--border-color);
      color: var(--text-main);
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 0.82rem;
      font-weight: 500;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
    }}
    button.btn:hover {{
      background: rgba(56, 189, 248, 0.18);
      border-color: var(--accent-cyan);
    }}
    button.btn.primary {{
      background: linear-gradient(135deg, var(--accent-blue), #0099ff);
      border: none;
      color: #fff;
    }}
    button.btn.primary:hover {{
      opacity: 0.92;
      box-shadow: 0 0 12px rgba(0, 153, 255, 0.5);
    }}
    select.speed-select {{
      background: rgba(255,255,255,0.08);
      color: var(--text-main);
      border: 1px solid var(--border-color);
      border-radius: 6px;
      padding: 5px 8px;
      font-size: 0.8rem;
      outline: none;
      cursor: pointer;
    }}
    body.theme-light select.speed-select {{
      background: #e2e8f0;
    }}

    /* Telemetry HUD Grid */
    .telemetry-bar {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 8px;
      padding: 8px 14px;
      background: rgba(0, 0, 0, 0.28);
      border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }}
    body.theme-light .telemetry-bar {{
      background: rgba(241, 245, 249, 0.9);
      border-color: rgba(0,0,0,0.06);
    }}
    .stat-item {{
      display: flex;
      flex-direction: column;
      gap: 2px;
    }}
    .stat-lbl {{
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
    }}
    .stat-val {{
      font-size: 0.92rem;
      font-weight: 700;
      color: var(--text-main);
      font-variant-numeric: tabular-nums;
    }}
  </style>

  <!-- External CDN Dependencies -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js"></script>
</head>
<body>
  <div id="webgl-canvas"></div>

  <!-- Header -->
  <div id="header">
    <h1>{title}</h1>
    <p>Conduit: D = {(R_m * 2) * 1e3:.1f} mm, L = {L_m * 1e3:.1f} mm | High-Speed Acquisition: {fps:.0f} Hz ({1000 / fps:.2f} ms/frame)</p>
  </div>

  <!-- Top Right Panel -->
  <div id="top-right">
    <div class="panel">
      <div class="legend-row">
        <div class="legend-box" style="background: #1a4f94; opacity: 0.7;"></div>
        <span>Cylindrical Conduit (D = {(R_m * 2) * 1e3:.1f} mm)</span>
      </div>
      <div class="legend-row">
        <div class="legend-box" style="background: #8b0000;"></div>
        <span>3D Vapor Interface (SI Geometries)</span>
      </div>
      <div style="margin-top: 8px; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 6px;">
        <span style="font-size:0.7rem; color:var(--text-muted); display:block; margin-bottom:4px;">CAMERA PRESETS</span>
        <div class="view-btn-group">
          <button class="mini-btn" onclick="setCameraView('side')">Side (X-Y)</button>
          <button class="mini-btn" onclick="setCameraView('top')">Top (X-Z)</button>
          <button class="mini-btn" onclick="setCameraView('front')">Cross-Section (Z-Y)</button>
          <button class="mini-btn" onclick="setCameraView('iso')">Isometric</button>
        </div>
      </div>
      <div style="margin-top: 8px; display: flex; justify-content: space-between; align-items: center;">
        <button class="mini-btn" id="theme-btn" onclick="toggleTheme()">☀️ White Mode</button>
        <button class="mini-btn" id="pipe-toggle-btn" onclick="togglePipe()">Hide Pipe</button>
      </div>
    </div>
  </div>

  <!-- Bottom Interactive Controls & HUD -->
  <div id="controls-container">
    <div class="control-card">
      <div class="scrubber-row">
        <input type="range" id="frame-slider" min="0" max="{total_frames - 1}" value="0" step="1">
        <span class="badge" id="frame-badge">Frame 1 / {total_frames}</span>
      </div>

      <div class="btn-row">
        <div class="button-group">
          <button class="btn primary" id="play-btn" onclick="togglePlay()">
            <span id="play-icon">▶</span> <span id="play-text">Play Flow</span>
          </button>
          <button class="btn" onclick="stepFrame(-1)">⏮ Prev</button>
          <button class="btn" onclick="stepFrame(1)">Next ⏭</button>
          <button class="btn" onclick="seekFrame(0)">Reset</button>
        </div>

        <div class="button-group">
          <span style="font-size:0.8rem; color:var(--text-muted);">Speed:</span>
          <select class="speed-select" id="speed-select" onchange="changeSpeed(this.value)">
            <option value="0.25">0.25x</option>
            <option value="0.5">0.5x</option>
            <option value="1.0" selected>1.0x (Slow-Mo)</option>
            <option value="2.0">2.0x</option>
            <option value="5.0">5.0x</option>
            <option value="10.0">10.0x</option>
          </select>
          <span style="font-size:0.75rem; color:var(--text-muted); margin-left: 6px;">Orbit: Left-Click | Pan: Right-Click | Zoom: Scroll</span>
        </div>
      </div>

      <!-- Live Real-Time Telemetry Bar -->
      <div class="telemetry-bar">
        <div class="stat-item">
          <span class="stat-lbl">Time Elapsed</span>
          <span class="stat-val" id="hud-time">0.00 ms</span>
        </div>
        <div class="stat-item">
          <span class="stat-lbl">Void Fraction (α)</span>
          <span class="stat-val" style="color:var(--accent-cyan);" id="hud-vf">0.000 %</span>
        </div>
        <div class="stat-item">
          <span class="stat-lbl">Active Bubbles</span>
          <span class="stat-val" id="hud-bubbles">0</span>
        </div>
        <div class="stat-item">
          <span class="stat-lbl">Gas Volume</span>
          <span class="stat-val" id="hud-gas-vol">0.00 mm³</span>
        </div>
        <div class="stat-item">
          <span class="stat-lbl">Conduit Volume</span>
          <span class="stat-val">{pipe_volume_m3 * 1e9:.1f} mm³</span>
        </div>
      </div>
    </div>
  </div>

  <script>
    // 1. Decompress Inlined Base64 Binary Payload
    const b64Data = "{b64_payload}";
    const binaryStr = atob(b64Data);
    const bytes = new Uint8Array(binaryStr.length);
    for (let i = 0; i < binaryStr.length; i++) {{
      bytes[i] = binaryStr.charCodeAt(i);
    }}
    const decompressed = pako.inflate(bytes, {{ to: 'string' }});
    const FRAMES = JSON.parse(decompressed);
    console.log("Loaded " + FRAMES.length + " 3D reconstructed frames successfully.");

    // 2. Three.js Scene Setup
    const container = document.getElementById('webgl-canvas');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0d14);

    const R_m = {R_m};
    const L_m = {L_m};

    const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.0005, 10);
    camera.position.set(L_m * 0.5, R_m * 1.5, R_m * 3.8);

    const renderer = new THREE.WebGLRenderer({{ antialias: true, powerPreference: "high-performance" }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.target.set(L_m * 0.5, 0, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.update();

    // 3. Lighting System
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xffffff, 0.8);
    keyLight.position.set(1, 2, 2).normalize();
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0xffffff, 0.5);
    fillLight.position.set(-1, -1, -2).normalize();
    scene.add(fillLight);

    const topLight = new THREE.DirectionalLight(0xffffff, 0.4);
    topLight.position.set(0, 3, 0).normalize();
    scene.add(topLight);

    // 4. Cylindrical Pipe Conduit Geometry
    const cylGeom = new THREE.CylinderGeometry(R_m, R_m, L_m, 64, 40, true);
    cylGeom.rotateZ(Math.PI / 2);
    cylGeom.translate(L_m / 2, 0, 0);

    const wireframeGeom = new THREE.WireframeGeometry(cylGeom);
    const pipeWireframe = new THREE.LineSegments(wireframeGeom);
    pipeWireframe.material.color.setHex(0x1a4f94);
    pipeWireframe.material.opacity = 0.6;
    pipeWireframe.material.transparent = true;
    scene.add(pipeWireframe);

    // Transparent Glass/Water Sheath
    const pipeSheathMat = new THREE.MeshPhysicalMaterial({{
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.06,
      roughness: 0.1,
      metalness: 0.1,
      clearcoat: 1.0,
      side: THREE.DoubleSide
    }});
    const pipeSheath = new THREE.Mesh(cylGeom, pipeSheathMat);
    scene.add(pipeSheath);

    // 5. Bubble Materials (Specularity, Subsurface Maroon Glow)
    const bubbleMat = new THREE.MeshPhongMaterial({{
      color: 0x8b0000,
      emissive: 0x1f0000,
      specular: 0x994444,
      shininess: 85,
      side: THREE.DoubleSide
    }});

    // Bubble Mesh Cache for active frame
    const bubbleGroup = new THREE.Group();
    scene.add(bubbleGroup);

    // Reusable Sphere Geometry Prototype
    const sphereProtoGeom = new THREE.SphereGeometry(1, 18, 14);

    function clearBubbleGroup() {{
      while (bubbleGroup.children.length > 0) {{
        const obj = bubbleGroup.children[0];
        bubbleGroup.remove(obj);
        if (obj.geometry && obj.geometry !== sphereProtoGeom) {{
          obj.geometry.dispose();
        }}
      }}
    }}

    function renderFrame(frameIdx) {{
      clearBubbleGroup();
      if (frameIdx < 0 || frameIdx >= FRAMES.length) return;
      const f = FRAMES[frameIdx];

      // Render Small Ellipsoid Bubbles
      if (f.spheres) {{
        f.spheres.forEach(s => {{
          const cx = s[0], cz = s[1], cy = s[2];
          const rx = s[3], ry = s[4], rz = s[5];
          const rot = s[6];

          const mesh = new THREE.Mesh(sphereProtoGeom, bubbleMat);
          mesh.scale.set(rx, ry, rz);
          mesh.position.set(cx, cz, cy);
          if (rot !== 0) {{
            mesh.rotation.z = rot;
          }}
          bubbleGroup.add(mesh);
        }});
      }}

      // Render Large Slugs (Extruded 3D Meshes)
      if (f.slugs) {{
        f.slugs.forEach(slug => {{
          if (slug.v && slug.v.length > 0 && slug.i && slug.i.length > 0) {{
            const geom = new THREE.BufferGeometry();
            geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(slug.v), 3));
            geom.setIndex(slug.i);
            geom.computeVertexNormals();
            const mesh = new THREE.Mesh(geom, bubbleMat);
            bubbleGroup.add(mesh);
          }}
        }});
      }}

      // Update HUD & Controls
      document.getElementById('frame-slider').value = frameIdx;
      document.getElementById('frame-badge').textContent = `Frame ${{frameIdx + 1}} / ${{FRAMES.length}} (${{f.label}})`;
      document.getElementById('hud-time').textContent = `${{f.time_ms.toFixed(2)}} ms`;
      document.getElementById('hud-vf').textContent = `${{f.vf_pct.toFixed(3)}} %`;
      document.getElementById('hud-bubbles').textContent = f.bubble_count;
      document.getElementById('hud-gas-vol').textContent = `${{f.gas_vol_mm3.toFixed(2)}} mm³`;
    }}

    // Initial Frame Render
    renderFrame(0);

    // 6. Playback Engine
    let isPlaying = false;
    let currentFrame = 0;
    let playbackSpeed = 1.0;
    const baseFPS = 15.0; // Slow-mo default
    let lastTime = performance.now();
    let accumulatedTime = 0;

    function togglePlay() {{
      isPlaying = !isPlaying;
      document.getElementById('play-icon').textContent = isPlaying ? "⏸" : "▶";
      document.getElementById('play-text').textContent = isPlaying ? "Pause" : "Play Flow";
      if (isPlaying) lastTime = performance.now();
    }}

    function stepFrame(delta) {{
      if (isPlaying) togglePlay();
      currentFrame = (currentFrame + delta + FRAMES.length) % FRAMES.length;
      renderFrame(currentFrame);
    }}

    function seekFrame(idx) {{
      currentFrame = Math.max(0, Math.min(FRAMES.length - 1, idx));
      renderFrame(currentFrame);
    }}

    function changeSpeed(val) {{
      playbackSpeed = parseFloat(val);
    }}

    document.getElementById('frame-slider').addEventListener('input', (e) => {{
      seekFrame(parseInt(e.target.value));
    }});

    // Camera Views
    function setCameraView(preset) {{
      controls.reset();
      controls.target.set(L_m * 0.5, 0, 0);
      if (preset === 'side') {{
        camera.position.set(L_m * 0.5, 0, R_m * 3.8);
      }} else if (preset === 'top') {{
        camera.position.set(L_m * 0.5, R_m * 3.8, 0.0001);
      }} else if (preset === 'front') {{
        camera.position.set(-R_m * 1.5, 0, 0);
      }} else if (preset === 'iso') {{
        camera.position.set(L_m * 0.5 + R_m * 1.8, R_m * 1.8, R_m * 3.2);
      }}
      controls.update();
    }}

    // Theme Toggle
    let isDark = true;
    function toggleTheme() {{
      isDark = !isDark;
      document.body.classList.toggle('theme-light', !isDark);
      document.getElementById('theme-btn').textContent = isDark ? "☀️ White Mode" : "🌙 Dark Mode";
      scene.background.setHex(isDark ? 0x0a0d14 : 0xf1f5f9);
      pipeWireframe.material.color.setHex(isDark ? 0x1a4f94 : 0x002277);
      ambientLight.intensity = isDark ? 0.85 : 0.95;
    }}

    let showPipe = true;
    function togglePipe() {{
      showPipe = !showPipe;
      pipeWireframe.visible = showPipe;
      pipeSheath.visible = showPipe;
      document.getElementById('pipe-toggle-btn').textContent = showPipe ? "Hide Pipe" : "Show Pipe";
    }}

    // Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {{
      if (e.code === 'Space') {{
        e.preventDefault();
        togglePlay();
      }} else if (e.code === 'ArrowRight') {{
        e.preventDefault();
        stepFrame(1);
      }} else if (e.code === 'ArrowLeft') {{
        e.preventDefault();
        stepFrame(-1);
      }}
    }});

    // Render & Animation Loop
    function animate(time) {{
      requestAnimationFrame(animate);
      controls.update();

      if (isPlaying) {{
        const delta = (time - lastTime) / 1000.0;
        lastTime = time;
        accumulatedTime += delta;
        const frameInterval = 1.0 / (baseFPS * playbackSpeed);

        if (accumulatedTime >= frameInterval) {{
          const advanceFrames = Math.floor(accumulatedTime / frameInterval);
          currentFrame = (currentFrame + advanceFrames) % FRAMES.length;
          renderFrame(currentFrame);
          accumulatedTime %= frameInterval;
        }}
      }} else {{
        lastTime = time;
      }}

      renderer.render(scene, camera);
    }}
    animate(performance.now());

    window.addEventListener('resize', () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="3D Animated Flow Reconstruction Pipeline (SI Units)"
    )
    parser.add_argument(
        "--masks",
        "-m",
        type=Path,
        required=True,
        help="Directory containing binary masks (*.png)",
    )
    parser.add_argument(
        "--output-html",
        "-o",
        type=Path,
        default=Path("results/3d_flow_animation.html"),
        help="Output standalone HTML interactive 3D player",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=5.936412e-05,
        help="Optical pixel scale in meters/pixel (SI, default: 5.936412e-05)",
    )
    parser.add_argument(
        "--diameter",
        type=float,
        default=0.020,
        help="Channel inner diameter in meters (SI, default: 0.020)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2000.0,
        help="Camera high-speed acquisition rate in Hz (default: 2000.0)",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Start frame index (default: 0)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Maximum frames to reconstruct for the 3D flow video (default: 100)",
    )
    parser.add_argument(
        "--stride",
        "-s",
        type=int,
        default=1,
        help="Frame stride (default: 1 for smooth continuous flow)",
    )
    parser.add_argument(
        "--slug-step",
        type=int,
        default=6,
        help="Sampling step for 3D slug mesh grid (default: 6 px)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="3D Multi-Phase Bubble Flow Animation",
        help="Header title for the 3D HTML viewer",
    )
    parser.add_argument(
        "--calib-file",
        "-c",
        type=Path,
        default=None,
        help="Path to calibration file (*calibration.txt) to automatically resolve pixel size, pipe diameter, and vertical channel centerline",
    )
    parser.add_argument(
        "--y-center",
        type=float,
        default=None,
        help="Explicit channel centerline vertical coordinate in pixels (defaults to mask center or calibrated midpoint)",
    )

    args = parser.parse_args()

    process_sequence_3d(
        mask_dir=args.masks,
        output_html_path=args.output_html,
        pixel_size_m=args.pixel_size,
        pipe_dia_m=args.diameter,
        fps=args.fps,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        stride=args.stride,
        slug_mesh_step=args.slug_step,
        title=args.title,
        calib_file=args.calib_file,
        y_center_px=args.y_center,
    )


if __name__ == "__main__":
    main()
