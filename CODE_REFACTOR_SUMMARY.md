# Phase 1 Refactor Report (Week 47)

Scope: `.py` scripts, their layout, and the `README.md` files. No logs, datasets, result folders, the report or the article were deleted or edited. The research articles in `0_docs/initial_research/` and `0_docs/papers/` were used as the mathematical reference.

**Status:** `ruff check .` and `ruff format --check .` pass, all 42 script/test commands in the verification run exit with code 0, and 6 runs without a calibration stop with exit code 1 as intended.

---

## 1. Structural changes

### New layout

```text
main.py, pipeline.py     entry points (unchanged location)
image_scripts/           Stage 0: optical preprocessing
model_training/          Stages 1–6: the numbered pipeline only
visualization/           renderers, plots, schematics            (new)
tools/                   evaluation, benchmarks, dataset utilities (new)
tests/                   unit and physics tests
```

### File moves (all with `git mv`, history preserved)

| From | To |
| :--- | :--- |
| `model_training/render_3d_flow.py` | `visualization/render_3d_flow.py` |
| `model_training/render_3d_flow_animation.py` | `visualization/render_3d_flow_animation.py` |
| `model_training/visualize_flow_animation.py` | `visualization/visualize_flow_animation.py` |
| `model_training/plot_void_fraction_timeline.py` | `visualization/plot_void_fraction_timeline.py` |
| `model_training/generate_methodology_schematics.py` | `visualization/generate_methodology_schematics.py` |
| `0_docs/assets/scripts/plot_resnet_*.py` (3 files) | `visualization/plot_resnet_*.py` |
| `model_training/evaluate_unet_metrics.py` | `tools/evaluate_unet_metrics.py` |
| `model_training/benchmark_inference_speed.py` | `tools/benchmark_inference_speed.py` |
| `model_training/create_dataset_2_from_divergente.py` | `tools/create_dataset_2_from_divergente.py` |
| `image_scripts/test_single_mask.py` | `image_scripts/preview_single_mask.py` (it is a preview tool, not a test) |

### DfD consolidation

`model_training/test_dfd_ground_truth.py` (2,576 lines, untracked) is merged into `model_training/6_depth_from_defocus.py`. The two files shared a near-identical physics core; the merged module keeps one copy and adds everything that only the test script had:

- Label Studio ingestion (brush RLE, polygon, ellipse, rectangle) and `--json` mode
- calibration auto-detection from `0_*calibration.txt`
- 3D spatial figure, volumetric figure and interactive WebGL viewer (`--render-3d`)
- frame sampler for annotation (`--prepare-tasks`)
- label decluttering in the 2-panel diagnostic (`--min-label-area`)

Archived, not deleted: `scratch/archive/test_dfd_ground_truth.py` (the file was never tracked by git, so deleting it would have been unrecoverable). The root symlink `test_dfd_ground_truth.py` pointing at it was removed.

`main.py dfd` now forwards its options to the module's own parser, so there is a single definition of the DfD CLI. The old form `main.py dfd -i … -m … -o results/dfd_results.csv` still works.

### Imports, paths and CLIs

- `main.py`, `model_training/4_*`, `5_*`, `6_*`, the tests and `render_3d_flow_animation.py` import from `visualization.*`. `pyproject.toml` lists `visualization` and `tools` as first-party for import sorting.
- Scripts that import a sibling package now run directly (`python folder/script.py`) as well as through `main.py`. Before this, `0_roi_crop.py` (headless crop) and `render_3d_flow_animation.py` failed with `ModuleNotFoundError` when run on their own.
- New CLIs where paths were hard-coded:
  - `visualization/generate_methodology_schematics.py` and the three `plot_resnet_*.py`: `--output-dir` (default unchanged: `ctf2_article/figures/schematics`). The `plot_resnet_*` bodies are now inside `main()`.
  - `tools/benchmark_inference_speed.py`: `--input`, `--model`, `--output-dir`, `--max-frames`, `--runs`, `--quick`.
  - `tools/create_dataset_2_from_divergente.py`: `--src-images`, `--src-masks`, `--dest`, `--diagnostic`.
- Removed from the `plot_resnet_*` scripts: a hard-coded copy of each figure into `/Users/…/.gemini/antigravity/brain/…`, which only worked on one machine.
- `benchmark_inference_speed.py` writes its temporary masks to a system temp directory instead of `scratch/benchmark_temp_out`.

---

## 2. Mathematical and SI-unit corrections

Reference for DfD: Rao, Sharma, Basu & Tropea (2024), *Experiments in Fluids* 65:55 (`paper_depth_from_defocus_calibration_free_rao2024.pdf`).

### Corrections that change results

| # | Where | Before | After | Reference |
| :-: | :--- | :--- | :--- | :--- |
| 1 | `DFDModel` inversion, out-of-range gradients | The interpolator's fill values were swapped: a gradient **weaker** than the table minimum returned $\tilde\sigma = 0.005$ (perfectly sharp, placed at the front wall and flagged reliable); a gradient **steeper** than the table maximum returned $\tilde\sigma = 0.35$ | Weak gradients map to the blurred end, steep gradients to the sharp end | $\tilde G = f_2(\tilde\sigma)$ decreases monotonically (Eq. 9, Fig. 5a) |
| 2 | `unreliable_defocus` status | Unreachable: the table stopped at $\tilde\sigma = 0.35$, so the test $\tilde\sigma > 0.35$ could never be true | Table extended with the same node spacing to $\tilde\sigma \approx 0.397$; measurements beyond the limit are now rejected | "results beyond this are disregarded", $\tilde\sigma_c = 0.35$ (Eq. 13) |
| 3 | Slug threshold in `analyze_frame_dfd` | Two defaults: $\pi R_{tube}^2$ px (old stage 6) and 5000 px (test script) | 5000 px, the dual-regime threshold of `render_3d_flow.py` | internal consistency |
| 4 | DfD curve schematic (`generate_dfd_curves_regularization`) | Placeholder curves $\tilde G = 0.35/(\tilde\sigma^{0.85}+0.01)$ and $\tilde\rho = 0.5 + 0.18\,\tilde\sigma^{0.65}$; the second **increases** with blur | Exact $f_1$, $f_2$ from the same lookup table the pipeline inverts; $\tilde\rho_t$ falls from 0.5 to 0.3211 at $\tilde\sigma_c$ | Eqs. 7, 9, 10; Figs. 3b, 5 |
| 5 | Pixel-size defaults | Three hidden fallbacks: `8.6e-5` (stage 4 and 5 CLIs), `6.7e-5` (function defaults), `6.1e-5` / `5.94e-5` elsewhere | **Removed.** The optical scale comes from a `0_*calibration.txt` next to the input or from `--pixel-size`; otherwise the command stops with an error. Library functions take `pixel_size_m` as a required argument | every acquisition has its own calibration |

Effect of 1–3 on `dataset/` (11 frames, 1,579 entities): near-wall regularised 51 → 26, unreliable 0 → 9, slugs 207 → 224, valid 1,321 → 1,320. Mean reliable diameter 0.745 → 0.602 mm; maximum reliable fluid depth 9.04 → 2.67 mm. The earlier values included large blobs and out-of-range bubbles that should not have been counted as reliable. **DfD CSVs and figures produced before this change will not reproduce exactly.**

### Verified as correct (no change needed)

- Blurred-disc integral (Eq. 7) including the exponentially scaled Bessel evaluation. The lookup table reproduces the article's published limit values: $f_1(0.35) = 0.3211$ and $f_2(0.35) = 0.2501$. This is now a unit test.
- $d_0 = r_t/\tilde\rho_t$, $\sigma = \tilde\sigma\,d_0$, $r_t = \sqrt{A/\pi}$.
- Depth units: $\lvert\Delta z\rvert\,[\text{m}] = \sigma\,[\text{px}] / \beta\,[\text{px/m}]$.
- Void fraction: the implementation is the Steiner form of the Rouhani–Axelsson drift-flux model, $C_0 = 1 + 0.12(1-x)$ with the $1.18\,(1-x)\,[g\sigma(\rho_l-\rho_v)]^{0.25}/(G\rho_l^{0.5})$ drift term, as written in Rodrigo Branco's MSc thesis. (The 1970 paper itself uses a constant $C = 1.12$.)
- Kinematics: $v = d\,s_{px}/\Delta t$ with $\Delta t = 1/\text{fps}$.

### Documentation corrections

- Citation: the module cited "Experiments in Fluids 65(9), 136"; the article is 65:55.
- $\beta$ units: the docstring said "meters/pixel or pixels/meter"; it is px/m.
- The near-focus smoothstep regulariser was attributed to Rao et al.; the article only states that depth is unreliable for $\tilde\sigma < 0.05$. It is now labelled as a project addition.

### Open points (need a decision, not changed)

1. **Focal-plane convention — settled.** The code takes $\sigma/\beta$ as the depth from the local front fluid boundary. This is the actual experimental setup (the focus is set on the front wall, the only position known with certainty), so it stays the default. `--focal-plane-z` remains as an optional planar alternative and is covered by a test.
2. **Measurement of $r_t$ and $\tilde G$.** The article takes both at the $g_t = 0.5$ iso-contour (thin strip $\pm 0.005$, 5×5 sub-pixel interpolation). The code uses the segmentation-mask area and a trimmed mean of the Sobel gradient on the mask contour. The article's pixelation correction $\gamma$ (Eq. 11) is not applied. Documented in the module docstring.
3. **$\beta = 2000$ px/m is a placeholder.** No in-situ calibration exists yet, so absolute depths are indicative.
4. **Tube geometry** (20 mm inner diameter, 3 mm wall) and the **water-at-100 °C defaults** in the drift-flux function could not be confirmed from the articles' extracted text.
5. **Calibration matching — fixed.** The DfD auto-detection used to fall back to the first calibration file it found even when it was recorded for another image size (for `dataset_2*` that silently applied the 891×328 calibration to 920×417 frames). It now only accepts a file whose image size matches; otherwise `--pixel-size` is required.

---

## 3. Verification log

Environment: Python 3.14.0, ruff 0.16.2, macOS (Apple Silicon). Every command wrote its outputs to a scratch folder outside the repository; `git status` was identical before and after the run. GUI-only entry points were checked with `--help`. Runs on `dataset_2/`, which has no calibration file, pass `--pixel-size 5.175358e-05` explicitly. The last block before the tests feeds masks with no calibration and no `--pixel-size`; exit code 1 is the expected result.

```text
ruff check .                                               exit=0
ruff format --check .                                      exit=0
image_scripts/0_roi_crop.py                                exit=0
image_scripts/1_background_subtraction.py                  exit=0
image_scripts/2_generate_masks.py                          exit=0
image_scripts/3_create_preannotations.py                   exit=0
image_scripts/calibrate_pixel_size.py (--help, GUI)        exit=0
image_scripts/preview_single_mask.py                       exit=0
model_training/1_convert_annotations.py                    exit=0
model_training/2_train_u_net.py (1 epoch)                  exit=0
model_training/3_batch_inference.py                        exit=0
model_training/4_void_fraction_reconstruct.py              exit=0
model_training/5_track_bubbles_kinematics.py               exit=0
model_training/6_depth_from_defocus.py (folders)           exit=0
model_training/6_depth_from_defocus.py (--json)            exit=0
model_training/6_depth_from_defocus.py (--prepare-tasks)   exit=0
visualization/render_3d_flow.py                            exit=0
visualization/render_3d_flow_animation.py                  exit=0
visualization/visualize_flow_animation.py                  exit=0
visualization/plot_void_fraction_timeline.py               exit=0
visualization/generate_methodology_schematics.py           exit=0
visualization/plot_resnet_gradient_highway.py              exit=0
visualization/plot_resnet_intra_stage_highway.py           exit=0
visualization/plot_resnet_unet_architecture.py             exit=0
tools/benchmark_inference_speed.py (--quick)               exit=0
tools/evaluate_unet_metrics.py                             exit=0
tools/create_dataset_2_from_divergente.py                  exit=0
pipeline.py prepare (--help)                               exit=0
pipeline.py train (--help)                                 exit=0
main.py void-fraction                                      exit=0
main.py track                                              exit=0
main.py animate-flow                                       exit=0
main.py animate-3d                                         exit=0
main.py dfd                                                exit=0
main.py render                                             exit=0
main.py plot-vf                                            exit=0
main.py gen-masks                                          exit=0
main.py bg-sub                                             exit=0
main.py crop                                               exit=0
main.py calibrate (--help, GUI)                            exit=0
no calibration: main.py void-fraction                      exit=1 (expected 1)
no calibration: main.py track                              exit=1 (expected 1)
no calibration: main.py dfd                                exit=1 (expected 1)
no calibration: render_3d_flow.py                          exit=1 (expected 1)
no calibration: render_3d_flow_animation.py                exit=1 (expected 1)
no calibration: visualize_flow_animation.py                exit=1 (expected 1)
tests/test_3d_render_and_si.py                             exit=0
tests/test_calibration.py                                  exit=0
tests/test_flow_animation.py                               exit=0
tests/test_plot_void_fraction_timeline.py                  exit=0
```

Tests: 40 `unittest` cases plus 13 physics checks (53 total), all passing. `test_depth_from_defocus_near_wall_regularization` was already failing before this refactor (it compared the absolute depth, which includes the 3 mm wall offset, against 0.1 mm); it now checks the fluid depth. Four DfD tests are new: the article's reference values, out-of-range gradients, rejection beyond $\tilde\sigma_c$, and the planar focal plane.

Not covered: `pipeline.py prepare` / `train` were checked with `--help` only (the underlying stage scripts ran for real), and the interactive calibration and crop GUIs were not driven. `scratch/` is git-ignored, excluded from ruff, and was not touched apart from receiving the archive.

---

## 4. Presentation content integrated into the READMEs

Source: `ctf2_presentation/w45_ctf2_presentation_final.pdf` (3 November 2026).

### Selling points (root `README.md`, "Key Results at a Glance")

| Presentation slide | Point |
| :--- | :--- |
| Slide 2 | Non-intrusive optical extraction of void fraction, velocity and 3D geometry at 2,000 FPS |
| Slide 10 | Training time 49.6 min → 1.17 min; 97 % F1; −99 % false negatives; 26× faster slug reconstruction; 6× faster batch analysis |
| Slide 9 | Single-camera DfD recovers the depth a monocular view loses |
| Slide 11 | Background removal: fewer false positives, higher F1 and IoU |

Two figures differ from the slides because the README follows the repository's own data:

- Training speed-up is written as **42×** (49.6 / 1.17 = 42.4); the slide says 43×.
- The background-removal table uses `models/model_comparison_metrics.csv`: F1 97.9 %, IoU +3.2 pt, false positives −59 %, false negatives −4 %. The slide states 98.2 %, +4 %, −60 % and −25 %. The −25 % false-negative figure in particular is not reproduced by the CSV.

### Figures (copied into `0_docs/assets/`, the tracked image folder)

| Asset | Source | Used in |
| :--- | :--- | :--- |
| `dfd_vs_nondfd_comparison.png` | `results/w44_results_3d/` (slide 9) | root, `model_training/` |
| `matlab_vs_python_head_to_head.png` | `models/head_to_head_options/option4_peak_performance_frame03.png` | root |
| `bg_vs_nobg_elongated_slug.png` | `models/head_to_head_options/bg_vs_nobg_frame48_elongated_slug.png` (slide 11) | root |
| `void_fraction_timeline_2791.png` | `results/w43_full_run_2k/` (slide 7) | root, `model_training/` |
| `bubble_velocity_distribution_2791.png` | `results/w43_full_run_2k/` (slide 8) | root |
| `dfd_diagnostic_frame_0071.png` | regenerated from `images/4rd_run/dfd_test` (DfD acquisition) | `model_training/` |

The three largest were downscaled to 2,000 px on the long side.

The two DfD figures were regenerated with the corrected code, so they differ from the slide versions. In `dfd_vs_nondfd_comparison.png` (frame 0071, 67 entities) the 18 bubbles previously drawn as "near-wall regularised" are now marked as rejected ($\tilde\sigma > 0.35$): their edge gradient was below the table range and the old inversion returned the sharp end instead of the blurred end. Valid bubbles (45) and slugs (4) are unchanged. `dfd_diagnostic_frame_0071.png` is the standard 2-panel output for the same DfD frame; it replaces an earlier diagnostic of `dataset` frame 0016, which is an in-focus (non-DfD) acquisition and was not a meaningful DfD example. The originals in `results/` are untouched.

### README files

- `README.md`: results summary, presentation figures, repository layout, DfD recipes, test instructions.
- `model_training/README.md`: new layout, DfD method table with equation numbers, three input modes, flag reference.
- `image_scripts/README.md`: renamed preview tool.
- `visualization/README.md`, `tools/README.md`: new.

All image links and script paths in these files were checked against the file system.

---

## 5. Left for the next phase

- Root-level debug images (`debug_*.png`, `sample_*.png`, `cross_test_*.png`, `zoom_*.png`, …) and `texput.log` are untouched.
- `scratch/` still holds older copies of the `plot_resnet_*` scripts and one-off analysis scripts.
- `tools/evaluate_unet_metrics.py --compare` writes a temporary CSV to `/tmp/eval_tmp.csv`.
- Nothing is committed; the changes are in the working tree.
