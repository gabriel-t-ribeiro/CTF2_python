# Tools (`tools`)

Evaluation, benchmarking and dataset utilities that sit outside the numbered pipeline.

| Script | Purpose |
| :--- | :--- |
| `evaluate_unet_metrics.py` | Accuracy, precision, recall, F1/Dice, IoU and MCC of a trained model on a labelled dataset; `--compare` runs the full model × dataset matrix |
| `benchmark_inference_speed.py` | Forward-pass and end-to-end inference timing against a MATLAB-equivalent U-Net |
| `create_dataset_2_from_divergente.py` | Rebuilds `dataset_2` by pairing raw frames with pixel-aligned masks |

## Usage

```bash
# Metrics for one model on one dataset
python tools/evaluate_unet_metrics.py -m models/run_3_unet.pth -d test_dataset \
  -o results/eval_metrics.csv

# Full comparison matrix (writes models/model_comparison_metrics.csv)
python tools/evaluate_unet_metrics.py --compare -o models/model_comparison_metrics.csv

# Benchmark: full run, or a few-second smoke run with --quick
python tools/benchmark_inference_speed.py --output-dir models
python tools/benchmark_inference_speed.py --quick -i test_dataset/images -o results/benchmark

# Rebuild dataset_2 (needs images/background1/divergente_40kHz, which is not tracked)
python tools/create_dataset_2_from_divergente.py --dest dataset_2
```

`create_dataset_2_from_divergente.py` overwrites the destination dataset; point `--dest` somewhere else to inspect the result first.

Results of the benchmark and comparison runs are summarised in [`../models/model_comparison_report.md`](../models/model_comparison_report.md).
