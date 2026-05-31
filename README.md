# Gallbladder Disease Image Classification Benchmark

This repository benchmarks lightweight convolutional neural network classifiers for gallbladder disease image classification on Apple Silicon using PyTorch, uv, and the MPS backend.

The goal is to compare practical medical-image classifiers under the same training, validation, testing, and inference protocol. The project intentionally avoids very large models such as VGG and focuses on efficient architectures that are realistic to train on a MacBook Pro with unified memory.

## What This Creates

The benchmark trains 10 models on 224x224 RGB images:

- `mobilenet_v3_small`
- `mobilenet_v3_large`
- `efficientnet_b0`
- `efficientnet_b1`
- `densenet121`
- `densenet169`
- `resnet18`
- `resnet34`
- `squeezenet1_1`
- `shufflenet_v2_x1_0`

For every model, the pipeline records training behavior, validation performance, final test-set classification quality, parameter count, model size estimate, training time, and inference latency/throughput.

## Dataset Layout

The script expects an ImageFolder-like dataset where each top-level folder is a class. Nested image folders are supported.

```text
Gallblader Diseases Dataset/
  1Gallstones/
  2Abdomen and retroperitoneum/
  3cholecystitis/
  ...
```

The local dataset used during setup contained 9 classes and 13,872 images. The dataset itself is not committed to this public repository.

## Methodology

The benchmark uses the same protocol for every model:

- Image size: 224x224
- Input mode: RGB
- Split: stratified train/validation/test
- Default split sizes: 70% train, 15% validation, 15% test
- Epochs: 50 by default
- Optimizer: AdamW
- Learning rate: `3e-4`
- Weight decay: `1e-4`
- Scheduler: cosine annealing
- Pretraining: torchvision ImageNet weights by default
- Class imbalance handling: weighted random sampler on the training split
- Best checkpoint selection: highest validation macro F1
- Final evaluation: best checkpoint on held-out test split
- Inference benchmark: warmup batches followed by measured test batches

Apple Silicon acceleration follows the official PyTorch MPS pattern: check `torch.backends.mps.is_available()`, select `torch.device("mps")`, move models and tensors to MPS, synchronize for timing, and report `torch.mps` memory statistics.

## Metrics Collected

Per model:

- Total parameters
- Trainable parameters
- Non-trainable parameters
- Parameter memory estimate
- Buffer memory estimate
- Model state memory estimate
- Training time in seconds
- Per-epoch train loss, accuracy, macro F1
- Per-epoch validation loss, accuracy, macro F1
- Per-epoch learning rate
- Per-epoch wall time
- MPS memory stats when available
- Test accuracy
- Test balanced accuracy
- Test macro F1
- Test weighted F1
- Matthews correlation coefficient
- Top-2 accuracy
- Top-3 accuracy
- Macro ROC-AUC one-vs-rest
- Weighted ROC-AUC one-vs-rest
- Macro average precision
- Per-class precision, recall, F1, and support
- Inference images/second
- Mean, p50, p95, and standard deviation latency per image

Overall benchmark:

- Dataset manifest CSVs
- Stratified split CSVs
- Class counts
- Model parameter comparison CSV
- Model metric comparison CSV
- Benchmark summary CSV
- Comparison plots and rank heatmaps

## Outputs

Each run writes to:

```text
outputs/runs/<timestamp>/
```

Overall files:

- `config.json`
- `class_to_idx.json`
- `all_files.csv`
- `train_files.csv`
- `val_files.csv`
- `test_files.csv`
- `class_counts.csv`
- `metrics_summary.csv`
- `benchmark_summary.csv`
- `model_parameters.csv`
- `model_comparison_metrics.png`
- `model_metric_ranks.png`
- `accuracy_size_throughput.png`

Per-model files:

- `best.pt`
- `history.csv`
- `metrics.json`
- `classification_report.csv`
- `test_predictions.csv`
- `test_probabilities.csv`
- `inference_benchmark_batches.csv`
- `learning_curves.png`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `roc_curves.png`
- `precision_recall_curves.png`

## Full Benchmark

```bash
uv run python scripts/train_all.py --device mps --epochs 50 --image-size 224 --batch-size 32 --num-workers 4
```

If memory pressure is high:

```bash
uv run python scripts/train_all.py --device mps --epochs 50 --image-size 224 --batch-size 16 --num-workers 4
```

If a PyTorch operation lacks MPS support on your installed version:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/train_all.py --device mps --epochs 50 --image-size 224 --batch-size 32 --num-workers 4
```

## Smoke Test

```bash
uv run python scripts/train_all.py --device mps --epochs 1 --models mobilenet_v3_small --limit-per-class 4 --batch-size 8 --num-workers 0 --val-size 0.25 --test-size 0.25 --benchmark-batches 2 --output-dir outputs/smoke
```

## Notes

This benchmark is intended for reproducible model comparison, not clinical deployment. Any medical use would require dataset documentation, leakage checks, external validation, calibration analysis, and review by qualified clinical experts.
