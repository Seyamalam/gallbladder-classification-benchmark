#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm.auto import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_MODELS = [
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "efficientnet_b1",
    "densenet121",
    "densenet169",
    "resnet18",
    "resnet34",
    "squeezenet1_1",
    "shufflenet_v2_x1_0",
]


@dataclass
class RunConfig:
    data_dir: str
    output_dir: str
    image_size: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    seed: int
    val_size: float
    test_size: float
    num_workers: int
    pretrained: bool
    weighted_sampler: bool
    limit_per_class: int | None
    models: list[str]
    device: str
    torch_threads: int | None
    verbose: bool
    benchmark_warmup_batches: int
    benchmark_batches: int


class ImageFrameDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, class_to_idx: dict[str, int], transform=None):
        self.frame = frame.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        with Image.open(row.path) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.class_to_idx[row.label]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        if requested == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                raise RuntimeError("MPS requested, but this PyTorch build does not include MPS support.")
            raise RuntimeError("MPS requested, but no MPS-capable Apple GPU/runtime is available.")
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def format_bytes(value: int | float | None) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(value) < 1024.0:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}PB"


def mps_memory_stats(device: torch.device) -> dict[str, str]:
    if device.type != "mps" or not torch.backends.mps.is_available():
        return {}
    stats = {
        "mps_current": format_bytes(torch.mps.current_allocated_memory()),
        "mps_driver": format_bytes(torch.mps.driver_allocated_memory()),
    }
    try:
        stats["mps_recommended_max"] = format_bytes(torch.mps.recommended_max_memory())
    except AttributeError:
        pass
    return stats


def sync_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def print_runtime_report(device: torch.device, config: RunConfig) -> None:
    print("\nRuntime")
    print(f"  platform: {platform.platform()}")
    print(f"  python: {platform.python_version()}")
    print(f"  torch: {torch.__version__}")
    print(f"  torchvision: {models.__name__.split('.')[0]}")
    print(f"  cpu threads: {torch.get_num_threads()}")
    print(f"  selected device: {device}")
    print(f"  mps built: {torch.backends.mps.is_built()}")
    print(f"  mps available: {torch.backends.mps.is_available()}")
    if device.type == "mps":
        print(f"  mps devices: {torch.mps.device_count()}")
        for key, value in mps_memory_stats(device).items():
            print(f"  {key}: {value}")
    print(f"  PYTORCH_ENABLE_MPS_FALLBACK: {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '<unset>')}")
    print(f"  pretrained weights: {config.pretrained}")
    print(f"  weighted sampler: {config.weighted_sampler}")


def scan_dataset(data_dir: Path, limit_per_class: int | None, seed: int) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    for class_dir in class_dirs:
        images = sorted(
            p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if limit_per_class is not None:
            rng = random.Random(seed)
            rng.shuffle(images)
            images = sorted(images[:limit_per_class])
        for path in images:
            rows.append({"path": str(path), "label": class_dir.name})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No images found under {data_dir}")
    classes = sorted(frame.label.unique().tolist())
    return frame, classes


def split_frame(frame: pd.DataFrame, val_size: float, test_size: float, seed: int):
    num_classes = frame.label.nunique()
    if round(len(frame) * test_size) < num_classes:
        raise ValueError(
            f"test split is too small for stratification: {len(frame)} images, "
            f"{num_classes} classes, test_size={test_size}. Increase --test-size or --limit-per-class."
        )
    if round(len(frame) * val_size) < num_classes:
        raise ValueError(
            f"validation split is too small for stratification: {len(frame)} images, "
            f"{num_classes} classes, val_size={val_size}. Increase --val-size or --limit-per-class."
        )
    train_val, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        stratify=frame.label,
    )
    val_fraction = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        random_state=seed,
        stratify=train_val.label,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def build_transforms(image_size: int):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tfms, eval_tfms


def make_loader(dataset, batch_size, num_workers, shuffle=False, sampler=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )


def make_sampler(frame: pd.DataFrame, class_to_idx: dict[str, int]):
    counts = frame.label.map(class_to_idx).value_counts().to_dict()
    weights = frame.label.map(lambda label: 1.0 / counts[class_to_idx[label]]).to_numpy(copy=True)
    return WeightedRandomSampler(torch.DoubleTensor(weights), len(weights), replacement=True)


def replace_classifier(model: nn.Module, model_name: str, num_classes: int) -> nn.Module:
    if model_name.startswith("mobilenet") or model_name.startswith("efficientnet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name.startswith("densenet"):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name.startswith("resnet") or model_name.startswith("shufflenet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name.startswith("squeezenet"):
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
        model.num_classes = num_classes
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


def build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    ctor = getattr(models, model_name)
    if pretrained:
        weights_enum = models.get_model_weights(model_name)
        model = ctor(weights=weights_enum.DEFAULT)
    else:
        model = ctor(weights=None)
    return replace_classifier(model, model_name, num_classes)


def model_profile(model: nn.Module) -> dict[str, int | float]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": total_params - trainable_params,
        "parameter_mb": param_bytes / (1024**2),
        "buffer_mb": buffer_bytes / (1024**2),
        "model_state_mb": (param_bytes + buffer_bytes) / (1024**2),
    }


def run_epoch(model, loader, criterion, device, optimizer=None, verbose: bool = True, model_name: str = ""):
    is_train = optimizer is not None
    model.train(is_train)
    running_loss = 0.0
    y_true, y_pred = [], []
    seen = 0
    correct = 0
    desc = f"{model_name} {'train' if is_train else 'val'}".strip()
    progress = tqdm(loader, leave=False, desc=desc, dynamic_ncols=True)
    start = time.perf_counter()
    for batch_idx, (images, labels) in enumerate(progress, start=1):
        images = images.to(device)
        labels = labels.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()
        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(1)
        seen += labels.size(0)
        correct += (preds == labels).sum().item()
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())
        if verbose:
            elapsed = max(time.perf_counter() - start, 1e-9)
            postfix = {
                "loss": f"{running_loss / seen:.4f}",
                "acc": f"{correct / seen:.4f}",
                "img/s": f"{seen / elapsed:.1f}",
            }
            if device.type == "mps":
                postfix["mps_mem"] = format_bytes(torch.mps.current_allocated_memory())
            progress.set_postfix(postfix)
    avg_loss = running_loss / max(len(loader.dataset), 1)
    return avg_loss, accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro", zero_division=0)


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for images, labels in tqdm(loader, leave=False, desc="test predict", dynamic_ncols=True):
        logits = model(images.to(device))
        probs = torch.softmax(logits, dim=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(probs.argmax(1).cpu().numpy().tolist())
        y_prob.extend(probs.cpu().numpy().tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


@torch.inference_mode()
def benchmark_inference(model, loader, device, warmup_batches: int, benchmark_batches: int, verbose: bool):
    model.eval()
    iterator = iter(loader)
    warmup_images = 0
    for _ in range(max(warmup_batches, 0)):
        try:
            images, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, _ = next(iterator)
        _ = model(images.to(device))
        warmup_images += images.size(0)
    sync_device(device)

    batch_rows = []
    measured_images = 0
    measured_seconds = 0.0
    progress = tqdm(
        range(max(benchmark_batches, 1)),
        leave=False,
        desc="inference benchmark",
        dynamic_ncols=True,
        disable=not verbose,
    )
    for batch_idx in progress:
        try:
            images, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, _ = next(iterator)
        images = images.to(device)
        sync_device(device)
        started = time.perf_counter()
        _ = model(images)
        sync_device(device)
        elapsed = time.perf_counter() - started
        batch_size = images.size(0)
        measured_images += batch_size
        measured_seconds += elapsed
        batch_rows.append(
            {
                "batch_index": batch_idx + 1,
                "batch_size": batch_size,
                "seconds": elapsed,
                "images_per_second": batch_size / elapsed if elapsed > 0 else math.nan,
                "latency_ms_per_image": elapsed * 1000 / batch_size if batch_size else math.nan,
            }
        )
        if verbose:
            progress.set_postfix(
                {
                    "lat/img_ms": f"{batch_rows[-1]['latency_ms_per_image']:.2f}",
                    "img/s": f"{batch_rows[-1]['images_per_second']:.1f}",
                }
            )

    latencies = np.array([row["latency_ms_per_image"] for row in batch_rows], dtype=float)
    throughput = measured_images / measured_seconds if measured_seconds > 0 else math.nan
    summary = {
        "inference_warmup_batches": warmup_batches,
        "inference_warmup_images": warmup_images,
        "inference_benchmark_batches": len(batch_rows),
        "inference_benchmark_images": measured_images,
        "inference_total_seconds": measured_seconds,
        "inference_images_per_second": throughput,
        "inference_latency_ms_per_image_mean": float(np.nanmean(latencies)),
        "inference_latency_ms_per_image_p50": float(np.nanpercentile(latencies, 50)),
        "inference_latency_ms_per_image_p95": float(np.nanpercentile(latencies, 95)),
        "inference_latency_ms_per_image_std": float(np.nanstd(latencies)),
    }
    return summary, pd.DataFrame(batch_rows)


def compute_metrics(y_true, y_pred, y_prob, classes):
    labels = list(range(len(classes)))
    y_bin = label_binarize(y_true, classes=labels)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    for idx, cls in enumerate(classes):
        metrics[f"{cls}__precision"] = precision[idx]
        metrics[f"{cls}__recall"] = recall[idx]
        metrics[f"{cls}__f1"] = f1[idx]
        metrics[f"{cls}__support"] = int(support[idx])
    try:
        metrics["top2_accuracy"] = top_k_accuracy_score(y_true, y_prob, k=min(2, len(classes)), labels=labels)
        metrics["top3_accuracy"] = top_k_accuracy_score(y_true, y_prob, k=min(3, len(classes)), labels=labels)
    except ValueError:
        pass
    try:
        metrics["macro_roc_auc_ovr"] = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
        metrics["weighted_roc_auc_ovr"] = roc_auc_score(y_bin, y_prob, average="weighted", multi_class="ovr")
    except ValueError:
        metrics["macro_roc_auc_ovr"] = math.nan
        metrics["weighted_roc_auc_ovr"] = math.nan
    try:
        metrics["macro_average_precision"] = average_precision_score(y_bin, y_prob, average="macro")
    except ValueError:
        metrics["macro_average_precision"] = math.nan
    return metrics


def save_plots(model_dir: Path, history: pd.DataFrame, y_true, y_pred, y_prob, classes):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for split in ["train", "val"]:
        axes[0].plot(history.epoch, history[f"{split}_loss"], label=split)
        axes[1].plot(history.epoch, history[f"{split}_accuracy"], label=split)
        axes[2].plot(history.epoch, history[f"{split}_macro_f1"], label=split)
    axes[0].set_title("Loss")
    axes[1].set_title("Accuracy")
    axes[2].set_title("Macro F1")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(model_dir / "learning_curves.png", dpi=180)
    plt.close(fig)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(model_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    cm_norm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))), normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix")
    fig.tight_layout()
    fig.savefig(model_dir / "confusion_matrix_normalized.png", dpi=180)
    plt.close(fig)

    labels = list(range(len(classes)))
    y_bin = label_binarize(y_true, classes=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, cls in enumerate(classes):
        if y_bin[:, idx].sum() in (0, len(y_bin)):
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, idx], y_prob[:, idx])
        ax.plot(fpr, tpr, label=cls)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_title("One-vs-Rest ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(model_dir / "roc_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, cls in enumerate(classes):
        if y_bin[:, idx].sum() in (0, len(y_bin)):
            continue
        precision, recall, _ = precision_recall_curve(y_bin[:, idx], y_prob[:, idx])
        ax.plot(recall, precision, label=cls)
    ax.set_title("One-vs-Rest Precision-Recall Curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(model_dir / "precision_recall_curves.png", dpi=180)
    plt.close(fig)


def save_comparison_plots(out_dir: Path, metrics_frame: pd.DataFrame):
    if metrics_frame.empty:
        return
    sns.set_theme(style="whitegrid")
    key_cols = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_roc_auc_ovr",
        "macro_average_precision",
        "best_val_macro_f1",
        "train_seconds",
        "inference_images_per_second",
        "inference_latency_ms_per_image_mean",
        "total_params",
        "model_state_mb",
    ]
    available = [c for c in key_cols if c in metrics_frame.columns]
    plot_frame = metrics_frame.melt(id_vars="model", value_vars=available, var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(data=plot_frame, x="model", y="value", hue="metric", ax=ax)
    ax.tick_params(axis="x", labelrotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.set_title("Model Comparison")
    fig.tight_layout()
    fig.savefig(out_dir / "model_comparison_metrics.png", dpi=180)
    plt.close(fig)

    rank_cols = [c for c in available if c != "train_seconds"]
    rank_frame = metrics_frame.set_index("model")[rank_cols].rank(ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(5, len(metrics_frame) * 0.5)))
    sns.heatmap(rank_frame, annot=True, fmt=".0f", cmap="viridis_r", ax=ax)
    ax.set_title("Metric Ranks (1 = Best)")
    fig.tight_layout()
    fig.savefig(out_dir / "model_metric_ranks.png", dpi=180)
    plt.close(fig)

    scatter_cols = {"total_params", "macro_f1", "inference_images_per_second"}
    if scatter_cols.issubset(metrics_frame.columns):
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.scatterplot(
            data=metrics_frame,
            x="total_params",
            y="macro_f1",
            size="inference_images_per_second",
            hue="model",
            sizes=(80, 420),
            ax=ax,
        )
        ax.set_title("Accuracy vs Model Size vs Inference Throughput")
        ax.set_xlabel("Parameters")
        ax.set_ylabel("Test Macro F1")
        fig.tight_layout()
        fig.savefig(out_dir / "accuracy_size_throughput.png", dpi=180)
        plt.close(fig)


def train_one_model(model_name, loaders, classes, config: RunConfig, device, out_dir: Path):
    model_dir = out_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(model_name, len(classes), config.pretrained).to(device)
    profile = model_profile(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    history = []
    best_score = -1.0
    best_path = model_dir / "best.pt"
    print(f"\n=== {model_name} ===")
    print(f"parameters: total={profile['total_params']:,} trainable={profile['trainable_params']:,}")
    print(f"model state: {profile['model_state_mb']:.2f} MB")
    print(f"batches: train={len(loaders['train'])} val={len(loaders['val'])} test={len(loaders['test'])}")
    print(f"initial memory: {mps_memory_stats(device) if device.type == 'mps' else 'n/a'}")
    sync_device(device)
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        print(f"\n[{model_name}] epoch {epoch}/{config.epochs} lr={optimizer.param_groups[0]['lr']:.8f}")
        sync_device(device)
        epoch_started = time.perf_counter()
        train_loss, train_acc, train_f1 = run_epoch(
            model, loaders["train"], criterion, device, optimizer, config.verbose, model_name
        )
        sync_device(device)
        val_loss, val_acc, val_f1 = run_epoch(
            model, loaders["val"], criterion, device, None, config.verbose, model_name
        )
        sync_device(device)
        epoch_seconds = time.perf_counter() - epoch_started
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "lr": scheduler.get_last_lr()[0],
            "epoch_seconds": epoch_seconds,
        }
        if device.type == "mps":
            row.update(mps_memory_stats(device))
        history.append(row)
        print(json.dumps(row, indent=2))
        if val_f1 > best_score:
            best_score = val_f1
            torch.save({"model": model.state_dict(), "classes": classes, "config": asdict(config)}, best_path)

    train_seconds = time.perf_counter() - started
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    y_true, y_pred, y_prob = predict(model, loaders["test"], device)
    sync_device(device)
    inference_summary, inference_batches = benchmark_inference(
        model,
        loaders["test"],
        device,
        config.benchmark_warmup_batches,
        config.benchmark_batches,
        config.verbose,
    )
    print("inference benchmark:")
    print(json.dumps(inference_summary, indent=2))
    metrics = compute_metrics(y_true, y_pred, y_prob, classes)
    metrics.update(
        {
            "model": model_name,
            "best_val_macro_f1": best_score,
            "train_seconds": train_seconds,
            "epochs": config.epochs,
            "test_samples": len(y_true),
        }
    )
    metrics.update(profile)
    metrics.update(inference_summary)
    if device.type == "mps":
        metrics.update(mps_memory_stats(device))
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(model_dir / "history.csv", index=False)
    pd.DataFrame(classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)).T.to_csv(
        model_dir / "classification_report.csv"
    )
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(model_dir / "test_predictions.csv", index=False)
    pd.DataFrame(y_prob, columns=classes).to_csv(model_dir / "test_probabilities.csv", index=False)
    inference_batches.to_csv(model_dir / "inference_benchmark_batches.csv", index=False)
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_plots(model_dir, history_frame, y_true, y_pred, y_prob, classes)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="Gallblader Diseases Dataset")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--quiet-progress", action="store_true")
    parser.add_argument("--benchmark-warmup-batches", type=int, default=3)
    parser.add_argument("--benchmark-batches", type=int, default=20)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--limit-per-class", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    seed_everything(args.seed)
    out_dir = Path(args.output_dir or f"outputs/runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    config = RunConfig(
        data_dir=args.data_dir,
        output_dir=str(out_dir),
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        val_size=args.val_size,
        test_size=args.test_size,
        num_workers=args.num_workers,
        pretrained=not args.no_pretrained,
        weighted_sampler=not args.no_weighted_sampler,
        limit_per_class=args.limit_per_class,
        models=args.models,
        device=str(device),
        torch_threads=args.torch_threads,
        verbose=not args.quiet_progress,
        benchmark_warmup_batches=args.benchmark_warmup_batches,
        benchmark_batches=args.benchmark_batches,
    )
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    frame, classes = scan_dataset(Path(args.data_dir), args.limit_per_class, args.seed)
    train_frame, val_frame, test_frame = split_frame(frame, args.val_size, args.test_size, args.seed)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    (out_dir / "class_to_idx.json").write_text(json.dumps(class_to_idx, indent=2), encoding="utf-8")
    for name, split in [("all", frame), ("train", train_frame), ("val", val_frame), ("test", test_frame)]:
        split.to_csv(out_dir / f"{name}_files.csv", index=False)
    counts = frame.label.value_counts().rename_axis("class").reset_index(name="count")
    counts.to_csv(out_dir / "class_counts.csv", index=False)

    print_runtime_report(device, config)
    print(f"\nClasses ({len(classes)}): {classes}")
    print(f"Images: all={len(frame)} train={len(train_frame)} val={len(val_frame)} test={len(test_frame)}")
    print("Class counts:")
    for _, row in counts.iterrows():
        print(f"  {row['class']}: {row['count']}")

    train_tfms, eval_tfms = build_transforms(args.image_size)
    train_ds = ImageFrameDataset(train_frame, class_to_idx, train_tfms)
    val_ds = ImageFrameDataset(val_frame, class_to_idx, eval_tfms)
    test_ds = ImageFrameDataset(test_frame, class_to_idx, eval_tfms)
    sampler = make_sampler(train_frame, class_to_idx) if config.weighted_sampler else None
    loaders = {
        "train": make_loader(train_ds, args.batch_size, args.num_workers, sampler=sampler),
        "val": make_loader(val_ds, args.batch_size, args.num_workers, shuffle=False),
        "test": make_loader(test_ds, args.batch_size, args.num_workers, shuffle=False),
    }

    all_metrics = []
    for model_name in args.models:
        metrics = train_one_model(model_name, loaders, classes, config, device, out_dir)
        all_metrics.append(metrics)
        summary_frame = pd.DataFrame(all_metrics)
        summary_frame.to_csv(out_dir / "metrics_summary.csv", index=False)
        benchmark_cols = [
            "model",
            "total_params",
            "trainable_params",
            "non_trainable_params",
            "model_state_mb",
            "train_seconds",
            "inference_images_per_second",
            "inference_latency_ms_per_image_mean",
            "inference_latency_ms_per_image_p50",
            "inference_latency_ms_per_image_p95",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "macro_roc_auc_ovr",
            "macro_average_precision",
        ]
        summary_frame[[c for c in benchmark_cols if c in summary_frame.columns]].to_csv(
            out_dir / "benchmark_summary.csv", index=False
        )
        summary_frame[
            [c for c in ["model", "total_params", "trainable_params", "non_trainable_params", "parameter_mb", "buffer_mb", "model_state_mb"] if c in summary_frame.columns]
        ].to_csv(out_dir / "model_parameters.csv", index=False)

    metrics_frame = pd.DataFrame(all_metrics)
    save_comparison_plots(out_dir, metrics_frame)
    print(f"\nDone. Outputs: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
