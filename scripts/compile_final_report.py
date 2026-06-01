#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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


def load_completed_runs(runs_dir: Path, model_names: list[str]):
    selected = {}
    candidates = []
    for metrics_path in sorted(runs_dir.glob("*/**/metrics.json")):
        model = metrics_path.parent.name
        if model not in model_names:
            continue
        history_path = metrics_path.parent / "history.csv"
        if not history_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        history = pd.read_csv(history_path)
        if len(history) < 1:
            continue
        row = {
            "model": model,
            "run": metrics_path.parents[1].name,
            "model_dir": str(metrics_path.parent),
            "history_path": str(history_path),
            "epochs_recorded": len(history),
            **metrics,
        }
        candidates.append(row)

    for model in model_names:
        model_candidates = [row for row in candidates if row["model"] == model]
        if not model_candidates:
            continue
        model_candidates.sort(key=lambda row: (row["epochs_recorded"], row.get("macro_f1", -1)), reverse=True)
        selected[model] = model_candidates[0]
    return selected, pd.DataFrame(candidates)


def plot_bar_grid(frame: pd.DataFrame, columns: list[str], output: Path, title: str, lower_is_better: set[str] | None = None):
    lower_is_better = lower_is_better or set()
    cols = [col for col in columns if col in frame.columns]
    if not cols:
        return
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(len(cols), 1, figsize=(12, max(4, len(cols) * 3.1)))
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        ordered = frame.sort_values(col, ascending=col in lower_is_better)
        sns.barplot(data=ordered, x=col, y="model", color="#4C78A8", ax=ax)
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel(col)
        ax.set_ylabel("")
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_history(history: pd.DataFrame, out_dir: Path):
    sns.set_theme(style="whitegrid")
    specs = [
        ("train_loss", "Training Loss", "history_train_loss_comparison.png"),
        ("val_loss", "Validation Loss", "history_val_loss_comparison.png"),
        ("train_accuracy", "Training Accuracy", "history_train_accuracy_comparison.png"),
        ("val_accuracy", "Validation Accuracy", "history_val_accuracy_comparison.png"),
        ("train_macro_f1", "Training Macro F1", "history_train_macro_f1_comparison.png"),
        ("val_macro_f1", "Validation Macro F1", "history_val_macro_f1_comparison.png"),
        ("epoch_seconds", "Epoch Time", "history_epoch_time_comparison.png"),
    ]
    for col, title, filename in specs:
        if col not in history.columns:
            continue
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.lineplot(data=history, x="epoch", y=col, hue="model", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(col.replace("_", " ").title())
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)

    dashboard = [
        ("train_loss", "Train Loss"),
        ("val_loss", "Val Loss"),
        ("train_accuracy", "Train Accuracy"),
        ("val_accuracy", "Val Accuracy"),
        ("train_macro_f1", "Train Macro F1"),
        ("val_macro_f1", "Val Macro F1"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    handles, labels = None, None
    for ax, (col, title) in zip(axes.ravel(), dashboard):
        sns.lineplot(data=history, x="epoch", y=col, hue="model", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("")
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    if handles and labels:
        fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_dir / "history_dashboard_all_models.png", dpi=180)
    plt.close(fig)


def plot_heatmaps(metrics: pd.DataFrame, out_dir: Path):
    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_roc_auc_ovr",
        "macro_average_precision",
        "inference_images_per_second",
        "inference_latency_ms_per_image_mean",
        "train_seconds",
        "total_params",
    ]
    metric_cols = [col for col in metric_cols if col in metrics.columns]
    rank = metrics.set_index("model")[metric_cols].copy()
    for col in ["inference_latency_ms_per_image_mean", "train_seconds", "total_params"]:
        if col in rank.columns:
            rank[col] = -rank[col]
    rank = rank.rank(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(rank, annot=True, fmt=".0f", cmap="viridis_r", ax=ax)
    ax.set_title("Benchmark Rank Heatmap (1 = Best)")
    fig.tight_layout()
    fig.savefig(out_dir / "benchmark_rank_heatmap.png", dpi=180)
    plt.close(fig)

    scaled = metrics.set_index("model")[metric_cols]
    scaled = (scaled - scaled.min()) / (scaled.max() - scaled.min())
    for col in ["inference_latency_ms_per_image_mean", "train_seconds", "total_params"]:
        if col in scaled.columns:
            scaled[col] = 1 - scaled[col]
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(scaled, annot=True, fmt=".2f", cmap="mako", ax=ax)
    ax.set_title("Normalized Benchmark Score Heatmap")
    fig.tight_layout()
    fig.savefig(out_dir / "benchmark_normalized_heatmap.png", dpi=180)
    plt.close(fig)

    per_class_rows = []
    for _, row in metrics.iterrows():
        for metric_name in ["precision", "recall", "f1", "support"]:
            suffix = f"__{metric_name}"
            for col in metrics.columns:
                if col.endswith(suffix):
                    per_class_rows.append(
                        {
                            "model": row["model"],
                            "class": col[: -len(suffix)],
                            "metric": metric_name,
                            "value": row[col],
                        }
                    )
    per_class = pd.DataFrame(per_class_rows)
    if per_class.empty:
        return
    per_class.to_csv(out_dir / "per_class_metrics_long.csv", index=False)
    for metric_name in ["precision", "recall", "f1"]:
        subset = per_class[per_class["metric"] == metric_name]
        pivot = subset.pivot(index="model", columns="class", values="value").loc[metrics["model"]]
        fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 1.25), 7))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="mako", vmin=0, vmax=1, ax=ax)
        ax.set_title(f"Per-Class {metric_name.title()} by Model")
        ax.set_xlabel("Class")
        ax.set_ylabel("Model")
        fig.tight_layout()
        fig.savefig(out_dir / f"per_class_{metric_name}_heatmap.png", dpi=180)
        plt.close(fig)


def plot_tradeoffs(metrics: pd.DataFrame, out_dir: Path):
    sns.set_theme(style="whitegrid")
    if {"total_params", "macro_f1", "inference_images_per_second"}.issubset(metrics.columns):
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(
            data=metrics,
            x="total_params",
            y="macro_f1",
            size="inference_images_per_second",
            hue="model",
            sizes=(80, 520),
            ax=ax,
        )
        ax.set_title("Macro F1 vs Model Size vs Inference Throughput")
        ax.set_xlabel("Parameters")
        ax.set_ylabel("Test Macro F1")
        fig.tight_layout()
        fig.savefig(out_dir / "accuracy_size_throughput.png", dpi=180)
        plt.close(fig)

    if {"train_seconds", "macro_f1", "inference_latency_ms_per_image_mean"}.issubset(metrics.columns):
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(
            data=metrics,
            x="train_seconds",
            y="macro_f1",
            size="inference_latency_ms_per_image_mean",
            hue="model",
            sizes=(80, 520),
            ax=ax,
        )
        ax.set_title("Macro F1 vs Training Time vs Inference Latency")
        ax.set_xlabel("Training Seconds")
        ax.set_ylabel("Test Macro F1")
        fig.tight_layout()
        fig.savefig(out_dir / "accuracy_train_time_latency.png", dpi=180)
        plt.close(fig)


def copy_representative_plots(selected: dict[str, dict], out_dir: Path):
    per_model_dir = out_dir / "per_model"
    per_model_dir.mkdir(parents=True, exist_ok=True)
    for model, row in selected.items():
        src_dir = Path(row["model_dir"])
        dst_dir = per_model_dir / model
        dst_dir.mkdir(parents=True, exist_ok=True)
        for name in [
            "learning_curves.png",
            "confusion_matrix_normalized.png",
            "roc_curves.png",
            "precision_recall_curves.png",
        ]:
            src = src_dir / name
            if src.exists():
                shutil.copy2(src, dst_dir / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--output-dir", default="reports/final_benchmark")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected, candidates = load_completed_runs(runs_dir, args.models)
    missing = [model for model in args.models if model not in selected]
    if missing:
        raise SystemExit(f"Missing completed metrics for: {', '.join(missing)}")

    candidates.to_csv(out_dir / "all_completed_run_candidates.csv", index=False)
    metrics = pd.DataFrame([selected[model] for model in args.models])
    metrics.to_csv(out_dir / "benchmark_summary.csv", index=False)
    parameter_cols = [
        "model",
        "run",
        "total_params",
        "trainable_params",
        "non_trainable_params",
        "parameter_mb",
        "buffer_mb",
        "model_state_mb",
    ]
    metrics[[col for col in parameter_cols if col in metrics.columns]].to_csv(out_dir / "model_parameters.csv", index=False)

    history_frames = []
    for model in args.models:
        history = pd.read_csv(selected[model]["history_path"])
        history["model"] = model
        history["run"] = selected[model]["run"]
        history_frames.append(history)
    history_all = pd.concat(history_frames, ignore_index=True)
    history_all.to_csv(out_dir / "history_all_models.csv", index=False)

    plot_bar_grid(
        metrics,
        ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_roc_auc_ovr", "macro_average_precision"],
        out_dir / "classification_metrics_comparison.png",
        "Classification Metrics by Model",
    )
    plot_bar_grid(
        metrics,
        ["train_seconds", "inference_images_per_second", "inference_latency_ms_per_image_mean", "total_params", "model_state_mb"],
        out_dir / "efficiency_metrics_comparison.png",
        "Efficiency and Model Size by Model",
        lower_is_better={"train_seconds", "inference_latency_ms_per_image_mean", "total_params", "model_state_mb"},
    )
    plot_history(history_all, out_dir)
    plot_heatmaps(metrics, out_dir)
    plot_tradeoffs(metrics, out_dir)
    copy_representative_plots(selected, out_dir)

    top_macro = metrics.sort_values("macro_f1", ascending=False).iloc[0]
    fastest = metrics.sort_values("inference_images_per_second", ascending=False).iloc[0]
    smallest = metrics.sort_values("total_params", ascending=True).iloc[0]
    summary = {
        "models": args.models,
        "selected_runs": {model: selected[model]["run"] for model in args.models},
        "best_macro_f1_model": top_macro["model"],
        "best_macro_f1": float(top_macro["macro_f1"]),
        "fastest_inference_model": fastest["model"],
        "fastest_inference_images_per_second": float(fastest["inference_images_per_second"]),
        "smallest_model": smallest["model"],
        "smallest_model_parameters": int(smallest["total_params"]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
