#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


def minmax_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    span = values.max() - values.min()
    if pd.isna(span) or span == 0:
        return pd.Series(1.0, index=series.index)
    scaled = (values - values.min()) / span
    return scaled if higher_is_better else 1 - scaled


def add_derived_scores(metrics: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    frame = metrics.copy()
    frame["million_params"] = frame["total_params"] / 1_000_000
    frame["train_minutes"] = frame["train_seconds"] / 60
    frame["macro_f1_per_million_params"] = frame["macro_f1"] / frame["million_params"]
    frame["macro_f1_per_training_minute"] = frame["macro_f1"] / frame["train_minutes"]
    frame["macro_f1_per_ms_latency"] = frame["macro_f1"] / frame["inference_latency_ms_per_image_mean"]
    frame["throughput_per_million_params"] = frame["inference_images_per_second"] / frame["million_params"]
    frame["accuracy_score_norm"] = minmax_score(frame["macro_f1"])
    frame["speed_score_norm"] = minmax_score(frame["inference_images_per_second"])
    frame["size_score_norm"] = minmax_score(frame["total_params"], higher_is_better=False)
    frame["training_cost_score_norm"] = minmax_score(frame["train_seconds"], higher_is_better=False)
    frame["balanced_deployment_score"] = (
        0.40 * frame["accuracy_score_norm"]
        + 0.25 * frame["speed_score_norm"]
        + 0.20 * frame["size_score_norm"]
        + 0.15 * frame["training_cost_score_norm"]
    )
    cols = [
        "model",
        "macro_f1_per_million_params",
        "macro_f1_per_training_minute",
        "macro_f1_per_ms_latency",
        "throughput_per_million_params",
        "balanced_deployment_score",
    ]
    frame[cols].sort_values("balanced_deployment_score", ascending=False).to_csv(
        out_dir / "efficiency_scores.csv", index=False
    )
    return frame


def save_decision_tables(metrics: pd.DataFrame, scored: pd.DataFrame, out_dir: Path):
    choices = [
        ("Best macro F1", metrics.sort_values("macro_f1", ascending=False).iloc[0]["model"], "macro_f1"),
        ("Best accuracy", metrics.sort_values("accuracy", ascending=False).iloc[0]["model"], "accuracy"),
        ("Best ROC-AUC", metrics.sort_values("macro_roc_auc_ovr", ascending=False).iloc[0]["model"], "macro_roc_auc_ovr"),
        (
            "Best average precision",
            metrics.sort_values("macro_average_precision", ascending=False).iloc[0]["model"],
            "macro_average_precision",
        ),
        (
            "Fastest inference",
            metrics.sort_values("inference_images_per_second", ascending=False).iloc[0]["model"],
            "inference_images_per_second",
        ),
        ("Lowest latency", metrics.sort_values("inference_latency_ms_per_image_mean").iloc[0]["model"], "inference_latency_ms_per_image_mean"),
        ("Smallest model", metrics.sort_values("total_params").iloc[0]["model"], "total_params"),
        ("Fastest training", metrics.sort_values("train_seconds").iloc[0]["model"], "train_seconds"),
        (
            "Best balanced deployment score",
            scored.sort_values("balanced_deployment_score", ascending=False).iloc[0]["model"],
            "balanced_deployment_score",
        ),
    ]
    rows = []
    for label, model, metric in choices:
        value = scored.loc[scored["model"] == model, metric].iloc[0]
        rows.append({"category": label, "model": model, "metric": metric, "value": value})
    pd.DataFrame(rows).to_csv(out_dir / "decision_table.csv", index=False)


def save_rank_tables(scored: pd.DataFrame, out_dir: Path):
    rank_specs = {
        "accuracy_rank": ("macro_f1", False),
        "speed_rank": ("inference_images_per_second", False),
        "latency_rank": ("inference_latency_ms_per_image_mean", True),
        "size_rank": ("total_params", True),
        "training_cost_rank": ("train_seconds", True),
        "balanced_score_rank": ("balanced_deployment_score", False),
        "roc_auc_rank": ("macro_roc_auc_ovr", False),
    }
    rank_frame = scored[["model"]].copy()
    for rank_name, (col, ascending) in rank_specs.items():
        rank_frame[rank_name] = scored[col].rank(ascending=ascending, method="min").astype(int)
    rank_cols = [col for col in rank_frame.columns if col != "model"]
    rank_frame["average_rank"] = rank_frame[rank_cols].mean(axis=1)
    rank_frame.sort_values("average_rank").to_csv(out_dir / "rank_aggregation.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    plot_data = rank_frame.sort_values("average_rank")
    sns.barplot(data=plot_data, x="average_rank", y="model", color="#59A14F", ax=ax)
    ax.set_title("Average Rank Across Accuracy, Speed, Size, Cost, and ROC-AUC")
    ax.set_xlabel("Average Rank (Lower is Better)")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out_dir / "average_rank_comparison.png", dpi=180)
    plt.close(fig)


def save_radar_chart(scored: pd.DataFrame, out_dir: Path):
    cols = [
        "accuracy_score_norm",
        "speed_score_norm",
        "size_score_norm",
        "training_cost_score_norm",
    ]
    labels = ["Macro F1", "Inference Speed", "Small Size", "Training Speed"]
    data = scored.set_index("model")[cols]
    angles = np.linspace(0, 2 * np.pi, len(cols), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    for model, row in data.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, label=model)
        ax.fill(angles, values, alpha=0.04)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Normalized Model Radar Chart")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "radar_model_profiles.png", dpi=180)
    plt.close(fig)


def save_pareto_frontier(scored: pd.DataFrame, out_dir: Path):
    objectives = ["macro_f1", "inference_images_per_second", "total_params", "train_seconds"]
    rows = []
    for _, candidate in scored.iterrows():
        dominated = False
        for _, other in scored.iterrows():
            if candidate["model"] == other["model"]:
                continue
            better_or_equal = (
                other["macro_f1"] >= candidate["macro_f1"]
                and other["inference_images_per_second"] >= candidate["inference_images_per_second"]
                and other["total_params"] <= candidate["total_params"]
                and other["train_seconds"] <= candidate["train_seconds"]
            )
            strictly_better = (
                other["macro_f1"] > candidate["macro_f1"]
                or other["inference_images_per_second"] > candidate["inference_images_per_second"]
                or other["total_params"] < candidate["total_params"]
                or other["train_seconds"] < candidate["train_seconds"]
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        rows.append({"model": candidate["model"], "pareto_frontier": not dominated, **{col: candidate[col] for col in objectives}})
    pareto = pd.DataFrame(rows)
    pareto.to_csv(out_dir / "pareto_frontier.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=pareto.merge(scored[["model", "inference_latency_ms_per_image_mean"]], on="model"),
        x="total_params",
        y="macro_f1",
        hue="pareto_frontier",
        size="inference_latency_ms_per_image_mean",
        sizes=(80, 480),
        ax=ax,
    )
    for _, row in pareto.iterrows():
        ax.text(row["total_params"], row["macro_f1"], row["model"], fontsize=7)
    ax.set_title("Pareto Frontier: Macro F1 vs Parameters")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Test Macro F1")
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_frontier_accuracy_params.png", dpi=180)
    plt.close(fig)


def save_generalization_and_convergence(history: pd.DataFrame, out_dir: Path):
    final = history.sort_values("epoch").groupby("model", as_index=False).tail(1).copy()
    final["accuracy_gap"] = final["train_accuracy"] - final["val_accuracy"]
    final["macro_f1_gap"] = final["train_macro_f1"] - final["val_macro_f1"]
    final["loss_gap"] = final["val_loss"] - final["train_loss"]
    final[["model", "accuracy_gap", "macro_f1_gap", "loss_gap"]].sort_values("macro_f1_gap").to_csv(
        out_dir / "generalization_gap.csv", index=False
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, col, title in zip(
        axes,
        ["accuracy_gap", "macro_f1_gap", "loss_gap"],
        ["Accuracy Gap", "Macro F1 Gap", "Loss Gap"],
    ):
        ordered = final.sort_values(col)
        sns.barplot(data=ordered, x=col, y="model", color="#F28E2B", ax=ax)
        ax.set_title(title)
        ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out_dir / "generalization_gap_comparison.png", dpi=180)
    plt.close(fig)

    rows = []
    for model, group in history.groupby("model"):
        group = group.sort_values("epoch")
        best_idx = group["val_macro_f1"].idxmax()
        best = group.loc[best_idx]
        final_row = group.iloc[-1]
        threshold = final_row["val_macro_f1"] * 0.95
        reached = group[group["val_macro_f1"] >= threshold]
        rows.append(
            {
                "model": model,
                "best_val_macro_f1": best["val_macro_f1"],
                "best_val_epoch": int(best["epoch"]),
                "final_val_macro_f1": final_row["val_macro_f1"],
                "final_epoch": int(final_row["epoch"]),
                "first_epoch_to_95pct_final_val_macro_f1": int(reached.iloc[0]["epoch"]) if not reached.empty else np.nan,
                "final_minus_best_val_macro_f1": final_row["val_macro_f1"] - best["val_macro_f1"],
                "late_val_macro_f1_std_last10": group.tail(10)["val_macro_f1"].std(),
                "late_val_loss_std_last10": group.tail(10)["val_loss"].std(),
            }
        )
    conv = pd.DataFrame(rows)
    conv.to_csv(out_dir / "convergence_stability.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    specs = [
        ("first_epoch_to_95pct_final_val_macro_f1", "Epoch to 95% Final Val Macro F1"),
        ("best_val_epoch", "Best Validation Epoch"),
        ("late_val_macro_f1_std_last10", "Late Validation F1 Std"),
    ]
    for ax, (col, title) in zip(axes, specs):
        ordered = conv.sort_values(col)
        sns.barplot(data=ordered, x=col, y="model", color="#B07AA1", ax=ax)
        ax.set_title(title)
        ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_stability_comparison.png", dpi=180)
    plt.close(fig)


def save_metric_correlation(scored: pd.DataFrame, out_dir: Path):
    cols = [
        "total_params",
        "model_state_mb",
        "train_seconds",
        "inference_images_per_second",
        "inference_latency_ms_per_image_mean",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_roc_auc_ovr",
        "macro_average_precision",
    ]
    cols = [col for col in cols if col in scored.columns]
    corr = scored[cols].corr()
    corr.to_csv(out_dir / "metric_correlation.csv")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Metric Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(out_dir / "metric_correlation_heatmap.png", dpi=180)
    plt.close(fig)


def save_family_comparison(scored: pd.DataFrame, out_dir: Path):
    def family(model: str) -> str:
        for prefix, name in [
            ("mobilenet", "MobileNet"),
            ("efficientnet", "EfficientNet"),
            ("densenet", "DenseNet"),
            ("resnet", "ResNet"),
            ("squeezenet", "SqueezeNet"),
            ("shufflenet", "ShuffleNet"),
        ]:
            if model.startswith(prefix):
                return name
        return "Other"

    frame = scored.copy()
    frame["family"] = frame["model"].map(family)
    family_summary = frame.groupby("family", as_index=False).agg(
        models=("model", "count"),
        macro_f1=("macro_f1", "mean"),
        accuracy=("accuracy", "mean"),
        train_seconds=("train_seconds", "mean"),
        inference_images_per_second=("inference_images_per_second", "mean"),
        total_params=("total_params", "mean"),
        balanced_deployment_score=("balanced_deployment_score", "mean"),
    )
    family_summary.to_csv(out_dir / "model_family_comparison.csv", index=False)
    plot_bar_grid(
        family_summary.rename(columns={"family": "model"}),
        ["macro_f1", "inference_images_per_second", "total_params", "train_seconds", "balanced_deployment_score"],
        out_dir / "model_family_comparison.png",
        "Model Family Comparison",
        lower_is_better={"total_params", "train_seconds"},
    )


def save_per_class_best_tables(metrics: pd.DataFrame, out_dir: Path):
    rows = []
    for metric_name in ["precision", "recall", "f1"]:
        suffix = f"__{metric_name}"
        for col in [c for c in metrics.columns if c.endswith(suffix)]:
            cls = col[: -len(suffix)]
            best = metrics.sort_values(col, ascending=False).iloc[0]
            weakest_model = metrics.sort_values(col, ascending=True).iloc[0]
            rows.append(
                {
                    "class": cls,
                    "metric": metric_name,
                    "best_model": best["model"],
                    "best_value": best[col],
                    "lowest_model": weakest_model["model"],
                    "lowest_value": weakest_model[col],
                    "mean_value": metrics[col].mean(),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "per_class_best_models.csv", index=False)


def save_confusion_tables(selected: dict[str, dict], out_dir: Path):
    rows = []
    for model, selected_row in selected.items():
        model_dir = Path(selected_row["model_dir"])
        preds_path = model_dir / "test_predictions.csv"
        class_map_path = model_dir.parents[0] / "class_to_idx.json"
        if not preds_path.exists() or not class_map_path.exists():
            continue
        idx_to_class = {idx: cls for cls, idx in json.loads(class_map_path.read_text()).items()}
        preds = pd.read_csv(preds_path)
        wrong = preds[preds["y_true"] != preds["y_pred"]]
        if wrong.empty:
            continue
        pair_counts = wrong.groupby(["y_true", "y_pred"]).size().reset_index(name="count")
        pair_counts["true_class"] = pair_counts["y_true"].map(lambda idx: idx_to_class[int(idx)])
        pair_counts["predicted_class"] = pair_counts["y_pred"].map(lambda idx: idx_to_class[int(idx)])
        pair_counts["model"] = model
        rows.append(pair_counts[["model", "true_class", "predicted_class", "count"]])
    if not rows:
        return
    confusion = pd.concat(rows, ignore_index=True)
    confusion.to_csv(out_dir / "confusion_pairs_by_model.csv", index=False)
    aggregate = confusion.groupby(["true_class", "predicted_class"], as_index=False)["count"].sum()
    aggregate.sort_values("count", ascending=False).to_csv(out_dir / "confusion_pairs_overall.csv", index=False)
    top = aggregate.sort_values("count", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(12, 8))
    top = top.assign(pair=top["true_class"] + " -> " + top["predicted_class"])
    sns.barplot(data=top, x="count", y="pair", color="#E15759", ax=ax)
    ax.set_title("Top Confused Class Pairs Across Models")
    ax.set_xlabel("Total Misclassifications")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out_dir / "top_confusion_pairs.png", dpi=180)
    plt.close(fig)


def save_report_cards(scored: pd.DataFrame, out_dir: Path):
    frame = scored.copy()
    grade_cols = {
        "accuracy_grade": "accuracy_score_norm",
        "speed_grade": "speed_score_norm",
        "size_grade": "size_score_norm",
        "training_cost_grade": "training_cost_score_norm",
        "overall_grade": "balanced_deployment_score",
    }

    def grade(value: float) -> str:
        if value >= 0.80:
            return "A"
        if value >= 0.60:
            return "B"
        if value >= 0.40:
            return "C"
        if value >= 0.20:
            return "D"
        return "E"

    report = frame[["model"]].copy()
    for grade_name, col in grade_cols.items():
        report[grade_name] = frame[col].map(grade)
    report["recommendation"] = frame.apply(
        lambda row: "Best balanced" if row["balanced_deployment_score"] == frame["balanced_deployment_score"].max()
        else "Fast deployment" if row["speed_score_norm"] >= 0.75
        else "Accuracy focused" if row["accuracy_score_norm"] >= 0.75
        else "Baseline/secondary option",
        axis=1,
    )
    report.to_csv(out_dir / "model_report_cards.csv", index=False)


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
    scored = add_derived_scores(metrics, out_dir)
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
    save_decision_tables(metrics, scored, out_dir)
    save_rank_tables(scored, out_dir)
    save_radar_chart(scored, out_dir)
    save_pareto_frontier(scored, out_dir)
    save_generalization_and_convergence(history_all, out_dir)
    save_metric_correlation(scored, out_dir)
    save_family_comparison(scored, out_dir)
    save_per_class_best_tables(metrics, out_dir)
    save_confusion_tables(selected, out_dir)
    save_report_cards(scored, out_dir)
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
