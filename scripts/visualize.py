#!/usr/bin/env python3
"""Generate visualizations from evaluation results."""

import sys
sys.path.insert(0, ".")

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
import seaborn as sns

from src.grpo_reasoning.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", font_scale=1.2)
COLORS = {"Base": "#636EFA", "SFT": "#EF553B", "GRPO": "#00CC96"}


def plot_accuracy_comparison(summary: dict, results_dir: Path):
    """Bar chart comparing accuracy across models and benchmarks."""
    models = list(summary.keys())
    labels = [summary[m].get("label", m.upper()) for m in models]

    # Overall accuracy
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall
    accs = [summary[m]["accuracy"] for m in models]
    colors = [COLORS.get(l, "#999") for l in labels]
    bars = axes[0].bar(labels, accs, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Overall Accuracy", fontweight="bold")
    axes[0].set_ylim(0, max(accs) * 1.3 + 0.01)
    for bar, acc in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{acc:.1%}", ha="center", va="bottom", fontweight="bold")

    # Per source
    sources = set()
    for m in models:
        sources.update(summary[m].get("per_source", {}).keys())
    sources = sorted(sources)

    x = np.arange(len(sources))
    width = 0.8 / len(models)
    for i, (m, label) in enumerate(zip(models, labels)):
        per_src = summary[m].get("per_source", {})
        vals = [per_src.get(s, 0) for s in sources]
        axes[1].bar(x + i * width, vals, width, label=label,
                    color=COLORS.get(label, "#999"), edgecolor="white")

    axes[1].set_xticks(x + width * (len(models) - 1) / 2)
    axes[1].set_xticklabels([s.upper() for s in sources])
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Per-Benchmark Accuracy", fontweight="bold")
    axes[1].legend()

    plt.tight_layout()
    path = results_dir / "accuracy_comparison.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")


def plot_reasoning_patterns(summary: dict, results_dir: Path):
    """Radar chart of reasoning pattern emergence."""
    models = list(summary.keys())
    labels = [summary[m].get("label", m.upper()) for m in models]

    pattern_keys = [
        "uses_step_numbers", "uses_equations", "uses_therefore",
        "uses_hashtag_delimiter", "shows_intermediate_calc",
    ]
    pattern_labels = [
        "Step\nNumbers", "Equations", "Therefore/\nThus",
        "####\nDelimiter", "Intermediate\nCalcs",
    ]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(pattern_keys), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    for m, label in zip(models, labels):
        patterns = summary[m].get("reasoning_patterns", {})
        values = [patterns.get(k, 0) for k in pattern_keys]
        values += values[:1]
        ax.plot(angles, values, "o-", label=label, color=COLORS.get(label, "#999"),
                linewidth=2, markersize=6)
        ax.fill(angles, values, alpha=0.15, color=COLORS.get(label, "#999"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(pattern_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Emergent Reasoning Patterns", fontweight="bold", pad=20, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    path = results_dir / "reasoning_patterns.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")


def plot_response_analysis(summary: dict, results_dir: Path):
    """Bar chart of response length and line count."""
    models = list(summary.keys())
    labels = [summary[m].get("label", m.upper()) for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Response length
    lengths = [summary[m].get("reasoning_patterns", {}).get("avg_response_length", 0)
               for m in models]
    colors = [COLORS.get(l, "#999") for l in labels]
    bars = axes[0].bar(labels, lengths, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_ylabel("Characters")
    axes[0].set_title("Avg Response Length", fontweight="bold")
    for bar, val in zip(bars, lengths):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     f"{val:.0f}", ha="center", va="bottom")

    # Line count
    line_counts = [summary[m].get("reasoning_patterns", {}).get("avg_num_lines", 0)
                   for m in models]
    bars = axes[1].bar(labels, line_counts, color=colors, edgecolor="white", linewidth=1.5)
    axes[1].set_ylabel("Lines")
    axes[1].set_title("Avg Response Lines", fontweight="bold")
    for bar, val in zip(bars, line_counts):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     f"{val:.1f}", ha="center", va="bottom")

    plt.tight_layout()
    path = results_dir / "response_analysis.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")


def main():
    config = load_config("configs/default.yaml")
    results_dir = Path(config["paths"]["results_dir"])

    summary_path = results_dir / "evaluation_summary.json"
    if not summary_path.exists():
        logger.error(f"No evaluation summary found at {summary_path}")
        logger.error("Run evaluate.py first.")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    # Add display labels
    label_map = {"base": "Base", "sft": "SFT", "grpo": "GRPO"}
    for key in summary:
        summary[key]["label"] = label_map.get(key, key.upper())

    logger.info(f"Found results for: {list(summary.keys())}")

    plot_accuracy_comparison(summary, results_dir)
    plot_reasoning_patterns(summary, results_dir)
    plot_response_analysis(summary, results_dir)

    logger.info("All visualizations generated.")


if __name__ == "__main__":
    main()
