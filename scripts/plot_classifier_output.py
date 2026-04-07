#!/usr/bin/env python3
"""
Generate a quick chart from classifier JSONL output.

Expected input format (one JSON object per line):
  {"text": "...", "tag": "...", "conf": 0.97}

Usage:
  python plot_classifier_output.py --input results.jsonl
  python plot_classifier_output.py --input results.jsonl --output results_chart.png
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(input_path: str):
    """Load classifier JSONL records with required fields."""
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"Line {i} is not valid JSON")

            if "tag" not in rec or "conf" not in rec:
                raise ValueError(f"Line {i} must contain 'tag' and 'conf' fields")

            rows.append({"tag": str(rec["tag"]), "conf": float(rec["conf"])})

    if not rows:
        raise ValueError("No valid rows found in input JSONL")
    return rows


def plot_results(rows, output_path: str, title: str = "Classifier Output Summary"):
    """Create a compact 2-panel chart: tag counts + confidence histogram."""
    tag_counts = Counter(r["tag"] for r in rows)
    tags = [t for t, _ in sorted(tag_counts.items(), key=lambda x: -x[1])]
    counts = [tag_counts[t] for t in tags]
    confs = [r["conf"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(title)

    ax1 = axes[0]
    bars = ax1.bar(tags, counts, color="#2B7A78")
    ax1.set_title("Tag Distribution")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis="x", rotation=30)
    for b in bars:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, h, f"{int(h)}",
                 ha="center", va="bottom", fontsize=8)

    ax2 = axes[1]
    ax2.hist(confs, bins=20, color="#F4A261", edgecolor="black", alpha=0.85)
    ax2.set_title("Confidence Histogram")
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Samples")
    ax2.set_xlim(0.0, 1.0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot classifier JSONL output")
    parser.add_argument("--input", "-i", required=True,
                        help="Classifier output JSONL path")
    parser.add_argument("--output", "-o", default=None,
                        help="Output image path (default: <input>_chart.png)")
    parser.add_argument("--title", default="Classifier Output Summary",
                        help="Figure title")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = args.output
    if output_path is None:
        output_path = str(input_path.with_suffix("")) + "_chart.png"

    rows = load_results(str(input_path))
    plot_results(rows, output_path, title=args.title)
    print(f"Saved chart to {output_path}")


if __name__ == "__main__":
    main()
