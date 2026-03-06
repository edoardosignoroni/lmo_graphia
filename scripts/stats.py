#!/usr/bin/env python3
"""
stats.py – Produce a statistics report for a .jsonl dataset.

Each line in the input file is expected to be a JSON object with at least:
  - "text" (str)
  - "tag"  (str)
and optionally:
  - "conf" (float)   confidence score

Usage:
    python stats.py <path_to_jsonl> [--top N] [--output FILE]
"""

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"WARNING: skipping line {lineno}: {exc}", file=sys.stderr)
    return records


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2
    return s[mid]


def _bar(value: float, max_value: float, width: int = 30) -> str:
    if max_value == 0:
        return ""
    filled = int(round(value / max_value * width))
    return "█" * filled + "░" * (width - filled)


# ── core analysis ────────────────────────────────────────────────────────────

def compute_stats(records: list[dict], top_n: int = 20) -> dict:
    """Return a nested dict with all computed statistics."""

    tags: list[str] = []
    texts: list[str] = []
    confs: list[float] = []
    has_conf = False

    # per-tag buckets
    tag_texts: dict[str, list[str]] = defaultdict(list)
    tag_confs: dict[str, list[float]] = defaultdict(list)

    for rec in records:
        tag = rec.get("tag", "UNKNOWN")
        text = rec.get("text", "")
        tags.append(tag)
        texts.append(text)
        tag_texts[tag].append(text)
        if "conf" in rec and rec["conf"] is not None:
            has_conf = True
            confs.append(float(rec["conf"]))
            tag_confs[tag].append(float(rec["conf"]))

    total = len(records)
    unique_tags = sorted(tag_texts.keys())

    # ── global stats ─────────────────────────────────────────────────────
    global_word_lens = [len(t.split()) for t in texts]
    global_char_lens = [len(t) for t in texts]
    global_words = Counter()
    global_chars = Counter()
    for t in texts:
        global_words.update(t.split())
        global_chars.update(t)

    global_info: dict = {
        "total_items": total,
        "unique_tags": len(unique_tags),
        "tags": unique_tags,
        "avg_word_len": _mean(global_word_lens),
        "median_word_len": _median(global_word_lens),
        "stdev_word_len": _stdev(global_word_lens, _mean(global_word_lens)),
        "avg_char_len": _mean(global_char_lens),
        "median_char_len": _median(global_char_lens),
        "stdev_char_len": _stdev(global_char_lens, _mean(global_char_lens)),
        "top_words": global_words.most_common(top_n),
        "top_chars": global_chars.most_common(top_n),
    }
    if has_conf:
        global_info["avg_conf"] = _mean(confs)
        global_info["median_conf"] = _median(confs)
        global_info["stdev_conf"] = _stdev(confs, _mean(confs))
        global_info["min_conf"] = min(confs)
        global_info["max_conf"] = max(confs)

    # ── per-tag stats ────────────────────────────────────────────────────
    per_tag: dict[str, dict] = {}
    for tag in unique_tags:
        t_texts = tag_texts[tag]
        word_lens = [len(t.split()) for t in t_texts]
        char_lens = [len(t) for t in t_texts]

        words = Counter()
        chars = Counter()
        for t in t_texts:
            words.update(t.split())
            chars.update(t)

        info: dict = {
            "count": len(t_texts),
            "pct": len(t_texts) / total * 100 if total else 0,
            "avg_word_len": _mean(word_lens),
            "median_word_len": _median(word_lens),
            "stdev_word_len": _stdev(word_lens, _mean(word_lens)),
            "min_word_len": min(word_lens) if word_lens else 0,
            "max_word_len": max(word_lens) if word_lens else 0,
            "avg_char_len": _mean(char_lens),
            "median_char_len": _median(char_lens),
            "stdev_char_len": _stdev(char_lens, _mean(char_lens)),
            "min_char_len": min(char_lens) if char_lens else 0,
            "max_char_len": max(char_lens) if char_lens else 0,
            "top_words": words.most_common(top_n),
            "top_chars": chars.most_common(top_n),
        }

        if tag in tag_confs and tag_confs[tag]:
            tc = tag_confs[tag]
            info["avg_conf"] = _mean(tc)
            info["median_conf"] = _median(tc)
            info["stdev_conf"] = _stdev(tc, _mean(tc))
            info["min_conf"] = min(tc)
            info["max_conf"] = max(tc)

        per_tag[tag] = info

    return {"global": global_info, "per_tag": per_tag}


# ── pretty-print report ─────────────────────────────────────────────────────

def format_report(stats: dict) -> str:
    lines: list[str] = []
    g = stats["global"]
    pt = stats["per_tag"]

    sep = "=" * 80

    lines.append(sep)
    lines.append("  JSONL STATISTICS REPORT")
    lines.append(sep)

    # ── global overview ──────────────────────────────────────────────────
    lines.append("")
    lines.append("GLOBAL OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"  Total items          : {g['total_items']:>10,}")
    lines.append(f"  Unique tags          : {g['unique_tags']:>10}")
    lines.append(f"  Tags                 : {', '.join(g['tags'])}")
    lines.append("")
    lines.append(f"  Avg  text length (words) : {g['avg_word_len']:>8.1f}")
    lines.append(f"  Med  text length (words) : {g['median_word_len']:>8.1f}")
    lines.append(f"  Std  text length (words) : {g['stdev_word_len']:>8.1f}")
    lines.append(f"  Avg  text length (chars) : {g['avg_char_len']:>8.1f}")
    lines.append(f"  Med  text length (chars) : {g['median_char_len']:>8.1f}")
    lines.append(f"  Std  text length (chars) : {g['stdev_char_len']:>8.1f}")

    if "avg_conf" in g:
        lines.append("")
        lines.append(f"  Avg  confidence          : {g['avg_conf']:>8.4f}")
        lines.append(f"  Med  confidence          : {g['median_conf']:>8.4f}")
        lines.append(f"  Std  confidence          : {g['stdev_conf']:>8.4f}")
        lines.append(f"  Min  confidence          : {g['min_conf']:>8.4f}")
        lines.append(f"  Max  confidence          : {g['max_conf']:>8.4f}")

    # ── tag distribution ─────────────────────────────────────────────────
    lines.append("")
    lines.append("TAG DISTRIBUTION")
    lines.append("-" * 40)
    max_count = max((v["count"] for v in pt.values()), default=1)
    for tag in sorted(pt, key=lambda t: pt[t]["count"], reverse=True):
        info = pt[tag]
        bar = _bar(info["count"], max_count, 30)
        lines.append(f"  {tag:<15} {info['count']:>8,}  ({info['pct']:5.1f}%)  {bar}")

    # ── global top words ─────────────────────────────────────────────────
    lines.append("")
    lines.append("TOP WORDS (GLOBAL)")
    lines.append("-" * 40)
    max_wc = g["top_words"][0][1] if g["top_words"] else 1
    for word, cnt in g["top_words"]:
        bar = _bar(cnt, max_wc, 20)
        lines.append(f"  {word:<25} {cnt:>8,}  {bar}")

    # ── global top chars ─────────────────────────────────────────────────
    lines.append("")
    lines.append("TOP CHARACTERS (GLOBAL)")
    lines.append("-" * 40)
    max_cc = g["top_chars"][0][1] if g["top_chars"] else 1
    for ch, cnt in g["top_chars"]:
        display = repr(ch) if ch in (" ", "\t", "\n") else ch
        bar = _bar(cnt, max_cc, 20)
        lines.append(f"  {display:<10} {cnt:>10,}  {bar}")

    # ── per-tag details ──────────────────────────────────────────────────
    for tag in sorted(pt, key=lambda t: pt[t]["count"], reverse=True):
        info = pt[tag]
        lines.append("")
        lines.append(sep)
        lines.append(f"  TAG: {tag}   ({info['count']:,} items, {info['pct']:.1f}%)")
        lines.append(sep)

        lines.append("")
        lines.append("  Text length (words)")
        lines.append(f"    avg / median / std  : {info['avg_word_len']:.1f} / {info['median_word_len']:.1f} / {info['stdev_word_len']:.1f}")
        lines.append(f"    min / max           : {info['min_word_len']} / {info['max_word_len']}")

        lines.append("")
        lines.append("  Text length (chars)")
        lines.append(f"    avg / median / std  : {info['avg_char_len']:.1f} / {info['median_char_len']:.1f} / {info['stdev_char_len']:.1f}")
        lines.append(f"    min / max           : {info['min_char_len']} / {info['max_char_len']}")

        if "avg_conf" in info:
            lines.append("")
            lines.append("  Confidence")
            lines.append(f"    avg / median / std  : {info['avg_conf']:.4f} / {info['median_conf']:.4f} / {info['stdev_conf']:.4f}")
            lines.append(f"    min / max           : {info['min_conf']:.4f} / {info['max_conf']:.4f}")

        lines.append("")
        lines.append("  Top words")
        if info["top_words"]:
            mw = info["top_words"][0][1]
            for word, cnt in info["top_words"]:
                bar = _bar(cnt, mw, 15)
                lines.append(f"    {word:<25} {cnt:>8,}  {bar}")
        else:
            lines.append("    (none)")

        lines.append("")
        lines.append("  Top characters")
        if info["top_chars"]:
            mc = info["top_chars"][0][1]
            for ch, cnt in info["top_chars"]:
                display = repr(ch) if ch in (" ", "\t", "\n") else ch
                bar = _bar(cnt, mc, 15)
                lines.append(f"    {display:<10} {cnt:>10,}  {bar}")
        else:
            lines.append("    (none)")

    lines.append("")
    lines.append(sep)
    lines.append("  END OF REPORT")
    lines.append(sep)
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate statistics for a .jsonl dataset (text/tag/conf)."
    )
    parser.add_argument(
        "input",
        help="Path to the .jsonl file to analyze.",
    )
    parser.add_argument(
        "--top", "-n",
        type=int,
        default=20,
        help="Number of top words/chars to show per tag (default: 20).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Write report to this file instead of stdout.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw statistics as JSON instead of the formatted report.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.input} …", file=sys.stderr)
    records = _load_jsonl(args.input)
    print(f"Loaded {len(records):,} records. Computing statistics …", file=sys.stderr)

    stats = compute_stats(records, top_n=args.top)

    if args.json:
        output = json.dumps(stats, ensure_ascii=False, indent=2)
    else:
        output = format_report(stats)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
            f.write("\n")
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
