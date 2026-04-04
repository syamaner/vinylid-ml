#!/usr/bin/env python3
"""Generate cross-model comparison reports from evaluation results (story #22).

Produces two report types:

1. **Simple comparison** (backward-compatible):
   Reads ``results/summary.csv`` and optionally ``results/latency_summary.csv``.
   Produces ``comparison.csv`` and ``comparison.html``.

2. **Multi-context comparison** (new in story #22):
   Merges test-split, phone-complete, and phone-sample evaluation results into
   a single report with clearly separated evaluation contexts.
   Sources: ``summary.csv``, ``phone_eval_summary.csv``,
   ``phone_sample_eval_summary.csv``.
   Outputs: ``multi_context_comparison.csv``, ``multi_context_comparison.html``.

Context labels:
  - ``test-split``     — augmented gallery/query evaluation (summary.csv)
  - ``phone-complete`` — 203 real-world phone photos (phone_eval_summary.csv)
  - ``phone-sample``   — 55-photo sample set matching C2 sample-mode (#53)

Usage:
    python scripts/compare_models.py
    python scripts/compare_models.py --results-dir results/
    python scripts/compare_models.py --latency-csv results/latency_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

__all__ = [
    "build_comparison_rows",
    "build_multi_context_rows",
    "decision_verdict",
    "generate_comparison_html",
    "generate_multi_context_html",
    "save_comparison_csv",
    "save_multi_context_csv",
]

# ── HTML template ──────────────────────────────────────────────────────────────

_HTML_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Zero-Shot Evaluation — Model Comparison</title>
<style>
body{font-family:system-ui,sans-serif;margin:2em;background:#fafafa;color:#333}
h1{margin-bottom:.25em}p.subtitle{color:#666;margin-top:0}
table{border-collapse:collapse;width:100%;background:#fff;
      box-shadow:0 1px 4px rgba(0,0,0,.12);border-radius:4px;overflow:hidden}
th{background:#2c3e50;color:#fff;padding:10px 14px;text-align:left;
   white-space:nowrap}
td{padding:9px 14px;border-bottom:1px solid #eee;white-space:nowrap}
tr:last-child td{border-bottom:none}
tr:hover td{background:#f0f4f8}
.r1-green{background:#d4edda;font-weight:700}
.r1-amber{background:#fff3cd;font-weight:700}
.r1-red{background:#f8d7da;font-weight:700}
.v-ok{color:#155724;font-weight:700}
.v-likely{color:#856404;font-weight:700}
.v-req{color:#721c24;font-weight:700}
.na{color:#bbb}
</style>
</head>
<body>
<h1>Zero-Shot Evaluation — Model Comparison</h1>
<p class="subtitle">
  Decision criteria:&nbsp;
  <strong style="color:#155724">Recall@1 &ge; 90%</strong> → zero-shot sufficient
  &nbsp;|&nbsp;
  <strong style="color:#856404">80-90%</strong> → fine-tuning likely needed
  &nbsp;|&nbsp;
  <strong style="color:#721c24">&lt;80%</strong> → fine-tuning required
</p>
"""

_HTML_TAIL = "</body>\n</html>\n"


def decision_verdict(recall_at_1: float) -> str:
    """Map Recall@1 to a production decision verdict.

    Applies the plan §6 decision criteria thresholds:
    * ≥ 0.90 → zero-shot sufficient
    * 0.80 - 0.89 → fine-tuning likely needed
    * < 0.80 → fine-tuning required

    Args:
        recall_at_1: Recall@1 score in [0, 1].

    Returns:
        One of ``"zero-shot sufficient"``, ``"fine-tuning likely needed"``,
        or ``"fine-tuning required"``.
    """
    if recall_at_1 >= 0.90:
        return "zero-shot sufficient"
    if recall_at_1 >= 0.80:
        return "fine-tuning likely needed"
    return "fine-tuning required"


def _recall_css_class(recall_at_1: float) -> str:
    if recall_at_1 >= 0.90:
        return "r1-green"
    if recall_at_1 >= 0.80:
        return "r1-amber"
    return "r1-red"


def _verdict_css_class(verdict: str) -> str:
    if verdict == "zero-shot sufficient":
        return "v-ok"
    if verdict == "fine-tuning likely needed":
        return "v-likely"
    if verdict == "fine-tuning required":
        return "v-req"
    return "na"  # N/A or any unexpected value


def _load_csv_by_model(csv_path: Path) -> dict[str, dict[str, str]]:
    """Load a model-comparison CSV, keeping the latest row per model_id.

    Works for ``summary.csv``, ``phone_eval_summary.csv``, and
    ``phone_sample_eval_summary.csv`` — all share the same schema.

    Args:
        csv_path: Absolute path to the CSV file.

    Returns:
        Dict mapping model_id → latest row (all values as strings).
        Returns empty dict if the file does not exist.
    """
    if not csv_path.exists():
        return {}
    by_model: dict[str, dict[str, str]] = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            model_id = str(row.get("model_id", ""))
            ts = str(row.get("timestamp", ""))
            existing = by_model.get(model_id)
            if existing is None or ts > str(existing.get("timestamp", "")):
                by_model[model_id] = {str(k): str(v) for k, v in row.items()}
    return by_model


def _load_summary_csv(results_dir: Path) -> dict[str, dict[str, str]]:
    """Load results/summary.csv, keeping the latest run per model.

    Args:
        results_dir: Top-level results directory.

    Returns:
        Dict mapping model_id → latest summary row (all values as strings).
    """
    return _load_csv_by_model(results_dir / "summary.csv")


def _load_latency_csv(latency_path: Path) -> dict[str, dict[str, float]]:
    """Load latency_summary.csv keyed by model_id.

    Args:
        latency_path: Path to latency_summary.csv.

    Returns:
        Dict mapping model_id → ``{"p50_ms": ..., "p95_ms": ..., "p99_ms": ...}``.
        Returns empty dict if file does not exist.
    """
    if not latency_path.exists():
        return {}
    result: dict[str, dict[str, float]] = {}
    with latency_path.open(newline="") as f:
        for row in csv.DictReader(f):
            model_id = str(row.get("model_id", ""))
            result[model_id] = {
                "p50_ms": float(row.get("p50_ms") or 0),
                "p95_ms": float(row.get("p95_ms") or 0),
                "p99_ms": float(row.get("p99_ms") or 0),
            }
    return result


def build_comparison_rows(
    summary: dict[str, dict[str, str]],
    latency: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Build unified comparison rows from summary and latency data.

    Sorts by Recall@1 descending. Applies decision verdict to each row.

    Args:
        summary: Per-model summary rows (from ``_load_summary_csv``).
        latency: Per-model latency data (from ``_load_latency_csv``).

    Returns:
        List of row dicts with keys: ``model_id``, ``recall_at_1``,
        ``recall_at_5``, ``map_at_5``, ``mrr``, ``num_gallery``,
        ``num_queries``, ``latency_p50_ms``, ``latency_p95_ms``,
        ``latency_p99_ms``, ``verdict``.
    """

    def _to_float(val: str) -> float | None:
        try:
            return float(val) if val.strip() else None
        except ValueError:
            return None

    rows: list[dict[str, Any]] = []
    for model_id, s in summary.items():
        r1 = _to_float(s.get("recall_at_1", ""))
        lat = latency.get(model_id, {})
        row: dict[str, Any] = {
            "model_id": model_id,
            "recall_at_1": r1,
            "recall_at_5": _to_float(s.get("recall_at_5", "")),
            "map_at_5": _to_float(s.get("map_at_5", "")),
            "mrr": _to_float(s.get("mrr", "")),
            "num_gallery": s.get("num_gallery", ""),
            "num_queries": s.get("num_queries", ""),
            "latency_p50_ms": lat.get("p50_ms"),
            "latency_p95_ms": lat.get("p95_ms"),
            "latency_p99_ms": lat.get("p99_ms"),
            "verdict": decision_verdict(r1) if r1 is not None else "N/A",
        }
        rows.append(row)

    rows.sort(
        key=lambda r: r["recall_at_1"] if r["recall_at_1"] is not None else -1.0, reverse=True
    )
    return rows


def save_comparison_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write comparison rows to a CSV file.

    Args:
        rows: Comparison rows from :func:`build_comparison_rows`.
        output_path: Destination CSV path.
    """
    fieldnames = [
        "model_id",
        "recall_at_1",
        "recall_at_5",
        "map_at_5",
        "mrr",
        "num_gallery",
        "num_queries",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "verdict",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_comparison_html(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    has_latency: bool = False,
) -> None:
    """Generate a colour-coded HTML comparison table.

    Recall@1 cells are colour-coded: green >= 90%, amber 80-90%, red < 80%.
    The verdict column is bold-coloured to match.

    Args:
        rows: Comparison rows from :func:`build_comparison_rows`.
        output_path: Destination HTML path.
        has_latency: Include p50/p95 latency columns if ``True``. Default False.
    """

    def _fmt(v: Any, decimals: int = 4) -> str:
        if v is None:
            return '<span class="na">N/A</span>'
        return f"{float(v):.{decimals}f}"

    # Build header row
    lat_headers = "<th>p50&nbsp;ms</th><th>p95&nbsp;ms</th>" if has_latency else ""
    header_row = (
        "<tr>"
        "<th>Model</th>"
        "<th>Recall@1</th><th>Recall@5</th><th>mAP@5</th><th>MRR</th>"
        "<th>Gallery</th><th>Queries</th>"
        f"{lat_headers}"
        "<th>Verdict</th>"
        "</tr>"
    )

    # Build data rows
    data_rows: list[str] = []
    for r in rows:
        r1 = r["recall_at_1"]
        verdict = str(r.get("verdict", ""))
        r1_cls = _recall_css_class(float(r1)) if r1 is not None else ""
        v_cls = _verdict_css_class(verdict)

        lat_cells = ""
        if has_latency:
            lat_cells = (
                f"<td>{_fmt(r.get('latency_p50_ms'), 1)}</td>"
                f"<td>{_fmt(r.get('latency_p95_ms'), 1)}</td>"
            )

        data_rows.append(
            "<tr>"
            f"<td>{r['model_id']}</td>"
            f'<td class="{r1_cls}">{_fmt(r1)}</td>'
            f"<td>{_fmt(r.get('recall_at_5'))}</td>"
            f"<td>{_fmt(r.get('map_at_5'))}</td>"
            f"<td>{_fmt(r.get('mrr'))}</td>"
            f"<td>{r.get('num_gallery', '')}</td>"
            f"<td>{r.get('num_queries', '')}</td>"
            f"{lat_cells}"
            f'<td class="{v_cls}">{verdict}</td>'
            "</tr>"
        )

    html = (
        _HTML_HEAD
        + "<table>\n<thead>\n"
        + header_row
        + "\n</thead>\n<tbody>\n"
        + "\n".join(data_rows)
        + "\n</tbody>\n</table>\n"
        + _HTML_TAIL
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


_MULTI_CTX_HTML_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>VinylID — Sprint 3 Multi-Context Model Comparison</title>
<style>
body{font-family:system-ui,sans-serif;margin:2em;background:#fafafa;color:#333}
h1{margin-bottom:.25em}h2{margin-top:2em;color:#2c3e50}p.subtitle{color:#666;margin-top:0}
table{border-collapse:collapse;width:100%;background:#fff;
      box-shadow:0 1px 4px rgba(0,0,0,.12);border-radius:4px;overflow:hidden;margin-bottom:1.5em}
th{background:#2c3e50;color:#fff;padding:10px 14px;text-align:left;white-space:nowrap}
td{padding:9px 14px;border-bottom:1px solid #eee;white-space:nowrap}
tr:last-child td{border-bottom:none}
tr:hover td{background:#f0f4f8}
.r1-green{background:#d4edda;font-weight:700}
.r1-amber{background:#fff3cd;font-weight:700}
.r1-red{background:#f8d7da;font-weight:700}
.v-ok{color:#155724;font-weight:700}
.v-likely{color:#856404;font-weight:700}
.v-req{color:#721c24;font-weight:700}
.na{color:#bbb}
.ctx-badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:.8em;
           background:#e3f2fd;color:#1565c0;font-weight:600;margin-left:.3em}
</style>
</head>
<body>
<h1>VinylID — Sprint 3 Multi-Context Model Comparison</h1>
<p class="subtitle">
  Decision criteria:&nbsp;
  <strong style="color:#155724">Recall@1 &ge; 90%</strong> → zero-shot sufficient
  &nbsp;|&nbsp;
  <strong style="color:#856404">80-90%</strong> → fine-tuning likely needed
  &nbsp;|&nbsp;
  <strong style="color:#721c24">&lt;80%</strong> → fine-tuning required
</p>
"""


def _render_table(
    section_title: str,
    context_desc: str,
    rows: list[dict[str, Any]],
) -> str:
    """Render one HTML section (h2 + table) for a single evaluation context.

    Args:
        section_title: Section heading text (e.g. ``"Test-Split Evaluation"``).
        context_desc: Short description shown below the heading.
        rows: Multi-context rows for this section (all same eval_context).

    Returns:
        HTML string fragment.
    """

    def _fmt(v: Any, decimals: int = 4) -> str:
        if v is None:
            return '<span class="na">N/A</span>'
        return f"{float(v):.{decimals}f}"

    header_row = (
        "<tr>"
        "<th>Model</th>"
        "<th>Recall@1</th><th>Recall@5</th><th>mAP@5</th><th>MRR</th>"
        "<th>Gallery</th><th>Queries</th>"
        "<th>Verdict</th>"
        "</tr>"
    )
    data_rows: list[str] = []
    for r in rows:
        r1 = r.get("recall_at_1")
        verdict = str(r.get("verdict", ""))
        r1_cls = _recall_css_class(float(r1)) if r1 is not None else ""
        v_cls = _verdict_css_class(verdict)
        data_rows.append(
            "<tr>"
            f"<td>{r['model_id']}</td>"
            f'<td class="{r1_cls}">{_fmt(r1)}</td>'
            f"<td>{_fmt(r.get('recall_at_5'))}</td>"
            f"<td>{_fmt(r.get('map_at_5'))}</td>"
            f"<td>{_fmt(r.get('mrr'))}</td>"
            f"<td>{r.get('num_gallery', '')}</td>"
            f"<td>{r.get('num_queries', '')}</td>"
            f'<td class="{v_cls}">{verdict}</td>'
            "</tr>"
        )
    return (
        f"<h2>{section_title}</h2>\n"
        f'<p style="color:#555;margin-top:-.5em">{context_desc}</p>\n'
        "<table>\n<thead>\n"
        + header_row
        + "\n</thead>\n<tbody>\n"
        + "\n".join(data_rows)
        + "\n</tbody>\n</table>\n"
    )


def build_multi_context_rows(
    test_split: dict[str, dict[str, str]],
    phone_complete: dict[str, dict[str, str]],
    phone_sample: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    """Build multi-context comparison rows from all evaluation sources.

    Each row represents one (model, eval_context) pair sorted by context then
    Recall@1 descending within each context.

    Args:
        test_split: Per-model rows from ``summary.csv``.
        phone_complete: Per-model rows from ``phone_eval_summary.csv``.
        phone_sample: Per-model rows from ``phone_sample_eval_summary.csv``.

    Returns:
        List of row dicts with keys: ``model_id``, ``eval_context``,
        ``recall_at_1``, ``recall_at_5``, ``map_at_5``, ``mrr``,
        ``num_gallery``, ``num_queries``, ``verdict``.
    """

    def _to_float(val: str) -> float | None:
        try:
            return float(val) if val.strip() else None
        except ValueError:
            return None

    def _make_rows(source: dict[str, dict[str, str]], context: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for model_id, s in source.items():
            r1 = _to_float(s.get("recall_at_1", ""))
            row: dict[str, Any] = {
                "model_id": model_id,
                "eval_context": context,
                "recall_at_1": r1,
                "recall_at_5": _to_float(s.get("recall_at_5", "")),
                "map_at_5": _to_float(s.get("map_at_5", "")),
                "mrr": _to_float(s.get("mrr", "")),
                "num_gallery": s.get("num_gallery", ""),
                "num_queries": s.get("num_queries", ""),
                "verdict": decision_verdict(r1) if r1 is not None else "N/A",
            }
            rows.append(row)
        rows.sort(
            key=lambda r: r["recall_at_1"] if r["recall_at_1"] is not None else -1.0,
            reverse=True,
        )
        return rows

    return (
        _make_rows(test_split, "test-split")
        + _make_rows(phone_complete, "phone-complete")
        + _make_rows(phone_sample, "phone-sample")
    )


def save_multi_context_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write multi-context comparison rows to a CSV file.

    Args:
        rows: Rows from :func:`build_multi_context_rows`.
        output_path: Destination CSV path.
    """
    fieldnames = [
        "eval_context",
        "model_id",
        "recall_at_1",
        "recall_at_5",
        "map_at_5",
        "mrr",
        "num_gallery",
        "num_queries",
        "verdict",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def generate_multi_context_html(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate a multi-context HTML report with one table per evaluation context.

    Sections rendered (in order, skipped if no data):
      1. Test-Split Evaluation (``test-split`` context — augmented gallery/query)
      2. Phone Photos — Complete Set (``phone-complete`` context)
      3. Phone Photos — Sample Set (``phone-sample`` context, #53)

    Args:
        rows: Rows from :func:`build_multi_context_rows`.
        output_path: Destination HTML path.
    """
    by_context: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        ctx = str(r.get("eval_context", "unknown"))
        by_context.setdefault(ctx, []).append(r)

    sections: list[str] = []
    ctx_specs = [
        (
            "test-split",
            "Test-Split Evaluation (A/B/C/D models)",
            "Augmented gallery/query split: 855 gallery images, ~4 500 queries. "
            "All models evaluated on identical data. D1-sscd-lightglue-k* rows show "
            "K-sweep results for the D1 hybrid pipeline (A4-sscd + LightGlue).",
        ),
        (
            "phone-complete",
            "Phone Photo Evaluation — Complete Set",
            "203 real-world iPhone captures vs 981 gallery images. "
            "Cross-domain challenge (phone photo → catalogue scan). "
            "Only global-embedding models (A-series) evaluated here.",
        ),
        (
            "phone-sample",
            "Phone Photo Evaluation — Sample Set (#53)",
            "55-photo / 50-matched-query sample — same dataset used by C2 sample-mode. "
            "Enables direct apples-to-apples comparison with C2 LightGlue (see summary.csv).",
        ),
    ]
    for ctx_key, title, desc in ctx_specs:
        ctx_rows = by_context.get(ctx_key, [])
        if ctx_rows:
            sections.append(_render_table(title, desc, ctx_rows))

    if not sections:
        sections.append("<p>No evaluation data found.</p>")

    html = _MULTI_CTX_HTML_HEAD + "\n".join(sections) + "</body>\n</html>\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    """Entry point for model comparison report generation."""
    parser = argparse.ArgumentParser(
        description="Generate cross-model comparison report from evaluation results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Top-level results directory (default: results/).",
    )
    parser.add_argument(
        "--latency-csv",
        type=Path,
        default=None,
        help="Path to latency_summary.csv (default: {results-dir}/latency_summary.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path (default: {results-dir}/comparison.html).",
    )
    args = parser.parse_args(argv)

    results_dir: Path = args.results_dir.resolve()
    latency_csv: Path = (
        args.latency_csv.resolve() if args.latency_csv else results_dir / "latency_summary.csv"
    )
    output_html: Path = args.output.resolve() if args.output else results_dir / "comparison.html"
    output_csv = output_html.parent / "comparison.csv"

    summary = _load_summary_csv(results_dir)
    if not summary:
        logger.error("no_summary_csv", path=str(results_dir / "summary.csv"))
        sys.exit(1)

    latency = _load_latency_csv(latency_csv)
    has_latency = bool(latency)

    rows = build_comparison_rows(summary, latency)
    save_comparison_csv(rows, output_csv)
    generate_comparison_html(rows, output_html, has_latency=has_latency)

    logger.info(
        "comparison_complete",
        num_models=len(rows),
        csv=str(output_csv),
        html=str(output_html),
    )
    print(f"\nComparison report: {output_html}")

    # ── Multi-context report (generated if any phone eval CSV exists) ─────
    phone_complete = _load_csv_by_model(results_dir / "phone_eval_summary.csv")
    phone_sample = _load_csv_by_model(results_dir / "phone_sample_eval_summary.csv")
    if phone_complete or phone_sample:
        mc_rows = build_multi_context_rows(summary, phone_complete, phone_sample)
        mc_csv = results_dir / "multi_context_comparison.csv"
        mc_html = results_dir / "multi_context_comparison.html"
        save_multi_context_csv(mc_rows, mc_csv)
        generate_multi_context_html(mc_rows, mc_html)
        logger.info(
            "multi_context_comparison_complete",
            num_rows=len(mc_rows),
            csv=str(mc_csv),
            html=str(mc_html),
        )
        print(f"Multi-context report: {mc_html}")


if __name__ == "__main__":
    main()
