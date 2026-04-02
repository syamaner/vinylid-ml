"""HTML evaluation report generation with interactive Plotly charts.

Produces an HTML report (charts require CDN access to plotly.js) for
each evaluation run, including metrics summary, stratified breakdowns,
NN ambiguity histogram, and confidence calibration curve.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownLambdaType=false
# Plotly has no type stubs — all plotly API calls produce "partially unknown" types.

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import structlog

try:
    import plotly.graph_objects as go  # type: ignore[import-untyped]
    from jinja2 import BaseLoader, Environment
    from markupsafe import Markup
except ImportError as exc:
    _msg = "Report generation requires the 'eval' extra. Install with: pip install -e '.[eval]'"
    raise ImportError(_msg) from exc

if TYPE_CHECKING:
    from vinylid_ml.eval_metrics import (
        CalibrationResult,
        NNAmbiguityResult,
        RetrievalMetrics,
        StratifiedMetrics,
    )

__all__ = ["generate_report"]

logger = structlog.get_logger()

# Jinja2 template for the HTML report — embedded inline so the module
# has no file-system dependencies beyond its own imports.
_REPORT_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>VinylID Evaluation — {{ model_id }}</title>
    <style>
        :root {
            --bg: #0f1117;
            --surface: #1a1d28;
            --border: #2d3040;
            --text: #e4e4e7;
            --text-muted: #9ca3af;
            --accent: #6366f1;
            --accent-light: #818cf8;
            --green: #34d399;
            --amber: #fbbf24;
            --red: #f87171;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 { font-size: 1.75rem; margin-bottom: 0.5rem; }
        h2 {
            font-size: 1.25rem;
            color: var(--accent-light);
            margin: 2rem 0 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
        }
        .meta {
            color: var(--text-muted);
            font-size: 0.875rem;
            margin-bottom: 2rem;
        }
        .meta span { margin-right: 1.5rem; }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.25rem;
            text-align: center;
        }
        .metric-card .value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--green);
        }
        .metric-card .label {
            font-size: 0.8rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.25rem;
        }
        .chart-container {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1.5rem;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 1rem;
        }
        th, td {
            padding: 0.5rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        th {
            color: var(--text-muted);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        td { font-variant-numeric: tabular-nums; }
        .footer {
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border);
            color: var(--text-muted);
            font-size: 0.75rem;
            text-align: center;
        }
    </style>
</head>
<body>
    <h1>🎵 VinylID Evaluation Report</h1>
    <div class="meta">
        <span><strong>Model:</strong> {{ model_id }}</span>
        <span><strong>Split:</strong> {{ split }}</span>
        <span><strong>Timestamp:</strong> {{ timestamp }}</span>
        {% if git_sha %}<span><strong>Git SHA:</strong> {{ git_sha[:8] }}</span>{% endif %}
    </div>

    <h2>Retrieval Metrics</h2>
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="value">{{ "%.1f"|format(metrics.recall_at_1 * 100) }}%</div>
            <div class="label">Recall@1</div>
        </div>
        <div class="metric-card">
            <div class="value">{{ "%.1f"|format(metrics.recall_at_5 * 100) }}%</div>
            <div class="label">Recall@5</div>
        </div>
        <div class="metric-card">
            <div class="value">{{ "%.3f"|format(metrics.map_at_5) }}</div>
            <div class="label">mAP@5</div>
        </div>
        <div class="metric-card">
            <div class="value">{{ "%.3f"|format(metrics.mrr) }}</div>
            <div class="label">MRR</div>
        </div>
        <div class="metric-card">
            <div class="value">{{ num_gallery }}</div>
            <div class="label">Gallery</div>
        </div>
        <div class="metric-card">
            <div class="value">{{ metrics.num_queries }}</div>
            <div class="label">Queries</div>
        </div>
    </div>

    {% if stratified_html %}
    <h2>Stratified Metrics</h2>
    {{ stratified_html }}
    {% endif %}

    {% if nn_ambiguity_html %}
    <h2>Nearest-Neighbor Ambiguity</h2>
    <p style="color: var(--text-muted); font-size: 0.875rem; margin-bottom: 1rem;">
        Distribution of max cosine similarity to a different-album gallery item.
        Higher values indicate greater false-positive risk.
    </p>
    <div class="chart-container">{{ nn_ambiguity_html }}</div>
    {% endif %}

    {% if calibration_html %}
    <h2>Confidence Calibration</h2>
    <p style="color: var(--text-muted); font-size: 0.875rem; margin-bottom: 1rem;">
        Top-1 similarity score vs. actual accuracy. Points on the diagonal
        indicate perfect calibration.
    </p>
    <div class="chart-container">{{ calibration_html }}</div>
    {% endif %}

    <div class="footer">
        Generated by VinylID ML evaluation framework
    </div>
</body>
</html>
"""

_PLOTLY_LAYOUT_DEFAULTS: dict[str, object] = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(26,29,40,1)",
    "font": {"color": "#e4e4e7", "family": "system-ui, sans-serif"},
    "margin": {"l": 50, "r": 30, "t": 40, "b": 50},
}


def _build_stratified_chart(stratified: StratifiedMetrics) -> str:
    """Build grouped bar charts for stratified metrics.

    Args:
        stratified: StratifiedMetrics with breakdown data.

    Returns:
        HTML string with Plotly chart divs, or empty string if no data.
    """
    charts: list[str] = []

    if stratified.by_album_image_count:
        fig = go.Figure()
        buckets = list(stratified.by_album_image_count.keys())
        values = [stratified.by_album_image_count[b] * 100 for b in buckets]
        fig.add_trace(
            go.Bar(
                x=buckets,
                y=values,
                marker_color="#6366f1",
                text=[f"{v:.1f}%" for v in values],
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Recall@1 by Album Image Count",
            xaxis_title="Images per Album",
            yaxis_title="Recall@1 (%)",
            yaxis_range=[0, 105],
            **_PLOTLY_LAYOUT_DEFAULTS,  # type: ignore[arg-type]
        )
        charts.append(
            '<div class="chart-container">'
            + fig.to_html(full_html=False, include_plotlyjs="cdn")
            + "</div>"
        )

    if stratified.by_resolution_bucket:
        fig = go.Figure()
        buckets = list(stratified.by_resolution_bucket.keys())
        values = [stratified.by_resolution_bucket[b] * 100 for b in buckets]
        fig.add_trace(
            go.Bar(
                x=buckets,
                y=values,
                marker_color="#818cf8",
                text=[f"{v:.1f}%" for v in values],
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Recall@1 by Source Resolution",
            xaxis_title="Resolution Bucket",
            yaxis_title="Recall@1 (%)",
            yaxis_range=[0, 105],
            **_PLOTLY_LAYOUT_DEFAULTS,  # type: ignore[arg-type]
        )
        charts.append(
            '<div class="chart-container">'
            + fig.to_html(full_html=False, include_plotlyjs="cdn")
            + "</div>"
        )

    if stratified.by_augmentation_type:
        fig = go.Figure()
        aug_types = list(stratified.by_augmentation_type.keys())
        values = [stratified.by_augmentation_type[a] * 100 for a in aug_types]
        fig.add_trace(
            go.Bar(
                x=aug_types,
                y=values,
                marker_color="#34d399",
                text=[f"{v:.1f}%" for v in values],
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Recall@1 by Augmentation Type",
            xaxis_title="Augmentation",
            yaxis_title="Recall@1 (%)",
            yaxis_range=[0, 105],
            **_PLOTLY_LAYOUT_DEFAULTS,  # type: ignore[arg-type]
        )
        charts.append(
            '<div class="chart-container">'
            + fig.to_html(full_html=False, include_plotlyjs="cdn")
            + "</div>"
        )

    return "\n".join(charts)


def _build_nn_ambiguity_chart(nn_ambiguity: NNAmbiguityResult) -> str:
    """Build overlaid histogram of NN similarities by album image count bucket.

    Args:
        nn_ambiguity: NNAmbiguityResult from compute_nn_ambiguity().

    Returns:
        Plotly HTML div string.
    """
    fig = go.Figure()

    # Bucket by album image count: 1, 2-5, 6+
    bucket_defs = [
        ("1 image", lambda c: c == 1, "#6366f1"),
        ("2-5 images", lambda c: 2 <= c <= 5, "#818cf8"),
        ("6+ images", lambda c: c >= 6, "#34d399"),
    ]

    for bucket_name, predicate, color in bucket_defs:
        mask = np.array(
            [
                predicate(nn_ambiguity.album_image_counts.get(int(label), 1))
                for label in nn_ambiguity.labels
            ]
        )
        if mask.any():
            fig.add_trace(
                go.Histogram(
                    x=nn_ambiguity.nn_similarities[mask].tolist(),
                    name=bucket_name,
                    marker_color=color,
                    opacity=0.7,
                    nbinsx=30,
                )
            )

    fig.update_layout(
        title="Nearest-Neighbor Ambiguity Distribution",
        xaxis_title="Max Cosine Similarity to Different Album",
        yaxis_title="Count",
        barmode="overlay",
        **_PLOTLY_LAYOUT_DEFAULTS,  # type: ignore[arg-type]
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _build_calibration_chart(calibration: CalibrationResult) -> str:
    """Build calibration curve with ideal diagonal reference.

    Args:
        calibration: CalibrationResult from compute_confidence_calibration().

    Returns:
        Plotly HTML div string.
    """
    fig = go.Figure()

    # Bin midpoints for x-axis
    bin_midpoints = [
        (float(calibration.bin_edges[i]) + float(calibration.bin_edges[i + 1])) / 2
        for i in range(len(calibration.bin_accuracies))
    ]

    # Filter out empty bins
    valid_mask = ~np.isnan(calibration.bin_accuracies)
    valid_x = [x for x, v in zip(bin_midpoints, valid_mask, strict=True) if v]
    valid_y = [float(y) for y, v in zip(calibration.bin_accuracies, valid_mask, strict=True) if v]
    valid_counts = [int(c) for c, v in zip(calibration.bin_counts, valid_mask, strict=True) if v]

    # Derive axis range from actual calibration data (cosine similarity can be [-1, 1])
    x_min = float(calibration.bin_edges[0])
    x_max = float(calibration.bin_edges[-1])

    # Ideal diagonal — perfect calibration: accuracy = score (clamped to [0, 1])
    ideal_lo = max(0.0, x_min)
    ideal_hi = min(1.0, x_max)
    fig.add_trace(
        go.Scatter(
            x=[ideal_lo, ideal_hi],
            y=[ideal_lo, ideal_hi],
            mode="lines",
            line={"dash": "dash", "color": "#4b5563"},
            name="Ideal",
            showlegend=True,
        )
    )

    # Actual calibration curve
    fig.add_trace(
        go.Scatter(
            x=valid_x,
            y=valid_y,
            mode="lines+markers",
            marker={"size": 8, "color": "#6366f1"},
            line={"color": "#6366f1"},
            name="Model",
            text=[f"n={c}" for c in valid_counts],
            hovertemplate="Score: %{x:.3f}<br>Accuracy: %{y:.3f}<br>%{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Confidence Calibration Curve",
        xaxis_title="Top-1 Similarity Score",
        yaxis_title="Accuracy",
        xaxis_range=[x_min - 0.02, x_max + 0.02],
        yaxis_range=[0, 1.05],
        **_PLOTLY_LAYOUT_DEFAULTS,  # type: ignore[arg-type]
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def generate_report(
    output_dir: Path,
    metrics: RetrievalMetrics,
    stratified: StratifiedMetrics,
    nn_ambiguity: NNAmbiguityResult | None,
    calibration: CalibrationResult | None,
    *,
    model_id: str,
    split: str,
    timestamp: str,
    num_gallery: int,
    git_sha: str | None = None,
) -> Path:
    """Generate an HTML evaluation report with interactive Plotly charts.

    Charts load Plotly JS from a CDN and require network access to render
    correctly.

    Args:
        output_dir: Directory to write report.html into.
        metrics: RetrievalMetrics from the evaluation run.
        stratified: StratifiedMetrics with breakdown data.
        nn_ambiguity: Optional NNAmbiguityResult for histogram.
        calibration: Optional CalibrationResult for calibration curve.
        model_id: Model identifier string.
        split: Split name (e.g., "test", "val").
        timestamp: ISO-format timestamp of the run.
        num_gallery: Number of gallery items.
        git_sha: Optional git commit SHA.

    Returns:
        Path to the generated report.html file.
    """
    env = Environment(loader=BaseLoader(), autoescape=True)
    template = env.from_string(_REPORT_TEMPLATE)

    stratified_html = Markup(_build_stratified_chart(stratified))
    nn_ambiguity_html = Markup(_build_nn_ambiguity_chart(nn_ambiguity)) if nn_ambiguity else ""
    calibration_html = Markup(_build_calibration_chart(calibration)) if calibration else ""

    html = template.render(
        model_id=model_id,
        split=split,
        timestamp=timestamp,
        git_sha=git_sha,
        metrics=metrics,
        num_gallery=num_gallery,
        stratified_html=stratified_html,
        nn_ambiguity_html=nn_ambiguity_html,
        calibration_html=calibration_html,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")

    logger.info("report_generated", path=str(report_path))
    return report_path
