"""Tests for scripts/compare_models.py."""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest

from scripts.compare_models import (
    build_comparison_rows,
    decision_verdict,
    generate_comparison_html,
    save_comparison_csv,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────

_FIXTURE_SUMMARY_ROWS: list[dict[str, str]] = [
    {
        "model_id": "A1-dinov2-cls",
        "timestamp": "2026-03-14T10-00-00",
        "recall_at_1": "0.92",
        "recall_at_5": "0.97",
        "map_at_5": "0.94",
        "mrr": "0.95",
        "num_gallery": "855",
        "num_queries": "4508",
    },
    {
        "model_id": "A4-sscd",
        "timestamp": "2026-03-14T11-00-00",
        "recall_at_1": "0.85",
        "recall_at_5": "0.93",
        "map_at_5": "0.88",
        "mrr": "0.89",
        "num_gallery": "855",
        "num_queries": "4508",
    },
    {
        "model_id": "A2-openclip",
        "timestamp": "2026-03-14T12-00-00",
        "recall_at_1": "0.72",
        "recall_at_5": "0.88",
        "map_at_5": "0.78",
        "mrr": "0.80",
        "num_gallery": "855",
        "num_queries": "4508",
    },
]


# ── decision_verdict ───────────────────────────────────────────────────────────


class TestDecisionVerdict:
    """Tests for decision_verdict()."""

    @pytest.mark.parametrize(
        "recall,expected",
        [
            (1.00, "zero-shot sufficient"),
            (0.90, "zero-shot sufficient"),
            (0.91, "zero-shot sufficient"),
            (0.899, "fine-tuning likely needed"),
            (0.80, "fine-tuning likely needed"),
            (0.84, "fine-tuning likely needed"),
            (0.799, "fine-tuning required"),
            (0.50, "fine-tuning required"),
            (0.00, "fine-tuning required"),
        ],
    )
    def test_thresholds(self, recall: float, expected: str) -> None:
        assert decision_verdict(recall) == expected

    def test_exact_boundary_90(self) -> None:
        assert decision_verdict(0.9) == "zero-shot sufficient"

    def test_exact_boundary_80(self) -> None:
        assert decision_verdict(0.8) == "fine-tuning likely needed"


# ── build_comparison_rows ──────────────────────────────────────────────────────


class TestBuildComparisonRows:
    """Tests for build_comparison_rows()."""

    def _summary_from_fixtures(self) -> dict[str, dict[str, str]]:
        return {r["model_id"]: r for r in _FIXTURE_SUMMARY_ROWS}

    def test_row_count_matches_summary(self) -> None:
        rows = build_comparison_rows(self._summary_from_fixtures(), {})
        assert len(rows) == 3

    def test_sorted_by_recall_at_1_descending(self) -> None:
        rows = build_comparison_rows(self._summary_from_fixtures(), {})
        recalls = [r["recall_at_1"] for r in rows]
        assert recalls == sorted([r for r in recalls if r is not None], reverse=True)

    def test_verdict_applied_correctly(self) -> None:
        rows = build_comparison_rows(self._summary_from_fixtures(), {})
        by_model = {r["model_id"]: r for r in rows}
        assert by_model["A1-dinov2-cls"]["verdict"] == "zero-shot sufficient"
        assert by_model["A4-sscd"]["verdict"] == "fine-tuning likely needed"
        assert by_model["A2-openclip"]["verdict"] == "fine-tuning required"

    def test_latency_included_when_provided(self) -> None:
        latency = {
            "A1-dinov2-cls": {"p50_ms": 12.0, "p95_ms": 15.0, "p99_ms": 18.0},
        }
        rows = build_comparison_rows(self._summary_from_fixtures(), latency)
        by_model = {r["model_id"]: r for r in rows}
        assert by_model["A1-dinov2-cls"]["latency_p50_ms"] == pytest.approx(12.0)
        assert by_model["A4-sscd"]["latency_p50_ms"] is None  # no latency for this model

    def test_latency_none_when_not_provided(self) -> None:
        rows = build_comparison_rows(self._summary_from_fixtures(), {})
        for r in rows:
            assert r["latency_p50_ms"] is None
            assert r["latency_p95_ms"] is None

    def test_recall_values_are_floats(self) -> None:
        rows = build_comparison_rows(self._summary_from_fixtures(), {})
        for r in rows:
            assert isinstance(r["recall_at_1"], float)
            assert isinstance(r["recall_at_5"], float)

    def test_empty_summary_returns_empty_list(self) -> None:
        assert build_comparison_rows({}, {}) == []


# ── save_comparison_csv ────────────────────────────────────────────────────────


class TestSaveComparisonCsv:
    """Tests for save_comparison_csv()."""

    def test_creates_file_with_correct_columns(self, tmp_path: Path) -> None:
        summary = {r["model_id"]: r for r in _FIXTURE_SUMMARY_ROWS}
        rows = build_comparison_rows(summary, {})
        out = tmp_path / "comparison.csv"
        save_comparison_csv(rows, out)

        assert out.exists()
        with out.open(newline="") as f:
            reader = csv.DictReader(f)
            assert "model_id" in (reader.fieldnames or [])
            assert "recall_at_1" in (reader.fieldnames or [])
            assert "verdict" in (reader.fieldnames or [])
            data = list(reader)
        assert len(data) == 3

    def test_creates_parent_dir(self, tmp_path: Path) -> None:
        """Parent directory is created if it does not exist."""
        out = tmp_path / "nested" / "dir" / "comparison.csv"
        save_comparison_csv([], out)
        assert out.parent.is_dir()

    def test_sorted_by_recall_in_output(self, tmp_path: Path) -> None:
        summary = {r["model_id"]: r for r in _FIXTURE_SUMMARY_ROWS}
        rows = build_comparison_rows(summary, {})
        out = tmp_path / "comparison.csv"
        save_comparison_csv(rows, out)

        with out.open(newline="") as f:
            data = list(csv.DictReader(f))
        recalls = [float(r["recall_at_1"]) for r in data if r["recall_at_1"]]
        assert recalls == sorted(recalls, reverse=True)


# ── generate_comparison_html ───────────────────────────────────────────────────


class TestGenerateComparisonHtml:
    """Tests for generate_comparison_html()."""

    def _make_rows(self) -> list[dict[str, Any]]:
        summary = {r["model_id"]: r for r in _FIXTURE_SUMMARY_ROWS}
        return build_comparison_rows(summary, {})

    def test_creates_html_file(self, tmp_path: Path) -> None:
        rows = self._make_rows()
        out = tmp_path / "comparison.html"
        generate_comparison_html(rows, out)
        assert out.exists()

    def test_html_contains_all_model_ids(self, tmp_path: Path) -> None:
        rows = self._make_rows()
        out = tmp_path / "comparison.html"
        generate_comparison_html(rows, out)
        html = out.read_text()
        assert "A1-dinov2-cls" in html
        assert "A4-sscd" in html
        assert "A2-openclip" in html

    def test_html_contains_recall_values(self, tmp_path: Path) -> None:
        rows = self._make_rows()
        out = tmp_path / "comparison.html"
        generate_comparison_html(rows, out)
        html = out.read_text()
        assert "0.9200" in html  # A1 Recall@1
        assert "0.7200" in html  # A2 Recall@1

    def test_html_colour_codes_applied(self, tmp_path: Path) -> None:
        rows = self._make_rows()
        out = tmp_path / "comparison.html"
        generate_comparison_html(rows, out)
        html = out.read_text()
        assert "r1-green" in html  # A1 ≥ 0.90
        assert "r1-amber" in html  # A4 0.80-0.90
        assert "r1-red" in html  # A2 < 0.80

    def test_html_verdict_classes(self, tmp_path: Path) -> None:
        rows = self._make_rows()
        out = tmp_path / "comparison.html"
        generate_comparison_html(rows, out)
        html = out.read_text()
        assert "v-ok" in html
        assert "v-likely" in html
        assert "v-req" in html

    def test_latency_columns_absent_when_not_provided(self, tmp_path: Path) -> None:
        rows = self._make_rows()
        out = tmp_path / "comparison.html"
        generate_comparison_html(rows, out, has_latency=False)
        html = out.read_text()
        assert "p50" not in html

    def test_latency_columns_present_when_provided(self, tmp_path: Path) -> None:
        latency = {"A1-dinov2-cls": {"p50_ms": 12.0, "p95_ms": 15.0, "p99_ms": 18.0}}
        summary = {r["model_id"]: r for r in _FIXTURE_SUMMARY_ROWS}
        rows = build_comparison_rows(summary, latency)
        out = tmp_path / "comparison.html"
        generate_comparison_html(rows, out, has_latency=True)
        html = out.read_text()
        assert "p50" in html

    def test_html_is_valid_doctype(self, tmp_path: Path) -> None:
        rows = self._make_rows()
        out = tmp_path / "comparison.html"
        generate_comparison_html(rows, out)
        html = out.read_text()
        assert html.startswith("<!DOCTYPE html>")
