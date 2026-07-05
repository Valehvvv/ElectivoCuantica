"""Tests for CSV append writers."""

from __future__ import annotations

from pathlib import Path

import pytest

from persistence.csv_writer import append_results_row, append_predictions_row


class TestResultsCSV:
    def test_creates_file_with_headers(self, tmp_path: Path) -> None:
        p = tmp_path / "results.csv"
        append_results_row(p, 0, 0.5, [0.1, -0.2], accuracy=0.75)
        assert p.exists()
        text = p.read_text()
        assert "iteration" in text.splitlines()[0]
        assert "0" in text.splitlines()[1]

    def test_appends_multiple_rows(self, tmp_path: Path) -> None:
        p = tmp_path / "results.csv"
        for i in range(3):
            append_results_row(p, i, float(i) * 0.1, [float(i)])
        lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
        assert len(lines) == 4  # header + 3 rows

    def test_extra_fields(self, tmp_path: Path) -> None:
        p = tmp_path / "results.csv"
        append_results_row(p, 0, 0.5, [], extra={"foo": "bar"})
        text = p.read_text()
        assert "foo" in text


class TestPredictionsCSV:
    def test_creates_file(self, tmp_path: Path) -> None:
        p = tmp_path / "predictions.csv"
        append_predictions_row(p, 0, 0, 1, 0, 0.85)
        assert p.exists()
        text = p.read_text()
        assert "label_true" in text.splitlines()[0]
        assert "1" in text.splitlines()[1]

    def test_appends(self, tmp_path: Path) -> None:
        p = tmp_path / "predictions.csv"
        for si in range(3):
            append_predictions_row(p, 0, si, 1, 1, 0.9)
        lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
        assert len(lines) == 4  # header + 3 rows