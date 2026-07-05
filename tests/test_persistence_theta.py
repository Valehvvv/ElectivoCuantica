"""Tests for theta persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from persistence.theta import save_theta, load_theta, latest_theta


class TestSaveLoad:
    def test_roundtrip(self, tmp_path: Path) -> None:
        theta = np.array([0.1, 0.2, 0.3, -0.5])
        path = save_theta(tmp_path, "ansatz_base", 0, theta, cost=1.5)
        assert path.exists()
        loaded = load_theta(path)
        np.testing.assert_array_almost_equal(loaded["theta"], theta)
        assert loaded["iteration"] == 0
        assert loaded["ansatz"] == "ansatz_base"
        assert loaded["cost"] == 1.5

    def test_without_cost(self, tmp_path: Path) -> None:
        theta = np.array([0.1, 0.2])
        path = save_theta(tmp_path, "hea", 5, theta)
        loaded = load_theta(path)
        assert "cost" not in loaded

    def test_load_nonexistent(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_theta(Path("/tmp/nonexistent.npz"))

    def test_latest_theta(self, tmp_path: Path) -> None:
        t1 = np.array([1.0])
        t2 = np.array([2.0])
        save_theta(tmp_path, "hea", 0, t1)
        save_theta(tmp_path, "hea", 1, t2)
        latest = latest_theta(tmp_path, "hea")
        assert latest is not None
        assert latest["iteration"] == 1
        np.testing.assert_array_equal(latest["theta"], t2)

    def test_latest_theta_none(self, tmp_path: Path) -> None:
        assert latest_theta(tmp_path, "nonexistent") is None