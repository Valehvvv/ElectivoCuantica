"""Tests for migration script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT_ROOT / "scripts" / "migrate_legacy_results.py"


def test_moves_legacy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = tmp_path / "proj"; fake.mkdir()
    (fake / "scripts").mkdir()
    shutil.copy(str(SCRIPT), str(fake / "scripts" / "migrate_legacy_results.py"))
    (fake / "results").mkdir()
    (fake / "results" / "data.csv").write_text("a,1\n")

    result = subprocess.run([sys.executable, str(fake / "scripts" / "migrate_legacy_results.py")],
                           capture_output=True, text=True, cwd=fake)
    assert result.returncode == 0
    assert (fake / "results" / "MIGRATION_DONE.flag").exists()
    backups = list(fake.glob("results.legacy.*"))
    assert len(backups) == 1


def test_idempotent(tmp_path: Path) -> None:
    fake = tmp_path / "proj"; fake.mkdir()
    (fake / "scripts").mkdir()
    shutil.copy(str(SCRIPT), str(fake / "scripts" / "migrate_legacy_results.py"))
    (fake / "results").mkdir()
    for _ in range(2):
        r = subprocess.run([sys.executable, str(fake / "scripts" / "migrate_legacy_results.py")],
                          capture_output=True, text=True, cwd=fake)
        assert r.returncode == 0