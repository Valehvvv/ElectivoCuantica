#!/usr/bin/env python3
"""One-shot migration: move legacy results/ aside, create fresh empty dir.

Previous results/ directory contained malformed/incomplete runs from
before the instrumentation work. This script:
1. Moves legacy results/ to results.legacy.<timestamp>/ (safety backup)
2. Creates a fresh empty results/ directory
3. Writes MIGRATION_DONE.flag documenting what happened
"""

import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-backup", action="store_true",
                        help="Delete instead of move")
    args = parser.parse_args()

    results = PROJECT_ROOT / "results"
    flag = results / "MIGRATION_DONE.flag"

    if flag.exists():
        print(f"Already migrated (flag at {flag}). Skip.")
        return 0

    if not results.exists():
        results.mkdir(parents=True)
        flag.write_text(f"migration_done_at={datetime.now().isoformat()}\nmode=no_legacy_found\n")
        print("Created fresh results/.")
        return 0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.no_backup:
        shutil.rmtree(results)
        mode = "deleted"
    else:
        backup = PROJECT_ROOT / f"results.legacy.{ts}"
        shutil.move(str(results), str(backup))
        print(f"Moved legacy to {backup}")
        mode = "moved"

    results.mkdir()
    flag.write_text(f"migration_done_at={datetime.now().isoformat()}\nmode={mode}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())