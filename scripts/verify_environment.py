"""Verifica a toolchain e registra um baseline reproduzível da suíte.

Execute com ``uv run python scripts/verify_environment.py``. O script usa o
mesmo interpretador do ambiente do projeto, roda a suíte completa e grava um
relatório JSON em ``outputs/environment_baseline.json``.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "outputs" / "environment_baseline.json"
LOCK_PATH = ROOT / "uv.lock"
PYTEST_SUMMARY_PATTERN = re.compile(
    r"(?P<passed>\d+) passed(?:, (?P<skipped>\d+) skipped)? in "
    r"(?P<duration>[\d.]+)s"
)


def _command_version(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> int:
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=short"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    duration_seconds = round(time.perf_counter() - started, 3)
    print(completed.stdout, end="")
    summary_match = PYTEST_SUMMARY_PATTERN.search(completed.stdout)

    report = {
        "schema_version": 1,
        "recorded_at": started_at.isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "uv_version": _command_version(["uv", "--version"]),
        "uv_lock_sha256": hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest(),
        "pytest_exit_code": completed.returncode,
        "tests_passed": int(summary_match.group("passed")) if summary_match else None,
        "tests_skipped": int(summary_match.group("skipped") or 0) if summary_match else None,
        "pytest_reported_duration_seconds": (
            float(summary_match.group("duration")) if summary_match else None
        ),
        "duration_seconds": duration_seconds,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Baseline gravado em {OUTPUT_PATH}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
