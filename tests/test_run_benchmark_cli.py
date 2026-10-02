import os
import subprocess
import sys

import pandas as pd
import pytest

SCRIPT = os.path.join("scripts", "run_benchmark.py")


def _skip_if_no_labour():
    if not os.path.exists(os.path.join("dataset", "labour", "meta.json")):
        pytest.skip("dataset/labour not converted yet - run scripts/convert_labour.py")


def test_cli_runs_and_prints_a_leaderboard():
    _skip_if_no_labour()
    result = subprocess.run(
        [sys.executable, SCRIPT, "--dataset", "labour", "--methods", "naive", "seasonal_naive", "--horizon", "8"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "naive" in result.stdout
    assert "seasonal_naive" in result.stdout
    assert "mae" in result.stdout


def test_cli_saves_leaderboard_csv(tmp_path):
    _skip_if_no_labour()
    out_path = tmp_path / "leaderboard.csv"
    result = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--dataset",
            "labour",
            "--methods",
            "naive",
            "--horizon",
            "8",
            "--save-path",
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert out_path.exists()
    df = pd.read_csv(out_path, index_col=0)
    assert list(df.index) == ["naive"]


def test_cli_min_trace_requires_no_extra_flags():
    # --reconcile min_trace should imply --all-levels on its own.
    _skip_if_no_labour()
    result = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--dataset",
            "labour",
            "--methods",
            "naive",
            "--horizon",
            "8",
            "--reconcile",
            "min_trace",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "naive" in result.stdout


def test_cli_rejects_unknown_method():
    result = subprocess.run(
        [sys.executable, SCRIPT, "--dataset", "labour", "--methods", "nope", "--horizon", "8"],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "invalid choice" in result.stderr
