# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Tests for the JAX TransformerLayer benchmark CLI helpers."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "benchmarks" / "jax" / "benchmark_transformer_layer.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.jax import benchmark_transformer_layer as benchmark  # noqa: E402


def test_requires_explicit_jax_framework():
    """The environment guard should fail before framework imports."""
    with pytest.raises(SystemExit, match="NVTE_FRAMEWORK=jax"):
        benchmark.require_jax_framework({})


def test_missing_framework_exits_without_stdout():
    """The CLI should keep stdout parseable by sending guard errors to stderr."""
    env = os.environ.copy()
    env.pop("NVTE_FRAMEWORK", None)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--warmup-iters", "0", "--iters", "1"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert "NVTE_FRAMEWORK=jax" in result.stderr


def test_timing_summary_rejects_non_positive_samples():
    with pytest.raises(ValueError, match="positive finite"):
        benchmark.timing_summary([1.0, 0.0])


def test_write_json_report_creates_parent_and_mirrors_stdout(tmp_path, capsys):
    output_path = tmp_path / "nested" / "report.json"
    report = {"schema_version": benchmark.SCHEMA_VERSION, "timing_ms": {"mean": 1.25}}

    benchmark.write_json_report(report, str(output_path))

    written = json.loads(output_path.read_text(encoding="utf-8"))
    mirrored = json.loads(capsys.readouterr().out)
    assert written == report
    assert mirrored == report


def test_parse_args_documents_smoke_defaults():
    args = benchmark.parse_args([])

    assert args.batch_size == 2
    assert args.seq_len == 128
    assert args.hidden_size == 512
    assert args.mlp_hidden_size == 2048
    assert args.num_heads == 8
    assert args.dtype == "bfloat16"
    assert args.warmup_iters == 3
    assert args.iters == 10
