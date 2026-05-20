# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Tests for the JAX TransformerLayer benchmark JSON contract."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = REPO_ROOT / "benchmarks" / "jax" / "benchmark_transformer_layer.py"


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location("benchmark_transformer_layer", BENCHMARK_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = _load_benchmark_module()


def test_parse_args_defaults():
    config = benchmark.parse_args([])

    assert config.batch_size == 1
    assert config.sequence_length == 128
    assert config.hidden_size == 512
    assert config.mlp_hidden_size == 2048
    assert config.num_attention_heads == 8
    assert config.warmup_iters == 3
    assert config.iters == 10


def test_parse_args_rejects_hidden_size_not_divisible_by_heads():
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--hidden-size", "130", "--num-attention-heads", "8"])


def test_build_report_has_stable_json_contract(monkeypatch):
    monkeypatch.setenv("NVTE_FRAMEWORK", "jax")
    config = benchmark.BenchmarkConfig(warmup_iters=1, iters=3)
    report = benchmark.build_report(
        config,
        step_times_ms=[1.0, 2.0, 3.0],
        device_info={"backend": "gpu", "platform": "gpu", "device_kind": "test"},
    )

    assert report["schema_version"] == "te_jax_transformer_layer_benchmark/v1"
    assert report["benchmark"] == "jax_transformer_layer"
    assert report["framework"] == "jax"
    assert report["mode"] == "forward_backward"
    assert report["environment"]["NVTE_FRAMEWORK"] == "jax"
    assert report["config"]["iters"] == 3
    assert report["config"]["dtype"] == "bfloat16"
    assert report["metrics"]["mean_step_ms"] == pytest.approx(2.0)
    assert report["metrics"]["median_step_ms"] == pytest.approx(2.0)
    assert report["metrics"]["tokens_per_second"] > 0.0
    assert report["measurements"][0]["case_id"] == "jax_transformer_layer_bf16_forward_backward"

    json.dumps(report, allow_nan=False)


def test_emit_report_writes_same_json_to_stdout_and_file(tmp_path, capsys):
    output_json = tmp_path / "report.json"
    report = {
        "schema_version": "te_jax_transformer_layer_benchmark/v1",
        "benchmark": "jax_transformer_layer",
        "metrics": {"mean_step_ms": 1.0},
    }

    benchmark.emit_report(report, str(output_json))

    stdout_report = json.loads(capsys.readouterr().out)
    file_report = json.loads(output_json.read_text(encoding="utf-8"))
    assert stdout_report == report
    assert file_report == report


def test_benchmark_script_has_no_pytorch_imports():
    source = BENCHMARK_PATH.read_text(encoding="utf-8")

    forbidden_terms = [
        "import torch",
        "from torch",
        "transformer_engine.pytorch",
        "transformer_engine_torch",
    ]
    for term in forbidden_terms:
        assert term not in source
