# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import importlib.util
import json
from pathlib import Path


def _load_smoke_benchmark():
    repo_root = Path(__file__).resolve().parents[2]
    benchmark_path = repo_root / "benchmarks" / "smoke_benchmark.py"
    spec = importlib.util.spec_from_file_location("te_smoke_benchmark", benchmark_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parser_defaults_match_smoke_contract():
    smoke_benchmark = _load_smoke_benchmark()

    args = smoke_benchmark.build_parser().parse_args(["--output", "/tmp/report.json"])

    assert args.iterations == 10
    assert args.warmup_iterations == 3
    assert args.batch_size == 32
    assert args.in_features == 128
    assert args.out_features == 128
    assert args.dtype == "bfloat16"
    assert args.device == "cuda"
    assert not args.profile


def test_report_shape_is_stable():
    smoke_benchmark = _load_smoke_benchmark()
    config = {
        "warmup_iterations": 3,
        "iterations": 10,
        "batch_size": 32,
        "in_features": 128,
        "out_features": 128,
        "dtype": "bfloat16",
        "device": "cuda",
        "seed": 1234,
        "profile": False,
    }

    report = smoke_benchmark.make_report(
        status="passed",
        config=config,
        metrics={
            "latency_ms_mean": 1.0,
            "latency_ms_median": 1.0,
            "latency_ms_min": 1.0,
            "latency_ms_max": 1.0,
            "samples_per_second": 32000.0,
        },
        device={"type": "cuda", "name": "test"},
        environment={"python": "test", "torch": "test", "transformer_engine": "test"},
    )

    assert report["schema_version"] == "te_benchmark_smoke/v1"
    assert report["benchmark"] == "pytorch_linear_forward_smoke"
    assert report["status"] == "passed"
    assert set(report) == {
        "schema_version",
        "benchmark",
        "status",
        "device",
        "config",
        "metrics",
        "environment",
    }


def test_write_report_creates_machine_readable_json(tmp_path):
    smoke_benchmark = _load_smoke_benchmark()
    output_path = tmp_path / "nested" / "report.json"
    report = smoke_benchmark.make_report(
        status="skipped",
        config={"iterations": 10},
        environment={"python": "test", "torch": "unavailable", "transformer_engine": "unavailable"},
        reason="CUDA is not available to PyTorch.",
    )

    smoke_benchmark.write_report(report, output_path)

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded == report
    assert output_path.read_text(encoding="utf-8").endswith("\n")
