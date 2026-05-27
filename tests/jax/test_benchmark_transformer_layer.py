# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Tests for the JAX TransformerLayer benchmark CLI helpers."""

import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "benchmarks" / "jax" / "benchmark_transformer_layer.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.jax import benchmark_transformer_layer as benchmark  # noqa: E402


def test_dtype_from_name_maps_supported_names():
    bf16 = object()
    f16 = object()
    f32 = object()
    jnp = types.SimpleNamespace(bfloat16=bf16, float16=f16, float32=f32)

    assert benchmark.dtype_from_name(jnp, "bfloat16") is bf16
    assert benchmark.dtype_from_name(jnp, "float16") is f16
    assert benchmark.dtype_from_name(jnp, "float32") is f32


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


def test_run_benchmark_constructs_report_measurements(monkeypatch):
    """Report construction should happen after timing data is available."""

    class FakeDevice:
        platform = "gpu"
        device_kind = "UnitTest GPU"

    class FakeTransformerLayer:
        def __init__(self, **kwargs):
            assert kwargs["dtype"] == "f16"

        def init(self, rngs, inputs, deterministic):
            assert deterministic
            return {"params": rngs["params"], "shape": inputs["shape"]}

        def apply(self, params, batch, deterministic):
            assert deterministic
            return {"output": batch, "params": params}

    class FakeTransformerLayerType:
        ENCODER = "encoder"

    class FakeAutocast:
        def __enter__(self):
            return None

        def __exit__(self, _exc_type, _exc, _traceback):
            return False

    def block_until_ready(_output):
        return None

    jnp = types.ModuleType("jax.numpy")
    jnp.bfloat16 = "bf16"
    jnp.float16 = "f16"
    jnp.float32 = "f32"

    jax = types.ModuleType("jax")
    jax.__path__ = []
    jax.__version__ = "test-jax"
    jax.numpy = jnp
    jax.default_backend = lambda: "gpu"
    jax.devices = lambda: [FakeDevice()]
    jax.local_device_count = lambda: 1
    jax.block_until_ready = block_until_ready
    jax.jit = lambda function: function
    jax.random = types.SimpleNamespace(
        PRNGKey=lambda seed: ("key", seed),
        split=lambda key: ("params_key", "input_key"),
        normal=lambda key, shape, dtype: {"key": key, "shape": shape, "dtype": dtype},
    )

    flax = types.ModuleType("flax")
    flax.__version__ = "test-flax"

    te = types.ModuleType("transformer_engine")
    te_jax = types.ModuleType("transformer_engine.jax")
    te_jax.__path__ = []
    te_jax.MeshResource = type("FakeMeshResource", (), {})
    te_jax.autocast = lambda **kwargs: FakeAutocast()
    te_jax_flax = types.ModuleType("transformer_engine.jax.flax")
    te_jax_flax.TransformerLayer = FakeTransformerLayer
    te_jax_flax.TransformerLayerType = FakeTransformerLayerType
    te.jax = te_jax
    te_jax.flax = te_jax_flax

    monkeypatch.setitem(sys.modules, "jax", jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", jnp)
    monkeypatch.setitem(sys.modules, "flax", flax)
    monkeypatch.setitem(sys.modules, "transformer_engine", te)
    monkeypatch.setitem(sys.modules, "transformer_engine.jax", te_jax)
    monkeypatch.setitem(sys.modules, "transformer_engine.jax.flax", te_jax_flax)
    monkeypatch.setattr(
        benchmark,
        "package_version",
        lambda package_name: {
            "jaxlib": "test-jaxlib",
            "transformer_engine": "test-transformer-engine",
        }[package_name],
    )

    times = iter([10.0, 10.001, 20.0, 20.004])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))

    args = benchmark.parse_args(
        [
            "--batch-size",
            "2",
            "--seq-len",
            "4",
            "--hidden-size",
            "8",
            "--mlp-hidden-size",
            "16",
            "--num-heads",
            "2",
            "--dtype",
            "float16",
            "--warmup-iters",
            "0",
            "--iters",
            "2",
            "--case-id",
            "unit_case",
        ]
    )

    report = benchmark.run_benchmark(args)

    assert report["case_id"] == "unit_case"
    assert report["config"]["dtype"] == "float16"
    assert report["device"] == {
        "jax_backend": "gpu",
        "platform": "gpu",
        "device_kind": "UnitTest GPU",
        "local_device_count": 1,
    }
    assert report["versions"] == {
        "jax": "test-jax",
        "jaxlib": "test-jaxlib",
        "flax": "test-flax",
        "transformer_engine": "test-transformer-engine",
    }
    assert report["timing_ms"]["iterations"] == pytest.approx([1.0, 4.0])
    assert report["timing_ms"]["mean"] == pytest.approx(2.5)
    assert report["throughput"]["tokens_per_iteration"] == 8
    assert report["throughput"]["tokens_per_second"] == pytest.approx(3200.0)

    measurements = report["measurements"]
    assert [measurement["metric"] for measurement in measurements] == [
        "latency_ms",
        "latency_ms",
        "latency_ms_mean",
        "tokens_per_second",
    ]
    assert [measurement["case_id"] for measurement in measurements] == ["unit_case"] * 4
    assert [measurement["iteration"] for measurement in measurements] == [0, 1, 2, 2]
    assert measurements[0]["value"] == pytest.approx(1.0)
    assert measurements[1]["value"] == pytest.approx(4.0)
    assert measurements[2]["value"] == pytest.approx(2.5)
    assert measurements[3]["value"] == pytest.approx(3200.0)
    assert measurements[0]["higher_is_better"] is False
    assert measurements[3]["higher_is_better"] is True


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
