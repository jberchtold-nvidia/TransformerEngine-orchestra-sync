# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX TransformerLayer benchmark with machine-readable JSON output."""

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Sequence


SCHEMA_VERSION = "te_jax_transformer_layer_benchmark/v1"
BENCHMARK_NAME = "jax_transformer_layer"
CASE_ID = "jax_transformer_layer_bf16_forward_backward"
MODE = "forward_backward"
DTYPE_NAME = "bfloat16"


@dataclass(frozen=True)
class BenchmarkConfig:
    """User-controlled benchmark shape and iteration settings."""

    batch_size: int = 1
    sequence_length: int = 128
    hidden_size: int = 512
    mlp_hidden_size: int = 2048
    num_attention_heads: int = 8
    warmup_iters: int = 3
    iters: int = 10
    seed: int = 0
    output_json: Optional[str] = None
    profile_dir: Optional[str] = None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value!r}")
    return parsed


def validate_config(config: BenchmarkConfig) -> None:
    """Validate cross-field CLI constraints."""

    if config.hidden_size % config.num_attention_heads != 0:
        raise ValueError(
            "--hidden-size must be divisible by --num-attention-heads; "
            f"got {config.hidden_size} and {config.num_attention_heads}"
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> BenchmarkConfig:
    """Parse CLI arguments into a validated benchmark config."""

    parser = argparse.ArgumentParser(
        description="Benchmark transformer_engine.jax.flax.TransformerLayer forward+backward.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch-size", type=_positive_int, default=1)
    parser.add_argument("--sequence-length", type=_positive_int, default=128)
    parser.add_argument("--hidden-size", type=_positive_int, default=512)
    parser.add_argument("--mlp-hidden-size", type=_positive_int, default=2048)
    parser.add_argument("--num-attention-heads", type=_positive_int, default=8)
    parser.add_argument("--warmup-iters", type=_non_negative_int, default=3)
    parser.add_argument("--iters", type=_positive_int, default=10)
    parser.add_argument("--seed", type=_non_negative_int, default=0)
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path that receives the same JSON object printed to stdout.",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default=None,
        help="Optional JAX profiler trace directory. Tracing starts after warmup.",
    )

    args = parser.parse_args(argv)
    config = BenchmarkConfig(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        mlp_hidden_size=args.mlp_hidden_size,
        num_attention_heads=args.num_attention_heads,
        warmup_iters=args.warmup_iters,
        iters=args.iters,
        seed=args.seed,
        output_json=args.output_json,
        profile_dir=args.profile_dir,
    )
    try:
        validate_config(config)
    except ValueError as exc:
        parser.error(str(exc))
    return config


def config_to_json(config: BenchmarkConfig) -> Dict[str, Any]:
    """Return the stable config fields emitted in reports."""

    return {
        "batch_size": config.batch_size,
        "sequence_length": config.sequence_length,
        "hidden_size": config.hidden_size,
        "mlp_hidden_size": config.mlp_hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "dtype": DTYPE_NAME,
        "warmup_iters": config.warmup_iters,
        "iters": config.iters,
        "seed": config.seed,
    }


def summarize_step_times(
    step_times_ms: Sequence[float],
    config: BenchmarkConfig,
) -> Dict[str, float]:
    """Summarize per-iteration timings."""

    if not step_times_ms:
        raise ValueError("at least one measured iteration is required")

    mean_step_ms = statistics.fmean(step_times_ms)
    tokens_per_second = config.batch_size * config.sequence_length / (mean_step_ms / 1000.0)
    return {
        "mean_step_ms": mean_step_ms,
        "median_step_ms": statistics.median(step_times_ms),
        "min_step_ms": min(step_times_ms),
        "max_step_ms": max(step_times_ms),
        "stddev_step_ms": statistics.pstdev(step_times_ms) if len(step_times_ms) > 1 else 0.0,
        "tokens_per_second": tokens_per_second,
    }


def _assert_finite_metrics(metrics: Dict[str, float]) -> None:
    for metric_name, value in metrics.items():
        if not math.isfinite(value):
            raise RuntimeError(f"metric {metric_name} is not finite: {value}")
        if metric_name != "stddev_step_ms" and value <= 0.0:
            raise RuntimeError(f"metric {metric_name} must be positive: {value}")
        if metric_name == "stddev_step_ms" and value < 0.0:
            raise RuntimeError(f"metric {metric_name} must be non-negative: {value}")


def build_measurements(metrics: Dict[str, float]) -> List[Dict[str, Any]]:
    """Create aggregate measurement records for benchmark collectors."""

    return [
        {
            "case_id": CASE_ID,
            "metric": "mean_step_ms",
            "value": metrics["mean_step_ms"],
            "unit": "ms",
            "iteration": 0,
            "higher_is_better": False,
        },
        {
            "case_id": CASE_ID,
            "metric": "median_step_ms",
            "value": metrics["median_step_ms"],
            "unit": "ms",
            "iteration": 0,
            "higher_is_better": False,
        },
        {
            "case_id": CASE_ID,
            "metric": "tokens_per_second",
            "value": metrics["tokens_per_second"],
            "unit": "tokens/s",
            "iteration": 0,
            "higher_is_better": True,
        },
    ]


def build_report(
    config: BenchmarkConfig,
    step_times_ms: Sequence[float],
    device_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the machine-readable benchmark report."""

    metrics = summarize_step_times(step_times_ms, config)
    _assert_finite_metrics(metrics)
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "framework": "jax",
        "mode": MODE,
        "environment": {
            "NVTE_FRAMEWORK": os.environ.get("NVTE_FRAMEWORK", ""),
        },
        "device": device_info,
        "config": config_to_json(config),
        "profile": {
            "enabled": config.profile_dir is not None,
            "profile_dir": config.profile_dir,
            "after_warmup": True,
        },
        "metrics": metrics,
        "measurements": build_measurements(metrics),
        "step_times_ms": list(step_times_ms),
    }


def _load_jax_components():
    import jax
    import jax.numpy as jnp
    import transformer_engine.jax as te
    from transformer_engine.jax.flax import TransformerLayer

    return jax, jnp, te, TransformerLayer


def _device_info(jax_module) -> Dict[str, Any]:
    devices = jax_module.devices()
    if not devices:
        raise RuntimeError("no JAX devices are visible")
    device = devices[0]
    platform = getattr(device, "platform", "unknown")
    if platform != "gpu":
        raise RuntimeError(
            "this benchmark requires a JAX GPU backend; "
            f"first visible device is platform={platform!r}"
        )
    return {
        "backend": platform,
        "platform": platform,
        "device_kind": getattr(device, "device_kind", "unknown"),
        "id": getattr(device, "id", None),
        "process_index": getattr(device, "process_index", None),
    }


def _block_until_ready(jax_module, value: Any) -> None:
    for leaf in jax_module.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _make_step(config: BenchmarkConfig, jax_module, jnp_module, te_module, layer_cls):
    key = jax_module.random.PRNGKey(config.seed)
    params_key, input_key = jax_module.random.split(key)
    inputs = jax_module.random.normal(
        input_key,
        (config.batch_size, config.sequence_length, config.hidden_size),
        dtype=jnp_module.float32,
    ).astype(jnp_module.bfloat16)

    layer = layer_cls(
        hidden_size=config.hidden_size,
        mlp_hidden_size=config.mlp_hidden_size,
        num_attention_heads=config.num_attention_heads,
        dtype=jnp_module.bfloat16,
        self_attn_mask_type="causal",
        self_attn_bias_type="no_bias",
        enable_relative_embedding=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        intermediate_dropout=0.0,
        transpose_batch_sequence=False,
    )

    with te_module.autocast(enabled=False, mesh_resource=te_module.MeshResource()):
        variables = layer.init(
            {"params": params_key, "dropout": params_key},
            inputs,
            deterministic=True,
        )

    def loss_fn(params, batch):
        output = layer.apply({"params": params}, batch, deterministic=True)
        return jnp_module.mean(output.astype(jnp_module.float32))

    step = jax_module.jit(jax_module.value_and_grad(loss_fn, argnums=(0, 1)))
    params = variables["params"]

    def run_step():
        with te_module.autocast(enabled=False, mesh_resource=te_module.MeshResource()):
            return step(params, inputs)

    return run_step


def _run_iterations(
    config: BenchmarkConfig,
    jax_module,
    run_step,
) -> List[float]:
    for _ in range(config.warmup_iters):
        _block_until_ready(jax_module, run_step())

    trace_started = False
    if config.profile_dir is not None:
        Path(config.profile_dir).mkdir(parents=True, exist_ok=True)
        jax_module.profiler.start_trace(config.profile_dir)
        trace_started = True

    step_times_ms: List[float] = []
    try:
        for _ in range(config.iters):
            start_ns = time.perf_counter_ns()
            _block_until_ready(jax_module, run_step())
            end_ns = time.perf_counter_ns()
            step_times_ms.append((end_ns - start_ns) / 1.0e6)
    finally:
        if trace_started:
            jax_module.profiler.stop_trace()

    return step_times_ms


def run_benchmark(config: BenchmarkConfig) -> Dict[str, Any]:
    """Run the JAX/Transformer Engine benchmark and return its report."""

    jax_module, jnp_module, te_module, layer_cls = _load_jax_components()
    device_info = _device_info(jax_module)
    run_step = _make_step(config, jax_module, jnp_module, te_module, layer_cls)
    step_times_ms = _run_iterations(config, jax_module, run_step)
    return build_report(config, step_times_ms, device_info)


def emit_report(report: Dict[str, Any], output_json: Optional[str] = None) -> None:
    """Print one JSON object to stdout and optionally write it to a file."""

    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output_json is not None:
        output_path = Path(output_json)
        if output_path.parent != Path(""):
            output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)


def main(argv: Optional[Sequence[str]] = None) -> int:
    config = parse_args(argv)
    report = run_benchmark(config)
    emit_report(report, config.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
