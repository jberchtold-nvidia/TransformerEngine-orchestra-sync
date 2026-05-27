# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Small JAX Transformer Engine TransformerLayer forward benchmark."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from importlib import metadata
from typing import Any, Mapping


BENCHMARK_NAME = "te_jax_transformer_layer_forward"
SCHEMA_VERSION = "te_jax_benchmark/v1"
REQUIRED_FRAMEWORK = "jax"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark a Transformer Engine JAX Flax TransformerLayer forward pass.",
        epilog=(
            "Example: NVTE_FRAMEWORK=jax python3 "
            "benchmarks/jax/benchmark_transformer_layer.py --warmup-iters 3 --iters 10 "
            "--output benchmark_output/jax_transformer_layer_smoke.json"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch-size", type=_positive_int, default=2)
    parser.add_argument("--seq-len", type=_positive_int, default=128)
    parser.add_argument("--hidden-size", type=_positive_int, default=512)
    parser.add_argument("--mlp-hidden-size", type=_positive_int, default=2048)
    parser.add_argument("--num-heads", type=_positive_int, default=8)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Input and parameter dtype.",
    )
    parser.add_argument("--warmup-iters", type=_non_negative_int, default=3)
    parser.add_argument("--iters", type=_positive_int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output",
        type=str,
        default="-",
        help="JSON output path. Use '-' to write only to stdout.",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default=None,
        help=(
            "Optional JAX trace directory. When set, JAX profiler tracing starts after "
            "warmup and stops before JSON output."
        ),
    )
    parser.add_argument(
        "--case-id",
        type=str,
        default="encoder_b2_s128_h512_heads8_bf16",
        help="Stable case identifier included in the JSON report.",
    )
    return parser.parse_args(argv)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value}")
    return parsed


def require_jax_framework(environ: Mapping[str, str] = os.environ) -> None:
    """Require an explicit JAX framework selection before importing TE JAX APIs."""
    framework = environ.get("NVTE_FRAMEWORK")
    if framework != REQUIRED_FRAMEWORK:
        raise SystemExit(
            "Set NVTE_FRAMEWORK=jax before running this benchmark. "
            f"Current value: {framework!r}."
        )


def validate_args(args: argparse.Namespace) -> None:
    """Validate cross-argument constraints."""
    if args.hidden_size % args.num_heads != 0:
        raise SystemExit(
            "--hidden-size must be divisible by --num-heads "
            f"(got hidden_size={args.hidden_size}, num_heads={args.num_heads})."
        )


def dtype_from_name(jnp: Any, dtype_name: str) -> Any:
    """Map CLI dtype names to JAX dtypes."""
    return {
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
        "float32": jnp.float32,
    }[dtype_name]


def timing_summary(iterations_ms: list[float]) -> dict[str, Any]:
    """Return aggregate timing statistics for a non-empty timing list."""
    if not iterations_ms:
        raise ValueError("timing summary requires at least one sample")
    if not all(math.isfinite(sample) and sample > 0.0 for sample in iterations_ms):
        raise ValueError(f"all timing samples must be positive finite values: {iterations_ms}")

    return {
        "iterations": iterations_ms,
        "mean": statistics.fmean(iterations_ms),
        "median": statistics.median(iterations_ms),
        "min": min(iterations_ms),
        "max": max(iterations_ms),
        "stdev": statistics.stdev(iterations_ms) if len(iterations_ms) > 1 else 0.0,
    }


def package_version(package_name: str) -> str:
    """Return an installed package version or 'unknown'."""
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "unknown"


def write_json_report(report: dict[str, Any], output: str) -> None:
    """Write a report to an output path and always mirror it to stdout."""
    payload = json.dumps(report, indent=2, sort_keys=True)
    if output != "-":
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    sys.stdout.write(payload)
    sys.stdout.write("\n")


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run the JAX benchmark and return a JSON-serializable report."""
    import flax
    import jax
    import jax.numpy as jnp
    from transformer_engine.jax import MeshResource, autocast
    from transformer_engine.jax.flax import TransformerLayer, TransformerLayerType

    if jax.default_backend() != "gpu":
        raise SystemExit(
            f"This benchmark requires a JAX GPU backend, got {jax.default_backend()!r}."
        )

    dtype = dtype_from_name(jnp, args.dtype)
    input_shape = (args.batch_size, args.seq_len, args.hidden_size)
    root_key = jax.random.PRNGKey(args.seed)
    params_key, input_key = jax.random.split(root_key)
    inputs = jax.random.normal(input_key, input_shape, dtype=dtype)

    layer = TransformerLayer(
        hidden_size=args.hidden_size,
        mlp_hidden_size=args.mlp_hidden_size,
        num_attention_heads=args.num_heads,
        layer_type=TransformerLayerType.ENCODER,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        intermediate_dropout=0.0,
        self_attn_mask_type="causal",
        enable_relative_embedding=False,
        dtype=dtype,
    )

    with autocast(enabled=False, mesh_resource=MeshResource()):
        variables = layer.init({"params": params_key}, inputs, deterministic=True)

        @jax.jit
        def forward(params, batch):
            return layer.apply(params, batch, deterministic=True)

        # First call compiles the function; it is intentionally outside timed samples.
        output = forward(variables, inputs)
        jax.block_until_ready(output)

        for _ in range(args.warmup_iters):
            output = forward(variables, inputs)
            jax.block_until_ready(output)

        profile_active = False
        if args.profile_dir is not None:
            profile_path = Path(args.profile_dir)
            profile_path.mkdir(parents=True, exist_ok=True)
            jax.profiler.start_trace(str(profile_path))
            profile_active = True

        iterations_ms: list[float] = []
        try:
            for _ in range(args.iters):
                start = time.perf_counter()
                output = forward(variables, inputs)
                jax.block_until_ready(output)
                iterations_ms.append((time.perf_counter() - start) * 1000.0)
        finally:
            if profile_active:
                jax.profiler.stop_trace()

    summary = timing_summary(iterations_ms)
    tokens_per_iteration = args.batch_size * args.seq_len
    tokens_per_second = tokens_per_iteration / (summary["mean"] / 1000.0)
    device = jax.devices()[0]
    measurements = [
        {
            "case_id": args.case_id,
            "metric": "latency_ms",
            "value": sample_ms,
            "unit": "ms",
            "iteration": index,
            "higher_is_better": False,
        }
        for index, sample_ms in enumerate(iterations_ms)
    ]
    measurements.extend(
        [
            {
                "case_id": args.case_id,
                "metric": "latency_ms_mean",
                "value": summary["mean"],
                "unit": "ms",
                "iteration": len(iterations_ms),
                "higher_is_better": False,
            },
            {
                "case_id": args.case_id,
                "metric": "tokens_per_second",
                "value": tokens_per_second,
                "unit": "tokens/s",
                "iteration": len(iterations_ms),
                "higher_is_better": True,
            },
        ]
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "case_id": args.case_id,
        "framework": "jax",
        "library": "transformer_engine.jax",
        "config": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "hidden_size": args.hidden_size,
            "mlp_hidden_size": args.mlp_hidden_size,
            "num_heads": args.num_heads,
            "dtype": args.dtype,
            "warmup_iters": args.warmup_iters,
            "iters": args.iters,
            "self_attn_mask_type": "causal",
            "dropout": 0.0,
        },
        "device": {
            "jax_backend": jax.default_backend(),
            "platform": getattr(device, "platform", "unknown"),
            "device_kind": getattr(device, "device_kind", str(device)),
            "local_device_count": jax.local_device_count(),
        },
        "versions": {
            "jax": getattr(jax, "__version__", "unknown"),
            "jaxlib": package_version("jaxlib"),
            "flax": getattr(flax, "__version__", "unknown"),
            "transformer_engine": package_version("transformer_engine"),
        },
        "timing_ms": summary,
        "throughput": {
            "tokens_per_iteration": tokens_per_iteration,
            "tokens_per_second": tokens_per_second,
        },
        "measurements": measurements,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    validate_args(args)
    require_jax_framework()
    report = run_benchmark(args)
    write_json_report(report, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
