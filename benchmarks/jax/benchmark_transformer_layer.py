# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX TransformerLayer benchmark with machine-readable JSON output."""

import argparse
import ctypes
import ctypes.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, List, Optional

import flax
import jax
import jax.numpy as jnp

import transformer_engine.jax as te
from transformer_engine.jax import flax as te_flax


BENCHMARK_NAME = "transformer_layer_fwd_bwd"
SCHEMA_VERSION = "te_jax_benchmark/v1"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a JAX Transformer Engine TransformerLayer forward+backward pass."
    )
    parser.add_argument("--batch-size", type=_positive_int, default=2)
    parser.add_argument("--sequence-length", type=_positive_int, default=128)
    parser.add_argument("--hidden-size", type=_positive_int, default=512)
    parser.add_argument("--mlp-hidden-size", type=_positive_int, default=2048)
    parser.add_argument("--num-attention-heads", type=_positive_int, default=8)
    parser.add_argument("--warmup-iters", type=_positive_int, default=5)
    parser.add_argument("--timing-iters", type=_positive_int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the JSON payload.")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow running on a CPU backend. By default the benchmark requires a JAX GPU backend.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Call cudaProfilerStart after warmup and cudaProfilerStop before output handling.",
    )
    return parser.parse_args()


def _load_cuda_runtime() -> Optional[ctypes.CDLL]:
    candidates = []
    found_library = ctypes.util.find_library("cudart")
    if found_library:
        candidates.append(found_library)
    candidates.extend(
        [
            "libcudart.so",
            "libcudart.so.12",
            "libcudart.so.11.0",
        ]
    )

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    return None


class _CudaProfiler:
    """CUDA profiler API wrapper used to bound Nsight collection to timed iterations."""

    def __init__(self, enabled: bool):
        self._enabled = enabled
        self._runtime = None

    def __enter__(self):
        if not self._enabled:
            return self
        self._runtime = _load_cuda_runtime()
        if self._runtime is None:
            raise RuntimeError("Unable to load CUDA runtime for --profile.")
        self._check(self._runtime.cudaProfilerStart(), "cudaProfilerStart")
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self._enabled and self._runtime is not None:
            self._check(self._runtime.cudaProfilerStop(), "cudaProfilerStop")
        return False

    @staticmethod
    def _check(status: int, name: str) -> None:
        if status != 0:
            raise RuntimeError(f"{name} failed with CUDA status {status}.")


def _require_supported_backend(allow_cpu: bool) -> None:
    backend = jax.default_backend()
    if not allow_cpu and backend not in ("cuda", "gpu", "rocm"):
        raise RuntimeError(
            f"JAX default backend is {backend!r}; rerun in a GPU JAX environment or pass "
            "--allow-cpu."
        )


def _block_until_ready(value: Any) -> None:
    jax.tree_util.tree_map(lambda item: item.block_until_ready(), value)


def _device_metadata() -> List[Dict[str, Any]]:
    devices = []
    for index, device in enumerate(jax.devices()):
        devices.append(
            {
                "index": index,
                "id": getattr(device, "id", None),
                "platform": getattr(device, "platform", None),
                "device_kind": getattr(device, "device_kind", None),
                "description": str(device),
            }
        )
    return devices


def _jaxlib_version() -> Optional[str]:
    try:
        import jaxlib  # pylint: disable=import-outside-toplevel

        return jaxlib.__version__
    except ImportError:
        return None


def _create_inputs(args: argparse.Namespace) -> jax.Array:
    data_shape = (args.batch_size, args.sequence_length, args.hidden_size)
    data_key = jax.random.PRNGKey(args.seed)
    return jax.random.normal(data_key, data_shape, dtype=jnp.bfloat16)


def _create_layer(args: argparse.Namespace) -> te_flax.TransformerLayer:
    return te_flax.TransformerLayer(
        hidden_size=args.hidden_size,
        mlp_hidden_size=args.mlp_hidden_size,
        num_attention_heads=args.num_attention_heads,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        intermediate_dropout=0.0,
        self_attn_mask_type="causal",
        enable_relative_embedding=False,
        dtype=jnp.bfloat16,
    )


def _build_step(args: argparse.Namespace):
    inputs = _create_inputs(args)
    layer = _create_layer(args)
    params_key, dropout_key = jax.random.split(jax.random.PRNGKey(args.seed + 1))
    variables = layer.init(
        {"params": params_key, "dropout": dropout_key}, inputs, deterministic=True
    )
    others, params = flax.core.pop(variables, "params")

    def loss_fn(current_params, current_inputs):
        current_variables = {"params": current_params, **others}
        outputs = layer.apply(current_variables, current_inputs, deterministic=True)
        return jnp.mean(outputs.astype(jnp.float32))

    return jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1))), params, inputs


def _statistics(samples_ms: List[float]) -> Dict[str, Any]:
    return {
        "latency_ms_mean": statistics.mean(samples_ms),
        "latency_ms_median": statistics.median(samples_ms),
        "latency_ms_min": min(samples_ms),
        "latency_ms_max": max(samples_ms),
        "latency_ms_stddev": statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0,
    }


def _run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    _require_supported_backend(args.allow_cpu)
    if args.hidden_size % args.num_attention_heads != 0:
        raise ValueError(
            "--hidden-size must be divisible by --num-attention-heads "
            f"({args.hidden_size} % {args.num_attention_heads} != 0)."
        )

    with te.autocast(enabled=False, mesh_resource=te.MeshResource()):
        step, params, inputs = _build_step(args)

        for _ in range(args.warmup_iters):
            _block_until_ready(step(params, inputs))

        samples_ms = []
        with _CudaProfiler(args.profile):
            for _ in range(args.timing_iters):
                start = time.perf_counter()
                _block_until_ready(step(params, inputs))
                samples_ms.append((time.perf_counter() - start) * 1000.0)

    metrics = _statistics(samples_ms)
    finite_latencies = all(math.isfinite(sample) and sample > 0 for sample in samples_ms)
    finite_metrics = all(math.isfinite(value) and value >= 0 for value in metrics.values())
    success = finite_latencies and finite_metrics and len(samples_ms) == args.timing_iters
    config = {
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "hidden_size": args.hidden_size,
        "mlp_hidden_size": args.mlp_hidden_size,
        "num_attention_heads": args.num_attention_heads,
        "warmup_iters": args.warmup_iters,
        "timing_iters": args.timing_iters,
        "dtype": "bf16",
        "layer_type": "encoder",
        "self_attn_mask_type": "causal",
        "compile_excluded": True,
        "profile": args.profile,
    }
    case_id = (
        f"b{args.batch_size}_s{args.sequence_length}_h{args.hidden_size}_"
        f"mlp{args.mlp_hidden_size}_heads{args.num_attention_heads}_bf16"
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "case_id": case_id,
        "framework": "jax",
        "precision": "bf16",
        "config": config,
        "environment": {
            "jax_version": jax.__version__,
            "jaxlib_version": _jaxlib_version(),
            "default_backend": jax.default_backend(),
            "device_count": jax.device_count(),
            "local_device_count": jax.local_device_count(),
            "devices": _device_metadata(),
        },
        "metrics": {
            **metrics,
            "samples_ms": samples_ms,
            "sample_count": len(samples_ms),
            "compile_excluded": True,
        },
        "measurements": [
            {
                "case_id": case_id,
                "metric": "latency_ms_mean",
                "value": metrics["latency_ms_mean"],
                "unit": "ms",
                "iteration": 0,
                "higher_is_better": False,
            },
            {
                "case_id": case_id,
                "metric": "latency_ms_median",
                "value": metrics["latency_ms_median"],
                "unit": "ms",
                "iteration": 0,
                "higher_is_better": False,
            },
        ],
        "success": success,
    }


def _emit_payload(payload: Dict[str, Any], output: Optional[Path], pretty: bool) -> None:
    indent = 2 if pretty else None
    text = json.dumps(payload, indent=indent, sort_keys=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


def main() -> None:
    args = _parse_args()
    payload = _run_benchmark(args)
    if not payload["success"]:
        raise RuntimeError("Benchmark completed but produced invalid timing metrics.")
    _emit_payload(payload, args.output, args.pretty)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"benchmark_transformer_layer.py: error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
