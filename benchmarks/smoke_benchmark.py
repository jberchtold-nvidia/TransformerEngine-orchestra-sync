# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Minimal Transformer Engine smoke benchmark with JSON output."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
from importlib import metadata
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "te_benchmark_smoke/v1"
BENCHMARK_NAME = "pytorch_linear_forward_smoke"


class BenchmarkSkipped(RuntimeError):
    """Raised when benchmark prerequisites are unavailable."""


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a small Transformer Engine PyTorch Linear forward benchmark."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the machine-readable JSON report.",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=_non_negative_int,
        default=3,
        help="Warmup iterations to run before measurement and profiler collection.",
    )
    parser.add_argument(
        "--iterations",
        type=_positive_int,
        default=10,
        help="Measured forward iterations.",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=32,
        help="Input batch size.",
    )
    parser.add_argument(
        "--in-features",
        type=_positive_int,
        default=128,
        help="Input feature dimension for the Linear module.",
    )
    parser.add_argument(
        "--out-features",
        type=_positive_int,
        default=128,
        help="Output feature dimension for the Linear module.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
        help="Input and parameter dtype.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="CUDA device specifier, for example 'cuda' or 'cuda:0'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Torch RNG seed used to create benchmark inputs.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Start CUDA profiler collection after warmup and stop before report writing.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Also print the JSON report to stdout.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "warmup_iterations": args.warmup_iterations,
        "iterations": args.iterations,
        "batch_size": args.batch_size,
        "in_features": args.in_features,
        "out_features": args.out_features,
        "dtype": args.dtype,
        "device": args.device,
        "seed": args.seed,
        "profile": args.profile,
    }


def collect_environment(torch_module: Any | None = None) -> dict[str, str]:
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if torch_module is not None:
        env["torch"] = str(torch_module.__version__)
    else:
        try:
            env["torch"] = metadata.version("torch")
        except metadata.PackageNotFoundError:
            env["torch"] = "unavailable"
    try:
        env["transformer_engine"] = metadata.version("transformer_engine")
    except metadata.PackageNotFoundError:
        env["transformer_engine"] = "unavailable"
    return env


def make_report(
    *,
    status: str,
    config: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    device: dict[str, Any] | None = None,
    environment: dict[str, str] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    report = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "status": status,
        "device": device or {},
        "config": config,
        "metrics": metrics or {},
        "environment": environment or collect_environment(),
    }
    if reason is not None:
        report["reason"] = reason
    return report


def requested_device_info(device: str) -> dict[str, str]:
    return {
        "type": device.split(":", maxsplit=1)[0],
        "requested": device,
    }


def write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, sort_keys=True)
        report_file.write("\n")
    tmp_path.replace(output_path)


def _runtime_modules() -> tuple[Any, Any]:
    try:
        import torch
        import transformer_engine.pytorch as te
    except (AssertionError, FileNotFoundError, ImportError, OSError, RuntimeError) as exc:
        raise BenchmarkSkipped(
            f"PyTorch Transformer Engine runtime is unavailable: {type(exc).__name__}: {exc}"
        ) from exc
    return torch, te


def _torch_dtype(torch_module: Any, dtype_name: str) -> Any:
    if dtype_name == "bfloat16":
        return torch_module.bfloat16
    if dtype_name == "float16":
        return torch_module.float16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _check_cuda_prerequisites(torch_module: Any, args: argparse.Namespace) -> Any:
    device = torch_module.device(args.device)
    if device.type != "cuda":
        raise BenchmarkSkipped("This smoke benchmark requires a CUDA device.")
    if not torch_module.cuda.is_available():
        raise BenchmarkSkipped("CUDA is not available to PyTorch.")
    if device.index is not None:
        torch_module.cuda.set_device(device)
    device = torch_module.device("cuda", torch_module.cuda.current_device())

    if args.dtype == "bfloat16" and not torch_module.cuda.is_bf16_supported():
        raise BenchmarkSkipped("CUDA device does not report bfloat16 support.")
    return device


def _device_info(torch_module: Any, device: Any) -> dict[str, Any]:
    props = torch_module.cuda.get_device_properties(device)
    return {
        "type": "cuda",
        "index": device.index,
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_bytes": props.total_memory,
    }


def _latency_metrics(samples_ms: list[float], batch_size: int) -> dict[str, Any]:
    if not samples_ms:
        raise RuntimeError("No latency samples were collected.")
    if any((not math.isfinite(sample) or sample <= 0.0) for sample in samples_ms):
        raise RuntimeError(f"Latency samples must be finite and positive: {samples_ms}")

    mean_ms = statistics.mean(samples_ms)
    return {
        "latency_ms_mean": mean_ms,
        "latency_ms_median": statistics.median(samples_ms),
        "latency_ms_min": min(samples_ms),
        "latency_ms_max": max(samples_ms),
        "samples_per_second": batch_size / (mean_ms / 1000.0),
        "latency_ms_samples": samples_ms,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    torch, te = _runtime_modules()
    device = _check_cuda_prerequisites(torch, args)
    dtype = _torch_dtype(torch, args.dtype)

    torch.manual_seed(args.seed)
    layer = te.Linear(
        args.in_features,
        args.out_features,
        bias=False,
        params_dtype=dtype,
    ).to(device)
    layer.eval()
    inputs = torch.randn((args.batch_size, args.in_features), dtype=dtype, device=device)

    with torch.no_grad():
        for _ in range(args.warmup_iterations):
            layer(inputs)
        torch.cuda.synchronize(device)

        if args.profile:
            torch.cuda.cudart().cudaProfilerStart()
        timings_ms = []
        try:
            for _ in range(args.iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                layer(inputs)
                end.record()
                torch.cuda.synchronize(device)
                timings_ms.append(start.elapsed_time(end))
        finally:
            if args.profile:
                torch.cuda.cudart().cudaProfilerStop()

    return make_report(
        status="passed",
        config=config_from_args(args),
        metrics=_latency_metrics(timings_ms, args.batch_size),
        device=_device_info(torch, device),
        environment=collect_environment(torch),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)

    exit_code = 0
    try:
        report = run_benchmark(args)
    except BenchmarkSkipped as exc:
        report = make_report(
            status="skipped",
            config=config,
            device=requested_device_info(args.device),
            reason=str(exc),
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        report = make_report(
            status="failed",
            config=config,
            device=requested_device_info(args.device),
            reason=f"{type(exc).__name__}: {exc}",
        )
        exit_code = 1

    write_report(report, args.output)
    if args.print_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
