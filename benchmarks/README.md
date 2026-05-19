# Transformer Engine Benchmarks

## Acceptance Smoke Benchmark

Run this command from the repository root to produce a machine-readable JSON report:

```bash
python3 benchmarks/smoke_benchmark.py --output /tmp/te-smoke-benchmark.json --warmup-iterations 3 --iterations 10 --batch-size 32 --in-features 128 --out-features 128
```

The benchmark runs a small PyTorch `transformer_engine.pytorch.Linear` forward pass on CUDA
with bfloat16 inputs and writes a report with schema version `te_benchmark_smoke/v1`.
Benchmark acceptance requires `status` to be `passed` and the latency and throughput metrics
to be finite positive numbers.

For Nsight capture, use the same benchmark with profiler collection enabled after warmup:

```bash
python3 benchmarks/smoke_benchmark.py --profile --output /tmp/te-smoke-benchmark.json --warmup-iterations 3 --iterations 5 --batch-size 32 --in-features 128 --out-features 128
```

The profiler range starts after warmup iterations and stops before JSON output handling.
