# JAX TransformerLayer Benchmark

This directory contains a small Transformer Engine JAX benchmark that measures a
single Flax `TransformerLayer` encoder forward pass and emits machine-readable
JSON.

Run from the repository root:

```bash
NVTE_FRAMEWORK=jax python3 benchmarks/jax/benchmark_transformer_layer.py \
  --warmup-iters 3 \
  --iters 10 \
  --output benchmark_output/jax_transformer_layer_smoke.json
```

The command writes the JSON report to the requested file and mirrors the same
JSON object to stdout. Use `--output -` to write only to stdout. The benchmark
checks `NVTE_FRAMEWORK=jax` before importing JAX or Transformer Engine JAX APIs.

Useful smoke command:

```bash
NVTE_FRAMEWORK=jax python3 benchmarks/jax/benchmark_transformer_layer.py \
  --warmup-iters 1 \
  --iters 2 \
  --output /tmp/te-jax-transformer-layer-smoke.json
```

For a short JAX profiler trace that starts after compilation and warmup:

```bash
NVTE_FRAMEWORK=jax python3 benchmarks/jax/benchmark_transformer_layer.py \
  --warmup-iters 3 \
  --iters 2 \
  --profile-dir /tmp/te-jax-transformer-layer-profile \
  --output /tmp/te-jax-transformer-layer-profile.json
```
