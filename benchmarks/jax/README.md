# JAX Transformer Engine Benchmarks

This directory contains JAX-only Transformer Engine benchmarks. The commands below keep
`NVTE_FRAMEWORK=jax` explicit and use JAX, Flax, and `transformer_engine.jax` APIs.

## TransformerLayer Forward and Backward

Run the default small BF16 encoder-layer benchmark and write the same JSON payload that is
printed to stdout:

```bash
NVTE_FRAMEWORK=jax python3 benchmarks/jax/benchmark_transformer_layer.py \
  --batch-size 2 \
  --sequence-length 128 \
  --hidden-size 512 \
  --mlp-hidden-size 2048 \
  --num-attention-heads 8 \
  --warmup-iters 5 \
  --timing-iters 10 \
  --output /tmp/te-jax-transformer-layer-benchmark.json
```

For a faster smoke run on a GPU JAX environment:

```bash
NVTE_FRAMEWORK=jax python3 benchmarks/jax/benchmark_transformer_layer.py \
  --batch-size 1 \
  --sequence-length 32 \
  --hidden-size 128 \
  --mlp-hidden-size 512 \
  --num-attention-heads 4 \
  --warmup-iters 1 \
  --timing-iters 1
```

The benchmark emits `te_jax_benchmark/v1` JSON with configuration, JAX backend/device metadata,
per-iteration `metrics.samples_ms`, and summary latency metrics. The primary metric is
`metrics.latency_ms_mean` in milliseconds per compiled forward+backward iteration. Lower is better.
Warmup iterations run before timing, so JIT compilation is excluded from measured samples.

Use `--profile` when running under an Nsight tool configured for CUDA profiler API capture. The
script starts profiler collection only after warmup iterations and stops it before JSON output.
