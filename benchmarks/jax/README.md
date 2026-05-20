# JAX TransformerLayer Benchmark

This benchmark runs a BF16 `transformer_engine.jax.flax.TransformerLayer` forward and backward pass with synthetic inputs. It prints one machine-readable JSON object to stdout and, when `--output-json` is provided, writes the same JSON object to that file.

Run from the repository root in a JAX-enabled Transformer Engine environment:

```bash
NVTE_FRAMEWORK=jax python3 benchmarks/jax/benchmark_transformer_layer.py \
  --batch-size 1 \
  --sequence-length 128 \
  --hidden-size 512 \
  --mlp-hidden-size 2048 \
  --num-attention-heads 8 \
  --warmup-iters 3 \
  --iters 10 \
  --output-json /tmp/te_jax_transformer_layer.json
```

The JSON report uses `schema_version` `te_jax_transformer_layer_benchmark/v1`, records the observed `NVTE_FRAMEWORK` value, and includes aggregate timing metrics under `metrics`.

To collect a JAX profiler trace after warmup, add `--profile-dir`:

```bash
NVTE_FRAMEWORK=jax python3 benchmarks/jax/benchmark_transformer_layer.py \
  --batch-size 1 \
  --sequence-length 128 \
  --hidden-size 512 \
  --mlp-hidden-size 2048 \
  --num-attention-heads 8 \
  --warmup-iters 3 \
  --iters 1 \
  --profile-dir /tmp/te_jax_transformer_layer_profile
```
