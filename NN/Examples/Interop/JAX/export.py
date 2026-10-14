"""Capture example JAX computations for ``torchlean jaxpr <artifact.json> ...``.

Change the functions below to try another computation. Parameters are closed over;
the input is one tensor, and a function may return several tensor outputs.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()
    if args.width < 1 or args.batch < 1:
        parser.error("width and batch must be positive")

    adapter_path = Path(__file__).resolve().parents[3] / "Runtime/JAX/jaxpr.py"
    spec = importlib.util.spec_from_file_location("torchlean_jaxpr", adapter_path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    # Do not silently truncate a requested float64 example to JAX's default float32.
    jax.config.update("jax_enable_x64", True)
    args.directory.mkdir(parents=True, exist_ok=True)

    n, b = args.width, args.batch
    vector = jnp.arange(1, n + 1, dtype=jnp.float64)
    matrix = jnp.arange(b * n, dtype=jnp.float64).reshape(b, n) / 8
    weights = jnp.arange(n * n, dtype=jnp.float64).reshape(n, n) / 16
    bias = jnp.arange(n, dtype=jnp.float64) / 4
    batched = jnp.arange(b * n * n, dtype=jnp.float64).reshape(b, n, n) / 32

    examples = {
        "sum_squares": (lambda x: jnp.sum(x * x), vector),
        "permuted_input": (lambda x: jnp.sum(x * x), vector[::-1]),
        "mlp": (lambda x: jax.nn.relu(x @ weights + bias) @ weights, matrix),
        "axis_mean": (lambda x: jnp.mean(x, axis=0), matrix),
        "transpose": (lambda x: x.T, matrix),
        "reshape": (lambda x: x.reshape(-1), matrix),
        "broadcast": (lambda x: jnp.broadcast_to(x[:, None], (n, b)), vector),
        "concatenate": (lambda x: jnp.concatenate((x, x), axis=-1), matrix),
        "batched_contraction": (lambda x: jnp.einsum("bij,bkj->bik", x, batched), batched),
        "multiple_outputs": (lambda x: (jnp.sin(x), jnp.cos(x), x * x), vector),
        "polynomial": (lambda x: x * x * x + 2, vector),
        "gradient": (jax.grad(lambda x: jnp.sum(x * x)), vector),
        "float32": (lambda x: jnp.sum(x * x), vector.astype(jnp.float32)),
        "mixed_precision": (
            lambda x: (x.astype(jnp.float32) * x.astype(jnp.float32)).astype(jnp.float16),
            vector.astype(jnp.float64)),
        "float16_sum": (lambda x: jnp.sum(x), vector.astype(jnp.float16)),
        "bfloat16_sum": (lambda x: jnp.sum(x), vector.astype(jnp.bfloat16)),
    }
    for name, (function, example) in examples.items():
        try:
            artifact = adapter.export(function, example)
        except adapter.Unsupported as error:
            print(f"{name}: unsupported: {error}")
            continue
        outputs = jax.tree.leaves(function(example))
        artifact["example_input"] = example.tolist()
        artifact["example_outputs"] = [value.tolist() for value in outputs]
        artifact["example_input_bits"] = adapter.bits(example)
        artifact["example_outputs_bits"] = [adapter.bits(value) for value in outputs]
        path = args.directory / f"{name}.json"
        path.write_text(json.dumps(artifact, allow_nan=False) + "\n")
        print(f"{name}: {len(artifact['nodes'])} IR nodes -> {path}")

if __name__ == "__main__":
    main()
