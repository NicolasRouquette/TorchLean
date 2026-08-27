#!/usr/bin/env python3
"""Export a tiny transformer-encoder-style interval certificate."""
from typing import Any

from common import (
    add_down,
    add_up,
    affine_interval,
    centered_box,
    layernorm_range_interval,
    matmul_interval,
    relu_interval,
    softmax_range_interval,
    write_json,
)

# Tiny transformer-encoder-like graph IBP, mirroring
# `NN.Verification.LiRPA.TransformerEncoder`.

nModel = 4
scoresDim = 5
nHidden = 6

def seed_params():
    """Return deterministic weights for the transformer-like fixture graph."""
    score_weight = [[float(1 + (i + 2*j)) for j in range(nModel)] for i in range(scoresDim)]
    score_bias = [0.1 * float(i) for i in range(scoresDim)]
    value_weight = [[float(2 + (i + j)) for j in range(scoresDim)] for i in range(nModel)]
    feed_forward_hidden_weight = [
        [float(1 + ((i + j) % 3)) for j in range(nModel)] for i in range(nHidden)
    ]
    feed_forward_hidden_bias = [0.05 * float(i) for i in range(nHidden)]
    feed_forward_output_weight = [
        [float(2 + ((i + j) % 4)) for j in range(nHidden)] for i in range(nModel)
    ]
    feed_forward_output_bias = [0.02 * float(i) for i in range(nModel)]
    return (
        score_weight,
        score_bias,
        value_weight,
        feed_forward_hidden_weight,
        feed_forward_hidden_bias,
        feed_forward_output_weight,
        feed_forward_output_bias,
    )


def seed_input_box(eps: float = 0.5):
    """Return the input interval box centered at `[1, 2, 3, 4]`."""
    input_center = [float(i + 1) for i in range(nModel)]
    return centered_box(input_center, eps)


def ibp_add(lo1: list[float], hi1: list[float], lo2: list[float], hi2: list[float]):
    """Add interval vectors with the outward rounding used by Lean's graph rule."""
    return [add_down(a, c) for a, c in zip(lo1, lo2)], [
        add_up(b, d) for b, d in zip(hi1, hi2)
    ]


def run_ibp() -> dict[str, Any]:
    """Compute the transformer-like certificate payload consumed by Lean."""
    (
        score_weight,
        score_bias,
        value_weight,
        feed_forward_hidden_weight,
        feed_forward_hidden_bias,
        feed_forward_output_weight,
        feed_forward_output_bias,
    ) = seed_params()
    x_lo, x_hi = seed_input_box(0.5)
    affine_interval(score_weight, score_bias, x_lo, x_hi)
    prob_lo, prob_hi = softmax_range_interval(scoresDim)
    attention_lo, attention_hi = matmul_interval(value_weight, prob_lo, prob_hi)
    attention_residual_lo, attention_residual_hi = ibp_add(x_lo, x_hi, attention_lo, attention_hi)
    normalized_attention_lo, normalized_attention_hi = layernorm_range_interval(nModel)
    hidden_lo, hidden_hi = affine_interval(
        feed_forward_hidden_weight,
        feed_forward_hidden_bias,
        normalized_attention_lo,
        normalized_attention_hi,
    )
    hidden_lo, hidden_hi = relu_interval(hidden_lo, hidden_hi)
    feed_forward_lo, feed_forward_hi = affine_interval(
        feed_forward_output_weight,
        feed_forward_output_bias,
        hidden_lo,
        hidden_hi,
    )
    feed_forward_residual_lo, feed_forward_residual_hi = ibp_add(
        normalized_attention_lo,
        normalized_attention_hi,
        feed_forward_lo,
        feed_forward_hi,
    )
    output_lo, output_hi = layernorm_range_interval(nModel)

    return {
        "graph": "transformer_encoder_workflow_v1",
        "input_box": {"id": 0, "dim": nModel, "lo": x_lo, "hi": x_hi},
        "result": {"node_id": 10, "dim": nModel, "lo": output_lo, "hi": output_hi}
    }


def main():
    """Write the transformer-like certificate to the bundled examples directory."""
    cert = run_ibp()
    out_path = "NN/Examples/Verification/LiRPA/transformer_encoder_cert.json"
    out = write_json(out_path, cert)
    print(f"Wrote certificate to {out}")

if __name__ == "__main__":
    main()
