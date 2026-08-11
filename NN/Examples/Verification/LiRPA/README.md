# LiRPA Verification Artifacts

These files are small offline artifacts for TorchLean's LiRPA/IBP certificate checkers. The
producer side writes finite JSON objects: graph metadata, node ids, input boxes, intermediate
bounds, and the final property being checked. Lean then parses the artifact and recomputes or
checks the represented bound condition.

The bundled artifacts cover several graph shapes:

- `mlp_cert.json` from `scripts/verification/lirpa/export_mlp_cert.py`
- `cnn_cert.json` from `scripts/verification/lirpa/export_cnn_cert.py`
- `attention_softmax_cert.json` from `scripts/verification/lirpa/export_attention_cert.py`
- `gru_gate_cert.json` from `scripts/verification/lirpa/export_gru_cert.py`
- `transformer_encoder_cert.json` from `scripts/verification/lirpa/export_crown_cert.py`

Run the curated group with:

```bash
python3 scripts/verification/regenerate_assets.py --group lirpa --run
lake exe verify -- all
```

Or run a single checker through the unified verifier:

```bash
lake exe verify -- lirpa-mlp
lake exe verify -- lirpa-cnn
lake exe verify -- lirpa-attention
lake exe verify -- lirpa-gru
lake exe verify -- lirpa-encoder
```

The producer and checker have separate roles. Python scripts may use ordinary numerical
code to construct the example artifact. TorchLean's Lean code checks the artifact it receives. If a
larger auto-LiRPA experiment is used as the producer, its raw logs, checkpoints, and large JSON
outputs should live in `_external/`, `/tmp`, or another documented generated-data directory. This
folder keeps only compact fixtures that are useful for review and CI.
