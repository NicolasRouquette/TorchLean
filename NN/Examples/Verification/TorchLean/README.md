# Native TorchLean Verification

These workflows lower a TorchLean model to the verifier IR and run Lean-side bound propagation:

```text
TorchLean model
  -> parameter payload
  -> verifier IR graph
  -> IBP/CROWN-style bound propagation
  -> checked margin, bound, or diagnostic report
```

Reusable workflow code belongs under `NN/Verification/TorchLean`. Reusable CROWN/LiRPA data
structures, transfer rules, and proof files belong under `NN/MLTheory/CROWN`.

Run the maintained entry points through the unified verifier:

```bash
lake exe verify -- torchlean-ibp
lake exe verify -- torchlean-crown-ops
lake exe verify -- torchlean-transformer-ibp
lake exe verify -- torchlean-mlp-workflow --scalar float32
```

Implementation map:

- `torchlean-ibp`: `NN.Verification.TorchLean.IBPWorkflow`
- `torchlean-crown-ops`: `NN.Verification.TorchLean.CrownOpsWorkflow`
- `torchlean-transformer-ibp`: `NN.Verification.TorchLean.TransformerIBPWorkflow`
- `torchlean-mlp-workflow`: `NN.Verification.TorchLean.MlpTrainVerifyWorkflow`

The `Proved/` subtree contains theorem-backed lowering and evaluator fragments. Runtime reports and
checker results remain separate from those theorems.

The model training examples elsewhere in `NN/Examples/Models` cover ordinary eager, typed graph, and
CUDA training. The workflows here are verifier workflows: after training, the parameters must be
available as Lean tensors so the verifier can lower and check the graph. Keep generated runtime
logs, checkpoints, and exported artifacts out of this directory; put them under an ignored
`generated/`, `outputs/`, or `_external/` directory if needed.
