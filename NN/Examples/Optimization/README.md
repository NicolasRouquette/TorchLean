# Optimization Examples

This folder contains example theorems for optimizer behavior. These files are different from a
training run: they do not tune a model or report a loss curve. They show how an optimizer update can
be named as a Lean object and how the public theorem API exposes the facts a later proof would
want to consume.

`MuonCertificates.lean` focuses on Muon-style updates. Muon combines a momentum buffer with an
orthogonalized update direction. In TorchLean, the orthogonalizer is not treated as a mysterious
black box. The state records which orthogonalizer is being used, and the theorem states what
that orthogonalizer certifies about the direction.

The examples cover three useful cases:

| Case | What the theorem exposes |
| --- | --- |
| QR checked backend | Positive QR pivots give an exact column-Gram certificate for the update direction. |
| Newton-Schulz residual backend | A residual check gives an approximate column-Gram certificate with an explicit tolerance. |
| Newton-Schulz fixed-point backend | Under the fixed-point hypotheses, the approximate iteration can be consumed as an exact certified step. |

The resulting theorems expose both pieces downstream code needs:

- a direction certificate, such as `HasExactColumnGram direction` or `HasApproxColumnGram eps direction`;
- the parameter equation saying the new parameters are `params - lr • direction`.

Build the optimization examples with:

```bash
lake build NN.Examples.Optimization
```

For the surrounding theory, read `NN/MLTheory/Optimization/OptimizerLaws.lean` and
`NN/MLTheory/Optimization/Muon.lean`. Runtime users configure Muon through
`TorchLean.optim.muon.optimizer`, because the orthogonalizer backend is part of the update. Proof
examples and theorem statements use the canonical `Optim.Muon` namespace.
