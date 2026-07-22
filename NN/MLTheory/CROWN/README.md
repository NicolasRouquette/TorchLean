# CROWN / LiRPA in TorchLean

This folder contains TorchLean's CROWN/LiRPA-style bound propagation code, certificate data
structures, and proof files. It is the mathematical engine behind several verification workflows:
TorchLean-native IBP/CROWN examples, external certificate checks, VNN-COMP-style exported suites,
and Lyapunov/controller experiments.

## Main Files

1. `Core.lean`: interval boxes (`Box`) and the basic affine form container (`AffineVec`).
2. `Models/Mlp.lean`: vector-in/vector-out CROWN development for small MLP-style networks. It uses
   the canonical executable ReLU relaxations from `Runtime/Ops.lean` together with its IBP bounds
   and model-specific affine composition.
3. `Graph.lean` and `Graph/`: graph-based LiRPA over TorchLean's op-tagged IR graphs. The graph
   engine stores per-node interval boxes, affine forms, parameter stores, and transfer state.
4. `Operators.lean` and `Operators/`: transfer rules for ReLU-family activations, arithmetic,
   convolution, pooling, batch normalization, reductions, slicing, and trigonometric operations.
5. `Cert/`: alpha-CROWN and alpha-beta-CROWN certificate data structures.
6. `Proofs/`: theorem-backed pieces of the CROWN development, including graph-IBP theorems,
   graph-certificate soundness, alpha/beta ReLU scalar soundness, and transfer-rule soundness
   interfaces. `Proofs/AlphaReLULowerBound.lean` is the shared scalar lower-bound lemma used by
   both alpha-CROWN and alpha/beta-CROWN proofs.
7. `Runtime/Ops.lean`: the canonical executable ReLU relaxation definitions used by both the graph
   engine and MLP development, kept separate from heavier proof imports.

## How To Read The Layering

CROWN-style code has three distinct jobs:

- represent bounds and affine relaxations;
- compute or check transfer rules over supported operators;
- prove that accepted bounds imply a semantic property of the graph or model.

Not every executable bound pass has the same theorem coverage. The proof files name the fragments
that currently have Lean support, while executable workflows and JSON checkers can still be useful
as diagnostics or artifact checks. When writing a claim, cite the strongest available support:
runtime report, checked certificate, transfer-rule assumption, or theorem.

## Claim Shapes

The same graph can support several levels of claim, and the wording should identify which one is
being used.

| Claim | Evidence to cite |
| --- | --- |
| A bound pass ran on a graph | the runtime command, graph id/output id, input box, and printed bound result |
| A JSON artifact was accepted | the checker module, schema name, artifact path, and recomputed predicate |
| A graph certificate is sound | the theorem in `Proofs/`, the graph semantics, and the hypotheses discharged by the checker |
| An external verifier found the leaf | the external producer/provenance plus the Lean-checked leaf artifact |
| A finite-precision bound is being used | the `FP32`, `IEEE32Exec`, or runtime bridge assumptions named by the caller |

For example, an alpha-beta-CROWN leaf artifact represents one exported terminal leaf: boxes, lower
bounds, thresholds, labels, and the witness comparison represented by the schema. A full producer
claim additionally needs provenance for the external branch-and-bound run that generated the leaf.

## Graph Engine And Proof Surface

The graph engine works over `NN.IR.Graph` node ids and payload stores. A typical workflow creates or
imports a graph, attaches an input box, computes per-node IBP boxes, and then propagates affine
forms for a selected output or objective. Operator files provide the transfer rules; proof files say
which rules have soundness statements or which assumptions remain.

Use this split when adding operators:

1. Add the executable interval/affine transfer rule.
2. State the shape and payload assumptions it needs.
3. Add or extend the proof layer soundness theorem when the operator supports formal
   graph-certificate claims.
4. Add a small verifier example or fixture if the rule is exposed through `lake exe verify`.

That keeps runtime diagnostics, accepted certificates, and theorem-backed graph claims connected
while preserving the distinction between execution evidence, checker acceptance, and theorem-backed
graph claims.

## Subfolders

- `Graph/`: graph engine, backward propagation, and graph-level theorem statements.
- `Operators/`: op-specific IBP and affine transfer rules.
- `Propagation/`: specialized propagation routines such as backward or sign-split passes. The
  canonical `IBP.matPos`/`IBP.matNeg` weight decomposition lives in `Core.lean` and is shared by
  interval and graph-CROWN linear rules.
- `Cert/`: alpha/alpha-beta certificate structures.
- `Lyapunov/`: controller and Lyapunov-oriented CROWN workflows, including the oracle boundary.
  The two Lean-executed pipelines share compiled gradient search and loss-box verification through
  `Lyapunov/TwoStage/CompiledLossAnalysis.lean` (`projectedGradientStep` and `checkLossBox`).
- `Proofs/`: soundness theorems and proof layer overviews.
- `Extras/`: optional helpers and proof toolboxes.
- `Tactics/`: small tactic support for CROWN oracle-style workflows.

## Optional Modules

- `Extras/IntervalLemmas.lean`: interval-arithmetic lemmas over `ℝ`.
- `Extras/AlphaConfig.lean`: data structures for alpha-optimized relaxations.
- `Extras/FloatIntegration.lean`: experiments integrating bound propagation with explicit rounding.
- `Extras/FP32.lean` and `Extras/BoundOpsIEEE32Exec.lean`: finite-precision specializations and
  executable IEEE32 connections.
