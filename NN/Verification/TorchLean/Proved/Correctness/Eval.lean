/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved.Correctness.Eval.Main
public import NN.Verification.TorchLean.Proved.Correctness.Eval.Concat
public import NN.Verification.TorchLean.Proved.Correctness.Eval.SourcesAndLosses
public import NN.Verification.TorchLean.Proved.Correctness.Eval.Reductions
public import NN.Verification.TorchLean.Proved.Correctness.Eval.Softmax
public import NN.Verification.TorchLean.Proved.Correctness.Eval.Transpose
public import NN.Verification.TorchLean.Proved.Correctness.Eval.LoweringPayload
public import NN.Verification.TorchLean.Proved.Correctness.Eval.PayloadBridge

/-!
Evaluation lemmas for proved TorchLean correctness.

This import point collects denotation-side facts used when moving from lowered and evaluated
TorchLean graphs back to their specification semantics.

Current bridge coverage includes:
- common elementwise arithmetic and activations emitted by PyTorch/ONNX import paths;
- shape-changing operations such as reshape, flatten, broadcast, scalar sum, leading-axis concat,
  axis permutation, supported transpose forms, and axis reductions;
- matrix and batched-matrix `matmul` through one typed operation;
- softmax along any valid tensor axis;
- payload-backed `linear` and arbitrary-rank no-dilation convolution;
- payload-backed constants;
- `layernorm axis` through a rank-independent matrix evaluation view;
- graph-structural nodes such as `input` and `detach`, plus scalar MSE loss;
- eval-mode BatchNorm over an arbitrary channel axis with payload-backed running statistics.
- exact `ParamStore` to IR `Payload` forwarding facts for every payload-backed op.
- lowering pass insertion facts for the payload-backed nodes in the proved forward fragment.
-/
