/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved.Correctness.Eval.Main
public import NN.Verification.TorchLean.Proved.Correctness.Eval.Coverage
public import NN.Verification.TorchLean.Proved.Correctness.Eval.CompilePayload
public import NN.Verification.TorchLean.Proved.Correctness.Eval.PayloadBridge

/-!
Evaluation lemmas for proved TorchLean correctness.

This import point collects denotation-side facts used when moving from compiled/evaluated
TorchLean graphs back to their specification semantics.

Current bridge coverage includes:
- common elementwise arithmetic and activations emitted by PyTorch/ONNX import paths;
- shape-changing plumbing such as reshape, flatten, broadcast, scalar sum, leading-axis concat,
  axis permutation, supported transpose forms, and axis reductions;
- rank-2 and rank-3 `matmul`;
- last-axis softmax and the evaluator's permutation path for non-last-axis softmax;
- payload-backed `linear` and no-dilation `conv2d`;
- payload-backed constants;
- CHW max/average pooling, including padded variants;
- `layernorm axis` through the reshape-to-2D spec LayerNorm path;
- graph plumbing nodes such as `input` and `detach`, plus scalar MSE loss;
- eval-mode NCHW BatchNorm with payload-backed running statistics.
- exact `ParamStore` to IR `Payload` forwarding facts for every payload-backed op.
- compiler insertion facts for the payload-backed nodes in the proved forward fragment.
-/
