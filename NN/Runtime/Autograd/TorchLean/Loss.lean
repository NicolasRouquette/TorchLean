/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.Functional

import Mathlib.Algebra.Order.Algebra

/-!
# Loss

TorchLean loss helpers in the style of `torch.nn.functional`.

These helpers keep training loops close to the familiar `torch.nn.functional` style:

```
loss = Loss.mse yhat y
loss = Loss.mse yhat y (reduction := .sum)
```

They are execution-mode generic: eager tape and typed SSA/DAG both work.

### PyTorch references

- `torch.nn.functional` (losses overview): https://pytorch.org/docs/stable/nn.functional.html
- `mse_loss`: https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html
- `cross_entropy`: https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
- `nll_loss`: https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html
- `binary_cross_entropy_with_logits`:
  https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html
-/

@[expose] public section


namespace TorchLean

open Spec
open Tensor
open _root_.Runtime.Autograd.Torch
open _root_.Runtime.Autograd.TorchLean

namespace Loss

/--
Reduction mode for losses that start as elementwise tensors.

PyTorch analogy: `reduction="mean"` or `reduction="sum"`.
-/
inductive Reduction where
  | mean
  | sum
  deriving Repr, DecidableEq

/--
  Reduce an elementwise loss tensor to a scalar according to `reduction`.

  This is the common final step for losses like MSE and cross-entropy.
  -/
  def reduce {α : Type} [Context α] [DecidableEq Shape]
      {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
      {s : Shape} (x : RefTy (m := m) (α := α) s) (reduction : Reduction) :
      m (RefTy (m := m) (α := α) Shape.scalar) := by
    cases reduction with
    | mean => exact F.mean (m := m) (α := α) (s := s) x
    | sum => exact sum (m := m) (α := α) (s := s) x

/--
Mean squared error (MSE) loss between predictions and targets.

This is backend-generic and supports both `mean` and `sum` reduction.
-/
def mse {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape}
    (yhat y : RefTy (m := m) (α := α) s)
    (reduction : Reduction := .mean) :
    m (RefTy (m := m) (α := α) Shape.scalar) := by
  cases reduction with
  | mean =>
      exact mseLoss (m := m) (α := α) (s := s) yhat y
  | sum =>
      exact (do
        let diff ← sub (m := m) (α := α) (s := s) yhat y
        let sq ← F.square (m := m) (α := α) (s := s) diff
        sum (m := m) (α := α) (s := s) sq
      )

/-- Negative log-likelihood (one-hot targets), assuming inputs are log-probabilities. -/
def nllOneHot {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (logProbs targetOneHot : RefTy (m := m) (α := α) s)
    (reduction : Reduction := .mean) :
    m (RefTy (m := m) (α := α) Shape.scalar) := do
  let prod ← mul (m := m) (α := α) (s := s) targetOneHot logProbs
  let negProd ← scale (m := m) (α := α) (s := s) prod (-1)
  match reduction with
  | .sum =>
      -- `sum` is already the correct reduction: `∑_{prefix,cls} -y * logp = ∑_{prefix} -logp_true`.
      reduce (m := m) (α := α) (s := s) negProd .sum
  | .mean =>
      -- Mean over samples, not classes: undo the class-dimension factor introduced by averaging
      -- every tensor entry.
      let avgAll ← reduce (m := m) (α := α) (s := s) negProd .mean
      scale (m := m) (α := α) (s := Shape.scalar) avgAll (Shape.axisSize s axis)

/--
Cross-entropy (one-hot targets), computed as `-sum(y * log(softmax(logits)))`.

Note: this is one-hot only. If your label is a single integer class index, use
`crossEntropyIndex`, whose `Fin n` target records the class bound in its type.
-/
def oneHotCrossEntropy {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (logits targetOneHot : RefTy (m := m) (α := α) s)
    (reduction : Reduction := .mean) :
    m (RefTy (m := m) (α := α) Shape.scalar) := do
  let logp ← F.logSoftmax (m := m) (α := α) (s := s) axis logits
  nllOneHot (m := m) (α := α) (s := s) axis logp targetOneHot (reduction := reduction)

/-- Negative log-likelihood for a single class index (vector logits/log-probs). -/
def nllIndex {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {n : Nat}
    (logProbs : RefTy (m := m) (α := α) (.dim n .scalar))
    (target : Fin n) :
    m (RefTy (m := m) (α := α) Shape.scalar) := do
  let picked ← gatherScalar (m := m) (α := α) (n := n) logProbs target
  scale (m := m) (α := α) (s := Shape.scalar) picked (-1)

/--
Cross-entropy for a single class index, computed as `-log(softmax(logits)[target])`.

This avoids one-hot targets, but the target index is a Lean `Fin n` (not a tensor value).
-/
def crossEntropyIndex {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {n : Nat}
    (logits : RefTy (m := m) (α := α) (.dim n .scalar))
    (target : Fin n) :
    m (RefTy (m := m) (α := α) Shape.scalar) := do
  let logp ← F.logSoftmax (m := m) (α := α) (s := .dim n .scalar) 0 logits
  nllIndex (m := m) (α := α) (n := n) logp target

/--
Convert per-row class labels into flat indices for a row-major `(rows × classes)` matrix.

An invalid class is mapped to the first index past the flattened matrix. The subsequent
`gatherVecNatOrZero` therefore returns zero for that row; it can never select a class from a
different row. Callers that attach the usual cross-entropy meaning must ensure every label is
strictly smaller than `classes`.
-/
def rowTargetFlatIndicesOrInvalid (rows classes : Nat) (target : Tensor Nat (.dim rows .scalar)) :
    Tensor Nat (.dim rows .scalar) :=
  match target with
  | Tensor.dim f =>
      Tensor.dim (fun r =>
        match f r with
        | Tensor.scalar cls =>
            Tensor.scalar <| if cls < classes then r.val * classes + cls else rows * classes)

/--
Unreduced negative log-likelihood for integer row labels.

`logProbs` has shape `(rows × classes)` and `target[r]` is the class id for row `r`. The result
contains one loss per row. Keeping this operation unreduced is useful for masked language modeling,
sample weighting, and any caller that needs to choose its own normalization.

The mathematical domain requires `target[r] < classes` for every row. Runtime objectives should
enforce that condition with `ObjectiveDef.validateNatInputs`; the low-level gather is totalized only
so malformed inputs cannot read a different row.
-/
def nllRowsNatUnreduced {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {rows classes : Nat}
    (logProbs : RefTy (m := m) (α := α) (.dim rows (.dim classes .scalar)))
    (target : _root_.Runtime.Autograd.Torch.NatTensorRef (m := m) (α := α)
      (.dim rows .scalar)) :
    m (RefTy (m := m) (α := α) (.dim rows .scalar)) := do
  let flat ← reshape (m := m) (α := α)
    (s₁ := .dim rows (.dim classes .scalar))
    (s₂ := .dim (rows * classes) .scalar)
    logProbs (by
      simp [Spec.Shape.size])
  let flatTarget := _root_.Runtime.Autograd.Torch.mapNatTensor (m := m) (α := α)
    (rowTargetFlatIndicesOrInvalid rows classes) target
  let picked ← gatherVecNatOrZero (m := m) (α := α)
    (n := rows * classes) (k := rows) flat flatTarget
  scale (m := m) (α := α) (s := .dim rows .scalar) picked (-1)

/--
Negative log-likelihood for a matrix of log-probabilities and integer row labels.

This is the integer-label counterpart of `nllOneHot`; it avoids materializing a one-hot target
matrix and then applies the requested mean or sum reduction over rows.
-/
def nllRowsNat {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {rows classes : Nat}
    (logProbs : RefTy (m := m) (α := α) (.dim rows (.dim classes .scalar)))
    (target : _root_.Runtime.Autograd.Torch.NatTensorRef (m := m) (α := α)
      (.dim rows .scalar))
    (reduction : Reduction := .mean) :
    m (RefTy (m := m) (α := α) Shape.scalar) := do
  let losses ← nllRowsNatUnreduced (m := m) (α := α)
    (rows := rows) (classes := classes) logProbs target
  reduce (m := m) (α := α) (s := .dim rows .scalar) losses reduction

/--
Weighted negative log-likelihood for integer row labels.

The result is

`∑ r, weights[r] * (-logProbs[r, target[r]])`.

No implicit normalization is performed. Use weights that sum to one for a weighted mean, zeros to
exclude rows, or arbitrary nonnegative weights for a weighted sum. Making the normalization
explicit avoids a hidden division-by-zero convention when every row is masked.
-/
def nllRowsNatWeighted {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {rows classes : Nat}
    (logProbs : RefTy (m := m) (α := α) (.dim rows (.dim classes .scalar)))
    (target : _root_.Runtime.Autograd.Torch.NatTensorRef (m := m) (α := α)
      (.dim rows .scalar))
    (weights : RefTy (m := m) (α := α) (.dim rows .scalar)) :
    m (RefTy (m := m) (α := α) Shape.scalar) := do
  let losses ← nllRowsNatUnreduced (m := m) (α := α)
    (rows := rows) (classes := classes) logProbs target
  let weighted ← mul (m := m) (α := α) (s := .dim rows .scalar) losses weights
  sum (m := m) (α := α) (s := .dim rows .scalar) weighted

/--
Cross-entropy for row-wise logits with integer labels.

This matches the common language-model/classification layout after flattening all prefix dimensions
into `rows`: logits are `(rows × classes)`, labels are a length-`rows` vector of class ids.
Every label must be strictly smaller than `classes`; model objectives validate this before execution.
-/
def crossEntropyRowsNat {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {rows classes : Nat}
    (logits : RefTy (m := m) (α := α) (.dim rows (.dim classes .scalar)))
    (target : _root_.Runtime.Autograd.Torch.NatTensorRef (m := m) (α := α)
      (.dim rows .scalar))
    (reduction : Reduction := .mean) :
    m (RefTy (m := m) (α := α) Shape.scalar) := do
  let logp ← F.logSoftmax (m := m) (α := α)
    (s := .dim rows (.dim classes .scalar)) 1 logits
  nllRowsNat (m := m) (α := α) (rows := rows) (classes := classes)
    logp target (reduction := reduction)

/--
Weighted row-wise cross entropy with integer labels.

This computes log-softmax, selects the target class in each row, multiplies by the caller-provided
row weights, and sums. It is the runtime primitive used by loss masks and packed variable-length
sequences; it is not specific to language models. The operation is a linear weighted sum for any
scalar weights. Callers that interpret it as a weighted cross-entropy mean must provide
nonnegative weights with sum one.
-/
def crossEntropyRowsNatWeighted {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {rows classes : Nat}
    (logits : RefTy (m := m) (α := α) (.dim rows (.dim classes .scalar)))
    (target : _root_.Runtime.Autograd.Torch.NatTensorRef (m := m) (α := α)
      (.dim rows .scalar))
    (weights : RefTy (m := m) (α := α) (.dim rows .scalar)) :
    m (RefTy (m := m) (α := α) Shape.scalar) := do
  let logp ← F.logSoftmax (m := m) (α := α)
    (s := .dim rows (.dim classes .scalar)) 1 logits
  nllRowsNatWeighted (m := m) (α := α) (rows := rows) (classes := classes)
    logp target weights

/--
Binary cross-entropy with logits (elementwise), using the stable identity:
`BCEWithLogits(x,y) = y * softplus(-x) + (1-y) * softplus(x)`.

Targets are expected in `[0,1]` (typically 0/1), same shape as `logits`.
-/
def bceWithLogits {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape}
    (logits target : RefTy (m := m) (α := α) s)
    (reduction : Reduction := .mean) :
    m (RefTy (m := m) (α := α) Shape.scalar) := do
  let onesT : Tensor α s := Spec.fill (1 : α) s
  let ones ← const (m := m) (α := α) (s := s) onesT
  let oneMinusY ← sub (m := m) (α := α) (s := s) ones target
  let negLogits ← scale (m := m) (α := α) (s := s) logits (-1)
  let spNeg ← softplus (m := m) (α := α) (s := s) negLogits
  let spPos ← softplus (m := m) (α := α) (s := s) logits
  let t1 ← mul (m := m) (α := α) (s := s) target spNeg
  let t2 ← mul (m := m) (α := α) (s := s) oneMinusY spPos
  let lossVec ← add (m := m) (α := α) (s := s) t1 t2
  reduce (m := m) (α := α) (s := s) lossVec reduction

/--
Binary cross-entropy on probabilities (elementwise):
`- (y * log(p) + (1-y) * log(1-p))`.

If you have logits, prefer `bceWithLogits`.
-/
def bce {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape}
    (probs target : RefTy (m := m) (α := α) s)
    (reduction : Reduction := .mean) (ε : α := Numbers.epsilon) :
    m (RefTy (m := m) (α := α) Shape.scalar) := do
  let onesT : Tensor α s := Spec.fill (1 : α) s
  let ones ← const (m := m) (α := α) (s := s) onesT
  let oneMinusP ← sub (m := m) (α := α) (s := s) ones probs
  let oneMinusY ← sub (m := m) (α := α) (s := s) ones target
  let logP ← safeLog (m := m) (α := α) (s := s) probs (ε := ε)
  let logOneMinusP ← safeLog (m := m) (α := α) (s := s) oneMinusP (ε := ε)
  let t1 ← mul (m := m) (α := α) (s := s) target logP
  let t2 ← mul (m := m) (α := α) (s := s) oneMinusY logOneMinusP
  let sumT ← add (m := m) (α := α) (s := s) t1 t2
  let negSumT ← scale (m := m) (α := α) (s := s) sumT (-1)
  reduce (m := m) (α := α) (s := s) negSumT reduction

end Loss

end TorchLean
