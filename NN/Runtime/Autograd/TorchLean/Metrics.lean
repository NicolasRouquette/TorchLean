/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape.LinearAlgebra
public import NN.Spec.Core.TensorReductionShape.Reductions
public import NN.Spec.Core.Tensor

/-!
# Metrics

TorchLean metrics helpers.

These are non-differentiable evaluation helpers for classification and accuracy reports.
-/

@[expose] public section


namespace TorchLean

open Spec
open Tensor

namespace Metrics

/--
Index of the first maximum in row-major storage order.

This operation accepts a tensor of any rank. It returns `none` exactly when the tensor has no
entries; ties are resolved in favor of the first index.
-/
def argmax? {α : Type} [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
    {s : Shape} (values : Tensor α s) : Option (Fin (Shape.size s)) :=
  let data := Spec.Tensor.toArray values
  match data[0]? with
  | none => none
  | some x0 =>
      let (_, bestIndex, _) := data.foldl (fun (index, bestIndex, bestValue) value =>
        if value > bestValue then (index + 1, index, value)
        else (index + 1, bestIndex, bestValue)) (0, 0, x0)
      if h : bestIndex < Shape.size s then some ⟨bestIndex, h⟩ else none

/--
Indices of the maxima along `axis`, in row-major order over all remaining dimensions.

The selected axis is moved to the innermost position before the tensor is flattened, so each
contiguous chunk is one class slice. An empty class axis contributes `none` for every slice of the
remaining shape. Ties are resolved in favor of the first index.
-/
def argmaxAxis? {α : Type} [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
    {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s] (values : Tensor α s) :
    Array (Option Nat) :=
  let argmaxChunk (xs : Array α) : Option Nat :=
    match xs[0]? with
    | none => none
    | some x0 =>
        let (_, bestIndex, _) := xs.foldl (fun (index, bestIndex, bestValue) value =>
          if value > bestValue then (index + 1, index, value)
          else (index + 1, bestIndex, bestValue)) (0, 0, x0)
        some bestIndex
  let axisExtent := Shape.axisSize s axis
  if axisExtent = 0 then
    Array.replicate (Shape.size (Tensor.shapeAfterSum s axis)) none
  else
    let swaps := Shape.moveAxisToInnermostSwaps (Shape.rank s) axis
    let moved := Tensor.permuteByAdjacentSwaps values swaps
    let data := Spec.Tensor.toArray moved
    (Array.finRange (Shape.size (Tensor.shapeAfterSum s axis))).map fun slice =>
      let start := slice.val * axisExtent
      argmaxChunk (data.extract start (start + axisExtent))

/--
Compare logits with one-hot targets along `axis`, once for every slice orthogonal to that axis.

An entry is `none` exactly when the selected class axis is empty. Otherwise it records whether the
two first-maximum indices agree.
-/
def correctOneHotAxis? {α : Type} [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
    {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (logits targetOneHot : Tensor α s) : Array (Option Bool) :=
  Array.zipWith (fun predicted target => do
    let predicted ← predicted
    let target ← target
    pure (predicted = target))
    (argmaxAxis? axis logits) (argmaxAxis? axis targetOneHot)

/-- Count correct and total one-hot classifications along `axis`. -/
def accuracyOneHotAxis {α : Type} [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
    {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (logits targetOneHot : Tensor α s) : Nat × Nat :=
  correctOneHotAxis? axis logits targetOneHot |>.foldl (fun (correct, total) result =>
    (if result = some true then correct + 1 else correct, total + 1)) (0, 0)

end Metrics

end TorchLean
