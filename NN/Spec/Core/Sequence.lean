/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape

/-!
# Shape-prefix and sequence operations

The prefix combinators act on arbitrary leading shapes. Sequence traversal records its length in the
result type.
-/

@[expose] public section


namespace Spec

namespace Sequence

/--
Traverse the indices `0, ..., n - 1`, threading a state and collecting one result per index.

Unlike a list fold, the result records in its type that the traversal produced exactly `n` values.
The indexed step is useful when the source is already represented as `Fin n → α`.
-/
def mapAccum {State Result : Type} :
    (n : Nat) → State → ((i : Fin n) → State → State × Result) →
      State × Tensor Result [n]
  | 0, state, _ => (state, Tensor.ofFn Fin.elim0)
  | n + 1, state, step =>
      let (next, result) := step 0 state
      let (final, results) := mapAccum n next (fun i => step i.succ)
      (final, Tensor.ofFn (Fin.cases result fun i => results.getScalar i))

/--
Traverse `Fin n` from right to left while returning results in their original index order.

This is the state-threading pattern used by reverse-mode passes through a fixed-length sequence.
-/
def mapAccumRight {State Result : Type} :
    (n : Nat) → State → ((i : Fin n) → State → State × Result) →
      State × Tensor Result [n]
  | 0, state, _ => (state, Tensor.ofFn Fin.elim0)
  | n + 1, state, step =>
      let last : Fin (n + 1) := ⟨n, Nat.lt_succ_self n⟩
      let (previous, result) := step last state
      let (initial, results) := mapAccumRight n previous (fun i => step i.castSucc)
      (initial, Tensor.ofFn fun i =>
        if h : i.val < n then results.getScalar ⟨i.val, h⟩ else result)

end Sequence

namespace Tensor

/-- Apply a function independently at every index of a leading shape. -/
def mapEach {α : Type} (leading : Shape) {inShape outShape : Shape}
    (f : Tensor α inShape → Tensor α outShape)
    (x : Tensor α (leading.concat inShape)) : Tensor α (leading.concat outShape) :=
  match leading with
  | .scalar => f x
  | .dim _ rest =>
      match x with
      | Tensor.dim values => Tensor.dim (fun i => mapEach rest f (values i))

/-- Zip two tensors pointwise across the same leading shape. -/
def zipEach {α : Type} (leading : Shape) {leftShape rightShape : Shape} (outShape : Shape)
    (f : Tensor α leftShape → Tensor α rightShape → Tensor α outShape)
    (left : Tensor α (leading.concat leftShape))
    (right : Tensor α (leading.concat rightShape)) : Tensor α (leading.concat outShape) :=
  match leading with
  | .scalar => f left right
  | .dim _ rest =>
      match left, right with
      | Tensor.dim left, Tensor.dim right =>
          Tensor.dim (fun i => zipEach rest outShape f (left i) (right i))

end Tensor
end Spec
