/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape

/-!
# Leading-axis tensor operations

These operations act on a leading shape prefix. The item shape is arbitrary: recurrent models often
use vectors, but the definitions do not assume a rank or scalar layout. Structural recursion keeps
the functions executable and gives proofs the same outer dimensions on which to induct.

These operations are not specific to recurrent models. They apply to any tensor with a leading
axis, regardless of the rank or meaning of each item.
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
    (n : Nat) → State → ((i : Fin n) → State → State × Result) → State × Vector Result n
  | 0, state, _ => (state, #v[])
  | n + 1, state, step =>
      let (next, result) := step 0 state
      let (final, results) := mapAccum n next (fun i => step i.succ)
      (final, ⟨#[result] ++ results.toArray, by simp [Nat.add_comm]⟩)

/--
Traverse `Fin n` from right to left while returning results in their original index order.

This is the state-threading pattern used by reverse-mode passes through a fixed-length sequence.
-/
def mapAccumRight {State Result : Type} :
    (n : Nat) → State → ((i : Fin n) → State → State × Result) → State × Vector Result n
  | 0, state, _ => (state, #v[])
  | n + 1, state, step =>
      let last : Fin (n + 1) := ⟨n, Nat.lt_succ_self n⟩
      let (previous, result) := step last state
      let (initial, results) := mapAccumRight n previous (fun i => step i.castSucc)
      (initial, results.push result)

end Sequence

variable {α : Type} [Context α]

namespace Tensor

/-- Apply a function independently at every index of a leading shape. -/
def mapLeading (leading : Shape) {inShape outShape : Shape}
    (f : Tensor α inShape → Tensor α outShape)
    (x : Tensor α (leading.concat inShape)) : Tensor α (leading.concat outShape) :=
  match leading with
  | .scalar => f x
  | .dim _ rest =>
      match x with
      | Tensor.dim values => Tensor.dim (fun i => mapLeading rest f (values i))

/-- Zip two tensors pointwise across the same leading shape. -/
def zipWithLeading (leading : Shape) {leftShape rightShape : Shape} (outShape : Shape)
    (f : Tensor α leftShape → Tensor α rightShape → Tensor α outShape)
    (left : Tensor α (leading.concat leftShape))
    (right : Tensor α (leading.concat rightShape)) : Tensor α (leading.concat outShape) :=
  match leading with
  | .scalar => f left right
  | .dim _ rest =>
      match left, right with
      | Tensor.dim left, Tensor.dim right =>
          Tensor.dim (fun i => zipWithLeading rest outShape f (left i) (right i))

/-- Sum along a nonempty leading dimension. -/
def sumLeadingAxis {length : Nat} {itemShape : Shape}
    (x : Tensor α (.dim length itemShape)) (h : length ≠ 0) : Tensor α itemShape :=
  reduceSum 0 x (Shape.hasNonemptyAxisZeroOfNe h).proof

/-- Reverse the leading dimension of a tensor. -/
def reverseLeadingAxis {length : Nat} {itemShape : Shape}
    (x : Tensor α (.dim length itemShape)) : Tensor α (.dim length itemShape) :=
  match x with
  | Tensor.dim values => Tensor.dim (fun i => values ⟨length - 1 - i.val, by grind⟩)

end Tensor
end Spec
