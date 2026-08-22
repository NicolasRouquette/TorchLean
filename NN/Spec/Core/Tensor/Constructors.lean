/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context
public import NN.Spec.Core.Tensor.Core

/-!
# Tensor constructors (spec layer)

These are small, **total** constructors for building `Spec.Tensor` values directly.

They are used heavily inside the spec layer (models/layers) and in proofs, where we want:

- straightforward definitional unfolding, and
- no dependence on `IO` or dynamic shape checks.

If you want a PyTorch-style user experience for examples (dynamic dims, runtime errors, and
Float-literal casting), prefer `NN/Tensor/API.lean` instead.

Design choice (why these are "total"):

- In the spec layer we would rather make edge cases explicit than throw runtime exceptions.
- If something is shape-invalid, we want Lean to reject it at elaboration time.
- If something is index-invalid at runtime (e.g. array-backed constructor with wrong size), we
  choose a predictable fallback (usually `Inhabited.default`) and let *verification* code decide
  whether that situation is allowed.
-/

@[expose] public section


namespace Spec

/-- Fill a tensor of shape `s` with a constant value.

PyTorch analogy: `torch.full(shape, value)`.
-/
def fill {α : Type} (value : α): (s : Shape) → Tensor α s
  | Shape.scalar => Tensor.scalar value
  | Shape.dim _ s' => Tensor.dim (fun _ => fill value s')

/-- Every outer coordinate of a filled tensor is the corresponding filled subtensor. -/
@[simp] theorem get_fill {α : Type} (value : α) (n : Nat) (s : Shape) (i : Fin n) :
    get (fill value (.dim n s)) i = fill value s := by
  rfl

/-- Every coordinate of a filled matrix contains its fill value. -/
@[simp] theorem get2_fill {α : Type} (value : α) (m n : Nat) (i : Fin m) (j : Fin n) :
    get2 (fill value (.dim m (.dim n .scalar))) i j = value := by
  rfl

namespace Tensor

/-- Construct a vector from its coordinate function.

PyTorch analogy: `torch.tensor([...])` with shape `(n,)`, but our input is a function, not a list.
-/
def vector {α : Type} {n : Nat} (values : Fin n → α) : Tensor α (.dim n .scalar) :=
  Tensor.dim (fun i => Tensor.scalar (values i))

/-- Construct a matrix from its row and column coordinate function.

PyTorch analogy: `torch.tensor([...]).reshape(m, n)` (again, function input rather than a list).
-/
def matrix {α : Type} {m n : Nat} (values : Fin m → Fin n → α) :
    Tensor α (.dim m (.dim n .scalar)) :=
  Tensor.dim (fun i => vector (fun j => values i j))

end Tensor

/-- A singleton vector.

PyTorch analogy: `x.unsqueeze(0)` for a scalar `x`.
-/
def singleton {α : Type} (x : α) : Tensor α (.dim 1 .scalar) :=
  Tensor.dim (fun _ => Tensor.scalar x)

/--
Pad a tensor with `n` leading dimensions of size 1.

This is the tensor-level companion of `Shape.padLeft`. It is useful for broadcasting-style
normalization: if you need a tensor to have extra leading batch dimensions of size `1`, this does
so without changing any underlying values.

PyTorch analogy: repeated `unsqueeze(0)` (or viewing a tensor as having extra leading singleton
  dims).
-/
def padLeft {α : Type} [Context α]
  {n : Nat} {s : Shape} (x : Tensor α s)
  : Tensor α (Shape.padLeft n s) :=
  match n with
  | 0 => x
  | Nat.succ _ =>
    let inner := padLeft x
    .dim (fun _ => inner)  -- Only 1 element along new dim

/-- Build one tensor dimension from an array of inner tensors.

The explicit size proof prevents silent truncation or padding. Taking tensors as array elements
makes this constructor independent of rank: use scalar tensors for a vector, vectors for a matrix,
or arbitrary inner tensors for higher-rank values.
-/
def Tensor.ofArray {α : Type} {n : Nat} {s : Shape}
    (xs : Array (Tensor α s)) (_h : n = xs.size) : Tensor α (.dim n s) :=
  Tensor.dim (fun i : Fin n =>
    xs[i.val]'(by simpa [_h] using i.2))

/--
View a length-`n` vector as an `n x 1` "column" matrix.

PyTorch analogy: `v.unsqueeze(-1)` for a 1D tensor `v` (or `v.reshape(n, 1)`).
-/
def Tensor.vecToCol {α : Type} {n : Nat}
    (v : Tensor α (.dim n .scalar)) : Tensor α (.dim n (.dim 1 .scalar)) :=
  Tensor.dim (fun i : Fin n =>
    Tensor.dim (fun _ : Fin 1 => get v i))

/-! ## Constant and list-backed constructors -/

/-- A zero-filled tensor of shape `s`. -/
def zeros (α : Type) [Zero α] (s : Shape) : Tensor α s :=
  fill (0 : α) s

/-- A one-filled tensor of shape `s`. -/
def ones (α : Type) [One α] (s : Shape) : Tensor α s :=
  fill (1 : α) s

/-- A filled tensor satisfies every pointwise property satisfied by its value. -/
theorem Tensor.forall_fill {α : Type} {p : α → Prop} {s : Shape} {x : α}
    (hx : p x) : Tensor.Forall p (Spec.fill x s) := by
  induction s with
  | scalar => exact hx
  | dim _ _ ih =>
      intro _
      exact ih

/-- Build a vector from a list, retaining the list length in its shape. -/
def vectorFromList {α : Type} (xs : List α) : Tensor α (.dim xs.length .scalar) :=
  Tensor.dim fun i => Tensor.scalar (xs.get i)

/-- Build a matrix when every row has the same length; reject ragged input. -/
def matrixFromRows? {α : Type} (rows : List (List α)) :
    Option (Tensor α (.dim rows.length
      (.dim (Option.getD (rows.head?.map List.length) 0) .scalar))) :=
  match rows with
  | [] => some (Tensor.dim fun i => nomatch i)
  | first :: rest =>
      let allRows := first :: rest
      let columnCount := first.length
      if hRectangular : ∀ row ∈ allRows, row.length = columnCount then
        some <| Tensor.dim fun i =>
          let row := allRows.get i
          have hLength : row.length = columnCount :=
            hRectangular row (List.get_mem allRows i)
          Tensor.vector fun j =>
            have hIndex : j.val < row.length := by simpa [hLength] using j.isLt
            row.get ⟨j.val, hIndex⟩
      else
        none

/-- Return the greatest row length, or zero for an empty list. -/
def maxRowLength {α : Type} (rows : List (List α)) : Nat :=
  rows.foldl (fun n row => Nat.max n row.length) 0

/-- Resize every row to `columnCount`, padding short rows with `default`. -/
def matrixFromRowsResize {α : Type} [Inhabited α]
    (columnCount : Nat) (rows : List (List α)) :
    Tensor α (.dim rows.length (.dim columnCount .scalar)) :=
  Tensor.dim fun i =>
    let row := rows.getD i.val []
    Tensor.vector fun j => row.getD j.val default

/-- Pad every row on the right to the length of the longest row. -/
def matrixFromRowsPadRight {α : Type} [Inhabited α] (rows : List (List α)) :
    Tensor α (.dim rows.length (.dim (maxRowLength rows) .scalar)) :=
  matrixFromRowsResize (maxRowLength rows) rows

end Spec
