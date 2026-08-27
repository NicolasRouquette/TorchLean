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
Float-literal casting), prefer `NN/Tensor.lean` instead.

Design choice (why these are "total"):

- In the spec layer we would rather make edge cases explicit than throw runtime exceptions.
- If something is shape-invalid, we want Lean to reject it at elaboration time.
- Dynamic storage is admitted only through constructors that either carry an exact size proof or
  return an explicit failure. The resizing constructors near the end of this file are named as
  such and are not used implicitly.
-/

@[expose] public section


namespace Spec

/-- Fill a tensor of shape `s` with a constant value.

PyTorch analogy: `torch.full(shape, value)`.
-/
def fill {α : Type} (value : α): (s : Shape) → Tensor α s
  | Shape.scalar => Tensor.scalar value
  | Shape.dim _ s' => Tensor.dim (fun _ => fill value s')

/-- Flattening a filled tensor produces one copy of the value for every scalar position. -/
@[simp] theorem Tensor.toList_fill {α : Type} (value : α) (shape : Shape) :
    (fill value shape).toList = List.replicate shape.size value := by
  induction shape with
  | scalar => rfl
  | dim n shape ih =>
      simp only [fill, Tensor.toList, ih, Shape.size]
      induction n with
      | zero => simp
      | succ n hn =>
          rw [List.finRange_succ]
          simp only [List.flatMap_cons, List.flatMap_map]
          rw [show (List.finRange n).flatMap
              (fun _ => List.replicate shape.size value) =
              List.replicate (n * shape.size) value by exact hn]
          rw [← List.replicate_add]
          congr 1
          simp [Nat.succ_mul, Nat.add_comm]

/-- Every outer coordinate of a filled tensor is the corresponding filled subtensor. -/
@[simp] theorem get_fill {α : Type} (value : α) (n : Nat) (s : Shape) (i : Fin n) :
    get (fill value (.dim n s)) i = fill value s := by
  rfl

/-- Every coordinate of a filled matrix contains its fill value. -/
@[simp] theorem get2_fill {α : Type} (value : α) (m n : Nat) (i : Fin m) (j : Fin n) :
    get2 (fill value (.dim m (.dim n .scalar))) i j = value := by
  rfl

namespace Tensor

/--
Construct an arbitrary-rank tensor from a coordinate function.

The dimension list is outermost first. At each coordinate, `f` receives one natural-number index
per dimension. The indices are in bounds by construction; the list representation keeps callers
independent of the recursive implementation of `Shape` and `Tensor`.

For example, `Tensor.generate [2, 3] f` has shape `[2, 3]`, and its entry at row `i` and column `j`
is `f [i, j]`.
-/
def generate {α : Type} (dims : List Nat) (f : List Nat → α) : Tensor α (Shape.ofList dims) :=
  match dims with
  | [] => Tensor.scalar (f [])
  | _ :: rest => Tensor.dim (fun i => generate rest (fun is => f (i.val :: is)))

/-- Construct a vector from its coordinate function.

PyTorch analogy: `torch.tensor([...])` with shape `(n,)`, but our input is a function, not a list.
-/
def ofFn {α : Type} {n : Nat} (values : Fin n → α) : Tensor α [n] :=
  Tensor.dim (fun i => Tensor.scalar (values i))

/-- Reading a coordinate from `ofFn` returns the value supplied at that coordinate. -/
@[simp] theorem getScalar_ofFn {α : Type} {n : Nat} (values : Fin n → α) (i : Fin n) :
    (ofFn values).getScalar i = values i := by
  rfl

/-- Rebuilding a vector from all of its coordinates returns the original vector. -/
@[simp] theorem ofFn_getScalar {α : Type} {n : Nat} (t : Tensor α [n]) :
    ofFn (fun i => t.getScalar i) = t := by
  apply Tensor.ext_vector
  intro i
  simp

/-- Build a vector from an array whose length is known statically. -/
def ofArrayExact {α : Type} {n : Nat} (values : Array α) (h : values.size = n) :
    Tensor α [n] :=
  ofFn fun i => values[i.val]'(h.symm ▸ i.2)

/-- Indexing an exact-length vector reads the corresponding array entry. -/
@[simp] theorem getScalar_ofArrayExact {α : Type} {n : Nat} (values : Array α)
    (h : values.size = n) (i : Fin n) :
    (ofArrayExact values h).getScalar i = values[i.val]'(h.symm ▸ i.2) := by
  rfl

/-- Flattening an exact array constructor preserves the original row-major values. -/
@[simp] theorem toList_ofArrayExact {α : Type} {n : Nat} (values : Array α)
    (h : values.size = n) :
    (ofArrayExact values h).toList = values.toList := by
  subst n
  simp only [ofArrayExact, ofFn, Tensor.toList]
  rw [← List.map_eq_flatMap, ← List.ofFn_eq_map, ← Array.toList_ofFn]
  simp

/--
Build an arbitrary-rank tensor from flat row-major data whose length matches the shape.

This is the checked boundary between dynamic array storage and TorchLean's canonical tensor type.
After the size proof is available, numerical code should use the resulting `Tensor α shape`.
-/
def ofFlatArrayExact {α : Type} :
    (shape : Shape) → (values : Array α) → values.size = shape.size → Tensor α shape
  | .scalar, values, hSize => by
      have hLength : values.size = 1 := by
        simpa [Shape.size] using hSize
      exact .scalar (values[0]'(by simp [hLength]))
  | .dim n tail, values, hSize => by
      let chunkSize := tail.size
      have hLength : values.size = n * chunkSize := by
        simpa [Shape.size, chunkSize] using hSize
      refine .dim fun i => ?_
      let chunk : Array α := Array.ofFn fun j : Fin chunkSize =>
        values[i.val * chunkSize + j.val]'(by
          have hOffset : i.val * chunkSize + j.val < (i.val + 1) * chunkSize := by
            rw [Nat.add_mul, Nat.one_mul]
            exact Nat.add_lt_add_left j.2 (i.val * chunkSize)
          have hChunkEnd : (i.val + 1) * chunkSize ≤ n * chunkSize :=
            Nat.mul_le_mul_right chunkSize (Nat.succ_le_of_lt i.2)
          simpa [hLength] using hOffset.trans_le hChunkEnd)
      have hChunk : chunk.size = tail.size := by simp [chunk, chunkSize]
      exact ofFlatArrayExact tail chunk hChunk

/-- Construct a matrix from its row and column coordinate function.

PyTorch analogy: `torch.tensor([...]).reshape(m, n)` (again, function input rather than a list).
-/
def matrix {α : Type} {m n : Nat} (values : Fin m → Fin n → α) :
    Tensor α [m, n] :=
  Tensor.dim (fun i => ofFn (fun j => values i j))

end Tensor

/-- Every coordinate of a filled vector contains the fill value. -/
@[simp] theorem Tensor.getScalar_fill {α : Type} (value : α) (n : Nat) (i : Fin n) :
    (fill value (.dim n .scalar)).getScalar i = value := by
  rfl

/-- A singleton vector.

PyTorch analogy: `x.unsqueeze(0)` for a scalar `x`.
-/
def singleton {α : Type} (x : α) : Tensor α [1] :=
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

/-! ## Constant and dynamic-data constructors -/

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

/-- Build a matrix when every row has the same length; reject ragged input. -/
def matrixFromRows? {α : Type} (rows : List (List α)) :
    Option (Tensor α [rows.length, Option.getD (rows.head?.map List.length) 0]) :=
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
          Tensor.ofFn fun j =>
            have hIndex : j.val < row.length := by simpa [hLength] using j.isLt
            row.get ⟨j.val, hIndex⟩
      else
        none

end Spec
