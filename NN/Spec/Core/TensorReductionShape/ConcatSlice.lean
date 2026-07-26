/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape.LinearAlgebra

@[expose] public section


namespace Spec
namespace Tensor

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-!
# Concatenation and Slicing

Concatenation, slicing, sequence concatenation, squeeze/unsqueeze, and channel layout transforms.
-/

/-- Runtime check that a tensor value matches a runtime `Shape`.

We use this in a few “dynamic” utilities where we have a runtime shape value and want to guard
access/casts in a total way.
-/
def matchShape {s : Shape} (t : Tensor α s) : Shape → Prop :=
  match t with
  | .scalar _ =>
      fun
      | .scalar => True
      | .dim _ _ => False
  | .dim (n := n) f =>
      fun
      | .scalar => False
      | .dim n' s' => n = n' ∧ ∀ i : Fin n, (f i).matchShape s'

/-- Concatenate a list of `(n,d)` tensors along the last axis, producing `(n, headCount*d)`.

This is mainly used by attention blocks that split/merge heads.

PyTorch analogy: `torch.cat(heads, dim=-1)` after splitting heads, followed by a reshape.
-/
def concatSpec
  {α : Type} [Inhabited α]
  {n d : Nat}
  (headCount : Nat)
  (tensors : List (Tensor α (.dim n (.dim d .scalar))))
  (_h_len : tensors.length = headCount) :
  Tensor α (.dim n (.dim (headCount * d) .scalar)) :=

  -- Helper to get a single row at index i
  let concatRow (i : Fin n) : Tensor α (.dim (headCount * d) .scalar) :=
    let rec buildRow (ts : List (Tensor α (.dim n (.dim d .scalar)))) : List α :=
      match ts with
      | [] => []
      | t :: rest =>
        match t with
        | Tensor.dim f =>
          match f i with
          | Tensor.dim g =>
            let rowElems := (List.finRange d).map (fun j =>
              match g j with
              | Tensor.scalar a => a)
            rowElems ++ buildRow rest

    let values := buildRow tensors
    Tensor.dim (fun j : Fin (headCount * d) =>
      Tensor.scalar (values.getD j.val Inhabited.default))

  Tensor.dim (fun i : Fin n => concatRow i)

/-- Concatenate two vectors by appending `v2` after `v1`. -/
def concatVectorsSpec {α : Type} {n m : Nat}
  (v1 : Tensor α (.dim n .scalar))
  (v2 : Tensor α (.dim m .scalar)) :
  Tensor α (.dim (n + m) .scalar) :=
  match v1, v2 with
  | .dim f1, .dim f2 =>
    .dim fun i =>
      if h : i.val < n then
        f1 ⟨i.val, h⟩
      else
        let j : Fin m :=
        ⟨i.val - n, Nat.sub_lt_left_of_lt_add (Nat.not_lt.mp h) i.is_lt⟩
        f2 j

/-- Concatenate along axis 0 (append `t2` after `t1`). -/
def concatLeadingAxisSpec {α : Type} {n m : Nat} {s : Shape}
  (t1 : Tensor α (.dim n s))
  (t2 : Tensor α (.dim m s)) :
  Tensor α (.dim (n + m) s) :=
  match t1, t2 with
  | .dim f1, .dim f2 =>
    .dim fun i =>
      if h : i.val < n then
        f1 ⟨i.val, h⟩
      else
        let j : Fin m :=
          ⟨i.val - n, Nat.sub_lt_left_of_lt_add (Nat.not_lt.mp h) i.is_lt⟩
        f2 j

/-!
## Slicing / concatenation on the leading axis

`concat_leading_axis_spec` is the "append on axis 0" primitive that powers many higher-level utilities
(sequence concatenation, channel skip connections, etc.).

For backprop and for "undoing" concatenations, it is convenient to have an explicit slice operation.
We keep the API compact and index-safe:

- `slice_leading_axis_range_spec start len` selects `len` consecutive entries starting at `start` along axis 0.
- `concat_leading_axis_backward_spec` is the adjoint of `concat_leading_axis_spec` (splits a gradient tensor).
-/

/-- Slice `len` entries along axis 0, starting at `start`.

This is the simplest "range slice" one typically needs to express:
- taking the first `n` channels/tokens,
- extracting the skip-connection half after a concat,
- implementing `take`/`drop` without changing the inner shape.

The proof `len + start ≤ n` makes the slice total (no out-of-bounds behavior). -/
def sliceLeadingAxisRangeSpec {α : Type} {n : Nat} {s : Shape}
  (start len : Nat) (h : len + start ≤ n)
  (t : Tensor α (.dim n s)) : Tensor α (.dim len s) :=
  match t with
  | .dim f =>
      .dim fun i =>
        let idx : Nat := start + i.val
        have h1 : idx < start + len := by
          simp [idx]
        have h2 : start + len ≤ n := by
          simpa [Nat.add_comm] using h
        f ⟨idx, lt_of_lt_of_le h1 h2⟩

/-- Backward (adjoint) of `concat_leading_axis_spec`.

If `y = concat_leading_axis_spec x1 x2`, then in reverse-mode we split the upstream gradient `δy` into:

- $\delta x_1$: the first $n$ entries of $\delta y$,
- $\delta x_2$: the last $m$ entries of $\delta y$. -/
def concatLeadingAxisBackwardSpec {α : Type} {n m : Nat} {s : Shape}
  (δ : Tensor α (.dim (n + m) s)) :
  Tensor α (.dim n s) × Tensor α (.dim m s) :=
  let δ₁ := sliceLeadingAxisRangeSpec (α := α) (n := n + m) (s := s) 0 n (Nat.le_add_right n m) δ
  let δ₂ :=
    sliceLeadingAxisRangeSpec (α := α) (n := n + m) (s := s) n m
      (by simp [Nat.add_comm]) δ
  (δ₁, δ₂)

/--
Backward (adjoint) of `slice_leading_axis_range_spec`.

If `y = slice_leading_axis_range_spec start len x`, then `slice_leading_axis_range_backward_spec start len δy` re-inserts
the gradient into the original shape and fills everything outside the slice with zeros.
-/
def sliceLeadingAxisRangeBackwardSpec {α : Type} [Zero α] {n : Nat} {s : Shape}
  (start len : Nat) (_h : len + start ≤ n)
  (δ : Tensor α (.dim len s)) : Tensor α (.dim n s) :=
  -- This is the adjoint of `slice_leading_axis_range_spec`: the gradient is re-inserted into the
  -- original shape and everything outside the slice is filled with zeros.
  Tensor.dim (fun i =>
    if h1 : i.val < start then
      fill (0 : α) s
    else if h2 : i.val < start + len then
      let j : Fin len :=
        ⟨i.val - start, Nat.sub_lt_left_of_lt_add (Nat.not_lt.mp h1) h2⟩
      getAtSpec δ j
    else
      fill (0 : α) s)

/--
Concatenate two sequences along time (axis 0), producing a longer sequence.

If `leftSeq : (seqLen1 x hidden)` and `rightSeq : (seqLen2 x hidden)`, this returns
`(seqLen1 + seqLen2) x hidden` by appending `rightSeq` after `leftSeq`.

Do not confuse this with `Spec.concatSequenceSpec` (defined in `NN.Spec.Core.Sequence`), which
concatenates along the feature dimension for *same-length* sequences.
-/
def concatSequenceSpec {α : Type} {seqLen1 seqLen2 hiddenSize : Nat}
  (leftSeq : Tensor α (.dim seqLen1 (.dim hiddenSize .scalar)))
  (rightSeq : Tensor α (.dim seqLen2 (.dim hiddenSize .scalar))) :
  Tensor α (.dim (seqLen1 + seqLen2) (.dim hiddenSize .scalar)) :=
  match leftSeq, rightSeq with
  | .dim f1, .dim f2 =>
    .dim fun i =>
      if h : i.val < seqLen1 then
        f1 ⟨i.val, h⟩
      else
        let j : Fin seqLen2 :=
          ⟨i.val - seqLen1, Nat.sub_lt_left_of_lt_add (Nat.not_lt.mp h) i.is_lt⟩
        f2 j

/-- Concatenate two sequences along the feature dimension (inner axis). -/
def concatSequenceInnerSpec {α : Type} {seqLen hiddenSize1 hiddenSize2 : Nat}
  (leftSeq : Tensor α (.dim seqLen (.dim hiddenSize1 .scalar)))
  (rightSeq : Tensor α (.dim seqLen (.dim hiddenSize2 .scalar))) :
  Tensor α (.dim seqLen (.dim (hiddenSize1 + hiddenSize2) .scalar)) :=
  match leftSeq, rightSeq with
  | .dim f1, .dim f2 =>
    .dim fun i =>
      match f1 i, f2 i with
      | .dim g1, .dim g2 =>
        .dim fun j =>
          if h : j.val < hiddenSize1 then
            g1 ⟨j.val, h⟩
          else
            let k : Fin hiddenSize2 :=
              ⟨j.val - hiddenSize1, Nat.sub_lt_left_of_lt_add (Nat.not_lt.mp h) j.is_lt⟩
            g2 k

/-- Expand a `(n, s)` tensor into `(n, 1, s)` by inserting a trailing dimension of size 1.

PyTorch analogy: `t.unsqueeze(-1)` for a rank-1 outer dimension (or `unsqueeze(dim=1)` in 2D terms).
  -/
def expandToColSpec {n s} (t : Tensor α (.dim n s)) : Tensor α (.dim n (.dim 1 s)) :=
  Tensor.dim (fun i => Tensor.dim (fun _ => getAtSpec t i))

/-- Same as `expand_to_col_spec`, specialized to vectors. -/
def expandToColSpecAlt {α : Type} {n : Nat} (v : Tensor α (Shape.dim n Shape.scalar)) :
    Tensor α (Shape.dim n (Shape.dim 1 Shape.scalar)) :=
  Tensor.dim (fun i => Tensor.dim (fun _ => getAtSpec v i))

/-- Squeeze a `(n,1,s)` tensor back into `(n,s)` by dropping the singleton dimension. -/
def squeezeColSpec {n s} (t : Tensor α (.dim n (.dim 1 s))) : Tensor α (.dim n s) :=
  Tensor.dim (fun i => getAtSpec (getAtSpec t i) 0)

/-- Same as `squeeze_col_spec`, specialized to vectors. -/
def squeezeColSpecAlt {α : Type} {n : Nat} (t : Tensor α (Shape.dim n (Shape.dim 1
  Shape.scalar))) :
    Tensor α (Shape.dim n Shape.scalar) :=
  Tensor.dim (fun i => getAtSpec (getAtSpec t i) 0)

/-- Unsqueeze (insert a singleton dim). Currently implemented as `expand_to_col_spec`.

Core uses singleton insertion mainly for column vectors, so this operation is specialized to that
use case.
General axis insertion can extend this definition. -/
def unsqueezeSpec {n s} (t : Tensor α (.dim n s)) (_dim : Nat) : Tensor α (.dim n (.dim 1 s)) :=
  expandToColSpec t

-- Expand vector to batch dimension
/-- Turn a vector `(n)` into a batch of size 1: `(1,n)`. -/
def expandVecToBatchSpec {α : Type} {n : Nat} (v : Tensor α (Shape.dim n Shape.scalar)) :
    Tensor α (Shape.dim 1 (Shape.dim n Shape.scalar)) :=
  Tensor.dim (fun _ => v)

-- Batch dimension manipulation: move batch to end
/-- Move a leading batch dimension to the innermost position. -/
def batchToEndSpec {α : Type} {batch : Nat} {s : Shape}
  (t : Tensor α (.dim batch s)) :
  Tensor α (s.appendDim batch) :=
  match s, t with
  | .scalar, .dim f =>
    -- Input: [batch, scalar] -> Output: [scalar, batch] = [batch]
    -- f : Fin batch -> Tensor α .scalar
    Tensor.dim fun i => f i
  | .dim _ _, .dim f =>
    -- Input: [batch, n, rest...] -> Output: [n, rest..., batch]
    -- f : Fin batch -> Tensor α (.dim n rest)
    -- We need to build: Tensor α (.dim n (rest.appendDim batch))
    Tensor.dim fun j =>
      -- For each position j in the new first dimension n
      -- We need: Tensor α (rest.appendDim batch)
      -- This comes from collecting f[i][j] for all i and restructuring
      collectAtIndexSpec (fun i => match f i with | Tensor.dim g => Tensor.dim (fun _ => g j)) j

-- Channel-first to channel-last (common in vision): (batch, channels, height, width) -> (batch,
-- height, width, channels)
/-- Convert channel-first images `(b,c,h,w)` into channel-last `(b,h,w,c)`. -/
def channelFirstToLastSpec {α : Type} {b c h w : Nat}
  (t : Tensor α (.dim b (.dim c (.dim h (.dim w .scalar))))) :
  Tensor α (.dim b (.dim h (.dim w (.dim c .scalar)))) :=
  match t with
  | .dim f_b =>
    .dim fun i_b =>
      match f_b i_b with
      | .dim f_c =>
        .dim fun i_h =>
          .dim fun i_w =>
            .dim fun i_c =>
              match f_c i_c with
              | .dim f_h =>
                match f_h i_h with
                | .dim f_w => .scalar (match f_w i_w with | .scalar x => x)
end Tensor
end Spec
