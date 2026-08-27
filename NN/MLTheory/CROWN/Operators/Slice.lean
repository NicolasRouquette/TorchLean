/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Core
public import NN.MLTheory.CROWN.Flatbox
public import NN.Spec.Core.Context
public import NN.Spec.Core.Tensor
public import NN.Spec.Core.TensorOps

/-!
# Slice / gather / split operator bounds

This file provides IBP and affine transfer rules for a small subset of indexing-like operations:
- `Slice`: extract a contiguous range `[start, stop)` from a flattened tensor,
- `Gather`: select entries by a runtime array of static indices, and
- `Split`: split a flattened tensor into an array of parts.

Important limitation: this does **not** model tensor-valued index dtypes inside the differentiable
graph (i.e. no PyTorch-style `LongTensor` indexing/gather/scatter driven by data tensors).
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Operators.Slice

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN

variable {α : Type} [Context α]

  /-- View a `(.dim n .scalar)` tensor as its underlying `Fin n → Tensor α .scalar` function. -/
  def getDimScalarFn {n : Nat} (t : Tensor α [n]) : Fin n → Tensor α .scalar :=
    match t with
    | .dim f => f

/-- IBP for Slice: extract elements [start, stop) from a flattened vector.
    Slice is a linear operation, so bounds propagate exactly.
-/
def ibpSlice? (xB : FlatBox α) (start stop : Nat) : Option (FlatBox α) :=
  let outDim := stop - start
  if start < stop ∧ stop ≤ xB.dim then
    let flo := getDimScalarFn xB.lo
    let fhi := getDimScalarFn xB.hi
    let outLo := Tensor.dim (fun i : Fin outDim =>
      let idx := start + i.val
      if hidx : idx < xB.dim then
        flo ⟨idx, hidx⟩
      else
        Tensor.scalar Numbers.zero)
    let outHi := Tensor.dim (fun i : Fin outDim =>
      let idx := start + i.val
      if hidx : idx < xB.dim then
        fhi ⟨idx, hidx⟩
      else
        Tensor.scalar Numbers.zero)
    some { dim := outDim, lo := outLo, hi := outHi }
  else
    none

/-- IBP for Gather: index into a vector using integer indices.

For input $x$ and a concrete index vector, the output satisfies
$y_j=x_{\mathrm{indices}[j]}$. This is a permutation or selection.
-/
def ibpGather? (xB : FlatBox α) (indices : Array Nat) : Option (FlatBox α) :=
  let outDim := indices.size
  let flo := getDimScalarFn xB.lo
  let fhi := getDimScalarFn xB.hi
  if indices.all (· < xB.dim) then
    let outLo := Tensor.dim (fun j : Fin outDim =>
      match indices[j.val]? with
      | some idx =>
        if hidx : idx < xB.dim then flo ⟨idx, hidx⟩ else Tensor.scalar Numbers.zero
      | none => Tensor.scalar Numbers.zero)
    let outHi := Tensor.dim (fun j : Fin outDim =>
      match indices[j.val]? with
      | some idx =>
        if hidx : idx < xB.dim then fhi ⟨idx, hidx⟩ else Tensor.scalar Numbers.zero
      | none => Tensor.scalar Numbers.zero)
    some { dim := outDim, lo := outLo, hi := outHi }
  else
    none

/-- IBP for Split: split a rank-one tensor into multiple parts.
    Returns an array of `FlatBox` values, one for each split.
-/
def ibpSplit? (xB : FlatBox α) (splitSizes : Array Nat) : Option (Array (FlatBox α)) :=
  let flo := getDimScalarFn xB.lo
  let fhi := getDimScalarFn xB.hi
  let buildSplits : Array (FlatBox α) × Nat :=
    splitSizes.foldl (fun (state : Array (FlatBox α) × Nat) size =>
      let boxes := state.1
      let offset := state.2
      let box := {
        dim := size
        lo := Tensor.dim (fun i : Fin size =>
          let idx := offset + i.val
          if hidx : idx < xB.dim then
            flo ⟨idx, hidx⟩
          else
            Tensor.scalar Numbers.zero)
        hi := Tensor.dim (fun i : Fin size =>
          let idx := offset + i.val
          if hidx : idx < xB.dim then
            fhi ⟨idx, hidx⟩
          else
            Tensor.scalar Numbers.zero)
      }
      (boxes.push box, offset + size)) (#[], 0)
  if splitSizes.foldl (· + ·) 0 = xB.dim then some buildSplits.1 else none

/-- Affine bounds for Slice: extract a subvector of an affine form.

If the input represents $y=Ax+c$, slicing selects the corresponding rows of $A$ and entries of $c$.
-/
def affSlice? {inDim outDim : Nat} (start sliceSize : Nat)
    (aff : AffineVec α inDim outDim) : Option (AffineVec α inDim sliceSize) :=
  if start + sliceSize ≤ outDim then
    match aff.A, aff.c with
    | .dim rows, .dim cv =>
      let A' := Tensor.dim (fun i : Fin sliceSize =>
        let srcIdx := start + i.val
        if hsrc : srcIdx < outDim then
          rows ⟨srcIdx, hsrc⟩
        else
          -- Out of bounds: return zero row
          Tensor.dim (fun _ : Fin inDim => Tensor.scalar Numbers.zero))
      let c' := Tensor.dim (fun i : Fin sliceSize =>
        let srcIdx := start + i.val
        if hsrc : srcIdx < outDim then
          cv ⟨srcIdx, hsrc⟩
        else
          Tensor.scalar Numbers.zero)
      some { A := A', c := c' }
  else
    none

/-- Affine bounds for Gather: permute/select rows of affine form. -/
def affGather? {inDim outDim : Nat} (indices : Array Nat)
    (aff : AffineVec α inDim outDim) : Option (AffineVec α inDim indices.size) :=
  if indices.all (· < outDim) then
    match aff.A, aff.c with
    | .dim rows, .dim cv =>
      let A' := Tensor.dim (fun j : Fin indices.size =>
        match indices[j.val]? with
        | some idx =>
          if hidx : idx < outDim then rows ⟨idx, hidx⟩
          else Tensor.dim (fun _ : Fin inDim => Tensor.scalar Numbers.zero)
        | none => Tensor.dim (fun _ : Fin inDim => Tensor.scalar Numbers.zero))
      let c' := Tensor.dim (fun j : Fin indices.size =>
        match indices[j.val]? with
        | some idx =>
          if hidx : idx < outDim then cv ⟨idx, hidx⟩ else Tensor.scalar Numbers.zero
        | none => Tensor.scalar Numbers.zero)
      some { A := A', c := c' }
  else
    none

/-- Derivative bounds for Slice: derivatives just slice through. -/
def derivSlice? (dB : FlatBox α) (start stop : Nat) : Option (FlatBox α) :=
  ibpSlice? dB start stop

/-- Derivative bounds for Gather: derivatives follow the same indexing. -/
def derivGather? (dB : FlatBox α) (indices : Array Nat) : Option (FlatBox α) :=
  ibpGather? dB indices

/-- Concatenate multiple FlatBoxes into one. -/
def ibpConcat (boxes : Array (FlatBox α)) : FlatBox α :=
  let totalDim := boxes.foldl (fun acc b => acc + b.dim) 0
  if h : totalDim > 0 then
    let buildConcat := boxes.foldl (fun (acc : Array α × Array α) b =>
      let (loArr, hiArr) := acc
      let flo := getDimScalarFn b.lo
      let fhi := getDimScalarFn b.hi
      let newLo := (Array.finRange b.dim).foldl (fun arr i =>
        match flo i with
        | .scalar v => arr.push v
      ) loArr
      let newHi := (Array.finRange b.dim).foldl (fun arr i =>
        match fhi i with
        | .scalar v => arr.push v
      ) hiArr
      (newLo, newHi)
    ) (#[], #[])
    let (loArr, hiArr) := buildConcat
    { dim := totalDim
    , lo := Tensor.dim (fun i : Fin totalDim =>
        Tensor.scalar (if h : i.val < loArr.size then loArr[i.val] else Numbers.zero))
    , hi := Tensor.dim (fun i : Fin totalDim =>
        Tensor.scalar (if h : i.val < hiArr.size then hiArr[i.val] else Numbers.zero))
    }
  else
    { dim := 0
    , lo := Tensor.dim (fun i : Fin 0 => i.elim0)
    , hi := Tensor.dim (fun i : Fin 0 => i.elim0) }

end NN.MLTheory.CROWN.Operators.Slice
