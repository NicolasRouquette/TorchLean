/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.Functional.Einsum

@[expose] public section


namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor

namespace F

/-! ## Shape/axis helpers -/

/--
Swap two adjacent axes at a given nesting depth.

This is the primitive used to implement general permutations via a sequence of adjacent swaps.
It corresponds to the backend op `Torch.swapAdjacentAtDepth`.
-/
def swapAdjacentAtDepth {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (depth : Nat) (x : RefTy (m := m) (α := α) s) :
    m (RefTy (m := m) (α := α) (s.swapAdjacentAtDepth depth)) :=
  _root_.Runtime.Autograd.Torch.swapAdjacentAtDepth (m := m) (α := α) (s := s) depth x

/-! ## Core tensor semantics (PyTorch-style) -/

/-- Detect duplicate `Nat`s (used to validate axis lists at runtime). -/
def hasDupNat (xs : List Nat) : Bool :=
  let rec go (seen : List Nat) : List Nat → Bool
    | [] => false
    | x :: xs => if seen.contains x then true else go (x :: seen) xs
  go [] xs

/-- Insert `x` into a list kept in descending order. -/
def insertDesc (x : Nat) : List Nat → List Nat
  | [] => [x]
  | y :: ys => if x ≥ y then x :: y :: ys else y :: insertDesc x ys

/-- Sort a list of `Nat`s in descending order (small insertion sort). -/
def sortDesc (xs : List Nat) : List Nat :=
  xs.foldl (fun acc x => insertDesc x acc) []

/-- Swap depths that move an axis to the last position (for “reduce along axis” lowering). -/
def moveAxisToLastSwaps (r axis : Nat) : List Nat :=
  let nSteps := r - (axis + 1)
  (List.range nSteps).map (fun i => axis + i)

/-- Swap depths that move an axis to the front position. -/
def moveAxisToFrontSwaps (axis : Nat) : List Nat :=
  (List.range axis).reverse

/-- Decidable `Shape.well_formed` for the dynamic reduction/slicing helpers. -/
def wellFormedDec : (s : Shape) → Decidable s.wellFormed
  | .scalar => isTrue trivial
  | .dim n s =>
      match (inferInstance : Decidable (n > 0)) with
      | isTrue hn =>
          match wellFormedDec s with
          | isTrue hs => isTrue ⟨hn, hs⟩
          | isFalse hs => isFalse (fun h => hs h.2)
      | isFalse hn =>
          isFalse (fun h => hn h.1)

/-- Local decidability instance for `Shape.well_formed` (used by dynamic reduction/slicing helpers).
  -/
instance (s : Shape) : Decidable s.wellFormed :=
  wellFormedDec s

/-- `Shape.appendDim s 1` preserves size (used to justify `reshape` in unsqueeze/keepdim code). -/
theorem size_appendDim_one' (s : Shape) : Shape.size (Shape.appendDim s 1) = Shape.size s := by
  induction s with
  | scalar => simp [Shape.appendDim, Shape.size]
  | dim n s ih => simp [Shape.appendDim, Shape.size, ih]

/--
Dynamic permutation: like `permute`, but returns an existential output shape.

PyTorch analogue: `torch.permute` / `Tensor.permute` (with runtime checks).
-/
def permuteDyn {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape}
    (axes : List Nat)
    (x : RefTy (m := m) (α := α) s) :
    m (Option (Σ s' : Shape, RefTy (m := m) (α := α) s')) := do
  let r := Shape.rank s
  if axes.length != r then
    return none
  if hasDupNat axes then
    return none
  if !(axes.all (fun a => a < r)) then
    return none
  let some swaps := Einsum.swapDepthsForPerm? axes r | return none
  let out ← Einsum.permuteBySwaps (α := α) (m := m) ⟨s, x⟩ swaps
  pure (some out)

/--
Permutation with an expected output shape.

This calls `permuteDyn` and checks that the computed shape equals `sOut`.
-/
def permute {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s sOut : Shape}
    (axes : List Nat)
    (x : RefTy (m := m) (α := α) s) :
    m (Option (RefTy (m := m) (α := α) sOut)) := do
  let y? ← permuteDyn (α := α) (m := m) (s := s) axes x
  match y? with
  | none => pure none
  | some ⟨s', y⟩ =>
      if h : s' = sOut then
        pure (some (h ▸ y))
      else
        pure none

namespace Internal

/--
Reduce along the last axis with `sum`, returning the new (existential) shape.

This is the primitive step used by `reduceDimsDynCore` after it has permuted the requested axis to
the last position.
-/
def reduceAlongLastSum {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    (x : Σ s : Shape, RefTy (m := m) (α := α) s) :
    m (Option (Σ s' : Shape, RefTy (m := m) (α := α) s')) := do
  let s := x.fst
  if hw : s.wellFormed then
    letI : Shape.WellFormed s := ⟨hw⟩
    if hRank : Shape.rank s > 0 then
      let axis := Shape.rank s - 1
      haveI : Shape.valid_axis_inst axis s :=
        Shape.validAxisLastInst (s := s) hRank hw
      _root_.Runtime.Autograd.Torch.reduceSum (m := m) (α := α) (s := s) axis x.snd >>= fun y =>
        pure (some ⟨Spec.Tensor.shapeAfterSum s axis, y⟩)
    else
      pure none
  else
    pure none

/-- Like `reduceAlongLastSum`, but using `mean`. -/
def reduceAlongLastMean {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    (x : Σ s : Shape, RefTy (m := m) (α := α) s) :
    m (Option (Σ s' : Shape, RefTy (m := m) (α := α) s')) := do
  let s := x.fst
  if hw : s.wellFormed then
    letI : Shape.WellFormed s := ⟨hw⟩
    if hRank : Shape.rank s > 0 then
      let axis := Shape.rank s - 1
      haveI : Shape.valid_axis_inst axis s :=
        Shape.validAxisLastInst (s := s) hRank hw
      _root_.Runtime.Autograd.Torch.reduceMean (m := m) (α := α) (s := s) axis x.snd >>= fun y =>
        pure (some ⟨Spec.Tensor.shapeAfterSum s axis, y⟩)
    else
      pure none
  else
    pure none

/--
Core implementation for dynamic reductions over multiple axes.

This lowers “reduce along axis k” to:
1. permute axis `k` to the last position,
2. call `reduceLast`, and
3. optionally re-insert a singleton dimension when `keepdim = true`.

`reduce_sum_dimsDyn` and `reduce_mean_dimsDyn` are just specializations.
-/
def reduceDimsDynCore {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    (reduceLast :
      (Σ s : Shape, RefTy (m := m) (α := α) s) →
        m (Option (Σ s' : Shape, RefTy (m := m) (α := α) s')))
    {s : Shape}
    (axes : List Nat)
    (keepdim : Bool)
    (x : RefTy (m := m) (α := α) s) :
    m (Option (Σ s' : Shape, RefTy (m := m) (α := α) s')) := do
  let r0 := Shape.rank s
  if hasDupNat axes then
    return none
  if !(axes.all (fun a => a < r0)) then
    return none
  let axes' := if keepdim then axes else sortDesc axes
  let mut cur : Σ s : Shape, RefTy (m := m) (α := α) s := ⟨s, x⟩
  for axis in axes' do
    let r := Shape.rank cur.fst
    if axis ≥ r then
      return none
    let swaps := moveAxisToLastSwaps r axis
    let curMoved ← Einsum.permuteBySwaps (α := α) (m := m) cur swaps
    let some curRed ← reduceLast curMoved | return none
    if keepdim then
      let sReshape : Shape := Shape.appendDim curRed.fst 1
      have hSz : Shape.size curRed.fst = Shape.size sReshape := by
        simpa [sReshape] using (Eq.symm (size_appendDim_one' curRed.fst))
      let xReshaped ← reshape (m := m) (α := α) (s₁ := curRed.fst) (s₂ := sReshape) curRed.snd hSz
      let curKeep : Σ s : Shape, RefTy (m := m) (α := α) s := ⟨sReshape, xReshaped⟩
      let curBack ← Einsum.permuteBySwaps (α := α) (m := m) curKeep swaps.reverse
      cur := curBack
    else
      cur := curRed
  pure (some cur)

end Internal

/-- Dynamic multi-axis sum reduction (like `torch.sum(x, dim=axes, keepdim=...)`). -/
def reduceSumDimsDyn {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape}
    (axes : List Nat)
    (x : RefTy (m := m) (α := α) s)
    (keepdim : Bool := false) :
    m (Option (Σ s' : Shape, RefTy (m := m) (α := α) s')) :=
  Internal.reduceDimsDynCore (α := α) (m := m) Internal.reduceAlongLastSum (s := s) axes keepdim x

/-- Dynamic multi-axis mean reduction (like `torch.mean(x, dim=axes, keepdim=...)`). -/
def reduceMeanDimsDyn {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape}
    (axes : List Nat)
    (x : RefTy (m := m) (α := α) s)
    (keepdim : Bool := false) :
    m (Option (Σ s' : Shape, RefTy (m := m) (α := α) s')) :=
  Internal.reduceDimsDynCore (α := α) (m := m) Internal.reduceAlongLastMean (s := s) axes keepdim x

/--
Dynamic slice on an arbitrary axis.

This lowers `slice_range_axisDyn axis start len` to:
1. permute `axis` to the front,
2. call the axis-0 slice primitive, then
3. permute back.
-/
def sliceRangeAxisDyn {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape}
    (axis start len : Nat)
    (x : RefTy (m := m) (α := α) s) :
    m (Option (Σ s' : Shape, RefTy (m := m) (α := α) s')) := do
  let r := Shape.rank s
  if axis ≥ r then
    return none
  let swapsToFront := moveAxisToFrontSwaps axis
  let swapsBack := List.range axis
  let xFront ← Einsum.permuteBySwaps (α := α) (m := m) ⟨s, x⟩ swapsToFront
  match xFront with
  | ⟨.scalar, _⟩ => pure none
  | ⟨.dim nDim rest, x0⟩ =>
      if h : len + start ≤ nDim then
        let y0 ← _root_.Runtime.Autograd.Torch.sliceRange0 (m := m) (α := α)
          (nDim := nDim) (s := rest) start len h x0
        let yFront : Σ s' : Shape, RefTy (m := m) (α := α) s' := ⟨.dim len rest, y0⟩
        let y ← Einsum.permuteBySwaps (α := α) (m := m) yFront swapsBack
        pure (some y)
      else
        pure none

/-- Dynamic `softmax` over an arbitrary axis (implemented by permuting to last, applying softmax,
  permuting back). -/
def softmaxDimDyn {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape}
    (axis : Nat)
    (x : RefTy (m := m) (α := α) s) :
    m (Option (RefTy (m := m) (α := α) s)) := do
  let r := Shape.rank s
  if axis ≥ r then
    return none
  let swaps := moveAxisToLastSwaps r axis
  let xMoved ← Einsum.permuteBySwaps (α := α) (m := m) ⟨s, x⟩ swaps
  let yMoved ← _root_.Runtime.Autograd.Torch.softmax (m := m) (α := α) (s := xMoved.fst) xMoved.snd
  let yBack ← Einsum.permuteBySwaps (α := α) (m := m) ⟨xMoved.fst, yMoved⟩ swaps.reverse
  if h : yBack.fst = s then
    pure (some (h ▸ yBack.snd))
  else
    pure none

/-- Dynamic `log_softmax` over an arbitrary axis (with optional epsilon for numerical stability). -/
def logSoftmaxDimDyn {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape}
    (axis : Nat)
    (x : RefTy (m := m) (α := α) s)
    (ε : α := Numbers.epsilon) :
    m (Option (RefTy (m := m) (α := α) s)) := do
  let r := Shape.rank s
  if axis ≥ r then
    return none
  let swaps := moveAxisToLastSwaps r axis
  let xMoved ← Einsum.permuteBySwaps (α := α) (m := m) ⟨s, x⟩ swaps
  let yMoved ← _root_.Runtime.Autograd.Torch.logSoftmax (m := m) (α := α) (s := xMoved.fst)
    xMoved.snd (ε := ε)
  let yBack ← Einsum.permuteBySwaps (α := α) (m := m) ⟨xMoved.fst, yMoved⟩ swaps.reverse
  if h : yBack.fst = s then
    pure (some (h ▸ yBack.snd))
  else
    pure none

/-- Helper: appending a trailing `1` dimension does not change `Shape.size`. -/
private theorem size_ofList_append_one (ds : List Nat) :
    Shape.size (Shape.ofList (ds ++ [1])) = Shape.size (Shape.ofList ds) := by
  induction ds with
  | nil => simp [Shape.ofList, Shape.size]
  | cons d ds ih =>
      simp [Shape.ofList, Shape.size, ih]

/--
Dynamic `unsqueeze`: insert a singleton dimension at `axis`.

PyTorch analogue: `torch.unsqueeze(x, dim=axis)`.
-/
def unsqueezeDyn {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape}
    (axis : Nat)
    (x : RefTy (m := m) (α := α) s) :
    m (Option (Σ s' : Shape, RefTy (m := m) (α := α) s')) := do
  let r := Shape.rank s
  if axis > r then
    return none
  let sApp : Shape := Shape.appendDim s 1
  have hSz : Shape.size s = Shape.size sApp := by
    simpa [sApp] using (Eq.symm (size_appendDim_one' s))
  let xApp ← reshape (m := m) (α := α) (s₁ := s) (s₂ := sApp) x hSz
  let swaps :=
    (List.range (r - axis)).map (fun i => (r - 1) - i)
  let out ← Einsum.permuteBySwaps (α := α) (m := m) ⟨sApp, xApp⟩ swaps
  pure (some out)

/--
Dynamic `squeeze` along a specific axis, requiring that axis to have size 1.

PyTorch analogue: `torch.squeeze(x, dim=axis)` (the `dim`-restricted variant).
-/
def squeezeDyn {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape}
    (axis : Nat)
    (x : RefTy (m := m) (α := α) s) :
    m (Option (Σ s' : Shape, RefTy (m := m) (α := α) s')) := do
  let r := Shape.rank s
  if axis ≥ r then
    return none
  let swaps := moveAxisToLastSwaps r axis
  let xMoved ← Einsum.permuteBySwaps (α := α) (m := m) ⟨s, x⟩ swaps
  let dims := Shape.toList xMoved.fst
  match hrev : dims.reverse with
  | [] => pure none
  | dLast :: revRest =>
      if hLast : dLast = 1 then
        let dims' := revRest.reverse
        let sDropped : Shape := Shape.ofList dims'
        have hx : Shape.ofList dims = xMoved.fst := by
          simp [dims]
        have hdims : dims = dims' ++ [dLast] := by
          calc
            dims = (dims.reverse).reverse := by
              simp
            _ = (dLast :: revRest).reverse := by simp [hrev]
            _ = revRest.reverse ++ [dLast] := by simp
            _ = dims' ++ [dLast] := by rfl
        have hSz : Shape.size xMoved.fst = Shape.size sDropped := by
          calc
            Shape.size xMoved.fst = Shape.size (Shape.ofList dims) := by simp [hx]
            _ = Shape.size (Shape.ofList (dims' ++ [dLast])) := by simp [hdims]
            _ = Shape.size (Shape.ofList dims') := by
              -- `dLast = 1` in this branch.
              simpa [hLast] using (size_ofList_append_one dims')
            _ = Shape.size sDropped := by rfl
        let xDropped ← reshape (m := m) (α := α) (s₁ := xMoved.fst) (s₂ := sDropped) xMoved.snd hSz
        -- Note: we *do not* permute back. After moving `axis` to the last position and then
        -- deleting it, the remaining axes are already in the correct order.
        pure (some ⟨sDropped, xDropped⟩)
      else
        pure none

/--
Dynamic concatenation of two tensors along `axis` (existential output shape).

This is the binary helper used by `cat_axisDyn`. It lowers to `concat_dim0` by moving the
requested axis to the front.
-/
def catAxis2Dyn {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    (axis : Nat)
    (x y : Σ s : Shape, RefTy (m := m) (α := α) s) :
    m (Option (Σ s' : Shape, RefTy (m := m) (α := α) s')) := do
  let r := Shape.rank x.fst
  if r = 0 then
    return none
  if axis ≥ r then
    return none
  let swapsToFront := moveAxisToFrontSwaps axis
  let swapsBack := List.range axis
  let xFront ← Einsum.permuteBySwaps (α := α) (m := m) x swapsToFront
  let yFront ← Einsum.permuteBySwaps (α := α) (m := m) y swapsToFront
  match xFront, yFront with
  | ⟨.dim nDim restX, xRef⟩, ⟨.dim mDim restY, yRef⟩ =>
      if hRest : restX = restY then
        match hRest with
        | rfl =>
            let zFront ← _root_.Runtime.Autograd.Torch.concatDim0 (m := m) (α := α)
              (nDim := nDim) (mDim := mDim) (s := restX) xRef yRef
            let outFront : Σ s' : Shape, RefTy (m := m) (α := α) s' := ⟨.dim (nDim + mDim) restX,
              zFront⟩
            let out ← Einsum.permuteBySwaps (α := α) (m := m) outFront swapsBack
            pure (some out)
      else
        pure none
  | _, _ => pure none

/-- Dynamic concatenation of a list of tensors along `axis` (folding `cat_axis2Dyn`). -/
def catAxisDyn {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    (axis : Nat)
    (xs : List (Σ s : Shape, RefTy (m := m) (α := α) s)) :
    m (Option (Σ s' : Shape, RefTy (m := m) (α := α) s')) := do
  match xs with
  | [] => pure none
  | x0 :: rest =>
      let mut cur := x0
      for x in rest do
        let some cur' ← catAxis2Dyn (α := α) (m := m) axis cur x | return none
        cur := cur'
      pure (some cur)

/--
Dynamic `stack` along a new axis.

PyTorch analogue: `torch.stack(xs, dim=axis)`.

Implementation: `unsqueeze` each input at `axis`, then `cat` along the same `axis`.
-/
def stackAxisDyn {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    (axis : Nat)
    (xs : List (Σ s : Shape, RefTy (m := m) (α := α) s)) :
    m (Option (Σ s' : Shape, RefTy (m := m) (α := α) s')) := do
  match xs with
  | [] => pure none
  | x0 :: rest =>
      -- Require all inputs have the same shape (PyTorch requirement).
      if !(rest.all (fun x => x.fst = x0.fst)) then
        return none
      let mut ys : List (Σ s : Shape, RefTy (m := m) (α := α) s) := []
      for x in xs do
        let some y ← unsqueezeDyn (α := α) (m := m) (s := x.fst) axis x.snd | return none
        ys := ys.concat y
      catAxisDyn (α := α) (m := m) axis ys

/--
Dynamic `split` along an axis with explicit split sizes.

PyTorch analogue: `torch.split(x, split_sizes, dim=axis)`.
-/
def splitAxisDyn {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape}
    (axis : Nat)
    (splitSizes : List Nat)
    (x : RefTy (m := m) (α := α) s) :
    m (Option (List (Σ s' : Shape, RefTy (m := m) (α := α) s'))) := do
  let r := Shape.rank s
  if r = 0 then
    return none
  if axis ≥ r then
    return none
  let swapsToFront := moveAxisToFrontSwaps axis
  let xFront ← Einsum.permuteBySwaps (α := α) (m := m) ⟨s, x⟩ swapsToFront
  match xFront.fst with
  | .scalar => pure none
  | .dim nDim _ =>
      if splitSizes.foldl (fun acc k => acc + k) 0 != nDim then
        return none
      let mut start : Nat := 0
      let mut outs : List (Σ s' : Shape, RefTy (m := m) (α := α) s') := []
      for len in splitSizes do
        let some y ← sliceRangeAxisDyn (α := α) (m := m) (s := s) axis start len x | return none
        outs := outs.concat y
        start := start + len
      pure (some outs)

/--
Dynamic `chunk` along an axis, given a desired chunk size.

PyTorch analogue: `torch.split(x, chunkSize, dim=axis)` or `torch.chunk` (size-based variant).
-/
def chunkAxisDyn {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape}
    (axis : Nat)
    (chunkSize : Nat)
    (x : RefTy (m := m) (α := α) s) :
    m (Option (List (Σ s' : Shape, RefTy (m := m) (α := α) s'))) := do
  if chunkSize = 0 then
    return none
  let r := Shape.rank s
  if r = 0 then
    return none
  if axis ≥ r then
    return none
  let swapsToFront := moveAxisToFrontSwaps axis
  let xFront ← Einsum.permuteBySwaps (α := α) (m := m) ⟨s, x⟩ swapsToFront
  match xFront.fst with
  | .scalar => pure none
  | .dim nDim _ =>
      -- Ceiling division to compute number of chunks.
      let nChunks : Nat := (nDim + chunkSize - 1) / chunkSize
      let sizes : List Nat :=
        (List.range nChunks).map (fun i =>
          if (i + 1) * chunkSize ≤ nDim then
            chunkSize
          else
            nDim - i * chunkSize)
      splitAxisDyn (α := α) (m := m) (s := s) axis sizes x

/-- NCHW → NHWC for 4D tensors, implemented via two adjacent swaps. -/
def nchwToNhwc {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {n c h w : Nat}
    (x : RefTy (m := m) (α := α) (.dim n (.dim c (.dim h (.dim w .scalar))))) :
    m (RefTy (m := m) (α := α) (.dim n (.dim h (.dim w (.dim c .scalar))))) := do
  let x1 : RefTy (m := m) (α := α) (.dim n (.dim h (.dim c (.dim w .scalar)))) ←
    swapAdjacentAtDepth (m := m) (α := α) (s := (.dim n (.dim c (.dim h (.dim w .scalar))))) 1 x
  swapAdjacentAtDepth (m := m) (α := α) (s := (.dim n (.dim h (.dim c (.dim w .scalar))))) 2 x1

/-- NHWC → NCHW for 4D tensors, implemented via two adjacent swaps. -/
def nhwcToNchw {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {n h w c : Nat}
    (x : RefTy (m := m) (α := α) (.dim n (.dim h (.dim w (.dim c .scalar))))) :
    m (RefTy (m := m) (α := α) (.dim n (.dim c (.dim h (.dim w .scalar))))) := do
  let x1 : RefTy (m := m) (α := α) (.dim n (.dim h (.dim c (.dim w .scalar)))) ←
    swapAdjacentAtDepth (m := m) (α := α) (s := (.dim n (.dim h (.dim w (.dim c .scalar))))) 2 x
  swapAdjacentAtDepth (m := m) (α := α) (s := (.dim n (.dim h (.dim c (.dim w .scalar))))) 1 x1

end F
end TorchLean
end Autograd
end Runtime
