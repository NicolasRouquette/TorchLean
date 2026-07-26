/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape.Broadcasting

@[expose] public section


namespace Spec
namespace Tensor

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-!
# Reductions

Fold, sum/product/mean/variance, axis reductions, and last-axis reductions.
-/

/-- Left fold over all tensor elements. -/
def tensorFoldlSpec {α β : Type} (f : β → α → β) (init : β) : ∀ {s : Shape}, Tensor α s → β
  | Shape.scalar, Tensor.scalar value => f init value
  | Shape.dim n s, Tensor.dim values =>
    let rec go (i : Nat) (acc : β) : β :=
      if h : i < n then
        go (i + 1) (tensorFoldlSpec f acc (values ⟨i, h⟩))
      else acc
    go 0 init

/-- Right fold over all tensor elements. -/
def tensorFoldrSpec {α β : Type} (f : α → β → β) (init : β) : ∀ {s : Shape}, Tensor α s → β
  | Shape.scalar, Tensor.scalar value => f value init
  | Shape.dim n s, Tensor.dim values =>
    let rec go (i : Nat) (acc : β) : β :=
      if h : i < n then
        if h_last : (n - 1 - i) < n then
          let idx := ⟨n - 1 - i, h_last⟩
          go (i + 1) (tensorFoldrSpec f acc (values idx))
        else acc
      else acc
    go 0 init

-- Reductions that collapse a tensor to scalar values.
/-- Sum all elements of a tensor. -/
def sumSpec {α : Type} [Add α] [Zero α] {s : Shape} (t : Tensor α s) : α :=
  tensorFoldlSpec (· + ·) 0 t

/-- Product of all elements of a tensor. -/
def prodSpec {s : Shape} (t : Tensor α s) : α :=
  tensorFoldlSpec (· * ·) 1 t

/-- Short name for `prodSpec`. -/
abbrev productSpec {s : Shape} (t : Tensor α s) : α :=
  prodSpec t

/-- Count the number of scalar entries in a tensor (= `Spec.Shape.size`). -/
def countSpec {s : Shape} (t : Tensor α s) : Nat :=
  tensorFoldlSpec (fun acc _ => acc + 1) 0 t

/-- `true` if any entry satisfies `p`. -/
def anySpec {s : Shape} (p : α → Bool) (t : Tensor α s) : Bool :=
  tensorFoldlSpec (fun acc x => acc || p x) false t

/-- `true` if all entries satisfy `p`. -/
def allSpec {s : Shape} (p : α → Bool) (t : Tensor α s) : Bool :=
  tensorFoldlSpec (fun acc x => acc && p x) true t

/-- Dot product: $\sum_i a_i b_i$. -/
def dotSpec {s : Shape} (a b : Tensor α s) : α :=
  sumSpec (mulSpec a b)

-- Statistics computed over all scalar leaves of a tensor.
/-- Mean of all elements (treats nested dims as one big collection). -/
def meanSpec : ∀ {s : Shape}, Tensor α s → α
  | .scalar, Tensor.scalar value => value
  | .dim n _, Tensor.dim values =>
      let sum := (List.finRange n).foldl (fun acc i => acc + meanSpec (values i)) 0
      sum / ↑n

/-- Variance of all scalar leaves (population variance, divides by the total leaf count).

For a higher-rank tensor this centers every entry around the tensor-wide mean. In particular, it
does not collapse each outer slice to its mean before measuring dispersion.
-/
def varianceSpec : ∀ {s : Shape}, Tensor α s → α
  | .scalar, Tensor.scalar _ => 0
  | .dim _ _, Tensor.dim values =>
      let t := Tensor.dim values
      let m := meanSpec t
      let sumSqDiff := tensorFoldlSpec (fun acc x =>
        let diff := x - m
        acc + diff * diff) 0 t
      sumSqDiff / ↑(countSpec t)

-- Shape-level bookkeeping for reductions that drop one axis.
/-- Output shape after summing along `axis` (drops that dimension). -/
def shapeAfterSum : Shape → Nat → Shape
  | .scalar, _ => .scalar
  | .dim _ inner, 0 => inner
  | .dim n inner, Nat.succ k => .dim n (shapeAfterSum inner k)

/-- `simp` lemma: dropping axis 1 from a 2D `(nQ+1)×(nK+1)` shape yields `(nQ+1)`. -/
@[simp]
theorem shape_after_sum_dim_1 (nQ nK : Nat) :
  shapeAfterSum (Shape.dim (nQ + 1) (Shape.dim (nK + 1) Shape.scalar)) 1 =
    Shape.dim (nQ + 1) Shape.scalar := by
  simp [shapeAfterSum]

/-- `simp` lemma: dropping axis 1 from a 2D `nQ×nK` shape yields `nQ`. -/
@[simp]
theorem shape_after_sum_dim_1_alt (nQ nK : Nat) :
  shapeAfterSum (Shape.dim nQ (Shape.dim nK Shape.scalar)) 1 =
    Shape.dim nQ Shape.scalar := by
  simp [shapeAfterSum]

/-- `simp` lemma: dropping axis 3 from a 4D `b×h×w×c` shape yields `b×h×w`. -/
@[simp]
theorem shape_after_sum_dim_3_alt (b h w c : Nat) :
  shapeAfterSum (Shape.dim b (Shape.dim h (Shape.dim w (Shape.dim c Shape.scalar)))) 3 =
    Shape.dim b (Shape.dim h (Shape.dim w Shape.scalar)) := by
  simp [shapeAfterSum]

/-- `simp` lemma: dropping axis 0 from a positive `.dim (n+1) s` yields `s`. -/
@[simp]
theorem shape_after_sum_zero {n s} :
  shapeAfterSum (.dim (n+1) s) 0 = s := by
  simp [shapeAfterSum]

/-- `simp` lemma: dropping axis `k+1` recurses into the tail shape. -/
@[simp]
theorem shape_after_sum_succ {n s k} :
  shapeAfterSum (.dim (n+1) s) (k+1) = .dim (n+1) (shapeAfterSum s k) := by
  simp [shapeAfterSum]

/-- `simp` lemma: dropping axis 0 from a 2D `(kH+1)×(kW+1)` yields `(kW+1)`. -/
@[simp]
theorem shape_after_sum_twice_zero {kH kW : Nat} :
  shapeAfterSum (Shape.dim (kH + 1) (Shape.dim (kW + 1) Shape.scalar)) 0
    = .dim (kW + 1) Shape.scalar := by simp [shapeAfterSum]

/-- `simp` lemma: dropping axis 0 from `.dim n inner` yields `inner` (even when `n=0`). -/
@[simp]
theorem shape_after_sum_zero_alt
  (n : Nat) (inner : Shape) :
  shapeAfterSum (.dim n inner) 0 = inner := by
  simp [shapeAfterSum]

-- Helper function for reflexivity
/-- Reflexive broadcast proof (`s` can broadcast to itself). -/
def canBroadcastToRefl (s : Shape) : Shape.CanBroadcastTo s s :=
  match s with
  | .scalar => Shape.CanBroadcastTo.scalar_to_any .scalar
  | .dim _ inner => Shape.CanBroadcastTo.dim_eq (canBroadcastToRefl inner)

/-- Build a broadcast proof from the reduced shape back to the original shape.

We use this when a backward pass computes something in the reduced shape (e.g. a mean/variance) and
we need to broadcast it back to match the original tensor shape.
-/
def shapeAfterSumBroadcastBack
  {s : Shape} (dim : Nat)
  (valid : Shape.valid_axis_inst dim s)
  (wf : Shape.WellFormed s) :
  Shape.CanBroadcastTo (shapeAfterSum s dim) s :=
match s, dim with
| .scalar, _ =>
  -- For scalar, shape_after_sum returns scalar
  Shape.CanBroadcastTo.scalar_to_any .scalar
| .dim n inner, 0 =>
  -- When dim = 0, shape_after_sum (.dim n inner) 0 = inner
  -- We need CanBroadcastTo inner (.dim n inner)
  Shape.CanBroadcastTo.expand_dims (canBroadcastToRefl inner)
| .dim n inner, Nat.succ k =>
  -- When dim = k+1, shape_after_sum (.dim n inner) (k+1) = .dim n (shape_after_sum inner k)
  -- We need CanBroadcastTo (.dim n (shape_after_sum inner k)) (.dim n inner)
  let valid_inner : Shape.valid_axis_inst k inner := by
    cases valid.proof with
    | valid_succ h => exact ⟨h⟩

  let inner_wf : Shape.WellFormed inner := ⟨wf.proof.right⟩

  Shape.CanBroadcastTo.dim_eq (shapeAfterSumBroadcastBack k valid_inner inner_wf)

-- The compact proof below uses the product-shape lemmas already established above.


-- Reducers parameterized by the scalar aggregation operation.

/-- Reduce a tensor of shape `(n, innerShape)` by applying `f` across the first axis.

This is the basic “reduce over axis 0” primitive that we reuse to implement broadcast-adjoints and
multi-axis reducers.
-/
def reduceFirstDim {α : Type} {innerShape : Shape} {n : Nat}
    (f : ∀ {sliceShape : Shape}, Tensor α sliceShape → α)
    (t : Tensor α (.dim n innerShape)) : Tensor α innerShape :=
    match innerShape with
    | .scalar =>
        match t with
        | .dim slices =>
            let collected := .dim (fun i => slices i)
            .scalar (f collected)
    | .dim _ _ =>
        match t with
        | .dim slices =>
            .dim (fun j =>
              let slice_at_j := .dim (fun i => sliceSpec (slices i) j)
              reduceFirstDim f slice_at_j)

/-!
Reduce a gradient from a broadcast target shape back to the original input shape.

This is the adjoint of `broadcastTo` for sum-reduction: broadcast duplicates values, so the
backward pass sums contributions across broadcasted dimensions.

PyTorch analogy: this is the logic behind "sum over broadcasted dimensions" that happens in
autograd for `expand` + elementwise ops.
-/
/-- Adjoint of `broadcastTo` under sum-reduction: collapse broadcasted dimensions by summing. -/
def reduceFromBroadcastTo {α : Type} [Add α] [Zero α] :
  {s₁ s₂ : Shape} → Shape.CanBroadcastTo s₁ s₂ → Tensor α s₂ → Tensor α s₁
| .scalar, s₂, Shape.CanBroadcastTo.scalar_to_any .(s₂), t =>
    Tensor.scalar (sumSpec (α := α) (s := s₂) t)
| .dim n s₁, .dim .(n) s₂, Shape.CanBroadcastTo.dim_eq tail, Tensor.dim xs =>
    Tensor.dim (fun i => reduceFromBroadcastTo (s₁ := s₁) (s₂ := s₂) tail (xs i))
| .dim 1 s₁, .dim n s₂, Shape.CanBroadcastTo.dim_1_to_n tail, t =>
    match t with
    | Tensor.dim xs =>
        let summed : Tensor α s₂ :=
          reduceFirstDim (α := α) (innerShape := s₂) (n := n)
            (fun {sliceShape} => sumSpec (α := α) (s := sliceShape)) (Tensor.dim xs)
        let reduced : Tensor α s₁ := reduceFromBroadcastTo (s₁ := s₁) (s₂ := s₂) tail summed
        Tensor.dim (fun _ => reduced)
| s₁, .dim n s₂, Shape.CanBroadcastTo.expand_dims tail, t =>
    match t with
    | Tensor.dim xs =>
        let summed : Tensor α s₂ :=
          reduceFirstDim (α := α) (innerShape := s₂) (n := n)
            (fun {sliceShape} => sumSpec (α := α) (s := sliceShape)) (Tensor.dim xs)
        reduceFromBroadcastTo (s₁ := s₁) (s₂ := s₂) tail summed

/-- Generic reduction along a (provably reducible) axis.

`reduce_dim f axis x` applies `f` to the slices along `axis`, and returns a tensor whose shape is
`shape_after_sum s axis` (i.e. that axis is dropped).
-/
def reduceDim
  {α : Type}
  {s : Shape}
  (f : ∀ {sliceShape : Shape}, Tensor α sliceShape → α)
  (axis : Nat)
  (x : Tensor α s)
  (_h : Shape.reducibleAlong axis s) : Tensor α (shapeAfterSum s axis) :=

  -- Design note:
  -- We implement `reduce_dim` by recursing down the shape tree until we hit the axis,
  -- then using `reduce_first_dim` at that level. This mirrors how you would implement
  -- `torch.sum(x, dim=axis)` via indexing/slicing, but keeps everything total and
  -- shape-correct by construction.
  let rec aux
    {inShape outShape : Shape} (axisAdjusted : Nat)
    (h_eq : outShape = shapeAfterSum inShape axisAdjusted)
    (t : Tensor α inShape) : Tensor α outShape :=

    match inShape, axisAdjusted with
    | .scalar, _ =>
      cast (congrArg (Tensor α) h_eq.symm) t

    | .dim n innerIn, 0 =>
      let reduced := reduceFirstDim f t
      cast (congrArg (Tensor α) h_eq.symm) reduced

    | .dim n innerIn, Nat.succ k =>
      let innerOut := shapeAfterSum innerIn k
      let recFun : Fin n → Tensor α innerOut := fun i =>
        aux k (by rfl) (getAtSpec t i)
      Tensor.dim recFun |> cast (congrArg (Tensor α) h_eq.symm)

  aux axis (by rfl) x

/-- Sum-reduction along a given axis. -/
def reduceSum {α : Type} [Add α] [Zero α] {s : Shape} (axis : Nat) (t : Tensor α s) (h :
  Shape.reducibleAlong axis s) :
    Tensor α (shapeAfterSum s axis) :=
  reduceDim sumSpec axis t h

/-- Sum-reduction along `axis`, with axis validity inferred via `valid_axis_inst`. -/
def reduceSumAuto {α : Type} [Add α] [Zero α] {s : Shape} (axis : Nat) [h : Shape.valid_axis_inst
  axis s] (t : Tensor α s) :
  Tensor α (shapeAfterSum s axis) :=
  reduceSum axis t (Shape.proveReducibleAlong axis s h.proof)

/-- Product-reduction along a given axis. -/
def reduceProd {s : Shape} (axis : Nat) (t : Tensor α s) (h : Shape.reducibleAlong axis s) :
    Tensor α (shapeAfterSum s axis) :=
  reduceDim prodSpec axis t h

/-- Product-reduction along `axis` when you already have a `valid_axis` proof. -/
def reduceProdAuto {s : Shape} (axis : Nat) (t : Tensor α s) (h : Shape.valid_axis axis s) :
    Tensor α (shapeAfterSum s axis) :=
  reduceProd axis t (Shape.proveReducibleAlong axis s h)

/-- Get the runtime size of the `k`-th dimension (0-based), if it exists. -/
def getDimSize : Shape → Nat → Option Nat
  | .scalar, _ => none
  | .dim n _, 0 => some n
  | .dim _ inner, k+1 => getDimSize inner k

/-- Mean-reduction along a given axis. -/
def reduceMean {s : Shape} (axis : Nat) (t : Tensor α s) (h : Shape.reducibleAlong axis s) :
  Tensor α (shapeAfterSum s axis) :=
  match s with
  | .scalar => t
  | .dim n inner =>
    match axis with
    | 0 =>
      let summed := reduceSum 0 t h
      mapSpec (fun x => x / (n : α)) summed
    | Nat.succ k =>
      let summed := reduceSum (Nat.succ k) t h
      -- When reducing along an *inner* axis, divide by the size of the axis being reduced,
      -- not the size of the output shape's leading dimension.
      --
      -- Example (2D): reducing axis=1 on shape `(seqLen, embedDim)` must divide by `embedDim`,
      -- but `shape_after_sum inner 0 = scalar` has `dim_size = 1`.
      --
      -- PyTorch analogy: `torch.mean(x, dim=k)` divides by the length of that `dim`, even when
      -- you reduce an inner axis of a higher-rank tensor.
      let denomNat :=
        match getDimSize inner k with
        | some m => m
        | none => 1
      mapSpec (fun x => x / (denomNat : α)) summed

/-- Mean-reduction along `axis`, with axis validity provided as a typeclass argument. -/
def reduceMeanAuto {s : Shape} (axis : Nat) (h : Shape.valid_axis_inst axis s) (t : Tensor α s) :
  Tensor α (shapeAfterSum s axis) :=
  reduceMean axis t (Shape.proveReducibleAlong axis s h.proof)

-- Reduce sum of squares (for variance)
/-- Sum of squares reduced along an axis (helper for variance). -/
def reduceSumSquared {n s} (axis : Nat) (t : Tensor α (.dim n s)) (h : Shape.reducibleAlong axis
  (.dim n s)) :
    Tensor α (shapeAfterSum (.dim n s) axis) :=
  reduceSum axis (mapSpec (fun x => x * x) t) h

/-- Variance-reduction along a given axis (population variance, divides by `n`).

The reduced axis is centered first and squared second. This two-pass arrangement avoids the
catastrophic cancellation of $\mathbb{E}[X^2]-\mathbb{E}[X]^2$ when values are large but tightly
clustered.
-/
def reduceVar
  {s : Shape} (axis : Nat) (t : Tensor α s) (h : Shape.reducibleAlong axis s) :
  Tensor α (shapeAfterSum s axis) :=
  match s with
  | .scalar =>
    mapSpec (fun _ => 0) t
  | .dim n inner =>
    match axis with
    | 0 =>
      -- PyTorch analogy: `torch.var(x, dim=0, unbiased=False)` (population variance).
      let mean := reduceMean 0 t h
      match t with
      | Tensor.dim slices =>
        let centered := Tensor.dim (fun i => subSpec (slices i) mean)
        let squared := mapSpec (fun x => x * x) centered
        reduceMean 0 squared h

    | Nat.succ k =>
      -- Reducing along axis k+1 in the inner dimensions
      -- Apply reduce_var recursively to each slice along the first dimension
      match t with
      | Tensor.dim f =>
        -- Extract the proof that inner is reducible along axis k
        let inner_reducible : Shape.reducibleAlong k inner := by
          -- We know h : Shape.reducibleAlong (k + 1) (Shape.dim n inner)
          -- This means reducibleAlong.tail (reducibleAlong k inner)
          -- So we can extract the inner proof
          cases h with
          | tail inner_h => exact inner_h

        -- For each slice along the first dimension, compute variance along axis k
        let variance_slices : Fin n → Tensor α (shapeAfterSum inner k) :=
          fun i => reduceVar k (f i) inner_reducible
        Tensor.dim variance_slices

/-- Variance-reduction along `axis`, with axis validity provided as a typeclass argument. -/
def reduceVarAuto {s : Shape} (axis : Nat) (h : Shape.valid_axis_inst axis s) (t : Tensor α s) :
  Tensor α (shapeAfterSum s axis) :=
  reduceVar axis t (Shape.proveReducibleAlong axis s h.proof)

/-- Min-reduction along a given axis. -/
def reduceMin {s : Shape}
  (axis : Nat) (t : Tensor α s) (h : Shape.reducibleAlong axis s) :
  Tensor α (shapeAfterSum s axis) :=
  match s with
  | .scalar =>
    -- Min of a single value is the value itself
    t

  | .dim n inner =>
    match axis with
    | 0 =>
      -- Reducing along the first axis - find min across the n slices
      --
      -- PyTorch analogy: `torch.amin(x, dim=0)` (or `torch.min` along a dim).
      match n, t with
      | 0, _ => nomatch h
      | Nat.succ n', Tensor.dim f =>
        -- We have at least one element, so we can safely reduce
        let rec loop (i : Nat) (acc : Tensor α inner) (hi : i ≤ n') : Tensor α inner :=
          if h_lt : i < n' then
            let next_idx : Fin (Nat.succ n') := ⟨i + 1, Nat.succ_lt_succ h_lt⟩
            loop (i + 1) (minSpec acc (f next_idx)) (Nat.le_of_succ_le_succ (Nat.succ_le_of_lt
              (Nat.succ_lt_succ h_lt)))
          else
            acc
        -- Start with first element (index 0) and loop through the rest
        let first_idx : Fin (Nat.succ n') := ⟨0, Nat.succ_pos n'⟩
        loop 0 (f first_idx) (Nat.zero_le n')

    | Nat.succ k =>
      -- Reducing along axis k+1 in the inner dimensions
      match t with
      | Tensor.dim f =>
        -- Extract the proof that inner is reducible along axis k
        let inner_reducible : Shape.reducibleAlong k inner := by
          cases h with
          | tail inner_h => exact inner_h

        -- For each slice along the first dimension, compute min along axis k
        let min_slices : Fin n → Tensor α (shapeAfterSum inner k) :=
          fun i => reduceMin k (f i) inner_reducible
        Tensor.dim min_slices

/-- Max-reduction along a given axis. -/
def reduceMax {s : Shape}
  (axis : Nat) (t : Tensor α s) (h : Shape.reducibleAlong axis s) :
  Tensor α (shapeAfterSum s axis) :=
  match s with
  | .scalar => t
  | .dim n inner =>
    match axis with
    | 0 =>
      -- PyTorch analogy: `torch.amax(x, dim=0)`.
      match n, t with
      | 0, _ => nomatch h
      | Nat.succ n', Tensor.dim f =>
        let rec loop (i : Nat) (acc : Tensor α inner) : Tensor α inner :=
          if h_lt : i < n' then
            let next_idx : Fin (Nat.succ n') := ⟨i + 1, Nat.succ_lt_succ h_lt⟩
            loop (i + 1) (maxSpec acc (f next_idx))
          else
            acc
        let first_idx : Fin (Nat.succ n') := ⟨0, Nat.succ_pos n'⟩
        loop 0 (f first_idx)
    | Nat.succ k =>
      match t with
      | Tensor.dim f =>
        let inner_reducible : Shape.reducibleAlong k inner := by
          cases h with
          | tail inner_h => exact inner_h
        let max_slices : Fin n → Tensor α (shapeAfterSum inner k) :=
          fun i => reduceMax k (f i) inner_reducible
        Tensor.dim max_slices

/-- Max-reduction along `axis`, with axis validity inferred via `valid_axis_inst`. -/
def reduceMaxAuto {s : Shape} (axis : Nat) [h : Shape.valid_axis_inst axis s] (t : Tensor α s) :
  Tensor α (shapeAfterSum s axis) :=
  reduceMax axis t (Shape.proveReducibleAlong axis s h.proof)

/-- Reduce along the last axis of `s` (i.e. axis `rank s - 1`). -/
def reduceLastDim {α : Type} [Context α] {s : Shape}
  (f : ∀ {sliceShape : Shape}, Tensor α sliceShape → α)
  (x : Tensor α s) (h : Shape.reducibleAlong (Spec.Shape.rank s - 1) s) :
  Tensor α (shapeAfterSum s (Spec.Shape.rank s - 1)) :=
  reduceDim f (Spec.Shape.rank s - 1) x h

/-- Like `reduce_last_dim`, but infers axis validity via `valid_axis_inst`. -/
def reduceLastDimAuto {α : Type} [Context α] {s : Shape}
  (f : ∀ {sliceShape : Shape}, Tensor α sliceShape → α)
  (x : Tensor α s) [h : Shape.valid_axis_inst (Spec.Shape.rank s - 1) s] :
  Tensor α (shapeAfterSum s (Spec.Shape.rank s - 1)) :=
  reduceLastDim f x (Shape.proveReducibleAlong (Spec.Shape.rank s - 1) s h.proof)

-- Reduce mean along the last dimension of any tensor shape
/-- Mean-reduce along the last axis. -/
def reduceMeanLast {α : Type} [Context α] {s : Shape} (x : Tensor α s) (h : Shape.reducibleAlong
  (Spec.Shape.rank s - 1) s) :
  Tensor α (shapeAfterSum s (Spec.Shape.rank s - 1)) :=
  reduceDim meanSpec (Spec.Shape.rank s - 1) x h

-- Reduce sum along the last dimension of a 2D tensor (specialized version)
/-- Sum-reduce along the last axis of a 2D tensor `(seqLen, embedDim)`. -/
def reduceSumLast {seqLen embedDim : Nat} (x : Tensor α (.dim seqLen (.dim embedDim .scalar))) (h
  : Shape.reducibleAlong (Spec.Shape.rank (.dim seqLen (.dim embedDim .scalar)) - 1) (.dim seqLen (.dim
  embedDim .scalar))) :
  Tensor α (.dim seqLen .scalar) :=
  reduceLastDim sumSpec x h

/-- Product-reduce along the last axis of a 2D tensor `(seqLen, embedDim)`. -/
def reduceProdLast {seqLen embedDim : Nat} (x : Tensor α (.dim seqLen (.dim embedDim .scalar))) (h
  : Shape.reducibleAlong (Spec.Shape.rank (.dim seqLen (.dim embedDim .scalar)) - 1) (.dim seqLen (.dim
  embedDim .scalar))) :
  Tensor α (.dim seqLen .scalar) :=
  reduceLastDim prodSpec x h

/-- Max-reduce along the last axis. -/
def reduceMaxLast {s : Shape} (x : Tensor α s) (h : Shape.reducibleAlong (Spec.Shape.rank s - 1) s) :
  Tensor α (shapeAfterSum s (Spec.Shape.rank s - 1)) :=
  reduceMax (Spec.Shape.rank s - 1) x h

/-- Min-reduce along the last axis. -/
def reduceMinLast {s : Shape} (x : Tensor α s) (h : Shape.reducibleAlong (Spec.Shape.rank s - 1) s) :
  Tensor α (shapeAfterSum s (Spec.Shape.rank s - 1)) :=
  reduceMin (Spec.Shape.rank s - 1) x h

/-- Variance-reduce along the last axis (specialized to a leading batch dimension). -/
def reduceVarLast
  {n : Nat} {s : Shape}
  (x : Tensor α (.dim n s)) (h : Shape.reducibleAlong (Spec.Shape.rank (.dim n s) - 1) (.dim n s)) :
  Tensor α (shapeAfterSum (.dim n s) (Spec.Shape.rank (.dim n s) - 1)) :=
  reduceVar (Spec.Shape.rank (.dim n s) - 1) x h

/-- Variance-reduce along the last axis (with axis validity as a typeclass argument). -/
def reduceVarLastGeneral {n : Nat}  {s : Shape}
  (x : Tensor α (.dim n s))
  (h : Shape.valid_axis_inst (Spec.Shape.rank (.dim n s) - 1) (.dim n s))
  : Tensor α (shapeAfterSum (.dim n s) (Spec.Shape.rank (.dim n s) - 1)) :=
  reduceVarAuto (Spec.Shape.rank (.dim n s) - 1) h x

/-- Mean-reduce along the last axis (with axis validity as a typeclass argument). -/
def reduceMeanLastGeneral {s : Shape}
  (x : Tensor α s)
  (h : Shape.valid_axis_inst (Spec.Shape.rank s - 1) s)
  : Tensor α (shapeAfterSum s (Spec.Shape.rank s - 1)) :=
  reduceMeanAuto (Spec.Shape.rank s - 1) h x

/-- Mean-reduce along the last axis, specialized for proofs that assume well-formedness. -/
def reduceMeanLastGeneralWf {s : Shape}
  (x : Tensor α s)
  [_h_wf : Shape.WellFormed s]
  (_h_rank : Spec.Shape.rank s > 0)
  (h_valid : Shape.valid_axis_inst (Spec.Shape.rank s - 1) s)
  : Tensor α (shapeAfterSum s (Spec.Shape.rank s - 1)) :=
  reduceMean (Spec.Shape.rank s - 1) x (Shape.proveReducibleAlong (Spec.Shape.rank s - 1) s h_valid.proof)

/-- Sum-reduce along the last axis (with axis validity inferred via `valid_axis_inst`). -/
def reduceSumLastGeneral {s : Shape}
  (x : Tensor α s)
  [h : Shape.valid_axis_inst (Spec.Shape.rank s - 1) s]
  : Tensor α (shapeAfterSum s (Spec.Shape.rank s - 1)) :=
  reduceSumAuto (Spec.Shape.rank s - 1) x

-- Transpose operations live in the linear-algebra extension modules.
end Tensor
end Spec
