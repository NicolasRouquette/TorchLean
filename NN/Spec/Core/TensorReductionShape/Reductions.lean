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

/-- Index of the first maximal entry in a nonempty vector.

The explicit nonemptiness hypothesis keeps the result total without inventing an index for an
empty vector. Ties retain the smaller index, matching a left-to-right `argmax` traversal. -/
def argmaxVector {n : Nat} (h : 0 < n) (x : Tensor α (.dim n .scalar)) : Fin n :=
  let value (i : Fin n) :=
    match get x i with
    | .scalar a => a
  let rec loop (i : Nat) (best : Fin n) : Fin n :=
    if hi : i < n then
      let current : Fin n := ⟨i, hi⟩
      loop (i + 1) (if value current > value best then current else best)
    else
      best
  termination_by n - i
  decreasing_by
    simpa using Nat.sub_succ_lt_self (a := n) (i := i) hi
  loop 1 ⟨0, h⟩

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

/-- Shape obtained by replacing the selected axis with a singleton dimension. -/
def shapeAfterSumKeepDim : Shape → Nat → Shape
  | .scalar, _ => .scalar
  | .dim _ inner, 0 => .dim 1 inner
  | .dim n inner, Nat.succ k => .dim n (shapeAfterSumKeepDim inner k)

@[simp] theorem shape_after_sum_keep_dim_size (s : Shape) (axis : Nat) :
    Shape.size (shapeAfterSumKeepDim s axis) = Shape.size (shapeAfterSum s axis) := by
  induction s generalizing axis with
  | scalar => simp [shapeAfterSumKeepDim, shapeAfterSum]
  | dim n inner ih =>
      cases axis with
      | zero => simp [shapeAfterSumKeepDim, shapeAfterSum, Shape.size]
      | succ axis => simp [shapeAfterSumKeepDim, shapeAfterSum, Shape.size, ih]

@[simp] theorem shape_after_sum_keep_dim_rank (s : Shape) (axis : Nat) :
    Shape.rank (shapeAfterSumKeepDim s axis) = Shape.rank s := by
  induction s generalizing axis with
  | scalar => simp [shapeAfterSumKeepDim, Shape.rank]
  | dim n inner ih =>
      cases axis with
      | zero => simp [shapeAfterSumKeepDim, Shape.rank]
      | succ axis => simp [shapeAfterSumKeepDim, Shape.rank, ih]

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

/-- Canonical broadcast from a keep-dimension reduction shape back to its input shape. -/
def shapeAfterSumKeepDimBroadcast : (s : Shape) → (axis : Nat) →
    Shape.CanBroadcastTo (shapeAfterSumKeepDim s axis) s
  | .scalar, _ => .scalar
  | .dim _ inner, 0 => .dim_1_to_n (Shape.CanBroadcastTo.refl inner)
  | .dim _ inner, Nat.succ axis =>
      letI : Shape.SameRank (shapeAfterSumKeepDim inner axis) inner :=
        ⟨shape_after_sum_keep_dim_rank inner axis⟩
      .dim_eq (shapeAfterSumKeepDimBroadcast inner axis)

/-- Reinsert an axis dropped by `shapeAfterSum`, repeating the reduced tensor along that axis.

This is deliberately separate from `broadcastTo`. Generic broadcasting aligns dimensions from the
right, whereas a reduction backward pass must restore the exact axis that was removed. -/
def broadcastAfterSum {α : Type} [Inhabited α] :
    (s : Shape) → (axis : Nat) → Tensor α (shapeAfterSum s axis) → Tensor α s
  | .scalar, _, x => x
  | .dim n _, 0, x => Tensor.dim (fun _ : Fin n => x)
  | .dim _ inner, Nat.succ axis, Tensor.dim xs =>
      Tensor.dim (fun i => broadcastAfterSum inner axis (xs i))

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
| .scalar, .scalar, Shape.CanBroadcastTo.scalar, t => t
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
  (_h : Shape.NonemptyAxis axis s) : Tensor α (shapeAfterSum s axis) :=
  -- Recurse until the selected axis becomes the outer axis. Returning `shapeAfterSum`
  -- directly keeps the shape change visible to Lean and avoids proof-carrying casts.
  let rec aux {inShape : Shape} (axisAdjusted : Nat) (t : Tensor α inShape) :
      Tensor α (shapeAfterSum inShape axisAdjusted) :=
    match inShape, axisAdjusted, t with
    | .scalar, _, t => t
    | .dim _ _, 0, t => reduceFirstDim f t
    | .dim _ _, Nat.succ k, .dim values =>
        Tensor.dim (fun i => aux k (values i))
  aux axis x

/-- Sum-reduction along a given axis. -/
def reduceSum {α : Type} [Add α] [Zero α] {s : Shape} (axis : Nat) (t : Tensor α s) (h :
  Shape.NonemptyAxis axis s) :
    Tensor α (shapeAfterSum s axis) :=
  reduceDim sumSpec axis t h

/-- Sum reduction on the leading axis is the corresponding first-dimension fold. -/
theorem reduceSum_zero_eq_reduceFirstDim {α : Type} [Add α] [Zero α]
    {n : Nat} {s : Shape} (h : 0 < n) (tensor : Tensor α (.dim n s)) :
    reduceSum 0 tensor (Shape.hasNonemptyAxisZeroOfPos h).proof =
      reduceFirstDim (fun {sliceShape} => sumSpec (s := sliceShape)) tensor := by
  rfl

/-- Product-reduction along a given axis. -/
def reduceProd {s : Shape} (axis : Nat) (t : Tensor α s) (h : Shape.NonemptyAxis axis s) :
    Tensor α (shapeAfterSum s axis) :=
  reduceDim prodSpec axis t h

/-- Mean-reduction along a given axis. -/
def reduceMean {s : Shape} (axis : Nat) (t : Tensor α s) (h : Shape.NonemptyAxis axis s) :
  Tensor α (shapeAfterSum s axis) :=
  let summed := reduceSum axis t h
  letI : Shape.AxisInBounds axis s := h.toAxisInBounds
  mapSpec (fun x => x / (Shape.axisSize s axis : α)) summed

-- Reduce sum of squares (for variance)
/-- Sum of squares reduced along an axis (helper for variance). -/
def reduceSumSquared {n s} (axis : Nat) (t : Tensor α (.dim n s)) (h : Shape.NonemptyAxis axis
  (.dim n s)) :
    Tensor α (shapeAfterSum (.dim n s) axis) :=
  reduceSum axis (mapSpec (fun x => x * x) t) h

/-- Variance-reduction along a given axis (population variance, divides by `n`).

The reduced axis is centered first and squared second. This two-pass arrangement avoids the
catastrophic cancellation of $\mathbb{E}[X^2]-\mathbb{E}[X]^2$ when values are large but tightly
clustered.
-/
def reduceVar
  {s : Shape} (axis : Nat) (t : Tensor α s) (h : Shape.NonemptyAxis axis s) :
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
        let inner_reducible : Shape.NonemptyAxis k inner := by
          -- We know h : Shape.NonemptyAxis (k + 1) (Shape.dim n inner)
          -- This means NonemptyAxis.succ (NonemptyAxis k inner)
          -- So we can extract the inner proof
          cases h with
          | succ inner_h => exact inner_h

        -- For each slice along the first dimension, compute variance along axis k
        let variance_slices : Fin n → Tensor α (shapeAfterSum inner k) :=
          fun i => reduceVar k (f i) inner_reducible
        Tensor.dim variance_slices

/-- Min-reduction along a given axis. -/
def reduceMin {s : Shape}
  (axis : Nat) (t : Tensor α s) (h : Shape.NonemptyAxis axis s) :
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
        let inner_reducible : Shape.NonemptyAxis k inner := by
          cases h with
          | succ inner_h => exact inner_h

        -- For each slice along the first dimension, compute min along axis k
        let min_slices : Fin n → Tensor α (shapeAfterSum inner k) :=
          fun i => reduceMin k (f i) inner_reducible
        Tensor.dim min_slices

/-- Max-reduction along a given axis. -/
def reduceMax {s : Shape}
  (axis : Nat) (t : Tensor α s) (h : Shape.NonemptyAxis axis s) :
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
        let inner_reducible : Shape.NonemptyAxis k inner := by
          cases h with
          | succ inner_h => exact inner_h
        let max_slices : Fin n → Tensor α (shapeAfterSum inner k) :=
          fun i => reduceMax k (f i) inner_reducible
        Tensor.dim max_slices

/-- Mean-reduce an arbitrary-rank tensor along its innermost axis. -/
def reduceMeanLast {s : Shape} (x : Tensor α s)
    (h : Shape.HasNonemptyAxis (Spec.Shape.rank s - 1) s) :
    Tensor α (shapeAfterSum s (Spec.Shape.rank s - 1)) :=
  reduceMean (Spec.Shape.rank s - 1) x h.proof

/-- Variance-reduce an arbitrary-rank tensor along its innermost axis. -/
def reduceVarLast {s : Shape} (x : Tensor α s)
    (h : Shape.HasNonemptyAxis (Spec.Shape.rank s - 1) s) :
    Tensor α (shapeAfterSum s (Spec.Shape.rank s - 1)) :=
  reduceVar (Spec.Shape.rank s - 1) x h.proof

/-- Sum-reduce an arbitrary-rank tensor along its innermost axis. -/
def reduceSumLast {s : Shape} (x : Tensor α s)
    [h : Shape.HasNonemptyAxis (Spec.Shape.rank s - 1) s] :
    Tensor α (shapeAfterSum s (Spec.Shape.rank s - 1)) :=
  reduceSum (Spec.Shape.rank s - 1) x h.proof

-- Transpose operations live in the linear-algebra extension modules.
end Tensor
end Spec
