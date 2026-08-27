/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.Group.Defs
public import Mathlib.Algebra.Group.Pi.Basic
public import Mathlib.Algebra.Group.TransferInstance
public import Mathlib.Logic.Equiv.Defs
public import NN.Spec.Core.Shape

/-!
# Core tensor datatype (`Spec.Tensor`)

This file defines the foundational, shape-indexed tensor type used throughout TorchLean's **spec**
layer:

`Tensor α s`

## Why an inductive / functional representation?

Instead of storing a flat array plus a shape, the spec tensor is a *function from indices*:

- a scalar is `α`
- an axis of extent `n` is represented by `Fin n → ...`

This has three practical benefits:

1) **Shape safety is enforced by the type.**
2) **Proofs are natural:** you reason by recursion on the `Shape` / `Tensor` constructors.
3) **No layout commitment:** the spec layer doesn’t bake in row-major vs column-major storage.

For long executable runs, repeated functional updates can create “closure chains”. Use
`Tensor.materialize` (documented below) to rebuild a tensor into an array-backed normal form.
-/

@[expose] public section


namespace Spec

/--
Shape-indexed tensor datatype for the spec layer.

This is a *functional* representation:
- a scalar tensor contains one `α`,
- an outer axis of extent `n` is a function `Fin n → Tensor α s`.

Fixed shapes should normally be written with dimension-list syntax, as in `Tensor α [4, 2]`.
The recursive `.dim` constructor is useful inside proofs and operations whose remaining rank is not
known statically; it is not a second user-facing tensor representation.

The spec layer does not commit to a concrete memory layout.
-/
inductive Tensor (α : Type) : Shape → Type where
  /-- Construct a rank-zero tensor from one scalar value. -/
  | scalar : α → Tensor α .scalar
  /-- Construct an outer dimension from its shape-indexed entries. -/
  | dim : ∀ {n s}, (Fin n → Tensor α s) → Tensor α (.dim n s)

/-- Return the value stored in a scalar tensor. -/
def Tensor.item {α : Type} : Tensor α .scalar → α
  | .scalar value => value

/-- Extracting the value of a scalar constructor returns that value. -/
@[simp] theorem Tensor.item_scalar {α : Type} (value : α) :
    (Tensor.scalar value).item = value := rfl

/-!
## Runtime note: materialization

`Tensor α s` is a *functional* representation (`Fin n → ...`). This is excellent for proofs, but
repeated updates (for example, many SGD steps) can build deep chains of closures
(`fun i => ... (old (old (old i))) ...`). Evaluating those chains later becomes progressively more
expensive.

`Tensor.materialize` rebuilds a tensor into an array-backed normal form (at every dimension), which
keeps long-running training loops from degrading.

It is extensionally the identity (the same mathematical tensor), but it is much friendlier to the
runtime evaluator.
-/

/-- Rebuild a tensor into an array-backed normal form (performance helper). -/
def Tensor.materialize {α : Type} : ∀ {s : Shape}, Tensor α s → Tensor α s
  | .scalar, t => t
  | .dim n s', Tensor.dim f =>
      let arr : Array (Tensor α s') := Array.ofFn (fun i : Fin n => Tensor.materialize (f i))
      Tensor.dim (fun i =>
        -- `arr` has size `n`, so this index is always in-bounds.
        let hn : arr.size = n := by
          simp [arr]
        let hi : i.1 < arr.size :=
          -- Avoid `simp` on `Fin.isLt` goals; transport along `hn.symm` directly.
          Eq.ndrec (motive := fun m => i.1 < m) i.2 hn.symm
        arr[i.1]'hi)

/-- `Tensor.materialize` preserves tensor values (it is extensionally the identity). -/
@[simp]
theorem Tensor.materialize_eq {α : Type} : ∀ {s : Shape} (t : Tensor α s), Tensor.materialize t = t
  := by
  intro s t
  induction t with
  | scalar x =>
      rfl
  | @dim n s f ih =>
      -- Unfold `materialize` and show the inner index function agrees with `f`.
      simp [Tensor.materialize]
      funext i
      simpa using ih i

/-- Default tensor value for any shape (uses `Inhabited.default` at scalars). -/
def Tensor.default {α : Type} [Inhabited α] : ∀ {s : Shape}, Tensor α s
  | .scalar => scalar (@Inhabited.default α _)
  | .dim _ _ => dim (fun _ => default)

/-- Make `Tensor α s` inhabited for any shape `s`. -/
@[reducible, instance]
def Tensor.inhabited {α : Type} [Inhabited α] : ∀ {s : Shape}, Inhabited (Tensor α s)
  | _ => ⟨Tensor.default⟩

/-- Recover the (data) shape from a tensor value. -/
def shapeOf {α : Type} : ∀ {s : Shape}, Tensor α s → Shape
  | .scalar, _ => .scalar
  | .dim 0 s', _ => .dim 0 s'            -- empty dimension
  | .dim (n'+1) _, .dim f => .dim (n'+1) (shapeOf (f ⟨0, Nat.zero_lt_succ n'⟩))

/-- Rebuilding a scalar tensor from its item returns the original tensor. -/
@[simp] theorem Tensor.scalar_item {α : Type} (x : Tensor α .scalar) :
    Tensor.scalar x.item = x := by
  cases x
  rfl

/-- Scalar tensors are equal when their stored values are equal. -/
@[ext] theorem Tensor.ext_scalar {α : Type} {x y : Tensor α .scalar}
    (h : x.item = y.item) : x = y := by
  cases x
  cases y
  simpa using h

/-- Equivalence between `Tensor α .scalar` and `α` (useful to reuse algebra instances). -/
def Tensor.scalarEquiv (α : Type) : Tensor α .scalar ≃ α where
  toFun := Tensor.item
  invFun := Tensor.scalar
  left_inv := Tensor.scalar_item
  right_inv := Tensor.item_scalar

/-- An outer tensor axis is equivalent to a coordinate function into the remaining tensor shape. -/
def Tensor.dimEquiv {α : Type} (n : Nat) (s : Shape) :
    Tensor α (.dim n s) ≃ (Fin n → Tensor α s) where
  toFun
    | .dim values => values
  invFun := Tensor.dim
  left_inv tensor := by cases tensor; rfl
  right_inv _ := rfl

/-- Tensor addition is pointwise at every rank. -/
@[instance_reducible] def Tensor.addCommMonoid {α : Type} [AddCommMonoid α] :
    (s : Shape) → AddCommMonoid (Tensor α s)
  | .scalar => Equiv.addCommMonoid (Tensor.scalarEquiv α)
  | .dim n s =>
      letI : AddCommMonoid (Tensor α s) := Tensor.addCommMonoid s
      Equiv.addCommMonoid (Tensor.dimEquiv n s)

/-- Every tensor shape inherits pointwise additive structure from its scalar type. -/
instance {α : Type} [AddCommMonoid α] {s : Shape} : AddCommMonoid (Tensor α s) :=
  Tensor.addCommMonoid s

/-- Equivalence between vectors and coordinate functions `Fin n → α`. -/
def Tensor.vectorEquiv {α : Type} (n : Nat) : Tensor α [n] ≃ (Fin n → α) :=
Equiv.mk
  (fun t i => match t with
              | Tensor.dim f => Tensor.item (f i))
  (fun f => Tensor.dim (fun i => Tensor.scalar (f i)))
  (by
    intro t
    cases t
    simp)
  (by
    intro f
    funext i
    simp)

namespace Tensor

/-- Cast a tensor along an equality of shapes. -/
def castShape {α : Type} {s₁ s₂ : Shape} (t : Tensor α s₁) (h : s₁ = s₂) : Tensor α s₂ :=
  Eq.mp (congrArg (Tensor α) h) t

/-!
### Cast lemmas

In dependently-typed proofs (especially graph/tape correctness proofs), the same cast may arise
with different proof terms. Since equality proofs are proof-irrelevant, we provide a few small
normalization lemmas for `Tensor.cast_shape`.
-/

/-- Casting a tensor along `rfl` is the identity. -/
@[simp] theorem cast_shape_rfl {α : Type} {s : Shape} (t : Tensor α s) :
    castShape (t := t) (h := rfl) = t := by
  rfl

/-- Casting a tensor along a reflexive equality proof is the identity. -/
@[simp] theorem cast_shape_self {α : Type} {s : Shape} (t : Tensor α s) (h : s = s) :
    castShape (t := t) h = t := by
  cases h
  rfl

/-- `Tensor.cast_shape` composes associatively (cast-by-eq is just `Eq.rec`). -/
@[simp] theorem cast_shape_trans {α : Type} {s₁ s₂ s₃ : Shape}
    (t : Tensor α s₁) (h₁₂ : s₁ = s₂) (h₂₃ : s₂ = s₃) :
    castShape (t := castShape (t := t) h₁₂) h₂₃ =
      castShape (t := t) (h₁₂.trans h₂₃) := by
  cases h₁₂
  cases h₂₃
  rfl

/-- Proof-irrelevance for `Tensor.cast_shape`. -/
theorem cast_shape_proof_irrel {α : Type} {s₁ s₂ : Shape}
    (t : Tensor α s₁) {p q : s₁ = s₂} :
    castShape (t := t) p = castShape (t := t) q := by
  have : p = q := Subsingleton.elim _ _
  cases this
  rfl

/-- Rewrite `Eq.rec` (`h ▸ t`) as `Tensor.cast_shape` for uniformity in larger proofs. -/
theorem eqRec_eq_cast_shape {α : Type} {s₁ s₂ : Shape}
    (t : Tensor α s₁) (h : s₁ = s₂) :
    (h ▸ t) = castShape (t := t) h := by
  cases h
  rfl

/-- Proof-irrelevance for `Eq.rec` casts of tensors. -/
theorem eqRec_proof_irrel {α : Type} {s₁ s₂ : Shape}
    (t : Tensor α s₁) {p q : s₁ = s₂} :
    (p ▸ t) = (q ▸ t) := by
  have : p = q := Subsingleton.elim _ _
  cases this
  rfl

-- Tell `grind` about the standard cast normalization lemmas.
attribute [grind =] cast_shape_rfl cast_shape_self cast_shape_trans eqRec_eq_cast_shape

end Tensor

-- Core tensor access operations

/-! ### Indexing helpers -/

/-!
Indexing design notes:

- `get_spec` takes a runtime multi-index (`List Nat`) and returns `Option α`.
  This permissive, frontend-friendly path is useful for debugging, JSON import/export checks, and
  any place you want to *try* an index without committing to proofs.
- For proof-driven code, you usually want `Fin n` indices and the `get`/`get2` helpers.

PyTorch analogy:
- `get_spec t [i,j,k]` is like `t[i,j,k]` but returns `none` instead of throwing.
- `get t i` is like slicing the first dimension: `t[i]`.
  (TorchLean also supports Lean’s indexing syntax: `t[i]` elaborates to `Spec.get t i`.)
- `get2 A i j` is like `A[i,j]`.
  (And `A[(i, j)]` elaborates to `Spec.get2 A i j` for matrix-shaped scalar tensors.)
-/

/-- Get a scalar by a multi‑index (list of Nats). -/
def getSpec {α : Type} {s : Shape} (t : Tensor α s) : List Nat → Option α :=
  match t with
  | .scalar value =>
      fun
      | [] => some value
      | _ :: _ => none
  | .dim (n := n) values =>
      fun
      | i :: is =>
          if h : i < n then
            getSpec (values ⟨i, h⟩) is
          else
            none
      | [] => none

@[simp] lemma get_spec_scalar_nil {α : Type} (value : α) :
    getSpec (Tensor.scalar value) [] = some value := rfl

@[simp] lemma get_spec_scalar_cons {α : Type} (value : α) (i : Nat) (is : List Nat) :
    getSpec (Tensor.scalar value) (i :: is) = none := rfl

@[simp] lemma get_spec_dim_nil {α : Type} {n : Nat} {s : Shape}
    (values : Fin n → Tensor α s) :
    getSpec (Tensor.dim values) [] = none := by
  simp [getSpec]

@[simp] lemma get_spec_dim_cons {α : Type} {n : Nat} {s : Shape}
    (values : Fin n → Tensor α s) (i : Nat) (is : List Nat) :
    getSpec (Tensor.dim values) (i :: is) =
      if h : i < n then getSpec (values ⟨i, h⟩) is else none := by
  simp [getSpec]

/-- Transporting a tensor across an equal shape does not change coordinate lookup. -/
@[simp] theorem get_spec_castShape {α : Type} {shape shape' : Shape}
    (tensor : Tensor α shape) (h : shape = shape') (index : List Nat) :
    getSpec (Tensor.castShape tensor h) index = getSpec tensor index := by
  cases h
  rfl

attribute [grind =] get_spec_scalar_nil get_spec_scalar_cons get_spec_dim_nil get_spec_dim_cons

namespace Tensor

/-- Select one coordinate along an arbitrary tensor axis. The selected axis is removed. -/
def selectSpec {α : Type} :
    (axis : Nat) → {s : Shape} → (tensor : Tensor α s) →
      [_h : Shape.AxisInBounds axis s] →
      Fin (Shape.axisSize s axis) → Tensor α (s.eraseAxis axis)
  | 0, .dim _ _, .dim values, _, i => values i
  | axis + 1, .dim _ rest, .dim values, h, i =>
      .dim fun outer =>
        @selectSpec α axis rest (values outer)
          ⟨by
            have := h.proof
            simp only [Shape.rank] at this
            grind⟩
          (Fin.cast (by rfl) i)

/-- Selecting axis zero from a tensor built by `Tensor.dim` returns the chosen entry. -/
@[simp] theorem selectSpec_zero_dim {α : Type} {n : Nat} {s : Shape}
    (values : Fin n → Tensor α s) (i : Fin n) :
    selectSpec 0 (Tensor.dim values) i = values i := by
  rfl

end Tensor

/-- Select one coordinate from the outermost tensor axis. -/
def get {α : Type} {n s} (t : Tensor α (.dim n s)) (i : Fin n) : Tensor α s :=
  Tensor.selectSpec 0 t i

/-- Indexing a tensor built from an outer coordinate function returns that coordinate. -/
@[simp] theorem get_dim {α : Type} {n : Nat} {s : Shape}
    (values : Fin n → Tensor α s) (i : Fin n) :
    get (Tensor.dim values) i = values i := by
  rfl

/--
Enable Lean’s indexing syntax for spec tensors.

After this instance, you can write `t[i]` as notation for `Spec.get t i`.

We use the domain condition `True` because the index is already a `Fin n`, so it is always
in-bounds by construction.
-/
instance {α : Type} {n : Nat} {s : Shape} :
    GetElem (Tensor α (.dim n s)) (Fin n) (Tensor α s) (fun _ _ => True) where
  getElem t i _ := _root_.Spec.get t i

namespace Tensor

/-- Two tensors of the same shape are equal when every coordinate lookup agrees. -/
@[ext] theorem ext_getSpec {α : Type} {shape : Shape} {x y : Tensor α shape}
    (h : ∀ index, getSpec x index = getSpec y index) : x = y := by
  induction shape with
  | scalar =>
      cases x with
      | scalar x =>
          cases y with
          | scalar y =>
              congr
              simpa using h []
  | dim n shape ih =>
      cases x with
      | dim x =>
          cases y with
          | dim y =>
              congr
              funext i
              apply ih
              intro index
              simpa [i.isLt] using h (i.val :: index)

/-- Extract a scalar from a vector at a statically valid coordinate. -/
def getScalar {α : Type} {n : Nat} (x : Tensor α [n]) (i : Fin n) : α :=
  Tensor.item (_root_.Spec.get x i)

/-- Scalar lookup from an outer tensor constructor exposes the selected scalar entry. -/
@[simp] theorem getScalar_dim_entry {α : Type} {n : Nat}
    (values : Fin n → Tensor α .scalar) (i : Fin n) :
    getScalar (Tensor.dim values) i = (values i).item := by
  rw [getScalar, _root_.Spec.get_dim]

/-- The coordinate-function view of a vector agrees with scalar indexing. -/
@[simp] theorem vectorEquiv_apply {α : Type} {n : Nat}
    (x : Tensor α [n]) (i : Fin n) :
    Tensor.vectorEquiv n x i = getScalar x i := by
  cases x
  rfl

/-- Reading a vector constructed coordinatewise returns the selected scalar. -/
@[simp] theorem getScalar_dim {α : Type} {n : Nat} (values : Fin n → α) (i : Fin n) :
    getScalar (Tensor.dim (fun j => Tensor.scalar (values j))) i = values i := by
  rfl

/-- Two vectors are equal when all corresponding scalar entries are equal. -/
@[ext] theorem ext_vector {α : Type} {n : Nat} {x y : Tensor α [n]}
    (h : ∀ i : Fin n, getScalar x i = getScalar y i) : x = y := by
  cases x with
  | dim xs =>
      cases y with
      | dim ys =>
          congr
          funext i
          apply Tensor.ext_scalar
          simpa [getScalar, _root_.Spec.get] using h i

end Tensor

/-- Matrix element access: `get2 A i j` returns `A[i, j]` as a scalar. -/
def get2 {α : Type} {m n : ℕ}
    (A : Tensor α [m, n]) (i : Fin m) (j : Fin n) : α :=
  match get A i with
  | Tensor.dim row => match row j with
    | Tensor.scalar v => v

/-- Matrix indexing agrees with scalar indexing after selecting the requested row. -/
theorem get2_eq_getScalar_get {α : Type} {m n : Nat}
    (A : Tensor α [m, n]) (i : Fin m) (j : Fin n) :
    get2 A i j = Tensor.getScalar (get A i) j := by
  cases A with
  | dim rows =>
      cases hrow : rows i
      case dim values =>
        cases hvalue : values j
        simp [get2, Tensor.getScalar, get, hrow, hvalue]

/-- Reading a matrix constructed coordinatewise returns the selected scalar. -/
@[simp] theorem get2_dim {α : Type} {m n : Nat} (values : Fin m → Fin n → α)
    (i : Fin m) (j : Fin n) :
    get2 (Tensor.dim (fun row => Tensor.dim (fun column =>
      Tensor.scalar (values row column)))) i j = values i j := by
  rfl

/--
Enable Lean’s indexing syntax for matrix-shaped scalar tensors.

After this instance, you can write `A[(i, j)]` as notation for `Spec.get2 A i j`.
-/
instance {α : Type} {m n : Nat} :
    GetElem (Tensor α [m, n]) (Fin m × Fin n) α (fun _ _ => True) where
  getElem A ij _ := get2 A ij.1 ij.2

-- Helper to safely get a scalar from a multi-dimensional tensor
/-!
`get_at_or_zero` is a total variant of `get_spec` used in places where a default value is more
convenient than `Option`.

We keep both `get_spec` and `get_at_or_zero` because they serve different roles:
- `get_spec` is better when you want to distinguish "out of bounds" explicitly.
- `get_at_or_zero` is better for formulas that naturally treat out-of-bounds as padding.
-/
/-- Total tensor indexing: returns `0` when the index list is out of bounds. -/
def getAtOrZero {α : Type} [Zero α] {s : Shape} (t : Tensor α s) : List Nat → α :=
  match t with
  | .scalar v =>
      fun
      | [] => v
      | _ :: _ => 0
  | .dim (n := n) f =>
      fun
      | i :: is =>
          if h : i < n then
            getAtOrZero (f ⟨i, h⟩) is
          else
            0
      | [] => 0

@[simp] lemma get_at_or_zero_scalar_nil {α : Type} [Zero α] (v : α) :
    getAtOrZero (Tensor.scalar v) [] = v := rfl

@[simp] lemma get_at_or_zero_scalar_cons {α : Type} [Zero α] (v : α) (i : Nat) (is : List Nat) :
    getAtOrZero (Tensor.scalar v) (i :: is) = 0 := rfl

@[simp] lemma get_at_or_zero_dim_nil {α : Type} [Zero α] {n : Nat} {s : Shape}
    (values : Fin n → Tensor α s) :
    getAtOrZero (Tensor.dim values) [] = 0 := by
  simp [getAtOrZero]

@[simp] lemma get_at_or_zero_dim_cons {α : Type} [Zero α] {n : Nat} {s : Shape}
    (values : Fin n → Tensor α s) (i : Nat) (is : List Nat) :
    getAtOrZero (Tensor.dim values) (i :: is) =
      if h : i < n then getAtOrZero (values ⟨i, h⟩) is else 0 := by
  simp [getAtOrZero]

attribute [grind =] get_at_or_zero_scalar_nil get_at_or_zero_scalar_cons get_at_or_zero_dim_nil
  get_at_or_zero_dim_cons

/-- Cast a tensor along an equality of shapes. -/
def tensorCast {α : Type} {s : Shape} (t : Shape) (h : s = t) : Tensor α s → Tensor α t :=
  fun x => Eq.mp (congrArg (Tensor α) h) x

/-- `tensor_cast` is definitionally `Tensor.cast_shape` (a uniform cast normal form). -/
@[simp] theorem tensor_cast_eq_cast_shape {α : Type} {s t : Shape} (h : s = t) (x : Tensor α s) :
    tensorCast (α := α) (s := s) t h x = Tensor.castShape (t := x) h := by
  rfl

attribute [grind =] tensor_cast_eq_cast_shape

/-- Replicate a scalar tensor to any shape. -/
def replicate {α : Type} : ∀ {s : Shape}, Tensor α .scalar → Tensor α s
  | .scalar, t => t
  | .dim n _, t =>
    Tensor.dim (fun _ : Fin n => replicate t)

namespace Tensor

/-- Keep a contiguous range of coordinates along an arbitrary tensor axis.

The selected axis is replaced by `count`; every other axis is preserved. The range proof makes
the operation total and rules out clipping. -/
def sliceAxisRangeSpec {α : Type} :
    (axis : Nat) → {s : Shape} → (tensor : Tensor α s) →
      [_h : Shape.AxisInBounds axis s] →
      (start count : Nat) → (start + count ≤ Shape.axisSize s axis) →
      Tensor α (s.replaceAxis axis count)
  | 0, .dim _ rest, .dim values, _, start, count, hRange =>
      .dim fun i =>
        values ⟨start + i.val,
          Nat.lt_of_lt_of_le (Nat.add_lt_add_left i.isLt start) hRange⟩
  | axis + 1, .dim _ rest, .dim values, hAxis, start, count, hRange =>
      let innerAxis : Shape.AxisInBounds axis rest :=
        ⟨by
          have := hAxis.proof
          simp only [Shape.rank] at this
          grind⟩
      letI := innerAxis
      have hRange' : start + count ≤ Shape.axisSize rest axis := by
        simpa only [Shape.axisSize_succ] using hRange
      .dim fun outer =>
        @sliceAxisRangeSpec α axis rest (values outer) innerAxis start count hRange'

end Tensor

/--
Slice a contiguous range along the first axis.

This is the spec-level analogue of `t[start : start+len]` in array/tensor libraries.
-/
def sliceRangeSpec {α : Type} {n : Nat} {s : Shape}
  (t : Tensor α (.dim n s)) (start : Nat) (len : Nat) (h : start + len ≤ n) :
  Tensor α (.dim len s) :=
  Tensor.sliceAxisRangeSpec 0 t start len (by simpa using h)

/-- Construct the first valid index of `Fin n` from an explicit nonempty proof. -/
def finZero {n : Nat} (h : 0 < n) : Fin n :=
  ⟨0, h⟩

/-- Get the first outer-axis entry, returning `none` for an empty axis. -/
def getHead {α : Type} {n : Nat} {s : Shape}
    (t : Tensor α (.dim n s)) : Option (Tensor α s) :=
  if h : 0 < n then some (get t (finZero h)) else none

/-- Drop the first outer-axis entry, returning `none` for an empty axis. -/
def getTail {α : Type} {n : Nat} {s : Shape}
    (t : Tensor α (.dim n s)) : Option (Tensor α (.dim (n - 1) s)) :=
  if h : 0 < n then
    some (Tensor.sliceAxisRangeSpec 0 t 1 (n - 1) (by
      simp only [Shape.axisSize_zero]
      grind))
  else
    none

/-! ## Scalar maps and pointwise predicates -/

namespace Tensor

/-- Map a scalar function across a tensor, preserving its shape. -/
def map {α β : Type} : ∀ {s : Shape}, (α → β) → Tensor α s → Tensor β s
  | .scalar, f, Tensor.scalar x => Tensor.scalar (f x)
  | .dim _ _, f, Tensor.dim values => Tensor.dim fun i => map f (values i)

/-- Mapping a function over a scalar tensor applies it to the stored value. -/
@[simp] theorem map_scalar {α β : Type} (f : α → β) (x : α) :
    map f (Tensor.scalar x) = Tensor.scalar (f x) :=
  rfl

/-- Mapping a function over a tensor dimension acts pointwise on its entries. -/
@[simp] theorem map_dim {α β : Type} {n : Nat} {s : Shape} (f : α → β)
    (values : Fin n → Tensor α s) :
    map f (Tensor.dim values) = Tensor.dim (fun i => map f (values i)) :=
  rfl

/-- Scalar lookup commutes with mapping over a vector. -/
@[simp] theorem getScalar_map {α β : Type} {n : Nat} (f : α → β)
    (x : Tensor α [n]) (i : Fin n) :
    (map f x).getScalar i = f (x.getScalar i) := by
  cases x with
  | dim values =>
      change Tensor.item (map f (values i)) = f (Tensor.item (values i))
      cases values i
      rfl

/-- `Forall p x` means that every scalar entry of `x` satisfies `p`. -/
def Forall {α : Type} (p : α → Prop) : {s : Shape} → Tensor α s → Prop
  | .scalar, Tensor.scalar x => p x
  | .dim n s, Tensor.dim values => ∀ i : Fin n, Forall p (s := s) (values i)

/-- `Forall₂ r x y` means that corresponding entries of `x` and `y` satisfy `r`. -/
def Forall₂ {α β : Type} (r : α → β → Prop) :
    {s : Shape} → Tensor α s → Tensor β s → Prop
  | .scalar, Tensor.scalar x, Tensor.scalar y => r x y
  | .dim n s, Tensor.dim xs, Tensor.dim ys =>
      ∀ i : Fin n, Forall₂ r (s := s) (xs i) (ys i)

@[simp] theorem forall_scalar {α : Type} {p : α → Prop} {x : α} :
    Forall p (Tensor.scalar x) ↔ p x := Iff.rfl

@[simp] theorem forall_dim {α : Type} {p : α → Prop} {n : Nat} {s : Shape}
    {values : Fin n → Tensor α s} :
    Forall p (Tensor.dim values) ↔ ∀ i, Forall p (values i) := Iff.rfl

@[simp] theorem forall₂_scalar {α β : Type} {r : α → β → Prop} {x : α} {y : β} :
    Forall₂ r (Tensor.scalar x) (Tensor.scalar y) ↔ r x y := Iff.rfl

@[simp] theorem forall₂_dim {α β : Type} {r : α → β → Prop} {n : Nat} {s : Shape}
    {xs : Fin n → Tensor α s} {ys : Fin n → Tensor β s} :
    Forall₂ r (Tensor.dim xs) (Tensor.dim ys) ↔ ∀ i, Forall₂ r (xs i) (ys i) := Iff.rfl

/-- The predicate that is always true holds at every tensor coordinate. -/
theorem forall_true {α : Type} {s : Shape} (x : Tensor α s) :
    Forall (fun _ => True) x := by
  induction s with
  | scalar =>
      cases x
      trivial
  | dim _ _ ih =>
      cases x with
      | dim values =>
          intro i
          exact ih (values i)

/-- Transport a pointwise property through a scalar map. -/
theorem forall_map {α β : Type} {p : α → Prop} {q : β → Prop}
    {f : α → β} {s : Shape} {x : Tensor α s}
    (hx : Forall p x) (hf : ∀ a, p a → q (f a)) :
    Forall q (map f x) := by
  induction s with
  | scalar =>
      cases x with
      | scalar a => exact hf a (by simpa using hx)
  | dim _ _ ih =>
      cases x with
      | dim values =>
          intro i
          exact ih (hx i)

/-- Replicating a scalar satisfying `p` produces a tensor satisfying `p` everywhere. -/
theorem forall_replicate {α : Type} {p : α → Prop} {s : Shape} {x : α}
    (hx : p x) : Forall p (Spec.replicate (s := s) (Tensor.scalar x)) := by
  induction s with
  | scalar => exact hx
  | dim _ _ ih =>
      intro _
      exact ih

end Tensor

/-! ## Conversion to host values -/

namespace Tensor

/-- Flatten a tensor to a row-major list. -/
def toList {α : Type} : ∀ {s : Shape}, Tensor α s → List α
  | .scalar, Tensor.scalar x => [x]
  | .dim n _, Tensor.dim values =>
      (List.finRange n).flatMap fun i => toList (values i)

/-- Flattening preserves the number of scalar entries recorded by the shape. -/
@[simp] theorem toList_length {α : Type} {s : Shape} (x : Tensor α s) :
    x.toList.length = s.size := by
  induction s with
  | scalar =>
      cases x
      rfl
  | dim n rest ih =>
      cases x with
      | dim values =>
          simp [toList, Shape.size, ih]

/-- Flatten a tensor directly to a row-major array at a dynamic storage boundary. -/
def toArray {α : Type} : ∀ {s : Shape}, Tensor α s → Array α
  | .scalar, Tensor.scalar x => #[x]
  | .dim n _, Tensor.dim values =>
      (Array.finRange n).foldl (fun result i => result ++ toArray (values i)) #[]

/-- The array and list views traverse tensor entries in the same row-major order. -/
@[simp] theorem toArray_toList {α : Type} : ∀ {s : Shape} (x : Tensor α s),
    x.toArray.toList = x.toList := by
  intro s x
  induction s with
  | scalar => cases x; rfl
  | dim n rest ih =>
      cases x with
      | dim values =>
        simp only [toArray, toList]
        rw [← Array.foldl_toList]
        rw [Array.foldl_toList_eq_flatMap (G := fun i => (values i).toList)]
        · simp [Array.finRange, List.finRange]
        · intro acc i
          simp [ih]

/-- Multiply the entries of a vector. -/
def prod {α : Type} [Mul α] [One α] {n : Nat} (x : Tensor α [n]) : α :=
  x.toArray.foldl (· * ·) 1

/-- The tensor product reduction agrees with the row-major array view. -/
@[simp] theorem prod_eq_toArray_foldl {α : Type} [Mul α] [One α] {n : Nat}
    (x : Tensor α [n]) :
    x.prod = x.toArray.foldl (· * ·) 1 := by
  rfl

/-- Tensor multiplication agrees with multiplication over the structural shape view. -/
theorem prod_eq_toList_prod {α : Type} [Monoid α] {n : Nat}
    (x : Tensor α [n]) :
    x.prod = x.toList.prod := by
  have foldl_mul (values : List α) (acc : α) :
      values.foldl (· * ·) acc = acc * values.prod := by
    induction values generalizing acc with
    | nil => simp
    | cons value values ih => simp [ih, mul_assoc]
  rw [prod_eq_toArray_foldl, ← Array.foldl_toList, toArray_toList, foldl_mul, one_mul]

/-- Render a tensor through its row-major scalar array. -/
instance {α : Type} {s : Shape} [Repr α] : Repr (Tensor α s) where
  reprPrec x _ := repr x.toArray

end Tensor

/-- Render a tensor recursively using the scalar `ToString` instance. -/
def pretty {α : Type} [ToString α] : ∀ {s : Shape}, Tensor α s → String
  | .scalar, Tensor.scalar x => toString x
  | .dim n _, Tensor.dim values =>
      "[" ++ String.intercalate ", " ((List.finRange n).map fun i => pretty (values i)) ++ "]"

end Spec
