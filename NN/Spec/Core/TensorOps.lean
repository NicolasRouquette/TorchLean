/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context
public import NN.Spec.Core.Tensor

/-!
# Elementwise tensor operations (`Spec.Tensor.*_spec`)

This file defines shape-preserving, elementwise operations on `Tensor α s`.

Naming convention:

- `foo_spec` means “pure spec definition” (no runtime side effects).
- most functions are defined by recursion on the tensor structure via `map_spec` / `map2_spec`.

## Domain / smoothness notes

Some operations are domain-sensitive or non-smooth:

- `sqrt_spec` uses `sqrt (max x 0)` to stay total on ordered rings.
- `log_spec` is total as a function call, but analytic properties require positivity assumptions.
- `relu` / `clamp` / comparisons are non-smooth; analytic backprop theorems treat these via
  pointwise assumptions, or by switching to smooth surrogates in verification workflows.

The **spec layer** is where these semantics are defined; the **proof layer** decides which
assumptions/variants to use for theorems.
-/

@[expose] public section


namespace Spec
namespace Tensor

variable {α : Type} [Context α]

/-! ## Axis-parametric indexing -/

/-- Place one selected slice back into an otherwise-zero tensor.

This is the adjoint of `selectSpec` and therefore its reverse-mode rule. -/
def selectBackwardSpec {α : Type} [Zero α] :
    (axis : Nat) → {s : Shape} →
      [_h : Shape.AxisInBounds axis s] →
      Fin (Shape.axisSize s axis) → Tensor α (s.eraseAxis axis) → Tensor α s
  | 0, .dim n rest, _, selected, gradient =>
      .dim fun coordinate =>
        if coordinate.val = selected.val then gradient else fill 0 rest
  | axis + 1, .dim n rest, h, selected, .dim gradients =>
      .dim fun outer =>
        @selectBackwardSpec α _ axis rest
          ⟨by
            have := h.proof
            simp only [Shape.rank] at this
            grind⟩
          (Fin.cast (by rfl) selected)
          (gradients outer)

/-- Select coordinates from an arbitrary axis using an index vector.

The selected axis is replaced by the index count. Repeated indices are preserved, matching
`torch.index_select`. -/
def indexSelectSpec {α : Type} :
    (axis : Nat) → {s : Shape} → (tensor : Tensor α s) →
      [_h : Shape.AxisInBounds axis s] →
      {count : Nat} → Tensor (Fin (Shape.axisSize s axis)) [count] →
      Tensor α (s.replaceAxis axis count)
  | 0, .dim _ _, .dim values, _, count, .dim indices =>
      .dim fun j =>
        match indices j with
        | .scalar i => values i
  | axis + 1, .dim n rest, .dim values, h, count, indices =>
      .dim fun outer =>
        @indexSelectSpec α axis rest (values outer)
          ⟨by
            have := h.proof
            simp only [Shape.rank] at this
            grind⟩
          count
          (cast (by rfl) indices)

/-- Insert a sliced gradient back along an arbitrary axis, filling coordinates outside it by zero. -/
def sliceAxisRangeBackwardSpec {α : Type} [Zero α] :
    (axis : Nat) → {s : Shape} →
      [_h : Shape.AxisInBounds axis s] →
      (start count : Nat) → (start + count ≤ Shape.axisSize s axis) →
      Tensor α (s.replaceAxis axis count) → Tensor α s
  | 0, .dim n rest, _, start, count, _hRange, .dim gradients =>
      .dim fun coordinate =>
        if hBefore : coordinate.val < start then
          fill 0 rest
        else if hInside : coordinate.val < start + count then
          gradients ⟨coordinate.val - start,
            Nat.sub_lt_left_of_lt_add (Nat.not_lt.mp hBefore) hInside⟩
        else
          fill 0 rest
  | axis + 1, .dim n rest, hAxis, start, count, hRange, .dim gradients =>
      let innerAxis : Shape.AxisInBounds axis rest :=
        ⟨by
          have := hAxis.proof
          simp only [Shape.rank] at this
          grind⟩
      letI := innerAxis
      have hRange' : start + count ≤ Shape.axisSize rest axis := by
        simpa only [Shape.axisSize_succ] using hRange
      .dim fun outer =>
        @sliceAxisRangeBackwardSpec α _ axis rest innerAxis start count hRange'
          (gradients outer)

/--
Map a scalar function over a tensor (shape preserved).

This is the core "structural recursion" combinator for spec tensors.
Most elementwise ops are direct instances of `mapSpec f`.

PyTorch analogy: `f` applied pointwise (like `torch.<op>` broadcasting over all entries),
but here shape is fixed and enforced by the type.
-/
def mapSpec {s : Shape} (f : α → α) : Tensor α s → Tensor α s :=
  Tensor.map f

omit [Context α] in
/-- Elementwise mapping computes directly on a scalar tensor. -/
@[simp] theorem mapSpec_scalar (f : α → α) (x : α) :
    mapSpec f (Tensor.scalar x) = Tensor.scalar (f x) := by
  rfl

omit [Context α] in
/-- Elementwise mapping distributes over the leading tensor dimension. -/
@[simp] theorem mapSpec_dim {n : Nat} {s : Shape} (f : α → α)
    (values : Fin n → Tensor α s) :
    mapSpec f (Tensor.dim values) = Tensor.dim (fun i => mapSpec f (values i)) := by
  rfl

omit [Context α] in
/-- Extracting a scalar after an elementwise map applies the scalar function once. -/
@[simp] theorem toScalar_mapSpec (f : α → α) (x : Tensor α .scalar) :
    (mapSpec f x).item = f x.item := by
  cases x
  rfl

omit [Context α] in
/-- Transport a pointwise tensor property through an elementwise operation. -/
theorem forall_mapSpec {p q : α → Prop} {f : α → α} {s : Shape} {x : Tensor α s}
    (hx : Forall p x) (hf : ∀ a, p a → q (f a)) :
    Forall q (mapSpec f x) := by
  induction s with
  | scalar =>
      cases x with
      | scalar a => exact hf a (by simpa using hx)
  | dim n inner ih =>
      cases x with
      | dim values =>
          intro i
          exact ih (hx i)

/--
Map a binary function over two tensors of the same shape.

This is the "zipWith" combinator for the spec tensor tree.
It is intentionally *shape-preserving*: if the shapes differ, the term is not well-typed.

PyTorch analogy: elementwise binary ops when tensors already have the same shape (no broadcasting).
Broadcasting is handled separately in `NN/Spec/Core/TensorReductionShape.lean`.
-/
def map2Spec {α β γ : Type} (f : α → β → γ) : ∀ {s : Shape}, Tensor α s → Tensor β s → Tensor γ s
  | Shape.scalar, Tensor.scalar x, Tensor.scalar y => Tensor.scalar (f x y)
  | Shape.dim _ _, Tensor.dim fx, Tensor.dim fy => Tensor.dim (fun i => map2Spec f (fx i) (fy i))

/-- Indexing a pointwise binary operation applies the scalar operation after indexing. -/
@[simp] theorem get_map2Spec {α β γ : Type} {f : α → β → γ} {n : Nat} {s : Shape}
    (left : Tensor α (.dim n s)) (right : Tensor β (.dim n s)) (i : Fin n) :
    _root_.Spec.get (map2Spec f left right) i =
      map2Spec f (_root_.Spec.get left i) (_root_.Spec.get right i) := by
  cases left
  cases right
  rfl

/-- Matrix indexing commutes with a pointwise binary tensor operation. -/
@[simp] theorem get2_map2Spec {α β γ : Type} {f : α → β → γ} {m n : Nat}
    (left : Tensor α [m, n])
    (right : Tensor β [m, n]) (i : Fin m) (j : Fin n) :
    _root_.Spec.get2 (map2Spec f left right) i j =
      f (_root_.Spec.get2 left i j) (_root_.Spec.get2 right i j) := by
  cases left with
  | dim leftRows =>
      cases right with
      | dim rightRows =>
          cases hLeft : leftRows i with
          | dim leftRow =>
              cases hRight : rightRows i with
              | dim rightRow =>
                  cases hLeftCell : leftRow j
                  cases hRightCell : rightRow j
                  simp [map2Spec, _root_.Spec.get2, _root_.Spec.get,
                    _root_.Spec.get, hLeft, hRight, hLeftCell, hRightCell]

/-- Transport two pointwise properties through a binary shape-preserving operation. -/
theorem forall_map2Spec {α β γ : Type} {p : α → Prop} {q : β → Prop} {r : γ → Prop}
    {f : α → β → γ} {s : Shape} {x : Tensor α s} {y : Tensor β s}
    (hx : Forall p x) (hy : Forall q y) (hf : ∀ a b, p a → q b → r (f a b)) :
    Forall r (map2Spec f x y) := by
  induction s with
  | scalar =>
      cases x with
      | scalar a =>
          cases y with
          | scalar b => exact hf a b (by simpa using hx) (by simpa using hy)
  | dim n inner ih =>
      cases x with
      | dim xs =>
          cases y with
          | dim ys =>
              intro i
              exact ih (hx i) (hy i)

/-- Element‑wise addition (shape preserved). -/
def addSpec {α : Type} [Add α] {s : Shape} (T₁ T₂ : Tensor α s) : Tensor α s :=
  map2Spec (· + ·) T₁ T₂

/-- Shape transport commutes with pointwise tensor addition. -/
theorem castShape_addSpec {α : Type} [Add α] {shape shape' : Shape}
    (left right : Tensor α shape) (h : shape = shape') :
    Tensor.castShape (addSpec left right) h =
      addSpec (Tensor.castShape left h) (Tensor.castShape right h) := by
  cases h
  rfl

/-- Add indexed source slices into an arbitrary axis of a tensor.

Repeated indices accumulate. This is the pure tensor semantics used by the backward rule for
`indexSelectSpec` and corresponds to `torch.scatter_add` with an index vector. -/
def scatterAddSpec {α : Type} [Add α] :
    (axis : Nat) → {s : Shape} → (base : Tensor α s) →
      [_h : Shape.AxisInBounds axis s] →
      {count : Nat} → Tensor (Fin (Shape.axisSize s axis)) [count] →
      Tensor α (s.replaceAxis axis count) → Tensor α s
  | 0, .dim _ _, .dim baseValues, _, count, .dim indices, .dim sourceValues =>
      .dim fun destination =>
        (List.finRange count).foldl
          (fun value source =>
            match indices source with
            | .scalar index =>
                if index.val = destination.val then addSpec value (sourceValues source) else value)
          (baseValues destination)
  | axis + 1, .dim n rest, .dim baseValues, h, count, indices, .dim sourceValues =>
      .dim fun outer =>
        @scatterAddSpec α _ axis rest (baseValues outer)
          ⟨by
            have := h.proof
            simp only [Shape.rank] at this
            grind⟩
          count
          (cast (by rfl) indices)
          (sourceValues outer)

/-- `Add` instance for shape-indexed tensors: add pointwise, preserving the shape. -/
instance {α : Type} [Add α] {s : Shape} : Add (Tensor α s) :=
  ⟨addSpec⟩

/-- Element‑wise multiplication (shape preserved). -/
def mulSpec {α : Type} [Mul α] {s : Shape} (T₁ T₂ : Tensor α s) : Tensor α s :=
  map2Spec (· * ·) T₁ T₂

/-- Scalar extraction commutes with pointwise tensor multiplication. -/
@[simp] theorem toScalar_mulSpec {α : Type} [Mul α]
    (left right : Tensor α .scalar) :
    (mulSpec left right).item = left.item * right.item := by
  cases left
  cases right
  rfl

/-- Multiplication by a filled tensor is coordinatewise multiplication by its scalar value. -/
@[simp] theorem mulSpec_fill_left {α : Type} [Mul α] {s : Shape} (coefficient : α) :
    ∀ t : Tensor α s, mulSpec (fill coefficient s) t = mapSpec (coefficient * ·) t
  | .scalar value => rfl
  | .dim values => by
      simp only [mulSpec, map2Spec, fill, mapSpec, Tensor.map]
      congr 1
      funext index
      exact mulSpec_fill_left coefficient (values index)

/-- `Mul` instance for shape-indexed tensors: multiply pointwise, preserving the shape. -/
instance {α : Type} [Mul α] {s : Shape} : Mul (Tensor α s) :=
  ⟨mulSpec⟩

/-- Element‑wise subtraction (shape preserved). -/
def subSpec {α : Type} [Sub α] {s : Shape} : Tensor α s → Tensor α s → Tensor α s :=
  map2Spec (· - ·)

/-- `Sub` instance for shape-indexed tensors: subtract pointwise, preserving the shape. -/
instance {α : Type} [Sub α] {s : Shape} : Sub (Tensor α s) :=
  ⟨subSpec⟩

/-- Element‑wise division (shape preserved). -/
def divSpec {s : Shape} : Tensor α s → Tensor α s → Tensor α s :=
  map2Spec (· / ·)

/-- `Div` instance for shape-indexed tensors: divide pointwise, preserving the shape. -/
instance {s : Shape} : Div (Tensor α s) :=
  ⟨divSpec⟩

/-- Safe division with epsilon protection, $x/(y+\varepsilon)$. -/
def safedivSpec {s : Shape} (t1 t2 : Tensor α s) : Tensor α s :=
  map2Spec (fun x y => x / (y + Numbers.epsilon)) t1 t2

/-- Scale a tensor by a scalar. -/
def scaleSpec {α : Type} [Mul α] {s : Shape} (t : Tensor α s) (scalar : α) : Tensor α s :=
  mapSpec (fun x => x * scalar) t

/-- Square each element of a tensor. -/
def squareSpec {s : Shape} (t : Tensor α s) : Tensor α s :=
  mapSpec (fun x => x * x) t

/-- Squaring is elementwise multiplication with the same tensor on both inputs. -/
theorem squareSpec_eq_mulSpec {s : Shape} (t : Tensor α s) :
    squareSpec t = mulSpec t t := by
  induction s with
  | scalar =>
      cases t
      rfl
  | dim n inner ih =>
      cases t with
      | dim values =>
          simp only [squareSpec, mapSpec, Tensor.map, mulSpec, map2Spec]
          congr 1
          funext i
          exact ih (values i)

/-- Square root of each element (clamped to `max x 0` to stay total). -/
def sqrtSpec {s : Shape} (t : Tensor α s) : Tensor α s :=
  mapSpec (fun x => MathFunctions.sqrt (Max.max x 0)) t

/-- Absolute value of each element. -/
def absSpec {s : Shape} (t : Tensor α s) : Tensor α s :=
  mapSpec MathFunctions.abs t

/-- Element‑wise natural log. -/
def logSpec {s : Shape} (t : Tensor α s) : Tensor α s :=
  mapSpec MathFunctions.log t

/-- Element‑wise exponential. -/
def expSpec {s : Shape} (t : Tensor α s) : Tensor α s :=
  mapSpec MathFunctions.exp t

/-- Element‑wise negation. -/
def negSpec {s : Shape} (t : Tensor α s) : Tensor α s :=
  mapSpec Neg.neg t

/-- `Neg` instance for shape-indexed tensors: negate pointwise, preserving the shape. -/
instance {s : Shape} : Neg (Tensor α s) :=
  ⟨negSpec⟩

/-- Multiply tensor by a constant. -/
def mulConstantSpec {s : Shape} (t : Tensor α s) (constant : α) : Tensor α s :=
  mapSpec (fun x => x * constant) t

/-- Element‑wise power. -/
def powSpec {s : Shape} (t1 t2 : Tensor α s) : Tensor α s :=
  map2Spec HPow.hPow t1 t2

/-- Element‑wise comparisons (returning Bool tensors). -/
def greaterThanSpec {s : Shape} (x y : Tensor α s) : Tensor Bool s :=
  map2Spec (fun a b => decide (a > b)) x y

/-- Element-wise $\le$ test, implemented via $\neg(>)$ so we only depend on
`DecidableRel (· > ·)`. -/
def lessEqualSpec {s : Shape} (x y : Tensor α s) : Tensor Bool s :=
  map2Spec (fun a b => decide (¬(a > b))) x y

/-- Element‑wise `<` test (defined as `y > x`). -/
def lessThanSpec {s : Shape} (x y : Tensor α s) : Tensor Bool s :=
  map2Spec (fun a b => decide (b > a)) x y

/-- Element-wise $\ge$ test (defined as $\neg(y>x)$). -/
def greaterEqualSpec {s : Shape} (x y : Tensor α s) : Tensor Bool s :=
  map2Spec (fun a b => decide (¬(b > a))) x y

/-- Boolean NOT, pointwise on a Bool tensor. -/
def notSpec {s : Shape} (x : Tensor Bool s) : Tensor Bool s :=
  mapSpec Bool.not x

/-- Element‑wise reciprocal (`1/x`). -/
def invSpec {s : Shape} (x : Tensor α s) : Tensor α s :=
  mapSpec (fun x => 1 / x) x

/-- Scalar extraction commutes with coordinatewise reciprocal. -/
@[simp] theorem toScalar_invSpec (x : Tensor α .scalar) :
    (invSpec x).item = 1 / x.item := by
  cases x
  rfl

/-- Element‑wise clamp into `[min_val, max_val]`. -/
def clampSpec {s : Shape} (x : Tensor α s) (min_val max_val : α) : Tensor α s :=
  mapSpec (fun v => Min.min max_val (Max.max min_val v)) x

/-- Element‑wise minimum. -/
def minSpec {s : Shape} (t1 t2 : Tensor α s) : Tensor α s :=
  map2Spec (fun x y => Min.min x y) t1 t2

/-- Element‑wise maximum. -/
def maxSpec {s : Shape} (t1 t2 : Tensor α s) : Tensor α s :=
  map2Spec (fun x y => Max.max x y) t1 t2

/-- Element‑wise sign function: returns `-1`, `0`, or `1`. -/
def signSpec {s : Shape} (t : Tensor α s) : Tensor α s :=
  mapSpec (fun x => if x > 0 then 1 else if x < 0 then -1 else 0) t

/-- Element‑wise cosh. -/
def coshSpec {s : Shape} : Tensor α s → Tensor α s :=
  mapSpec MathFunctions.cosh

/-- Element‑wise sinh. -/
def sinhSpec {s : Shape} : Tensor α s → Tensor α s :=
  mapSpec MathFunctions.sinh

/-- Derivative mask for clamp: `1` strictly inside `(min_val, max_val)`, else `0`. -/
def clampDerivativeSpec {s : Shape} (x : Tensor α s) (min_val max_val : α) : Tensor α s :=
  mapSpec (fun v => if v > min_val ∧ v < max_val then 1 else 0) x

/-- Numeric mask: `1` where `a > b`, else `0`. -/
def gtMaskSpec {s : Shape} (a b : Tensor α s) : Tensor α s :=
  map2Spec (fun x y => if x > y then 1 else 0) a b

/-- Numeric mask: `1` where `a < b`, else `0`. -/
def ltMaskSpec {s : Shape} (a b : Tensor α s) : Tensor α s :=
  map2Spec (fun x y => if x < y then 1 else 0) a b

/-- Convert a Bool to `α` using `1`/`0`. -/
def boolToAlphaSpec : Bool → α :=
  fun b => if b then 1 else 0

/-- Multiply a tensor by a Bool mask (casts the mask to `0/1`). -/
def mulBoolMaskSpec {s : Shape} (t : Tensor α s) (mask : Tensor Bool s)
  : Tensor α s :=
  map2Spec (fun x b => x * boolToAlphaSpec b) t mask

/-- Apply a Huber-style clamp on entries selected by `mask` (leaves others unchanged). -/
def clampHuberMaskSpec {s : Shape}
  (t : Tensor α s) (mask : Tensor Bool s) (delta : α) : Tensor α s :=
  map2Spec (fun x m =>
    if m then
      if x > delta then delta
      else if (-delta > x) then -delta
      else x
    else
      x
  ) t mask

/-- Update a tensor at a (runtime) index path.

The index path is interpreted outermost-first. Out-of-bounds indices leave the tensor unchanged.
This is an executable convenience helper; most proof layer code prefers total, shape-indexed access.
-/
def updateTensorSpec {α : Type} : ∀ {s : Shape}, Tensor α s → List Nat → α → Tensor α s
  | .scalar, .scalar _, [], new_val => .scalar new_val
  | .scalar, .scalar val, _ :: _, _ => .scalar val  -- Can't index into scalar
  | .dim _ _, .dim values, [], _ => .dim values     -- No index provided
  | .dim n _, .dim values, i :: rest, new_val =>
      if h : i < n then
        .dim (Function.update values ⟨i, h⟩
          (updateTensorSpec (values ⟨i, h⟩) rest new_val))
      else
        .dim values  -- Index out of bounds

/-- Like `update_tensor_spec`, but replaces a subtree with another tensor. -/
def updateTensorWithTensorSpec {α : Type} : ∀ {s : Shape}, Tensor α s → List Nat → Tensor α s →
  Tensor α s
  | .scalar, Tensor.scalar _, [], new_tensor => new_tensor
  | .scalar, Tensor.scalar val, _ :: _, _ => Tensor.scalar val  -- Can't index into scalar
  | .dim _ _, Tensor.dim values, [], _ => Tensor.dim values     -- No index provided
  | .dim n _, Tensor.dim values, i :: rest, new_tensor =>
      if h : i < n then
        .dim (Function.update values ⟨i, h⟩
          (updateTensorWithTensorSpec (values ⟨i, h⟩) rest (match new_tensor with
            | Tensor.dim new_values => new_values ⟨i, h⟩)))
      else
        .dim values  -- Index out of bounds

/-- Specialization of `update_tensor_spec` for a top-level vector dimension. -/
def updateSpec {α : Type} {n : ℕ} {s : Shape} :
     Tensor α (.dim n s) → List Nat → α → Tensor α (.dim n s)
  | .dim values, [], _ => .dim values  -- No index provided, return original
  | .dim values, i :: rest, new_val =>
    if h : i < n then
      .dim (Function.update values ⟨i, h⟩
         (updateTensorSpec (values ⟨i, h⟩) rest new_val))
    else
      .dim values  -- Index out of bounds, return original

end Tensor
end Spec
