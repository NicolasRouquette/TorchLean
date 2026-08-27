/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.SpecialFunctions.Exp
public import Mathlib.Data.Fin.Basic
public import Mathlib.Data.Nat.Basic
public import NN.Proofs.Analysis.Lipschitz
public import NN.Proofs.Tensor.Basic
public import NN.Spec.Core.Context
public import NN.Spec.Core.Shape
public import NN.Spec.Core.Tensor
public import NN.Spec.Core.TensorOps

/-!
# Tensor-shape induction and lifting lemmas

This file collects reusable *proof patterns* for reasoning about `Tensor` by structural induction
on its `Shape` (i.e. the nested `scalar`/`dim` structure), plus a few higher-level lifting lemmas
that are easiest to state once the Lipschitz/norm library is available.

## Why this exists
Many lemmas in TorchLean are naturally phrased as “for all shapes / for all dimensions …”.
Rather than re-proving the same induction scaffolding (or writing deeply nested `cases`/`induction`
blocks) throughout the repo, we keep a few canonical lemmas here.

- `tensor_induction_principle` for predicates `P : Tensor ℝ s → Prop`,
- `binary_tensor_induction` for predicates `P : Tensor ℝ s → Tensor ℝ s → Prop`.

These are especially useful when proving algebraic properties of `*_spec` tensor operations, or
norm/metric bounds that are proved “componentwise” and then lifted to the whole tensor.

Why this is not under `NN/Spec`: `NN/Spec` should define the mathematical objects and operations.
The induction principles below are theorem/proof conveniences about those objects, so they belong
under `NN/Proofs`.

## References
- This is standard structural induction on an inductive family; no external paper is required.
  The main detail is that TorchLean encodes tensors as a tree indexed by `Shape`,
  rather than (say) a flat array with a runtime `shape`.
-/

@[expose] public section


namespace Proofs

open Spec
open Tensor
open Shape

-- ====================================================================
-- DIMENSIONAL INDUCTION PATTERNS
-- ====================================================================

/--
Structural induction on tensors by their `Shape`.

Informally: to prove `P t` for all tensors `t`, it suffices to prove it for scalars, and to prove
that it is preserved when we build a higher-dimensional tensor `Tensor.dim f` from its components.
-/
theorem tensor_induction_principle
  (P : ∀ {s : Shape}, Tensor ℝ s → Prop)
  (base : ∀ x : ℝ, P (Tensor.scalar x))
  (step : ∀ {n : Nat} {s : Shape} (f : Fin n → Tensor ℝ s),
    (∀ i : Fin n, P (f i)) → P (Tensor.dim f))
  : ∀ {s : Shape} (t : Tensor ℝ s), P t := by
  intro s t
  induction s with
  | scalar =>
    cases t with | scalar x => exact base x
  | dim n s ih =>
    cases t with | dim f =>
    apply step
    intro i
    exact ih (f i)

/--
Structural induction for *binary* tensor predicates.

Informally: to prove `P t₁ t₂` for all tensors of the same shape, it suffices to prove it for
scalar pairs, and to prove it componentwise for `Tensor.dim f`/`Tensor.dim g`.
-/
theorem binary_tensor_induction
  (P : ∀ {s : Shape}, Tensor ℝ s → Tensor ℝ s → Prop)
  (base : ∀ x y : ℝ, P (Tensor.scalar x) (Tensor.scalar y))
  (step : ∀ {n : Nat} {s : Shape} (f g : Fin n → Tensor ℝ s),
    (∀ i : Fin n, P (f i) (g i)) → P (Tensor.dim f) (Tensor.dim g))
  : ∀ {s : Shape} (t₁ t₂ : Tensor ℝ s), P t₁ t₂ := by
  intro s t₁ t₂
  induction s with
  | scalar =>
    cases t₁ with | scalar x =>
    cases t₂ with | scalar y =>
    exact base x y
  | dim n s ih =>
    cases t₁ with | dim f =>
    cases t₂ with | dim g =>
    apply step
    intro i
    exact ih (f i) (g i)

-- ====================================================================
-- NORM PRESERVATION UNDER DIMENSIONAL SCALING
-- ====================================================================

/--
The squared $\ell_2$ norm of a concatenation is the sum of the squared $\ell_2$ norms.

Informally: `Tensor.dim f` is a “stack/concat along the outer dimension”. The Euclidean norm
satisfies
$\left\lVert\operatorname{concat}_i f_i\right\rVert_2^2
=\sum_i\lVert f_i\rVert_2^2$.
-/
theorem l2_norm_concatenation {n : Nat} {s : Shape}
  (f : Fin n → Tensor ℝ s) :
  (tensorL2Norm (Tensor.dim f))^2 =
  (List.finRange n).foldl (fun acc i => acc + (tensorL2Norm (f i))^2) 0 := by
  classical
  have l2_sq : ∀ {s : Shape} (t : Tensor ℝ s), (tensorL2Norm t)^2 = tensorNormSquared t := by
    intro s t
    simp [tensorL2Norm, Real.sq_sqrt (tensor_norm_squared_nonneg (t := t))]
  calc
    (tensorL2Norm (Tensor.dim f))^2 = tensorNormSquared (Tensor.dim f) := l2_sq (t := Tensor.dim
      f)
    _ = (Finset.univ : Finset (Fin n)).sum (fun i => tensorNormSquared (f i)) := by
      calc
        tensorNormSquared (Tensor.dim f) = sumSpec (Tensor.dim (fun i => mulSpec (f i) (f i)))
          := by rfl
        _ = (Finset.univ : Finset (Fin n)).sum (fun i => sumSpec (mulSpec (f i) (f i))) := by
          -- Use the canonical lemma from `NN/Proofs/Tensor/Basic.lean` instead of duplicating the
          -- outer-fold-to-`Finset.sum` proof here.
            simpa [Spec.get] using
              (Spec.sum_spec_dim (t := Tensor.dim (fun i => mulSpec (f i) (f i))))
        _ = (Finset.univ : Finset (Fin n)).sum (fun i => tensorNormSquared (f i)) := by
          refine Finset.sum_congr rfl ?_
          intro i _
          rfl
    _ = (Finset.univ : Finset (Fin n)).sum (fun i => (tensorL2Norm (f i))^2) := by
      refine Finset.sum_congr rfl ?_
      intro i _
      simpa using (l2_sq (t := f i)).symm
    _ = (List.finRange n).foldl (fun acc i => acc + (tensorL2Norm (f i))^2) 0 := by
      simpa using
        (List.finRange_foldl_add_eq_finset_sum (f := fun i : Fin n => (tensorL2Norm (f
          i))^2)).symm

/--
Component-wise bounds extend to full tensors.
Key principle for lifting scalar bounds to tensor bounds.
-/
theorem componentwise_bound_extension {n : Nat} {s : Shape}
  (f g : Fin n → Tensor ℝ s) (C : ℝ)
  (h : ∀ i : Fin n, tensorL2Norm (f i) ≤ C * tensorL2Norm (g i)) :
  tensorL2Norm (Tensor.dim f) ≤ C * tensorL2Norm (Tensor.dim g) := by
  classical
  by_cases hC : 0 ≤ C
  · -- Compare squares and use `le_of_sq_le_sq` (RHS nonnegative).
    have hSf :
        (tensorL2Norm (Tensor.dim f))^2 =
          (Finset.univ : Finset (Fin n)).sum (fun i => (tensorL2Norm (f i))^2) := by
      calc
        (tensorL2Norm (Tensor.dim f))^2 =
            (List.finRange n).foldl (fun acc i => acc + (tensorL2Norm (f i))^2) 0 := by
              simpa using (l2_norm_concatenation (f := f))
        _ = (Finset.univ : Finset (Fin n)).sum (fun i => (tensorL2Norm (f i))^2) := by
          simpa using
            (List.finRange_foldl_add_eq_finset_sum (f := fun i : Fin n => (tensorL2Norm (f i))^2))
    have hSg :
        (tensorL2Norm (Tensor.dim g))^2 =
          (Finset.univ : Finset (Fin n)).sum (fun i => (tensorL2Norm (g i))^2) := by
      calc
        (tensorL2Norm (Tensor.dim g))^2 =
            (List.finRange n).foldl (fun acc i => acc + (tensorL2Norm (g i))^2) 0 := by
              simpa using (l2_norm_concatenation (f := g))
        _ = (Finset.univ : Finset (Fin n)).sum (fun i => (tensorL2Norm (g i))^2) := by
          simpa using
            (List.finRange_foldl_add_eq_finset_sum (f := fun i : Fin n => (tensorL2Norm (g i))^2))

    have h_term :
        ∀ i : Fin n, (tensorL2Norm (f i))^2 ≤ C^2 * (tensorL2Norm (g i))^2 := by
      intro i
      have hi := h i
      have hf_nonneg : 0 ≤ tensorL2Norm (f i) := by
        simpa [ge_iff_le] using tensor_l2_norm_nonneg (t := f i)
      have hg_nonneg : 0 ≤ tensorL2Norm (g i) := by
        simpa [ge_iff_le] using tensor_l2_norm_nonneg (t := g i)
      have hCg_nonneg : 0 ≤ C * tensorL2Norm (g i) := mul_nonneg hC hg_nonneg
      have hsq :
          (tensorL2Norm (f i))^2 ≤ (C * tensorL2Norm (g i))^2 := by
        have hmul :
            tensorL2Norm (f i) * tensorL2Norm (f i) ≤
              (C * tensorL2Norm (g i)) * (C * tensorL2Norm (g i)) :=
          mul_le_mul hi hi hf_nonneg hCg_nonneg
        simpa [pow_two] using hmul
      simpa [mul_pow] using hsq

    have hsquared :
        (tensorL2Norm (Tensor.dim f))^2 ≤ (C * tensorL2Norm (Tensor.dim g))^2 := by
      -- Convert to `Finset` sums and bound termwise.
      rw [hSf]
      simp [mul_pow]
      rw [hSg]
      have hsum :
          (Finset.univ : Finset (Fin n)).sum (fun i => (tensorL2Norm (f i))^2) ≤
            (Finset.univ : Finset (Fin n)).sum (fun i => C^2 * (tensorL2Norm (g i))^2) := by
        refine Finset.sum_le_sum ?_
        intro i _
        exact h_term i
      calc
        (Finset.univ : Finset (Fin n)).sum (fun i => (tensorL2Norm (f i))^2)
            ≤ (Finset.univ : Finset (Fin n)).sum (fun i => C^2 * (tensorL2Norm (g i))^2) := hsum
        _ = C^2 * (Finset.univ : Finset (Fin n)).sum (fun i => (tensorL2Norm (g i))^2) := by
          simpa using
            (Finset.mul_sum (s := (Finset.univ : Finset (Fin n)))
              (f := fun i : Fin n => (tensorL2Norm (g i))^2) (a := C^2)).symm

    have hR_nonneg : 0 ≤ C * tensorL2Norm (Tensor.dim g) := by
      have hg_nonneg : 0 ≤ tensorL2Norm (Tensor.dim g) := by
        simpa [ge_iff_le] using tensor_l2_norm_nonneg (t := Tensor.dim g)
      exact mul_nonneg hC hg_nonneg
    exact le_of_sq_le_sq hsquared hR_nonneg

  · -- If `C < 0`, the hypotheses force both sides to be zero.
    have hCneg : C < 0 := lt_of_not_ge hC

    have hg_norm0 : ∀ i : Fin n, tensorL2Norm (g i) = 0 := by
      intro i
      have hf_nonneg : 0 ≤ tensorL2Norm (f i) := by
        simpa [ge_iff_le] using tensor_l2_norm_nonneg (t := f i)
      have hCg_nonneg : 0 ≤ C * tensorL2Norm (g i) := le_trans hf_nonneg (h i)
      have hg_nonneg : 0 ≤ tensorL2Norm (g i) := by
        simpa [ge_iff_le] using tensor_l2_norm_nonneg (t := g i)
      have hCg_nonpos : C * tensorL2Norm (g i) ≤ 0 :=
        mul_nonpos_of_nonpos_of_nonneg (le_of_lt hCneg) hg_nonneg
      have hCg0 : C * tensorL2Norm (g i) = 0 := le_antisymm hCg_nonpos hCg_nonneg
      have hCne : C ≠ 0 := ne_of_lt hCneg
      rcases (mul_eq_zero.mp hCg0) with hC0 | hg0
      · exact (hCne hC0).elim
      · exact hg0

    have hf_norm0 : ∀ i : Fin n, tensorL2Norm (f i) = 0 := by
      intro i
      have hg0 := hg_norm0 i
      have hf_nonneg : 0 ≤ tensorL2Norm (f i) := by
        simpa [ge_iff_le] using tensor_l2_norm_nonneg (t := f i)
      have hf_le0 : tensorL2Norm (f i) ≤ 0 := by simpa [hg0] using (h i)
      exact le_antisymm hf_le0 hf_nonneg

    have hg0 : ∀ i : Fin n, g i = fill (0 : ℝ) s := by
      intro i
      exact (tensor_l2_norm_zero_iff (t := g i)).1 (hg_norm0 i)
    have hf0 : ∀ i : Fin n, f i = fill (0 : ℝ) s := by
      intro i
      exact (tensor_l2_norm_zero_iff (t := f i)).1 (hf_norm0 i)

    have hg_dim : Tensor.dim g = fill (0 : ℝ) (.dim n s) := by
      have : g = (fun _ : Fin n => fill (0 : ℝ) s) := by
        funext i
        exact hg0 i
      simp [this, fill]

    have hf_dim : Tensor.dim f = fill (0 : ℝ) (.dim n s) := by
      have : f = (fun _ : Fin n => fill (0 : ℝ) s) := by
        funext i
        exact hf0 i
      simp [this, fill]

    have nf0 : tensorL2Norm (Tensor.dim f) = 0 :=
      (tensor_l2_norm_zero_iff (t := Tensor.dim f)).2 hf_dim
    have ng0 : tensorL2Norm (Tensor.dim g) = 0 :=
      (tensor_l2_norm_zero_iff (t := Tensor.dim g)).2 hg_dim
    simp [nf0, ng0]

-- ====================================================================
-- ACTIVATION FUNCTION INDUCTIVE ANALYSIS
-- ====================================================================

/--
ReLU preserves non-negativity inductively over all dimensions.

PyTorch analogue: `relu` is defined pointwise as $\max(x,0)$, so its outputs are always
$\ge 0$.
https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html
-/
theorem relu_nonneg_inductive {s : Shape} (t : Tensor ℝ s) :
  ∀ indices : List Nat,
  match getSpec (Activation.reluSpec t) indices with
  | some x => x ≥ 0
  | none => True := by
  apply tensor_induction_principle
    (P := fun {s} t =>
      ∀ indices : List Nat,
        match getSpec (Activation.reluSpec t) indices with
        | some x => x ≥ 0
        | none => True)
    (t := t)
  · -- Base case: scalar
    intro x indices
    cases indices with
    | nil =>
      -- ReLU(x) = max x 0, so it is always nonnegative.
      simp [Activation.reluSpec, Activation.Math.reluSpec, mapSpec]
    | cons _ _ => simp [Activation.reluSpec, Activation.Math.reluSpec, mapSpec]
  · -- Inductive case
    intro n s f ih indices
    simp [Activation.reluSpec, mapSpec]
    cases indices with
    | nil =>
      simp
    | cons head tail =>
        simp
        by_cases h : head < n
        · simpa [Activation.reluSpec, mapSpec, h] using ih ⟨head, h⟩ tail
        · simp [h]

/--
Sigmoid output bounds extend inductively.
Shows $0<\sigma(x)<1$ for all tensor components.

PyTorch analogue: `torch.sigmoid` maps reals to the open interval (0, 1) pointwise.
https://pytorch.org/docs/stable/generated/torch.sigmoid.html
-/
theorem sigmoid_bounds_inductive {s : Shape} (t : Tensor ℝ s) :
  ∀ indices : List Nat,
  match getSpec (mapSpec (fun x => 1 / (1 + Real.exp (-x))) t) indices with
  | some y => 0 < y ∧ y < 1
  | none => True := by
  refine tensor_induction_principle
    (P := fun {s} t =>
      ∀ indices : List Nat,
        match getSpec (mapSpec (fun x => 1 / (1 + Real.exp (-x))) t) indices with
        | some y => 0 < y ∧ y < 1
        | none => True)
    (t := t) ?_ ?_
  · -- Base case: scalar
    intro x indices
    cases indices with
    | nil =>
      simp [mapSpec]
      have hden_pos : 0 < (1 + Real.exp (-x)) := by
        have : 0 < Real.exp (-x) := by simpa using Real.exp_pos (-x)
        linarith
      have hden_lt : (1 : ℝ) < (1 + Real.exp (-x)) := by
        have : 0 < Real.exp (-x) := by simpa using Real.exp_pos (-x)
        linarith
      constructor
      · exact hden_pos
      ·
        have : (1 : ℝ) / (1 + Real.exp (-x)) < 1 := (div_lt_one hden_pos).2 hden_lt
        simpa [one_div] using this
    | cons _ _ =>
      simp [mapSpec]
  · -- Inductive case
    intro n s f ih indices
    cases indices with
    | nil =>
      simp
    | cons head tail =>
      rw [mapSpec_dim, get_spec_dim_cons]
      by_cases h : head < n
      · simp only [h, dite_true]
        exact ih ⟨head, h⟩ tail
      · simp [h]

-- ====================================================================
-- LINEAR TRANSFORMATION INDUCTIVE PROPERTIES
-- ====================================================================

/--
Matrix-vector multiplication dimension consistency.
Proves output dimensions are correct regardless of input tensor structure.
-/
theorem matvec_dimension_consistency {m n : Nat}
  (A : Tensor ℝ [m, n])
  (x : Tensor ℝ [n]) :
  shapeOf (matVecMulSpec A x) = .dim m .scalar := by
  exact shapeOf_eq_shape (matVecMulSpec A x)

/--
Linear transformation preserves tensor structure inductively.
Shows that linearity holds component-wise across all dimensions.
-/
theorem linear_structure_preservation {m n : Nat}
  (A : Tensor ℝ [m, n]) :
  ∀ (x y : Tensor ℝ [n]) (a b : ℝ),
  matVecMulSpec A (addSpec (scaleSpec x a) (scaleSpec y b)) =
  addSpec (scaleSpec (matVecMulSpec A x) a) (scaleSpec (matVecMulSpec A y) b) := by
  intro x y a b
  simpa using (Spec.mat_vec_linear_combination (W := A) (x := x) (y := y) (a := a) (b := b))

-- ====================================================================
-- COMPOSITION INDUCTIVE THEOREMS
-- ====================================================================

/-- A tensor map packaged with a proved nonnegative Lipschitz constant. -/
structure LipschitzLayer (s : Shape) where
  /-- The layer's forward map. -/
  forward : Tensor ℝ s → Tensor ℝ s
  /-- A global Lipschitz constant for `forward`. -/
  constant : ℝ
  /-- Lipschitz constants are nonnegative. -/
  constant_nonneg : 0 ≤ constant
  /-- The distance bound witnessed by `constant`. -/
  dist_le : ∀ x y, tensorL2Dist (forward x) (forward y) ≤ constant * tensorL2Dist x y

/-- Apply a runtime-sized sequence of shape-preserving layers from left to right. -/
def composeFunctions {s : Shape} (layers : Array (LipschitzLayer s))
    (x : Tensor ℝ s) : Tensor ℝ s :=
  layers.foldl (fun value layer => layer.forward value) x

/-- Product of the Lipschitz constants attached to a layer sequence. -/
def composedLipschitzConstant {s : Shape} (layers : Array (LipschitzLayer s)) : ℝ :=
  layers.foldr (fun layer bound => layer.constant * bound) 1

private def composeLayerList {s : Shape} (layers : List (LipschitzLayer s))
    (x : Tensor ℝ s) : Tensor ℝ s :=
  layers.foldl (fun value layer => layer.forward value) x

private def layerListConstant {s : Shape} (layers : List (LipschitzLayer s)) : ℝ :=
  layers.foldr (fun layer bound => layer.constant * bound) 1

private theorem layerListConstant_nonneg {s : Shape} (layers : List (LipschitzLayer s)) :
    0 ≤ layerListConstant layers := by
  induction layers with
  | nil => simp [layerListConstant]
  | cons layer layers ih =>
      simpa [layerListConstant] using mul_nonneg layer.constant_nonneg ih

private theorem composeLayerList_dist_le {s : Shape} (layers : List (LipschitzLayer s))
    (x y : Tensor ℝ s) :
    tensorL2Dist (composeLayerList layers x) (composeLayerList layers y) ≤
      layerListConstant layers * tensorL2Dist x y := by
  induction layers generalizing x y with
  | nil => simp [composeLayerList, layerListConstant]
  | cons layer layers ih =>
      have tailConstantNonneg : 0 ≤ layerListConstant layers :=
        layerListConstant_nonneg layers
      calc
        tensorL2Dist (composeLayerList (layer :: layers) x)
            (composeLayerList (layer :: layers) y) =
            tensorL2Dist (composeLayerList layers (layer.forward x))
              (composeLayerList layers (layer.forward y)) := by
                rfl
        _ ≤ layerListConstant layers *
              tensorL2Dist (layer.forward x) (layer.forward y) := ih _ _
        _ ≤ layerListConstant layers * (layer.constant * tensorL2Dist x y) :=
          mul_le_mul_of_nonneg_left (layer.dist_le x y) tailConstantNonneg
        _ = layerListConstant (layer :: layers) * tensorL2Dist x y := by
          simp only [layerListConstant, List.foldr_cons]
          ring

/-- The composition of proved Lipschitz layers is Lipschitz with the product bound. -/
theorem nested_lipschitz_composition {s : Shape} (layers : Array (LipschitzLayer s))
    (x y : Tensor ℝ s) :
    tensorL2Dist (composeFunctions layers x) (composeFunctions layers y) ≤
      composedLipschitzConstant layers * tensorL2Dist x y := by
  simpa [composeFunctions, composedLipschitzConstant, composeLayerList, layerListConstant,
    Array.foldl_toList, Array.foldr_toList] using composeLayerList_dist_le layers.toList x y

end Proofs
