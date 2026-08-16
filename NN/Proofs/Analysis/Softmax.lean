/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.BigOperators.Field
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Analysis.SpecialFunctions.Exp
public import NN.Proofs.Tensor.Basic
public import NN.Proofs.Utils.List
public import NN.Proofs.Utils.MathFunctions
public import NN.Spec.Layers.Activation

/-!
# Softmax analysis properties

This module proves theorem-level facts about TorchLean's spec-level softmax operators. The
definitions themselves live in `NN.Spec.Layers.Activation`; this file belongs under
`NN.Proofs.Analysis` because it imports real-analysis and finite-sum proof machinery to establish
properties of those definitions.

Current theorem surface:

- `toVec_le_maxVecSpec` and `exists_toVec_eq_maxVecSpec`: the shared stable-shift maximum is an
  attained upper bound;
- `softmax_shift_nonpos` and `exists_softmax_shift_eq_zero`: shifted logits are nonpositive and
  one is exactly zero;
- `softmax_shift_exp_le_one` and `softmax_shift_denom_bounds`: exponentials cannot overflow and
  their denominator lies between `1` and the axis length;
- `softmax_vec_spec_normalized`: exposes the positive normalized weights used by the stable
  max-shifted implementation;
- `softmax_vec_spec_pos`: every coordinate is strictly positive;
- `softmax_vec_spec_mem_unitInterval`: every coordinate lies in `[0,1]`;
- `sum_spec_softmax_vec_spec`: a nonempty vector softmax sums to `1`;
- `sum_spec_softmax_backward_spec`: the concrete stable softmax VJP has coordinate sum zero;
- `abs_toVec_softmax_backward_spec_le_two_mul`: bounded upstream coordinates give a
  dimension-independent coordinate bound for that VJP;
- `sum_spec_softmax_spec_row`: matrix softmax is rowwise, so each nonempty row sums to `1`;
- `sum_spec_softmax_spec_row_of_ne_zero`: the same row theorem with nonemptiness supplied as
  `nK ≠ 0`.

We intentionally state these over `ℝ`: positivity of `exp` and division by a positive denominator
are the mathematical facts that make the probabilistic interpretation precise.
-/

@[expose] public section

open scoped BigOperators

noncomputable section

namespace Proofs

open Spec
open Tensor
open Activation

/-! ## Scalar helpers

`softmaxVecSpec` is written over tensors, so even one coordinate has type `Tensor ℝ .scalar`.
Local helper definitions expose scalar coordinates to the proof without adding public API.
-/

/--
Eliminate a scalar tensor using the same matcher as `Activation.softmaxVecSpec`.

This local eliminator avoids depending on compiler-generated matcher names, which are not a stable
interface and can change when an earlier definition is inserted in `Activation.lean`.
-/
private def scalarElim {β : Sort _} (t : Tensor ℝ .scalar) (k : ℝ → β) : β :=
  match t with
  | Tensor.scalar value => k value

@[simp] private theorem scalarElim_scalar {β : Sort _} (k : ℝ → β) (v : ℝ) :
    scalarElim (β := β) (Tensor.scalar v) k = k v := rfl

/-- Extract the real value from a scalar tensor for local proof steps. -/
private abbrev scalarVal (t : Tensor ℝ .scalar) : ℝ :=
  scalarElim (β := ℝ) t (fun v => v)

/-! ## Stable max shift -/

/-- Every coordinate is bounded above by the exact maximum used by softmax and log-softmax. -/
theorem toVec_le_maxVecSpec {n : Nat}
    (t : Tensor ℝ (.dim (Nat.succ n) .scalar)) (i : Fin (Nat.succ n)) :
    Spec.toVec t i <= Tensor.toScalar (Activation.maxVecSpec t) := by
  cases t with
  | dim values =>
      change Tensor.toScalar (values i) <= _
      change Tensor.toScalar (values i) <=
        (List.finRange (Nat.succ n)).foldl
          (fun acc j => max acc (Tensor.toScalar (values j)))
          (Tensor.toScalar (values ⟨0, Nat.succ_pos n⟩))
      exact List.le_foldl_max_of_mem (List.finRange (Nat.succ n))
        (fun j => Tensor.toScalar (values j))
        (acc := Tensor.toScalar (values ⟨0, Nat.succ_pos n⟩)) (i := i)
        (List.mem_finRange i)

/-- The maximum used by stable softmax is attained by an input coordinate. -/
theorem exists_toVec_eq_maxVecSpec {n : Nat}
    (t : Tensor ℝ (.dim (Nat.succ n) .scalar)) :
    ∃ i, Spec.toVec t i = Tensor.toScalar (Activation.maxVecSpec t) := by
  cases t with
  | dim values =>
      let firstIndex : Fin (Nat.succ n) := ⟨0, Nat.succ_pos n⟩
      let value : Fin (Nat.succ n) -> ℝ := fun i => Tensor.toScalar (values i)
      change ∃ i, value i =
        (List.finRange (Nat.succ n)).foldl (fun acc j => max acc (value j))
          (value firstIndex)
      rcases List.foldl_max_eq_init_or_mem (List.finRange (Nat.succ n)) value
          (value firstIndex) with hfirst | ⟨i, hi, hvalue⟩
      · exact ⟨firstIndex, hfirst.symm⟩
      · exact ⟨i, hvalue.symm⟩

/-- Every logit shifted by the implementation's maximum is nonpositive. -/
theorem softmax_shift_nonpos {n : Nat}
    (t : Tensor ℝ (.dim (Nat.succ n) .scalar)) (i : Fin (Nat.succ n)) :
    Spec.toVec
      (Spec.Tensor.subSpec t (Spec.replicate (Activation.maxVecSpec t))) i <= 0 := by
  cases t with
  | dim values =>
      cases hvalue : values i with
      | scalar value =>
          cases hmaxEq : Activation.maxVecSpec (Tensor.dim values) with
          | scalar maximum =>
              have hle := toVec_le_maxVecSpec (t := Tensor.dim values) i
              have : value <= maximum := by
                simpa [Spec.toVec, hvalue, hmaxEq] using hle
              simpa [Spec.toVec, Spec.Tensor.subSpec, Spec.Tensor.map2Spec, Spec.replicate, hvalue,
                hmaxEq] using sub_nonpos.mpr this

/-- At least one max-shifted logit is exactly zero. -/
theorem exists_softmax_shift_eq_zero {n : Nat}
    (t : Tensor ℝ (.dim (Nat.succ n) .scalar)) :
    ∃ i, Spec.toVec
      (Spec.Tensor.subSpec t (Spec.replicate (Activation.maxVecSpec t))) i = 0 := by
  rcases exists_toVec_eq_maxVecSpec t with ⟨i, hi⟩
  refine ⟨i, ?_⟩
  cases t with
  | dim values =>
      cases hvalue : values i with
      | scalar value =>
          cases hmax : Activation.maxVecSpec (Tensor.dim values) with
          | scalar maximum =>
              have hi' : value = maximum := by
                simpa [Spec.toVec, hvalue, hmax] using hi
              simpa [Spec.toVec, Spec.Tensor.subSpec, Spec.Tensor.map2Spec, Spec.replicate,
                hvalue, hmax] using sub_eq_zero.mpr hi'

/-- Exponentiating a max-shifted real logit produces a value at most one. This is the central
overflow-prevention fact behind stable softmax. -/
theorem softmax_shift_exp_le_one {n : Nat}
    (t : Tensor ℝ (.dim (Nat.succ n) .scalar)) (i : Fin (Nat.succ n)) :
    Spec.toVec (Activation.maxShiftedExpVecSpec t) i <= 1 := by
  have hshift := softmax_shift_nonpos t i
  cases t with
  | dim values =>
      cases hvalue : values i with
      | scalar value =>
          cases hmax : Activation.maxVecSpec (Tensor.dim values) with
          | scalar maximum =>
              have hshift' : value - maximum <= 0 := by
                simpa [Spec.toVec, Spec.Tensor.subSpec, Spec.Tensor.map2Spec, Spec.replicate,
                  hvalue, hmax] using hshift
              simpa [Spec.toVec, Spec.Tensor.expSpec, Spec.Tensor.subSpec, Spec.Tensor.mapSpec,
                Spec.Tensor.map2Spec, Spec.replicate, Activation.maxShiftedExpVecSpec, hvalue,
                hmax, mathfunc_exp_eq_rexp] using
                (Real.exp_le_one_iff.mpr hshift')

/-- Max-shifted real exponentials remain strictly positive. -/
theorem softmax_shift_exp_pos {n : Nat}
    (t : Tensor ℝ (.dim (Nat.succ n) .scalar)) (i : Fin (Nat.succ n)) :
    0 < Spec.toVec (Activation.maxShiftedExpVecSpec t) i := by
  cases t with
  | dim values =>
      cases hvalue : values i with
      | scalar value =>
          cases hmax : Activation.maxVecSpec (Tensor.dim values) with
          | scalar maximum =>
              simpa [Activation.maxShiftedExpVecSpec, Spec.toVec, Spec.Tensor.expSpec,
                Spec.Tensor.subSpec, Spec.Tensor.mapSpec, Spec.Tensor.map2Spec, Spec.replicate,
                hvalue, hmax, mathfunc_exp_eq_rexp] using Real.exp_pos (value - maximum)

/-- One max-shifted exponential is exactly one, because the maximum is attained. -/
theorem exists_softmax_shift_exp_eq_one {n : Nat}
    (t : Tensor ℝ (.dim (Nat.succ n) .scalar)) :
    ∃ i, Spec.toVec (Activation.maxShiftedExpVecSpec t) i = 1 := by
  rcases exists_softmax_shift_eq_zero t with ⟨i, hzero⟩
  refine ⟨i, ?_⟩
  cases t with
  | dim values =>
      cases hvalue : values i with
      | scalar value =>
          cases hmax : Activation.maxVecSpec (Tensor.dim values) with
          | scalar maximum =>
              have hzero' : value - maximum = 0 := by
                simpa [Spec.toVec, Spec.Tensor.subSpec, Spec.Tensor.map2Spec, Spec.replicate,
                  hvalue, hmax] using hzero
              simpa [Activation.maxShiftedExpVecSpec, Spec.toVec, Spec.Tensor.expSpec,
                Spec.Tensor.subSpec, Spec.Tensor.mapSpec, Spec.Tensor.map2Spec, Spec.replicate,
                hvalue, hmax, mathfunc_exp_eq_rexp] using (Real.exp_eq_one_iff _).mpr hzero'

/-- The stable softmax denominator lies in `[1,n]` for a nonempty vector of length `n`.

The lower bound rules out division by zero. The upper bound follows because every shifted
exponential is at most one. Together with `softmax_shift_exp_le_one`, this makes overflow
prevention an explicit theorem of the max-shifted implementation rather than an empirical claim.
-/
theorem softmax_shift_denom_bounds {n : Nat}
    (t : Tensor ℝ (.dim (Nat.succ n) .scalar)) :
    1 <= Spec.Tensor.sumSpec (Activation.maxShiftedExpVecSpec t) ∧
      Spec.Tensor.sumSpec (Activation.maxShiftedExpVecSpec t) <= Nat.succ n := by
  classical
  let ex := Activation.maxShiftedExpVecSpec t
  have hpos : ∀ i, 0 <= Spec.toVec ex i := fun i => le_of_lt (softmax_shift_exp_pos t i)
  have hle : ∀ i, Spec.toVec ex i <= 1 := fun i => softmax_shift_exp_le_one t i
  rcases exists_softmax_shift_exp_eq_one t with ⟨witness, hwitness⟩
  rw [Spec.sum_spec_vec]
  constructor
  · calc
      1 = Spec.toVec ex witness := hwitness.symm
      _ <= ∑ i, Spec.toVec ex i :=
        Finset.single_le_sum (fun i _ => hpos i) (Finset.mem_univ witness)
  · calc
      (∑ i, Spec.toVec ex i) <= ∑ _i : Fin (Nat.succ n), (1 : ℝ) := by
        exact Finset.sum_le_sum fun i _ => hle i
      _ = Nat.succ n := by simp

/-! ## Normalized coordinates -/

/--
The stable vector softmax has positive weights normalized by their sum.

This lemma exposes exactly one reusable algebraic description of the implementation. The weights
are the max-shifted exponentials computed by `softmaxVecSpec`; subsequent proofs of positivity,
range, and normalization do not unfold the implementation again.
-/
theorem softmax_vec_spec_normalized {n : Nat}
    (t : Tensor ℝ (.dim (Nat.succ n) .scalar)) :
    ∃ weights : Fin (Nat.succ n) → ℝ,
      (∀ i, 0 < weights i) ∧
      ∀ i,
        Spec.toVec (Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) t) i =
          weights i / ∑ j, weights j := by
  classical
  cases t with
  | dim f =>
      let input : Tensor ℝ (.dim (Nat.succ n) .scalar) := Tensor.dim f
      let maxT : Tensor ℝ .scalar := Activation.maxVecSpec input
      let shifted := Spec.Tensor.subSpec input (Spec.replicate maxT)
      let exponentials := Spec.Tensor.expSpec shifted
      let weights : Fin (Nat.succ n) → ℝ := fun j => Spec.toVec exponentials j
      have hweightsPos : ∀ j, 0 < weights j := by
        intro j
        cases hj : f j with
        | scalar xj =>
            simp only [weights, exponentials, shifted, input, Spec.toVec, Spec.Tensor.expSpec,
              Spec.Tensor.subSpec, Spec.Tensor.mapSpec, Spec.Tensor.map2Spec, Spec.replicate, hj]
            exact Real.exp_pos _
      let denom : ℝ := Spec.Tensor.sumSpec exponentials
      have hdenom : denom = ∑ j : Fin (Nat.succ n), weights j := by
        simpa [denom, weights] using Spec.sum_spec_vec exponentials
      have hcoord : ∀ i : Fin (Nat.succ n),
          Spec.toVec (Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) (Tensor.dim f)) i =
            weights i / denom := by
        intro i
        change Spec.toVec
            (Spec.Tensor.divSpec exponentials (Spec.replicate (Tensor.scalar denom))) i =
          Spec.toVec exponentials i / denom
        cases exponentials with
        | dim values =>
            cases hvalue : values i with
            | scalar value =>
                simp [Spec.toVec, Spec.Tensor.divSpec, Spec.Tensor.map2Spec, Spec.replicate,
                  hvalue]
      refine ⟨weights, hweightsPos, ?_⟩
      intro i
      simpa [hdenom] using hcoord i

/-- Coordinate equation for the concrete stable vector softmax.

This is the small unfolding lemma that downstream algebraic proofs should use. It exposes the
max-shifted numerator and its tensor sum while hiding the implementation chosen for tensor
reduction. -/
theorem toVec_softmaxVecSpec {n : Nat}
    (t : Tensor ℝ (.dim (Nat.succ n) .scalar)) (i : Fin (Nat.succ n)) :
    Spec.toVec (Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) t) i =
      Spec.toVec (Activation.maxShiftedExpVecSpec t) i /
        Spec.Tensor.sumSpec (Activation.maxShiftedExpVecSpec t) := by
  cases t with
  | dim values =>
      let exponentials := Activation.maxShiftedExpVecSpec (Tensor.dim values)
      change Spec.toVec
          (Spec.Tensor.divSpec exponentials
            (Spec.replicate (Tensor.scalar (Spec.Tensor.sumSpec exponentials)))) i = _
      cases hEx : exponentials with
      | dim exValues =>
          cases hvalue : exValues i with
          | scalar value =>
              simp [Spec.toVec, Spec.Tensor.divSpec, Spec.Tensor.map2Spec, Spec.replicate,
                exponentials, hEx, hvalue]

/-! ## Probability-simplex properties -/

/-- Every coordinate of a nonempty real softmax vector is strictly positive. -/
theorem softmax_vec_spec_pos {n : Nat}
    (t : Tensor ℝ (.dim (Nat.succ n) .scalar)) (i : Fin (Nat.succ n)) :
    0 < Spec.toVec (Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) t) i := by
  classical
  rcases softmax_vec_spec_normalized t with ⟨weights, hpos, hcoord⟩
  rw [hcoord i]
  exact div_pos (hpos i) (Finset.sum_pos (fun j _ => hpos j) Finset.univ_nonempty)

/-- `softmaxVecSpec` produces a vector whose entries sum to `1` over `ℝ`. -/
theorem sum_spec_softmax_vec_spec {n : Nat}
    (t : Tensor ℝ (.dim (Nat.succ n) .scalar)) :
    Spec.Tensor.sumSpec (Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) t) = 1 := by
  classical
  rcases softmax_vec_spec_normalized t with ⟨weights, hpos, hcoord⟩
  rw [Spec.sum_spec_vec]
  simp_rw [hcoord]
  calc
    (∑ i, weights i / ∑ j, weights j) = (∑ i, weights i) / ∑ j, weights j := by
      simpa using
        (Finset.sum_div (s := (Finset.univ : Finset (Fin (Nat.succ n))))
          (f := weights) (a := ∑ j, weights j)).symm
    _ = 1 := div_self (ne_of_gt (Finset.sum_pos (fun j _ => hpos j) Finset.univ_nonempty))

/-- Every coordinate of a nonempty real softmax vector lies in the closed unit interval. -/
theorem softmax_vec_spec_mem_unitInterval {n : Nat}
    (t : Tensor ℝ (.dim (Nat.succ n) .scalar)) (i : Fin (Nat.succ n)) :
    Spec.toVec (Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) t) i ∈ Set.Icc 0 1 := by
  classical
  let y := Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) t
  have hpos : ∀ j, 0 < Spec.toVec y j := fun j => softmax_vec_spec_pos t j
  have hsum : ∑ j, Spec.toVec y j = 1 := by
    simpa [Spec.sum_spec_vec] using sum_spec_softmax_vec_spec t
  constructor
  · exact le_of_lt (hpos i)
  · calc
      Spec.toVec y i <= ∑ j, Spec.toVec y j :=
        Finset.single_le_sum (fun j _ => le_of_lt (hpos j)) (Finset.mem_univ i)
      _ = 1 := hsum

/-! ## Backward conservation -/

/-- The concrete stable softmax backward is tangent to the probability simplex.

`Activation.softmaxBackwardSpec` is the VJP used by the spec and tape layers. Its coordinate sum is
zero because the stable forward weights sum to one. This statement is about the actual tensor
definition, not the separate analytic `EuclideanSpace` presentation of the same derivative. -/
theorem sum_spec_softmax_backward_spec {n : Nat}
    (x dY : Tensor ℝ (.dim (Nat.succ n) .scalar)) :
    Spec.Tensor.sumSpec
      (Activation.softmaxBackwardSpec (α := ℝ) (s := .dim (Nat.succ n) .scalar) x dY) = 0 := by
  classical
  let y := Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) x
  let s : ℝ := Spec.Tensor.sumSpec (Spec.Tensor.mulSpec dY y)
  have hy : (∑ i, Spec.toVec y i) = 1 := by
    simpa [y, Spec.sum_spec_vec] using sum_spec_softmax_vec_spec x
  have hs : s = ∑ i, Spec.toVec y i * Spec.toVec dY i := by
    rw [show s = Spec.Tensor.sumSpec (Spec.Tensor.mulSpec dY y) by rfl,
      Spec.sum_spec_vec]
    refine Finset.sum_congr rfl ?_
    intro i _
    rw [Spec.toVec_mul_spec]
    ring
  have hsub : ∀ i,
      Spec.toVec
        (Spec.Tensor.subSpec dY (Spec.replicate (Tensor.scalar s))) i =
          Spec.toVec dY i - s := by
    intro i
    cases dY with
    | dim values =>
        cases hvalue : values i with
        | scalar value =>
            simp [Spec.toVec, Spec.Tensor.subSpec, Spec.Tensor.map2Spec, Spec.replicate, hvalue]
  rw [show Activation.softmaxBackwardSpec (α := ℝ) (s := .dim (Nat.succ n) .scalar) x dY =
      Spec.Tensor.mulSpec y
        (Spec.Tensor.subSpec dY (Spec.replicate (Tensor.scalar s))) by
          simp [Activation.softmaxBackwardSpec, y, s]]
  rw [Spec.sum_spec_vec]
  simp_rw [Spec.toVec_mul_spec, hsub]
  calc
    (∑ i, Spec.toVec y i * (Spec.toVec dY i - s)) =
        (∑ i, Spec.toVec y i * Spec.toVec dY i) - s * (∑ i, Spec.toVec y i) := by
      calc
        (∑ i, Spec.toVec y i * (Spec.toVec dY i - s)) =
            ∑ i, (Spec.toVec y i * Spec.toVec dY i - s * Spec.toVec y i) := by
          refine Finset.sum_congr rfl ?_
          intro i _
          ring
        _ = (∑ i, Spec.toVec y i * Spec.toVec dY i) -
            ∑ i, s * Spec.toVec y i := by rw [Finset.sum_sub_distrib]
        _ = (∑ i, Spec.toVec y i * Spec.toVec dY i) -
            s * (∑ i, Spec.toVec y i) := by rw [Finset.mul_sum]
    _ = 0 := by rw [hy, hs]; ring

/-- Coordinatewise bound for the concrete stable softmax VJP.

If every upstream coordinate has magnitude at most `G`, each input-gradient coordinate has
magnitude at most `2G`. The estimate does not grow with the axis length because the softmax output
is a nonnegative vector of total mass one. -/
theorem abs_toVec_softmax_backward_spec_le_two_mul {n : Nat}
    (x dY : Tensor ℝ (.dim (Nat.succ n) .scalar)) (G : ℝ)
    (hdY : ∀ i, |Spec.toVec dY i| <= G) (i : Fin (Nat.succ n)) :
    |Spec.toVec
      (Activation.softmaxBackwardSpec (α := ℝ) (s := .dim (Nat.succ n) .scalar) x dY) i| <=
        2 * G := by
  classical
  let y := Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) x
  let s : ℝ := Spec.Tensor.sumSpec (Spec.Tensor.mulSpec dY y)
  have hyPos : ∀ j, 0 < Spec.toVec y j := by
    intro j
    exact softmax_vec_spec_pos x j
  have hySum : (∑ j, Spec.toVec y j) = 1 := by
    simpa [y, Spec.sum_spec_vec] using sum_spec_softmax_vec_spec x
  have hs : s = ∑ j, Spec.toVec y j * Spec.toVec dY j := by
    rw [show s = Spec.Tensor.sumSpec (Spec.Tensor.mulSpec dY y) by rfl,
      Spec.sum_spec_vec]
    refine Finset.sum_congr rfl ?_
    intro j _
    rw [Spec.toVec_mul_spec]
    ring
  have hsAbs : |s| <= G := by
    rw [hs]
    calc
      |∑ j, Spec.toVec y j * Spec.toVec dY j| <=
          ∑ j, |Spec.toVec y j * Spec.toVec dY j| :=
        Finset.abs_sum_le_sum_abs _ _
      _ = ∑ j, Spec.toVec y j * |Spec.toVec dY j| := by
        refine Finset.sum_congr rfl ?_
        intro j _
        rw [abs_mul, abs_of_pos (hyPos j)]
      _ <= ∑ j, Spec.toVec y j * G := by
        refine Finset.sum_le_sum ?_
        intro j _
        exact mul_le_mul_of_nonneg_left (hdY j) (le_of_lt (hyPos j))
      _ = G := by rw [← Finset.sum_mul, hySum, one_mul]
  have hyLeOne : Spec.toVec y i <= 1 := by
    calc
      Spec.toVec y i <= ∑ j, Spec.toVec y j :=
        Finset.single_le_sum (fun j _ => le_of_lt (hyPos j)) (Finset.mem_univ i)
      _ = 1 := hySum
  have hsub : Spec.toVec
      (Spec.Tensor.subSpec dY (Spec.replicate (Tensor.scalar s))) i =
        Spec.toVec dY i - s := by
    cases dY with
    | dim values =>
        cases hvalue : values i with
        | scalar value =>
            simp [Spec.toVec, Spec.Tensor.subSpec, Spec.Tensor.map2Spec, Spec.replicate, hvalue]
  have hbackward :
      Activation.softmaxBackwardSpec (α := ℝ) (s := .dim (Nat.succ n) .scalar) x dY =
        Spec.Tensor.mulSpec y
          (Spec.Tensor.subSpec dY (Spec.replicate (Tensor.scalar s))) := by
    simp [Activation.softmaxBackwardSpec, y, s]
  have hdiff : |Spec.toVec dY i - s| <= 2 * G := by
    calc
      |Spec.toVec dY i - s| <= |Spec.toVec dY i| + |s| := abs_sub _ _
      _ <= G + G := add_le_add (hdY i) hsAbs
      _ = 2 * G := by ring
  rw [hbackward, Spec.toVec_mul_spec, hsub, abs_mul, abs_of_pos (hyPos i)]
  calc
    Spec.toVec y i * |Spec.toVec dY i - s| <=
        1 * |Spec.toVec dY i - s| :=
      mul_le_mul_of_nonneg_right hyLeOne (abs_nonneg _)
    _ <= 1 * (2 * G) := mul_le_mul_of_nonneg_left hdiff zero_le_one
    _ = 2 * G := one_mul _

/-!
`softmaxSpec` on matrices is rowwise, so each row sums to `1`.

This is the attention-shaped theorem: for score matrices, the key axis is the last/vector axis, and
softmax is applied independently to every query row.
-/
theorem sum_spec_softmax_spec_row {nQ nK : Nat}
    (maskedScores : Tensor ℝ (.dim nQ (.dim (Nat.succ nK) .scalar))) (i : Fin nQ) :
    Spec.Tensor.sumSpec
        (Spec.get (Activation.softmaxSpec (α := ℝ)
          (s := .dim nQ (.dim (Nat.succ nK) .scalar)) maskedScores) i)
      = 1 := by
  cases maskedScores with
  | dim rows =>
      -- `softmax_spec` on a matrix is rowwise, and `get` picks a row.
      simpa [Activation.softmaxSpec, Spec.Tensor.get, Spec.Tensor.getAtSpec] using
        (sum_spec_softmax_vec_spec (t := rows i))

/-!
Convenience row-sum theorem when the key dimension is written as an arbitrary `nK` plus a proof
`nK ≠ 0`.

Many model statements quantify over a natural key length `nK`; this wrapper converts that style
into the `Nat.succ _` shape required by `sum_spec_softmax_spec_row`.
-/
theorem sum_spec_softmax_spec_row_of_ne_zero {nQ nK : Nat} (hK : nK ≠ 0)
    (scores : Tensor ℝ (.dim nQ (.dim nK .scalar))) (i : Fin nQ) :
    Spec.Tensor.sumSpec
        (Spec.get (Activation.softmaxSpec (α := ℝ) (s := .dim nQ (.dim nK .scalar)) scores) i)
      = 1 := by
  cases nK with
  | zero =>
      cases (hK rfl)
  | succ nK' =>
      -- Reduce to the `Nat.succ _` specialization.
      simpa using (sum_spec_softmax_spec_row (nQ := nQ) (nK := nK') (maskedScores := scores) i)

end Proofs
