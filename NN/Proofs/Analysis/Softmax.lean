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

- `getScalar_le_maxVecSpec` and `exists_getScalar_eq_maxVecSpec`: the shared stable-shift maximum is an
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
- `abs_getScalar_softmax_backward_spec_le_two_mul`: bounded upstream coordinates give a
  dimension-independent coordinate bound for that VJP;
- `sum_spec_softmax_spec_row`: axis-`1` matrix softmax has rows summing to `1` when the key
  dimension is nonempty.

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
theorem getScalar_le_maxVecSpec {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) (i : Fin (Nat.succ n)) :
    Spec.Tensor.getScalar t i <= Tensor.item (Activation.maxVecSpec t) := by
  cases t with
  | dim values =>
      change Tensor.item (values i) <= _
      change Tensor.item (values i) <=
        (List.finRange (Nat.succ n)).foldl
          (fun acc j => max acc (Tensor.item (values j)))
          (Tensor.item (values ⟨0, Nat.succ_pos n⟩))
      exact List.le_foldl_max_of_mem (List.finRange (Nat.succ n))
        (fun j => Tensor.item (values j))
        (acc := Tensor.item (values ⟨0, Nat.succ_pos n⟩)) (i := i)
        (List.mem_finRange i)

/-- The maximum used by stable softmax is attained by an input coordinate. -/
theorem exists_getScalar_eq_maxVecSpec {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) :
    ∃ i, Spec.Tensor.getScalar t i = Tensor.item (Activation.maxVecSpec t) := by
  cases t with
  | dim values =>
      let firstIndex : Fin (Nat.succ n) := ⟨0, Nat.succ_pos n⟩
      let value : Fin (Nat.succ n) -> ℝ := fun i => Tensor.item (values i)
      change ∃ i, value i =
        (List.finRange (Nat.succ n)).foldl (fun acc j => max acc (value j))
          (value firstIndex)
      rcases List.foldl_max_eq_init_or_mem (List.finRange (Nat.succ n)) value
          (value firstIndex) with hfirst | ⟨i, hi, hvalue⟩
      · exact ⟨firstIndex, hfirst.symm⟩
      · exact ⟨i, hvalue.symm⟩

/-- Every logit shifted by the implementation's maximum is nonpositive. -/
theorem softmax_shift_nonpos {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) (i : Fin (Nat.succ n)) :
    Spec.Tensor.getScalar
      (Spec.Tensor.subSpec t (Spec.replicate (Activation.maxVecSpec t))) i <= 0 := by
  cases t with
  | dim values =>
      cases hvalue : values i with
      | scalar value =>
          cases hmaxEq : Activation.maxVecSpec (Tensor.dim values) with
          | scalar maximum =>
              have hle := getScalar_le_maxVecSpec (t := Tensor.dim values) i
              have : value <= maximum := by
                simpa [Spec.Tensor.getScalar, hvalue, hmaxEq] using hle
              simpa [Spec.Tensor.getScalar, Spec.Tensor.subSpec, Spec.Tensor.map2Spec, Spec.replicate, hvalue,
                hmaxEq] using sub_nonpos.mpr this

/-- At least one max-shifted logit is exactly zero. -/
theorem exists_softmax_shift_eq_zero {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) :
    ∃ i, Spec.Tensor.getScalar
      (Spec.Tensor.subSpec t (Spec.replicate (Activation.maxVecSpec t))) i = 0 := by
  rcases exists_getScalar_eq_maxVecSpec t with ⟨i, hi⟩
  refine ⟨i, ?_⟩
  cases t with
  | dim values =>
      cases hvalue : values i with
      | scalar value =>
          cases hmax : Activation.maxVecSpec (Tensor.dim values) with
          | scalar maximum =>
              have hi' : value = maximum := by
                simpa [Spec.Tensor.getScalar, hvalue, hmax] using hi
              simpa [Spec.Tensor.getScalar, Spec.Tensor.subSpec, Spec.Tensor.map2Spec, Spec.replicate,
                hvalue, hmax] using sub_eq_zero.mpr hi'

/-- Exponentiating a max-shifted real logit produces a value at most one. This is the central
overflow-prevention fact behind stable softmax. -/
theorem softmax_shift_exp_le_one {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) (i : Fin (Nat.succ n)) :
    Spec.Tensor.getScalar (Activation.maxShiftedExpVecSpec t) i <= 1 := by
  have hshift := softmax_shift_nonpos t i
  cases t with
  | dim values =>
      cases hvalue : values i with
      | scalar value =>
          cases hmax : Activation.maxVecSpec (Tensor.dim values) with
          | scalar maximum =>
              have hshift' : value - maximum <= 0 := by
                simpa [Spec.Tensor.getScalar, Spec.Tensor.subSpec, Spec.Tensor.map2Spec, Spec.replicate,
                  hvalue, hmax] using hshift
              simpa [Spec.Tensor.getScalar, Spec.Tensor.expSpec, Spec.Tensor.subSpec, Spec.Tensor.mapSpec,
                Spec.Tensor.map2Spec, Spec.replicate, Activation.maxShiftedExpVecSpec, hvalue,
                hmax, mathfunc_exp_eq_rexp] using
                (Real.exp_le_one_iff.mpr hshift')

/-- Max-shifted real exponentials remain strictly positive. -/
theorem softmax_shift_exp_pos {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) (i : Fin (Nat.succ n)) :
    0 < Spec.Tensor.getScalar (Activation.maxShiftedExpVecSpec t) i := by
  cases t with
  | dim values =>
      cases hvalue : values i with
      | scalar value =>
          cases hmax : Activation.maxVecSpec (Tensor.dim values) with
          | scalar maximum =>
              simpa [Activation.maxShiftedExpVecSpec, Spec.Tensor.getScalar, Spec.Tensor.expSpec,
                Spec.Tensor.subSpec, Spec.Tensor.mapSpec, Spec.Tensor.map2Spec, Spec.replicate,
                hvalue, hmax, mathfunc_exp_eq_rexp] using Real.exp_pos (value - maximum)

/-- One max-shifted exponential is exactly one, because the maximum is attained. -/
theorem exists_softmax_shift_exp_eq_one {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) :
    ∃ i, Spec.Tensor.getScalar (Activation.maxShiftedExpVecSpec t) i = 1 := by
  rcases exists_softmax_shift_eq_zero t with ⟨i, hzero⟩
  refine ⟨i, ?_⟩
  cases t with
  | dim values =>
      cases hvalue : values i with
      | scalar value =>
          cases hmax : Activation.maxVecSpec (Tensor.dim values) with
          | scalar maximum =>
              have hzero' : value - maximum = 0 := by
                simpa [Spec.Tensor.getScalar, Spec.Tensor.subSpec, Spec.Tensor.map2Spec, Spec.replicate,
                  hvalue, hmax] using hzero
              simpa [Activation.maxShiftedExpVecSpec, Spec.Tensor.getScalar, Spec.Tensor.expSpec,
                Spec.Tensor.subSpec, Spec.Tensor.mapSpec, Spec.Tensor.map2Spec, Spec.replicate,
                hvalue, hmax, mathfunc_exp_eq_rexp] using (Real.exp_eq_one_iff _).mpr hzero'

/-- The stable softmax denominator lies in `[1,n]` for a nonempty vector of length `n`.

The lower bound rules out division by zero. The upper bound follows because every shifted
exponential is at most one. Together with `softmax_shift_exp_le_one`, this makes overflow
prevention an explicit theorem of the max-shifted implementation rather than an empirical claim.
-/
theorem softmax_shift_denom_bounds {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) :
    1 <= Spec.Tensor.sumSpec (Activation.maxShiftedExpVecSpec t) ∧
      Spec.Tensor.sumSpec (Activation.maxShiftedExpVecSpec t) <= Nat.succ n := by
  classical
  let ex := Activation.maxShiftedExpVecSpec t
  have hpos : ∀ i, 0 <= Spec.Tensor.getScalar ex i := fun i => le_of_lt (softmax_shift_exp_pos t i)
  have hle : ∀ i, Spec.Tensor.getScalar ex i <= 1 := fun i => softmax_shift_exp_le_one t i
  rcases exists_softmax_shift_exp_eq_one t with ⟨witness, hwitness⟩
  rw [Spec.sum_spec_vec]
  constructor
  · calc
      1 = Spec.Tensor.getScalar ex witness := hwitness.symm
      _ <= ∑ i, Spec.Tensor.getScalar ex i :=
        Finset.single_le_sum (fun i _ => hpos i) (Finset.mem_univ witness)
  · calc
      (∑ i, Spec.Tensor.getScalar ex i) <= ∑ _i : Fin (Nat.succ n), (1 : ℝ) := by
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
    (t : Tensor ℝ [Nat.succ n]) :
    ∃ weights : Fin (Nat.succ n) → ℝ,
      (∀ i, 0 < weights i) ∧
      ∀ i,
        Spec.Tensor.getScalar (Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) t) i =
          weights i / ∑ j, weights j := by
  classical
  cases t with
  | dim f =>
      let input : Tensor ℝ [Nat.succ n] := Tensor.dim f
      let maxT : Tensor ℝ .scalar := Activation.maxVecSpec input
      let shifted := Spec.Tensor.subSpec input (Spec.replicate maxT)
      let exponentials := Spec.Tensor.expSpec shifted
      let weights : Fin (Nat.succ n) → ℝ := fun j => Spec.Tensor.getScalar exponentials j
      have hweightsPos : ∀ j, 0 < weights j := by
        intro j
        simpa [weights, exponentials, Spec.Tensor.expSpec, Spec.Tensor.mapSpec] using
          (show 0 < MathFunctions.exp (Spec.Tensor.getScalar shifted j) by
            simpa [mathfunc_exp_eq_rexp] using Real.exp_pos (Spec.Tensor.getScalar shifted j))
      let denom : ℝ := Spec.Tensor.sumSpec exponentials
      have hdenom : denom = ∑ j : Fin (Nat.succ n), weights j := by
        simpa [denom, weights] using Spec.sum_spec_vec exponentials
      have hcoord : ∀ i : Fin (Nat.succ n),
          Spec.Tensor.getScalar (Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) (Tensor.dim f)) i =
            weights i / denom := by
        intro i
        change Spec.Tensor.getScalar
            (Spec.Tensor.divSpec exponentials (Spec.replicate (Tensor.scalar denom))) i =
          Spec.Tensor.getScalar exponentials i / denom
        cases exponentials with
        | dim values =>
            cases hvalue : values i with
            | scalar value =>
                simp [Spec.Tensor.getScalar, Spec.Tensor.divSpec, Spec.Tensor.map2Spec, Spec.replicate,
                  hvalue]
      refine ⟨weights, hweightsPos, ?_⟩
      intro i
      simpa [hdenom] using hcoord i

/-- Coordinate equation for the concrete stable vector softmax.

This is the small unfolding lemma that downstream algebraic proofs should use. It exposes the
max-shifted numerator and its tensor sum while hiding the implementation chosen for tensor
reduction. -/
theorem getScalar_softmaxVecSpec {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) (i : Fin (Nat.succ n)) :
    Spec.Tensor.getScalar (Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) t) i =
      Spec.Tensor.getScalar (Activation.maxShiftedExpVecSpec t) i /
        Spec.Tensor.sumSpec (Activation.maxShiftedExpVecSpec t) := by
  cases t with
  | dim values =>
      let exponentials := Activation.maxShiftedExpVecSpec (Tensor.dim values)
      change Spec.Tensor.getScalar
          (Spec.Tensor.divSpec exponentials
            (Spec.replicate (Tensor.scalar (Spec.Tensor.sumSpec exponentials)))) i = _
      cases hEx : exponentials with
      | dim exValues =>
          cases hvalue : exValues i with
          | scalar value =>
              simp [Spec.Tensor.getScalar, Spec.Tensor.divSpec, Spec.Tensor.map2Spec, Spec.replicate,
                exponentials, hEx, hvalue]

/-! ## Probability-simplex properties -/

/-- Every coordinate of a nonempty real softmax vector is strictly positive. -/
theorem softmax_vec_spec_pos {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) (i : Fin (Nat.succ n)) :
    0 < Spec.Tensor.getScalar (Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) t) i := by
  classical
  rcases softmax_vec_spec_normalized t with ⟨weights, hpos, hcoord⟩
  rw [hcoord i]
  exact div_pos (hpos i) (Finset.sum_pos (fun j _ => hpos j) Finset.univ_nonempty)

/-- `softmaxVecSpec` produces a vector whose entries sum to `1` over `ℝ`. -/
theorem sum_spec_softmax_vec_spec {n : Nat}
    (t : Tensor ℝ [Nat.succ n]) :
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
    (t : Tensor ℝ [Nat.succ n]) (i : Fin (Nat.succ n)) :
    Spec.Tensor.getScalar (Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) t) i ∈ Set.Icc 0 1 := by
  classical
  let y := Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) t
  have hpos : ∀ j, 0 < Spec.Tensor.getScalar y j := fun j => softmax_vec_spec_pos t j
  have hsum : ∑ j, Spec.Tensor.getScalar y j = 1 := by
    simpa [Spec.sum_spec_vec] using sum_spec_softmax_vec_spec t
  constructor
  · exact le_of_lt (hpos i)
  · calc
      Spec.Tensor.getScalar y i <= ∑ j, Spec.Tensor.getScalar y j :=
        Finset.single_le_sum (fun j _ => le_of_lt (hpos j)) (Finset.mem_univ i)
      _ = 1 := hsum

/-! ## Backward conservation -/

/-- The concrete stable softmax backward is tangent to the probability simplex.

`Activation.softmaxBackwardSpec 0` is the vector VJP used by the spec and tape layers. Its
coordinate sum is zero because the stable forward weights sum to one. This statement is about the
actual tensor definition, not the separate analytic `EuclideanSpace` presentation of the same
derivative. -/
theorem sum_spec_softmax_backward_spec {n : Nat}
    (x dY : Tensor ℝ [Nat.succ n]) :
    Spec.Tensor.sumSpec
      (Activation.softmaxBackwardSpec (α := ℝ) (s := [Nat.succ n]) 0 x dY) = 0 := by
  change Spec.Tensor.sumSpec
    (Activation.Internal.softmaxInnermostBackwardSpec x dY) = 0
  classical
  let y := Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) x
  let s : ℝ := Spec.Tensor.sumSpec (Spec.Tensor.mulSpec dY y)
  have hy : (∑ i, Spec.Tensor.getScalar y i) = 1 := by
    simpa [y, Spec.sum_spec_vec] using sum_spec_softmax_vec_spec x
  have hs : s = ∑ i, Spec.Tensor.getScalar y i * Spec.Tensor.getScalar dY i := by
    rw [show s = Spec.Tensor.sumSpec (Spec.Tensor.mulSpec dY y) by rfl,
      Spec.sum_spec_vec]
    refine Finset.sum_congr rfl ?_
    intro i _
    rw [Spec.getScalar_mul_spec]
    ring
  have hsub : ∀ i,
      Spec.Tensor.getScalar
        (Spec.Tensor.subSpec dY (Spec.replicate (Tensor.scalar s))) i =
          Spec.Tensor.getScalar dY i - s := by
    intro i
    cases dY with
    | dim values =>
        cases hvalue : values i with
        | scalar value =>
            simp [Spec.Tensor.getScalar, Spec.Tensor.subSpec, Spec.Tensor.map2Spec, Spec.replicate, hvalue]
  rw [show Activation.Internal.softmaxInnermostBackwardSpec x dY =
      Spec.Tensor.mulSpec y
        (Spec.Tensor.subSpec dY (Spec.replicate (Tensor.scalar s))) by
          simp [Activation.Internal.softmaxInnermostBackwardSpec, y, s]]
  rw [Spec.sum_spec_vec]
  simp_rw [Spec.getScalar_mul_spec, hsub]
  calc
    (∑ i, Spec.Tensor.getScalar y i * (Spec.Tensor.getScalar dY i - s)) =
        (∑ i, Spec.Tensor.getScalar y i * Spec.Tensor.getScalar dY i) - s * (∑ i, Spec.Tensor.getScalar y i) := by
      calc
        (∑ i, Spec.Tensor.getScalar y i * (Spec.Tensor.getScalar dY i - s)) =
            ∑ i, (Spec.Tensor.getScalar y i * Spec.Tensor.getScalar dY i - s * Spec.Tensor.getScalar y i) := by
          refine Finset.sum_congr rfl ?_
          intro i _
          ring
        _ = (∑ i, Spec.Tensor.getScalar y i * Spec.Tensor.getScalar dY i) -
            ∑ i, s * Spec.Tensor.getScalar y i := by rw [Finset.sum_sub_distrib]
        _ = (∑ i, Spec.Tensor.getScalar y i * Spec.Tensor.getScalar dY i) -
            s * (∑ i, Spec.Tensor.getScalar y i) := by rw [Finset.mul_sum]
    _ = 0 := by rw [hy, hs]; ring

/-- Coordinatewise bound for the concrete stable softmax VJP.

If every upstream coordinate has magnitude at most `G`, each input-gradient coordinate has
magnitude at most `2G`. The estimate does not grow with the axis length because the softmax output
is a nonnegative vector of total mass one. -/
theorem abs_getScalar_softmax_backward_spec_le_two_mul {n : Nat}
    (x dY : Tensor ℝ [Nat.succ n]) (G : ℝ)
    (hdY : ∀ i, |Spec.Tensor.getScalar dY i| <= G) (i : Fin (Nat.succ n)) :
    |Spec.Tensor.getScalar
      (Activation.softmaxBackwardSpec (α := ℝ) (s := [Nat.succ n]) 0 x dY) i| <=
        2 * G := by
  change |Spec.Tensor.getScalar
    (Activation.Internal.softmaxInnermostBackwardSpec x dY) i| <= 2 * G
  classical
  let y := Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) x
  let s : ℝ := Spec.Tensor.sumSpec (Spec.Tensor.mulSpec dY y)
  have hyPos : ∀ j, 0 < Spec.Tensor.getScalar y j := by
    intro j
    exact softmax_vec_spec_pos x j
  have hySum : (∑ j, Spec.Tensor.getScalar y j) = 1 := by
    simpa [y, Spec.sum_spec_vec] using sum_spec_softmax_vec_spec x
  have hs : s = ∑ j, Spec.Tensor.getScalar y j * Spec.Tensor.getScalar dY j := by
    rw [show s = Spec.Tensor.sumSpec (Spec.Tensor.mulSpec dY y) by rfl,
      Spec.sum_spec_vec]
    refine Finset.sum_congr rfl ?_
    intro j _
    rw [Spec.getScalar_mul_spec]
    ring
  have hsAbs : |s| <= G := by
    rw [hs]
    calc
      |∑ j, Spec.Tensor.getScalar y j * Spec.Tensor.getScalar dY j| <=
          ∑ j, |Spec.Tensor.getScalar y j * Spec.Tensor.getScalar dY j| :=
        Finset.abs_sum_le_sum_abs _ _
      _ = ∑ j, Spec.Tensor.getScalar y j * |Spec.Tensor.getScalar dY j| := by
        refine Finset.sum_congr rfl ?_
        intro j _
        rw [abs_mul, abs_of_pos (hyPos j)]
      _ <= ∑ j, Spec.Tensor.getScalar y j * G := by
        refine Finset.sum_le_sum ?_
        intro j _
        exact mul_le_mul_of_nonneg_left (hdY j) (le_of_lt (hyPos j))
      _ = G := by rw [← Finset.sum_mul, hySum, one_mul]
  have hyLeOne : Spec.Tensor.getScalar y i <= 1 := by
    calc
      Spec.Tensor.getScalar y i <= ∑ j, Spec.Tensor.getScalar y j :=
        Finset.single_le_sum (fun j _ => le_of_lt (hyPos j)) (Finset.mem_univ i)
      _ = 1 := hySum
  have hsub : Spec.Tensor.getScalar
      (Spec.Tensor.subSpec dY (Spec.replicate (Tensor.scalar s))) i =
        Spec.Tensor.getScalar dY i - s := by
    cases dY with
    | dim values =>
        cases hvalue : values i with
        | scalar value =>
            simp [Spec.Tensor.getScalar, Spec.Tensor.subSpec, Spec.Tensor.map2Spec, Spec.replicate, hvalue]
  have hbackward :
      Activation.Internal.softmaxInnermostBackwardSpec x dY =
        Spec.Tensor.mulSpec y
          (Spec.Tensor.subSpec dY (Spec.replicate (Tensor.scalar s))) := by
    simp [Activation.Internal.softmaxInnermostBackwardSpec, y, s]
  have hdiff : |Spec.Tensor.getScalar dY i - s| <= 2 * G := by
    calc
      |Spec.Tensor.getScalar dY i - s| <= |Spec.Tensor.getScalar dY i| + |s| := abs_sub _ _
      _ <= G + G := add_le_add (hdY i) hsAbs
      _ = 2 * G := by ring
  rw [hbackward, Spec.getScalar_mul_spec, hsub, abs_mul, abs_of_pos (hyPos i)]
  calc
    Spec.Tensor.getScalar y i * |Spec.Tensor.getScalar dY i - s| <=
        1 * |Spec.Tensor.getScalar dY i - s| :=
      mul_le_mul_of_nonneg_right hyLeOne (abs_nonneg _)
    _ <= 1 * (2 * G) := mul_le_mul_of_nonneg_left hdiff zero_le_one
    _ = 2 * G := one_mul _

/-!
Axis-`1` softmax on matrices is rowwise, so each row sums to `1`.

This is the attention-shaped theorem: for score matrices, the key axis is the last/vector axis, and
softmax is applied independently to every query row.
-/
theorem sum_spec_softmax_spec_row {nQ nK : Nat} (hK : nK ≠ 0)
    (scores : Tensor ℝ [nQ, nK]) (i : Fin nQ) :
    Spec.Tensor.sumSpec
        (Spec.get (Activation.softmaxSpec (α := ℝ) (s := [nQ, nK]) 1 scores) i)
      = 1 := by
  cases nK with
  | zero => exact (hK rfl).elim
  | succ nK' =>
      change Spec.Tensor.sumSpec
        (Spec.get (Activation.Internal.softmaxInnermostSpec scores) i) = 1
      cases scores with
      | dim rows =>
          simpa [Activation.Internal.softmaxInnermostSpec, Spec.Tensor.get,
            Spec.Tensor.get] using
            (sum_spec_softmax_vec_spec (t := rows i))

end Proofs
