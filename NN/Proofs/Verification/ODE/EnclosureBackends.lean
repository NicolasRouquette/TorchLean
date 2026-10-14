/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Verification.ODE.Enclosure
public import NN.Floats.FP32
public import FloatLib.Floats.Formats.BinaryInterchange.Configured
import Mathlib.Topology.Separation.Connected

/-!
# Backend Views for ODE Enclosures

The generic adapters apply the real-valued corridor theorems to a supplied interpretation
`toReal : α → ℝ`. The `FP32` wrappers use TorchLean's rounded-real model.

Continuity, exact derivatives, and corridor inequalities of the interpreted functions remain
caller hypotheses. Correct rounding or finite-error bounds alone do not supply these hypotheses:
an ordinary sampled floating-point trajectory is not automatically a continuous ODE solution.
These adapters establish neither backend numerical soundness nor a certificate-to-ODE bridge.

A directly decoded binary32 trajectory has finite range, so continuity on an interval forces it
to be constant. To reason about sampled trajectories, use a real-valued interpolant instead.
The corridor theorem still needs the interpolant's clamped-ODE derivative hypothesis; sample
values and interval boxes alone do not establish that it solves the ODE.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)

namespace NN.Proofs.Verification.ODE.Enclosure

open Set

namespace Backend

open TorchLean.Floats

noncomputable section

/-- Interpret a backend-valued trajectory `g : ℝ → α` as a real-valued function via `toReal`. -/
abbrev realView {α : Type} (toReal : α → ℝ) (g : ℝ → α) : ℝ → ℝ :=
  fun t => toReal (g t)

namespace LocalCorridor

/-! ## Local corridor wrapper -/

/--
Generic backend wrapper for the local corridor theorem.

The hypotheses are deliberately phrased over `realView toReal ...`: a caller must already have
translated backend computations into real continuity, derivative, and inequality facts. Given those
facts, the result is the same enclosure + unclamping statement for the backend trajectory's real
interpretation.
-/
theorem fromRealView
    {α : Type} (toReal : α → ℝ)
    {T : ℝ} {f : ℝ → ℝ → ℝ}
    {u uL uU uL' uU' : ℝ → α} {a : ℝ}
    (hu_cont : ContinuousOn (realView toReal u) (Icc 0 T))
    (hu_der :
      ∀ t ∈ Ico 0 T,
        HasDerivWithinAt (realView toReal u)
          (f t (clampToCorridor
            (realView toReal uL) (realView toReal uU) t ((realView toReal u) t)))
            (Ici t) t)
    (hu0 : (realView toReal u) 0 = a)
    (hL_cont : ContinuousOn (realView toReal uL) (Icc 0 T))
    (hL_der :
      ∀ t ∈ Ico 0 T,
        HasDerivWithinAt (realView toReal uL) ((realView toReal uL') t) (Ici t) t)
    (hL_sub : ∀ t ∈ Ico 0 T, (realView toReal uL') t ≤ f t ((realView toReal uL) t))
    (hL0 : (realView toReal uL) 0 ≤ a)
    (hU_cont : ContinuousOn (realView toReal uU) (Icc 0 T))
    (hU_der :
      ∀ t ∈ Ico 0 T,
        HasDerivWithinAt (realView toReal uU) ((realView toReal uU') t) (Ici t) t)
    (hU_sup : ∀ t ∈ Ico 0 T, f t ((realView toReal uU) t) ≤ (realView toReal uU') t)
    (hU0 : a ≤ (realView toReal uU) 0)
    (hLU : ∀ t ∈ Icc 0 T, (realView toReal uL) t ≤ (realView toReal uU) t) :
    (∀ t ∈ Icc 0 T,
        (realView toReal uL) t ≤ (realView toReal u) t ∧
          (realView toReal u) t ≤ (realView toReal uU) t) ∧
      (∀ t ∈ Ico 0 T, HasDerivWithinAt (realView toReal u) (f t ((realView toReal u) t)) (Ici t)
        t) := by
  simpa [realView] using
    (localSolutionEnclosed_fromClampedDynamics (T := T) (f := f)
      (u := realView toReal u)
      (uL := realView toReal uL) (uU := realView toReal uU)
      (uL' := realView toReal uL') (uU' := realView toReal uU') (a := a)
      hu_cont hu_der hu0 hL_cont hL_der hL_sub hL0 hU_cont hU_der hU_sup hU0 hLU)

/-- Specialization of `fromRealView` to TorchLean's proof-level `FP32` model. -/
theorem forFP32
    {T : ℝ} {f : ℝ → ℝ → ℝ}
    {u uL uU uL' uU' : ℝ → FP32} {a : ℝ}
    (hu_cont : ContinuousOn (realView FP32.toReal u) (Icc 0 T))
    (hu_der :
      ∀ t ∈ Ico 0 T,
        HasDerivWithinAt (realView FP32.toReal u)
          (f t (clampToCorridor (realView FP32.toReal uL) (realView FP32.toReal uU) t ((realView
            FP32.toReal u) t)))
          (Ici t) t)
    (hu0 : (realView FP32.toReal u) 0 = a)
    (hL_cont : ContinuousOn (realView FP32.toReal uL) (Icc 0 T))
    (hL_der :
      ∀ t ∈ Ico 0 T,
        HasDerivWithinAt (realView FP32.toReal uL) ((realView FP32.toReal uL') t) (Ici t) t)
    (hL_sub : ∀ t ∈ Ico 0 T, (realView FP32.toReal uL') t ≤ f t ((realView FP32.toReal uL) t))
    (hL0 : (realView FP32.toReal uL) 0 ≤ a)
    (hU_cont : ContinuousOn (realView FP32.toReal uU) (Icc 0 T))
    (hU_der :
      ∀ t ∈ Ico 0 T,
        HasDerivWithinAt (realView FP32.toReal uU) ((realView FP32.toReal uU') t) (Ici t) t)
    (hU_sup : ∀ t ∈ Ico 0 T, f t ((realView FP32.toReal uU) t) ≤ (realView FP32.toReal uU') t)
    (hU0 : a ≤ (realView FP32.toReal uU) 0)
    (hLU : ∀ t ∈ Icc 0 T, (realView FP32.toReal uL) t ≤ (realView FP32.toReal uU) t) :
    (∀ t ∈ Icc 0 T,
        (realView FP32.toReal uL) t ≤ (realView FP32.toReal u) t ∧
          (realView FP32.toReal u) t ≤ (realView FP32.toReal uU) t) ∧
      (∀ t ∈ Ico 0 T,
        HasDerivWithinAt (realView FP32.toReal u) (f t ((realView FP32.toReal u) t)) (Ici t) t)
          := by
  simpa using
    (fromRealView (α := FP32) FP32.toReal (T := T) (f := f)
      (u := u) (uL := uL) (uU := uU) (uL' := uL') (uU' := uU') (a := a)
      hu_cont hu_der hu0 hL_cont hL_der hL_sub hL0 hU_cont hU_der hU_sup hU0 hLU)

end LocalCorridor

namespace ConstantExtension

/-! ## Constant-extension wrapper -/

/--
Generic backend wrapper for the constant-extension theorem.

The corridor is interpreted as real-valued first and then frozen after `T` with
`constantExtensionAfter`. This mirrors the paper's global step while keeping all backend arithmetic
outside this proof.
-/
theorem fromRealView
    {α : Type} (toReal : α → ℝ)
    {T τ : ℝ} (hT : 0 ≤ T) {f : ℝ → ℝ → ℝ}
    {u uL uU uL' uU' : ℝ → α} {a : ℝ}
    (hu_cont : ContinuousOn (realView toReal u) (Icc 0 τ))
    (hu_der : ∀ t ∈ Ico 0 τ,
      HasDerivWithinAt (realView toReal u)
        (f t (clampToCorridor
          (constantExtensionAfter T (realView toReal uL))
          (constantExtensionAfter T (realView toReal uU)) t
          ((realView toReal u) t))) (Ici t) t)
    (hu0 : (realView toReal u) 0 = a)
    (hL_cont : ContinuousOn (realView toReal uL) (Icc 0 T))
    (hL_der :
      ∀ t ∈ Ico 0 T,
        HasDerivWithinAt (realView toReal uL) ((realView toReal uL') t) (Ici t) t)
    (hL_sub : ∀ t ∈ Ico 0 T, (realView toReal uL') t ≤ f t ((realView toReal uL) t))
    (hL0 : (realView toReal uL) 0 ≤ a)
    (hU_cont : ContinuousOn (realView toReal uU) (Icc 0 T))
    (hU_der :
      ∀ t ∈ Ico 0 T,
        HasDerivWithinAt (realView toReal uU) ((realView toReal uU') t) (Ici t) t)
    (hU_sup : ∀ t ∈ Ico 0 T, f t ((realView toReal uU) t) ≤ (realView toReal uU') t)
    (hU0 : a ≤ (realView toReal uU) 0)
    (hLU : ∀ t ∈ Icc 0 T, (realView toReal uL) t ≤ (realView toReal uU) t)
    (hLower :
      ∀ t, T < t →
        0 ≤ f T ((realView toReal uL) T) ∧ f T ((realView toReal uL) T) ≤ f t ((realView
          toReal uL) T))
    (hUpper :
      ∀ t, T < t →
        f t ((realView toReal uU) T) ≤ f T ((realView toReal uU) T) ∧ f T ((realView toReal
          uU) T) ≤ 0) :
    (∀ t ∈ Icc 0 τ,
        constantExtensionAfter T (realView toReal uL) t ≤ (realView toReal u) t ∧
          (realView toReal u) t ≤ constantExtensionAfter T (realView toReal uU) t) ∧
      (∀ t ∈ Ico 0 τ,
        HasDerivWithinAt (realView toReal u) (f t ((realView toReal u) t)) (Ici t) t) := by
  simpa [realView] using
    (extendedSolutionEnclosed_fromClampedDynamics (T := T) (τ := τ) hT (f := f)
      (u := realView toReal u)
      (uL := realView toReal uL) (uU := realView toReal uU)
      (uL' := realView toReal uL') (uU' := realView toReal uU') (a := a)
      hu_cont hu_der hu0 hL_cont hL_der hL_sub hL0 hU_cont hU_der hU_sup hU0 hLU hLower hUpper)

/-- Specialization of `fromRealView` to TorchLean's proof-level `FP32` model. -/
theorem forFP32
    {T τ : ℝ} (hT : 0 ≤ T) {f : ℝ → ℝ → ℝ}
    {u uL uU uL' uU' : ℝ → FP32} {a : ℝ}
    (hu_cont : ContinuousOn (realView FP32.toReal u) (Icc 0 τ))
    (hu_der : ∀ t ∈ Ico 0 τ,
      HasDerivWithinAt (realView FP32.toReal u)
        (f t
          (clampToCorridor
            (constantExtensionAfter T (realView FP32.toReal uL))
            (constantExtensionAfter T (realView FP32.toReal uU)) t
            ((realView FP32.toReal u) t)))
        (Ici t) t)
    (hu0 : (realView FP32.toReal u) 0 = a)
    (hL_cont : ContinuousOn (realView FP32.toReal uL) (Icc 0 T))
    (hL_der :
      ∀ t ∈ Ico 0 T,
        HasDerivWithinAt (realView FP32.toReal uL) ((realView FP32.toReal uL') t) (Ici t) t)
    (hL_sub : ∀ t ∈ Ico 0 T, (realView FP32.toReal uL') t ≤ f t ((realView FP32.toReal uL) t))
    (hL0 : (realView FP32.toReal uL) 0 ≤ a)
    (hU_cont : ContinuousOn (realView FP32.toReal uU) (Icc 0 T))
    (hU_der :
      ∀ t ∈ Ico 0 T,
        HasDerivWithinAt (realView FP32.toReal uU) ((realView FP32.toReal uU') t) (Ici t) t)
    (hU_sup : ∀ t ∈ Ico 0 T, f t ((realView FP32.toReal uU) t) ≤ (realView FP32.toReal uU') t)
    (hU0 : a ≤ (realView FP32.toReal uU) 0)
    (hLU : ∀ t ∈ Icc 0 T, (realView FP32.toReal uL) t ≤ (realView FP32.toReal uU) t)
    (hLower :
      ∀ t, T < t →
        0 ≤ f T ((realView FP32.toReal uL) T) ∧
          f T ((realView FP32.toReal uL) T) ≤ f t ((realView FP32.toReal uL) T))
    (hUpper :
      ∀ t, T < t →
        f t ((realView FP32.toReal uU) T) ≤ f T ((realView FP32.toReal uU) T) ∧
          f T ((realView FP32.toReal uU) T) ≤ 0) :
    (∀ t ∈ Icc 0 τ,
        constantExtensionAfter T (realView FP32.toReal uL) t ≤ (realView FP32.toReal u) t ∧
          (realView FP32.toReal u) t ≤ constantExtensionAfter T (realView FP32.toReal uU) t) ∧
      (∀ t ∈ Ico 0 τ,
        HasDerivWithinAt (realView FP32.toReal u) (f t ((realView FP32.toReal u) t)) (Ici t) t)
          := by
  simpa using
    (fromRealView (α := FP32) FP32.toReal (T := T) (τ := τ) hT (f := f)
      (u := u) (uL := uL) (uU := uU) (uL' := uL') (uU' := uU') (a := a)
      hu_cont hu_der hu0 hL_cont hL_der hL_sub hL0 hU_cont hU_der hU_sup hU0 hLU hLower hUpper)

end ConstantExtension

/-! ## Continuity of a directly decoded binary32 trajectory -/

/--
Real interpretation of an executable IEEE-754 binary32 trajectory.

This view does not establish continuity, derivative identities, or finiteness of the trajectory.
-/
abbrev ieee32RealView (g : ℝ → (ExecFloat.Binary 8 23)) : ℝ → ℝ := fun t =>
  (ExecFloat.Binary.toModel (g t)).toReal

private theorem finite_model_reals
    (format : FloatLib.Floats.Formats.BinaryInterchange.FloatFormat) :
    (Set.range
      (FloatLib.Floats.Formats.BinaryInterchange.Model.toReal (fmt := format))).Finite := by
  let : Finite (FloatLib.Floats.Formats.BinaryInterchange.Model format) :=
    Finite.of_injective (fun x => x.bits.toFin) (by
      intro left right same
      cases left
      cases right
      have same := congrArg BitVec.ofFin same
      simp only [BitVec.ofFin_toFin] at same
      cases same
      rfl)
  exact Set.finite_range _

/-- A directly decoded binary32 trajectory that is continuous on an interval is constant there.
Use a real-valued interpolant when stating an ODE theorem about discrete samples.
-/
theorem ieee32RealView_eq_of_continuousOn
    (u : ℝ → ExecFloat.Binary 8 23) {a b s t : ℝ}
    (continuous : ContinuousOn (ieee32RealView u) (Icc a b))
    (hs : s ∈ Icc a b) (ht : t ∈ Icc a b) :
    ieee32RealView u s = ieee32RealView u t := by
  let f := ieee32RealView u
  have finite : (f '' Icc a b).Finite := (finite_model_reals _).subset (by
    rintro _ ⟨x, _, rfl⟩
    exact ⟨ExecFloat.Binary.toModel (u x), rfl⟩)
  have connected := isPreconnected_Icc.image f continuous
  have singleton : (f '' Icc a b).Subsingleton := by
    by_contra nontrivial
    exact (connected.infinite_of_nontrivial (Set.not_subsingleton_iff.mp nontrivial)).not_finite
      finite
  exact singleton ⟨s, hs, rfl⟩ ⟨t, ht, rfl⟩

end

end Backend

end NN.Proofs.Verification.ODE.Enclosure
