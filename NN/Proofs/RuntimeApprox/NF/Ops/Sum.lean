/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.Ops.Scalar
public import NN.Proofs.Tensor.Basic.Folds

/-!
# NF Sum Reduction Bounds

Forward-error bounds for rounded sum reductions.  The accumulator carries both the runtime value
and a proof budget, so every addition contributes the incoming element error plus one rounding term.
-/

@[expose] public section

namespace Proofs
namespace RuntimeApprox

open Spec
open Tensor
open NN.MLTheory.Robustness.Spec

noncomputable section

namespace NFBackend

open TorchLean.Floats
open Proofs.RuntimeRoundingApprox

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]
variable {rnd : ℝ → ℤ} [NeuralValidRndToNearest rnd]

local notation "R" => TorchLean.Floats.NF β fexp rnd

-- ---------------------------------------------------------------------------
-- Sum reduction bound (fold with rounded addition)
-- ---------------------------------------------------------------------------

/--
One fold step for `sum_spec` that tracks an explicit forward error budget.

State is `(accR, epsAcc)` where `accR` is the runtime accumulator and `epsAcc` bounds the absolute
error `|toSpec accR - accS|` for the corresponding spec accumulator `accS`. Each step adds:
- the incoming per-element budget `epsElem`;
- one rounding-ULP term for the addition.
-/
def sumStep (epsElem : ℝ) : (R × ℝ) → R → (R × ℝ)
  | (accR, epsAcc), xR =>
      let epsAcc' : ℝ :=
        epsAcc + epsElem +
          neuralUlp β fexp
              (toSpec (β := β) (fexp := fexp) (rnd := rnd) accR +
                toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) / 2
      (accR + xR, epsAcc')

/--
Fold `sum_step` over a tensor via `tensor_foldl_spec`.

This is the shared helper behind `sum_bound` and `approxT_sum_spec`: it simultaneously computes the
runtime sum (in `.1`) and the accumulated error bound (in `.2`).
-/
def sumFoldState {s : Shape} (epsElem : ℝ) (st : R × ℝ) (tR : Tensor R s) : (R × ℝ) :=
  tensorFoldlSpec (sumStep (β := β) (fexp := fexp) (rnd := rnd) epsElem) st tR

/--
Forward absolute-error bound for `sum_spec`.

`sum_bound epsElem tR` is the `.2` component of `sum_fold_state` started at 0, assuming each element
is approximated within `epsElem`. This corresponds to naive sequential summation with a rounding
term added at each step (cf. standard floating-point summation analyses).
-/
def sumBound {s : Shape} (epsElem : ℝ) (tR : Tensor R s) : ℝ :=
  (sumFoldState (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsElem
    ((0 : R), neuralUlp β fexp 0 / 2) tR).2

omit [NeuralValidRndToNearest rnd] in
/--
The accumulator component of `sum_fold_state` matches the plain spec fold.

Informal: `sum_fold_state` only adds bookkeeping to `.2`; `.1` is exactly `tensor_foldl_spec (·+·)`.
-/
private lemma sum_fold_state_fst_eq {s : Shape} (epsElem : ℝ) (st : R × ℝ) (tR : Tensor R s) :
    (sumFoldState (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsElem st tR).1 =
      tensorFoldlSpec (· + ·) st.1 tR := by
  induction s generalizing st with
  | scalar =>
      cases tR with
      | scalar xR =>
          cases st with
          | mk accR epsAcc =>
              simp [sumFoldState, sumStep, tensorFoldlSpec]
  | dim n s ih =>
      cases tR with
      | dim valuesR =>
          -- Compare the `go` loops for the pair-valued fold vs the scalar fold.
          have go_fst :
              ∀ k (st : R × ℝ), k ≤ n →
                (tensorFoldlSpec.go (sumStep (β := β) (fexp := fexp) (rnd := rnd) epsElem) n s
                  valuesR k st).1 =
                  tensorFoldlSpec.go (· + ·) n s valuesR k st.1 := by
            intro k st hk
            induction hn : n - k generalizing k st with
            | zero =>
                have hk' : k = n := by grind
                subst k
                simp [Spec.tensor_foldl_spec_go_of_not_lt]
            | succ m ih_go =>
                have hlt : k < n := by grind
                have hk1 : k + 1 ≤ n := Nat.succ_le_of_lt hlt
                rw [Spec.tensor_foldl_spec_go_of_lt (f := sumStep (β := β) (fexp := fexp) (rnd := rnd) epsElem)
                  (values := valuesR) (k := k) (acc := st) hlt]
                rw [Spec.tensor_foldl_spec_go_of_lt (f := (· + ·)) (values := valuesR) (k := k) (acc := st.1) hlt]
                have h_next : n - (k + 1) = m := by grind
                -- The recursive fold over the sub-tensor updates only the accumulator in `.1`.
                have h_step :
                    (tensorFoldlSpec (sumStep (β := β) (fexp := fexp) (rnd := rnd) epsElem) st
                        (valuesR ⟨k, hlt⟩)).1 =
                      tensorFoldlSpec (· + ·) st.1 (valuesR ⟨k, hlt⟩) := by
                  simpa [sumFoldState] using
                    ih (st := st) (tR := valuesR ⟨k, hlt⟩)
                have := ih_go (k := k + 1)
                  (st := tensorFoldlSpec (sumStep (β := β) (fexp := fexp) (rnd := rnd) epsElem)
                    st
                    (valuesR ⟨k, hlt⟩)) hk1
                simpa [h_next, h_step] using this
          have h0 := go_fst (k := 0) (st := st) (by exact Nat.zero_le n)
          simpa [sumFoldState, tensorFoldlSpec] using h0

/--
Core summation induction: `sum_fold_state` preserves a forward bound.

In words: if the current accumulator `st.1` approximates a spec value `accS` within
  `st.2`,
and each tensor entry is approximated within `epsElem`, then folding `sum_fold_state` over the
  tensor
produces an accumulator whose spec value is within the final `.2` budget of the corresponding spec
fold.
-/
private theorem approx_sum_fold_state {s : Shape} :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {accS : ℝ} {st : R × ℝ} {epsElem : ℝ},
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) st.1 - accS) ≤ st.2 →
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR epsElem →
        abs
            (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                (sumFoldState (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsElem st xR).1 -
              tensorFoldlSpec (· + ·) accS xS) ≤
          (sumFoldState (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsElem st xR).2 := by
  intro xS xR accS st epsElem hAcc hx
  induction s generalizing accS st with
  | scalar =>
      cases xS with
      | scalar x =>
          cases xR with
          | scalar xR =>
              cases st with
              | mk accR epsAcc =>
                  have hx' :=
                    (approxT_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                      rnd))
                      (x := x) (xR := xR) (eps := epsElem)).1 hx
                  have h :=
                    approx_add_nf (β := β) (fexp := fexp) (rnd := rnd)
                      (x := accS) (y := x) (xR := accR) (yR := xR)
                      (epsx := epsAcc) (epsy := epsElem) hAcc hx'
                  simpa [sumFoldState, sumStep, tensorFoldlSpec, add_assoc, add_left_comm,
                    add_comm] using h
  | dim n s ih =>
      cases xS with
      | dim valuesS =>
          cases xR with
          | dim valuesR =>
              -- Prove the accumulator/error invariant for the recursive fold loops.
              have go_sound :
                  ∀ k (accS : ℝ) (st : R × ℝ), k ≤ n →
                    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) st.1 - accS) ≤ st.2 →
                      abs
                          (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                              (tensorFoldlSpec.go
                                  (sumStep (β := β) (fexp := fexp) (rnd := rnd) epsElem) n s
                                    valuesR k st).1 -
                            tensorFoldlSpec.go (· + ·) n s valuesS k accS) ≤
                        (tensorFoldlSpec.go
                            (sumStep (β := β) (fexp := fexp) (rnd := rnd) epsElem) n s valuesR k
                              st).2 := by
                intro k accS st hk hAcc
                induction hn : n - k generalizing k accS st with
                | zero =>
                    have hk' : k = n := by grind
                    subst k
                    simpa [Spec.tensor_foldl_spec_go_of_not_lt] using hAcc
                | succ m ih_go =>
                    have hlt : k < n := by grind
                    have hk1 : k + 1 ≤ n := Nat.succ_le_of_lt hlt
                    rw [Spec.tensor_foldl_spec_go_of_lt (f := sumStep (β := β) (fexp := fexp) (rnd := rnd) epsElem)
                      (values := valuesR) (k := k) (acc := st) hlt]
                    rw [Spec.tensor_foldl_spec_go_of_lt (f := (· + ·)) (values := valuesS) (k := k) (acc := accS) hlt]
                    have h_next : n - (k + 1) = m := by grind
                    -- Apply the shape IH to fold over the current slice `valuesR ⟨k, hlt⟩`.
                    have hx_k :=
                      approxT_dim_get (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                        rnd))
                        (xS := Tensor.dim valuesS) (xR := Tensor.dim valuesR) (eps := epsElem) hx
                          ⟨k, hlt⟩
                    have h_step :=
                      ih (xS := valuesS ⟨k, hlt⟩) (xR := valuesR ⟨k, hlt⟩) (accS := accS) (st := st)
                        hAcc hx_k
                    -- Use IH on the tail of the outer `go`.
                    have htail :=
                      ih_go (k := k + 1)
                        (accS := tensorFoldlSpec (· + ·) accS (valuesS ⟨k, hlt⟩))
                        (st := tensorFoldlSpec
                          (sumStep (β := β) (fexp := fexp) (rnd := rnd) epsElem) st (valuesR ⟨k,
                            hlt⟩))
                        hk1 (by simpa [h_next, sumFoldState] using h_step)
                    simpa [h_next] using htail
              have h0 := go_sound (k := 0) (accS := accS) (st := st) (by exact Nat.zero_le n) hAcc
              simpa [sumFoldState, tensorFoldlSpec] using h0

/--
Forward approximation bound for `sum_spec` over an arbitrary tensor shape.

If `xR` approximates `xS` elementwise within `eps`, then the scalar sums `sum_spec xR` and
`sum_spec xS` differ by at most `sum_bound eps xR`.
-/
theorem approxT_sum_spec {s : Shape} :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ},
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
        approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (Tensor.scalar (sumSpec (α := ℝ) (s := s) xS))
          (Tensor.scalar (sumSpec (α := R) (s := s) xR))
          (sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR) := by
  intro xS xR eps hx
  -- Start from accumulator 0 with a conservative rounding bound.
  let initEps : ℝ := neuralUlp β fexp 0 / 2
  have hAcc : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (0 : R) - (0 : ℝ)) ≤ initEps := by
    have hnonneg : 0 ≤ initEps := by
      exact div_nonneg (neuralUlp.nonneg β fexp 0) (by norm_num)
    simpa [initEps] using hnonneg
  have h :=
    approx_sum_fold_state (β := β) (fexp := fexp) (rnd := rnd) (s := s)
      (xS := xS) (xR := xR) (accS := (0 : ℝ)) (st := ((0 : R), initEps)) (epsElem := eps) hAcc hx
  -- Relate the accumulator component to `sum_spec`.
  have hfst :
      (sumFoldState (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps ((0 : R), initEps) xR).1 =
        sumSpec (α := R) (s := s) xR := by
    simpa [sumSpec] using
      (sum_fold_state_fst_eq (β := β) (fexp := fexp) (rnd := rnd) (s := s) (epsElem := eps)
        (st := ((0 : R), initEps)) (tR := xR))
  -- Wrap back into `approxT` on scalar tensors.
  refine (approxT_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (x := sumSpec (α := ℝ) (s := s) xS) (xR := sumSpec (α := R) (s := s) xR)
      (eps := sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR)).2 ?_
  have h' :
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd) (sumSpec (α := R) (s := s) xR) -
            sumSpec (α := ℝ) (s := s) xS) ≤
        (sumFoldState (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps ((0 : R), initEps) xR).2
          := by
    simpa [hfst, sumSpec] using h
  simpa [sumBound, sumFoldState, initEps] using h'


end NFBackend

end

end RuntimeApprox
end Proofs
