/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.Ops.Elementwise.Softmax

/-!
# NF Elementwise Bounds: Binary Arithmetic
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

omit [NeuralValidRndToNearest rnd] in
/--
`approxTensor` bound for elementwise addition (`add_spec`) over arbitrary tensor shapes.

The output epsilon is computed as `linf_norm (add_bound_tensor epsx epsy xR yR)`, which combines the
input epsilons and one rounding-ULP term per element.
-/
theorem approxTensor_add_spec {s : Shape} [NeuralValidRndToNearest rnd] :
    ∀ {xS yS : SpecTensor s} {xR yR : Tensor R s} {epsx epsy : ℝ},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR epsx →
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) yS yR epsy →
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (addSpec xS yS) (addSpec xR yR)
          (linfNorm (addBoundTensor (β := β) (fexp := fexp) epsx epsy xR yR)) := by
  intro xS yS xR yR epsx epsy hx hy
  induction s with
  | scalar =>
      cases xS with
      | scalar x =>
          cases yS with
          | scalar y =>
              cases xR with
              | scalar xR =>
                  cases yR with
                  | scalar yR =>
                      -- Reduce to the scalar rounding lemma and wrap back into `approxTensor`.
                      have hx' :=
                        (approxTensor_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd
                          := rnd))
                          (x := x) (xR := xR) (eps := epsx)).1 hx
                      have hy' :=
                        (approxTensor_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd
                          := rnd))
                          (x := y) (xR := yR) (eps := epsy)).1 hy
                      have hxy := approx_add_nf (β := β) (fexp := fexp) (rnd := rnd) (x := x) (y :=
                        y) hx' hy'
                      have hle :
                          abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (xR + yR) - (x + y)) ≤
                            linfNorm
                              (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                                (s := Shape.scalar) epsx epsy (Tensor.scalar xR) (Tensor.scalar yR))
                                  := by
                        -- The RHS is `abs` of the scalar bound; widen using `le_abs_self`.
                        refine le_trans hxy ?_
                        -- `linf_norm` of the scalar bound tensor is `abs` of its scalar entry.
                        simpa [addBoundTensor, tensorToSpec, Spec.Tensor.map, map2Spec,
                          MathFunctions.abs,
                          linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm] using
                          (le_abs_self (epsx + epsy +
                            neuralUlp β fexp
                              (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR +
                                toSpec (β := β) (fexp := fexp) (rnd := rnd) yR) / 2))
                      exact
                        (approxTensor_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd
                          := rnd))
                          (x := x + y) (xR := xR + yR)
                          (eps := linfNorm
                            (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                              (s := Shape.scalar) epsx epsy (Tensor.scalar xR) (Tensor.scalar
                                yR)))).2 (by
                                simpa using hle)
  | dim n s ih =>
      cases xS with
      | dim xSf =>
          cases yS with
          | dim ySf =>
              cases xR with
              | dim xRf =>
                  cases yR with
                  | dim yRf =>
                      -- Let `B` be the global bound (max over component bounds).
                      let B : ℝ :=
                        linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                          (s := Shape.dim n s) epsx epsy (Tensor.dim xRf) (Tensor.dim yRf))
                      have hB_nonneg : 0 ≤ B := by
                        simpa [B] using (linf_norm_nonneg
                          (t := addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                            (s := Shape.dim n s) epsx epsy (Tensor.dim xRf) (Tensor.dim yRf)))

                      -- Show each component output distance is ≤ B, then take the max fold.
                      have hcomp :
                          ∀ i : Fin n,
                            tensorDistance (α := SpecScalar) linfNorm
                                (addSpec (xSf i) (ySf i))
                                (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp)
                                  (rnd := rnd))
                                  (addSpec (xRf i) (yRf i)))
                              ≤ B := by
                        intro i
                        -- Project input approximations to the component.
                        have hx_i :=
                          approxTensor_dim_get (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                            rnd))
                            (xS := Tensor.dim xSf) (xR := Tensor.dim xRf) (eps := epsx) hx i
                        have hy_i :=
                          approxTensor_dim_get (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                            rnd))
                            (xS := Tensor.dim ySf) (xR := Tensor.dim yRf) (eps := epsy) hy i

                        have hih :=
                          ih (xS := xSf i) (yS := ySf i) (xR := xRf i) (yR := yRf i) hx_i hy_i

                        -- The component bound is ≤ the global bound `B`.
                        have hB_ge :
                            linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                              (s := s) epsx epsy (xRf i) (yRf i)) ≤ B := by
                          -- `add_bound_tensor` is shape-preserving, so this is a
                          -- max-over-components inequality.
                          simpa [B, addBoundTensor, tensorToSpec, Spec.Tensor.map, map2Spec]
                            using
                            (linf_norm_le_get_dim
                              (t :=
                                addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                                  (s := Shape.dim n s) epsx epsy (Tensor.dim xRf) (Tensor.dim yRf))
                              i)

                        -- Convert the IH (an `approxTensor` statement) into a `tensor_distance`
                        -- inequality and weaken the bound.
                        have hdist : tensorDistance (α := SpecScalar) linfNorm
                            (addSpec (xSf i) (ySf i))
                            (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd
                              := rnd))
                              (addSpec (xRf i) (yRf i)))
                          ≤ linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                              (s := s) epsx epsy (xRf i) (yRf i)) := by
                          simpa [approxTensor, approxWith] using hih
                        exact le_trans hdist hB_ge

                      -- Fold the component distances with `max` and bound by `B`.
                      have : tensorDistance (α := SpecScalar) linfNorm
                          (Tensor.dim fun i => addSpec (xSf i) (ySf i))
                          (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                            rnd))
                            (Tensor.dim fun i => addSpec (xRf i) (yRf i)))
                        ≤ B := by
                        -- Unfold `tensor_distance` on `.dim`: it becomes a `foldl max` over
                        -- component distances.
                        have hf :
                            ∀ i ∈ List.finRange n,
                              tensorDistance (α := SpecScalar) linfNorm
                                  (addSpec (xSf i) (ySf i))
                                  (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp)
                                    (rnd := rnd))
                                    (addSpec (xRf i) (yRf i)))
                                ≤ B := by
                          intro i _hi
                          exact hcomp i
                        -- Apply the list lemma.
                        have hfold :=
                          List.foldl_max_le_of_le (List.finRange n)
                            (fun i =>
                              tensorDistance (α := SpecScalar) linfNorm
                                (addSpec (xSf i) (ySf i))
                                (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp)
                                  (rnd := rnd))
                                  (addSpec (xRf i) (yRf i))))
                            (acc := (0 : ℝ)) (eps := B) hB_nonneg hf
                        -- Rewrite back into the `tensorDistance` fold over the outer dimension.
                        simp [tensorDistance,
                          linfNorm, RuntimeApprox.linfNorm, tensorToSpec]
                        change
                          List.foldl
                            (fun a i =>
                              max a
                                (tensorLinfNorm
                                  (((xSf i).addSpec (ySf i)).subSpec
                                    (Tensor.map (toSpec (β := β) (fexp := fexp) (rnd := rnd))
                                      ((xRf i).addSpec (yRf i))))))
                            0 (List.finRange n) ≤ B
                        simpa [tensorDistance, linfNorm, RuntimeApprox.linfNorm, tensorToSpec,
                          MathFunctions.abs, Spec.Tensor.map] using hfold

                      -- Conclude `approxTensor` for the whole tensor.
                      simpa [approxTensor, approxWith, B, addSpec, map2Spec] using this

/--
`approxTensor` bound for elementwise subtraction (`sub_spec`) over arbitrary tensor shapes.

This is obtained by lifting the scalar subtraction bound `approx_sub_nf` via
`approxTensor_map2_spec_of_scalar_bound`.
-/
theorem approxTensor_sub_spec {s : Shape} :
    ∀ {xS yS : SpecTensor s} {xR yR : Tensor R s} {epsx epsy : ℝ},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR epsx →
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) yS yR epsy →
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (subSpec xS yS) (subSpec xR yR)
          (linfNorm (subBoundTensor (β := β) (fexp := fexp) epsx epsy xR yR)) := by
  intro xS yS xR yR epsx epsy hx hy
  have h :=
    approxTensor_map2_spec_of_scalar_bound (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
      rnd))
      (s := s)
      (fS := fun a b => a - b) (fR := fun a b => a - b)
      (bnd := fun a b epsx epsy =>
        epsx + epsy + neuralUlp β fexp (a - b) / 2)
      (xS := xS) (yS := yS) (xR := xR) (yR := yR) (epsx := epsx) (epsy := epsy)
      hx hy (by
        intro x y xR yR hx hy
        simpa using
          (approx_sub_nf (β := β) (fexp := fexp) (rnd := rnd) (x := x) (y := y) (xR := xR) (yR :=
            yR) hx hy))
  simpa [subSpec, subBoundTensor] using h

omit [NeuralValidRndToNearest rnd] in
/--
`approxTensor` bound for elementwise multiplication (`mul_spec`) over arbitrary tensor shapes.

The scalar core is `approx_mul_nf`, lifted componentwise; the resulting bound is packaged as
`mul_bound_tensor` and reduced with `linf_norm`.
-/
theorem approxTensor_mul_spec {s : Shape} [NeuralValidRndToNearest rnd] :
    ∀ {xS yS : SpecTensor s} {xR yR : Tensor R s} {epsx epsy : ℝ},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR epsx →
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) yS yR epsy →
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (mulSpec xS yS) (mulSpec xR yR)
          (linfNorm (mulBoundTensor (β := β) (fexp := fexp) epsx epsy xR yR)) := by
  intro xS yS xR yR epsx epsy hx hy
  induction s with
  | scalar =>
      cases xS with
      | scalar x =>
          cases yS with
          | scalar y =>
              cases xR with
              | scalar xR =>
                  cases yR with
                  | scalar yR =>
                      have hx' :=
                        (approxTensor_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd
                          := rnd))
                          (x := x) (xR := xR) (eps := epsx)).1 hx
                      have hy' :=
                        (approxTensor_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd
                          := rnd))
                          (x := y) (xR := yR) (eps := epsy)).1 hy
                      have hxy := approx_mul_nf (β := β) (fexp := fexp) (rnd := rnd) (x := x) (y :=
                        y) hx' hy'
                      have hle :
                          abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (xR * yR) - (x * y)) ≤
                            linfNorm
                              (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                                (s := Shape.scalar) epsx epsy (Tensor.scalar xR) (Tensor.scalar yR))
                                  := by
                        refine le_trans hxy ?_
                        simpa [mulBoundTensor, tensorToSpec, Spec.Tensor.map, map2Spec,
                          MathFunctions.abs,
                          linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm] using
                          (le_abs_self
                            ((abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) + epsx) * epsy +
                              (abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR) + epsy) * epsx +
                              neuralUlp β fexp
                                  (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR *
                                    toSpec (β := β) (fexp := fexp) (rnd := rnd) yR) / 2))
                      exact
                        (approxTensor_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd
                          := rnd))
                          (x := x * y) (xR := xR * yR)
                          (eps := linfNorm
                            (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                              (s := Shape.scalar) epsx epsy (Tensor.scalar xR) (Tensor.scalar
                                yR)))).2 (by
                                simpa using hle)
  | dim n s ih =>
      cases xS with
      | dim xSf =>
          cases yS with
          | dim ySf =>
              cases xR with
              | dim xRf =>
                  cases yR with
                  | dim yRf =>
                      let B : ℝ :=
                        linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                          (s := Shape.dim n s) epsx epsy (Tensor.dim xRf) (Tensor.dim yRf))
                      have hB_nonneg : 0 ≤ B := by
                        simpa [B] using (linf_norm_nonneg
                          (t := mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                            (s := Shape.dim n s) epsx epsy (Tensor.dim xRf) (Tensor.dim yRf)))

                      have hcomp :
                          ∀ i : Fin n,
                            tensorDistance (α := SpecScalar) linfNorm
                                (mulSpec (xSf i) (ySf i))
                                (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp)
                                  (rnd := rnd))
                                  (mulSpec (xRf i) (yRf i)))
                              ≤ B := by
                        intro i
                        have hx_i :=
                          approxTensor_dim_get (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                            rnd))
                            (xS := Tensor.dim xSf) (xR := Tensor.dim xRf) (eps := epsx) hx i
                        have hy_i :=
                          approxTensor_dim_get (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                            rnd))
                            (xS := Tensor.dim ySf) (xR := Tensor.dim yRf) (eps := epsy) hy i

                        have hih :=
                          ih (xS := xSf i) (yS := ySf i) (xR := xRf i) (yR := yRf i) hx_i hy_i

                        have hB_ge :
                            linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                              (s := s) epsx epsy (xRf i) (yRf i)) ≤ B := by
                          simpa [B, mulBoundTensor, tensorToSpec, Spec.Tensor.map, map2Spec]
                            using
                            (linf_norm_le_get_dim
                              (t :=
                                mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                                  (s := Shape.dim n s) epsx epsy (Tensor.dim xRf) (Tensor.dim yRf))
                              i)

                        have hdist : tensorDistance (α := SpecScalar) linfNorm
                            (mulSpec (xSf i) (ySf i))
                            (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd
                              := rnd))
                              (mulSpec (xRf i) (yRf i)))
                          ≤ linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                              (s := s) epsx epsy (xRf i) (yRf i)) := by
                          simpa [approxTensor, approxWith] using hih
                        exact le_trans hdist hB_ge

                      have : tensorDistance (α := SpecScalar) linfNorm
                          (Tensor.dim fun i => mulSpec (xSf i) (ySf i))
                          (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                            rnd))
                            (Tensor.dim fun i => mulSpec (xRf i) (yRf i)))
                        ≤ B := by
                        have hf :
                            ∀ i ∈ List.finRange n,
                              tensorDistance (α := SpecScalar) linfNorm
                                  (mulSpec (xSf i) (ySf i))
                                  (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp)
                                    (rnd := rnd))
                                    (mulSpec (xRf i) (yRf i)))
                                ≤ B := by
                          intro i _hi
                          exact hcomp i
                        have hfold :=
                          List.foldl_max_le_of_le (List.finRange n)
                            (fun i =>
                              tensorDistance (α := SpecScalar) linfNorm
                                (mulSpec (xSf i) (ySf i))
                                (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp)
                                  (rnd := rnd))
                                  (mulSpec (xRf i) (yRf i))))
                            (acc := (0 : ℝ)) (eps := B) hB_nonneg hf
                        simp [tensorDistance,
                          linfNorm, RuntimeApprox.linfNorm, tensorToSpec]
                        change
                          List.foldl
                            (fun a i =>
                              max a
                                (tensorLinfNorm
                                  (((xSf i).mulSpec (ySf i)).subSpec
                                    (Tensor.map (toSpec (β := β) (fexp := fexp) (rnd := rnd))
                                      ((xRf i).mulSpec (yRf i))))))
                            0 (List.finRange n) ≤ B
                        simpa [tensorDistance, linfNorm, RuntimeApprox.linfNorm, tensorToSpec,
                          MathFunctions.abs, Spec.Tensor.map] using hfold

                      simpa [approxTensor, approxWith, B, mulSpec, map2Spec] using this

/-- Squaring is the diagonal specialization of elementwise multiplication, so it uses the same
rounded multiplication bound rather than a separate numerical rule. -/
theorem approxTensor_square_spec {s : Shape} :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (squareSpec xS) (squareSpec xR)
          (linfNorm (mulBoundTensor (β := β) (fexp := fexp) eps eps xR xR)) := by
  intro xS xR eps hx
  have h := approxTensor_mul_spec
    (β := β) (fexp := fexp) (rnd := rnd) hx hx
  simpa only [squareSpec_eq_mulSpec] using h
end NFBackend

end

end RuntimeApprox
end Proofs
