/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Ops.Conv.BackwardDot.Kernel

/-!
# Conv2D input-gradient adjointness

This file proves the inner-product identity for the convolution input cotangent:

`⟪conv(dInput, kernel), δ⟫ = ⟪dInput, conv2dInputDeriv(kernel, δ)⟫`.

The statement is over the specification tensors, not over a backend implementation.  That lets the
runtime proof later use this file as a clean algebraic contract for the input-gradient rule.  Most of
the proof is finite-sum normalization: expand both dot products, rewrite each convolution entry, and
reindex the sums so every contribution to an input pixel is collected in the same place.
-/

@[expose] public section

namespace Proofs
namespace Autograd
namespace Conv2D

open Spec
open Tensor

open scoped BigOperators

noncomputable section

/--
Adjointness of the Conv2D forward map with respect to input perturbations.

The left side perturbs the input and pairs the resulting output perturbation with the output
cotangent `δ`.  The right side pairs that same input perturbation with the specification-level input
derivative.  This is the theorem used by the tape proof to justify the input-gradient part of Conv2D
backpropagation.
-/
lemma dot_conv2d_input
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (layer : Spec.Conv2DSpec inC outC kH kW stride padding ℝ h1 h2 h3)
    (input : Spec.Tensor ℝ (.dim inC (.dim inH (.dim inW .scalar))))
    (dInput : Spec.Tensor ℝ (.dim inC (.dim inH (.dim inW .scalar))))
    (δ : Spec.Tensor ℝ (.dim outC (.dim (outH inH kH stride padding) (.dim (outW inW kW stride padding) .scalar)))) :
    let layer0 : Spec.Conv2DSpec inC outC kH kW stride padding ℝ h1 h2 h3 :=
      { kernel := layer.kernel, bias := fill (0 : ℝ) (.dim outC .scalar) }
    dot (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput) δ
      =
    dot dInput (Spec.conv2dInputDerivSpec (α := ℝ) (layer := layer) (input := input)
      (grad_output := δ)) := by
  intro layer0
  classical

  -- A canonical expanded sum for both sides.
  let S : ℝ :=
    ∑ oc : Fin outC,
      ∑ oi : Fin (outH inH kH stride padding),
        ∑ oj : Fin (outW inW kW stride padding),
          ∑ ic : Fin inC,
            ∑ di : Fin kH,
              ∑ dj : Fin kW,
                (getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]) *
                  (getAtOrZero
                    (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) dInput)
                    [ic.val, oi.val * stride + di.val, oj.val * stride + dj.val]) *
                  (getAtOrZero δ [oc.val, oi.val, oj.val])

  have hLHS :
      dot (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput) δ = S := by
    -- Expand the dot and rewrite each conv entry via `conv2d_spec_noBias_get`.
    rw [dot3_eq_sum (a := Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput) (b := δ)]
    -- Match the binder sizes explicitly.
    change
      (∑ oc : Fin outC, ∑ oi : Fin (outH inH kH stride padding), ∑ oj : Fin (outW inW kW stride
        padding),
          getAtOrZero (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput) [oc.val, oi.val,
            oj.val] *
            getAtOrZero δ [oc.val, oi.val, oj.val])
        = S
    -- Rewrite each conv entry and distribute the final `* δ` into the `(ic,di,dj)` sums.
    refine (by
      refine Fintype.sum_congr _ _ ?_
      intro oc
      refine Fintype.sum_congr _ _ ?_
      intro oi
      refine Fintype.sum_congr _ _ ?_
      intro oj
      have hEntry :=
        (by
          simpa [layer0] using
            (conv2d_spec_noBias_get (h1 := h1) (h2 := h2) (h3 := h3)
              (dKernel := layer.kernel) (input := dInput) (oc := oc) (i := oi) (j := oj)))
      have hEntry' :
          getAtOrZero (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput) [oc.val, oi.val,
            oj.val]
            =
          ∑ ic : Fin inC,
            ∑ di : Fin kH,
              ∑ dj : Fin kW,
                getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val] *
                  getAtOrZero
                    (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) dInput)
                    [ic.val, oi.val * stride + di.val, oj.val * stride + dj.val] := by
        exact hEntry
      -- Multiply by `δ[oc,oi,oj]` and distribute across the nested sums.
      have hMul :
          (getAtOrZero (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInput) [oc.val, oi.val,
            oj.val] *
              getAtOrZero δ [oc.val, oi.val, oj.val])
            =
          ∑ ic : Fin inC,
            ∑ di : Fin kH,
              ∑ dj : Fin kW,
                (getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]) *
                  (getAtOrZero
                    (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) dInput)
                    [ic.val, oi.val * stride + di.val, oj.val * stride + dj.val]) *
                  (getAtOrZero δ [oc.val, oi.val, oj.val]) := by
        rw [hEntry']
        simp_rw [Finset.sum_mul]
      simpa [S, mul_assoc] using hMul)

  have hRHS :
      dot dInput (Spec.conv2dInputDerivSpec (α := ℝ) (layer := layer) (input := input)
        (grad_output := δ)) = S := by
    -- Expand the dot.
    rw [dot3_eq_sum (a := dInput)
      (b := Spec.conv2dInputDerivSpec (α := ℝ) (layer := layer) (input := input) (grad_output :=
        δ))]
    -- Match binder sizes.
    change
      (∑ ic : Fin inC, ∑ i : Fin inH, ∑ j : Fin inW,
          getAtOrZero dInput [ic.val, i.val, j.val] *
            getAtOrZero
              (Spec.conv2dInputDerivSpec (α := ℝ) (layer := layer) (input := input) (grad_output
                := δ))
              [ic.val, i.val, j.val]) = S
    -- Rewrite each gradient entry using `conv2d_input_deriv_get`.
    have hRewrite :
        (∑ ic : Fin inC, ∑ i : Fin inH, ∑ j : Fin inW,
            getAtOrZero dInput [ic.val, i.val, j.val] *
              getAtOrZero
                (Spec.conv2dInputDerivSpec (α := ℝ) (layer := layer) (input := input)
                  (grad_output := δ))
                [ic.val, i.val, j.val])
          =
        (∑ ic : Fin inC, ∑ i : Fin inH, ∑ j : Fin inW,
            getAtOrZero dInput [ic.val, i.val, j.val] *
              (∑ oc : Fin outC,
                ∑ oi : Fin (outH inH kH stride padding),
                  ∑ oj : Fin (outW inW kW stride padding),
                    ∑ di : Fin kH,
                      ∑ dj : Fin kW,
                        if h :
                            (oi.val * stride + di.val = i.val + padding) ∧
                            (oj.val * stride + dj.val = j.val + padding) then
                          getAtOrZero δ [oc.val, oi.val, oj.val] *
                            getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                        else 0)) := by
      refine Fintype.sum_congr _ _ ?_
      intro ic
      refine Fintype.sum_congr _ _ ?_
      intro i
      refine Fintype.sum_congr _ _ ?_
      intro j
      simp [conv2d_input_deriv_get (layer := layer) (input := input) (δ := δ) (ic := ic) (i := i) (j
        := j)]
    -- Distribute the multiplication into the nested sums, then commute sums so that `(i,j)` are the
    -- innermost sums.
    -- Finally, collapse the `(i,j)` sum with `sum_shift_eq_paddedInput`.
    rw [hRewrite]
    -- Push `get_at_or_zero dInput[...]` into the sums.
    have hDist :
        (∑ ic : Fin inC, ∑ i : Fin inH, ∑ j : Fin inW,
            getAtOrZero dInput [ic.val, i.val, j.val] *
              (∑ oc : Fin outC,
                ∑ oi : Fin (outH inH kH stride padding),
                  ∑ oj : Fin (outW inW kW stride padding),
                    ∑ di : Fin kH,
                      ∑ dj : Fin kW,
                        if h :
                            (oi.val * stride + di.val = i.val + padding) ∧
                            (oj.val * stride + dj.val = j.val + padding) then
                          getAtOrZero δ [oc.val, oi.val, oj.val] *
                            getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                        else 0))
          =
        (∑ ic : Fin inC, ∑ i : Fin inH, ∑ j : Fin inW,
            ∑ oc : Fin outC,
              ∑ oi : Fin (outH inH kH stride padding),
                ∑ oj : Fin (outW inW kW stride padding),
                  ∑ di : Fin kH,
                    ∑ dj : Fin kW,
                      getAtOrZero dInput [ic.val, i.val, j.val] *
                        (if h :
                            (oi.val * stride + di.val = i.val + padding) ∧
                            (oj.val * stride + dj.val = j.val + padding) then
                          getAtOrZero δ [oc.val, oi.val, oj.val] *
                            getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                        else 0)) := by
      refine Fintype.sum_congr _ _ ?_
      intro ic
      refine Fintype.sum_congr _ _ ?_
      intro i
      refine Fintype.sum_congr _ _ ?_
      intro j
      -- Distribute across each nested sum using `mul_sum`, one level at a time.
      let a : ℝ := getAtOrZero dInput [ic.val, i.val, j.val]
      calc
        a *
            (∑ oc : Fin outC,
              ∑ oi : Fin (outH inH kH stride padding),
                ∑ oj : Fin (outW inW kW stride padding),
                  ∑ di : Fin kH,
                    ∑ dj : Fin kW,
                      if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0)
            =
          ∑ oc : Fin outC,
            a *
              (∑ oi : Fin (outH inH kH stride padding),
                ∑ oj : Fin (outW inW kW stride padding),
                  ∑ di : Fin kH,
                    ∑ dj : Fin kW,
                      if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0) := by
            simpa [a] using
              (mul_sum (ι := Fin outC) (a := a)
                (f := fun oc =>
                  ∑ oi : Fin (outH inH kH stride padding),
                    ∑ oj : Fin (outW inW kW stride padding),
                      ∑ di : Fin kH,
                        ∑ dj : Fin kW,
                          if h :
                              (oi.val * stride + di.val = i.val + padding) ∧
                              (oj.val * stride + dj.val = j.val + padding) then
                            getAtOrZero δ [oc.val, oi.val, oj.val] *
                              getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                          else 0))
        _ =
          ∑ oc : Fin outC,
            ∑ oi : Fin (outH inH kH stride padding),
              a *
                (∑ oj : Fin (outW inW kW stride padding),
                  ∑ di : Fin kH,
                    ∑ dj : Fin kW,
                      if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0) := by
            refine Fintype.sum_congr _ _ ?_
            intro oc
            simpa [a] using
              (mul_sum (ι := Fin (outH inH kH stride padding)) (a := a)
                (f := fun oi =>
                  ∑ oj : Fin (outW inW kW stride padding),
                    ∑ di : Fin kH,
                      ∑ dj : Fin kW,
                        if h :
                            (oi.val * stride + di.val = i.val + padding) ∧
                            (oj.val * stride + dj.val = j.val + padding) then
                          getAtOrZero δ [oc.val, oi.val, oj.val] *
                            getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                        else 0))
        _ =
          ∑ oc : Fin outC,
            ∑ oi : Fin (outH inH kH stride padding),
              ∑ oj : Fin (outW inW kW stride padding),
                a *
                  (∑ di : Fin kH,
                    ∑ dj : Fin kW,
                      if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0) := by
            refine Fintype.sum_congr _ _ ?_
            intro oc
            refine Fintype.sum_congr _ _ ?_
            intro oi
            simpa [a] using
              (mul_sum (ι := Fin (outW inW kW stride padding)) (a := a)
                (f := fun oj =>
                  ∑ di : Fin kH,
                    ∑ dj : Fin kW,
                      if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0))
        _ =
          ∑ oc : Fin outC,
            ∑ oi : Fin (outH inH kH stride padding),
              ∑ oj : Fin (outW inW kW stride padding),
                ∑ di : Fin kH,
                  a *
                    (∑ dj : Fin kW,
                      if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0) := by
            refine Fintype.sum_congr _ _ ?_
            intro oc
            refine Fintype.sum_congr _ _ ?_
            intro oi
            refine Fintype.sum_congr _ _ ?_
            intro oj
            simpa [a] using
              (mul_sum (ι := Fin kH) (a := a)
                (f := fun di =>
                  ∑ dj : Fin kW,
                    if h :
                        (oi.val * stride + di.val = i.val + padding) ∧
                        (oj.val * stride + dj.val = j.val + padding) then
                      getAtOrZero δ [oc.val, oi.val, oj.val] *
                        getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                    else 0))
        _ =
          ∑ oc : Fin outC,
            ∑ oi : Fin (outH inH kH stride padding),
              ∑ oj : Fin (outW inW kW stride padding),
                ∑ di : Fin kH,
                  ∑ dj : Fin kW,
                    a *
                      (if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0) := by
            refine Fintype.sum_congr _ _ ?_
            intro oc
            refine Fintype.sum_congr _ _ ?_
            intro oi
            refine Fintype.sum_congr _ _ ?_
            intro oj
            refine Fintype.sum_congr _ _ ?_
            intro di
            simpa [a] using
              (mul_sum (ι := Fin kW) (a := a)
                (f := fun dj =>
                  if h :
                      (oi.val * stride + di.val = i.val + padding) ∧
                      (oj.val * stride + dj.val = j.val + padding) then
                    getAtOrZero δ [oc.val, oi.val, oj.val] *
                      getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                  else 0))
        _ =
          (∑ oc : Fin outC,
              ∑ oi : Fin (outH inH kH stride padding),
                ∑ oj : Fin (outW inW kW stride padding),
                  ∑ di : Fin kH,
                    ∑ dj : Fin kW,
                      getAtOrZero dInput [ic.val, i.val, j.val] *
                        (if h :
                            (oi.val * stride + di.val = i.val + padding) ∧
                            (oj.val * stride + dj.val = j.val + padding) then
                          getAtOrZero δ [oc.val, oi.val, oj.val] *
                            getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                        else 0)) := by
            simp [a]
    rw [hDist]
    -- Put output coordinates first, then collapse the unique matching input coordinate.
    rw [sum_eight_conv_input_order]
    dsimp only [S]
    refine Fintype.sum_congr _ _ ?_
    intro oc
    refine Fintype.sum_congr _ _ ?_
    intro oi
    refine Fintype.sum_congr _ _ ?_
    intro oj
    refine Fintype.sum_congr _ _ ?_
    intro ic
    refine Fintype.sum_congr _ _ ?_
    intro di
    refine Fintype.sum_congr _ _ ?_
    intro dj
    let d := getAtOrZero δ [oc.val, oi.val, oj.val]
    let k := getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
    calc
      (∑ i : Fin inH, ∑ j : Fin inW,
          getAtOrZero dInput [ic.val, i.val, j.val] *
            (if
                oi.val * stride + di.val = i.val + padding ∧
                oj.val * stride + dj.val = j.val + padding then
              d * k
            else 0)) =
        getAtOrZero
            (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) dInput)
            [ic.val, oi.val * stride + di.val, oj.val * stride + dj.val] * (d * k) := by
          exact sum_shift_mul_eq_paddedInput_mul
            (x := dInput) (ic := ic)
            (p := oi.val * stride + di.val) (q := oj.val * stride + dj.val) (c := d * k)
      _ = k *
          getAtOrZero
            (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) dInput)
            [ic.val, oi.val * stride + di.val, oj.val * stride + dj.val] * d := by
          ring

  exact hLHS.trans hRHS.symm


end

end Conv2D
end Autograd
end Proofs
