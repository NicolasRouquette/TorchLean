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

set_option maxHeartbeats 12000000

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
    (input : Spec.MultiChannelImage inC inH inW ℝ)
    (dInput : Spec.MultiChannelImage inC inH inW ℝ)
    (δ : Spec.MultiChannelImage outC (outH inH kH stride padding) (outW inW kW stride padding) ℝ) :
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
    -- Commute sums to get the order `oc, oi, oj, ic, di, dj, i, j`.
    have hComm₁ :
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
                        else 0))
          =
        (∑ oc : Fin outC,
          ∑ oi : Fin (outH inH kH stride padding),
            ∑ oj : Fin (outW inW kW stride padding),
              ∑ ic : Fin inC,
                ∑ di : Fin kH,
                  ∑ dj : Fin kW,
                    ∑ i : Fin inH,
                      ∑ j : Fin inW,
                        getAtOrZero dInput [ic.val, i.val, j.val] *
                          (if h :
                              (oi.val * stride + di.val = i.val + padding) ∧
                              (oj.val * stride + dj.val = j.val + padding) then
                            getAtOrZero δ [oc.val, oi.val, oj.val] *
                              getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                          else 0)) := by
      -- Move `oc` outward past `j`, `i`, and `ic`, then move `oi/oj/di/dj` outward past `i/j`
      -- similarly.
      -- Do this by adjacent swaps using `sum_comm`.
      -- Swap `j` with `oc`.
      have h1 :
          (∑ ic : Fin inC, ∑ i : Fin inH, ∑ j : Fin inW, ∑ oc : Fin outC,
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
                          else 0))
            =
          (∑ ic : Fin inC, ∑ i : Fin inH, ∑ oc : Fin outC, ∑ j : Fin inW,
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
        simpa using
          (sum_comm (α := Fin inW) (β := Fin outC)
            (f := fun j oc =>
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
                        else 0)))
      -- Swap `i` with `oc`.
      have h2 :
          (∑ ic : Fin inC, ∑ i : Fin inH, ∑ oc : Fin outC, ∑ j : Fin inW,
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
                        else 0))
            =
          (∑ ic : Fin inC, ∑ oc : Fin outC, ∑ i : Fin inH, ∑ j : Fin inW,
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
        simpa using
          (sum_comm (α := Fin inH) (β := Fin outC)
            (f := fun i oc =>
              ∑ j : Fin inW,
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
                          else 0)))
      -- Swap `ic` with `oc`.
      have h3 :
          (∑ ic : Fin inC, ∑ oc : Fin outC, ∑ i : Fin inH, ∑ j : Fin inW,
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
                        else 0))
            =
          (∑ oc : Fin outC, ∑ ic : Fin inC, ∑ i : Fin inH, ∑ j : Fin inW,
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
        simpa using
          (sum_comm (α := Fin inC) (β := Fin outC)
            (f := fun ic oc =>
              ∑ i : Fin inH,
                ∑ j : Fin inW,
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
                            else 0)))
      -- Now move `oi` outward past `i/j` (and similarly `oj/di/dj`), and reorder to
      -- `oc,oi,oj,ic,di,dj,i,j`.
      -- We do this by applying `sum_comm` repeatedly under `oc` and `ic`.
      have h4 :
          (∑ oc : Fin outC, ∑ ic : Fin inC, ∑ i : Fin inH, ∑ j : Fin inW,
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
                          else 0))
            =
          (∑ oc : Fin outC,
            ∑ oi : Fin (outH inH kH stride padding),
              ∑ oj : Fin (outW inW kW stride padding),
                ∑ ic : Fin inC,
                  ∑ di : Fin kH,
                    ∑ dj : Fin kW,
                      ∑ i : Fin inH,
                        ∑ j : Fin inW,
                          getAtOrZero dInput [ic.val, i.val, j.val] *
                            (if h :
                                (oi.val * stride + di.val = i.val + padding) ∧
                                (oj.val * stride + dj.val = j.val + padding) then
                              getAtOrZero δ [oc.val, oi.val, oj.val] *
                                getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                            else 0)) := by
        refine Fintype.sum_congr _ _ ?_
        intro oc
        -- Swap `oi` outward past `j`, then `i`, then past `ic`.
        -- Step A: under `ic`, swap `j` with `oi` and `i` with `oi`.
        have hA :
            (∑ ic : Fin inC, ∑ i : Fin inH, ∑ j : Fin inW,
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
                            else 0))
              =
            (∑ ic : Fin inC,
              ∑ oi : Fin (outH inH kH stride padding),
                ∑ i : Fin inH,
                  ∑ j : Fin inW,
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
          -- swap `j` and `oi`, then `i` and `oi`
          have hAj :
              (∑ i : Fin inH, ∑ j : Fin inW, ∑ oi : Fin (outH inH kH stride padding),
                    ∑ oj : Fin (outW inW kW stride padding),
                      ∑ di : Fin kH,
                        ∑ dj : Fin kW,
                          getAtOrZero dInput [ic.val, i.val, j.val] *
                            (if h :
                                (oi.val * stride + di.val = i.val + padding) ∧
                                (oj.val * stride + dj.val = j.val + padding) then
                              getAtOrZero δ [oc.val, oi.val, oj.val] *
                                getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                            else 0))
                =
              (∑ i : Fin inH, ∑ oi : Fin (outH inH kH stride padding), ∑ j : Fin inW,
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
            intro i
            simpa using
              (sum_comm (α := Fin inW) (β := Fin (outH inH kH stride padding))
                (f := fun j oi =>
                  ∑ oj : Fin (outW inW kW stride padding),
                    ∑ di : Fin kH,
                      ∑ dj : Fin kW,
                        getAtOrZero dInput [ic.val, i.val, j.val] *
                          (if h :
                              (oi.val * stride + di.val = i.val + padding) ∧
                              (oj.val * stride + dj.val = j.val + padding) then
                            getAtOrZero δ [oc.val, oi.val, oj.val] *
                              getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                          else 0)))
          have hAi :
              (∑ i : Fin inH, ∑ oi : Fin (outH inH kH stride padding), ∑ j : Fin inW,
                    ∑ oj : Fin (outW inW kW stride padding),
                      ∑ di : Fin kH,
                        ∑ dj : Fin kW,
                          getAtOrZero dInput [ic.val, i.val, j.val] *
                            (if h :
                                (oi.val * stride + di.val = i.val + padding) ∧
                                (oj.val * stride + dj.val = j.val + padding) then
                              getAtOrZero δ [oc.val, oi.val, oj.val] *
                                getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                            else 0))
                =
              (∑ oi : Fin (outH inH kH stride padding), ∑ i : Fin inH, ∑ j : Fin inW,
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
            simpa using
              (sum_comm (α := Fin inH) (β := Fin (outH inH kH stride padding))
                (f := fun i oi =>
                  ∑ j : Fin inW,
                    ∑ oj : Fin (outW inW kW stride padding),
                      ∑ di : Fin kH,
                        ∑ dj : Fin kW,
                          getAtOrZero dInput [ic.val, i.val, j.val] *
                            (if h :
                                (oi.val * stride + di.val = i.val + padding) ∧
                                (oj.val * stride + dj.val = j.val + padding) then
                              getAtOrZero δ [oc.val, oi.val, oj.val] *
                                getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                            else 0)))
          exact hAj.trans hAi
        -- Now swap `ic` and `oi`.
        have hB :
            (∑ ic : Fin inC, ∑ oi : Fin (outH inH kH stride padding), ∑ i : Fin inH, ∑ j : Fin inW,
                  ∑ oj : Fin (outW inW kW stride padding),
                    ∑ di : Fin kH,
                      ∑ dj : Fin kW,
                        getAtOrZero dInput [ic.val, i.val, j.val] *
                          (if h :
                              (oi.val * stride + di.val = i.val + padding) ∧
                              (oj.val * stride + dj.val = j.val + padding) then
                            getAtOrZero δ [oc.val, oi.val, oj.val] *
                              getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                          else 0))
              =
            (∑ oi : Fin (outH inH kH stride padding), ∑ ic : Fin inC, ∑ i : Fin inH, ∑ j : Fin inW,
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
          simpa using
            (sum_comm (α := Fin inC) (β := Fin (outH inH kH stride padding))
              (f := fun ic oi =>
                ∑ i : Fin inH,
                  ∑ j : Fin inW,
                    ∑ oj : Fin (outW inW kW stride padding),
                      ∑ di : Fin kH,
                        ∑ dj : Fin kW,
                          getAtOrZero dInput [ic.val, i.val, j.val] *
                            (if h :
                                (oi.val * stride + di.val = i.val + padding) ∧
                                (oj.val * stride + dj.val = j.val + padding) then
                              getAtOrZero δ [oc.val, oi.val, oj.val] *
                                getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                            else 0)))
        -- Now move `oj` outside `ic,i,j` (so it sits after `oi`), then move `di/dj` outside `i/j`.
        have hC :
            (∑ oi : Fin (outH inH kH stride padding),
                ∑ ic : Fin inC,
                  ∑ i : Fin inH,
                    ∑ j : Fin inW,
                      ∑ oj : Fin (outW inW kW stride padding),
                        ∑ di : Fin kH,
                          ∑ dj : Fin kW,
                            getAtOrZero dInput [ic.val, i.val, j.val] *
                              (if h :
                                  (oi.val * stride + di.val = i.val + padding) ∧
                                  (oj.val * stride + dj.val = j.val + padding) then
                                getAtOrZero δ [oc.val, oi.val, oj.val] *
                                  getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                              else 0))
              =
            (∑ oi : Fin (outH inH kH stride padding),
                ∑ oj : Fin (outW inW kW stride padding),
                  ∑ ic : Fin inC,
                    ∑ i : Fin inH,
                      ∑ j : Fin inW,
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
          intro oi
          -- Swap `oj` outward past `j`, then `i`, then `ic`.
          have hC1 :
              (∑ ic : Fin inC, ∑ i : Fin inH, ∑ j : Fin inW,
                    ∑ oj : Fin (outW inW kW stride padding),
                      ∑ di : Fin kH,
                        ∑ dj : Fin kW,
                          getAtOrZero dInput [ic.val, i.val, j.val] *
                            (if h :
                                (oi.val * stride + di.val = i.val + padding) ∧
                                (oj.val * stride + dj.val = j.val + padding) then
                              getAtOrZero δ [oc.val, oi.val, oj.val] *
                                getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                            else 0))
                =
              (∑ ic : Fin inC, ∑ i : Fin inH, ∑ oj : Fin (outW inW kW stride padding), ∑ j : Fin
                inW,
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
            simpa using
              (sum_comm (α := Fin inW) (β := Fin (outW inW kW stride padding))
                (f := fun j oj =>
                  ∑ di : Fin kH,
                    ∑ dj : Fin kW,
                      getAtOrZero dInput [ic.val, i.val, j.val] *
                        (if h :
                            (oi.val * stride + di.val = i.val + padding) ∧
                            (oj.val * stride + dj.val = j.val + padding) then
                          getAtOrZero δ [oc.val, oi.val, oj.val] *
                            getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                        else 0)))
          have hC2 :
              (∑ ic : Fin inC, ∑ i : Fin inH, ∑ oj : Fin (outW inW kW stride padding), ∑ j : Fin
                inW,
                    ∑ di : Fin kH,
                      ∑ dj : Fin kW,
                        getAtOrZero dInput [ic.val, i.val, j.val] *
                          (if h :
                              (oi.val * stride + di.val = i.val + padding) ∧
                              (oj.val * stride + dj.val = j.val + padding) then
                            getAtOrZero δ [oc.val, oi.val, oj.val] *
                              getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                          else 0))
                =
              (∑ ic : Fin inC, ∑ oj : Fin (outW inW kW stride padding), ∑ i : Fin inH, ∑ j : Fin
                inW,
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
            simpa using
              (sum_comm (α := Fin inH) (β := Fin (outW inW kW stride padding))
                (f := fun i oj =>
                  ∑ j : Fin inW,
                    ∑ di : Fin kH,
                      ∑ dj : Fin kW,
                        getAtOrZero dInput [ic.val, i.val, j.val] *
                          (if h :
                              (oi.val * stride + di.val = i.val + padding) ∧
                              (oj.val * stride + dj.val = j.val + padding) then
                            getAtOrZero δ [oc.val, oi.val, oj.val] *
                              getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                          else 0)))
          have hC3 :
              (∑ ic : Fin inC, ∑ oj : Fin (outW inW kW stride padding), ∑ i : Fin inH, ∑ j : Fin
                inW,
                    ∑ di : Fin kH,
                      ∑ dj : Fin kW,
                        getAtOrZero dInput [ic.val, i.val, j.val] *
                          (if h :
                              (oi.val * stride + di.val = i.val + padding) ∧
                              (oj.val * stride + dj.val = j.val + padding) then
                            getAtOrZero δ [oc.val, oi.val, oj.val] *
                              getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                          else 0))
                =
              (∑ oj : Fin (outW inW kW stride padding), ∑ ic : Fin inC, ∑ i : Fin inH, ∑ j : Fin
                inW,
                    ∑ di : Fin kH,
                      ∑ dj : Fin kW,
                        getAtOrZero dInput [ic.val, i.val, j.val] *
                          (if h :
                              (oi.val * stride + di.val = i.val + padding) ∧
                              (oj.val * stride + dj.val = j.val + padding) then
                            getAtOrZero δ [oc.val, oi.val, oj.val] *
                              getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                          else 0)) := by
            simpa using
              (sum_comm (α := Fin inC) (β := Fin (outW inW kW stride padding))
                (f := fun ic oj =>
                  ∑ i : Fin inH,
                    ∑ j : Fin inW,
                      ∑ di : Fin kH,
                        ∑ dj : Fin kW,
                          getAtOrZero dInput [ic.val, i.val, j.val] *
                            (if h :
                                (oi.val * stride + di.val = i.val + padding) ∧
                                (oj.val * stride + dj.val = j.val + padding) then
                              getAtOrZero δ [oc.val, oi.val, oj.val] *
                                getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                            else 0)))
          exact hC1.trans (hC2.trans hC3)

        have hD :
            (∑ oi : Fin (outH inH kH stride padding),
                ∑ oj : Fin (outW inW kW stride padding),
                  ∑ ic : Fin inC,
                    ∑ i : Fin inH,
                      ∑ j : Fin inW,
                        ∑ di : Fin kH,
                          ∑ dj : Fin kW,
                            getAtOrZero dInput [ic.val, i.val, j.val] *
                              (if h :
                                  (oi.val * stride + di.val = i.val + padding) ∧
                                  (oj.val * stride + dj.val = j.val + padding) then
                                getAtOrZero δ [oc.val, oi.val, oj.val] *
                                  getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                              else 0))
              =
            (∑ oi : Fin (outH inH kH stride padding),
                ∑ oj : Fin (outW inW kW stride padding),
                  ∑ ic : Fin inC,
                    ∑ di : Fin kH,
                      ∑ dj : Fin kW,
                        ∑ i : Fin inH,
                          ∑ j : Fin inW,
                            getAtOrZero dInput [ic.val, i.val, j.val] *
                              (if h :
                                  (oi.val * stride + di.val = i.val + padding) ∧
                                  (oj.val * stride + dj.val = j.val + padding) then
                                getAtOrZero δ [oc.val, oi.val, oj.val] *
                                  getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                              else 0)) := by
          refine Fintype.sum_congr _ _ ?_
          intro oi
          refine Fintype.sum_congr _ _ ?_
          intro oj
          refine Fintype.sum_congr _ _ ?_
          intro ic
          -- Move `di` out past `(i,j)`, then move `dj` out past `(i,j)`.
          have hD1 :
              (∑ i : Fin inH, ∑ j : Fin inW, ∑ di : Fin kH, ∑ dj : Fin kW,
                    getAtOrZero dInput [ic.val, i.val, j.val] *
                      (if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0))
                =
              (∑ di : Fin kH, ∑ i : Fin inH, ∑ j : Fin inW, ∑ dj : Fin kW,
                    getAtOrZero dInput [ic.val, i.val, j.val] *
                      (if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0)) := by
            -- swap `j` and `di`, then `i` and `di`
            calc
              (∑ i : Fin inH, ∑ j : Fin inW, ∑ di : Fin kH, ∑ dj : Fin kW,
                    getAtOrZero dInput [ic.val, i.val, j.val] *
                      (if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0))
                  =
                (∑ i : Fin inH, ∑ di : Fin kH, ∑ j : Fin inW, ∑ dj : Fin kW,
                    getAtOrZero dInput [ic.val, i.val, j.val] *
                      (if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0)) := by
                  refine Fintype.sum_congr _ _ ?_
                  intro i
                  simpa using
                    (sum_comm (α := Fin inW) (β := Fin kH)
                      (f := fun j di =>
                        ∑ dj : Fin kW,
                          getAtOrZero dInput [ic.val, i.val, j.val] *
                            (if h :
                                (oi.val * stride + di.val = i.val + padding) ∧
                                (oj.val * stride + dj.val = j.val + padding) then
                              getAtOrZero δ [oc.val, oi.val, oj.val] *
                                getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                            else 0)))
              _ =
                (∑ di : Fin kH, ∑ i : Fin inH, ∑ j : Fin inW, ∑ dj : Fin kW,
                    getAtOrZero dInput [ic.val, i.val, j.val] *
                      (if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0)) := by
                  simpa using
                    (sum_comm (α := Fin inH) (β := Fin kH)
                      (f := fun i di =>
                        ∑ j : Fin inW, ∑ dj : Fin kW,
                          getAtOrZero dInput [ic.val, i.val, j.val] *
                            (if h :
                                (oi.val * stride + di.val = i.val + padding) ∧
                                (oj.val * stride + dj.val = j.val + padding) then
                              getAtOrZero δ [oc.val, oi.val, oj.val] *
                                getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                            else 0)))
          have hD2 :
              (∑ di : Fin kH, ∑ i : Fin inH, ∑ j : Fin inW, ∑ dj : Fin kW,
                    getAtOrZero dInput [ic.val, i.val, j.val] *
                      (if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0))
                =
              (∑ di : Fin kH, ∑ dj : Fin kW, ∑ i : Fin inH, ∑ j : Fin inW,
                    getAtOrZero dInput [ic.val, i.val, j.val] *
                      (if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0)) := by
            refine Fintype.sum_congr _ _ ?_
            intro di
            -- swap `j` and `dj`, then `i` and `dj`
            calc
              (∑ i : Fin inH, ∑ j : Fin inW, ∑ dj : Fin kW,
                    getAtOrZero dInput [ic.val, i.val, j.val] *
                      (if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0))
                  =
                (∑ i : Fin inH, ∑ dj : Fin kW, ∑ j : Fin inW,
                    getAtOrZero dInput [ic.val, i.val, j.val] *
                      (if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0)) := by
                  refine Fintype.sum_congr _ _ ?_
                  intro i
                  simpa using
                    (sum_comm (α := Fin inW) (β := Fin kW)
                      (f := fun j dj =>
                        getAtOrZero dInput [ic.val, i.val, j.val] *
                          (if h :
                              (oi.val * stride + di.val = i.val + padding) ∧
                              (oj.val * stride + dj.val = j.val + padding) then
                            getAtOrZero δ [oc.val, oi.val, oj.val] *
                              getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                          else 0)))
              _ =
                (∑ dj : Fin kW, ∑ i : Fin inH, ∑ j : Fin inW,
                    getAtOrZero dInput [ic.val, i.val, j.val] *
                      (if h :
                          (oi.val * stride + di.val = i.val + padding) ∧
                          (oj.val * stride + dj.val = j.val + padding) then
                        getAtOrZero δ [oc.val, oi.val, oj.val] *
                          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                      else 0)) := by
                  simpa using
                    (sum_comm (α := Fin inH) (β := Fin kW)
                      (f := fun i dj =>
                        ∑ j : Fin inW,
                          getAtOrZero dInput [ic.val, i.val, j.val] *
                            (if h :
                                (oi.val * stride + di.val = i.val + padding) ∧
                                (oj.val * stride + dj.val = j.val + padding) then
                              getAtOrZero δ [oc.val, oi.val, oj.val] *
                                getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                            else 0)))
          exact hD1.trans hD2

        -- Assemble: first use `hA`/`hB` to move `oi`, then `hC` to move `oj`, then `hD` to move
        -- `di/dj`.
        calc
          (∑ ic : Fin inC, ∑ i : Fin inH, ∑ j : Fin inW,
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
                          else 0))
              =
            (∑ oi : Fin (outH inH kH stride padding),
                ∑ ic : Fin inC,
                  ∑ i : Fin inH,
                    ∑ j : Fin inW,
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
                exact hA.trans hB
          _ =
            (∑ oi : Fin (outH inH kH stride padding),
                ∑ oj : Fin (outW inW kW stride padding),
                  ∑ ic : Fin inC,
                    ∑ i : Fin inH,
                      ∑ j : Fin inW,
                        ∑ di : Fin kH,
                          ∑ dj : Fin kW,
                            getAtOrZero dInput [ic.val, i.val, j.val] *
                              (if h :
                                  (oi.val * stride + di.val = i.val + padding) ∧
                                  (oj.val * stride + dj.val = j.val + padding) then
                                getAtOrZero δ [oc.val, oi.val, oj.val] *
                                  getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                              else 0)) := by
                exact hC
          _ =
            (∑ oi : Fin (outH inH kH stride padding),
                ∑ oj : Fin (outW inW kW stride padding),
                  ∑ ic : Fin inC,
                    ∑ di : Fin kH,
                      ∑ dj : Fin kW,
                        ∑ i : Fin inH,
                          ∑ j : Fin inW,
                            getAtOrZero dInput [ic.val, i.val, j.val] *
                              (if h :
                                  (oi.val * stride + di.val = i.val + padding) ∧
                                  (oj.val * stride + dj.val = j.val + padding) then
                                getAtOrZero δ [oc.val, oi.val, oj.val] *
                                  getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                              else 0)) := by
                exact hD
      -- Combine the pieces.
      exact h1.trans (h2.trans (h3.trans h4))
    rw [hComm₁]
    -- Evaluate the inner `(i,j)` sum using `sum_shift_eq_paddedInput`, then simplify to `S`.
    refine (by
      -- Work under the outer sums.
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
      -- Define the constant factor chosen by the indicator.
      let c : ℝ :=
        getAtOrZero δ [oc.val, oi.val, oj.val] * getAtOrZero layer.kernel [oc.val, ic.val,
          di.val, dj.val]
      -- Show the inner sum collapses to a padded read times `c`.
      have hInner :
          (∑ i : Fin inH, ∑ j : Fin inW,
              getAtOrZero dInput [ic.val, i.val, j.val] *
                (if h :
                    (oi.val * stride + di.val = i.val + padding) ∧
                    (oj.val * stride + dj.val = j.val + padding) then
                  getAtOrZero δ [oc.val, oi.val, oj.val] *
                    getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                else 0))
            =
          getAtOrZero
              (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) dInput)
              [ic.val, oi.val * stride + di.val, oj.val * stride + dj.val] * c := by
        -- Rewrite each summand to match `sum_shift_eq_paddedInput`, then factor out `c` and apply
        -- the shift lemma.
        have hSummand :
            (fun (i : Fin inH) (j : Fin inW) =>
                getAtOrZero dInput [ic.val, i.val, j.val] *
                  (if h :
                      (oi.val * stride + di.val = i.val + padding) ∧
                      (oj.val * stride + dj.val = j.val + padding) then
                    getAtOrZero δ [oc.val, oi.val, oj.val] *
                      getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
                  else 0))
              =
            (fun i j =>
                (if (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val = j.val +
                  padding) then
                  getAtOrZero dInput [ic.val, i.val, j.val]
                else 0) * c) := by
          funext i j
          by_cases h :
              (oi.val * stride + di.val = i.val + padding) ∧
              (oj.val * stride + dj.val = j.val + padding)
          · simp [c, h]
          · simp [c, h]
        -- Apply the summand rewrite.
        simp []
        -- Factor `c` out of the inner sums and use `sum_shift_eq_paddedInput` to collapse the
        -- indicator sum.
        -- First collapse `j`, then collapse `i`.
        have hPullJ :
            (∑ i : Fin inH, ∑ j : Fin inW,
                (if (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val = j.val +
                  padding) then
                    getAtOrZero dInput [ic.val, i.val, j.val]
                  else 0) * c)
              =
            (∑ i : Fin inH, ∑ j : Fin inW,
                if (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val = j.val +
                  padding) then
                    getAtOrZero dInput [ic.val, i.val, j.val]
                  else 0) * c := by
          -- Pull `c` out of the `j` sum, then out of the `i` sum (reverse direction of `sum_mul`
          -- twice).
          have hj (i : Fin inH) :
              (∑ j : Fin inW,
                  (if (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val = j.val
                    + padding) then
                      getAtOrZero dInput [ic.val, i.val, j.val]
                    else 0) * c)
                =
              (∑ j : Fin inW,
                  if (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val = j.val
                    + padding) then
                      getAtOrZero dInput [ic.val, i.val, j.val]
                    else 0) * c := by
            simpa using
              (sum_mul (ι := Fin inW)
                (f := fun j =>
                  if (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val = j.val
                    + padding) then
                      getAtOrZero dInput [ic.val, i.val, j.val]
                    else 0)
                (a := c)).symm
          calc
            (∑ i : Fin inH, ∑ j : Fin inW,
                (if (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val = j.val +
                  padding) then
                    getAtOrZero dInput [ic.val, i.val, j.val]
                  else 0) * c)
                =
              (∑ i : Fin inH,
                  (∑ j : Fin inW,
                      if (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val =
                        j.val + padding) then
                          getAtOrZero dInput [ic.val, i.val, j.val]
                        else 0) * c) := by
                refine Fintype.sum_congr _ _ ?_
                intro i
                simpa using hj (i := i)
            _ =
              (∑ i : Fin inH, ∑ j : Fin inW,
                  if (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val = j.val
                    + padding) then
                      getAtOrZero dInput [ic.val, i.val, j.val]
                    else 0) * c := by
                simpa using
                  (sum_mul (ι := Fin inH)
                    (f := fun i =>
                      ∑ j : Fin inW,
                        if (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val =
                          j.val + padding) then
                            getAtOrZero dInput [ic.val, i.val, j.val]
                          else 0)
                    (a := c)).symm
        -- Now apply `sum_shift_eq_paddedInput` to the indicator sum (without `c`) and finish.
        have hShift :
            (∑ i : Fin inH, ∑ j : Fin inW,
                if (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val = j.val +
                  padding) then
                  getAtOrZero dInput [ic.val, i.val, j.val]
                else 0)
              =
            getAtOrZero
                (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) dInput)
                [ic.val, oi.val * stride + di.val, oj.val * stride + dj.val] := by
          -- This is exactly `sum_shift_eq_paddedInput` with `p = oi*stride+di`, `q = oj*stride+dj`.
          simpa [and_left_comm, and_assoc, and_comm] using
            (sum_shift_eq_paddedInput (x := dInput) (ic := ic)
              (p := oi.val * stride + di.val) (q := oj.val * stride + dj.val))
        -- Combine: first rewrite `ite`-products into the form `(..) * c`, then use `hPullJ` and
        -- `hShift`.
        have hIteMul :
            (∑ i : Fin inH, ∑ j : Fin inW,
                if (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val = j.val +
                  padding) then
                  getAtOrZero dInput [ic.val, i.val, j.val] * c
                else 0)
              =
            (∑ i : Fin inH, ∑ j : Fin inW,
                (if (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val = j.val +
                  padding) then
                  getAtOrZero dInput [ic.val, i.val, j.val]
                else 0) * c) := by
          refine Fintype.sum_congr _ _ ?_
          intro i
          refine Fintype.sum_congr _ _ ?_
          intro j
          by_cases h :
              (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val = j.val +
                padding)
          · simp [h]
          · simp [h]
        calc
          (∑ i : Fin inH, ∑ j : Fin inW,
              if (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val = j.val +
                padding) then
                getAtOrZero dInput [ic.val, i.val, j.val] * c
              else 0)
              =
            (∑ i : Fin inH, ∑ j : Fin inW,
                (if (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val = j.val +
                  padding) then
                  getAtOrZero dInput [ic.val, i.val, j.val]
                else 0) * c) := hIteMul
          _ =
            (∑ i : Fin inH, ∑ j : Fin inW,
                if (oi.val * stride + di.val = i.val + padding ∧ oj.val * stride + dj.val = j.val +
                  padding) then
                  getAtOrZero dInput [ic.val, i.val, j.val]
                else 0) * c := hPullJ
          _ =
            getAtOrZero
                (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) dInput)
                [ic.val, oi.val * stride + di.val, oj.val * stride + dj.val] * c := by
              simp [hShift]
      -- Use the collapsed inner sum and simplify to `S`.
      -- Replace `c` and reorder multiplications to match `S`.
      have hReorder :
          getAtOrZero
              (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) dInput)
              [ic.val, oi.val * stride + di.val, oj.val * stride + dj.val] * c
            =
          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val] *
              getAtOrZero
                (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) dInput)
                [ic.val, oi.val * stride + di.val, oj.val * stride + dj.val] *
              getAtOrZero δ [oc.val, oi.val, oj.val] := by
        -- Avoid `simp` with `mul_comm`: it rewrites `Nat` multiplication inside indices.
        dsimp [c]
        -- Let-bind the scalar factors to keep rewrites local.
        set x :=
          getAtOrZero
            (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) dInput)
            [ic.val, oi.val * stride + di.val, oj.val * stride + dj.val]
        set d := getAtOrZero δ [oc.val, oi.val, oj.val]
        set k := getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
        -- Goal becomes `x * (d * k) = k * x * d`.
        calc
          x * (d * k) = x * (k * d) := by
            rw [mul_comm d k]
          _ = (x * k) * d := by
            rw [mul_assoc x k d]
          _ = (k * x) * d := by
            rw [mul_comm x k]
          _ = k * x * d := by
            rw [mul_assoc k x d]
      -- Combine `hInner` with the scalar reordering, then unfold the `S` summand.
      have : (∑ i : Fin inH, ∑ j : Fin inW,
          getAtOrZero dInput [ic.val, i.val, j.val] *
            (if h :
                (oi.val * stride + di.val = i.val + padding) ∧
                (oj.val * stride + dj.val = j.val + padding) then
              getAtOrZero δ [oc.val, oi.val, oj.val] *
                getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
            else 0))
          =
          getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val] *
              getAtOrZero
                (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) dInput)
                [ic.val, oi.val * stride + di.val, oj.val * stride + dj.val] *
              getAtOrZero δ [oc.val, oi.val, oj.val] := by
        exact hInner.trans hReorder
      simpa [S] using this
    )

  exact hLHS.trans hRHS.symm


end

end Conv2D
end Autograd
end Proofs
