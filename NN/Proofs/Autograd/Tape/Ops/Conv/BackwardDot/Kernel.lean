/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Ops.Conv.BackwardDot.Common

/-!
# Conv2d kernel-gradient adjointness

This file proves the inner-product identity for the convolution kernel cotangent.  The result is the
algebraic heart of the kernel-gradient backward rule:

`⟪conv(dKernel, input), δ⟫ = ⟪dKernel, conv2dKernelDeriv(input, δ)⟫`.

The proof is written over finite tensors indexed by `Fin`, so the main work is bookkeeping rather
than analysis: expand the two dot products, commute the finite sums, and factor the candidate
kernel perturbation back out.  Keeping this proof separate from the input-gradient proof makes the
three conv adjointness obligations easier to audit.
-/

@[expose] public section

namespace Proofs
namespace Autograd
namespace Conv2d

open Spec
open Tensor

open scoped BigOperators

noncomputable section


/--
Adjointness of the Conv2d forward map with respect to kernel perturbations.

The left side applies a no-bias convolution whose kernel is `dKernel`, then pairs the output with the
cotangent `δ`.  The right side pairs `dKernel` with the specification-level kernel derivative.  This
is the theorem used by the tape proof to justify the kernel part of Conv2d backpropagation.
-/
lemma dot_conv2d_kernel
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (layer : Spec.Conv2dSpec inC outC kH kW stride padding ℝ h1 h2 h3)
    (input : Spec.Tensor ℝ (.dim inC (.dim inH (.dim inW .scalar))))
    (dKernel : Tensor ℝ (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))))
    (δ : Spec.Tensor ℝ (.dim outC (.dim (outH inH kH stride padding) (.dim (outW inW kW stride padding) .scalar)))) :
    let layerK : Spec.Conv2dSpec inC outC kH kW stride padding ℝ h1 h2 h3 :=
      { kernel := dKernel, bias := fill (0 : ℝ) (.dim outC .scalar) }
    dot (Spec.conv2dSpec (α := ℝ) (layer := layerK) input) δ
      =
    dot dKernel (Spec.conv2dKernelDerivSpec (α := ℝ) (layer := layer) (input := input)
      (grad_output := δ)) := by
  intro layerK
  classical
  -- Expand both dots into explicit sums and show they coincide.
  -- LHS: sum over output indices.
  rw [dot3_eq_sum (a := Spec.conv2dSpec (α := ℝ) (layer := layerK) input) (b := δ)]
  -- Make the binder sizes explicit using the local `outH/outW` abbreviations so we can rewrite with
  -- `hL`.
  change
      (∑ oc : Fin outC, ∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride
        padding),
          getAtOrZero (Spec.conv2dSpec (α := ℝ) (layer := layerK) input) [oc.val, i.val, j.val]
            *
            getAtOrZero δ [oc.val, i.val, j.val])
        =
      dot dKernel (Spec.conv2dKernelDerivSpec (α := ℝ) (layer := layer) (input := input)
        (grad_output := δ))
  -- Rewrite the conv output entries via the no-bias formula.
  have hL :
      (∑ oc : Fin outC, ∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride
        padding),
          getAtOrZero (Spec.conv2dSpec (α := ℝ) (layer := layerK) input) [oc.val, i.val, j.val]
            *
            getAtOrZero δ [oc.val, i.val, j.val])
        =
      (∑ oc : Fin outC, ∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride
        padding),
          (∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW,
              getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                  padding) input)
                  [ic.val, i.val * stride + di.val, j.val * stride + dj.val]) *
            getAtOrZero δ [oc.val, i.val, j.val]) := by
    refine Finset.sum_congr rfl ?_
    intro oc _
    refine Finset.sum_congr rfl ?_
    intro i _
    refine Finset.sum_congr rfl ?_
    intro j _
    -- Avoid `simp` here: it may cancel the common `* δ` factor and introduce a disjunction.
    have hEntry :=
      (by
        simpa [layerK] using
          (conv2d_spec_noBias_get (h1 := h1) (h2 := h2) (h3 := h3)
            (dKernel := dKernel) (input := input) (oc := oc) (i := i) (j := j)))
    -- Rewrite the conv entry, then close by reflexivity.
    simpa [mul_assoc] using congrArg (fun x => x * getAtOrZero δ [oc.val, i.val, j.val]) hEntry
  rw [hL]
  -- Push the output cotangent inside and reassociate the sums.
  -- This makes the expression match the kernel-gradient dot expansion.
  have hReorder :
      (∑ oc : Fin outC, ∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride
        padding),
          (∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW,
              getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                  padding) input)
                  [ic.val, i.val * stride + di.val, j.val * stride + dj.val]) *
            getAtOrZero δ [oc.val, i.val, j.val])
        =
      (∑ oc : Fin outC, ∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW,
          getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
            (∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride padding),
              getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                padding) input)
                [ic.val, i.val * stride + di.val, j.val * stride + dj.val] *
                getAtOrZero δ [oc.val, i.val, j.val])) := by
    classical
    -- Expand to a 6-index sum, commute sums, then factor the kernel entry out of the `(i,j)` sums.
    let G (oc : Fin outC) (i : Fin (outH inH kH stride padding)) (j : Fin (outW inW kW stride
      padding))
        (ic : Fin inC) (di : Fin kH) (dj : Fin kW) : ℝ :=
      (getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
          getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding)
            input)
            [ic.val, i.val * stride + di.val, j.val * stride + dj.val]) *
        getAtOrZero δ [oc.val, i.val, j.val]

    have hMul3
        (oc : Fin outC) (i : Fin (outH inH kH stride padding)) (j : Fin (outW inW kW stride
          padding)) :
        (∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW,
              getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                  padding) input)
                  [ic.val, i.val * stride + di.val, j.val * stride + dj.val]) *
            getAtOrZero δ [oc.val, i.val, j.val]
          =
        ∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW, G oc i j ic di dj := by
      -- Distribute the final `* δ` into the nested sums.
      -- First over `ic`, then `di`, then `dj`.
      have hIc :
          (∑ ic : Fin inC,
                (∑ di : Fin kH, ∑ dj : Fin kW,
                      getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                        getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding
                          := padding) input)
                          [ic.val, i.val * stride + di.val, j.val * stride + dj.val])) *
              getAtOrZero δ [oc.val, i.val, j.val]
            =
          ∑ ic : Fin inC,
              ((∑ di : Fin kH, ∑ dj : Fin kW,
                    getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                      getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                        padding) input)
                        [ic.val, i.val * stride + di.val, j.val * stride + dj.val]) *
                getAtOrZero δ [oc.val, i.val, j.val]) := by
        -- `sum_mul` over `ic`
        simpa [mul_assoc] using
          (sum_mul (ι := Fin inC)
            (f := fun ic =>
              ∑ di : Fin kH, ∑ dj : Fin kW,
                getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                  getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                    padding) input)
                    [ic.val, i.val * stride + di.val, j.val * stride + dj.val])
            (a := getAtOrZero δ [oc.val, i.val, j.val]))
      -- Now distribute inside each `ic`.
      calc
        (∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW,
              getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                  padding) input)
                  [ic.val, i.val * stride + di.val, j.val * stride + dj.val]) *
            getAtOrZero δ [oc.val, i.val, j.val]
            =
          ∑ ic : Fin inC,
              ((∑ di : Fin kH, ∑ dj : Fin kW,
                    getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                      getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                        padding) input)
                        [ic.val, i.val * stride + di.val, j.val * stride + dj.val]) *
                getAtOrZero δ [oc.val, i.val, j.val]) := hIc
        _ =
          ∑ ic : Fin inC, ∑ di : Fin kH,
              ((∑ dj : Fin kW,
                    getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                      getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                        padding) input)
                        [ic.val, i.val * stride + di.val, j.val * stride + dj.val]) *
                getAtOrZero δ [oc.val, i.val, j.val]) := by
          refine Fintype.sum_congr _ _ ?_
          intro ic
          simpa [mul_assoc] using
            (sum_mul (ι := Fin kH)
              (f := fun di =>
                ∑ dj : Fin kW,
                  getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                    getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                      padding) input)
                      [ic.val, i.val * stride + di.val, j.val * stride + dj.val])
              (a := getAtOrZero δ [oc.val, i.val, j.val]))
        _ =
          ∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW,
              (getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                  getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                    padding) input)
                    [ic.val, i.val * stride + di.val, j.val * stride + dj.val]) *
                getAtOrZero δ [oc.val, i.val, j.val] := by
          refine Fintype.sum_congr _ _ ?_
          intro ic
          refine Fintype.sum_congr _ _ ?_
          intro di
          simpa [mul_assoc] using
            (sum_mul (ι := Fin kW)
              (f := fun dj =>
                getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                  getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                    padding) input)
                    [ic.val, i.val * stride + di.val, j.val * stride + dj.val])
              (a := getAtOrZero δ [oc.val, i.val, j.val]))
        _ = ∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW, G oc i j ic di dj := by
          rfl

    -- LHS: rewrite using `hMul3`, then commute sums to match the RHS's ordering.
    have hLHS :
        (∑ oc : Fin outC, ∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride
          padding),
              (∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW,
                    getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                      getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                        padding) input)
                        [ic.val, i.val * stride + di.val, j.val * stride + dj.val]) *
                getAtOrZero δ [oc.val, i.val, j.val])
          =
        (∑ oc : Fin outC, ∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW,
              ∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride padding), G oc i
                j ic di dj) := by
      refine Fintype.sum_congr _ _ ?_
      intro oc
      -- Apply `hMul3` pointwise under the `(i,j)` sums.
      have h1 :
          (∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride padding),
                (∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW,
                      getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                        getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding
                          := padding) input)
                          [ic.val, i.val * stride + di.val, j.val * stride + dj.val]) *
                  getAtOrZero δ [oc.val, i.val, j.val])
            =
          (∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride padding),
                ∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW, G oc i j ic di dj) := by
        refine Fintype.sum_congr _ _ ?_
        intro i
        refine Fintype.sum_congr _ _ ?_
        intro j
        simpa using (hMul3 oc i j)
      -- Commute `(i,j)` past `(ic,di,dj)` using `sum_comm`.
      -- Step 1: swap `j` and `ic`.
      have h2 :
          (∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride padding),
                ∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW, G oc i j ic di dj)
            =
          (∑ i : Fin (outH inH kH stride padding), ∑ ic : Fin inC, ∑ j : Fin (outW inW kW stride
            padding),
                ∑ di : Fin kH, ∑ dj : Fin kW, G oc i j ic di dj) := by
        refine Fintype.sum_congr _ _ ?_
        intro i
        simpa using
          (sum_comm (α := Fin (outW inW kW stride padding)) (β := Fin inC)
            (f := fun j ic => ∑ di : Fin kH, ∑ dj : Fin kW, G oc i j ic di dj))
      -- Step 2: swap `i` and `ic`.
      have h3 :
          (∑ i : Fin (outH inH kH stride padding), ∑ ic : Fin inC, ∑ j : Fin (outW inW kW stride
            padding),
                ∑ di : Fin kH, ∑ dj : Fin kW, G oc i j ic di dj)
            =
          (∑ ic : Fin inC, ∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride
            padding),
                ∑ di : Fin kH, ∑ dj : Fin kW, G oc i j ic di dj) := by
        simpa using
          (sum_comm (α := Fin (outH inH kH stride padding)) (β := Fin inC)
            (f := fun i ic => ∑ j : Fin (outW inW kW stride padding),
              ∑ di : Fin kH, ∑ dj : Fin kW, G oc i j ic di dj))
      -- Step 3: swap `j` with `di`, then `i` with `di`, and similarly for `dj`.
      have h4 :
          (∑ ic : Fin inC, ∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride
            padding),
                ∑ di : Fin kH, ∑ dj : Fin kW, G oc i j ic di dj)
            =
          (∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW,
                ∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride padding), G oc
                  i j ic di dj) := by
        -- Move `di` out past `(i,j)`, then `dj`.
        -- We do this by a fixed sequence of adjacent swaps.
        calc
          (∑ ic : Fin inC, ∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride
            padding),
                ∑ di : Fin kH, ∑ dj : Fin kW, G oc i j ic di dj)
              =
            (∑ ic : Fin inC, ∑ i : Fin (outH inH kH stride padding), ∑ di : Fin kH, ∑ j : Fin (outW
              inW kW stride padding),
                ∑ dj : Fin kW, G oc i j ic di dj) := by
              refine Fintype.sum_congr _ _ ?_
              intro ic
              refine Fintype.sum_congr _ _ ?_
              intro i
              -- swap `j` and `di`
              simpa using
                (sum_comm (α := Fin (outW inW kW stride padding)) (β := Fin kH)
                  (f := fun j di => ∑ dj : Fin kW, G oc i j ic di dj))
          _ =
            (∑ ic : Fin inC, ∑ di : Fin kH, ∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW
              inW kW stride padding),
                ∑ dj : Fin kW, G oc i j ic di dj) := by
              refine Fintype.sum_congr _ _ ?_
              intro ic
              -- swap `i` and `di`
              simpa using
                (sum_comm (α := Fin (outH inH kH stride padding)) (β := Fin kH)
                  (f := fun i di => ∑ j : Fin (outW inW kW stride padding),
                    ∑ dj : Fin kW, G oc i j ic di dj))
          _ =
            (∑ ic : Fin inC, ∑ di : Fin kH, ∑ i : Fin (outH inH kH stride padding), ∑ dj : Fin kW, ∑
              j : Fin (outW inW kW stride padding),
                G oc i j ic di dj) := by
              refine Fintype.sum_congr _ _ ?_
              intro ic
              refine Fintype.sum_congr _ _ ?_
              intro di
              refine Fintype.sum_congr _ _ ?_
              intro i
              -- swap `j` and `dj`
              simpa using
                (sum_comm (α := Fin (outW inW kW stride padding)) (β := Fin kW)
                  (f := fun j dj => G oc i j ic di dj))
          _ =
            (∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW, ∑ i : Fin (outH inH kH stride padding), ∑
              j : Fin (outW inW kW stride padding),
                G oc i j ic di dj) := by
              refine Fintype.sum_congr _ _ ?_
              intro ic
              refine Fintype.sum_congr _ _ ?_
              intro di
              -- swap `i` and `dj`
              simpa using
                (sum_comm (α := Fin (outH inH kH stride padding)) (β := Fin kW)
                  (f := fun i dj => ∑ j : Fin (outW inW kW stride padding), G oc i j ic di dj))
      -- Assemble.
      simp [h1, h2, h3, h4]

    -- RHS: factor kernel entry out of the `(i,j)` sums.
    have hRHS :
        (∑ oc : Fin outC, ∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW,
              getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                (∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride padding),
                  getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                    padding) input)
                    [ic.val, i.val * stride + di.val, j.val * stride + dj.val] *
                    getAtOrZero δ [oc.val, i.val, j.val]))
          =
        (∑ oc : Fin outC, ∑ ic : Fin inC, ∑ di : Fin kH, ∑ dj : Fin kW,
              ∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride padding), G oc i
                j ic di dj) := by
      refine Fintype.sum_congr _ _ ?_
      intro oc
      refine Fintype.sum_congr _ _ ?_
      intro ic
      refine Fintype.sum_congr _ _ ?_
      intro di
      refine Fintype.sum_congr _ _ ?_
      intro dj
      -- Expand the product into the `(i,j)` sums.
      -- First over `i`, then over `j`.
      calc
        getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
            (∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride padding),
              getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                padding) input)
                [ic.val, i.val * stride + di.val, j.val * stride + dj.val] *
                getAtOrZero δ [oc.val, i.val, j.val])
            =
          ∑ i : Fin (outH inH kH stride padding),
              getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                (∑ j : Fin (outW inW kW stride padding),
                  getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                    padding) input)
                    [ic.val, i.val * stride + di.val, j.val * stride + dj.val] *
                    getAtOrZero δ [oc.val, i.val, j.val]) := by
          simpa [mul_assoc] using
            (mul_sum (ι := Fin (outH inH kH stride padding))
              (a := getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val])
              (f := fun i =>
                ∑ j : Fin (outW inW kW stride padding),
                  getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                    padding) input)
                    [ic.val, i.val * stride + di.val, j.val * stride + dj.val] *
                    getAtOrZero δ [oc.val, i.val, j.val]))
        _ =
          ∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride padding),
              (getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                  getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                    padding) input)
                    [ic.val, i.val * stride + di.val, j.val * stride + dj.val]) *
                getAtOrZero δ [oc.val, i.val, j.val] := by
          refine Fintype.sum_congr _ _ ?_
          intro i
          simpa [mul_assoc] using
            (mul_sum (ι := Fin (outW inW kW stride padding))
              (a := getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val])
              (f := fun j =>
                getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding :=
                  padding) input)
                  [ic.val, i.val * stride + di.val, j.val * stride + dj.val] *
                  getAtOrZero δ [oc.val, i.val, j.val]))
        _ = ∑ i : Fin (outH inH kH stride padding), ∑ j : Fin (outW inW kW stride padding), G oc i j
          ic di dj := by
          rfl

    -- Conclude by a common expanded form.
    exact hLHS.trans hRHS.symm
  rw [hReorder]
  -- RHS: expand dot on the 4D kernel.
  rw [dot4_eq_sum (a := dKernel) (b := Spec.conv2dKernelDerivSpec (α := ℝ) (layer := layer)
    (input := input) (grad_output := δ))]
  -- Rewrite each gradient entry and conclude.
  refine Finset.sum_congr rfl ?_
  intro oc _
  refine Finset.sum_congr rfl ?_
  intro ic _
  refine Finset.sum_congr rfl ?_
  intro di _
  refine Finset.sum_congr rfl ?_
  intro dj _
  simp [conv2d_kernel_deriv_get (layer := layer) (input := input) (δ := δ) (oc := oc) (ic := ic) (di
    := di) (dj := dj),
    mul_comm]


end

end Conv2d
end Autograd
end Proofs
