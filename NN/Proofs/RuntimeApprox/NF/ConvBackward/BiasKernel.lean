/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.ConvBackward.Common

/-!
# NeuralFloat Conv2d Bias/Kernel Backward Bounds

This file proves pointwise NeuralFloat approximation bounds for the Conv2d bias and kernel
gradients. The proofs replay the same summation structure used by the runtime backward pass and
accumulate an explicit error budget for each output coordinate.
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

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]
variable {rnd : ℝ → ℤ} [NeuralValidRndToNearest rnd]

local notation "R" => TorchLean.Floats.NF β fexp rnd


/--
Pointwise error bound for the Conv2d **bias** gradient (NF runtime vs spec).

This bound is a replay of the bias-gradient summation with per-term error `epsδ` coming from the
`grad_output` approximation hypothesis.
-/
def conv2dBiasPointBound
    {outC kH kW stride padding inH inW : Nat}
    (δR : Spec.Tensor R (.dim outC (.dim (conv2dOutH inH kH stride padding) (.dim (conv2dOutW inW kW stride
      padding) .scalar))))
    (epsδ : ℝ)
    (out_ch : Fin outC) : ℝ :=
  let out_h := conv2dOutH inH kH stride padding
  let out_w := conv2dOutW inW kW stride padding
  let idxs : List (Fin out_h × Fin out_w) :=
    (List.finRange out_h).flatMap (fun i => (List.finRange out_w).map (fun j => (i, j)))
  let termR : (Fin out_h × Fin out_w) → R := fun t =>
    getAtOrZero δR [out_ch.val, t.1.val, t.2.val]
  let epsTerm : (Fin out_h × Fin out_w) → ℝ := fun _ => epsδ
  (foldAddState (β := β) (fexp := fexp) (rnd := rnd) idxs termR epsTerm).2

/--
Soundness of the Conv2d **bias**-gradient pointwise bound.

Given `approxTensor` for `grad_output`, this shows the spec bias-gradient entry is approximated by the
NF runtime entry within `conv2dBiasPointBound`.
-/
theorem approx_conv2d_bias_point
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    {kernelS : Tensor ℝ (.dim outC (.dim inC (.dim kH (.dim kW .scalar))))}
    {kernelR : Tensor R (.dim outC (.dim inC (.dim kH (.dim kW .scalar))))}
    {biasS : Tensor ℝ (.dim outC .scalar)} {biasR : Tensor R (.dim outC .scalar)}
    {inputS : Spec.Tensor ℝ (.dim inC (.dim inH (.dim inW .scalar)))}
    {inputR : Spec.Tensor R (.dim inC (.dim inH (.dim inW .scalar)))}
    {δS : Spec.Tensor ℝ (.dim outC (.dim (conv2dOutH inH kH stride padding) (.dim (conv2dOutW inW kW stride
      padding) .scalar)))}
    {δR : Spec.Tensor R (.dim outC (.dim (conv2dOutH inH kH stride padding) (.dim (conv2dOutW inW kW stride
      padding) .scalar)))}
    {epsδ : ℝ}
    (hδ : approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) δS δR epsδ)
    (out_ch : Fin outC) :
    let layerS : Spec.Conv2dSpec inC outC kH kW stride padding ℝ h1 h2 h3 :=
      { kernel := kernelS, bias := biasS }
    let layerR : Spec.Conv2dSpec inC outC kH kW stride padding R h1 h2 h3 :=
      { kernel := kernelR, bias := biasR }
    abs
        (toSpec (β := β) (fexp := fexp) (rnd := rnd)
            (getAtOrZero (Spec.conv2dBiasDerivSpec (α := R) (layer := layerR) (input :=
              inputR) (grad_output := δR))
              [out_ch.val]) -
            getAtOrZero (Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layerS) (input :=
              inputS) (grad_output := δS))
                [out_ch.val]) ≤
        conv2dBiasPointBound (β := β) (fexp := fexp) (rnd := rnd)
          (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding) (inH := inH)
            (inW := inW)
          δR epsδ out_ch := by
  intro layerS layerR
  classical
  let out_h := conv2dOutH inH kH stride padding
  let out_w := conv2dOutW inW kW stride padding
  let idxs : List (Fin out_h × Fin out_w) :=
    (List.finRange out_h).flatMap (fun i => (List.finRange out_w).map (fun j => (i, j)))
  let termR : (Fin out_h × Fin out_w) → R := fun t =>
    getAtOrZero δR [out_ch.val, t.1.val, t.2.val]
  let termS : (Fin out_h × Fin out_w) → ℝ := fun t =>
    getAtOrZero δS [out_ch.val, t.1.val, t.2.val]
  let epsTerm : (Fin out_h × Fin out_w) → ℝ := fun _ => epsδ
  let sumR : R := idxs.foldl (fun acc t => acc + termR t) 0
  let sumS : ℝ := idxs.foldl (fun acc t => acc + termS t) 0
  let sumEps : ℝ := (foldAddState (β := β) (fexp := fexp) (rnd := rnd) idxs termR epsTerm).2

  have hTermIdx : ∀ t ∈ idxs, abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (termR t) - termS t)
    ≤ epsTerm t := by
    intro t _ht
    rcases t with ⟨i, j⟩
    simpa [termR, termS, epsTerm] using
      (approx_get_at_or_zero (β := β) (fexp := fexp) (rnd := rnd)
        (s := .dim outC (.dim out_h (.dim out_w .scalar)))
        (xS := δS) (xR := δR) (eps := epsδ) hδ [out_ch.val, i.val, j.val])

  have hSum :
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) sumR - sumS) ≤ sumEps := by
    simpa [sumR, sumS, sumEps] using
      (approx_fold_add (β := β) (fexp := fexp) (rnd := rnd)
        (l := idxs) (termS := termS) (termR := termR) (epsTerm := epsTerm) hTermIdx)

  have hsumR_nested :
      (List.finRange out_h).foldl (fun acc i =>
          (List.finRange out_w).foldl (fun acc j => acc + termR (i, j)) acc) 0 = sumR := by
    let fR : R → (Fin out_h × Fin out_w) → R := fun acc t => acc + termR t
    have hW :
        ∀ (acc : R) (i : Fin out_h),
          List.foldl fR acc ((List.finRange out_w).map (fun j => (i, j))) =
            (List.finRange out_w).foldl (fun acc j => acc + termR (i, j)) acc := by
      intro acc i
      exact
        (List.foldl_map (f := fun j => (i, j)) (g := fR) (l := List.finRange out_w) (init := acc))
    have hC :
        idxs.foldl fR (0 : R) =
          (List.finRange out_h).foldl (fun acc i =>
            List.foldl fR acc ((List.finRange out_w).map (fun j => (i, j)))) 0 := by
      simpa [idxs] using
        (foldl_flatMap (l := List.finRange out_h) (g := fun i => (List.finRange out_w).map (fun j =>
          (i, j)))
          (f := fR) (init := (0 : R)))
    have hC' :
        idxs.foldl fR 0 =
          (List.finRange out_h).foldl (fun acc i => (List.finRange out_w).foldl (fun acc j => acc +
            termR (i, j)) acc) 0 := by
      have :=
        foldl_congr (l := List.finRange out_h)
          (f := fun acc i => List.foldl fR acc ((List.finRange out_w).map (fun j => (i, j))))
          (g := fun acc i => (List.finRange out_w).foldl (fun acc j => acc + termR (i, j)) acc)
          (init := (0 : R)) (by intro acc i; simpa using (hW acc i))
      simpa [hC] using this
    simpa [sumR, fR] using hC'.symm

  have hsumS_nested :
      (List.finRange out_h).foldl (fun acc i =>
          (List.finRange out_w).foldl (fun acc j => acc + termS (i, j)) acc) 0 = sumS := by
    -- identical proof over `ℝ`
    let fS : ℝ → (Fin out_h × Fin out_w) → ℝ := fun acc t => acc + termS t
    have hW :
        ∀ (acc : ℝ) (i : Fin out_h),
          List.foldl fS acc ((List.finRange out_w).map (fun j => (i, j))) =
            (List.finRange out_w).foldl (fun acc j => acc + termS (i, j)) acc := by
      intro acc i
      exact
        (List.foldl_map (f := fun j => (i, j)) (g := fS) (l := List.finRange out_w) (init := acc))
    have hC :
        idxs.foldl fS (0 : ℝ) =
          (List.finRange out_h).foldl (fun acc i =>
            List.foldl fS acc ((List.finRange out_w).map (fun j => (i, j)))) 0 := by
      simpa [idxs] using
        (foldl_flatMap (l := List.finRange out_h) (g := fun i => (List.finRange out_w).map (fun j =>
          (i, j)))
          (f := fS) (init := (0 : ℝ)))
    have hC' :
        idxs.foldl fS 0 =
          (List.finRange out_h).foldl (fun acc i => (List.finRange out_w).foldl (fun acc j => acc +
            termS (i, j)) acc) 0 := by
      have :=
        foldl_congr (l := List.finRange out_h)
          (f := fun acc i => List.foldl fS acc ((List.finRange out_w).map (fun j => (i, j))))
          (g := fun acc i => (List.finRange out_w).foldl (fun acc j => acc + termS (i, j)) acc)
          (init := (0 : ℝ)) (by intro acc i; simpa using (hW acc i))
      simpa [hC] using this
    simpa [sumS, fS] using hC'.symm

  have houtR :
      getAtOrZero (Spec.conv2dBiasDerivSpec (α := R) (layer := layerR) (input := inputR)
        (grad_output := δR))
        [out_ch.val] = sumR := by
    have hFoldR :
        (List.finRange out_h).foldl (fun acc i =>
            (List.finRange out_w).foldl (fun acc j =>
                acc + getAtOrZero δR [out_ch.val, i.val, j.val]) acc) 0 = sumR := by
      have hNested :
          (List.finRange out_h).foldl (fun acc i =>
              (List.finRange out_w).foldl (fun acc j =>
                  acc + getAtOrZero δR [out_ch.val, i.val, j.val]) acc) 0 =
            (List.finRange out_h).foldl (fun acc i =>
              (List.finRange out_w).foldl (fun acc j => acc + termR (i, j)) acc) 0 := by
        refine foldl_congr (l := List.finRange out_h)
          (f := fun acc i =>
            (List.finRange out_w).foldl (fun acc j => acc + getAtOrZero δR [out_ch.val, i.val,
              j.val]) acc)
          (g := fun acc i =>
            (List.finRange out_w).foldl (fun acc j => acc + termR (i, j)) acc)
          (init := (0 : R)) ?_
        intro acc i
        refine foldl_congr (l := List.finRange out_w)
          (f := fun acc j => acc + getAtOrZero δR [out_ch.val, i.val, j.val])
          (g := fun acc j => acc + termR (i, j))
          (init := acc) ?_
        intro acc j
        simp [termR]
      simpa [hNested] using hsumR_nested
    have hFoldR' :
        (List.finRange (Shape.slidingWindowOutDim inH kH stride padding)).foldl (fun acc i =>
            (List.finRange (Shape.slidingWindowOutDim inW kW stride padding)).foldl (fun acc j =>
                acc + getAtOrZero δR [out_ch.val, i.val, j.val]) acc) 0 = sumR := by
      simpa [out_h, out_w, conv2dOutH, conv2dOutW] using hFoldR
    have hSpec :
        getAtOrZero (Spec.conv2dBiasDerivSpec (α := R) (layer := layerR) (input := inputR)
          (grad_output := δR))
          [out_ch.val] =
        (List.finRange (Shape.slidingWindowOutDim inH kH stride padding)).foldl (fun acc i =>
            (List.finRange (Shape.slidingWindowOutDim inW kW stride padding)).foldl (fun acc j =>
                acc + getAtOrZero δR [out_ch.val, i.val, j.val]) acc) 0 := by
      simp [Spec.conv2dBiasDerivSpec, out_ch.isLt]
      rfl
    exact hSpec.trans hFoldR'

  have houtS :
      getAtOrZero (Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
        (grad_output := δS))
        [out_ch.val] = sumS := by
    have hFoldS :
        (List.finRange out_h).foldl (fun acc i =>
            (List.finRange out_w).foldl (fun acc j =>
                acc + getAtOrZero δS [out_ch.val, i.val, j.val]) acc) 0 = sumS := by
      have hNested :
          (List.finRange out_h).foldl (fun acc i =>
              (List.finRange out_w).foldl (fun acc j =>
                  acc + getAtOrZero δS [out_ch.val, i.val, j.val]) acc) 0 =
            (List.finRange out_h).foldl (fun acc i =>
              (List.finRange out_w).foldl (fun acc j => acc + termS (i, j)) acc) 0 := by
        refine foldl_congr (l := List.finRange out_h)
          (f := fun acc i =>
            (List.finRange out_w).foldl (fun acc j => acc + getAtOrZero δS [out_ch.val, i.val,
              j.val]) acc)
          (g := fun acc i =>
            (List.finRange out_w).foldl (fun acc j => acc + termS (i, j)) acc)
          (init := (0 : ℝ)) ?_
        intro acc i
        refine foldl_congr (l := List.finRange out_w)
          (f := fun acc j => acc + getAtOrZero δS [out_ch.val, i.val, j.val])
          (g := fun acc j => acc + termS (i, j))
          (init := acc) ?_
        intro acc j
        simp [termS]
      simpa [hNested] using hsumS_nested
    have hFoldS' :
        (List.finRange (Shape.slidingWindowOutDim inH kH stride padding)).foldl (fun acc i =>
            (List.finRange (Shape.slidingWindowOutDim inW kW stride padding)).foldl (fun acc j =>
                acc + getAtOrZero δS [out_ch.val, i.val, j.val]) acc) 0 = sumS := by
      simpa [out_h, out_w, conv2dOutH, conv2dOutW] using hFoldS
    have hSpec :
        getAtOrZero (Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
          (grad_output := δS))
          [out_ch.val] =
        (List.finRange (Shape.slidingWindowOutDim inH kH stride padding)).foldl (fun acc i =>
            (List.finRange (Shape.slidingWindowOutDim inW kW stride padding)).foldl (fun acc j =>
                acc + getAtOrZero δS [out_ch.val, i.val, j.val]) acc) 0 := by
      simp [Spec.conv2dBiasDerivSpec, out_ch.isLt]
      rfl
    exact hSpec.trans hFoldS'

  have hFinal :
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (getAtOrZero (Spec.conv2dBiasDerivSpec (α := R) (layer := layerR) (input :=
                inputR) (grad_output := δR))
                [out_ch.val]) -
            getAtOrZero (Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layerS) (input :=
              inputS) (grad_output := δS))
              [out_ch.val]) ≤ sumEps := by
    simpa [houtR, houtS] using hSum

  simpa [conv2dBiasPointBound, out_h, out_w, idxs, termR, epsTerm, sumEps] using hFinal

/--
Tensor-shaped bias-gradient bound.

This packages `conv2dBiasPointBound` into a `Tensor` so later `approxTensor` statements can use
`linfNorm` to obtain a single scalar error bound.
-/
def conv2dBiasBoundTensor
    {outC kH kW stride padding inH inW : Nat}
    (δR : Spec.Tensor R (.dim outC (.dim (conv2dOutH inH kH stride padding) (.dim (conv2dOutW inW kW stride
      padding) .scalar))))
    (epsδ : ℝ) : Tensor ℝ (.dim outC .scalar) :=
  Tensor.dim (fun out_ch =>
    Tensor.scalar <| abs <|
      conv2dBiasPointBound (β := β) (fexp := fexp) (rnd := rnd)
        (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding) (inH := inH)
          (inW := inW)
        δR epsδ out_ch)

-- ---------------------------------------------------------------------------
-- Conv2d kernel gradient: pointwise bound
-- ---------------------------------------------------------------------------

/--
Pointwise error bound for the Conv2d **kernel** gradient (NF runtime vs spec).

The kernel-gradient entry accumulates products of padded input values and upstream gradients.
The bound is a replay of this accumulation with per-term errors derived from `epsX` and `epsδ`.
-/
def conv2dKernelPointBound
    {inC outC kH kW stride padding inH inW : Nat}
    (inputR : Spec.Tensor R (.dim inC (.dim inH (.dim inW .scalar))))
    (δR : Spec.Tensor R (.dim outC (.dim (conv2dOutH inH kH stride padding) (.dim (conv2dOutW inW kW stride
      padding) .scalar))))
    (epsX epsδ : ℝ)
    (out_ch : Fin outC) (in_ch : Fin inC) (di : Fin kH) (dj : Fin kW) : ℝ :=
  let out_h := conv2dOutH inH kH stride padding
  let out_w := conv2dOutW inW kW stride padding
  let padded_inputR :=
    if h4 : padding = 0 then
      tensorCast
        (Shape.dim inC (Shape.dim (inH + 2 * padding) (Shape.dim (inW + 2 * padding) .scalar)))
        (by simp; rw [h4])
        inputR
    else
      Spec.padChannelsFirst2d inputR padding
  let idxs : List (Fin out_h × Fin out_w) :=
    (List.finRange out_h).flatMap (fun i => (List.finRange out_w).map (fun j => (i, j)))
  let termR : (Fin out_h × Fin out_w) → R := fun t =>
    let i := t.1
    let j := t.2
    let input_val := getAtOrZero padded_inputR [in_ch.val, i.val * stride + di.val, j.val *
      stride + dj.val]
    let grad_val := getAtOrZero δR [out_ch.val, i.val, j.val]
    input_val * grad_val
  let epsTerm : (Fin out_h × Fin out_w) → ℝ := fun t =>
    let i := t.1
    let j := t.2
    let input_val := getAtOrZero padded_inputR [in_ch.val, i.val * stride + di.val, j.val *
      stride + dj.val]
    let grad_val := getAtOrZero δR [out_ch.val, i.val, j.val]
    mulEps (β := β) (fexp := fexp) (rnd := rnd) input_val grad_val epsX epsδ
  (foldAddState (β := β) (fexp := fexp) (rnd := rnd) idxs termR epsTerm).2

/--
Soundness of the Conv2d **kernel**-gradient pointwise bound.

Given `approxTensor` hypotheses for the input and upstream gradient (`grad_output`), this shows the spec
kernel-gradient entry is approximated by the NF runtime entry within `conv2dKernelPointBound`.
-/
theorem approx_conv2d_kernel_point
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    {kernelS : Tensor ℝ (.dim outC (.dim inC (.dim kH (.dim kW .scalar))))}
    {kernelR : Tensor R (.dim outC (.dim inC (.dim kH (.dim kW .scalar))))}
    {biasS : Tensor ℝ (.dim outC .scalar)} {biasR : Tensor R (.dim outC .scalar)}
    {inputS : Spec.Tensor ℝ (.dim inC (.dim inH (.dim inW .scalar)))}
    {inputR : Spec.Tensor R (.dim inC (.dim inH (.dim inW .scalar)))}
    {δS : Spec.Tensor ℝ (.dim outC (.dim (conv2dOutH inH kH stride padding) (.dim (conv2dOutW inW kW stride
      padding) .scalar)))}
    {δR : Spec.Tensor R (.dim outC (.dim (conv2dOutH inH kH stride padding) (.dim (conv2dOutW inW kW stride
      padding) .scalar)))}
    {epsX epsδ : ℝ}
    (hX : approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) inputS inputR
      epsX)
    (hδ : approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) δS δR epsδ)
    (out_ch : Fin outC) (in_ch : Fin inC) (di : Fin kH) (dj : Fin kW) :
    let layerS : Spec.Conv2dSpec inC outC kH kW stride padding ℝ h1 h2 h3 :=
      { kernel := kernelS, bias := biasS }
    let layerR : Spec.Conv2dSpec inC outC kH kW stride padding R h1 h2 h3 :=
      { kernel := kernelR, bias := biasR }
    abs
        (toSpec (β := β) (fexp := fexp) (rnd := rnd)
            (getAtOrZero
              (Spec.conv2dKernelDerivSpec (α := R) (layer := layerR) (input := inputR)
                (grad_output := δR))
              [out_ch.val, in_ch.val, di.val, dj.val]) -
    getAtOrZero
      (Spec.conv2dKernelDerivSpec (α := ℝ) (layer := layerS) (input := inputS) (grad_output :=
        δS))
      [out_ch.val, in_ch.val, di.val, dj.val]) ≤
      conv2dKernelPointBound (β := β) (fexp := fexp) (rnd := rnd)
        (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
          (inH := inH) (inW := inW)
        inputR δR epsX epsδ out_ch in_ch di dj := by
  intro layerS layerR
  classical
  let out_h := conv2dOutH inH kH stride padding
  let out_w := conv2dOutW inW kW stride padding
  let paddedR :=
    if h4 : padding = 0 then
      tensorCast
        (Shape.dim inC (Shape.dim (inH + 2 * padding) (Shape.dim (inW + 2 * padding) .scalar)))
        (by simp; rw [h4])
        inputR
    else
      Spec.padChannelsFirst2d inputR padding
  let paddedS :=
    if h4 : padding = 0 then
      tensorCast
        (Shape.dim inC (Shape.dim (inH + 2 * padding) (Shape.dim (inW + 2 * padding) .scalar)))
        (by simp; rw [h4])
        inputS
    else
      Spec.padChannelsFirst2d inputS padding
  let idxs : List (Fin out_h × Fin out_w) :=
    (List.finRange out_h).flatMap (fun i => (List.finRange out_w).map (fun j => (i, j)))
  let termR : (Fin out_h × Fin out_w) → R := fun t =>
    let i := t.1; let j := t.2
    let input_val := getAtOrZero paddedR [in_ch.val, i.val * stride + di.val, j.val * stride +
      dj.val]
    let grad_val := getAtOrZero δR [out_ch.val, i.val, j.val]
    input_val * grad_val
  let termS : (Fin out_h × Fin out_w) → ℝ := fun t =>
    let i := t.1; let j := t.2
    let input_val := getAtOrZero paddedS [in_ch.val, i.val * stride + di.val, j.val * stride +
      dj.val]
    let grad_val := getAtOrZero δS [out_ch.val, i.val, j.val]
    input_val * grad_val
  let epsTerm : (Fin out_h × Fin out_w) → ℝ := fun t =>
    let i := t.1; let j := t.2
    let input_val := getAtOrZero paddedR [in_ch.val, i.val * stride + di.val, j.val * stride +
      dj.val]
    let grad_val := getAtOrZero δR [out_ch.val, i.val, j.val]
    mulEps (β := β) (fexp := fexp) (rnd := rnd) input_val grad_val epsX epsδ
  let sumR : R := idxs.foldl (fun acc t => acc + termR t) 0
  let sumS : ℝ := idxs.foldl (fun acc t => acc + termS t) 0
  let sumEps : ℝ := (foldAddState (β := β) (fexp := fexp) (rnd := rnd) idxs termR epsTerm).2

  have hInputVal :
      ∀ (p q : Nat),
        abs
            (toSpec (β := β) (fexp := fexp) (rnd := rnd) (getAtOrZero paddedR [in_ch.val, p, q])
              -
              getAtOrZero paddedS [in_ch.val, p, q]) ≤ epsX := by
    intro p q
    simpa [paddedR, paddedS] using
      (approx_padded_input_read (β := β) (fexp := fexp) (rnd := rnd)
        (inC := inC) (inH := inH) (inW := inW) (padding := padding) (xS := inputS) (xR := inputR)
          (epsX := epsX) hX in_ch p q)

  have hGradVal :
      ∀ (i : Fin out_h) (j : Fin out_w),
        abs
            (toSpec (β := β) (fexp := fexp) (rnd := rnd) (getAtOrZero δR [out_ch.val, i.val,
              j.val]) -
              getAtOrZero δS [out_ch.val, i.val, j.val]) ≤ epsδ := by
    intro i j
    simpa using
      (approx_get_at_or_zero (β := β) (fexp := fexp) (rnd := rnd)
        (s := .dim outC (.dim out_h (.dim out_w .scalar)))
        (xS := δS) (xR := δR) (eps := epsδ) hδ [out_ch.val, i.val, j.val])

  have hTermIdx :
      ∀ t ∈ idxs, abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (termR t) - termS t) ≤ epsTerm t
        := by
    intro t _ht
    rcases t with ⟨i, j⟩
    have hx := hInputVal (i.val * stride + di.val) (j.val * stride + dj.val)
    have hy := hGradVal i j
    have :=
      approx_mul_nf (β := β) (fexp := fexp) (rnd := rnd)
        (x := getAtOrZero paddedS [in_ch.val, i.val * stride + di.val, j.val * stride + dj.val])
        (y := getAtOrZero δS [out_ch.val, i.val, j.val])
        (xR := getAtOrZero paddedR [in_ch.val, i.val * stride + di.val, j.val * stride + dj.val])
        (yR := getAtOrZero δR [out_ch.val, i.val, j.val])
        (epsx := epsX) (epsy := epsδ) hx hy
    simpa [termR, termS, epsTerm, mulEps] using this

  have hSum :
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) sumR - sumS) ≤ sumEps := by
    simpa [sumR, sumS, sumEps] using
      (approx_fold_add (β := β) (fexp := fexp) (rnd := rnd)
        (l := idxs) (termS := termS) (termR := termR) (epsTerm := epsTerm) hTermIdx)

  have hsumR_nested :
      (List.finRange out_h).foldl (fun acc i =>
          (List.finRange out_w).foldl (fun acc j => acc + termR (i, j)) acc) 0 = sumR := by
    let fR : R → (Fin out_h × Fin out_w) → R := fun acc t => acc + termR t
    have hW :
        ∀ (acc : R) (i : Fin out_h),
          List.foldl fR acc ((List.finRange out_w).map (fun j => (i, j))) =
            (List.finRange out_w).foldl (fun acc j => acc + termR (i, j)) acc := by
      intro acc i
      exact
        (List.foldl_map (f := fun j => (i, j)) (g := fR) (l := List.finRange out_w) (init := acc))
    have hC :
        idxs.foldl fR (0 : R) =
          (List.finRange out_h).foldl (fun acc i =>
            List.foldl fR acc ((List.finRange out_w).map (fun j => (i, j)))) 0 := by
      simpa [idxs] using
        (foldl_flatMap (l := List.finRange out_h) (g := fun i => (List.finRange out_w).map (fun j =>
          (i, j)))
          (f := fR) (init := (0 : R)))
    have hC' :
        idxs.foldl fR 0 =
          (List.finRange out_h).foldl (fun acc i => (List.finRange out_w).foldl (fun acc j => acc +
            termR (i, j)) acc) 0 := by
      have :=
        foldl_congr (l := List.finRange out_h)
          (f := fun acc i => List.foldl fR acc ((List.finRange out_w).map (fun j => (i, j))))
          (g := fun acc i => (List.finRange out_w).foldl (fun acc j => acc + termR (i, j)) acc)
          (init := (0 : R)) (by intro acc i; simpa using (hW acc i))
      simpa [hC] using this
    simpa [sumR, fR] using hC'.symm

  have hsumS_nested :
      (List.finRange out_h).foldl (fun acc i =>
          (List.finRange out_w).foldl (fun acc j => acc + termS (i, j)) acc) 0 = sumS := by
    let fS : ℝ → (Fin out_h × Fin out_w) → ℝ := fun acc t => acc + termS t
    have hW :
        ∀ (acc : ℝ) (i : Fin out_h),
          List.foldl fS acc ((List.finRange out_w).map (fun j => (i, j))) =
            (List.finRange out_w).foldl (fun acc j => acc + termS (i, j)) acc := by
      intro acc i
      exact
        (List.foldl_map (f := fun j => (i, j)) (g := fS) (l := List.finRange out_w) (init := acc))
    have hC :
        idxs.foldl fS (0 : ℝ) =
          (List.finRange out_h).foldl (fun acc i =>
            List.foldl fS acc ((List.finRange out_w).map (fun j => (i, j)))) 0 := by
      simpa [idxs] using
        (foldl_flatMap (l := List.finRange out_h) (g := fun i => (List.finRange out_w).map (fun j =>
          (i, j)))
          (f := fS) (init := (0 : ℝ)))
    have hC' :
        idxs.foldl fS 0 =
          (List.finRange out_h).foldl (fun acc i => (List.finRange out_w).foldl (fun acc j => acc +
            termS (i, j)) acc) 0 := by
      have :=
        foldl_congr (l := List.finRange out_h)
          (f := fun acc i => List.foldl fS acc ((List.finRange out_w).map (fun j => (i, j))))
          (g := fun acc i => (List.finRange out_w).foldl (fun acc j => acc + termS (i, j)) acc)
          (init := (0 : ℝ)) (by intro acc i; simpa using (hW acc i))
      simpa [hC] using this
    simpa [sumS, fS] using hC'.symm

  have hFoldR :
      (List.finRange out_h).foldl (fun acc i =>
          (List.finRange out_w).foldl (fun acc j =>
              acc +
                getAtOrZero paddedR [in_ch.val, i.val * stride + di.val, j.val * stride + dj.val]
                  *
                  getAtOrZero δR [out_ch.val, i.val, j.val]) acc) 0 = sumR := by
    have hNested :
        (List.finRange out_h).foldl (fun acc i =>
            (List.finRange out_w).foldl (fun acc j =>
                acc +
                  getAtOrZero paddedR [in_ch.val, i.val * stride + di.val, j.val * stride +
                    dj.val] *
                    getAtOrZero δR [out_ch.val, i.val, j.val]) acc) 0 =
          (List.finRange out_h).foldl (fun acc i =>
            (List.finRange out_w).foldl (fun acc j => acc + termR (i, j)) acc) 0 := by
      refine foldl_congr (l := List.finRange out_h)
        (f := fun acc i =>
          (List.finRange out_w).foldl (fun acc j =>
              acc +
                getAtOrZero paddedR [in_ch.val, i.val * stride + di.val, j.val * stride + dj.val]
                  *
                  getAtOrZero δR [out_ch.val, i.val, j.val]) acc)
        (g := fun acc i =>
          (List.finRange out_w).foldl (fun acc j => acc + termR (i, j)) acc)
        (init := (0 : R)) ?_
      intro acc i
      refine foldl_congr (l := List.finRange out_w)
        (f := fun acc j =>
          acc +
            getAtOrZero paddedR [in_ch.val, i.val * stride + di.val, j.val * stride + dj.val] *
              getAtOrZero δR [out_ch.val, i.val, j.val])
        (g := fun acc j => acc + termR (i, j))
        (init := acc) ?_
      intro acc j
      simp [termR]
    simpa [hNested] using hsumR_nested

  have hFoldS :
      (List.finRange out_h).foldl (fun acc i =>
          (List.finRange out_w).foldl (fun acc j =>
              acc +
                getAtOrZero paddedS [in_ch.val, i.val * stride + di.val, j.val * stride + dj.val]
                  *
                  getAtOrZero δS [out_ch.val, i.val, j.val]) acc) 0 = sumS := by
    have hNested :
        (List.finRange out_h).foldl (fun acc i =>
            (List.finRange out_w).foldl (fun acc j =>
                acc +
                  getAtOrZero paddedS [in_ch.val, i.val * stride + di.val, j.val * stride +
                    dj.val] *
                    getAtOrZero δS [out_ch.val, i.val, j.val]) acc) 0 =
          (List.finRange out_h).foldl (fun acc i =>
            (List.finRange out_w).foldl (fun acc j => acc + termS (i, j)) acc) 0 := by
      refine foldl_congr (l := List.finRange out_h)
        (f := fun acc i =>
          (List.finRange out_w).foldl (fun acc j =>
              acc +
                getAtOrZero paddedS [in_ch.val, i.val * stride + di.val, j.val * stride + dj.val]
                  *
                  getAtOrZero δS [out_ch.val, i.val, j.val]) acc)
        (g := fun acc i =>
          (List.finRange out_w).foldl (fun acc j => acc + termS (i, j)) acc)
        (init := (0 : ℝ)) ?_
      intro acc i
      refine foldl_congr (l := List.finRange out_w)
        (f := fun acc j =>
          acc +
            getAtOrZero paddedS [in_ch.val, i.val * stride + di.val, j.val * stride + dj.val] *
              getAtOrZero δS [out_ch.val, i.val, j.val])
        (g := fun acc j => acc + termS (i, j))
        (init := acc) ?_
      intro acc j
      simp [termS]
    simpa [hNested] using hsumS_nested

  have houtR :
      getAtOrZero (Spec.conv2dKernelDerivSpec (α := R) (layer := layerR) (input := inputR)
        (grad_output := δR))
          [out_ch.val, in_ch.val, di.val, dj.val] = sumR := by
    have hFoldR' :
        (List.finRange (conv2dOutH inH kH stride padding)).foldl (fun acc i =>
            (List.finRange (conv2dOutW inW kW stride padding)).foldl (fun acc j =>
                acc +
                  getAtOrZero
                      (paddedInput (inC := inC) (inH := inH) (inW := inW)
                        (padding := padding) inputR)
                      [in_ch.val, i.val * stride + di.val, j.val * stride + dj.val] *
                    getAtOrZero δR [out_ch.val, i.val, j.val]) acc) 0 = sumR := by
      simpa [out_h, out_w, paddedR, paddedInput] using hFoldR
    have hGet :
        getAtOrZero (Spec.conv2dKernelDerivSpec (α := R) (layer := layerR) (input := inputR)
          (grad_output := δR))
            [out_ch.val, in_ch.val, di.val, dj.val] =
          (List.finRange (conv2dOutH inH kH stride padding)).foldl (fun acc i =>
              (List.finRange (conv2dOutW inW kW stride padding)).foldl (fun acc j =>
                  acc +
                    getAtOrZero
                        (paddedInput (inC := inC) (inH := inH) (inW := inW)
                          (padding := padding) inputR)
                        [in_ch.val, i.val * stride + di.val, j.val * stride + dj.val] *
                        getAtOrZero δR [out_ch.val, i.val, j.val]) acc) 0 := by
        convert
          (conv2dKernelFoldRead_eq_paddedFold (input := inputR) (grad := δR)
            (out_ch := out_ch) (in_ch := in_ch) (di := di.val) (dj := dj.val)
            (stride := stride) (padding := padding)) using 1
        · simp [Spec.conv2dKernelDerivSpec, out_ch.isLt, in_ch.isLt,
            di.isLt, dj.isLt]
          rfl
    exact hGet.trans hFoldR'

  have houtS :
      getAtOrZero (Spec.conv2dKernelDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
        (grad_output := δS))
          [out_ch.val, in_ch.val, di.val, dj.val] = sumS := by
    have hFoldS' :
        (List.finRange (conv2dOutH inH kH stride padding)).foldl (fun acc i =>
            (List.finRange (conv2dOutW inW kW stride padding)).foldl (fun acc j =>
                acc +
                  getAtOrZero
                      (paddedInput (inC := inC) (inH := inH) (inW := inW)
                        (padding := padding) inputS)
                      [in_ch.val, i.val * stride + di.val, j.val * stride + dj.val] *
                    getAtOrZero δS [out_ch.val, i.val, j.val]) acc) 0 = sumS := by
      simpa [out_h, out_w, paddedS, paddedInput] using hFoldS
    have hGet :
        getAtOrZero (Spec.conv2dKernelDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
          (grad_output := δS))
            [out_ch.val, in_ch.val, di.val, dj.val] =
          (List.finRange (conv2dOutH inH kH stride padding)).foldl (fun acc i =>
              (List.finRange (conv2dOutW inW kW stride padding)).foldl (fun acc j =>
                  acc +
                    getAtOrZero
                        (paddedInput (inC := inC) (inH := inH) (inW := inW)
                          (padding := padding) inputS)
                        [in_ch.val, i.val * stride + di.val, j.val * stride + dj.val] *
                        getAtOrZero δS [out_ch.val, i.val, j.val]) acc) 0 := by
        convert
          (conv2dKernelFoldRead_eq_paddedFold (input := inputS) (grad := δS)
            (out_ch := out_ch) (in_ch := in_ch) (di := di.val) (dj := dj.val)
            (stride := stride) (padding := padding)) using 1
        · simp [Spec.conv2dKernelDerivSpec, out_ch.isLt, in_ch.isLt,
            di.isLt, dj.isLt]
          rfl
    exact hGet.trans hFoldS'

  have hFinal :
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (getAtOrZero
                (Spec.conv2dKernelDerivSpec (α := R) (layer := layerR) (input := inputR)
                  (grad_output := δR))
                [out_ch.val, in_ch.val, di.val, dj.val]) -
            getAtOrZero
              (Spec.conv2dKernelDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
                (grad_output := δS))
              [out_ch.val, in_ch.val, di.val, dj.val]) ≤ sumEps := by
    simpa [houtR, houtS] using hSum

  simpa [conv2dKernelPointBound, out_h, out_w, idxs, termR, epsTerm, sumEps, paddedR] using
    hFinal

/--
Tensor-shaped kernel-gradient bound.

This packages `conv2dKernelPointBound` into the full 4D kernel-tensor shape so later `approxTensor`
lemmas can use a single global bound via `linfNorm`.
-/
def conv2dKernelBoundTensor
    {inC outC kH kW stride padding inH inW : Nat}
    (inputR : Spec.Tensor R (.dim inC (.dim inH (.dim inW .scalar))))
    (δR : Spec.Tensor R (.dim outC (.dim (conv2dOutH inH kH stride padding) (.dim (conv2dOutW inW kW stride
      padding) .scalar))))
    (epsX epsδ : ℝ) :
    Tensor ℝ (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))) :=
  Tensor.dim (fun out_ch =>
    Tensor.dim (fun in_ch =>
      Tensor.dim (fun di =>
        Tensor.dim (fun dj =>
          Tensor.scalar <| abs <|
            conv2dKernelPointBound (β := β) (fexp := fexp) (rnd := rnd)
              (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding :=
                padding) (inH := inH) (inW := inW)
              inputR δR epsX epsδ out_ch in_ch di dj))))

-- ---------------------------------------------------------------------------
-- Tensor-level backward bounds (kernel + bias)
-- ---------------------------------------------------------------------------

/--
Tensor-level `approxTensor` bound for the Conv2d **bias** gradient.

This lifts `approx_conv2d_bias_point` entrywise and packages the error into
`linfNorm (conv2dBiasBoundTensor ...)`.
-/
theorem approxTensor_conv2d_bias_deriv_spec
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    {kernelS : Tensor ℝ (.dim outC (.dim inC (.dim kH (.dim kW .scalar))))}
    {kernelR : Tensor R (.dim outC (.dim inC (.dim kH (.dim kW .scalar))))}
    {biasS : Tensor ℝ (.dim outC .scalar)} {biasR : Tensor R (.dim outC .scalar)}
    {inputS : Spec.Tensor ℝ (.dim inC (.dim inH (.dim inW .scalar)))}
    {inputR : Spec.Tensor R (.dim inC (.dim inH (.dim inW .scalar)))}
    {δS : Spec.Tensor ℝ (.dim outC (.dim (conv2dOutH inH kH stride padding) (.dim (conv2dOutW inW kW stride
      padding) .scalar)))}
    {δR : Spec.Tensor R (.dim outC (.dim (conv2dOutH inH kH stride padding) (.dim (conv2dOutW inW kW stride
      padding) .scalar)))}
    {epsδ : ℝ}
    (hδ : approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) δS δR epsδ) :
    let layerS : Spec.Conv2dSpec inC outC kH kW stride padding ℝ h1 h2 h3 :=
      { kernel := kernelS, bias := biasS }
    let layerR : Spec.Conv2dSpec inC outC kH kW stride padding R h1 h2 h3 :=
      { kernel := kernelR, bias := biasR }
    let outS := Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
      (grad_output := δS)
    let outR := Spec.conv2dBiasDerivSpec (α := R) (layer := layerR) (input := inputR)
      (grad_output := δR)
    let bT :=
      conv2dBiasBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
        (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding) (inH := inH)
            (inW := inW)
          δR epsδ
    approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) outS outR (linfNorm
      bT) := by
  intro layerS layerR outS outR bT
  classical
  have hε : 0 ≤ linfNorm bT := linf_norm_nonneg (t := bT)
  refine approxTensor_dim_of_forall (n := outC) (s := .scalar)
    (xS := outS) (xR := outR) (eps := linfNorm bT) hε ?_
  intro oc
  have hpt :=
    approx_conv2d_bias_point (β := β) (fexp := fexp) (rnd := rnd)
      (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding) (inH
        := inH) (inW := inW)
      (h1 := h1) (h2 := h2) (h3 := h3)
      (kernelS := kernelS) (kernelR := kernelR) (biasS := biasS) (biasR := biasR)
      (inputS := inputS) (inputR := inputR) (δS := δS) (δR := δR) (epsδ := epsδ) hδ oc
  have hEntry :
      abs (conv2dBiasPointBound (β := β) (fexp := fexp) (rnd := rnd)
          (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding) (inH := inH)
            (inW := inW)
          δR epsδ oc) ≤ linfNorm bT := by
    have hcoord := (linf_norm_le_get_dim (t := bT) oc)
    let bound :=
      conv2dBiasPointBound (β := β) (fexp := fexp) (rnd := rnd)
        (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding) (inH := inH)
          (inW := inW)
        δR epsδ oc
    have hdouble' : (MathFunctions.abs (abs bound) : ℝ) ≤ linfNorm bT := by
      simpa [bound, bT, conv2dBiasBoundTensor, linfNorm, RuntimeApprox.linfNorm,
        tensorLinfNorm] using hcoord
    have habs : (MathFunctions.abs (abs bound) : ℝ) = abs (abs bound) := by
      rfl
    have hdouble : abs (abs bound) ≤ linfNorm bT := by
      simpa [habs] using hdouble'
    simpa [bound, abs_abs] using hdouble

  have hscalar :
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (getAtOrZero outR [oc.val]) -
            getAtOrZero outS [oc.val]) ≤ linfNorm bT :=
    le_trans (le_trans hpt (le_abs_self _)) hEntry

  let entryS : Tensor ℝ .scalar := (match outS with | .dim f => f oc)
  let entryR : Tensor R .scalar := (match outR with | .dim f => f oc)
  change
    approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) entryS entryR
      (linfNorm bT)
  have hEntryS :
      entryS = Tensor.scalar (getAtOrZero outS [oc.val]) := by
    convert (entry_eq_scalar_get_at_or_zero1 (t := outS) oc) using 1
    rfl
  have hEntryR :
      entryR = Tensor.scalar (getAtOrZero outR [oc.val]) := by
    convert (entry_eq_scalar_get_at_or_zero1 (t := outR) oc) using 1
    rfl
  have happ :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Tensor.scalar (getAtOrZero outS [oc.val]))
        (Tensor.scalar (getAtOrZero outR [oc.val]))
        (linfNorm bT) :=
    (approxTensor_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (x := getAtOrZero outS [oc.val]) (xR := getAtOrZero outR [oc.val]) (eps := linfNorm
          bT)).2 (by
          simpa using hscalar)
  simpa [hEntryS, hEntryR] using happ

/--
Tensor-level `approxTensor` bound for the Conv2d **kernel** gradient.

This lifts `approx_conv2d_kernel_point` entrywise and packages the error into
`linfNorm (conv2dKernelBoundTensor ...)`.
-/
theorem approxTensor_conv2d_kernel_deriv_spec
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    {kernelS : Tensor ℝ (.dim outC (.dim inC (.dim kH (.dim kW .scalar))))}
    {kernelR : Tensor R (.dim outC (.dim inC (.dim kH (.dim kW .scalar))))}
    {biasS : Tensor ℝ (.dim outC .scalar)} {biasR : Tensor R (.dim outC .scalar)}
    {inputS : Spec.Tensor ℝ (.dim inC (.dim inH (.dim inW .scalar)))}
    {inputR : Spec.Tensor R (.dim inC (.dim inH (.dim inW .scalar)))}
    {δS : Spec.Tensor ℝ (.dim outC (.dim (conv2dOutH inH kH stride padding) (.dim (conv2dOutW inW kW stride
      padding) .scalar)))}
    {δR : Spec.Tensor R (.dim outC (.dim (conv2dOutH inH kH stride padding) (.dim (conv2dOutW inW kW stride
      padding) .scalar)))}
    {epsX epsδ : ℝ}
    (hX : approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) inputS inputR
      epsX)
    (hδ : approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) δS δR epsδ) :
    let layerS : Spec.Conv2dSpec inC outC kH kW stride padding ℝ h1 h2 h3 :=
      { kernel := kernelS, bias := biasS }
    let layerR : Spec.Conv2dSpec inC outC kH kW stride padding R h1 h2 h3 :=
      { kernel := kernelR, bias := biasR }
    let outS := Spec.conv2dKernelDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
      (grad_output := δS)
    let outR := Spec.conv2dKernelDerivSpec (α := R) (layer := layerR) (input := inputR)
      (grad_output := δR)
    let bT :=
        conv2dKernelBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
            (inH := inH) (inW := inW)
          inputR δR epsX epsδ
    approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) outS outR (linfNorm
      bT) := by
  intro layerS layerR outS outR bT
  classical
  have hε : 0 ≤ linfNorm bT := linf_norm_nonneg (t := bT)
  refine approxTensor_dim_of_forall (n := outC)
    (s := .dim inC (.dim kH (.dim kW .scalar)))
    (xS := outS) (xR := outR) (eps := linfNorm bT) hε ?_
  intro oc
  refine approxTensor_dim_of_forall (n := inC)
    (s := .dim kH (.dim kW .scalar))
    (xS := (match outS with | .dim f => f oc)) (xR := (match outR with | .dim f => f oc))
    (eps := linfNorm bT) hε ?_
  intro ic
  refine approxTensor_dim_of_forall (n := kH)
    (s := .dim kW .scalar)
    (xS := (match (match outS with | .dim f => f oc) with | .dim g => g ic))
    (xR := (match (match outR with | .dim f => f oc) with | .dim g => g ic))
    (eps := linfNorm bT) hε ?_
  intro di
  refine approxTensor_dim_of_forall (n := kW)
    (s := .scalar)
    (xS :=
      (match
        match
          match outS with
          | .dim f => f oc
        with
        | .dim g => g ic
      with
      | .dim h => h di))
    (xR :=
      (match
        match
          match outR with
          | .dim f => f oc
        with
        | .dim g => g ic
      with
      | .dim h => h di))
    (eps := linfNorm bT) hε ?_
  intro dj

  have hpt :=
    approx_conv2d_kernel_point (β := β) (fexp := fexp) (rnd := rnd)
      (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding) (inH
        := inH) (inW := inW)
      (h1 := h1) (h2 := h2) (h3 := h3)
      (kernelS := kernelS) (kernelR := kernelR) (biasS := biasS) (biasR := biasR)
      (inputS := inputS) (inputR := inputR) (δS := δS) (δR := δR)
      (epsX := epsX) (epsδ := epsδ) hX hδ oc ic di dj

  have hEntry :
      abs (conv2dKernelPointBound (β := β) (fexp := fexp) (rnd := rnd)
          (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
            (inH := inH) (inW := inW)
          inputR δR epsX epsδ oc ic di dj) ≤ linfNorm bT := by
    let bToc := match bT with | .dim f => f oc
    let bTocic := match bToc with | .dim f => f ic
    let bTocicdi := match bTocic with | .dim f => f di
    let bTocicdidj := match bTocicdi with | .dim f => f dj
    have h0 : linfNorm bToc ≤ linfNorm bT := by
      convert (linf_norm_le_get_dim (t := bT) oc) using 1
      rfl
    have h1' : linfNorm bTocic ≤ linfNorm bToc := by
      convert (linf_norm_le_get_dim (t := bToc) ic) using 1
      rfl
    have h2' : linfNorm bTocicdi ≤ linfNorm bTocic := by
      convert (linf_norm_le_get_dim (t := bTocic) di) using 1
      rfl
    have h3' : linfNorm bTocicdidj ≤ linfNorm bTocicdi := by
      convert (linf_norm_le_get_dim (t := bTocicdi) dj) using 1
      rfl
    have hchain : linfNorm bTocicdidj ≤ linfNorm bT :=
      le_trans (le_trans (le_trans h3' h2') h1') h0
    let bound :=
      conv2dKernelPointBound (β := β) (fexp := fexp) (rnd := rnd)
        (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
          (inH := inH) (inW := inW)
        inputR δR epsX epsδ oc ic di dj
    have hdouble' : (MathFunctions.abs (abs bound) : ℝ) ≤ linfNorm bT := by
      simpa [bound, bT, bToc, bTocic, bTocicdi, bTocicdidj, conv2dKernelBoundTensor, linfNorm,
        RuntimeApprox.linfNorm, tensorLinfNorm] using hchain
    have habs : (MathFunctions.abs (abs bound) : ℝ) = abs (abs bound) := by
      rfl
    have hdouble : abs (abs bound) ≤ linfNorm bT := by
      simpa [habs] using hdouble'
    simpa [bound, abs_abs] using hdouble

  have hscalar :
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (getAtOrZero outR [oc.val, ic.val, di.val, dj.val]) -
            getAtOrZero outS [oc.val, ic.val, di.val, dj.val]) ≤ linfNorm bT :=
    le_trans (le_trans hpt (le_abs_self _)) hEntry

  let entryS : Tensor ℝ .scalar :=
    (match
      match
        match
          match outS with
          | .dim f => f oc with
        | .dim g => g ic with
      | .dim h => h di with
    | .dim k => k dj)
  let entryR : Tensor R .scalar :=
    (match
      match
        match
          match outR with
          | .dim f => f oc with
        | .dim g => g ic with
      | .dim h => h di with
    | .dim k => k dj)

  change
    approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) entryS entryR
      (linfNorm bT)
  have hEntryS :
      entryS = Tensor.scalar (getAtOrZero outS [oc.val, ic.val, di.val, dj.val]) := by
    convert (entry_eq_scalar_get_at_or_zero4 (t := outS) oc ic di dj) using 1
    rfl
  have hEntryR :
      entryR = Tensor.scalar (getAtOrZero outR [oc.val, ic.val, di.val, dj.val]) := by
    convert (entry_eq_scalar_get_at_or_zero4 (t := outR) oc ic di dj) using 1
    rfl

  have happ :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Tensor.scalar (getAtOrZero outS [oc.val, ic.val, di.val, dj.val]))
        (Tensor.scalar (getAtOrZero outR [oc.val, ic.val, di.val, dj.val]))
        (linfNorm bT) :=
    (approxTensor_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (x := getAtOrZero outS [oc.val, ic.val, di.val, dj.val])
        (xR := getAtOrZero outR [oc.val, ic.val, di.val, dj.val])
        (eps := linfNorm bT)).2 (by
          simpa using hscalar)

  simpa [hEntryS, hEntryR] using happ

-- ---------------------------------------------------------------------------
-- Conv2d input gradient: pointwise bound
-- ---------------------------------------------------------------------------


end NFBackend

end
end RuntimeApprox
end Proofs
