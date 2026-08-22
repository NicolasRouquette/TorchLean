/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.ConvBackward.BiasKernel

/-!
# NeuralFloat Conv2d Input-Gradient Bounds

This file completes the pointwise approximation argument for Conv2d backward by handling the input
gradient. Each input coordinate collects exactly the output/kernel positions that contribute to it
under the stride and padding relation, then bounds the resulting NeuralFloat accumulation.
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
Pointwise error bound for the Conv2d **input** gradient (NF runtime vs spec).

The input-gradient entry accumulates contributions from all output channels and spatial positions
that “hit” the given input coordinate under the stride/padding relation. The bound is a replay of
that accumulation with per-term errors derived from `epsK` and `epsδ`.
-/
def conv2dInputPointBound
    {inC outC kH kW stride padding inH inW : Nat}
    (kernelR : Tensor R (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))))
    (δR : Spec.Tensor R (.dim outC (.dim (conv2dOutH inH kH stride padding) (.dim (conv2dOutW inW kW stride
      padding) .scalar))))
    (epsK epsδ : ℝ)
    (in_ch : Fin inC) (i : Fin inH) (j : Fin inW) : ℝ :=
  let out_h := conv2dOutH inH kH stride padding
  let out_w := conv2dOutW inW kW stride padding
  let idxs : List (Fin outC × Fin out_h × Fin out_w × Fin kH × Fin kW) :=
    (List.finRange outC).flatMap (fun out_ch =>
      (List.finRange out_h).flatMap (fun out_i =>
        (List.finRange out_w).flatMap (fun out_j =>
          (List.finRange kH).flatMap (fun di =>
            (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))))
  let termR : (Fin outC × Fin out_h × Fin out_w × Fin kH × Fin kW) → R := fun t =>
    let out_ch := t.1
    let out_i := t.2.1
    let out_j := t.2.2.1
    let di := t.2.2.2.1
    let dj := t.2.2.2.2
      if _ :
          (out_i.val * stride + di.val = i.val + padding) ∧
          (out_j.val * stride + dj.val = j.val + padding) then
      let grad_val := getAtOrZero δR [out_ch.val, out_i.val, out_j.val]
      let kernel_val := getAtOrZero kernelR [out_ch.val, in_ch.val, di.val, dj.val]
      grad_val * kernel_val
    else
      0
  let epsTerm : (Fin outC × Fin out_h × Fin out_w × Fin kH × Fin kW) → ℝ := fun t =>
    let out_ch := t.1
    let out_i := t.2.1
    let out_j := t.2.2.1
    let di := t.2.2.2.1
    let dj := t.2.2.2.2
      if _ :
          (out_i.val * stride + di.val = i.val + padding) ∧
          (out_j.val * stride + dj.val = j.val + padding) then
      let grad_val := getAtOrZero δR [out_ch.val, out_i.val, out_j.val]
      let kernel_val := getAtOrZero kernelR [out_ch.val, in_ch.val, di.val, dj.val]
      mulEps (β := β) (fexp := fexp) (rnd := rnd) grad_val kernel_val epsδ epsK
    else
      0
  (foldAddState (β := β) (fexp := fexp) (rnd := rnd) idxs termR epsTerm).2

/--
Tensor-shaped input-gradient bound.

This packages `conv2dInputPointBound` into the full input image shape so later `approxTensor` lemmas
can use a single global bound via `linfNorm`.
-/
def conv2dInputBoundTensor
    {inC outC kH kW stride padding inH inW : Nat}
    (kernelR : Tensor R (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))))
    (δR : Spec.Tensor R (.dim outC (.dim (conv2dOutH inH kH stride padding) (.dim (conv2dOutW inW kW stride
      padding) .scalar))))
    (epsK epsδ : ℝ) :
    Spec.Tensor ℝ (.dim inC (.dim inH (.dim inW .scalar))) :=
  Tensor.dim (fun in_ch =>
    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        Tensor.scalar <| abs <|
          conv2dInputPointBound (β := β) (fexp := fexp) (rnd := rnd)
            (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding :=
              padding) (inH := inH) (inW := inW)
            kernelR δR epsK epsδ in_ch i j)))

/--
Soundness of the Conv2d **input**-gradient pointwise bound.

Given `approxTensor` hypotheses for the kernel and upstream gradient (`grad_output`), this shows the spec
input-gradient entry is approximated by the NF runtime entry within `conv2dInputPointBound`.
-/
theorem approx_conv2d_input_point
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
    {epsK epsδ : ℝ}
    (hK : approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) kernelS kernelR
      epsK)
    (hδ : approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) δS δR epsδ)
    (in_ch : Fin inC) (i : Fin inH) (j : Fin inW) :
    let layerS : Spec.Conv2dSpec inC outC kH kW stride padding ℝ h1 h2 h3 :=
      { kernel := kernelS, bias := biasS }
    let layerR : Spec.Conv2dSpec inC outC kH kW stride padding R h1 h2 h3 :=
      { kernel := kernelR, bias := biasR }
    abs
        (toSpec (β := β) (fexp := fexp) (rnd := rnd)
            (getAtOrZero
              (Spec.conv2dInputDerivSpec (α := R) (layer := layerR) (input := inputR)
                (grad_output := δR))
              [in_ch.val, i.val, j.val]) -
    getAtOrZero
      (Spec.conv2dInputDerivSpec (α := ℝ) (layer := layerS) (input := inputS) (grad_output :=
        δS))
      [in_ch.val, i.val, j.val]) ≤
      conv2dInputPointBound (β := β) (fexp := fexp) (rnd := rnd)
        (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
          (inH := inH) (inW := inW)
        kernelR δR epsK epsδ in_ch i j := by
  intro layerS layerR
  classical
  let out_h := conv2dOutH inH kH stride padding
  let out_w := conv2dOutW inW kW stride padding
  let idxs : List (Fin outC × Fin out_h × Fin out_w × Fin kH × Fin kW) :=
    (List.finRange outC).flatMap (fun out_ch =>
      (List.finRange out_h).flatMap (fun out_i =>
        (List.finRange out_w).flatMap (fun out_j =>
          (List.finRange kH).flatMap (fun di =>
            (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))))
  let termR : (Fin outC × Fin out_h × Fin out_w × Fin kH × Fin kW) → R := fun t =>
    let out_ch := t.1
    let out_i := t.2.1
    let out_j := t.2.2.1
    let di := t.2.2.2.1
    let dj := t.2.2.2.2
    if h :
        (out_i.val * stride + di.val = i.val + padding) ∧
        (out_j.val * stride + dj.val = j.val + padding) then
      let grad_val := getAtOrZero δR [out_ch.val, out_i.val, out_j.val]
      let kernel_val := getAtOrZero kernelR [out_ch.val, in_ch.val, di.val, dj.val]
      grad_val * kernel_val
    else
      0
  let termS : (Fin outC × Fin out_h × Fin out_w × Fin kH × Fin kW) → ℝ := fun t =>
    let out_ch := t.1
    let out_i := t.2.1
    let out_j := t.2.2.1
    let di := t.2.2.2.1
    let dj := t.2.2.2.2
    if h :
        (out_i.val * stride + di.val = i.val + padding) ∧
        (out_j.val * stride + dj.val = j.val + padding) then
      let grad_val := getAtOrZero δS [out_ch.val, out_i.val, out_j.val]
      let kernel_val := getAtOrZero kernelS [out_ch.val, in_ch.val, di.val, dj.val]
      grad_val * kernel_val
    else
      0
  let epsTerm : (Fin outC × Fin out_h × Fin out_w × Fin kH × Fin kW) → ℝ := fun t =>
    let out_ch := t.1
    let out_i := t.2.1
    let out_j := t.2.2.1
    let di := t.2.2.2.1
    let dj := t.2.2.2.2
    if h :
        (out_i.val * stride + di.val = i.val + padding) ∧
        (out_j.val * stride + dj.val = j.val + padding) then
      let grad_val := getAtOrZero δR [out_ch.val, out_i.val, out_j.val]
      let kernel_val := getAtOrZero kernelR [out_ch.val, in_ch.val, di.val, dj.val]
      mulEps (β := β) (fexp := fexp) (rnd := rnd) grad_val kernel_val epsδ epsK
    else
      0
  let sumR : R := idxs.foldl (fun acc t => acc + termR t) 0
  let sumS : ℝ := idxs.foldl (fun acc t => acc + termS t) 0
  let sumEps : ℝ := (foldAddState (β := β) (fexp := fexp) (rnd := rnd) idxs termR epsTerm).2

  have hGradVal :
      ∀ (out_ch : Fin outC) (out_i : Fin out_h) (out_j : Fin out_w),
        abs
            (toSpec (β := β) (fexp := fexp) (rnd := rnd) (getAtOrZero δR [out_ch.val, out_i.val,
              out_j.val]) -
              getAtOrZero δS [out_ch.val, out_i.val, out_j.val]) ≤ epsδ := by
    intro out_ch out_i out_j
    simpa using
      (approx_get_at_or_zero (β := β) (fexp := fexp) (rnd := rnd)
        (s := .dim outC (.dim out_h (.dim out_w .scalar)))
        (xS := δS) (xR := δR) (eps := epsδ) hδ [out_ch.val, out_i.val, out_j.val])

  have hKernelVal :
      ∀ (out_ch : Fin outC) (di : Fin kH) (dj : Fin kW),
        abs
            (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                (getAtOrZero kernelR [out_ch.val, in_ch.val, di.val, dj.val]) -
              getAtOrZero kernelS [out_ch.val, in_ch.val, di.val, dj.val]) ≤ epsK := by
    intro out_ch di dj
    simpa using
      (approx_get_at_or_zero (β := β) (fexp := fexp) (rnd := rnd)
        (s := .dim outC (.dim inC (.dim kH (.dim kW .scalar))))
        (xS := kernelS) (xR := kernelR) (eps := epsK) hK [out_ch.val, in_ch.val, di.val, dj.val])

  have hTermIdx :
      ∀ t ∈ idxs, abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (termR t) - termS t) ≤ epsTerm t
        := by
    intro t _ht
    rcases t with ⟨out_ch, out_i, out_j, di, dj⟩
    by_cases h :
        (out_i.val * stride + di.val = i.val + padding) ∧
        (out_j.val * stride + dj.val = j.val + padding)
    · have hx := hGradVal out_ch out_i out_j
      have hy := hKernelVal out_ch di dj
      have hmul :
          abs
              (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                  (getAtOrZero δR [out_ch.val, out_i.val, out_j.val] *
                    getAtOrZero kernelR [out_ch.val, in_ch.val, di.val, dj.val]) -
                getAtOrZero δS [out_ch.val, out_i.val, out_j.val] *
                  getAtOrZero kernelS [out_ch.val, in_ch.val, di.val, dj.val]) ≤
            mulEps (β := β) (fexp := fexp) (rnd := rnd)
              (getAtOrZero δR [out_ch.val, out_i.val, out_j.val])
              (getAtOrZero kernelR [out_ch.val, in_ch.val, di.val, dj.val]) epsδ epsK :=
        approx_mul_nf (β := β) (fexp := fexp) (rnd := rnd)
          (x := getAtOrZero δS [out_ch.val, out_i.val, out_j.val])
          (y := getAtOrZero kernelS [out_ch.val, in_ch.val, di.val, dj.val])
          (xR := getAtOrZero δR [out_ch.val, out_i.val, out_j.val])
          (yR := getAtOrZero kernelR [out_ch.val, in_ch.val, di.val, dj.val])
          (epsx := epsδ) (epsy := epsK) hx hy
      simpa [termR, termS, epsTerm, h] using hmul
    · simp [termR, termS, epsTerm, h]

  have hSum :
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) sumR - sumS) ≤ sumEps := by
    simpa [sumR, sumS, sumEps] using
      (approx_fold_add (β := β) (fexp := fexp) (rnd := rnd)
        (l := idxs) (termS := termS) (termR := termR) (epsTerm := epsTerm) hTermIdx)

  -- Rewrite the `Spec.conv2dInputDerivSpec` nested fold into the flattened `idxs` fold (`sumR/sumS`).
  have hsumR_nested :
      (List.finRange outC).foldl (fun acc out_ch =>
          (List.finRange out_h).foldl (fun acc out_i =>
              (List.finRange out_w).foldl (fun acc out_j =>
                  (List.finRange kH).foldl (fun acc di =>
                      (List.finRange kW).foldl (fun acc dj => acc + termR (out_ch, out_i, out_j, di,
                        dj)) acc) acc) acc) acc) 0 =
        sumR := by
    let fR : R → (Fin outC × Fin out_h × Fin out_w × Fin kH × Fin kW) → R := fun acc t => acc +
      termR t
    have hC0 :
        idxs.foldl fR (0 : R) =
          (List.finRange outC).foldl (fun acc out_ch =>
            List.foldl fR acc
              ((List.finRange out_h).flatMap (fun out_i =>
                (List.finRange out_w).flatMap (fun out_j =>
                  (List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))))) (0 : R)
                      := by
      simpa [idxs] using
        (foldl_flatMap (l := List.finRange outC)
          (g := fun out_ch =>
            (List.finRange out_h).flatMap (fun out_i =>
              (List.finRange out_w).flatMap (fun out_j =>
                (List.finRange kH).flatMap (fun di =>
                  (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))))
          (f := fR) (init := (0 : R)))
    have hOut :
        ∀ (acc : R) (out_ch : Fin outC),
          List.foldl fR acc
              ((List.finRange out_h).flatMap (fun out_i =>
                (List.finRange out_w).flatMap (fun out_j =>
                  (List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))))
            =
          (List.finRange out_h).foldl (fun acc out_i =>
              (List.finRange out_w).foldl (fun acc out_j =>
                  (List.finRange kH).foldl (fun acc di =>
                      (List.finRange kW).foldl (fun acc dj => acc + termR (out_ch, out_i, out_j, di,
                        dj)) acc) acc) acc) acc := by
      intro acc out_ch
      -- fold over `out_h`
      have hH' :
          List.foldl fR acc
              ((List.finRange out_h).flatMap (fun out_i =>
                (List.finRange out_w).flatMap (fun out_j =>
                  (List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))))
            =
          (List.finRange out_h).foldl (fun acc out_i =>
              List.foldl fR acc
                ((List.finRange out_w).flatMap (fun out_j =>
                  (List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj)))))) acc := by
        simpa using
          (foldl_flatMap (l := List.finRange out_h)
            (g := fun out_i =>
              (List.finRange out_w).flatMap (fun out_j =>
                (List.finRange kH).flatMap (fun di =>
                  (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj)))))
            (f := fR) (init := acc))
      -- fold over `out_w`/`kH`/`kW`
      have hH :
          ∀ (acc : R) (out_i : Fin out_h),
            List.foldl fR acc
                ((List.finRange out_w).flatMap (fun out_j =>
                  (List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj)))))
              =
            (List.finRange out_w).foldl (fun acc out_j =>
                (List.finRange kH).foldl (fun acc di =>
                    (List.finRange kW).foldl (fun acc dj => acc + termR (out_ch, out_i, out_j, di,
                      dj)) acc) acc) acc := by
        intro acc out_i
        have hW' :
            List.foldl fR acc
                ((List.finRange out_w).flatMap (fun out_j =>
                  (List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj)))))
              =
            (List.finRange out_w).foldl (fun acc out_j =>
                List.foldl fR acc
                  ((List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))) acc := by
          simpa using
            (foldl_flatMap (l := List.finRange out_w)
              (g := fun out_j =>
                (List.finRange kH).flatMap (fun di =>
                  (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))
              (f := fR) (init := acc))
        have hW :
            ∀ (acc : R) (out_j : Fin out_w),
              List.foldl fR acc
                  ((List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))
                =
              (List.finRange kH).foldl (fun acc di =>
                (List.finRange kW).foldl (fun acc dj => acc + termR (out_ch, out_i, out_j, di, dj))
                  acc) acc :=
        by
          intro acc out_j
          have hK' :
              List.foldl fR acc
                  ((List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))
                =
              (List.finRange kH).foldl (fun acc di =>
                  List.foldl fR acc ((List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di,
                    dj)))) acc :=
          by
            simpa using
              (foldl_flatMap (l := List.finRange kH)
                (g := fun di => (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj)))
                (f := fR) (init := acc))
          have hWk :
              ∀ (acc : R) (di : Fin kH),
                List.foldl fR acc ((List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di,
                  dj))) =
                  (List.finRange kW).foldl (fun acc dj => acc + termR (out_ch, out_i, out_j, di,
                    dj)) acc :=
          by
            intro acc di
            exact
              (List.foldl_map (f := fun dj => (out_ch, out_i, out_j, di, dj)) (g := fR) (l :=
                List.finRange kW) (init := acc))
          simpa [hK'] using
            (foldl_congr (l := List.finRange kH)
              (f := fun acc di => List.foldl fR acc ((List.finRange kW).map (fun dj => (out_ch,
                out_i, out_j, di, dj))))
              (g := fun acc di => (List.finRange kW).foldl (fun acc dj => acc + termR (out_ch,
                out_i, out_j, di, dj)) acc)
                (init := acc) (by intro acc di; simpa using (hWk acc di)))
        simpa [hW'] using
          (foldl_congr (l := List.finRange out_w)
            (f := fun acc out_j =>
              List.foldl fR acc
                ((List.finRange kH).flatMap (fun di => (List.finRange kW).map (fun dj => (out_ch,
                  out_i, out_j, di, dj)))))
            (g := fun acc out_j =>
              (List.finRange kH).foldl (fun acc di =>
                  (List.finRange kW).foldl (fun acc dj => acc + termR (out_ch, out_i, out_j, di,
                    dj)) acc) acc)
            (init := acc) (by intro acc out_j; simpa using (hW acc out_j)))
      simpa [hH'] using
        (foldl_congr (l := List.finRange out_h)
          (f := fun acc out_i =>
            List.foldl fR acc
              ((List.finRange out_w).flatMap (fun out_j =>
                (List.finRange kH).flatMap (fun di => (List.finRange kW).map (fun dj => (out_ch,
                  out_i, out_j, di, dj))))))
          (g := fun acc out_i =>
            (List.finRange out_w).foldl (fun acc out_j =>
                (List.finRange kH).foldl (fun acc di =>
                    (List.finRange kW).foldl (fun acc dj => acc + termR (out_ch, out_i, out_j, di,
                      dj)) acc) acc) acc)
          (init := acc) (by intro acc out_i; simpa using (hH acc out_i)))
    have hC0' :
        idxs.foldl fR 0 =
          (List.finRange outC).foldl (fun acc out_ch =>
            (List.finRange out_h).foldl (fun acc out_i =>
                (List.finRange out_w).foldl (fun acc out_j =>
                    (List.finRange kH).foldl (fun acc di =>
                        (List.finRange kW).foldl (fun acc dj => acc + termR (out_ch, out_i, out_j,
                          di, dj)) acc) acc) acc) acc) 0 := by
      have :=
        foldl_congr (l := List.finRange outC)
          (f := fun acc out_ch =>
            List.foldl fR acc
              ((List.finRange out_h).flatMap (fun out_i =>
                (List.finRange out_w).flatMap (fun out_j =>
                  (List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj)))))))
          (g := fun acc out_ch =>
            (List.finRange out_h).foldl (fun acc out_i =>
                (List.finRange out_w).foldl (fun acc out_j =>
                    (List.finRange kH).foldl (fun acc di =>
                        (List.finRange kW).foldl (fun acc dj => acc + termR (out_ch, out_i, out_j,
                          di, dj)) acc) acc) acc) acc)
          (init := (0 : R)) (by intro acc out_ch; simpa using (hOut acc out_ch))
      simpa [hC0] using this
    simpa [sumR, fR] using hC0'.symm

  have hsumS_nested :
      (List.finRange outC).foldl (fun acc out_ch =>
          (List.finRange out_h).foldl (fun acc out_i =>
              (List.finRange out_w).foldl (fun acc out_j =>
                  (List.finRange kH).foldl (fun acc di =>
                      (List.finRange kW).foldl (fun acc dj => acc + termS (out_ch, out_i, out_j, di,
                        dj)) acc) acc) acc) acc) 0 =
        sumS := by
    -- same proof as `hsumR_nested`, over `ℝ`
    let fS : ℝ → (Fin outC × Fin out_h × Fin out_w × Fin kH × Fin kW) → ℝ := fun acc t => acc +
      termS t
    have hC0 :
        idxs.foldl fS (0 : ℝ) =
          (List.finRange outC).foldl (fun acc out_ch =>
            List.foldl fS acc
              ((List.finRange out_h).flatMap (fun out_i =>
                (List.finRange out_w).flatMap (fun out_j =>
                  (List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))))) (0 : ℝ)
                      := by
      simpa [idxs] using
        (foldl_flatMap (l := List.finRange outC)
          (g := fun out_ch =>
            (List.finRange out_h).flatMap (fun out_i =>
              (List.finRange out_w).flatMap (fun out_j =>
                (List.finRange kH).flatMap (fun di =>
                  (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))))
          (f := fS) (init := (0 : ℝ)))
    have hOut :
        ∀ (acc : ℝ) (out_ch : Fin outC),
          List.foldl fS acc
              ((List.finRange out_h).flatMap (fun out_i =>
                (List.finRange out_w).flatMap (fun out_j =>
                  (List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))))
            =
          (List.finRange out_h).foldl (fun acc out_i =>
              (List.finRange out_w).foldl (fun acc out_j =>
                  (List.finRange kH).foldl (fun acc di =>
                      (List.finRange kW).foldl (fun acc dj => acc + termS (out_ch, out_i, out_j, di,
                        dj)) acc) acc) acc) acc := by
      intro acc out_ch
      have hH' :
          List.foldl fS acc
              ((List.finRange out_h).flatMap (fun out_i =>
                (List.finRange out_w).flatMap (fun out_j =>
                  (List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))))
            =
          (List.finRange out_h).foldl (fun acc out_i =>
              List.foldl fS acc
                ((List.finRange out_w).flatMap (fun out_j =>
                  (List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj)))))) acc := by
        simpa using
          (foldl_flatMap (l := List.finRange out_h)
            (g := fun out_i =>
              (List.finRange out_w).flatMap (fun out_j =>
                (List.finRange kH).flatMap (fun di =>
                  (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj)))))
            (f := fS) (init := acc))
      have hH :
          ∀ (acc : ℝ) (out_i : Fin out_h),
            List.foldl fS acc
                ((List.finRange out_w).flatMap (fun out_j =>
                  (List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj)))))
              =
            (List.finRange out_w).foldl (fun acc out_j =>
                (List.finRange kH).foldl (fun acc di =>
                    (List.finRange kW).foldl (fun acc dj => acc + termS (out_ch, out_i, out_j, di,
                      dj)) acc) acc) acc := by
        intro acc out_i
        have hW' :
            List.foldl fS acc
                ((List.finRange out_w).flatMap (fun out_j =>
                  (List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj)))))
              =
            (List.finRange out_w).foldl (fun acc out_j =>
                List.foldl fS acc
                  ((List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))) acc := by
          simpa using
            (foldl_flatMap (l := List.finRange out_w)
              (g := fun out_j =>
                (List.finRange kH).flatMap (fun di =>
                  (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))
              (f := fS) (init := acc))
        have hW :
            ∀ (acc : ℝ) (out_j : Fin out_w),
              List.foldl fS acc
                  ((List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))
                =
              (List.finRange kH).foldl (fun acc di =>
                  (List.finRange kW).foldl (fun acc dj => acc + termS (out_ch, out_i, out_j, di,
                    dj)) acc) acc := by
          intro acc out_j
          have hK' :
              List.foldl fS acc
                  ((List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj))))
                =
              (List.finRange kH).foldl (fun acc di =>
                  List.foldl fS acc ((List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di,
                    dj)))) acc := by
            simpa using
              (foldl_flatMap (l := List.finRange kH)
                (g := fun di => (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj)))
                (f := fS) (init := acc))
          have hWk :
              ∀ (acc : ℝ) (di : Fin kH),
                List.foldl fS acc ((List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di,
                  dj))) =
                  (List.finRange kW).foldl (fun acc dj => acc + termS (out_ch, out_i, out_j, di,
                    dj)) acc := by
            intro acc di
            exact
              (List.foldl_map (f := fun dj => (out_ch, out_i, out_j, di, dj)) (g := fS) (l :=
                List.finRange kW) (init := acc))
          simpa [hK'] using
            (foldl_congr (l := List.finRange kH)
              (f := fun acc di => List.foldl fS acc ((List.finRange kW).map (fun dj => (out_ch,
                out_i, out_j, di, dj))))
              (g := fun acc di => (List.finRange kW).foldl (fun acc dj => acc + termS (out_ch,
                out_i, out_j, di, dj)) acc)
              (init := acc) (by intro acc di; simpa using (hWk acc di)))
        simpa [hW'] using
          (foldl_congr (l := List.finRange out_w)
            (f := fun acc out_j =>
              List.foldl fS acc
                ((List.finRange kH).flatMap (fun di => (List.finRange kW).map (fun dj => (out_ch,
                  out_i, out_j, di, dj)))))
            (g := fun acc out_j =>
              (List.finRange kH).foldl (fun acc di =>
                  (List.finRange kW).foldl (fun acc dj => acc + termS (out_ch, out_i, out_j, di,
                    dj)) acc) acc)
            (init := acc) (by intro acc out_j; simpa using (hW acc out_j)))
      simpa [hH'] using
        (foldl_congr (l := List.finRange out_h)
          (f := fun acc out_i =>
            List.foldl fS acc
              ((List.finRange out_w).flatMap (fun out_j =>
                (List.finRange kH).flatMap (fun di => (List.finRange kW).map (fun dj => (out_ch,
                  out_i, out_j, di, dj))))))
          (g := fun acc out_i =>
            (List.finRange out_w).foldl (fun acc out_j =>
                (List.finRange kH).foldl (fun acc di =>
                    (List.finRange kW).foldl (fun acc dj => acc + termS (out_ch, out_i, out_j, di,
                      dj)) acc) acc) acc)
          (init := acc) (by intro acc out_i; simpa using (hH acc out_i)))
    have hC0' :
        idxs.foldl fS 0 =
          (List.finRange outC).foldl (fun acc out_ch =>
            (List.finRange out_h).foldl (fun acc out_i =>
                (List.finRange out_w).foldl (fun acc out_j =>
                    (List.finRange kH).foldl (fun acc di =>
                        (List.finRange kW).foldl (fun acc dj => acc + termS (out_ch, out_i, out_j,
                          di, dj)) acc) acc) acc) acc) 0 := by
      have :=
        foldl_congr (l := List.finRange outC)
          (f := fun acc out_ch =>
            List.foldl fS acc
              ((List.finRange out_h).flatMap (fun out_i =>
                (List.finRange out_w).flatMap (fun out_j =>
                  (List.finRange kH).flatMap (fun di =>
                    (List.finRange kW).map (fun dj => (out_ch, out_i, out_j, di, dj)))))))
          (g := fun acc out_ch =>
            (List.finRange out_h).foldl (fun acc out_i =>
                (List.finRange out_w).foldl (fun acc out_j =>
                    (List.finRange kH).foldl (fun acc di =>
                        (List.finRange kW).foldl (fun acc dj => acc + termS (out_ch, out_i, out_j,
                          di, dj)) acc) acc) acc) acc)
          (init := (0 : ℝ)) (by intro acc out_ch; simpa using (hOut acc out_ch))
      simpa [hC0] using this
    simpa [sumS, fS] using hC0'.symm

  have houtR :
      getAtOrZero (Spec.conv2dInputDerivSpec (α := R) (layer := layerR) (input := inputR)
        (grad_output := δR))
          [in_ch.val, i.val, j.val] = sumR := by
    let foldIfR : R :=
      (List.finRange outC).foldl (fun accC out_ch =>
          (List.finRange out_h).foldl (fun accH out_i =>
              (List.finRange out_w).foldl (fun accW out_j =>
                  (List.finRange kH).foldl (fun accKH di =>
                      (List.finRange kW).foldl (fun accKW dj =>
                          accKW +
                            if (out_i.val * stride + di.val = i.val + padding) ∧
                                (out_j.val * stride + dj.val = j.val + padding) then
                              getAtOrZero δR [out_ch.val, out_i.val, out_j.val] *
                                getAtOrZero kernelR [out_ch.val, in_ch.val, di.val, dj.val]
                            else 0)
                        accKH)
                    accW)
                accH)
            accC)
        (0 : R)
    have hGet :
        getAtOrZero
            (Spec.conv2dInputDerivSpec (α := R) (layer := layerR) (input := inputR) (grad_output
              := δR))
            [in_ch.val, i.val, j.val] = foldIfR := by
      have hEntry :=
        entry_eq_scalar_get_at_or_zero3
          (t := Spec.conv2dInputDerivSpec (α := R) (layer := layerR) (input := inputR)
            (grad_output := δR))
          in_ch i j
      have hSpecEntry :
          (match
              match
                match
                  Spec.conv2dInputDerivSpec (α := R) (layer := layerR) (input := inputR)
                    (grad_output := δR) with
                | Tensor.dim f => f in_ch with
              | Tensor.dim f => f i with
            | Tensor.dim f => f j) =
            Tensor.scalar foldIfR := by
        dsimp [Spec.conv2dInputDerivSpec, foldIfR, layerR, out_h, out_w, conv2dOutH,
          conv2dOutW]
        rfl
      have hTensor :
          Tensor.scalar foldIfR =
            Tensor.scalar
              (getAtOrZero
                (Spec.conv2dInputDerivSpec (α := R) (layer := layerR) (input := inputR)
                  (grad_output := δR))
                [in_ch.val, i.val, j.val]) := by
        exact hSpecEntry.symm.trans hEntry
      have :
          foldIfR =
            getAtOrZero
              (Spec.conv2dInputDerivSpec (α := R) (layer := layerR) (input := inputR)
                (grad_output := δR))
              [in_ch.val, i.val, j.val] := by
        simpa [Tensor.scalar.injEq] using hTensor
      simpa using this.symm
    have hFold : foldIfR = sumR := by
      simpa [foldIfR, termR] using hsumR_nested
    exact hGet.trans hFold

  have houtS :
      getAtOrZero (Spec.conv2dInputDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
        (grad_output := δS))
          [in_ch.val, i.val, j.val] = sumS := by
    let foldIfS : ℝ :=
      (List.finRange outC).foldl (fun accC out_ch =>
          (List.finRange out_h).foldl (fun accH out_i =>
              (List.finRange out_w).foldl (fun accW out_j =>
                  (List.finRange kH).foldl (fun accKH di =>
                      (List.finRange kW).foldl (fun accKW dj =>
                          accKW +
                            if (out_i.val * stride + di.val = i.val + padding) ∧
                                (out_j.val * stride + dj.val = j.val + padding) then
                              getAtOrZero δS [out_ch.val, out_i.val, out_j.val] *
                                getAtOrZero kernelS [out_ch.val, in_ch.val, di.val, dj.val]
                            else 0)
                        accKH)
                    accW)
                accH)
            accC)
        0
    have hGet :
        getAtOrZero
            (Spec.conv2dInputDerivSpec (α := ℝ) (layer := layerS) (input := inputS) (grad_output
              := δS))
            [in_ch.val, i.val, j.val] = foldIfS := by
      have hEntry :=
        entry_eq_scalar_get_at_or_zero3
          (t := Spec.conv2dInputDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
            (grad_output := δS))
          in_ch i j
      have hSpecEntry :
          (match
              match
                match
                  Spec.conv2dInputDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
                    (grad_output := δS) with
                | Tensor.dim f => f in_ch with
              | Tensor.dim f => f i with
            | Tensor.dim f => f j) =
            Tensor.scalar foldIfS := by
        dsimp [Spec.conv2dInputDerivSpec, foldIfS, layerS, out_h, out_w, conv2dOutH,
          conv2dOutW]
        rfl
      have hTensor :
          Tensor.scalar foldIfS =
            Tensor.scalar
              (getAtOrZero
                (Spec.conv2dInputDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
                  (grad_output := δS))
                [in_ch.val, i.val, j.val]) := by
        exact hSpecEntry.symm.trans hEntry
      have :
          foldIfS =
            getAtOrZero
              (Spec.conv2dInputDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
                (grad_output := δS))
              [in_ch.val, i.val, j.val] := by
        simpa [Tensor.scalar.injEq] using hTensor
      simpa using this.symm
    have hFold : foldIfS = sumS := by
      simpa [foldIfS, termS] using hsumS_nested
    exact hGet.trans hFold

  have hFinal :
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (getAtOrZero
                (Spec.conv2dInputDerivSpec (α := R) (layer := layerR) (input := inputR)
                  (grad_output := δR))
                [in_ch.val, i.val, j.val]) -
            getAtOrZero
              (Spec.conv2dInputDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
                (grad_output := δS))
              [in_ch.val, i.val, j.val]) ≤ sumEps := by
    simpa [houtR, houtS] using hSum

  have hSumEps :
      sumEps =
        conv2dInputPointBound (β := β) (fexp := fexp) (rnd := rnd)
          (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
            (inH := inH)
          (inW := inW) kernelR δR epsK epsδ in_ch i j := by
    simp [sumEps, conv2dInputPointBound, out_h, out_w, idxs, termR, epsTerm]
  simpa [hSumEps] using hFinal

-- ---------------------------------------------------------------------------
-- Tensor-level backward bound (input gradient)
-- ---------------------------------------------------------------------------

/--
Tensor-level `approxTensor` bound for the Conv2d **input** gradient.

This lifts `approx_conv2d_input_point` entrywise and packages the error into
`linfNorm (conv2dInputBoundTensor ...)`.
-/
theorem approxTensor_conv2d_input_deriv_spec
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
    {epsK epsδ : ℝ}
    (hK : approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) kernelS kernelR
      epsK)
    (hδ : approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) δS δR epsδ) :
    let layerS : Spec.Conv2dSpec inC outC kH kW stride padding ℝ h1 h2 h3 :=
      { kernel := kernelS, bias := biasS }
    let layerR : Spec.Conv2dSpec inC outC kH kW stride padding R h1 h2 h3 :=
      { kernel := kernelR, bias := biasR }
    let outS := Spec.conv2dInputDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
      (grad_output := δS)
    let outR := Spec.conv2dInputDerivSpec (α := R) (layer := layerR) (input := inputR)
      (grad_output := δR)
    let bT :=
        conv2dInputBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
            (inH := inH) (inW := inW)
          kernelR δR epsK epsδ
    approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) outS outR (linfNorm
      bT) := by
  intro layerS layerR outS outR bT
  classical
  have hε : 0 ≤ linfNorm bT := linf_norm_nonneg (t := bT)
  refine approxTensor_dim_of_forall (n := inC)
    (s := .dim inH (.dim inW .scalar))
    (xS := outS) (xR := outR) (eps := linfNorm bT) hε ?_
  intro ic
  refine approxTensor_dim_of_forall (n := inH)
    (s := .dim inW .scalar)
    (xS := (match outS with | .dim f => f ic))
    (xR := (match outR with | .dim f => f ic))
    (eps := linfNorm bT) hε ?_
  intro ii
  refine approxTensor_dim_of_forall (n := inW)
    (s := .scalar)
    (xS := (match (match outS with | .dim f => f ic) with | .dim g => g ii))
    (xR := (match (match outR with | .dim f => f ic) with | .dim g => g ii))
    (eps := linfNorm bT) hε ?_
  intro jj

  have hpt :=
    approx_conv2d_input_point (β := β) (fexp := fexp) (rnd := rnd)
      (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding) (inH
        := inH) (inW := inW)
      (h1 := h1) (h2 := h2) (h3 := h3)
      (kernelS := kernelS) (kernelR := kernelR) (biasS := biasS) (biasR := biasR)
      (inputS := inputS) (inputR := inputR) (δS := δS) (δR := δR)
      (epsK := epsK) (epsδ := epsδ) hK hδ ic ii jj

  have hEntry :
      abs (conv2dInputPointBound (β := β) (fexp := fexp) (rnd := rnd)
          (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
            (inH := inH) (inW := inW)
          kernelR δR epsK epsδ ic ii jj) ≤ linfNorm bT := by
    let bTic := match bT with | .dim f => f ic
    let bTicii := match bTic with | .dim f => f ii
    let bTiciijj := match bTicii with | .dim f => f jj
    have h0 : linfNorm bTic ≤ linfNorm bT := by
      change linfNorm (match bT with | .dim f => f ic) ≤ linfNorm bT
      exact linf_norm_le_get_dim (t := bT) ic
    have h1' : linfNorm bTicii ≤ linfNorm bTic := by
      change linfNorm (match bTic with | .dim f => f ii) ≤ linfNorm bTic
      exact linf_norm_le_get_dim (t := bTic) ii
    have h2' : linfNorm bTiciijj ≤ linfNorm bTicii := by
      change linfNorm (match bTicii with | .dim f => f jj) ≤ linfNorm bTicii
      exact linf_norm_le_get_dim (t := bTicii) jj
    have hchain : linfNorm bTiciijj ≤ linfNorm bT :=
      le_trans (le_trans h2' h1') h0
    let bound :=
      conv2dInputPointBound (β := β) (fexp := fexp) (rnd := rnd)
        (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
          (inH := inH) (inW := inW)
        kernelR δR epsK epsδ ic ii jj
    have hdouble' : (MathFunctions.abs (abs bound) : ℝ) ≤ linfNorm bT := by
      simpa [bound, bT, bTic, bTicii, bTiciijj, conv2dInputBoundTensor, linfNorm,
        RuntimeApprox.linfNorm,
        tensorLinfNorm] using hchain
    have habs : (MathFunctions.abs (abs bound) : ℝ) = abs (abs bound) := by
      rfl
    have hdouble : abs (abs bound) ≤ linfNorm bT := by
      simpa [habs] using hdouble'
    simpa [bound, abs_abs] using hdouble

  have hscalar :
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (getAtOrZero outR [ic.val, ii.val, jj.val]) -
            getAtOrZero outS [ic.val, ii.val, jj.val]) ≤ linfNorm bT :=
    le_trans (le_trans hpt (le_abs_self _)) hEntry

  let entryS : Tensor ℝ .scalar :=
    (match match match outS with
      | .dim f => f ic with
    | .dim g => g ii with
    | .dim h => h jj)
  let entryR : Tensor R .scalar :=
    (match match match outR with
      | .dim f => f ic with
    | .dim g => g ii with
    | .dim h => h jj)
  change
    approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) entryS entryR
      (linfNorm bT)
  have hEntryS :
      entryS = Tensor.scalar (getAtOrZero outS [ic.val, ii.val, jj.val]) := by
    exact entry_eq_scalar_get_at_or_zero3 (t := outS) ic ii jj
  have hEntryR :
      entryR = Tensor.scalar (getAtOrZero outR [ic.val, ii.val, jj.val]) := by
    exact entry_eq_scalar_get_at_or_zero3 (t := outR) ic ii jj
  have happ :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Tensor.scalar (getAtOrZero outS [ic.val, ii.val, jj.val]))
        (Tensor.scalar (getAtOrZero outR [ic.val, ii.val, jj.val]))
        (linfNorm bT) :=
    (approxTensor_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (x := getAtOrZero outS [ic.val, ii.val, jj.val]) (xR := getAtOrZero outR [ic.val,
          ii.val, jj.val])
        (eps := linfNorm bT)).2 (by
          simpa using hscalar)
  simpa [hEntryS, hEntryR] using happ

-- ---------------------------------------------------------------------------
-- `RevNode` packaging for Conv2d.
-- ---------------------------------------------------------------------------

lemma idx_i_ne_of_shape_ne {Γ : List Shape} {s₁ s₂ : Shape} (a : Idx Γ s₁) (b : Idx Γ s₂) (hs : s₁ ≠
  s₂) :
    a.i ≠ b.i := by
  intro hEq
  apply hs
  exact idx_shape_eq_of_i_eq (a := a) (b := b) hEq


end NFBackend

end
end RuntimeApprox
end Proofs
