/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.ConvBackward.Input

/-!
# Conv2D Backward as a RuntimeApprox Reverse Node

The previous files prove pointwise bounds for the bias, kernel, and input-gradient pieces. This file
packages those pieces into the `RevNode` interface used by the generic reverse-graph approximation
theorems, so Conv2D backward can participate in end-to-end NeuralFloat runtime proofs.
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
Package Conv2D forward + backward (VJP) as a `RevNode` for `RevGraph.backprop_approx`.

The node uses the forward bound from `conv2d_forward` and the three gradient bounds proved in this
file (kernel/bias/input) to provide a compositional reverse-mode approximation theorem.
-/
def conv2dRevNode
    {Γ : List Shape}
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (kernelIdx : Idx Γ (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))))
    (biasIdx : Idx Γ (.dim outC .scalar))
    (inputIdx : Idx Γ (.dim inC (.dim inH (.dim inW .scalar)))) :
    RevNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ
      (.dim outC (.dim (conv2dOutH inH kH stride padding) (.dim (conv2dOutW inW kW stride padding)
        .scalar))) :=
by
  classical
  have hShapeKB :
      (Shape.dim outC (Shape.dim inC (Shape.dim kH (Shape.dim kW Shape.scalar)))) ≠
        (Shape.dim outC Shape.scalar) := by
    intro h
    injection h with _ hRest
    cases hRest
  have hShapeKX :
      (Shape.dim outC (Shape.dim inC (Shape.dim kH (Shape.dim kW Shape.scalar)))) ≠
        (Shape.dim inC (Shape.dim inH (Shape.dim inW Shape.scalar))) := by
    intro h
    injection h with _ hRest1
    injection hRest1 with _ hRest2
    injection hRest2 with _ hRest3
    cases hRest3
  have hShapeBX :
      (Shape.dim outC Shape.scalar) ≠ (Shape.dim inC (Shape.dim inH (Shape.dim inW Shape.scalar)))
        := by
    intro h
    injection h with _ hRest
    cases hRest
  have hab : kernelIdx.i ≠ biasIdx.i :=
    idx_i_ne_of_shape_ne (a := kernelIdx) (b := biasIdx) hShapeKB
  have hac : kernelIdx.i ≠ inputIdx.i :=
    idx_i_ne_of_shape_ne (a := kernelIdx) (b := inputIdx) hShapeKX
  have hbc : biasIdx.i ≠ inputIdx.i :=
    idx_i_ne_of_shape_ne (a := biasIdx) (b := inputIdx) hShapeBX
  refine
    { toFwdNode := conv2dNode (β := β) (fexp := fexp) (rnd := rnd)
        (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
          (inH := inH) (inW := inW)
        (h1 := h1) (h2 := h2) (h3 := h3)
        kernelIdx biasIdx inputIdx
      vjpSpec := fun ctx δ =>
        let kernelS := getIdx (α := SpecScalar) ctx kernelIdx
        let biasS := getIdx (α := SpecScalar) ctx biasIdx
        let inputS := getIdx (α := SpecScalar) ctx inputIdx
        let layerS : Spec.Conv2DSpec inC outC kH kW stride padding ℝ h1 h2 h3 :=
          { kernel := kernelS, bias := biasS }
        let dK := Spec.conv2dKernelDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
          (grad_output := δ)
        let dB := Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
          (grad_output := δ)
        let dX := Spec.conv2dInputDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
          (grad_output := δ)
        TList.set3IdxNe (α := SpecScalar) (Γ := Γ) (s₁ := (.dim outC (.dim inC (.dim kH (.dim kW
          .scalar)))))
          (s₂ := (.dim outC .scalar)) (s₃ := (.dim inC (.dim inH (.dim inW .scalar))))
          kernelIdx dK biasIdx dB inputIdx dX hab hac hbc
      vjpRuntime := fun ctx δ =>
        let kernelR := getIdx (α := R) ctx kernelIdx
        let biasR := getIdx (α := R) ctx biasIdx
        let inputR := getIdx (α := R) ctx inputIdx
        let layerR : Spec.Conv2DSpec inC outC kH kW stride padding R h1 h2 h3 :=
          { kernel := kernelR, bias := biasR }
        let dK := Spec.conv2dKernelDerivSpec (α := R) (layer := layerR) (input := inputR)
          (grad_output := δ)
        let dB := Spec.conv2dBiasDerivSpec (α := R) (layer := layerR) (input := inputR)
          (grad_output := δ)
        let dX := Spec.conv2dInputDerivSpec (α := R) (layer := layerR) (input := inputR)
          (grad_output := δ)
        TList.set3IdxNe (α := R) (Γ := Γ) (s₁ := (.dim outC (.dim inC (.dim kH (.dim kW
          .scalar)))))
          (s₂ := (.dim outC .scalar)) (s₃ := (.dim inC (.dim inH (.dim inW .scalar))))
          kernelIdx dK biasIdx dB inputIdx dX hab hac hbc
      vjpBound := fun epsCtx ctxR epsδ δR =>
        let kernelR := getIdx (α := R) ctxR kernelIdx
        let inputR := getIdx (α := R) ctxR inputIdx
        let epsK := getIdxEps (Γ := Γ) (s := (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))))
          epsCtx kernelIdx
        let epsX := getIdxEps (Γ := Γ) (s := (.dim inC (.dim inH (.dim inW .scalar)))) epsCtx
          inputIdx
          let epsDK :=
            linfNorm
              (conv2dKernelBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding :=
                  padding) (inH := inH) (inW := inW)
                inputR δR epsX epsδ)
          let epsDB :=
            linfNorm
              (conv2dBiasBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding) (inH :=
                  inH) (inW := inW)
                δR epsδ)
          let epsDX :=
            linfNorm
              (conv2dInputBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding :=
                  padding) (inH := inH) (inW := inW)
                kernelR δR epsK epsδ)
        EList.set3IdxNe (Γ := Γ) (s₁ := (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))))
          (s₂ := (.dim outC .scalar)) (s₃ := (.dim inC (.dim inH (.dim inW .scalar))))
          kernelIdx epsDK biasIdx epsDB inputIdx epsDX hab hac hbc
      vjpSound := ?_ }
  intro ctxS ctxR epsCtx δS δR epsδ hctx hδ
  have hK :=
    approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx kernelIdx
  have hB :=
    approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx biasIdx
  have hX :=
    approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx inputIdx
  -- Unpack the context tensors once, to share with both spec and runtime sides.
  let kernelS := getIdx (α := SpecScalar) ctxS kernelIdx
  let kernelR := getIdx (α := R) ctxR kernelIdx
  let biasS := getIdx (α := SpecScalar) ctxS biasIdx
  let biasR := getIdx (α := R) ctxR biasIdx
  let inputS := getIdx (α := SpecScalar) ctxS inputIdx
  let inputR := getIdx (α := R) ctxR inputIdx
  let layerS : Spec.Conv2DSpec inC outC kH kW stride padding ℝ h1 h2 h3 :=
    { kernel := kernelS, bias := biasS }
  let layerR : Spec.Conv2DSpec inC outC kH kW stride padding R h1 h2 h3 :=
    { kernel := kernelR, bias := biasR }
  let epsK := getIdxEps (Γ := Γ) (s := (.dim outC (.dim inC (.dim kH (.dim kW .scalar))))) epsCtx
    kernelIdx
  let epsX := getIdxEps (Γ := Γ) (s := (.dim inC (.dim inH (.dim inW .scalar)))) epsCtx inputIdx
  have hBias :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layerS) (input := inputS) (grad_output :=
          δS))
      (Spec.conv2dBiasDerivSpec (α := R) (layer := layerR) (input := inputR) (grad_output :=
        δR))
      (linfNorm
        (conv2dBiasBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding) (inH :=
                inH) (inW := inW)
              δR epsδ)) := by
    -- `bias_deriv` depends only on `δ`, but we keep the same `layer`/`input` arguments.
    simpa [kernelS, kernelR, biasS, biasR, inputS, inputR, layerS, layerR] using
      (approxT_conv2d_bias_deriv_spec (β := β) (fexp := fexp) (rnd := rnd)
        (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
          (inH := inH) (inW := inW)
        (h1 := h1) (h2 := h2) (h3 := h3)
        (kernelS := kernelS) (kernelR := kernelR) (biasS := biasS) (biasR := biasR)
        (inputS := inputS) (inputR := inputR) (δS := δS) (δR := δR) (epsδ := epsδ) hδ)
  have hKernel :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.conv2dKernelDerivSpec (α := ℝ) (layer := layerS) (input := inputS) (grad_output :=
          δS))
      (Spec.conv2dKernelDerivSpec (α := R) (layer := layerR) (input := inputR) (grad_output :=
        δR))
      (linfNorm
        (conv2dKernelBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
            (inH := inH) (inW := inW)
          inputR δR epsX epsδ)) := by
    have hX' : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) inputS
      inputR epsX := by
      simpa [epsX] using hX
    simpa [kernelS, kernelR, biasS, biasR, inputS, inputR, layerS, layerR] using
      (approxT_conv2d_kernel_deriv_spec (β := β) (fexp := fexp) (rnd := rnd)
        (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
          (inH := inH) (inW := inW)
        (h1 := h1) (h2 := h2) (h3 := h3)
        (kernelS := kernelS) (kernelR := kernelR) (biasS := biasS) (biasR := biasR)
        (inputS := inputS) (inputR := inputR) (δS := δS) (δR := δR) (epsX := epsX) (epsδ := epsδ)
        hX' hδ)
  have hInput :
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.conv2dInputDerivSpec (α := ℝ) (layer := layerS) (input := inputS) (grad_output :=
          δS))
      (Spec.conv2dInputDerivSpec (α := R) (layer := layerR) (input := inputR) (grad_output :=
        δR))
      (linfNorm
        (conv2dInputBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
            (inH := inH) (inW := inW)
          kernelR δR epsK epsδ)) := by
    have hK' : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) kernelS
      kernelR epsK := by
      simpa [epsK] using hK
    simpa [kernelS, kernelR, biasS, biasR, inputS, inputR, layerS, layerR] using
      (approxT_conv2d_input_deriv_spec (β := β) (fexp := fexp) (rnd := rnd)
        (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
          (inH := inH) (inW := inW)
        (h1 := h1) (h2 := h2) (h3 := h3)
        (kernelS := kernelS) (kernelR := kernelR) (biasS := biasS) (biasR := biasR)
        (inputS := inputS) (inputR := inputR) (δS := δS) (δR := δR) (epsK := epsK) (epsδ := epsδ)
        hK' hδ)
  have hctx' :=
    approxCtx_set3Idx_ne (β := β) (fexp := fexp) (rnd := rnd)
      (Γ := Γ)
      (s₁ := (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))))
      (s₂ := (.dim outC .scalar))
      (s₃ := (.dim inC (.dim inH (.dim inW .scalar))))
      kernelIdx biasIdx inputIdx
      (t₁S := Spec.conv2dKernelDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
        (grad_output := δS))
      (t₁R := Spec.conv2dKernelDerivSpec (α := R) (layer := layerR) (input := inputR)
        (grad_output := δR))
        (eps₁ :=
          linfNorm
            (conv2dKernelBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding :=
                padding) (inH := inH) (inW := inW)
              inputR δR epsX epsδ))
      (t₂S := Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
        (grad_output := δS))
      (t₂R := Spec.conv2dBiasDerivSpec (α := R) (layer := layerR) (input := inputR)
        (grad_output := δR))
        (eps₂ :=
          linfNorm
            (conv2dBiasBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding) (inH :=
                inH) (inW := inW)
              δR epsδ))
      (t₃S := Spec.conv2dInputDerivSpec (α := ℝ) (layer := layerS) (input := inputS)
        (grad_output := δS))
      (t₃R := Spec.conv2dInputDerivSpec (α := R) (layer := layerR) (input := inputR)
        (grad_output := δR))
        (eps₃ :=
          linfNorm
            (conv2dInputBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding :=
                padding) (inH := inH) (inW := inW)
              kernelR δR epsK epsδ))
      hKernel hBias hInput hab hac hbc
  -- Goal matches the `set3Idx_ne` packaging in `vjpSpec/vjpRuntime/vjpBound`.
  simpa [kernelS, kernelR, biasS, biasR, inputS, inputR, layerS, layerR, epsK, epsX] using hctx'

end NFBackend

end
end RuntimeApprox
end Proofs
