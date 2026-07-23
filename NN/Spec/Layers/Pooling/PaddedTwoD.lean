/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Pooling.TwoD

@[expose] public section


namespace Spec
open Tensor
open Spec (Image MultiChannelImage getValueAtPosition extractWindow)

variable {α : Type} [Context α]

/-!
# Padded 2D Pooling

Symmetric-padding variants of the 2D pooling specs and their backward maps.
-/

/-!
## Padded pooling (symmetric padding)

For max-pooling, padded locations are not real input elements and are ignored when selecting the
maximum. This is the scalar-polymorphic way to model PyTorch's `-∞` max-pool padding without adding
a backend-specific infinity constant to `Context α`.

For average pooling, this corresponds to including padded zeros in the average (PyTorch's default
`count_include_pad = true`).
-/

/-- Remove symmetric zero-padding from a single-channel image. -/
def unpadImage {α : Type} [Context α] {H W padding : ℕ}
    (img : Image (H + 2 * padding) (W + 2 * padding) α) : Image H W α :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      have hi0 : i.val + padding < H + padding := Nat.add_lt_add_right i.isLt padding
      have hj0 : j.val + padding < W + padding := Nat.add_lt_add_right j.isLt padding
      have hleH : H + padding ≤ H + 2 * padding := by
        have hle : padding ≤ 2 * padding := by
          simp [two_mul]
        exact Nat.add_le_add_left hle H
      have hleW : W + padding ≤ W + 2 * padding := by
        have hle : padding ≤ 2 * padding := by
          simp [two_mul]
        exact Nat.add_le_add_left hle W
      have hi : i.val + padding < H + 2 * padding := Nat.lt_of_lt_of_le hi0 hleH
      have hj : j.val + padding < W + 2 * padding := Nat.lt_of_lt_of_le hj0 hleW
      getAtSpec (getAtSpec img ⟨i.val + padding, hi⟩) ⟨j.val + padding, hj⟩))

/-- Remove symmetric zero-padding from a multi-channel image (channel-wise `unpad_image`). -/
def unpadMultiChannel {α : Type} [Context α] {C H W padding : ℕ}
    (img : MultiChannelImage C (H + 2 * padding) (W + 2 * padding) α) : MultiChannelImage C H W α :=
  Tensor.dim (fun c =>
    unpadImage (α := α) (H := H) (W := W) (padding := padding) (getAtSpec img c))

/-- Multi-channel max-pooling forward pass with PyTorch-style padding (`-∞` outside bounds). -/
def maxPool2dMultiSpecPad {kH kW inH inW inC stride padding : ℕ} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
    {hStride : stride ≠ 0}
    (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
    (input : MultiChannelImage inC inH inW α) :
    MultiChannelImage inC (Shape.slidingWindowOutDim inH kH stride padding)
      (Shape.slidingWindowOutDim inW kW stride padding) α :=
  Tensor.dim (fun c =>
    Tensor.dim (fun oh =>
      Tensor.dim (fun ow =>
        let best? :=
          (List.range kH).foldl (fun rowBest ky =>
            (List.range kW).foldl (fun best kx =>
              let ph := oh.val * layer.stride + ky
              let pw := ow.val * layer.stride + kx
              if ph < padding then
                best
              else if pw < padding then
                best
              else
                let ih := ph - padding
                let iw := pw - padding
                if hIh : ih < inH then
                  if hIw : iw < inW then
                    let valT := getAtSpec (getAtSpec (getAtSpec input c) ⟨ih, hIh⟩) ⟨iw, hIw⟩
                    match valT with
                    | Tensor.scalar v =>
                        match best with
                        | none => some v
                        | some b => if v > b then some v else some b
                  else best
                else best
            ) rowBest
          ) none
        Tensor.scalar (best?.getD 0))))

/--
Selected-branch linearization for padded hard max-pooling.

Padding cells are ignored exactly as in `maxPool2dMultiSpecPad`, so the tangent is taken from the
primal winner among real input locations only. If a window contains no real input cells, the
forward value and tangent are both `0`.
-/
def maxPool2dMultiLinearizationSpecPad {kH kW inH inW inC stride padding : ℕ} {h1 : kH ≠ 0}
    {h2 : kW ≠ 0} {hStride : stride ≠ 0}
    (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
    (input tangent : MultiChannelImage inC inH inW α) :
    MultiChannelImage inC (Shape.slidingWindowOutDim inH kH stride padding)
      (Shape.slidingWindowOutDim inW kW stride padding) α :=
  Tensor.dim (fun c =>
    Tensor.dim (fun oh =>
      Tensor.dim (fun ow =>
        let best? :=
          (List.range kH).foldl (fun rowBest ky =>
            (List.range kW).foldl (fun best kx =>
              let ph := oh.val * layer.stride + ky
              let pw := ow.val * layer.stride + kx
              if ph < padding then
                best
              else if pw < padding then
                best
              else
                let ih := ph - padding
                let iw := pw - padding
                if hIh : ih < inH then
                  if hIw : iw < inW then
                    let valT := getAtSpec (getAtSpec (getAtSpec input c) ⟨ih, hIh⟩) ⟨iw, hIw⟩
                    match valT with
                    | Tensor.scalar v =>
                        match best with
                        | none => some (ih, iw, v)
                        | some (_, _, b) => if v > b then some (ih, iw, v) else best
                  else best
                else best
            ) rowBest
          ) none
        match best? with
        | none => Tensor.scalar 0
        | some (ih, iw, _) =>
            if hIh : ih < inH then
              if hIw : iw < inW then
                getAtSpec (getAtSpec (getAtSpec tangent c) ⟨ih, hIh⟩) ⟨iw, hIw⟩
              else
                Tensor.scalar 0
            else
              Tensor.scalar 0)))

/-- Multi-channel average pooling forward pass with symmetric zero padding. -/
def avgPool2dMultiSpecPad {kH kW inH inW inC stride padding : ℕ} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
    {hStride : stride ≠ 0}
    (layer : AvgPool2DSpec kH kW stride h1 h2 hStride)
    (input : MultiChannelImage inC inH inW α) :
    MultiChannelImage inC (Shape.slidingWindowOutDim inH kH stride padding)
      (Shape.slidingWindowOutDim inW kW stride padding) α := by
  -- PyTorch note: this matches `count_include_pad=true` (the padded zeros are part of the average).
  let inputPad : MultiChannelImage inC (inH + 2 * padding) (inW + 2 * padding) α :=
    padMultiChannel (inC := inC) (inH := inH) (inW := inW) input padding
  simpa [Shape.slidingWindowOutDim, Nat.add_assoc] using
    avgPool2dMultiSpec (h1 := h1) (h2 := h2) (layer := layer) (input := inputPad)

/-- Multi-channel max-pooling backward pass with PyTorch-style padding (`-∞` outside bounds). -/
def maxPool2dMultiBackwardSpecPad {kH kW inH inW inC stride padding : ℕ} {h1 : kH ≠ 0} {h2 : kW
  ≠ 0}
    {hStride : stride ≠ 0}
    (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
    (input : MultiChannelImage inC inH inW α)
    (grad_output :
    MultiChannelImage inC (Shape.slidingWindowOutDim inH kH stride padding)
      (Shape.slidingWindowOutDim inW kW stride padding) α) :
    MultiChannelImage inC inH inW α :=
  let outH := Shape.slidingWindowOutDim inH kH stride padding
  let outW := Shape.slidingWindowOutDim inW kW stride padding
  let grad_init : MultiChannelImage inC inH inW α :=
    Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.scalar 0)))
  (List.range inC).foldl (fun acc cNat =>
    if hC : cNat < inC then
      (List.range outH).foldl (fun accH oh =>
        (List.range outW).foldl (fun accW ow =>
          let best? :=
            (List.range kH).foldl (fun rowBest ky =>
              (List.range kW).foldl (fun best kx =>
                let ph := oh * layer.stride + ky
                let pw := ow * layer.stride + kx
                if ph < padding then
                  best
                else if pw < padding then
                  best
                else
                  let ih := ph - padding
                  let iw := pw - padding
                  if hIh : ih < inH then
                    if hIw : iw < inW then
                      let valT := getAtSpec (getAtSpec (getAtSpec input ⟨cNat, hC⟩) ⟨ih, hIh⟩)
                        ⟨iw, hIw⟩
                      match valT with
                      | Tensor.scalar v =>
                          match best with
                          | none => some (ih, iw, v)
                          | some (_, _, b) => if v > b then some (ih, iw, v) else best
                    else best
                  else best
              ) rowBest
            ) none
          match best? with
          | none => accW
          | some (ih, iw, _) =>
              if hOh : oh < outH then
                if hOw : ow < outW then
                  let gOutT :=
                    getAtSpec (getAtSpec (getAtSpec grad_output ⟨cNat, hC⟩) ⟨oh, hOh⟩)
                      ⟨ow, hOw⟩
                  match gOutT with
                  | Tensor.scalar gOut =>
                      let idx := [cNat, ih, iw]
                      let current : α := getAtOrZero accW idx
                      updateTensorSpec accW idx (current + gOut)
                else accW
              else accW
        ) accH
      ) acc
    else acc
  ) grad_init

/-- Multi-channel average-pooling backward pass with symmetric padding (backprop then unpad). -/
def avgPool2dMultiBackwardSpecPad {kH kW inH inW inC stride padding : ℕ} (h1 : kH ≠ 0) (h2 : kW
  ≠ 0)
    {hStride : stride ≠ 0}
    (layer : AvgPool2DSpec kH kW stride h1 h2 hStride)
    (grad_output :
      MultiChannelImage inC (Shape.slidingWindowOutDim inH kH stride padding)
        (Shape.slidingWindowOutDim inW kW stride padding) α) :
    MultiChannelImage inC inH inW α := by
  let gradOutputPadded :
      MultiChannelImage inC
        (Shape.slidingWindowOutDim (inH + 2 * padding) kH stride 0)
        (Shape.slidingWindowOutDim (inW + 2 * padding) kW stride 0) α := by
    simpa [Shape.slidingWindowOutDim, Nat.add_assoc] using grad_output
  let gradPad : MultiChannelImage inC (inH + 2 * padding) (inW + 2 * padding) α :=
    Tensor.dim (fun c =>
      avgPool2dBackwardSpec (α := α) h1 h2 layer (getAtSpec gradOutputPadded c))
  exact unpadMultiChannel (α := α) (C := inC) (H := inH) (W := inW) (padding := padding) gradPad

-- Smooth max pooling backward pass (log-sum-exp surrogate)
/-- Backward/VJP for `smooth_max_pool2d_spec` (log-sum-exp surrogate). -/
def smoothMaxPool2dBackwardSpec {kH kW inH inW stride : ℕ} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  {hStride : stride ≠ 0}
  (_layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
  (beta : α)
  (input : Image inH inW α)
  (grad_output : Image (Shape.slidingWindowOutDim inH kH stride 0) (Shape.slidingWindowOutDim inW kW stride 0) α) :
  Image inH inW α :=
  -- This is the VJP of the log-sum-exp surrogate:
  --   smooth_max(x) = (1/beta) * log(sum(exp(beta*x))).
  -- The gradient distributes `grad_output` using algebraically equivalent shifted softmax
  -- weights, avoiding overflow in `exp(beta*x)`.
  let input_grad_init : Image inH inW α := createZeroImage inH inW
  let outH := Shape.slidingWindowOutDim inH kH stride 0
  let outW := Shape.slidingWindowOutDim inW kW stride 0
  let coeff : α := 1
  (List.finRange outH).foldl (fun acc_grad (out_i : Fin outH) =>
    (List.finRange outW).foldl (fun acc_grad_inner (out_j : Fin outW) =>
      let window := extractWindow kW kH input (out_i.val * stride) (out_j.val * stride)
      let pivot := Private.smoothMaxPool2dPivot h1 h2 beta window
      let expWindow :=
        mapSpec (s := Shape.dim kH (Shape.dim kW Shape.scalar))
          (fun x => MathFunctions.exp (beta * (x - pivot))) window
      have instH : Shape.valid_axis_inst 0 (Shape.dim kH (Shape.dim kW Shape.scalar)) :=
        Shape.validAxisInstZeroAlt h1
      have instW : Shape.valid_axis_inst 0 (Shape.dim kW Shape.scalar) :=
        Shape.validAxisInstZeroAlt h2
      let sumH := reduceSumAuto 0 expWindow
      have hcast : (Shape.dim kW Shape.scalar) = shapeAfterSum (Shape.dim kH (Shape.dim kW
        Shape.scalar)) 0 := by
        simp [shapeAfterSum]
      let sumH' := tensorCast (Shape.dim kW Shape.scalar) hcast.symm sumH
      let sumAll := reduceSumAuto 0 sumH'
      match sumAll with
      | Tensor.scalar sumExp =>
          let gOut := getAtSpec (getAtSpec grad_output out_i) out_j
          -- Distribute gradient over the pooling window.
          (List.finRange kH).foldl (fun acc_di (di : Fin kH) =>
            (List.finRange kW).foldl (fun acc_dj (dj : Fin kW) =>
              let inp_i := out_i.val * stride + di.val
              let inp_j := out_j.val * stride + dj.val
              if h_inp_i : inp_i < inH then
                if h_inp_j : inp_j < inW then
                  let expVal := getAtSpec (getAtSpec expWindow di) dj
                  match expVal with
                  | Tensor.scalar eVal =>
                      let w : α := coeff * (eVal / sumExp)
                      let contrib : Tensor α .scalar :=
                        match gOut with
                        | Tensor.scalar g => Tensor.scalar (g * w)
                      let current := getAtSpec (getAtSpec acc_dj ⟨inp_i, h_inp_i⟩) ⟨inp_j,
                        h_inp_j⟩
                      let new := addSpec current contrib
                      updateTensorSpec acc_dj [inp_i, inp_j] (Tensor.toScalar new)
                else acc_dj
              else acc_dj
            ) acc_di
          ) acc_grad_inner
    ) acc_grad
  ) input_grad_init

/-- Multi-channel backward for `smooth_max_pool2d_multi_spec` (apply per channel). -/
def smoothMaxPool2dMultiBackwardSpec {kH kW inH inW inC stride : ℕ} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  {hStride : stride ≠ 0}
  (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
  (beta : α)
  (input : MultiChannelImage inC inH inW α)
  (grad_output : MultiChannelImage inC (Shape.slidingWindowOutDim inH kH stride 0) (Shape.slidingWindowOutDim inW kW stride 0) α) :
  MultiChannelImage inC inH inW α :=
  Tensor.dim (fun c =>
    smoothMaxPool2dBackwardSpec (_layer := layer) (beta := beta)
      (input := getAtSpec input c) (grad_output := getAtSpec grad_output c))
end Spec
