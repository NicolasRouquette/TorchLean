/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import NN.Runtime.Autograd.Engine.Core.Linear

/-!
# Core Tape Convolution and Pooling

This file implements the pure tape nodes for convolution, transposed convolution, and pooling. These
nodes are backend-independent: they record forward values, parents, and backward closures using the
spec-layer definitions before CUDA or compiled backends enter the picture.
-/

@[expose] public section

namespace Runtime
namespace Autograd

open Spec
open Tensor

namespace Tape

/--
N-D convolution for channels-first tensors `(inC, spatial...)` (no batch axis).

This is the generic counterpart to `conv2d`; `conv2d` is implemented as a specialization with
`d = 2`, scalar stride, and scalar padding.
-/
def conv {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  (t : Tape α) (kernelId biasId inputId : Nat) (name : String := "conv") :
  Result (Tape α × Nat) := do
  let k ← requireValue (α:=α) (t:=t)
    (s:=Shape.ofList (outC :: inC :: kernel.toList)) kernelId
  let b ← requireValue (α:=α) (t:=t) (s:=.dim outC .scalar) biasId
  let x ← requireValue (α:=α) (t:=t)
    (s:=Shape.ofList (inC :: inSpatial.toList)) inputId
  let layer : Spec.ConvSpec d inC outC kernel stride padding α :=
    { kernel := k, bias := b }
  let y := Spec.convSpec (layer := layer) x
  let outSpatial := Spec.convOutSpatial inSpatial kernel stride padding
  let outSh : Shape := Shape.ofList (outC :: outSpatial.toList)
  let node : Node α :=
    { name := some name
      value := AnyTensor.mk y
      requires_grad := true
      parents := [kernelId, biasId, inputId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := outSh) dLdyAny
        let (dK, dB, dX) := Spec.convBackwardSpec (layer := layer) x dLdy
        pure [
          (kernelId, AnyTensor.mk dK),
          (biasId, AnyTensor.mk dB),
          (inputId, AnyTensor.mk dX)
        ]
    }
  pure (t.addNode node)

/--
2D convolution for channel-first images `(inC,inH,inW)` (no batch axis).

PyTorch comparison: `torch.nn.functional.conv2d` specialized to a single image.
-/
def conv2d {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (t : Tape α) (kernelId biasId inputId : Nat) : Result (Tape α × Nat) := by
  let _ := h1
  let _ := h2
  let _ := h3
  exact
    conv (α := α)
      (d := 2)
      (inC := inC)
      (outC := outC)
      (kernel := ⟨#[kH, kW], by simp⟩)
      (stride := ⟨#[stride, stride], by simp⟩)
      (padding := ⟨#[padding, padding], by simp⟩)
      (inSpatial := ⟨#[inH, inW], by simp⟩)
      t kernelId biasId inputId
      (name := "conv2d")

/--
N-D transpose convolution for channels-first tensors `(inC, spatial...)` (no batch axis).

This is the generic counterpart to `conv_transpose2d`.

Kernel layout matches the spec/PyTorch convention `(inC, outC, kernel[0], ..., kernel[d-1])`.

PyTorch comparison: `torch.nn.functional.conv_transpose{d}d` specialized to a single sample
(no batch axis).
-/
def convTranspose {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [DecidableEq Shape]
  {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  (t : Tape α) (kernelId biasId inputId : Nat) (name : String := "conv_transpose") :
  Result (Tape α × Nat) := do
  let w ← requireValue (α := α) (t := t)
    (s := Shape.ofList (inC :: outC :: kernel.toList)) kernelId
  let b ← requireValue (α := α) (t := t) (s := .dim outC .scalar) biasId
  let x ← requireValue (α := α) (t := t)
    (s := Shape.ofList (inC :: inSpatial.toList)) inputId

  let layer : Spec.ConvTransposeSpec d inC outC kernel stride padding α :=
    { kernel := w, bias := b }
  let y := Spec.convTransposeSpec (layer := layer) x
  let outSpatial := Spec.convTransposeOutSpatial inSpatial kernel stride padding
  let outSh : Shape := Shape.ofList (outC :: outSpatial.toList)

  let node : Node α :=
    { name := some name
      value := AnyTensor.mk y
      requires_grad := true
      parents := [kernelId, biasId, inputId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := outSh) dLdyAny
        let (dW, dB, dX) := Spec.convTransposeBackwardSpec (layer := layer) x dLdy
        pure [
          (kernelId, AnyTensor.mk dW),
          (biasId, AnyTensor.mk dB),
          (inputId, AnyTensor.mk dX)
        ]
    }
  pure (t.addNode node)

/--
2D transpose convolution for channel-first images `(inC,inH,inW)` (no batch axis).

This is implemented as a specialization of `conv_transpose` with `d = 2`, scalar stride, and
scalar padding.
Kernel layout matches the spec/PyTorch convention `(inC,outC,kH,kW)`.

PyTorch comparison: `torch.nn.functional.conv_transpose2d` specialized to a single image.
-/
def convTranspose2d {α : Type} [Context α] [DecidableEq Shape]
  {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (t : Tape α) (kernelId biasId inputId : Nat) : Result (Tape α × Nat) := by
  let _ := h1
  let _ := h2
  let _ := h3
  exact
    convTranspose (α := α)
      (d := 2)
      (inC := inC)
      (outC := outC)
      (kernel := ⟨#[kH, kW], by simp⟩)
      (stride := ⟨#[stride, stride], by simp⟩)
      (padding := ⟨#[padding, padding], by simp⟩)
      (inSpatial := ⟨#[inH, inW], by simp⟩)
      t kernelId biasId inputId
      (name := "conv_transpose2d")

/--
N-D max pooling for channels-first tensors `(C, spatial...)` (no batch axis).

Padding is symmetric per-axis and uses zeros. To model unpadded pooling, pass `padding := 0` on
every axis.
-/
def maxPool {α : Type} [Context α] [DecidableEq Shape]
  {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
  {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t)
    (s:=Shape.ofList (C :: inSpatial.toList)) xId
  if hStride : (∀ i : Fin d, stride.get i ≠ 0) then
    let layer : Spec.MaxPoolSpec d kernel stride padding hKernel hStride := {}
    let y := Spec.maxPoolSpec (layer := layer) x
    let outSpatial := Spec.poolOutSpatialPad inSpatial kernel stride padding
    let outSh : Shape := Shape.ofList (C :: outSpatial.toList)
    let node : Node α :=
      { name := some "max_pool"
        value := AnyTensor.mk y
        requires_grad := true
        parents := [xId]
        backward := fun dLdyAny => do
          let dLdy ← requireGrad (α := α) (τ := outSh) dLdyAny
          let dx := Spec.maxPoolBackwardSpec (layer := layer) (input := x) (grad_output := dLdy)
          pure [(xId, AnyTensor.mk dx)]
      }
    pure (t.addNode node)
  else
    throw "autograd: max_pool requires stride > 0 on every spatial axis"

/--
N-D average pooling for channels-first tensors `(C, spatial...)` (no batch axis).

Padding is symmetric per-axis and uses zeros; pooling uses `count_include_pad=true` semantics.
-/
def avgPool {α : Type} [Context α] [DecidableEq Shape]
  {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
  (hKernel : ∀ i : Fin d, kernel.get i ≠ 0)
  (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t)
    (s:=Shape.ofList (C :: inSpatial.toList)) xId
  if hStride : (∀ i : Fin d, stride.get i ≠ 0) then
    let layer : Spec.AvgPoolSpec d kernel stride padding hKernel hStride := {}
    let y := Spec.avgPoolSpec (layer := layer) x
    let outSpatial := Spec.poolOutSpatialPad inSpatial kernel stride padding
    let outSh : Shape := Shape.ofList (C :: outSpatial.toList)
    let node : Node α :=
      { name := some "avg_pool"
        value := AnyTensor.mk y
        requires_grad := true
        parents := [xId]
        backward := fun dLdyAny => do
          let dLdy ← requireGrad (α := α) (τ := outSh) dLdyAny
          let dx := Spec.avgPoolBackwardSpec (layer := layer) (grad_output := dLdy)
          pure [(xId, AnyTensor.mk dx)]
      }
    pure (t.addNode node)
  else
    throw "autograd: avg_pool requires stride > 0 on every spatial axis"

/--
N-D smooth max pooling (log-sum-exp surrogate) for channels-first tensors `(C, spatial...)`.

The executable tape requires a finite, nonzero `beta`. Finiteness is checked through the scalar
arithmetic contract: finite scalar backends satisfy `beta - beta == 0`, whereas IEEE NaN and
infinity do not. At least one spatial dimension is required, matching the native runtime contract.
-/
def smoothMaxPool {α : Type} [Context α] [DecidableEq Shape]
  {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
  {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (t : Tape α) (xId : Nat) (beta : α) : Result (Tape α × Nat) := do
  if beta == 0 || !(beta - beta == 0) then
    throw "autograd: smooth_max_pool requires finite nonzero beta"
  if d = 0 then
    throw "autograd: smooth_max_pool requires at least one spatial dimension"
  let x ← requireValue (α:=α) (t:=t)
    (s:=Shape.ofList (C :: inSpatial.toList)) xId
  if hStride : (∀ i : Fin d, stride.get i ≠ 0) then
    let layer : Spec.MaxPoolSpec d kernel stride padding hKernel hStride := {}
    let y := Spec.smoothMaxPoolSpec (layer := layer) (beta := beta) x
    let outSpatial := Spec.poolOutSpatialPad inSpatial kernel stride padding
    let outSh : Shape := Shape.ofList (C :: outSpatial.toList)
    let node : Node α :=
      { name := some "smooth_max_pool"
        value := AnyTensor.mk y
        requires_grad := true
        parents := [xId]
        backward := fun dLdyAny => do
          let dLdy ← requireGrad (α := α) (τ := outSh) dLdyAny
          let dx :=
            Spec.smoothMaxPoolBackwardSpec (layer := layer) (beta := beta)
              (input := x) (grad_output := dLdy)
          pure [(xId, AnyTensor.mk dx)]
      }
    pure (t.addNode node)
  else
    throw "autograd: smooth_max_pool requires stride > 0 on every spatial axis"

/--
2D max-pooling for channel-first images (no batch axis).

PyTorch comparison: `torch.nn.functional.max_pool2d`.
-/
def maxPool2d {α : Type} [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t)
    (s:=.dim inC (.dim inH (.dim inW .scalar))) xId
  if hStride : stride ≠ 0 then
    let layer : Spec.MaxPool2DSpec kH kW stride h1 h2 hStride := {}
    let y := Spec.maxPool2dMultiSpec (layer := layer) x
    let node : Node α :=
      { name := some "max_pool2d"
        value := AnyTensor.mk y
        requires_grad := true
        parents := [xId]
        backward := fun dLdyAny => do
          let outH := Spec.poolOutDim inH kH stride 0
          let outW := Spec.poolOutDim inW kW stride 0
          let dLdy ←
            requireGrad (α := α) (τ := .dim inC (.dim outH (.dim outW .scalar))) dLdyAny
          let dx :=
            Tensor.dim (fun c =>
              Spec.maxPool2dBackwardSpec (_layer := layer)
                (input := getAtSpec x c) (grad_output := getAtSpec dLdy c))
          pure [(xId, AnyTensor.mk dx)]
      }
    pure (t.addNode node)
  else
    throw "autograd: max_pool2d requires stride > 0"

/--
2D max-pooling with padding for channel-first images (no batch axis).

PyTorch comparison: `max_pool2d(..., padding=...)`.
-/
def maxPool2dPad {α : Type} [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride padding : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t)
    (s:=.dim inC (.dim inH (.dim inW .scalar))) xId
  if hStride : stride ≠ 0 then
    let layer : Spec.MaxPool2DSpec kH kW stride h1 h2 hStride := {}
    let y := Spec.maxPool2dMultiSpecPad (layer := layer) (padding := padding) x
    let node : Node α :=
      { name := some "max_pool2d_pad"
        value := AnyTensor.mk y
        requires_grad := true
        parents := [xId]
        backward := fun dLdyAny => do
          let outH := Spec.poolOutDim inH kH stride padding
          let outW := Spec.poolOutDim inW kW stride padding
          let dLdy ←
            requireGrad (α := α) (τ := .dim inC (.dim outH (.dim outW .scalar))) dLdyAny
          let dx :=
            Spec.maxPool2dMultiBackwardSpecPad (layer := layer) (padding := padding) x dLdy
          pure [(xId, AnyTensor.mk dx)]
      }
    pure (t.addNode node)
  else
    throw "autograd: max_pool2d_pad requires stride > 0"

/--
Smooth approximation of max-pooling (softmax pooling).

This is not a standard PyTorch primitive; it is useful for differentiable relaxations.
The executable tape requires a finite, nonzero `beta`.
-/
def smoothMaxPool2d {α : Type} [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (t : Tape α) (xId : Nat) (beta : α) : Result (Tape α × Nat) := do
  if beta == 0 || !(beta - beta == 0) then
    throw "autograd: smooth_max_pool2d requires finite nonzero beta"
  let x ← requireValue (α:=α) (t:=t)
    (s:=.dim inC (.dim inH (.dim inW .scalar))) xId
  if hStride : stride ≠ 0 then
    let layer : Spec.MaxPool2DSpec kH kW stride h1 h2 hStride := {}
    let y := Spec.smoothMaxPool2dMultiSpec (layer := layer) (beta := beta) x
    let node : Node α :=
      { name := some "smooth_max_pool2d"
        value := AnyTensor.mk y
        requires_grad := true
        parents := [xId]
        backward := fun dLdyAny => do
          let outH := Spec.poolOutDim inH kH stride 0
          let outW := Spec.poolOutDim inW kW stride 0
          let dLdy ←
            requireGrad (α := α) (τ := .dim inC (.dim outH (.dim outW .scalar))) dLdyAny
          let dx :=
            Spec.smoothMaxPool2dMultiBackwardSpec (layer := layer) (beta := beta) x dLdy
          pure [(xId, AnyTensor.mk dx)]
      }
    pure (t.addNode node)
  else
    throw "autograd: smooth_max_pool2d requires stride > 0"

/--
2D average-pooling for channel-first images (no batch axis).

PyTorch comparison: `torch.nn.functional.avg_pool2d`.
-/
def avgPool2d {α : Type} [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t)
    (s:=.dim inC (.dim inH (.dim inW .scalar))) xId
  if hStride : stride ≠ 0 then
    let layer : Spec.AvgPool2DSpec kH kW stride h1 h2 hStride := {}
    let y := Spec.avgPool2dMultiSpec (h1 := h1) (h2 := h2) (layer := layer) x
    let node : Node α :=
      { name := some "avg_pool2d"
        value := AnyTensor.mk y
        requires_grad := true
        parents := [xId]
        backward := fun dLdyAny => do
          let outH := Spec.poolOutDim inH kH stride 0
          let outW := Spec.poolOutDim inW kW stride 0
          let dLdy ←
            requireGrad (α := α) (τ := .dim inC (.dim outH (.dim outW .scalar))) dLdyAny
          let dx :=
            Tensor.dim (fun c =>
              Spec.avgPool2dBackwardSpec (α := α) h1 h2 layer (getAtSpec dLdy c))
          pure [(xId, AnyTensor.mk dx)]
      }
    pure (t.addNode node)
  else
    throw "autograd: avg_pool2d requires stride > 0"

/--
2D average-pooling with padding for channel-first images (no batch axis).

PyTorch comparison: `avg_pool2d(..., padding=...)`.
-/
def avgPool2dPad {α : Type} [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride padding : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t)
    (s:=.dim inC (.dim inH (.dim inW .scalar))) xId
  if hStride : stride ≠ 0 then
    let layer : Spec.AvgPool2DSpec kH kW stride h1 h2 hStride := {}
    let y := Spec.avgPool2dMultiSpecPad (h1 := h1) (h2 := h2) (layer := layer) (padding :=
      padding) x
    let node : Node α :=
      { name := some "avg_pool2d_pad"
        value := AnyTensor.mk y
        requires_grad := true
        parents := [xId]
        backward := fun dLdyAny => do
          let outH := Spec.poolOutDim inH kH stride padding
          let outW := Spec.poolOutDim inW kW stride padding
          let dLdy ←
            requireGrad (α := α) (τ := .dim inC (.dim outH (.dim outW .scalar))) dLdyAny
          let dx :=
            Spec.avgPool2dMultiBackwardSpecPad (h1 := h1) (h2 := h2) (layer := layer)
              (padding := padding) dLdy
          pure [(xId, AnyTensor.mk dx)]
      }
    pure (t.addNode node)
  else
    throw "autograd: avg_pool2d_pad requires stride > 0"
