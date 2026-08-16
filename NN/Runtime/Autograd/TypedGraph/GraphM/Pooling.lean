/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TypedGraph.GraphM.Elementwise

/-!
# GraphM Pooling Ops

N-dimensional and two-dimensional pooling builders with forward, JVP, and VJP payloads.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace TypedGraph
namespace GraphM

open Spec
open Tensor
open Proofs.Autograd.Algebra
open Runtime.Autograd.TorchLean

/--
N-D max pooling (channels-first) on a single sample tensor (no batch axis).

PyTorch comparison: `torch.nn.functional.max_pool1d` / `max_pool2d` / `max_pool3d` depending on
the spatial rank `d`.

Forward-mode status: implemented as the selected-branch linearization from
`Spec.maxPoolLinearizationSpec`. At ties this follows the documented first-winner convention.
-/
def maxPool {α : Type} {Δ : Type} [Context α] [DecidableEq Shape]
  {Γ : List Shape} {d C : Nat}
  {inSpatial kernel stride padding : Vector Nat d}
  {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (x : Var (Shape.ofList (C :: inSpatial.toList))) :
  MWith α Δ Γ (Var (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  if hStride : (∀ i : Fin d, stride.get i ≠ 0) then
    let layer : Spec.MaxPoolSpec d kernel stride padding hKernel hStride := {}
    let outSpatial := Spec.poolOutSpatialPad inSpatial kernel stride padding
    let outShape : Shape := Shape.ofList (C :: outSpatial.toList)
    let inShape : Shape := Shape.ofList (C :: inSpatial.toList)
    let node : NodeData α Δ (Γ ++ ss) outShape :=
      { forward := fun ctx _d =>
          let xv := getIdx (α := α) (xs := ctx) ix
          Spec.maxPoolSpec (α := α) (d := d) (C := C)
            (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
            (layer := layer) xv
        jvp := fun ctx dctx _d =>
          let xv := getIdx (α := α) (xs := ctx) ix
          let dx := getIdx (α := α) (xs := dctx) ix
          Spec.maxPoolLinearizationSpec (α := α) (d := d) (C := C)
            (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
            (layer := layer) xv dx
        vjp := fun ctx _d δ =>
          let xv := getIdx (α := α) (xs := ctx) ix
          let dx :=
            Spec.maxPoolBackwardSpec (α := α) (d := d) (C := C)
              (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
              (layer := layer) (input := xv) (grad_output := δ)
          TList.single (α := α) (Γ := Γ ++ ss) (s := inShape) ix dx }
    push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outShape) g node
  else
    throw "typed GraphM: max_pool requires stride > 0 on every spatial axis"

/--
N-D average pooling (channels-first) on a single sample tensor (no batch axis).

PyTorch comparison: `torch.nn.functional.avg_pool1d` / `avg_pool2d` / `avg_pool3d` depending on
the spatial rank `d`.

  Forward-mode status: implemented. Average pooling is linear, so the JVP is the same average-pool
  map applied to the input tangent.
-/
def avgPool {α : Type} {Δ : Type} [Context α] [DecidableEq Shape]
  {Γ : List Shape} {d C : Nat}
  {inSpatial kernel stride padding : Vector Nat d}
  (hKernel : ∀ i : Fin d, kernel.get i ≠ 0)
  (x : Var (Shape.ofList (C :: inSpatial.toList))) :
  MWith α Δ Γ (Var (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  if hStride : (∀ i : Fin d, stride.get i ≠ 0) then
    let layer : Spec.AvgPoolSpec d kernel stride padding hKernel hStride := {}
    let outSpatial := Spec.poolOutSpatialPad inSpatial kernel stride padding
    let outShape : Shape := Shape.ofList (C :: outSpatial.toList)
    let inShape : Shape := Shape.ofList (C :: inSpatial.toList)
    let node : NodeData α Δ (Γ ++ ss) outShape :=
      { forward := fun ctx _d =>
          let xv := getIdx (α := α) (xs := ctx) ix
          Spec.avgPoolSpec (α := α) (d := d) (C := C)
            (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
            (layer := layer) xv
        jvp := fun _ctx dctx _d =>
          let dx := getIdx (α := α) (xs := dctx) ix
          Spec.avgPoolSpec (α := α) (d := d) (C := C)
            (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
            (layer := layer) dx
        vjp := fun _ctx _d δ =>
          let dx :=
            Spec.avgPoolBackwardSpec (α := α) (d := d) (C := C)
              (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
              (layer := layer) (grad_output := δ)
          TList.single (α := α) (Γ := Γ ++ ss) (s := inShape) ix dx }
    push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outShape) g node
  else
    throw "typed GraphM: avg_pool requires stride > 0 on every spatial axis"

/--
N-D smooth max pooling (log-sum-exp surrogate) on a single sample tensor (no batch axis).

PyTorch comparison: there is no direct primitive; this is a differentiable approximation to
max pooling.

Forward-mode status: implemented. The JVP is the softmax-weighted tangent of the log-sum-exp
pooling window. Executable graphs require a finite, nonzero `beta` and at least one spatial
dimension.
-/
def smoothMaxPool {α : Type} {Δ : Type} [Context α] [DecidableEq Shape]
  {Γ : List Shape} {d C : Nat}
  {inSpatial kernel stride padding : Vector Nat d}
  {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (x : Var (Shape.ofList (C :: inSpatial.toList))) (beta : α) :
  MWith α Δ Γ (Var (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) := do
  -- `Context` has no generic `isFinite`; supported executable scalars self-subtract to zero exactly
  -- when finite, so this probe rejects NaN and infinities without adding a stronger scalar class.
  if beta == 0 || !(beta - beta == 0) then
    throw "typed GraphM: smooth_max_pool requires finite nonzero beta"
  if d = 0 then
    throw "typed GraphM: smooth_max_pool requires at least one spatial dimension"
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  if hStride : (∀ i : Fin d, stride.get i ≠ 0) then
    let layer : Spec.MaxPoolSpec d kernel stride padding hKernel hStride := {}
    let outSpatial := Spec.poolOutSpatialPad inSpatial kernel stride padding
    let outShape : Shape := Shape.ofList (C :: outSpatial.toList)
    let inShape : Shape := Shape.ofList (C :: inSpatial.toList)
    let node : NodeData α Δ (Γ ++ ss) outShape :=
      { forward := fun ctx _d =>
          let xv := getIdx (α := α) (xs := ctx) ix
          Spec.smoothMaxPoolSpec (α := α) (d := d) (C := C)
            (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
            (layer := layer) (beta := beta) xv
        jvp := fun ctx dctx _d =>
          let xv := getIdx (α := α) (xs := ctx) ix
          let dx := getIdx (α := α) (xs := dctx) ix
          Spec.smoothMaxPoolJvpSpec (α := α) (d := d) (C := C)
            (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
            (layer := layer) (beta := beta) xv dx
        vjp := fun ctx _d δ =>
          let xv := getIdx (α := α) (xs := ctx) ix
          let dx :=
            Spec.smoothMaxPoolBackwardSpec (α := α) (d := d) (C := C)
              (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
              (layer := layer) (beta := beta) (input := xv) (grad_output := δ)
          TList.single (α := α) (Γ := Γ ++ ss) (s := inShape) ix dx }
    push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outShape) g node
  else
    throw "typed GraphM: smooth_max_pool requires stride > 0 on every spatial axis"

/--
2D max-pooling (channel-first) on a single image tensor.

PyTorch comparison: `torch.nn.functional.max_pool2d` (without a batch dimension).

Forward-mode status: implemented. The JVP routes each output tangent through the
argmax selected by the primal input.
-/

def maxPool2d {α : Type} {Δ : Type} [Context α] [DecidableEq Shape]
  {Γ : List Shape} {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : Var (.dim inC (.dim inH (.dim inW .scalar)))) :
  MWith α Δ Γ (Var (.dim inC (.dim (Spec.poolOutDim inH kH stride 0)
    (.dim (Spec.poolOutDim inW kW stride 0) .scalar)))) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  if hStride : stride ≠ 0 then
    let layer : Spec.MaxPool2DSpec kH kW stride h1 h2 hStride := {}
    let outH := Spec.poolOutDim inH kH stride 0
    let outW := Spec.poolOutDim inW kW stride 0
    let outShape : Shape := .dim inC (.dim outH (.dim outW .scalar))
    let inShape : Shape := .dim inC (.dim inH (.dim inW .scalar))
    let node : NodeData α Δ (Γ ++ ss) outShape :=
      { forward := fun ctx _d =>
          let xv := getIdx (α := α) (xs := ctx) ix
          Spec.maxPool2dMultiSpec (layer := layer) xv
        jvp := fun ctx dctx _d =>
          let xv := getIdx (α := α) (xs := ctx) ix
          let dx := getIdx (α := α) (xs := dctx) ix
          Spec.maxPool2dMultiLinearizationSpec (layer := layer) (input := xv) (tangent := dx)
        vjp := fun ctx _d δ =>
          let xv := getIdx (α := α) (xs := ctx) ix
          let dx :=
            Tensor.dim (fun c =>
              Spec.maxPool2dBackwardSpec (α := α) (_layer := layer)
                (input := getAtSpec xv c) (grad_output := getAtSpec δ c))
          TList.single (α := α) (Γ := Γ ++ ss) (s := inShape) ix dx }
    push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outShape) g node
  else
    throw "typed GraphM: max_pool2d requires stride > 0"

/--
2D max-pooling with explicit padding.

PyTorch comparison: `torch.nn.functional.max_pool2d` with padding.

Forward-mode status: implemented as a selected-branch linearization. Padding is fixed and the
chosen tangent follows the real primal winner, ignoring padded cells just like the forward pass.
-/
def maxPool2dPad {α : Type} {Δ : Type} [Context α] [DecidableEq Shape]
  {Γ : List Shape} {kH kW inH inW inC stride padding : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : Var (.dim inC (.dim inH (.dim inW .scalar)))) :
  MWith α Δ Γ
    (Var (.dim inC (.dim (Spec.poolOutDim inH kH stride padding)
      (.dim (Spec.poolOutDim inW kW stride padding) .scalar)))) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  if hStride : stride ≠ 0 then
    let layer : Spec.MaxPool2DSpec kH kW stride h1 h2 hStride := {}
    let outH := Spec.poolOutDim inH kH stride padding
    let outW := Spec.poolOutDim inW kW stride padding
    let outShape : Shape := .dim inC (.dim outH (.dim outW .scalar))
    let inShape : Shape := .dim inC (.dim inH (.dim inW .scalar))
    let node : NodeData α Δ (Γ ++ ss) outShape :=
      { forward := fun ctx _d =>
          let xv := getIdx (α := α) (xs := ctx) ix
          Spec.maxPool2dMultiSpecPad (layer := layer) (padding := padding) xv
        jvp := fun ctx dctx _d =>
          let xv := getIdx (α := α) (xs := ctx) ix
          let dx := getIdx (α := α) (xs := dctx) ix
          Spec.maxPool2dMultiLinearizationSpecPad (layer := layer) (padding := padding)
            (input := xv) (tangent := dx)
        vjp := fun ctx _d δ =>
          let xv := getIdx (α := α) (xs := ctx) ix
          let dx :=
            Spec.maxPool2dMultiBackwardSpecPad (layer := layer) (padding := padding) xv δ
          TList.single (α := α) (Γ := Γ ++ ss) (s := inShape) ix dx }
    push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outShape) g node
  else
    throw "typed GraphM: max_pool2d_pad requires stride > 0"

/--
Smooth (soft) max-pooling, controlled by `beta`.

This is a differentiable approximation to max-pooling.

Forward-mode status: implemented. The JVP is the softmax-weighted tangent of the
log-sum-exp pooling window. Executable graphs require a finite, nonzero `beta`.
-/
def smoothMaxPool2d {α : Type} {Δ : Type} [Context α] [DecidableEq Shape]
  {Γ : List Shape} {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : Var (.dim inC (.dim inH (.dim inW .scalar)))) (beta : α) :
  MWith α Δ Γ (Var (.dim inC (.dim (Spec.poolOutDim inH kH stride 0)
    (.dim (Spec.poolOutDim inW kW stride 0) .scalar)))) := do
  -- `Context` has no generic `isFinite`; supported executable scalars self-subtract to zero exactly
  -- when finite, so this probe rejects NaN and infinities without adding a stronger scalar class.
  if beta == 0 || !(beta - beta == 0) then
    throw "typed GraphM: smooth_max_pool2d requires finite nonzero beta"
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  if hStride : stride ≠ 0 then
    let layer : Spec.MaxPool2DSpec kH kW stride h1 h2 hStride := {}
    let outH := Spec.poolOutDim inH kH stride 0
    let outW := Spec.poolOutDim inW kW stride 0
    let outShape : Shape := .dim inC (.dim outH (.dim outW .scalar))
    let inShape : Shape := .dim inC (.dim inH (.dim inW .scalar))
    let node : NodeData α Δ (Γ ++ ss) outShape :=
      { forward := fun ctx _d =>
          let xv := getIdx (α := α) (xs := ctx) ix
          Spec.smoothMaxPool2dMultiSpec (layer := layer) (beta := beta) xv
        jvp := fun ctx dctx _d =>
          let xv := getIdx (α := α) (xs := ctx) ix
          let dx := getIdx (α := α) (xs := dctx) ix
          Spec.smoothMaxPool2dMultiJvpSpec (layer := layer) (beta := beta)
            (input := xv) (tangent := dx)
        vjp := fun ctx _d δ =>
          let xv := getIdx (α := α) (xs := ctx) ix
          let dx :=
            Spec.smoothMaxPool2dMultiBackwardSpec (layer := layer) (beta := beta) xv δ
          TList.single (α := α) (Γ := Γ ++ ss) (s := inShape) ix dx }
    push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outShape) g node
  else
    throw "typed GraphM: smooth_max_pool2d requires stride > 0"

/--
Average pooling (channel-first) on a single image tensor.

PyTorch comparison: `torch.nn.functional.avg_pool2d` (without a batch dimension).

Forward-mode status: implemented. Average pooling is linear, so the JVP is average pooling of the
input tangent.
-/
def avgPool2d {α : Type} {Δ : Type} [Context α] [DecidableEq Shape]
  {Γ : List Shape} {kH kW inH inW inC stride : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  (x : Var (.dim inC (.dim inH (.dim inW .scalar)))) :
  MWith α Δ Γ (Var (.dim inC (.dim (Spec.poolOutDim inH kH stride 0)
    (.dim (Spec.poolOutDim inW kW stride 0) .scalar)))) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  if hStride : stride ≠ 0 then
    let layer : Spec.AvgPool2DSpec kH kW stride h1 h2 hStride := {}
    let outH := Spec.poolOutDim inH kH stride 0
    let outW := Spec.poolOutDim inW kW stride 0
    let outShape : Shape := .dim inC (.dim outH (.dim outW .scalar))
    let inShape : Shape := .dim inC (.dim inH (.dim inW .scalar))
    let node : NodeData α Δ (Γ ++ ss) outShape :=
      { forward := fun ctx _d =>
          let xv := getIdx (α := α) (xs := ctx) ix
          Spec.avgPool2dMultiSpec (h1 := h1) (h2 := h2) (layer := layer) xv
        jvp := fun _ctx dctx _d =>
          let dx := getIdx (α := α) (xs := dctx) ix
          Spec.avgPool2dMultiSpec (h1 := h1) (h2 := h2) (layer := layer) dx
        vjp := fun _ctx _d δ =>
          let dx :=
            Tensor.dim (fun c =>
              Spec.avgPool2dBackwardSpec (α := α) (_h1 := h1) (_h2 := h2) (_layer := layer)
                (grad_output := getAtSpec δ c))
          TList.single (α := α) (Γ := Γ ++ ss) (s := inShape) ix dx }
    push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outShape) g node
  else
    throw "typed GraphM: avg_pool2d requires stride > 0"

/--
Average pooling with explicit padding.

PyTorch comparison: `torch.nn.functional.avg_pool2d` with padding.

  Forward-mode status: implemented. Padding is fixed and average pooling is linear, so the JVP is
  the padded average-pool map applied to the input tangent.
-/
def avgPool2dPad {α : Type} {Δ : Type} [Context α] [DecidableEq Shape]
  {Γ : List Shape} {kH kW inH inW inC stride padding : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  (x : Var (.dim inC (.dim inH (.dim inW .scalar)))) :
  MWith α Δ Γ
    (Var (.dim inC (.dim (Spec.poolOutDim inH kH stride padding)
      (.dim (Spec.poolOutDim inW kW stride padding) .scalar)))) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  if hStride : stride ≠ 0 then
    let layer : Spec.AvgPool2DSpec kH kW stride h1 h2 hStride := {}
    let outH := Spec.poolOutDim inH kH stride padding
    let outW := Spec.poolOutDim inW kW stride padding
    let outShape : Shape := .dim inC (.dim outH (.dim outW .scalar))
    let inShape : Shape := .dim inC (.dim inH (.dim inW .scalar))
    let node : NodeData α Δ (Γ ++ ss) outShape :=
      { forward := fun ctx _d =>
          let xv := getIdx (α := α) (xs := ctx) ix
          Spec.avgPool2dMultiSpecPad (h1 := h1) (h2 := h2) (layer := layer) (padding := padding)
            xv
        jvp := fun _ctx dctx _d =>
          let dx := getIdx (α := α) (xs := dctx) ix
          Spec.avgPool2dMultiSpecPad (h1 := h1) (h2 := h2) (layer := layer) (padding := padding)
            dx
        vjp := fun _ctx _d δ =>
          let dx :=
            Spec.avgPool2dMultiBackwardSpecPad (h1 := h1) (h2 := h2) (layer := layer)
              (padding := padding) δ
          TList.single (α := α) (Γ := Γ ++ ss) (s := inShape) ix dx }
    push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outShape) g node
  else
    throw "typed GraphM: avg_pool2d_pad requires stride > 0"

end GraphM
end TypedGraph
end Autograd
end Runtime
