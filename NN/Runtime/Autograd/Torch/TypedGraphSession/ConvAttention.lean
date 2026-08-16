/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.TypedGraphSession.Neural

/-!
# Typed Graph Session: Convolution and Attention Operations
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec
open Tensor

namespace Internal

namespace TypedGraphSession

/--
N-D convolution for channels-first tensors `(inC, spatial...)` (no batch axis).

Kernel layout is `(outC, inC, kernelSpatial...)`, bias is `(outC)`.

PyTorch comparison: `torch.nn.functional.conv{d}d` specialized to a single sample.
-/
def conv {α : Type} (s : TypedGraphSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (w : TensorRef α (Shape.ofList (outC :: inC :: kernel.toList)))
  (b : TensorRef α (.dim outC .scalar))
  (x : TensorRef α (Shape.ofList (inC :: inSpatial.toList))) :
  IO (TensorRef α
    (Shape.ofList (outC :: (Spec.convOutSpatial inSpatial kernel stride padding).toList))) :=
  commitGraphM (α := α) s
    (β := TensorRef α
      (Shape.ofList (outC :: (Spec.convOutSpatial inSpatial kernel stride padding).toList)))
    (fun {Γ} {ss} xv nat g => do
      let (v, st') ← runGraphM (α := α) (Γ := Γ)
        (Runtime.Autograd.TypedGraph.GraphM.conv (α := α) (Γ := Γ)
          (d := d) (inC := inC) (outC := outC)
          (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
          (hInC := hInC) (hKernel := hKernel)
          { id := w.id } { id := b.id } { id := x.id })
        ss g
      let ⟨ss', g'⟩ := st'
      let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
      pure ({ id := v.id }, st1))

/--
N-D transpose convolution for channels-first tensors `(inC, spatial...)` (no batch axis).

Kernel layout is `(inC, outC, kernelSpatial...)` (PyTorch convention), bias is `(outC)`.

PyTorch comparison: `torch.nn.functional.conv_transpose{d}d` specialized to a single sample.
-/
def convTranspose {α : Type} (s : TypedGraphSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (w : TensorRef α (Shape.ofList (inC :: outC :: kernel.toList)))
  (b : TensorRef α (.dim outC .scalar))
  (x : TensorRef α (Shape.ofList (inC :: inSpatial.toList))) :
  IO (TensorRef α
    (Shape.ofList (outC ::
      (Spec.convTransposeOutSpatial inSpatial kernel stride padding).toList))) :=
  commitGraphM (α := α) s
    (β := TensorRef α
      (Shape.ofList (outC :: (Spec.convTransposeOutSpatial inSpatial kernel stride padding).toList)))
    (fun {Γ} {ss} xv nat g => do
      let (v, st') ← runGraphM (α := α) (Γ := Γ)
        (Runtime.Autograd.TypedGraph.GraphM.convTranspose (α := α) (Γ := Γ)
          (d := d) (inC := inC) (outC := outC)
          (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
          (hInC := hInC) (hKernel := hKernel)
          { id := w.id } { id := b.id } { id := x.id })
        ss g
      let ⟨ss', g'⟩ := st'
      let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
      pure ({ id := v.id }, st1))

/--
2D convolution for channel-first images `(inC, inH, inW)` (no batch axis).

Type-level shapes fix the kernel layout `(outC, inC, kH, kW)` and output spatial dimensions derived
from `stride` and `padding`.
PyTorch comparison: `torch.nn.functional.conv2d` (conceptually), specialized to a single image.
-/
def conv2d {α : Type} (s : TypedGraphSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (kernel : TensorRef α (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))))
  (bias : TensorRef α (.dim outC .scalar))
  (input : TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (TensorRef α (.dim outC (.dim (Spec.Shape.slidingWindowOutDim inH kH stride padding) (.dim (Spec.Shape.slidingWindowOutDim inW kW stride padding) .scalar)))) :=
  commitGraphM (α := α) s
    (β := TensorRef α (.dim outC (.dim (Spec.Shape.slidingWindowOutDim inH kH stride padding) (.dim (Spec.Shape.slidingWindowOutDim inW kW stride padding) .scalar))))
    (fun {Γ} {ss} xv nat g => do
      let (v, st') ← runGraphM (α := α) (Γ := Γ)
        (Runtime.Autograd.TypedGraph.GraphM.conv2d (α := α) (Γ := Γ)
          (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
          (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 := h3)
          { id := kernel.id } { id := bias.id } { id := input.id })
        ss g
      let ⟨ss', g'⟩ := st'
      let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
      pure ({ id := v.id }, st1))

/--
2D transpose convolution for channel-first images `(inC, inH, inW)` (no batch axis).

Kernel layout matches the spec/PyTorch convention `(inC, outC, kH, kW)`.
PyTorch comparison: `torch.nn.functional.conv_transpose2d` specialized to a single image.
-/
def convTranspose2d {α : Type} (s : TypedGraphSession α) [Context α]
  [DecidableEq Shape]
  {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (kernel : TensorRef α (.dim inC (.dim outC (.dim kH (.dim kW .scalar)))))
  (bias : TensorRef α (.dim outC .scalar))
  (input : TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (TensorRef α (.dim outC (.dim (Spec.convTransposeOutDim inH kH stride padding)
    (.dim (Spec.convTransposeOutDim inW kW stride padding) .scalar)))) :=
  commitGraphM (α := α) s
    (β := TensorRef α (.dim outC (.dim (Spec.convTransposeOutDim inH kH stride padding)
      (.dim (Spec.convTransposeOutDim inW kW stride padding) .scalar))))
    (fun {Γ} {ss} xv nat g => do
      let (v, st') ← runGraphM (α := α) (Γ := Γ)
        (Runtime.Autograd.TypedGraph.GraphM.convTranspose2d (α := α) (Γ := Γ)
          (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
          (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 := h3)
          { id := kernel.id } { id := bias.id } { id := input.id })
        ss g
      let ⟨ss', g'⟩ := st'
      let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
      pure ({ id := v.id }, st1))

/--
Multi-head self-attention.

This is a shape-specialized attention primitive used by transformer-style examples:
- input `x` has shape `(n, dModel)`
- `wq`, `wk`, `wv` map `dModel → numHeads*headDim`
- `wo` maps `numHeads*headDim → dModel`
- optional `mask` is a boolean `(n,n)` attention mask

PyTorch comparison: similar to `torch.nn.MultiheadAttention` / scaled dot-product attention, but
encoded in a fully typed graph for lowering and later semantic analysis.
-/
def multiHeadAttention {α : Type} (s : TypedGraphSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (wq : TensorRef α (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wk : TensorRef α (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wv : TensorRef α (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wo : TensorRef α (.dim (numHeads * headDim) (.dim dModel .scalar)))
  (x : TensorRef α (.dim n (.dim dModel .scalar)))
  (mask : Option (Tensor Bool (.dim n (.dim n .scalar))) := none) :
  IO (TensorRef α (.dim n (.dim dModel .scalar))) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim n (.dim dModel .scalar))) (fun {Γ} {ss} xv nat g
    => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.multiHeadAttention (α := α) (Γ := Γ)
        (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim) (h1 := h1)
        { id := wq.id } { id := wk.id } { id := wv.id } { id := wo.id } { id := x.id } (mask :=
          mask))
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))
end TypedGraphSession

end Internal

end Torch
end Autograd
end Runtime
