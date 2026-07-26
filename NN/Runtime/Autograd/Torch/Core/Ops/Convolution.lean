/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Ops.Dispatch

/-!
# Eager Tensor Operations

PyTorch-style tensor operations backed by the eager CPU/CUDA tapes. These wrappers record runtime
nodes, dispatch CUDA kernels when requested, and preserve the typed `TensorRef` surface.
-/


@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec
open Tensor
open Proofs.Autograd.Algebra

namespace Internal

namespace EagerSession

/-! ## Convolution operations -/

/--
N-D convolution for channels-first tensors `(inC, spatial...)` (no batch axis).

This is the generic counterpart to `conv2d`.
PyTorch comparison: `torch.nn.functional.conv{d}d` specialized to a single sample.
-/
def conv {α : Type} (s : EagerSession α) [Context α]
  [DecidableEq Shape]
  {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (w : TensorRef α (Shape.ofList (outC :: inC :: kernel.toList)))
  (b : TensorRef α (.dim outC .scalar))
  (x : TensorRef α (Shape.ofList (inC :: inSpatial.toList))) :
  IO (TensorRef α
    (Shape.ofList (outC :: (Spec.convOutSpatial inSpatial kernel stride padding).toList))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.conv (t := t0)
      (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      w.id b.id x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.conv (t := t0)
      (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      w.id b.id x.id
      (hInC := hInC) (hKernel := hKernel))
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .conv cpu cuda

/-- 2D convolution for channel-first images `(C,H,W)` (no batch axis). PyTorch:
  `torch.nn.functional.conv2d`. -/
def conv2d {α : Type} (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (kernel : TensorRef α (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))))
  (bias : TensorRef α (.dim outC .scalar))
  (input : TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (TensorRef α (.dim outC (.dim (Spec.Shape.slidingWindowOutDim inH kH stride padding)
    (.dim (Spec.Shape.slidingWindowOutDim inW kW stride padding) .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.conv2d (t := t0)
      (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
      (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 := h3)
      kernel.id bias.id input.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.conv2d (t := t0)
        (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
        (inH := inH) (inW := inW)
        (h1 := h1) (h2 := h2) (h3 := h3)
        kernel.id bias.id input.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .conv cpu cuda

/--
N-D transpose convolution for channels-first tensors `(inC, spatial...)` (no batch axis).

This is the generic counterpart to `conv_transpose2d`.
PyTorch comparison: `torch.nn.functional.conv_transpose{d}d` specialized to a single sample.
-/
def convTranspose {α : Type} (s : EagerSession α) [Context α]
  [DecidableEq Shape]
  {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (w : TensorRef α (Shape.ofList (inC :: outC :: kernel.toList)))
  (b : TensorRef α (.dim outC .scalar))
  (x : TensorRef α (Shape.ofList (inC :: inSpatial.toList))) :
  IO (TensorRef α
    (Shape.ofList (outC :: (Spec.convTransposeOutSpatial inSpatial kernel stride padding).toList)))
    := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.convTranspose (t := t0)
      (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      w.id b.id x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.convTranspose (t := t0)
      (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      w.id b.id x.id
      (hInC := hInC) (hKernel := hKernel))
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .convTranspose cpu cuda

/-- 2D transpose convolution for channel-first images `(C,H,W)` (no batch axis). PyTorch:
  `torch.nn.functional.conv_transpose2d`. -/
def convTranspose2d {α : Type} (s : EagerSession α) [Context α]
  [DecidableEq Shape]
  {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (kernel : TensorRef α (.dim inC (.dim outC (.dim kH (.dim kW .scalar)))))
  (bias : TensorRef α (.dim outC .scalar))
  (input : TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (TensorRef α (.dim outC (.dim (Spec.convTransposeOutDim inH kH stride padding)
    (.dim (Spec.convTransposeOutDim inW kW stride padding) .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.convTranspose2d (t := t0)
      (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
      (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 := h3)
      kernel.id bias.id input.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.convTranspose2d (t := t0)
        (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
        (inH := inH) (inW := inW)
        (h1 := h1) (h2 := h2) (h3 := h3)
        kernel.id bias.id input.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .convTranspose cpu cuda

end EagerSession

end Internal
end Torch
end Autograd
end Runtime
