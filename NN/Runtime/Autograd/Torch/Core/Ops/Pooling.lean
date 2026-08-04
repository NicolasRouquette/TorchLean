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

/-! ## Pooling operations -/

/--
N-D max pooling for channels-first tensors `(C, spatial...)` (no batch axis).

PyTorch comparison: `torch.nn.functional.max_pool1d` / `max_pool2d` / `max_pool3d` depending on the
spatial rank `d`.
-/
def maxPool {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
  {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (x : TensorRef α (Shape.ofList (C :: inSpatial.toList))) :
  IO (TensorRef α
    (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.maxPool (t := t0)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.maxPool (t := t0)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel) x.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .maxPool cpu cuda

/--
N-D average pooling for channels-first tensors `(C, spatial...)` (no batch axis).

PyTorch comparison: `torch.nn.functional.avg_pool1d` / `avg_pool2d` / `avg_pool3d` depending on the
spatial rank `d`.
-/
def avgPool {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
  (hKernel : ∀ i : Fin d, kernel.get i ≠ 0)
  (x : TensorRef α (Shape.ofList (C :: inSpatial.toList))) :
  IO (TensorRef α
    (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.avgPool (t := t0)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      hKernel x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.avgPool (t := t0)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      hKernel x.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .avgPool cpu cuda

/--
N-D smooth max pooling (log-sum-exp surrogate) for channels-first tensors `(C, spatial...)`.

This is a differentiable approximation to max pooling; PyTorch does not expose it as a single
primitive, but it can be emulated with `logsumexp` over local windows. Executable backends require
at least one spatial dimension and a finite, nonzero `beta`; evaluation uses an input-space
max/min shift so the exponential weights remain stable for large finite values.
-/
def smoothMaxPool {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) [Context α]
  [DecidableEq Shape]
  {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
  {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (x : TensorRef α (Shape.ofList (C :: inSpatial.toList))) (beta : α) :
  IO (TensorRef α
    (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.smoothMaxPool (t := t0)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel) x.id beta)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let betaF ← CudaBridge.TensorConv.toFloat (α := α) beta
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.smoothMaxPool (t := t0)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel) x.id betaF)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .smoothMaxPool cpu cuda

/-- 2D max-pooling (no batch axis). PyTorch: `torch.nn.functional.max_pool2d`. -/
def maxPool2d {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (TensorRef α (.dim inC (.dim (Spec.poolOutDim inH kH stride 0) (.dim (Spec.poolOutDim inW kW stride 0)
    .scalar)))) := do
  if stride == 0 then
    throw <| IO.userError "torch: max_pool2d requires stride > 0"
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.maxPool2d (t := t0)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := h1) (h2 := h2) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let inCU32 ← okOrThrow <| Runtime.Autograd.Cuda.AnyBuffer.natToU32Checked inC
    let inHU32 ← okOrThrow <| Runtime.Autograd.Cuda.AnyBuffer.natToU32Checked inH
    let inWU32 ← okOrThrow <| Runtime.Autograd.Cuda.AnyBuffer.natToU32Checked inW
    let kHU32 ← okOrThrow <| Runtime.Autograd.Cuda.AnyBuffer.natToU32Checked kH
    let kWU32 ← okOrThrow <| Runtime.Autograd.Cuda.AnyBuffer.natToU32Checked kW
    let strideU32 ← okOrThrow <| Runtime.Autograd.Cuda.AnyBuffer.natToU32Checked stride
    let paddingU32 : UInt32 := 0
    let outSh : Shape :=
      .dim inC (.dim (Spec.poolOutDim inH kH stride 0) (.dim (Spec.poolOutDim inW kW stride 0) .scalar))
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.unary (t := t0) "max_pool2d" x.id
        (.dim inC (.dim inH (.dim inW .scalar))) outSh
        (forward := fun xBuf =>
          Runtime.Autograd.Cuda.torchleanMaxPool2dFwdCuda xBuf inCU32 inHU32 inWU32 kHU32 kWU32 strideU32 paddingU32)
        (backward := fun xBuf dLdy =>
          Runtime.Autograd.Cuda.torchleanMaxPool2dBwdCuda xBuf dLdy inCU32 inHU32 inWU32 kHU32 kWU32 strideU32 paddingU32)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .maxPool cpu cuda

/-- 2D max-pooling with padding (no batch axis). PyTorch: `max_pool2d(..., padding=...)`. -/
def maxPool2dPad {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride padding : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (TensorRef α (.dim inC (.dim (Spec.poolOutDim inH kH stride padding)
    (.dim (Spec.poolOutDim inW kW stride padding) .scalar)))) := do
  if stride == 0 then
    throw <| IO.userError "torch: max_pool2d with padding requires stride > 0"
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.maxPool2dPad (t := t0)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
        padding)
      (h1 := h1) (h2 := h2) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let inCU32 ← okOrThrow <| Runtime.Autograd.Cuda.AnyBuffer.natToU32Checked inC
    let inHU32 ← okOrThrow <| Runtime.Autograd.Cuda.AnyBuffer.natToU32Checked inH
    let inWU32 ← okOrThrow <| Runtime.Autograd.Cuda.AnyBuffer.natToU32Checked inW
    let kHU32 ← okOrThrow <| Runtime.Autograd.Cuda.AnyBuffer.natToU32Checked kH
    let kWU32 ← okOrThrow <| Runtime.Autograd.Cuda.AnyBuffer.natToU32Checked kW
    let strideU32 ← okOrThrow <| Runtime.Autograd.Cuda.AnyBuffer.natToU32Checked stride
    let paddingU32 ← okOrThrow <| Runtime.Autograd.Cuda.AnyBuffer.natToU32Checked padding
    let outSh : Shape :=
      .dim inC (.dim (Spec.poolOutDim inH kH stride padding)
        (.dim (Spec.poolOutDim inW kW stride padding) .scalar))
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.unary (t := t0) "max_pool2d_pad" x.id
        (.dim inC (.dim inH (.dim inW .scalar))) outSh
        (forward := fun xBuf =>
          Runtime.Autograd.Cuda.torchleanMaxPool2dFwdCuda xBuf inCU32 inHU32 inWU32 kHU32 kWU32 strideU32 paddingU32)
        (backward := fun xBuf dLdy =>
          Runtime.Autograd.Cuda.torchleanMaxPool2dBwdCuda xBuf dLdy inCU32 inHU32 inWU32 kHU32 kWU32 strideU32 paddingU32)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .maxPool cpu cuda

/--
Smooth max-pooling (softmax pooling). Not a standard PyTorch primitive; see
`Torch.LinkedSession.smooth_max_pool2d`. Executable backends require finite, nonzero `beta` and
use max/min-shifted exponential weights.
-/
def smoothMaxPool2d {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) [Context α]
  [DecidableEq Shape]
  {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) (beta : α) :
  IO (TensorRef α (.dim inC (.dim (Spec.poolOutDim inH kH stride 0) (.dim (Spec.poolOutDim inW kW stride 0)
    .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.smoothMaxPool2d (t := t0)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := h1) (h2 := h2) x.id beta)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let betaF ← CudaBridge.TensorConv.toFloat (α := α) beta
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.smoothMaxPool2d (t := t0)
        (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
        (h1 := h1) (h2 := h2) x.id betaF
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .smoothMaxPool cpu cuda

/-- 2D average-pooling (no batch axis). PyTorch: `torch.nn.functional.avg_pool2d`. -/
def avgPool2d {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  (x : TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (TensorRef α (.dim inC (.dim (Spec.poolOutDim inH kH stride 0) (.dim (Spec.poolOutDim inW kW stride 0)
    .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.avgPool2d (t := t0)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := h1) (h2 := h2) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.avgPool2d (t := t0)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      h1 h2 x.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .avgPool cpu cuda

/-- 2D average-pooling with padding (no batch axis). PyTorch: `avg_pool2d(..., padding=...)`. -/
def avgPool2dPad {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride padding : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  (x : TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (TensorRef α (.dim inC (.dim (Spec.poolOutDim inH kH stride padding)
    (.dim (Spec.poolOutDim inW kW stride padding) .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.avgPool2dPad (t := t0)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
        padding)
      (h1 := h1) (h2 := h2) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.avgPool2dPad (t := t0)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
        padding)
      h1 h2 x.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .avgPool cpu cuda

end EagerSession

end Internal
end Torch
end Autograd
end Runtime
