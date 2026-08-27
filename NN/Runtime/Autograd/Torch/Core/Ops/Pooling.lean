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
  {d C : Nat} {inSpatial kernel stride padding : Spec.Tensor Nat [d]}
  {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
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
  dispatchCudaOpt (α := α) s .maxPool #[x.identity?] cpu cuda

/--
N-D average pooling for channels-first tensors `(C, spatial...)` (no batch axis).

PyTorch comparison: `torch.nn.functional.avg_pool1d` / `avg_pool2d` / `avg_pool3d` depending on the
spatial rank `d`.
-/
def avgPool {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {d C : Nat} {inSpatial kernel stride padding : Spec.Tensor Nat [d]}
  (hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0)
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
  dispatchCudaOpt (α := α) s .avgPool #[x.identity?] cpu cuda

/--
N-D smooth max pooling (log-sum-exp surrogate) for channels-first tensors `(C, spatial...)`.

This is a differentiable approximation to max pooling; PyTorch does not expose it as a single
primitive, but it can be emulated with `logsumexp` over local windows. Executable backends require
at least one spatial dimension and a finite, nonzero `beta`; evaluation uses an input-space
max/min shift so the exponential weights remain stable for large finite values.
-/
def smoothMaxPool {α : Type} [TensorTransfer α] (s : EagerSession α) [Context α]
  [DecidableEq α] [DecidableEq Shape]
  {d C : Nat} {inSpatial kernel stride padding : Spec.Tensor Nat [d]}
  {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
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
    let betaF ← TensorTransfer.toFloat (α := α) beta
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.smoothMaxPool (t := t0)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel) x.id betaF)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .smoothMaxPool #[x.identity?] cpu cuda

end EagerSession

end Internal
end Torch
end Autograd
end Runtime
