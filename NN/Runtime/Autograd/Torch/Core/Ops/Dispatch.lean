/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Session

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

/-!
## Tensor ops (eager tape wrappers)

The following definitions are the eager front-end for `Runtime.Autograd.Tape.*` primitives. Each one:
- reads the current tape from `s.tape`,
- appends a new node/leaf via a `Tape.*` constructor,
- writes the updated tape back, and
- returns a fresh `TensorRef` pointing to the new node id.

PyTorch comparison: this is the standard eager autograd mechanism (a dynamic tape of ops).
-/

/--
Dispatch an eager operation through its selected CPU or CUDA capsule.

`cudaProviders` names the providers implemented by the supplied CUDA handler. The selected capsule
is bound to the matching handler before any implementation runs. Returning `none` still means that
the operation has no implementation in this CUDA runtime; there is no per-operation CPU fallback.
-/
def dispatchCudaCapsuleOpt {α β : Type} (s : EagerSession α) (op : NN.Backend.BackendOp)
    (cudaProviders : List NN.Backend.Provider) (cpu : IO β)
    (cuda : NN.Backend.KernelCapsule → IO (Option β)) : IO β := do
  let cpuHandler : NN.Backend.KernelHandler β :=
    { name := "TorchLean reference CPU"
      op
      provider := .reference
      device := .cpu
      execute := fun _ => cpu }
  let cudaHandlers : List (NN.Backend.KernelHandler β) :=
    cudaProviders.map fun provider =>
      { name := s!"CUDA executor for {reprStr provider}"
        op
        provider
        device := .cuda
        execute := fun capsule => do
          match ← cuda capsule with
          | some result => pure result
          | none =>
              throw <| IO.userError <|
                s!"torch: cuda: `{op.name}` is unsupported by `{reprStr provider}`" }
  s.executeSelected op (cpuHandler :: cudaHandlers)

/--
Dispatch an eager operation implemented by the reference CPU and TorchLean native CUDA runtimes.
-/
def dispatchCudaOpt {α β : Type} (s : EagerSession α) (op : NN.Backend.BackendOp)
    (cpu : IO β) (cuda : IO (Option β)) : IO β :=
  dispatchCudaCapsuleOpt s op [.nativeCuda] cpu (fun _ => cuda)

/--
Validate one float-encoded token id and return the corresponding `Nat`.

This is intentionally stricter than `Float.floor`: language-model targets are discrete labels, so a
fractional value is almost certainly a bad dataset or adapter boundary. Rejecting it here prevents a
quiet change of class label before the embedding or cross-entropy code sees the id.
-/
def natOfTokenFloat (i : Nat) (x : Float) : IO Nat := do
  if x.isNaN || x.isInf then
    throw <| IO.userError s!"torch: token id at index {i} is not finite"
  else if x < 0.0 then
    throw <| IO.userError s!"torch: token id at index {i} is negative: {x}"
  else
    let y := Float.floor x
    if y != x then
      throw <| IO.userError s!"torch: token id at index {i} is not an integer: {x}"
    else
      let n := y.toUInt64.toNat
      if Float.ofNat n == x then
        pure n
      else
        throw <| IO.userError s!"torch: token id at index {i} is outside the supported Nat range: {x}"

end EagerSession

end Internal
end Torch
end Autograd
end Runtime
