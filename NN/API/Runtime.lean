/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Report
public import NN.API.Scalar
public import NN.Runtime.Autograd.Torch.Core.TensorTransfer
public import NN.Runtime.Autograd.TorchLean.Functional.ShapeOps

/-!
# Runtime Selection

Scalar semantics, execution mode, device, and backend-contract inspection.
-/

@[expose] public section

namespace TorchLean

export _root_.Runtime.Autograd.Torch (Options)

namespace Runtime

export _root_.Runtime.Autograd.Torch (Ops)
export _root_.Runtime.Autograd.TorchLean (Program)
export _root_.Runtime.Autograd.Torch (TensorTransfer)
export _root_.Runtime.Autograd.Torch.TensorTransfer (toFloatTensor)

open _root_.Spec

/--
A shape-indexed handle to a value owned by a runtime program.

Unlike `Tensor`, a `ValueRef` does not contain tensor elements. It names an intermediate value in
an eager session or typed graph and is valid only in the program that created it.
-/
abbrev ValueRef (m : Type → Type) (α : Type)
    [Context α] [DecidableEq Shape] [Monad m] [Ops (m := m) (α := α)]
    (shape : Shape) :=
  _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) shape

/-- Apply an affine map to the final axis, independently over every index in `leading`. -/
def linear {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    (leading : List Nat := []) {inDim outDim : Nat}
    (weight : ValueRef (m := m) (α := α) [outDim, inDim])
    (bias : ValueRef (m := m) (α := α) [outDim])
    (input : ValueRef (m := m) (α := α) (leading ++ [inDim])) :
    m (ValueRef (m := m) (α := α) (leading ++ [outDim])) :=
  by
    let input' : ValueRef (m := m) (α := α)
        ((Shape.ofList leading).concat [inDim]) := by
      simpa only [Shape.ofList_append] using input
    simpa only [Shape.ofList_append] using
      (_root_.Runtime.Autograd.TorchLean.linearEach
        (m := m) (α := α) (leadingShape := Shape.ofList leading) weight bias input')

/-!
## Operation-Polymorphic Programs

These operations build or execute runtime programs. They consume shape-indexed references rather
than materialized `Tensor` values, so they live under `Runtime` instead of the pure tensor API.
Ordinary models should use `nn` and `Trainer`; this lower-level surface is useful for custom losses,
verification programs, and graph-lowering tools.
-/

export _root_.Runtime.Autograd.Torch
  (const add sub mul scale abs sqrt clamp max min
   broadcastTo reshape reduceSum reduceMean select indexSelect scatterAdd
   matmul
   relu silu gelu sigmoid tanh softplus exp log inv safeLog
   sum flatten mseLoss)
export _root_.Runtime.Autograd.TorchLean
  (mapEach maxPool avgPool smoothMaxPool layerNorm multiHeadAttention
   multiHeadAttentionOutputBias conv convTranspose)
export _root_.Runtime.Autograd.TorchLean.F (permute softmax logSoftmax)

export _root_.Runtime.Autograd.Torch (ExecutionMode)

namespace ExecutionMode

export _root_.Runtime.Autograd.Torch.ExecutionMode (eager typedGraph)

/-- Parse the stable command-line spelling of an execution mode. -/
def parse (value : String) : Except String ExecutionMode :=
  match value with
  | "eager" => pure .eager
  | "typed-graph" => pure .typedGraph
  | _ => throw s!"unknown execution mode `{value}` (supported: eager | typed-graph)"

end ExecutionMode

export NN.Backend (Device)

namespace Device

export NN.Backend.Device (cpu cuda rocm metal wasm tpu trainium custom external)

/--
Parse a public device selector. `auto` chooses the portable CPU runtime; every other value is
validated against the devices known to the backend registry.
-/
def parse (value : String) : Except String Device :=
  if value == "auto" then pure .cpu else NN.Backend.Device.parse value

end Device

namespace BackendContracts

/-- Plan operations under the runtime-selected backend-contract profile. -/
def planReport (opts : Options) (ops : Array NN.Backend.BackendOp) : Except String String := do
  let profile ← opts.effectiveBackendProfile
  profile.planReport ops

/-- Print the selected backend capsules for operations. -/
def printPlan (opts : Options) (ops : Array NN.Backend.BackendOp) : IO Unit := do
  match planReport opts ops with
  | .ok report => IO.println report
  | .error msg => IO.println s!"kernel plan unavailable: {msg}"

end BackendContracts

end Runtime


end TorchLean
