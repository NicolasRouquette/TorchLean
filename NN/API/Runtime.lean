/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Report
public import NN.API.Scalar
public import NN.Runtime.Autograd.Torch.Core.Functional
public import NN.Runtime.Autograd.Torch.Core.Types
public import NN.Runtime.Autograd.TorchLean.Program

/-!
# Runtime Selection

Scalar semantics, execution mode, device, and backend-contract inspection.
-/

@[expose] public section

namespace TorchLean

export _root_.Runtime.Autograd.Torch (Options)

namespace Runtime

export _root_.Runtime.Autograd.Torch (Ops)
export _root_.Runtime.Autograd.TorchLean (RefTy Program)

/-- Runtime execution strategy: eager evaluation or typed-graph execution. -/
abbrev ExecutionMode := _root_.Runtime.Autograd.Torch.ExecutionMode

namespace ExecutionMode

export _root_.Runtime.Autograd.Torch.ExecutionMode (eager typedGraph)

/-- Parse the stable command-line spelling of an execution mode. -/
def parse (value : String) : Except String ExecutionMode :=
  match value with
  | "eager" => pure .eager
  | "typed-graph" => pure .typedGraph
  | _ => throw s!"unknown execution mode `{value}` (supported: eager | typed-graph)"

end ExecutionMode

/-- Physical or logical device selected for runtime execution. -/
abbrev Device := NN.Backend.Device

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

/-- Backend-contract profile corresponding to the selected runtime options. -/
def profileForOptions (opts : Options) : Except String NN.Backend.BackendProfile :=
  opts.effectiveBackendProfile

/-- Plan operations under the runtime-selected backend-contract profile. -/
def planReport (opts : Options) (ops : List NN.Backend.BackendOp) : Except String String := do
  let profile ← profileForOptions opts
  profile.planReport ops

/-- Print the selected backend capsules for operations. -/
def printPlan (opts : Options) (ops : List NN.Backend.BackendOp) : IO Unit := do
  match planReport opts ops with
  | .ok report => IO.println report
  | .error msg => IO.println s!"kernel plan unavailable: {msg}"

end BackendContracts

end Runtime


end TorchLean
