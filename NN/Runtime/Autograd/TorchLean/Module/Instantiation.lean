/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.Module.Objective

/-!
# Module Instantiation

Constructors that instantiate scalar-objective definitions from semantic tensors or storage-first
runtime initialization plans.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor
open Proofs.Autograd.Algebra

namespace Module

namespace ObjectiveDef

/--
Instantiate an `ObjectiveDef` by casting Float initializers to `α` and choosing Torch options.

This is the most general constructor. The shorter `instantiate` entrypoint chooses standard runtime
options before calling this function.
-/
def instantiateWith {α β : Type} [Context α] [DecidableEq Shape]
        [_root_.Runtime.Autograd.Torch.TensorTransfer α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (d : ObjectiveDef β stateShapes inputShapes dataInputShapes)
    (cast : Float → α) (opts : Torch.Options) :
    IO (Objective α β stateShapes inputShapes dataInputShapes) := do
  let initState : TorchLean.TensorPack α stateShapes := castPack (α := α) cast d.initState
  Objective.create (α := α) (stateShapes := stateShapes) (inputShapes := inputShapes)
    (dataInputShapes := dataInputShapes)
    (opts := opts) (requiresGrad := d.requiresGrad)
    (validateDataInputs := d.validateDataInputs)
    (loss := d.loss (α := α)) initState

/--
Instantiate a module using runtime parameter initializers.

This is the runtime-initialized sibling of `instantiateWith`. Instead of first building every
initial parameter as a full Lean tensor, it creates minimal zero host tensors and then applies a
shape-indexed runtime plan to the module parameters.  In CUDA mode those initializers allocate
device buffers directly and mark the host copies stale; public parameter readback still
synchronizes them through the existing CUDA mirror machinery.
-/
def instantiateWithPlan {α β : Type} [Context α] [DecidableEq Shape]
    [Torch.TensorTransfer α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (d : ObjectiveDef β stateShapes inputShapes dataInputShapes)
    (cast : Float → α)
    (opts : Torch.Options)
    (plan : RuntimeInit.Plan stateShapes) :
    IO (Objective α β stateShapes inputShapes dataInputShapes) := do
  let initState := RuntimeInit.zeroPack (cast 0.0) (ss := stateShapes)
  let module ← Objective.create (α := α) (stateShapes := stateShapes)
    (inputShapes := inputShapes) (dataInputShapes := dataInputShapes)
    (opts := opts) (requiresGrad := d.requiresGrad)
    (validateDataInputs := d.validateDataInputs)
    (loss := d.loss (α := α)) initState
  RuntimeInit.applyPlan (α := α) cast (opts := opts) module.trainer.state plan
  pure module

/--
Instantiate a module over Lean's binary64 `Float` type.

This is an explicit binary64 path rather than the public runtime default. Definitions with a
storage-first plan use it; other definitions retain tensor-valued initialization.
-/
def instantiateFloat64 {β : Type}
    {stateShapes inputShapes dataInputShapes : List Shape}
    (d : ObjectiveDef β stateShapes inputShapes dataInputShapes) (opts : Torch.Options) :
    IO (Objective Float β stateShapes inputShapes dataInputShapes) :=
  match d.runtimeInit with
  | some plan => instantiateWithPlan d id opts plan
  | none => instantiateWith (α := Float) d id opts

/-- Convenience instantiator that chooses only the execution mode. -/
def instantiate {α β : Type} [Context α] [DecidableEq Shape]
    [_root_.Runtime.Autograd.Torch.TensorTransfer α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (d : ObjectiveDef β stateShapes inputShapes dataInputShapes)
    (cast : Float → α) (execution : Torch.ExecutionMode := .eager) :
    IO (Objective α β stateShapes inputShapes dataInputShapes) := do
  instantiateWith (α := α) (stateShapes := stateShapes) (inputShapes := inputShapes)
    (dataInputShapes := dataInputShapes)
    d cast { execution := execution }

end ObjectiveDef

end Module

end TorchLean
end Autograd
end Runtime
