/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Runtime
public import NN.Runtime.Autograd.TorchLean.Module

@[expose] public section

namespace TorchLean
namespace Module

/-!
# Module Execution

This file connects typed module definitions to executable scalar modules. It provides state
initialization and explicit execution settings. Command-line selection lives in
`NN.API.Module.Command`.

`ObjectiveDef` describes a scalar objective together with model state and input shapes.
Instantiating it produces a mutable `Objective` that can evaluate the objective, return explicit
state gradients, and update trainable entries. The shape lists remain part of both types, so
construction and execution use the same state ordering.
-/

/-- An immutable scalar objective together with its typed initial state. -/
abbrev ObjectiveDef (β : Type) (stateShapes inputShapes : List Spec.Shape)
    (dataInputShapes : List Spec.Shape := []) :=
  _root_.Runtime.Autograd.TorchLean.Module.ObjectiveDef β stateShapes inputShapes dataInputShapes

/-- Executable state for a model and scalar objective. -/
abbrev Objective (α β : Type) [_root_.Context α] [DecidableEq Spec.Shape]
    (stateShapes inputShapes : List Spec.Shape) (dataInputShapes : List Spec.Shape := []) :=
  _root_.Runtime.Autograd.TorchLean.Module.Objective α β stateShapes inputShapes dataInputShapes

/-- A reusable evaluator over an existing typed model state. -/
abbrev Evaluator (α β : Type) (stateShapes inputShapes dataInputShapes : List Spec.Shape)
    (outputShape : Spec.Shape) :=
  _root_.Runtime.Autograd.TorchLean.Module.Evaluator α β stateShapes inputShapes dataInputShapes
    outputShape

namespace RuntimeInit
export _root_.Runtime.Autograd.TorchLean.Module.RuntimeInit
  (FloatInit Plan xavierUniformForShape kaimingUniformForShape)
end RuntimeInit
export _root_.Runtime.Autograd.TorchLean.Module.Objective
  (create loss lossAndGradState gradState sgdStepWithLoss sgdStep initOptimizer optimizerStep
   optimizerStepWithLoss state loadState trainSGD trainWithOptimizer meanLoss)
export _root_.Runtime.Autograd.TorchLean.Module.Evaluator (withState)
namespace Evaluator
export _root_.Runtime.Autograd.TorchLean.Module.Evaluator (run)
end Evaluator
export _root_.Runtime.Autograd.TorchLean.Module.ObjectiveDef
  (evaluatorWithState lossWithState instantiateFloat64 instantiateWithPlan)

/--
Instantiate an `ObjectiveDef` under explicit Torch options such as `execution` and `device`.

The supplied options are passed unchanged to module construction, including the selected device and
execution strategy.
-/
def instantiateAs
    {α β : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    [Runtime.TensorTransfer α]
    {stateShapes inputShapes dataInputShapes : List Spec.Shape}
    (defn : ObjectiveDef β stateShapes inputShapes dataInputShapes)
    (cast : Float → α) (opts : Options) :
    IO (Objective α β stateShapes inputShapes dataInputShapes) :=
  _root_.Runtime.Autograd.TorchLean.Module.ObjectiveDef.instantiateWith
    (α := α) (β := β) (stateShapes := stateShapes) (inputShapes := inputShapes)
    (dataInputShapes := dataInputShapes) defn cast opts

/--
Instantiate an executable objective using the runtime scalar's standard `Float` conversion.

This is the low-level constructor for custom losses and multi-input programs. The higher-level
`nn.Module` and `Trainer` APIs should be preferred for ordinary sequential models.
-/
def instantiate
    {α β : Type} [_root_.Context α] [DecidableEq Spec.Shape] [Runtime.FromFloat α]
    [Runtime.TensorTransfer α]
    {stateShapes inputShapes dataInputShapes : List Spec.Shape}
    (opts : Options)
    (defn : ObjectiveDef β stateShapes inputShapes dataInputShapes)
    (cast : Float → α := Runtime.ofFloat) :
    IO (Objective α β stateShapes inputShapes dataInputShapes) :=
  instantiateAs (α := α) (β := β) defn cast opts

end Module
end TorchLean
