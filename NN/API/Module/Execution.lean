/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Runtime
public import NN.GraphSpec.Models.TorchLean
public import NN.Runtime.Autograd.TorchLean

import Mathlib.Algebra.Order.Algebra
import NN.Spec.Autograd.AutogradSpec

@[expose] public section

namespace TorchLean
namespace Module

export _root_.Runtime.Autograd.Torch (ExecutionMode Options)

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

export _root_.Runtime.Autograd.TorchLean.Module
  (Evaluator ObjectiveEvaluator ObjectiveDef Objective)
namespace RuntimeInit
export _root_.Runtime.Autograd.TorchLean.Module.RuntimeInit
  (FloatInit Plan xavierUniformForShape kaimingUniformForShape)
end RuntimeInit
export _root_.Runtime.Autograd.TorchLean.Module.Objective
  (create loss lossAndGradState gradState sgdStepWithLoss sgdStep initOptimizer optimizerStep
   optimizerStepWithLoss state loadState trainSGD trainWithOptimizer meanLoss)
export _root_.Runtime.Autograd.TorchLean.Module.Evaluator (evaluatePacked withState)
export _root_.Runtime.Autograd.TorchLean.Module.ObjectiveDef
  (evaluatorWithState lossWithState instantiate instantiateFloat64 instantiateWithPlan
   instantiateWithInit)

/--
Instantiate an `ObjectiveDef` under explicit Torch options such as `execution` and `device`.

The supplied options are passed unchanged to module construction, including the selected device and
execution strategy.
-/
def instantiateAs
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    {stateShapes inputShapes natInputShapes : List Spec.Shape}
    (defn : ObjectiveDef stateShapes inputShapes natInputShapes)
    (cast : Float → α) (opts : Options) :
    IO (Objective α stateShapes inputShapes natInputShapes) :=
  _root_.Runtime.Autograd.TorchLean.Module.ObjectiveDef.instantiateWith
    (α := α) (stateShapes := stateShapes) (inputShapes := inputShapes)
    (natInputShapes := natInputShapes) defn cast opts

end Module
end TorchLean
