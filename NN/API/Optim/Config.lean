/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# Optimizer Configuration

Optimizer algorithms and hyperparameters shared by the public training and runtime APIs.
-/

@[expose] public section

namespace TorchLean
namespace optim

/-- Optimizer algorithm and hyperparameters used by the public training APIs. -/
inductive Optimizer where
  /-- SGD optimizer configuration. -/
  | sgd (lr : Float) (momentum : Float := 0.0)
  /-- AdaGrad optimizer configuration. -/
  | adagrad (lr : Float) (epsilon : Float := 1e-10)
  /-- RMSProp optimizer configuration. -/
  | rmsprop (lr : Float) (decay : Float := 0.99) (epsilon : Float := 1e-8)
  /-- Adam optimizer configuration. -/
  | adam (lr : Float) (beta1 : Float := 0.9) (beta2 : Float := 0.999)
      (epsilon : Float := 1e-8)
  /-- AdamW optimizer configuration with decoupled weight decay. -/
  | adamw (lr : Float) (weightDecay : Float := 0.01)
      (beta1 : Float := 0.9) (beta2 : Float := 0.999) (epsilon : Float := 1e-8)
  /-- Adadelta optimizer configuration. -/
  | adadelta (lr : Float := 1.0) (rho : Float := 0.9) (epsilon : Float := 1e-6)
deriving Repr

namespace Optimizer

/-- Return the base learning rate encoded in an optimizer configuration. -/
def learningRate : Optimizer -> Float
  | .sgd lr _ => lr
  | .adagrad lr _ => lr
  | .rmsprop lr _ _ => lr
  | .adam lr _ _ _ => lr
  | .adamw lr _ _ _ _ => lr
  | .adadelta lr _ _ => lr

namespace Internal

def requireFiniteNonnegative (name : String) (value : Float) : Except String Unit := do
  unless value.isFinite && 0.0 <= value do
    throw s!"optimizer: {name} must be finite and nonnegative"

def requireUnitInterval (name : String) (value : Float) : Except String Unit := do
  unless value.isFinite && 0.0 <= value && value < 1.0 do
    throw s!"optimizer: {name} must be finite and satisfy 0 <= {name} < 1"

def requireFinitePositive (name : String) (value : Float) : Except String Unit := do
  unless value.isFinite && 0.0 < value do
    throw s!"optimizer: {name} must be finite and positive"

end Internal

/--
Check the numerical domain of an optimizer configuration before allocating optimizer state.

The checks rule out undefined bias corrections and non-finite updates. They are shared by the
trainer, manual-module, and reinforcement-learning entry points.
-/
def validate : Optimizer -> Except String Unit
  | .sgd lr momentum => do
      Internal.requireFiniteNonnegative "learning rate" lr
      Internal.requireUnitInterval "momentum" momentum
  | .adagrad lr epsilon => do
      Internal.requireFiniteNonnegative "learning rate" lr
      Internal.requireFinitePositive "epsilon" epsilon
  | .rmsprop lr decay epsilon => do
      Internal.requireFiniteNonnegative "learning rate" lr
      Internal.requireUnitInterval "decay" decay
      Internal.requireFinitePositive "epsilon" epsilon
  | .adam lr beta1 beta2 epsilon => do
      Internal.requireFiniteNonnegative "learning rate" lr
      Internal.requireUnitInterval "beta1" beta1
      Internal.requireUnitInterval "beta2" beta2
      Internal.requireFinitePositive "epsilon" epsilon
  | .adamw lr weightDecay beta1 beta2 epsilon => do
      Internal.requireFiniteNonnegative "learning rate" lr
      Internal.requireFiniteNonnegative "weight decay" weightDecay
      Internal.requireUnitInterval "beta1" beta1
      Internal.requireUnitInterval "beta2" beta2
      Internal.requireFinitePositive "epsilon" epsilon
  | .adadelta lr rho epsilon => do
      Internal.requireFiniteNonnegative "learning rate" lr
      Internal.requireUnitInterval "rho" rho
      Internal.requireFinitePositive "epsilon" epsilon

end Optimizer
end optim
end TorchLean
