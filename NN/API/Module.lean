/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Module.Execution
public import NN.API.Neural.Builders
public import NN.API.Tensor
public import NN.API.Trainer.Manual.Core

/-!
# Executable Modules

Executable module operations for manual runtime and example code.
-/

@[expose] public section

namespace TorchLean

namespace Module

export _root_.Runtime.Autograd.TorchLean.Module (BoundOptimizer bindOptimizer)

/--
Live parameter and buffer storage shared by executable modules.

The shape list remains in the type, so replacing the state cannot silently reorder parameters or
load tensors with incompatible dimensions. Mutation is confined to this runtime object; model and
layer definitions remain immutable values that can be lowered and reasoned about.
-/
structure RuntimeState (α : Type) [_root_.Context α] [DecidableEq Shape]
    (stateShapes : List Shape) where
  /-- Live parameter and buffer storage. -/
  stateRef : _root_.Runtime.Autograd.Torch.ParamList α stateShapes
  /-- Runtime options fixed when the state is instantiated. -/
  options : Options
  /-- Current behavior of training-sensitive layers such as dropout and normalization. -/
  modeRef : IO.Ref _root_.Runtime.Autograd.TorchLean.NN.Mode
  /-- Host/device conversion used by the selected runtime scalar. -/
  tensorConv : _root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α

namespace RuntimeState

/-- Instantiate runtime state after converting semantic initializer tensors to `α`. -/
def instantiateAs {α : Type} [_root_.Context α] [DecidableEq Shape]
    [tensorConv : _root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    {stateShapes : List Shape} (initial : TensorPack Float stateShapes)
    (runtimeInit : Option
      (_root_.Runtime.Autograd.TorchLean.Module.RuntimeInit.Plan stateShapes))
    (requiresGrad : List Bool) (options : Options) (cast : Float → α) :
    IO (RuntimeState α stateShapes) := do
  let stateRef ←
    match runtimeInit with
    | some plan => do
        let empty := _root_.Runtime.Autograd.TorchLean.Module.RuntimeInit.zeroTList
          (cast 0.0) (ss := stateShapes)
        let stateRef ← _root_.Runtime.Autograd.Torch.ParamList.ofTListWithRequiresGrad
          empty requiresGrad
        _root_.Runtime.Autograd.TorchLean.Module.RuntimeInit.applyPlan
          (α := α) cast options stateRef plan
        pure stateRef
    | none => do
        let values := _root_.Runtime.Autograd.TorchLean.Module.castTList cast initial
        _root_.Runtime.Autograd.Torch.ParamList.ofTListWithRequiresGrad values requiresGrad
  let modeRef ← IO.mkRef (_root_.Runtime.Autograd.TorchLean.NN.Mode.train)
  pure { stateRef, options, modeRef, tensorConv }

/-- Read the current behavior of training-sensitive layers. -/
def mode {α : Type} [_root_.Context α] [DecidableEq Shape] {stateShapes : List Shape}
    (state : RuntimeState α stateShapes) : IO _root_.Runtime.Autograd.TorchLean.NN.Mode :=
  state.modeRef.get

/-- Enable training behavior. -/
def train {α : Type} [_root_.Context α] [DecidableEq Shape] {stateShapes : List Shape}
    (state : RuntimeState α stateShapes) : IO Unit :=
  state.modeRef.set .train

/-- Enable evaluation behavior. -/
def eval {α : Type} [_root_.Context α] [DecidableEq Shape] {stateShapes : List Shape}
    (state : RuntimeState α stateShapes) : IO Unit :=
  state.modeRef.set .eval

/-- Return `true` exactly when the state uses training behavior. -/
def isTraining {α : Type} [_root_.Context α] [DecidableEq Shape]
    {stateShapes : List Shape} (state : RuntimeState α stateShapes) : IO Bool := do
  pure ((← state.mode) == .train)

/-- Read all parameters and buffers, synchronizing device storage when necessary. -/
def state {α : Type} [_root_.Context α] [DecidableEq Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    {stateShapes : List Shape} (runtimeState : RuntimeState α stateShapes) :
    IO (TensorPack α stateShapes) :=
  _root_.Runtime.Autograd.Torch.ParamList.valuesSynced runtimeState.stateRef

/-- Replace all parameters and buffers with a shape-compatible state. -/
def loadState {α : Type} [_root_.Context α] [DecidableEq Shape] {stateShapes : List Shape}
    (runtimeState : RuntimeState α stateShapes) (values : TensorPack α stateShapes) : IO Unit :=
  _root_.Runtime.Autograd.Torch.ParamList.setValues runtimeState.stateRef values

end RuntimeState

end Module

namespace nn

/--
An ordinary tensor model with live, mutable runtime state.

`nn.Sequential` remains the immutable, shape-checked model definition used by proofs and graph
lowering. An `nn.Module` is its runtime counterpart: it owns one parameter set and a mutable
train/eval flag. This separation gives executable code the familiar module lifecycle without
hiding mutation inside the mathematical model.
-/
structure Module (α : Type) [_root_.Context α] [DecidableEq Shape]
    {σ τ : Shape} (model : nn.Sequential σ τ)
    extends _root_.TorchLean.Module.RuntimeState α (nn.stateShapes model)

/-- An indexed-input model with live, mutable runtime state. -/
structure IndexedModule (α : Type) [_root_.Context α] [DecidableEq Shape]
    {σ τ : Shape} (model : nn.IndexedModel σ τ)
    extends _root_.TorchLean.Module.RuntimeState α model.stateShapes

namespace Module

/-- Instantiate a checked model under an explicitly selected scalar type. -/
def instantiateAs {σ τ : Shape} {α : Type}
    [_root_.Context α] [DecidableEq Shape]
    [tensorConv : _root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (model : nn.Sequential σ τ) (options : Options) (cast : Float → α) :
    IO (Module α model) := do
  let state ← _root_.TorchLean.Module.RuntimeState.instantiateAs
    (nn.initState model) (nn.runtimeInit? model) (nn.requiresGrad model) options cast
  pure { toRuntimeState := state }

/--
Instantiate a native binary32 model with mutable parameter and buffer storage.
-/
def instantiate {σ τ : Shape} (model : nn.Sequential σ τ) (options : Options) :
    IO (Module Float32 model) :=
  instantiateAs model options Float.toFloat32

/-- Read whether training-sensitive layers currently use training or evaluation behavior. -/
def mode {σ τ : Shape} {α : Type} [_root_.Context α] [DecidableEq Shape]
    {model : nn.Sequential σ τ} (module : Module α model) :
    IO _root_.Runtime.Autograd.TorchLean.NN.Mode :=
  module.toRuntimeState.mode

/-- Enable training behavior for dropout, normalization buffers, and similar layers. -/
def train {σ τ : Shape} {α : Type} [_root_.Context α] [DecidableEq Shape]
    {model : nn.Sequential σ τ} (module : Module α model) : IO Unit :=
  module.toRuntimeState.train

/-- Enable deterministic evaluation behavior for training-sensitive layers. -/
def eval {σ τ : Shape} {α : Type} [_root_.Context α] [DecidableEq Shape]
    {model : nn.Sequential σ τ} (module : Module α model) : IO Unit :=
  module.toRuntimeState.eval

/-- Return `true` exactly when this module uses training behavior. -/
def isTraining {σ τ : Shape} {α : Type} [_root_.Context α] [DecidableEq Shape]
    {model : nn.Sequential σ τ} (module : Module α model) : IO Bool := do
  module.toRuntimeState.isTraining

/-- Read the complete parameter-and-buffer state, synchronizing device storage when necessary. -/
def state {σ τ : Shape} {α : Type}
    [_root_.Context α] [DecidableEq Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    {model : nn.Sequential σ τ} (module : Module α model) :
    IO (TensorPack α (nn.stateShapes model)) :=
  module.toRuntimeState.state

/-- Replace the complete shape-indexed parameter-and-buffer state. -/
def loadState {σ τ : Shape} {α : Type} [_root_.Context α] [DecidableEq Shape]
    {model : nn.Sequential σ τ} (module : Module α model)
    (state : TensorPack α (nn.stateShapes model)) : IO Unit :=
  module.toRuntimeState.loadState state

/--
Run one concrete input without constructing a backward tape.

The module's mode controls training-sensitive layer behavior. This concrete runtime call does not
construct a backward tape; differentiable model programs use `nn.forward`.
-/
def run {σ τ : Shape} {α : Type}
    [_root_.Context α] [DecidableEq Shape]
    [tensorConv : _root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    {model : nn.Sequential σ τ} (module : Module α model)
    (input : Tensor α σ) : IO (Tensor α τ) := do
  _root_.Runtime.Autograd.TorchLean.NN.Seq.forwardNoGrad
    (α := α) (tensorConv := tensorConv)
    module.options model module.stateRef input
    (mode := ← module.mode)

/-- Evaluation-mode inference that restores the previous mode after the call. -/
def predict {σ τ : Shape} {α : Type}
    [_root_.Context α] [DecidableEq Shape]
    [tensorConv : _root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    {model : nn.Sequential σ τ} (module : Module α model)
    (input : Tensor α σ) : IO (Tensor α τ) := do
  let previous ← module.mode
  module.eval
  try
    module.run input
  finally
    module.modeRef.set previous

end Module

namespace IndexedModule

/-- Instantiate an indexed model under an explicitly selected scalar type. -/
def instantiateAs {σ τ : Shape} {α : Type}
    [_root_.Context α] [DecidableEq Shape]
    [tensorConv : _root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (model : nn.IndexedModel σ τ) (options : Options) (cast : Float → α) :
    IO (IndexedModule α model) := do
  let state ← _root_.TorchLean.Module.RuntimeState.instantiateAs
    model.initState model.runtimeInit model.requiresGrad options cast
  pure { toRuntimeState := state }

/-- Instantiate a native binary32 indexed model. -/
def instantiate {σ τ : Shape} (model : nn.IndexedModel σ τ) (options : Options) :
    IO (IndexedModule Float32 model) :=
  instantiateAs model options Float.toFloat32

/-- Read the current behavior of training-sensitive layers. -/
def mode {σ τ : Shape} {α : Type} [_root_.Context α] [DecidableEq Shape]
    {model : nn.IndexedModel σ τ} (module : IndexedModule α model) :
    IO _root_.Runtime.Autograd.TorchLean.NN.Mode :=
  module.toRuntimeState.mode

/-- Enable training behavior. -/
def train {σ τ : Shape} {α : Type} [_root_.Context α] [DecidableEq Shape]
    {model : nn.IndexedModel σ τ} (module : IndexedModule α model) : IO Unit :=
  module.toRuntimeState.train

/-- Enable evaluation behavior. -/
def eval {σ τ : Shape} {α : Type} [_root_.Context α] [DecidableEq Shape]
    {model : nn.IndexedModel σ τ} (module : IndexedModule α model) : IO Unit :=
  module.toRuntimeState.eval

/-- Return `true` exactly when the module uses training behavior. -/
def isTraining {σ τ : Shape} {α : Type} [_root_.Context α] [DecidableEq Shape]
    {model : nn.IndexedModel σ τ} (module : IndexedModule α model) : IO Bool :=
  module.toRuntimeState.isTraining

/-- Read the complete parameter-and-buffer state. -/
def state {σ τ : Shape} {α : Type}
    [_root_.Context α] [DecidableEq Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    {model : nn.IndexedModel σ τ} (module : IndexedModule α model) :
    IO (TensorPack α model.stateShapes) :=
  module.toRuntimeState.state

/-- Replace the complete shape-indexed parameter-and-buffer state. -/
def loadState {σ τ : Shape} {α : Type} [_root_.Context α] [DecidableEq Shape]
    {model : nn.IndexedModel σ τ} (module : IndexedModule α model)
    (state : TensorPack α model.stateShapes) : IO Unit :=
  module.toRuntimeState.loadState state

/-- Run one validated index tensor without constructing a backward tape. -/
def run {σ τ : Shape} {α : Type}
    [_root_.Context α] [DecidableEq Shape]
    [tensorConv : _root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    {model : nn.IndexedModel σ τ} (module : IndexedModule α model)
    (input : Tensor Nat σ) : IO (Tensor α τ) := do
  match model.validateInput input with
  | .error message => throw <| IO.userError message
  | .ok () =>
      let currentMode ← module.mode
      let program : _root_.Runtime.Autograd.TorchLean.ProgramWithNatInputs α
          (model.stateShapes ++ []) [σ] τ :=
        fun {m} _ _ => by
          simpa using (model.forward currentMode (α := α) (m := m))
      let evaluator ← _root_.Runtime.Autograd.TorchLean.Module.Evaluator.withState
        (program := program)
        module.options module.stateRef
      _root_.Runtime.Autograd.TorchLean.Module.Evaluator.evaluatePacked
        evaluator .nil (.cons input .nil)

/-- Evaluation-mode inference that restores the previous mode after the call. -/
def predict {σ τ : Shape} {α : Type}
    [_root_.Context α] [DecidableEq Shape]
    [tensorConv : _root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    {model : nn.IndexedModel σ τ} (module : IndexedModule α model)
    (input : Tensor Nat σ) : IO (Tensor α τ) := do
  let previous ← module.mode
  module.eval
  try
    module.run input
  finally
    module.modeRef.set previous

end IndexedModule

end nn

namespace Module

/--
Instantiate an executable runtime objective from an `ObjectiveDef`.

This is the low-level entry point for custom losses and multi-input runtime programs. Ordinary
sequential models use `nn.Module.instantiate` or the `Trainer` API.
-/
def instantiate
    {α : Type} [_root_.Context α] [DecidableEq Shape] [Runtime.FromFloat α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    {stateShapes inputShapes natInputShapes : List Shape}
    (opts : Options)
    (defn : ObjectiveDef stateShapes inputShapes natInputShapes)
    (cast : Float → α := Runtime.ofFloat) :
    IO (Objective α stateShapes inputShapes natInputShapes) :=
  instantiateAs (α := α) defn cast opts

namespace Supervised

/-- Run evaluation-mode prediction through a supervised runtime module. -/
def predict {σ τ : Shape} {α : Type}
    [_root_.Context α] [DecidableEq Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (opts : Options)
    (model : TorchLean.nn.Sequential σ τ)
    (m : Objective α (nn.stateShapes model) [σ, τ])
    (x : Tensor α σ) : IO (Tensor α τ) :=
  _root_.Runtime.Autograd.TorchLean.NN.Seq.predict
    (α := α) opts model m.trainer.state x

end Supervised

/--
Evaluate one supervised sample through a runtime module and return the scalar loss value.

This packages the common public example pattern `Module.loss ...; Tensor.item`.
-/
def lossValue {σ τ : Shape} {α : Type}
    [_root_.Context α] [DecidableEq Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (model : TorchLean.nn.Sequential σ τ)
    (m : Objective α (nn.stateShapes model) [σ, τ])
    (sample : Sample.Supervised α σ τ) : IO α := do
  let loss ← loss (α := α) m sample .nil
  pure (Tensor.item loss)

/--
Bind Adam state and updates to a concrete runtime module.

This packages the common public pattern of constructing Adam and binding it to a module.
-/
def bindAdam {α : Type}
    [_root_.Context α] [DecidableEq Shape]
    {stateShapes inputShapes : List Shape}
    (m : Objective α stateShapes inputShapes)
    (lr beta1 beta2 epsilon : α) := do
  let opt := _root_.Runtime.Autograd.TorchLean.Optim.adam (α := α) lr beta1 beta2 epsilon
    (paramShapes := stateShapes)
  bindOptimizer (α := α) m opt

/--
Bind AdamW state and updates to a concrete runtime module.
-/
def bindAdamW {α : Type}
    [_root_.Context α] [DecidableEq Shape]
    {stateShapes inputShapes : List Shape}
    (m : Objective α stateShapes inputShapes)
    (lr weightDecay beta1 beta2 epsilon : α) := do
  let opt := _root_.Runtime.Autograd.TorchLean.Optim.adamw (α := α)
    (paramShapes := stateShapes)
    lr weightDecay beta1 beta2 epsilon
  bindOptimizer (α := α) m opt

/--
Bind SGD state and updates to a concrete runtime module.
-/
def bindSGD {α : Type}
    [_root_.Context α] [DecidableEq Shape]
    {stateShapes inputShapes : List Shape}
    (m : Objective α stateShapes inputShapes)
    (lr : α) := do
  let opt := _root_.Runtime.Autograd.TorchLean.Optim.sgd (α := α) lr
    (paramShapes := stateShapes)
  bindOptimizer (α := α) m opt

/--
Create a one-step update function for any typed module input pack from the public optimizer config
used by the trainer API.

Generic bridge for custom training loops: richer examples can keep their own control flow while
still choosing a public `optim.*` config through the same API as `Trainer.RunConfig`.
-/
def makeOptimizerStep {α : Type}
    [_root_.Context α] [Runtime.FromFloat α] [DecidableEq Shape]
    {stateShapes inputShapes : List Shape}
    (m : Objective α stateShapes inputShapes)
    (cfg : TorchLean.Trainer.Manual.OptimizerConfig) :
    IO (TensorPack α inputShapes → IO Unit) := do
  match cfg with
  | .sgd lr momentum =>
      if momentum == 0.0 then
        let bound ← bindSGD m (Runtime.ofFloat lr)
        pure bound.step
      else
        let opt := _root_.Runtime.Autograd.TorchLean.Optim.momentumSGD
          (α := α)
          (Runtime.ofFloat lr)
          (Runtime.ofFloat momentum)
          (paramShapes := stateShapes)
        let bound ← bindOptimizer (α := α) m opt
        pure bound.step
  | .adagrad lr epsilon =>
      let opt := _root_.Runtime.Autograd.TorchLean.Optim.adagrad
        (α := α)
        (Runtime.ofFloat lr)
        (Runtime.ofFloat epsilon)
        (paramShapes := stateShapes)
      let bound ← bindOptimizer (α := α) m opt
      pure bound.step
  | .rmsprop lr decay epsilon =>
      let opt := _root_.Runtime.Autograd.TorchLean.Optim.rmsprop
        (α := α)
        (Runtime.ofFloat lr)
        (Runtime.ofFloat decay)
        (Runtime.ofFloat epsilon)
        (paramShapes := stateShapes)
      let bound ← bindOptimizer (α := α) m opt
      pure bound.step
  | .adam lr beta1 beta2 epsilon =>
      let bound ← bindAdam m
        (Runtime.ofFloat lr)
        (Runtime.ofFloat beta1)
        (Runtime.ofFloat beta2)
        (Runtime.ofFloat epsilon)
      pure bound.step
  | .adadelta lr rho epsilon =>
      let opt := _root_.Runtime.Autograd.TorchLean.Optim.adadelta
        (α := α)
        (Runtime.ofFloat lr)
        (Runtime.ofFloat rho)
        (Runtime.ofFloat epsilon)
        (paramShapes := stateShapes)
      let bound ← bindOptimizer (α := α) m opt
      pure bound.step
  | .adamw lr weightDecay beta1 beta2 epsilon =>
      let bound ← bindAdamW m
        (Runtime.ofFloat lr)
        (Runtime.ofFloat weightDecay)
        (Runtime.ofFloat beta1)
        (Runtime.ofFloat beta2)
        (Runtime.ofFloat epsilon)
      pure bound.step

/--
Create a sample-step function from the public optimizer config used by the trainer API.

Bridge for custom training loops: richer examples can keep their own control flow while still
choosing a public `optim.*` config through the same API as `Trainer.RunConfig`.
-/
def makeSupervisedStep {α : Type} {σ τ : Shape}
    [_root_.Context α] [Runtime.FromFloat α] [DecidableEq Shape]
    {stateShapes : List Shape}
    (m : Objective α stateShapes [σ, τ])
    (cfg : TorchLean.Trainer.Manual.OptimizerConfig) : IO (Sample.Supervised α σ τ → IO Unit) := do
  makeOptimizerStep m cfg

end Module


end TorchLean
