/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.NN
public import NN.API.Public.Facade.Runtime.RL

/-!
# TorchLean Module Runtime API

Executable module operations for manual runtime and example code.
-/

@[expose] public section

namespace TorchLean

namespace Module

export NN.API.TorchLean.Module
  (instantiateConfigured instantiateFloat forwardWithParams forward lossAndBackward backward
   stepWithLoss step initOptim stepWith stepWithOptimizerAndLoss params setParams trainSGD trainWith
   meanLoss run)

@[inherit_doc NN.API.TorchLean.Module.ScalarModule]
abbrev ScalarModule := NN.API.TorchLean.Module.ScalarModule

@[inherit_doc NN.API.TorchLean.Module.ScalarModuleDef]
abbrev ScalarModuleDef := NN.API.TorchLean.Module.ScalarModuleDef

/--
Instantiate an executable runtime module from a public `ScalarModuleDef`.

Generic public entrypoint for custom runtime tasks outside the standard supervised module
constructors such as `Module.instantiateMse` or `Module.instantiateCrossEntropyOneHot`.
-/
def instantiate
    {α : Type} [Runtime.SemanticScalar α] [DecidableEq Shape] [Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    {paramShapes inputShapes : List Shape}
    (opts : Options)
    (defn : ScalarModuleDef paramShapes inputShapes)
    (cast : Float → α := Runtime.ofFloat) :
    IO (ScalarModule α paramShapes inputShapes) :=
  instantiateConfigured (α := α) defn cast opts

/--
Run one inference step through a supervised runtime module.

Public sibling of the direct runtime pattern `model.predict opts m.trainer.params x`.
-/
def predict {σ τ : Shape} {α : Type}
    [Runtime.TensorScalar α] [DecidableEq Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (opts : Options)
    (model : TorchLean.nn.Sequential σ τ)
    (m : ScalarModule α (nn.paramShapes model) [σ, τ])
    (x : Tensor.T α σ) : IO (Tensor.T α τ) :=
  nn.predict (α := α) model opts m.trainer.params x

/--
Instantiate a supervised MSE module directly from a sequential model.
-/
def instantiateMse {σ τ : Shape} {α : Type}
    [Runtime.SemanticScalar α] [DecidableEq Shape] [Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (opts : Options)
    (model : TorchLean.nn.Sequential σ τ)
    (reduction : NN.API.TorchLean.Loss.Reduction := .mean)
    (cast : Float → α := Runtime.ofFloat) :
    IO (ScalarModule α (nn.paramShapes model) [σ, τ]) :=
  instantiate (α := α) opts
    (nn.mseScalarModuleDef model (reduction := reduction)) cast

/--
Instantiate a supervised one-hot cross-entropy module directly from a sequential model.
-/
def instantiateCrossEntropyOneHot {σ τ : Shape} {α : Type}
    [Runtime.SemanticScalar α] [DecidableEq Shape] [Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (opts : Options)
    (model : TorchLean.nn.Sequential σ τ)
    (reduction : NN.API.TorchLean.Loss.Reduction := .mean)
    (cast : Float → α := Runtime.ofFloat) :
    IO (ScalarModule α (nn.paramShapes model) [σ, τ]) :=
  instantiate (α := α) opts
    (nn.crossEntropyOneHotScalarModuleDef model (reduction := reduction))
    cast

/--
Instantiate a custom supervised runtime module directly from a sequential model.

Use this when a public example keeps the ordinary `TorchLean.nn.Sequential` model API but needs a custom
loss/module definition instead of the standard MSE or cross-entropy module constructors.
-/
def instantiateModuleDefModel
    {σ τ : Shape} {α : Type}
    [Runtime.SemanticScalar α] [DecidableEq Shape] [Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (opts : Options)
    (model : TorchLean.nn.Sequential σ τ)
    (moduleDefOf : (model : TorchLean.nn.Sequential σ τ) →
      ScalarModuleDef (nn.paramShapes model) [σ, τ])
    (cast : Float → α := Runtime.ofFloat) :
    IO (ScalarModule α (nn.paramShapes model) [σ, τ]) :=
  instantiate (α := α) opts (moduleDefOf model) cast

/-- Float specialization of `instantiateModuleDefModel` with storage-first initialization. -/
def instantiateModuleDefModelFloat
    {σ τ : Shape}
    (opts : Options)
    (model : TorchLean.nn.Sequential σ τ)
    (moduleDefOf : (model : TorchLean.nn.Sequential σ τ) →
      ScalarModuleDef (nn.paramShapes model) [σ, τ]) :
    IO (ScalarModule Float (nn.paramShapes model) [σ, τ]) :=
  NN.API.TorchLean.Module.instantiateFloat (moduleDefOf model) opts

/--
Instantiate the standard PPO actor-critic supervised runtime module from rollout-shaped actor and
critic networks.
-/
def instantiatePpoActorCritic
    {stateShape : Shape} {batch nActions : Nat} {α : Type}
    [Fact (0 < batch)] [Fact (0 < nActions)]
    [Runtime.SemanticScalar α] [DecidableEq Shape] [Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (opts : Options)
    (actor : TorchLean.nn.Sequential stateShape (.dim batch (.dim nActions .scalar)))
    (critic : TorchLean.nn.Sequential stateShape (.dim batch (.dim 1 .scalar)))
    (cast : Float → α := Runtime.ofFloat) :
    IO (ScalarModule α
      (nn.paramShapes actor ++ nn.paramShapes critic)
      [stateShape, (.dim batch (.dim nActions .scalar)), (.dim batch .scalar), (.dim batch .scalar),
        (.dim batch (.dim 1 .scalar))]) :=
  instantiate (α := α) opts
    (rl.policy.autograd.ppoActorCriticScalarModuleDef
      (batch := batch) (nActions := nActions) actor critic)
    cast

/--
Build a sequential model, instantiate a one-hot cross-entropy runtime module for it, and continue
with both values.

This packages the common public example pattern
`nn.withModel mkModel fun model => let m ← Module.instantiateCrossEntropyOneHot ...`.
-/
def withCrossEntropyOneHotModel
    {σ τ : Shape} {α : Type} {β : Type}
    [Runtime.SemanticScalar α] [DecidableEq Shape] [Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (mkModel : TorchLean.nn.M (TorchLean.nn.Sequential σ τ))
    (opts : Options)
    (reduction : NN.API.TorchLean.Loss.Reduction := .mean)
    (cast : Float → α := Runtime.ofFloat)
    (k : (model : TorchLean.nn.Sequential σ τ) →
      ScalarModule α (nn.paramShapes model) [σ, τ] → IO β) : IO β :=
  nn.withModel mkModel fun model => do
    let m ← instantiateCrossEntropyOneHot
      (α := α) opts model (reduction := reduction) (cast := cast)
    k model m

/--
Build a sequential model, instantiate an MSE runtime module for it, and continue with both values.
-/
def withMseModel
    {σ τ : Shape} {α : Type} {β : Type}
    [Runtime.SemanticScalar α] [DecidableEq Shape] [Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (mkModel : TorchLean.nn.M (TorchLean.nn.Sequential σ τ))
    (opts : Options)
    (reduction : NN.API.TorchLean.Loss.Reduction := .mean)
    (cast : Float → α := Runtime.ofFloat)
    (k : (model : TorchLean.nn.Sequential σ τ) →
      ScalarModule α (nn.paramShapes model) [σ, τ] → IO β) : IO β :=
  nn.withModel mkModel fun model => do
    let m ← instantiateMse
      (α := α) opts model (reduction := reduction) (cast := cast)
    k model m

/--
Build a sequential model, instantiate a custom supervised runtime module for it, and continue with
both values.

This packages the common public example pattern
`nn.withModel mkModel fun model => let m ← Module.instantiate ... (moduleDefOf model)`.
-/
def withModuleDefModel
    {σ τ : Shape} {α : Type} {β : Type}
    [Runtime.SemanticScalar α] [DecidableEq Shape] [Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (mkModel : TorchLean.nn.M (TorchLean.nn.Sequential σ τ))
    (opts : Options)
    (moduleDefOf : (model : TorchLean.nn.Sequential σ τ) →
      ScalarModuleDef (nn.paramShapes model) [σ, τ])
    (cast : Float → α := Runtime.ofFloat)
    (k : (model : TorchLean.nn.Sequential σ τ) →
      ScalarModule α (nn.paramShapes model) [σ, τ] → IO β) : IO β :=
  nn.withModel mkModel fun model => do
    let m ← instantiateModuleDefModel
      (α := α) opts model moduleDefOf (cast := cast)
    k model m

/--
Build a sequential model, instantiate a runtime module for a custom scalar loss program, and
continue with both values.

Custom-loss sibling of `withMseModel` / `withCrossEntropyOneHotModel`. Use it when the model is
ordinary `TorchLean.nn.Sequential`, but the loss needs task-specific logic beyond the standard MSE or
cross-entropy module constructors.
-/
def withScalarLossModel
    {σ τ : Shape} {α : Type} {β : Type}
    [Runtime.SemanticScalar α] [DecidableEq Shape] [Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (mkModel : TorchLean.nn.M (TorchLean.nn.Sequential σ τ))
    (opts : Options)
    (loss : ∀ {α : Type}, [Runtime.TensorScalar α] → [DecidableEq Shape] →
      _root_.Runtime.Autograd.TorchLean.Program α [τ, τ] Shape.scalar)
    (cast : Float → α := Runtime.ofFloat)
    (k : (model : TorchLean.nn.Sequential σ τ) →
      ScalarModule α (nn.paramShapes model) [σ, τ] → IO β) : IO β :=
  withModuleDefModel
    (α := α) (mkModel := mkModel) (opts := opts)
    (moduleDefOf := fun model => nn.scalarModuleDef model (loss := loss))
    (cast := cast) k

/-- Float specialization of `withScalarLossModel` with storage-first parameter initialization. -/
def withScalarLossModelFloat
    {σ τ : Shape} {β : Type}
    (mkModel : TorchLean.nn.M (TorchLean.nn.Sequential σ τ))
    (opts : Options)
    (loss : ∀ {α : Type}, [Runtime.TensorScalar α] → [DecidableEq Shape] →
      _root_.Runtime.Autograd.TorchLean.Program α [τ, τ] Shape.scalar)
    (k : (model : TorchLean.nn.Sequential σ τ) →
      ScalarModule Float (nn.paramShapes model) [σ, τ] → IO β) : IO β :=
  nn.withModel mkModel fun model => do
    let m ← instantiateModuleDefModelFloat opts model
      (fun current => nn.scalarModuleDef current (loss := loss))
    k model m

/--
Evaluate one supervised sample through a runtime module and return the scalar loss value.

This packages the common public example pattern `Module.forward ...; Tensor.toScalar`.
-/
def lossScalar {σ τ : Shape} {α : Type}
    [Runtime.TensorScalar α] [DecidableEq Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (model : TorchLean.nn.Sequential σ τ)
    (m : ScalarModule α (nn.paramShapes model) [σ, τ])
    (sample : SupervisedSample α σ τ) : IO α := do
  let loss ← forward (α := α) m sample
  pure (Tensor.toScalar loss)

/--
Create an Adam optimizer handle bound to a concrete runtime module.

This packages the common public example pattern `optim.runtimeAdam ...; optim.handle m opt`.
-/
def adamHandle {α : Type}
    [Runtime.TensorScalar α] [DecidableEq Shape]
    {paramShapes inputShapes : List Shape}
    (m : ScalarModule α paramShapes inputShapes)
    (lr beta1 beta2 epsilon : α) := do
  let opt := NN.API.TorchLean.Optim.adam (α := α) lr beta1 beta2 epsilon
    (paramShapes := paramShapes)
  NN.API.TorchLean.Optim.handle (α := α) m opt

/--
Create an AdamW optimizer handle bound to a concrete runtime module.
-/
def adamWHandle {α : Type}
    [Runtime.TensorScalar α] [DecidableEq Shape]
    {paramShapes inputShapes : List Shape}
    (m : ScalarModule α paramShapes inputShapes)
    (lr weightDecay beta1 beta2 epsilon : α) := do
  let opt := NN.API.TorchLean.Optim.adamw (α := α)
    (paramShapes := paramShapes)
    lr weightDecay beta1 beta2 epsilon
  NN.API.TorchLean.Optim.handle (α := α) m opt

/--
Create an SGD optimizer handle bound to a concrete runtime module.
-/
def sgdHandle {α : Type}
    [Runtime.TensorScalar α] [DecidableEq Shape]
    {paramShapes inputShapes : List Shape}
    (m : ScalarModule α paramShapes inputShapes)
    (lr : α) := do
  let opt := NN.API.TorchLean.Optim.sgd (α := α) lr (paramShapes := paramShapes)
  NN.API.TorchLean.Optim.handle (α := α) m opt

/--
Create a one-step update function for any typed module input pack from the public optimizer config
used by the trainer API.

Generic bridge for custom training loops: richer examples can keep their own control flow while
still choosing a public `optim.*` config through the same API as `Trainer.RunConfig`.
-/
def optimizerInputs {α : Type}
    [Runtime.TensorScalar α] [Runtime.Scalar α] [DecidableEq Shape]
    {paramShapes inputShapes : List Shape}
    (m : ScalarModule α paramShapes inputShapes)
    (cfg : NN.API.TorchLean.Trainer.Optimizer) :
    IO (TensorPack α inputShapes → IO Unit) := do
  match cfg with
  | .sgd lr momentum =>
      if momentum == 0.0 then
        let optH ← sgdHandle m (Runtime.ofFloat lr)
        pure optH.step
      else
        let opt := NN.API.TorchLean.Optim.momentumSGD
          (α := α)
          (Runtime.ofFloat lr)
          (Runtime.ofFloat momentum)
          (paramShapes := paramShapes)
        let optH ← NN.API.TorchLean.Optim.handle (α := α) m opt
        pure optH.step
  | .adagrad lr epsilon =>
      let opt := NN.API.TorchLean.Optim.adagrad
        (α := α)
        (Runtime.ofFloat lr)
        (Runtime.ofFloat epsilon)
        (paramShapes := paramShapes)
      let optH ← NN.API.TorchLean.Optim.handle (α := α) m opt
      pure optH.step
  | .rmsprop lr decay epsilon =>
      let opt := NN.API.TorchLean.Optim.rmsprop
        (α := α)
        (Runtime.ofFloat lr)
        (Runtime.ofFloat decay)
        (Runtime.ofFloat epsilon)
        (paramShapes := paramShapes)
      let optH ← NN.API.TorchLean.Optim.handle (α := α) m opt
      pure optH.step
  | .adam lr beta1 beta2 epsilon =>
      let optH ← adamHandle m
        (Runtime.ofFloat lr)
        (Runtime.ofFloat beta1)
        (Runtime.ofFloat beta2)
        (Runtime.ofFloat epsilon)
      pure optH.step
  | .adadelta lr rho epsilon =>
      let opt := NN.API.TorchLean.Optim.adadelta
        (α := α)
        (Runtime.ofFloat lr)
        (Runtime.ofFloat rho)
        (Runtime.ofFloat epsilon)
        (paramShapes := paramShapes)
      let optH ← NN.API.TorchLean.Optim.handle (α := α) m opt
      pure optH.step
  | .adamw lr weightDecay beta1 beta2 epsilon =>
      let optH ← adamWHandle m
        (Runtime.ofFloat lr)
        (Runtime.ofFloat weightDecay)
        (Runtime.ofFloat beta1)
        (Runtime.ofFloat beta2)
        (Runtime.ofFloat epsilon)
      pure optH.step

/--
Create a sample-step function from the public optimizer config used by the trainer API.

Bridge for custom training loops: richer examples can keep their own control flow while still
choosing a public `optim.*` config through the same API as `Trainer.RunConfig`.
-/
def optimizerStep {α : Type} {σ τ : Shape}
    [Runtime.TensorScalar α] [Runtime.Scalar α] [DecidableEq Shape]
    {paramShapes : List Shape}
    (m : ScalarModule α paramShapes [σ, τ])
    (cfg : NN.API.TorchLean.Trainer.Optimizer) : IO (SupervisedSample α σ τ → IO Unit) := do
  optimizerInputs m cfg

end Module


end TorchLean
