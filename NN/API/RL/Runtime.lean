/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.RL.Definitions
public import NN.API.Neural.Execution
public import NN.API.Trainer.Manual.Core
public import NN.Runtime.RL.Boundary
public import NN.Runtime.RL.Gymnasium
public import NN.Runtime.RL.Numerics
public import NN.Runtime.RL.PPO
public import NN.Runtime.RL.Session

/-!
# RL Runtime

Rollout boundary checks, Gymnasium sessions, Float32 and interval numerics, and PPO actor-critic
wiring exposed under `TorchLean.rl`.
-/

@[expose] public section

namespace TorchLean
namespace rl

namespace boundary
export _root_.Runtime.RL.Boundary
  (isFiniteFloat tensorAll tensorFinite tensorInClosedInterval
   Contract Transition
   checkAction
   checkObservation checkReward checkDoneFlags
   checkTransitionFin checkTransition
   parseTransitionJson loadRollout)
export _root_.Runtime.RL.Boundary.Transition (done)

/-!
## Casting to Other Scalar Backends

The trust-boundary checker validates rollout JSON in host `Float`, because that is the interchange
format. The functions below cast accepted rollouts into the runtime scalar chosen for the proof or
training path.
-/

/-- Cast a `Float` observation tensor into a runtime scalar backend `α`. -/
def castObs {α : Type} [Runtime.FromFloat α] {obsShape : _root_.Spec.Shape}
    (t : _root_.Spec.Tensor Float obsShape) : _root_.Spec.Tensor α obsShape :=
  _root_.Spec.mapTensor (Runtime.ofFloat (α := α)) t

/-- Cast a validated `Float` transition into a runtime scalar backend `α`. -/
def castTransition {α : Type} [Runtime.FromFloat α]
    {obsShape : _root_.Spec.Shape} {nActions : Nat}
    (tr : _root_.Runtime.RL.Boundary.Transition obsShape nActions) :
    _root_.Spec.RL.ObservedTransition (_root_.Spec.Tensor α obsShape) (Fin nActions) α :=
  { observation := castObs (α := α) tr.observation
    action := tr.action
    reward := Runtime.ofFloat (α := α) tr.reward
    nextObservation := castObs (α := α) tr.nextObservation
    terminated := tr.terminated
    truncated := tr.truncated }

/-- Cast a whole rollout into a runtime scalar backend `α`. -/
def castRollout {α : Type} [Runtime.FromFloat α]
    {obsShape : _root_.Spec.Shape} {nActions : Nat}
    (xs : Array (_root_.Runtime.RL.Boundary.Transition obsShape nActions)) :
    Array (_root_.Spec.RL.ObservedTransition (_root_.Spec.Tensor α obsShape) (Fin nActions) α) :=
  xs.map (castTransition (α := α) (obsShape := obsShape) (nActions := nActions))

/-- Load a rollout JSON file, validate it with the boundary contract, then cast to scalar `α`. -/
def loadRolloutCast {α : Type} [Runtime.FromFloat α]
    {obsShape : _root_.Spec.Shape} {nActions : Nat}
    (path : String)
    (c : _root_.Runtime.RL.Boundary.Contract obsShape nActions) :
    IO (Array (_root_.Spec.RL.ObservedTransition (_root_.Spec.Tensor α obsShape) (Fin nActions) α)) := do
  let xs ← _root_.Runtime.RL.Boundary.loadRollout (obsShape := obsShape) (nActions := nActions) path c
  pure (castRollout (α := α) (obsShape := obsShape) (nActions := nActions) xs)

end boundary

namespace numerics
namespace float32
export _root_.Runtime.RL.Numerics.Float32
  (Float32Exec Interval32
   ofFloatIEEE32ExecChecked castTensorIEEE32ExecChecked castTransitionIEEE32ExecChecked
   discountedBackupIEEE32ExecChecked discountedReturnsVecFromIEEE32ExecChecked
   tdResidualIEEE32ExecChecked
   generalizedAdvantageEstimationVecIEEE32ExecChecked
   normalizeZScoreIEEE32ExecChecked
   importanceRatioIEEE32ExecChecked
   ppoClippedObjectiveFromRatioIEEE32ExecChecked
   discountedBackupInterval32 tdResidualInterval32
   ppoClippedObjectiveFromRatioInterval32
   discountedReturnsIntervals32 generalizedAdvantageEstimationIntervals32
   returnsWithinIntervals32)
end float32
end numerics

namespace session
export _root_.Runtime.RL.Session (CheckedSession)
export _root_.Runtime.RL.Session.CheckedSession (gymnasium ofEnv)
end session

namespace gym
export _root_.Runtime.RL.Gymnasium (Client Session)

namespace client
-- Only export the stable high-level entry points. The JSON request/response protocol and raw-step
-- protocol remain behind `NN.Runtime.RL.Gymnasium`.
export _root_.Runtime.RL.Gymnasium.Client (spawn reset close withClient)
end client

namespace session
export _root_.Runtime.RL.Gymnasium.Session (start reset stepChecked close withSession)
end session

end gym

namespace ppo
export _root_.Runtime.RL.PPO
  (StateBatchShape LogitsBatchShape ScalarBatchShape ValueBatchShape
   Step Rollout
   collectRolloutSessionWith collectRolloutCheckedSessionWith collectRolloutWith)
export _root_.Runtime.RL.PPO.Rollout (toActorCriticSample)

/-- Instantiate the standard PPO actor-critic runtime module. -/
def instantiateActorCritic
    {stateShape : _root_.Spec.Shape} {batch nActions : Nat} {α : Type}
    [Fact (0 < batch)] [Fact (0 < nActions)]
    [_root_.Context α] [DecidableEq _root_.Spec.Shape] [_root_.TorchLean.Runtime.FromFloat α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (opts : _root_.Runtime.Autograd.Torch.Options)
    (actor : _root_.Runtime.Autograd.TorchLean.NN.Seq stateShape (.dim batch (.dim nActions .scalar)))
    (critic : _root_.Runtime.Autograd.TorchLean.NN.Seq stateShape (.dim batch (.dim 1 .scalar)))
    (cast : Float → α := _root_.TorchLean.Runtime.ofFloat) :
    IO (_root_.Runtime.Autograd.TorchLean.Module.ScalarModule α
      (_root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes actor ++ _root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes critic)
      [stateShape, (.dim batch (.dim nActions .scalar)), (.dim batch .scalar), (.dim batch .scalar),
        (.dim batch (.dim 1 .scalar))]) :=
  _root_.TorchLean.Module.instantiateConfigured (α := α)
    (_root_.Runtime.RL.PolicyGradient.Autograd.ppoActorCriticScalarModuleDef
      (batch := batch) (nActions := nActions) actor critic)
    cast opts

/-- Create a PPO actor-critic update function from the public optimizer config. -/
def optimizerInputs {α : Type}
    [_root_.Context α] [_root_.TorchLean.Runtime.FromFloat α]
    {paramShapes inputShapes : List _root_.Spec.Shape}
    (m : _root_.Runtime.Autograd.TorchLean.Module.ScalarModule α paramShapes inputShapes)
    (cfg : _root_.TorchLean.Trainer.Manual.OptimizerConfig) :
    IO (_root_.Runtime.Autograd.Torch.TList α inputShapes → IO Unit) := do
  match cfg with
  | .sgd lr momentum =>
      if momentum == 0.0 then
        let opt := _root_.Runtime.Autograd.TorchLean.Optim.sgd
          (α := α) (_root_.TorchLean.Runtime.ofFloat lr) (paramShapes := paramShapes)
        let optH ← _root_.Runtime.Autograd.TorchLean.Module.optimizerHandle (α := α) m opt
        pure optH.step
      else
        let opt := _root_.Runtime.Autograd.TorchLean.Optim.momentumSGD
          (α := α)
          (_root_.TorchLean.Runtime.ofFloat lr)
          (_root_.TorchLean.Runtime.ofFloat momentum)
          (paramShapes := paramShapes)
        let optH ← _root_.Runtime.Autograd.TorchLean.Module.optimizerHandle (α := α) m opt
        pure optH.step
  | .adagrad lr epsilon =>
      let opt := _root_.Runtime.Autograd.TorchLean.Optim.adagrad
        (α := α)
        (_root_.TorchLean.Runtime.ofFloat lr)
        (_root_.TorchLean.Runtime.ofFloat epsilon)
        (paramShapes := paramShapes)
      let optH ← _root_.Runtime.Autograd.TorchLean.Module.optimizerHandle (α := α) m opt
      pure optH.step
  | .rmsprop lr decay epsilon =>
      let opt := _root_.Runtime.Autograd.TorchLean.Optim.rmsprop
        (α := α)
        (_root_.TorchLean.Runtime.ofFloat lr)
        (_root_.TorchLean.Runtime.ofFloat decay)
        (_root_.TorchLean.Runtime.ofFloat epsilon)
        (paramShapes := paramShapes)
      let optH ← _root_.Runtime.Autograd.TorchLean.Module.optimizerHandle (α := α) m opt
      pure optH.step
  | .adam lr beta1 beta2 epsilon =>
      let opt := _root_.Runtime.Autograd.TorchLean.Optim.adam
        (α := α)
        (_root_.TorchLean.Runtime.ofFloat lr)
        (_root_.TorchLean.Runtime.ofFloat beta1)
        (_root_.TorchLean.Runtime.ofFloat beta2)
        (_root_.TorchLean.Runtime.ofFloat epsilon)
        (paramShapes := paramShapes)
      let optH ← _root_.Runtime.Autograd.TorchLean.Module.optimizerHandle (α := α) m opt
      pure optH.step
  | .adamw lr weightDecay beta1 beta2 epsilon =>
      let opt := _root_.Runtime.Autograd.TorchLean.Optim.adamw
        (α := α)
        (_root_.TorchLean.Runtime.ofFloat lr)
        (_root_.TorchLean.Runtime.ofFloat weightDecay)
        (_root_.TorchLean.Runtime.ofFloat beta1)
        (_root_.TorchLean.Runtime.ofFloat beta2)
        (_root_.TorchLean.Runtime.ofFloat epsilon)
        (paramShapes := paramShapes)
      let optH ← _root_.Runtime.Autograd.TorchLean.Module.optimizerHandle (α := α) m opt
      pure optH.step
  | .adadelta lr rho epsilon =>
      let opt := _root_.Runtime.Autograd.TorchLean.Optim.adadelta
        (α := α)
        (_root_.TorchLean.Runtime.ofFloat lr)
        (_root_.TorchLean.Runtime.ofFloat rho)
        (_root_.TorchLean.Runtime.ofFloat epsilon)
        (paramShapes := paramShapes)
      let optH ← _root_.Runtime.Autograd.TorchLean.Module.optimizerHandle (α := α) m opt
      pure optH.step

/-- Read the concatenated actor-critic parameter pack from a PPO runtime module. -/
def params {α : Type} [_root_.Context α] {paramShapes inputShapes : List _root_.Spec.Shape}
    (m : _root_.Runtime.Autograd.TorchLean.Module.ScalarModule α paramShapes inputShapes) :
    IO (_root_.Runtime.Autograd.Torch.TList α paramShapes) :=
  _root_.Runtime.Autograd.TorchLean.Module.ScalarModule.params m

/-- Split a concatenated actor-critic parameter pack into `(actorParams, criticParams)`. -/
def splitActorCriticParams
    {σ₁ τ₁ σ₂ τ₂ : _root_.Spec.Shape}
    (actor : _root_.Runtime.Autograd.TorchLean.NN.Seq σ₁ τ₁)
    (critic : _root_.Runtime.Autograd.TorchLean.NN.Seq σ₂ τ₂)
    {α : Type}
    (ps :
      _root_.Runtime.Autograd.Torch.TList α
        (_root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes actor ++
          _root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes critic)) :
    _root_.Runtime.Autograd.Torch.TList α (_root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes actor) ×
      _root_.Runtime.Autograd.Torch.TList α (_root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes critic) :=
  _root_.Proofs.Autograd.Algebra.TList.splitAppend (α := α)
    (ss₁ := _root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes actor)
    (ss₂ := _root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes critic)
    ps

/--
Build a single-observation actor policy from the parameter pack of a rollout-shaped actor-critic
module.

The compiled actor records its parameter layout, while `sameActorParams` states that the rollout
actor uses that layout as well.
-/
def actorPolicyFromParams
    {obsShape logitsShape rolloutStateShape rolloutLogitsShape rolloutValueShape : _root_.Spec.Shape}
    {actorParamShapes : List _root_.Spec.Shape}
    {α : Type} [_root_.Context α]
    (actorCompiled : nn.Compiled actorParamShapes obsShape logitsShape α)
    (actorRollout : nn.Sequential rolloutStateShape rolloutLogitsShape)
    (criticRollout : nn.Sequential rolloutStateShape rolloutValueShape)
    (psAll : _root_.Runtime.Autograd.Torch.TList α
      (nn.paramShapes actorRollout ++ nn.paramShapes criticRollout))
    (sameActorParams : nn.paramShapes actorRollout = actorParamShapes := by rfl) :
    Tensor.T α obsShape → Tensor.T α logitsShape :=
  let (psActor, _psCritic) := splitActorCriticParams actorRollout criticRollout psAll
  let psActorObs : TensorPack α actorParamShapes :=
    Eq.mp (by rw [← sameActorParams]) psActor
  fun obs => actorCompiled.forward psActorObs obs

/--
Build a single-observation critic function from the parameter pack of a rollout-shaped actor-critic
module.

The result is scalar because the compiled critic has a checked one-element output shape.
-/
def criticValueFromParams
    {obsShape rolloutStateShape rolloutLogitsShape rolloutValueShape : _root_.Spec.Shape}
    {criticParamShapes : List _root_.Spec.Shape}
    {α : Type} [_root_.Context α]
    (criticCompiled : nn.Compiled criticParamShapes obsShape (.dim 1 .scalar) α)
    (actorRollout : nn.Sequential rolloutStateShape rolloutLogitsShape)
    (criticRollout : nn.Sequential rolloutStateShape rolloutValueShape)
    (psAll : _root_.Runtime.Autograd.Torch.TList α
      (nn.paramShapes actorRollout ++ nn.paramShapes criticRollout))
    (sameCriticParams : nn.paramShapes criticRollout = criticParamShapes := by rfl) :
    Tensor.T α obsShape → α :=
  let (_psActor, psCritic) := splitActorCriticParams actorRollout criticRollout psAll
  let psCriticObs : TensorPack α criticParamShapes :=
    Eq.mp (by rw [← sameCriticParams]) psCritic
  fun obs =>
    _root_.Spec.Tensor.vecGet (criticCompiled.forward psCriticObs obs) ⟨0, by decide⟩

end ppo

end rl
end TorchLean
