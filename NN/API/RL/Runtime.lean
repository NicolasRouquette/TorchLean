/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.RL.Core
public import NN.API.Module
public import NN.API.Neural.Execution
public import NN.API.Optim
public import NN.Tensor
public import NN.API.Sample
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
def castObs {α : Type} [Runtime.FromFloat α] {obsShape : Shape}
    (t : Tensor Float obsShape) : Tensor α obsShape :=
  _root_.Spec.Tensor.map (Runtime.ofFloat (α := α)) t

/-- Cast a validated `Float` transition into a runtime scalar backend `α`. -/
def castTransition {α : Type} [Runtime.FromFloat α]
    {obsShape : Shape} {nActions : Nat}
    (tr : Transition obsShape nActions) :
    _root_.Spec.RL.ObservedTransition (Tensor α obsShape) (Fin nActions) α :=
  { observation := castObs (α := α) tr.observation
    action := tr.action
    reward := Runtime.ofFloat (α := α) tr.reward
    nextObservation := castObs (α := α) tr.nextObservation
    terminated := tr.terminated
    truncated := tr.truncated }

/-- Cast a whole rollout into a runtime scalar backend `α`. -/
def castRollout {α : Type} [Runtime.FromFloat α]
    {obsShape : Shape} {nActions : Nat}
    (xs : Array (Transition obsShape nActions)) :
    Array (_root_.Spec.RL.ObservedTransition (Tensor α obsShape) (Fin nActions) α) :=
  xs.map (castTransition (α := α) (obsShape := obsShape) (nActions := nActions))

/-- Load a rollout JSON file, validate it with the boundary contract, then cast to scalar `α`. -/
def loadRolloutAs {α : Type} [Runtime.FromFloat α]
    {obsShape : Shape} {nActions : Nat}
    (path : String)
    (c : Contract obsShape nActions) :
    IO (Array (_root_.Spec.RL.ObservedTransition (Tensor α obsShape) (Fin nActions) α)) := do
  let xs ← loadRollout path c
  pure (castRollout (α := α) xs)

end boundary

namespace numerics
namespace float32
export _root_.Runtime.RL.Numerics.Float32
  (Float32Exec Interval32
   ofFloatChecked castTensorChecked castTransitionChecked
   discountedBackupChecked discountedReturnsChecked
   tdResidualChecked
   generalizedAdvantageEstimationChecked
   normalizeZScoreChecked
   importanceRatioChecked
   ppoClippedObjectiveFromRatioChecked
   discountedBackupInterval tdResidualInterval
   ppoClippedObjectiveFromRatioInterval
   discountedReturnsIntervals generalizedAdvantageEstimationIntervals
   returnsWithinIntervals)
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
    [NeZero batch] [NeZero nActions]
    [_root_.Context α] [DecidableEq _root_.Spec.Shape] [_root_.TorchLean.Runtime.FromFloat α]
    [Runtime.TensorTransfer α]
    (opts : _root_.Runtime.Autograd.Torch.Options)
    (actor : _root_.Runtime.Autograd.TorchLean.NN.Seq stateShape [batch, nActions])
    (critic : _root_.Runtime.Autograd.TorchLean.NN.Seq stateShape [batch, 1])
    (cast : Float → α := _root_.TorchLean.Runtime.ofFloat) :
    IO (_root_.Runtime.Autograd.TorchLean.Module.Objective α Unit
      (_root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes actor ++ _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes critic)
      [stateShape, [batch, nActions], [batch], [batch],
        [batch, 1]]) :=
  _root_.TorchLean.Module.instantiateAs (α := α)
    (_root_.Runtime.RL.PolicyGradient.Autograd.ppoActorCriticObjectiveDef
      (batch := batch) (nActions := nActions) actor critic)
    cast opts

/-- Create a PPO actor-critic update function from the public optimizer config. -/
def makeOptimizerStep {α : Type}
    [_root_.Context α] [_root_.TorchLean.Runtime.FromFloat α]
    {paramShapes inputShapes : List _root_.Spec.Shape}
    (m : _root_.Runtime.Autograd.TorchLean.Module.Objective α Unit paramShapes inputShapes)
    (cfg : _root_.TorchLean.optim.Optimizer) :
    IO (_root_.TorchLean.TensorPack α inputShapes → IO Unit) :=
  _root_.TorchLean.Module.makeOptimizerStep m cfg

/-- Read the concatenated actor-critic state from a PPO runtime module. -/
def state {α : Type} [_root_.Context α] {stateShapes inputShapes : List _root_.Spec.Shape}
    (m : _root_.Runtime.Autograd.TorchLean.Module.Objective α Unit stateShapes inputShapes) :
    IO (_root_.TorchLean.TensorPack α stateShapes) :=
  _root_.Runtime.Autograd.TorchLean.Module.Objective.state m

/-- Split concatenated actor-critic state into its actor and critic components. -/
def splitState
    {σ₁ τ₁ σ₂ τ₂ : _root_.Spec.Shape}
    (actor : _root_.Runtime.Autograd.TorchLean.NN.Seq σ₁ τ₁)
    (critic : _root_.Runtime.Autograd.TorchLean.NN.Seq σ₂ τ₂)
    {α : Type}
    (state :
      _root_.TorchLean.TensorPack α
        (_root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes actor ++
          _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes critic)) :
    _root_.TorchLean.TensorPack α (_root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes actor) ×
      _root_.TorchLean.TensorPack α (_root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes critic) :=
  TorchLean.TensorPack.split (α := α)
    (ss₁ := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes actor)
    (ss₂ := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes critic)
    state

/--
Build a single-observation actor policy from the state of a rollout-shaped actor-critic module.

The typed actor graph records its state layout, while `sameActorState` states that the rollout actor
uses that layout as well.
-/
def actorPolicy
    {obsShape logitsShape rolloutStateShape rolloutLogitsShape rolloutValueShape : _root_.Spec.Shape}
    {actorStateShapes : List _root_.Spec.Shape}
    {α : Type} [_root_.Context α]
    (actorGraph : nn.TypedGraphModel actorStateShapes obsShape logitsShape α)
    (actorRollout : nn.Sequential rolloutStateShape rolloutLogitsShape)
    (criticRollout : nn.Sequential rolloutStateShape rolloutValueShape)
    (state : _root_.TorchLean.TensorPack α
      (nn.stateShapes actorRollout ++ nn.stateShapes criticRollout))
    (sameActorState : nn.stateShapes actorRollout = actorStateShapes := by rfl) :
    Tensor α obsShape → Tensor α logitsShape :=
  let (actorState, _) := splitState actorRollout criticRollout state
  let actorState : _root_.TorchLean.TensorPack α actorStateShapes :=
    Eq.mp (by rw [← sameActorState]) actorState
  fun obs => actorGraph.forward actorState obs

/--
Build a single-observation critic function from the state of a rollout-shaped actor-critic module.

The result is scalar because the typed critic graph has a checked one-element output shape.
-/
def criticValue
    {obsShape rolloutStateShape rolloutLogitsShape rolloutValueShape : _root_.Spec.Shape}
    {criticStateShapes : List _root_.Spec.Shape}
    {α : Type} [_root_.Context α]
    (criticGraph : nn.TypedGraphModel criticStateShapes obsShape [1] α)
    (actorRollout : nn.Sequential rolloutStateShape rolloutLogitsShape)
    (criticRollout : nn.Sequential rolloutStateShape rolloutValueShape)
    (state : _root_.TorchLean.TensorPack α
      (nn.stateShapes actorRollout ++ nn.stateShapes criticRollout))
    (sameCriticState : nn.stateShapes criticRollout = criticStateShapes := by rfl) :
    Tensor α obsShape → α :=
  let (_, criticState) := splitState actorRollout criticRollout state
  let criticState : _root_.TorchLean.TensorPack α criticStateShapes :=
    Eq.mp (by rw [← sameCriticState]) criticState
  fun obs =>
    Tensor.item (Tensor.get (criticGraph.forward criticState obs) ⟨0, by decide⟩)

end ppo

end rl
end TorchLean
