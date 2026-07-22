/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.NN

/-!
# TorchLean RL Runtime API

RL environment, algorithm, replay, PPO, Gymnasium, and training names.
-/

@[expose] public section

namespace TorchLean

namespace rl

/-!
Reinforcement-learning APIs used by the executable examples.

The public namespace mirrors the shape of the checked RL library:

- `rl.core`, `rl.mdp`, and friends expose the mathematical objects,
- `rl.replay`, `rl.dqn`, and `rl.policy` expose runtime update formulas,
- `rl.boundary`, `rl.gym`, and `rl.eval` expose the executable trust-boundary APIs,
- `rl.cli` keeps runnable examples on one shared command-line convention.
-/

namespace env
export NN.API.rl.env
  (StepResult ObservedTransition Env SafeEnv
   reset stepGym evolve evolveFrom
   states statesFrom rollout rolloutFrom)
export _root_.Spec.RL.StepResult (done)
export _root_.Spec.RL.SafeEnv (actionPathOk)
end env

namespace core
export NN.API.rl.core
  (AdvantageStep
   Transition IndexedTransition
   continueMask discountedBackup tdTarget tdResidual
   discountedReturns discountedReturnsFrom discountedReturnsDone
   generalizedAdvantageEstimation returnsFromAdvantages
   discountedReturnsVecFrom discountedReturnsVec discountedReturnsVecDone
   generalizedAdvantageEstimationVec returnsFromAdvantagesVec
   squaredError huberLoss)
end core

namespace mdp
export NN.API.rl.mdp
  (ValueFunction Policy FiniteMDP
   valueAt stateActionValue actionValues
   bellmanPolicy bellmanOptimality)
export _root_.Spec.RL.FiniteMDP (toEnv)
end mdp

namespace markov
export NN.API.rl.markov
  (ValueFunction Policy MDP Valid
   transitionMeasure
   expectedNextValue actionValue
   bellmanPolicy bellmanOptimality)
end markov

namespace finiteStochastic
export NN.API.rl.finiteStochastic
  (MDP Valid
   expectedNextValue actionValue actionValues
   bellmanPolicy bellmanOptimality)
end finiteStochastic

namespace bandits
export NN.API.rl.bandits
  (ValueState PreferenceState
   greedyAction? epsilonGreedyAction?
   sampleAverageStep totalPulls
   ucb1Bonus ucb1Scores ucb1Action?
   gradientPolicy gradientBanditStep)
export _root_.Runtime.RL.Bandits.ValueState (init)
export _root_.Runtime.RL.Bandits.PreferenceState (init)
end bandits

namespace tabular
export NN.API.rl.tabular
  (actionRow maxActionValue greedyAction? expectedActionValue
   td0Update
   sarsaTarget expectedSarsaTarget qLearningTarget doubleQTarget
   sarsaUpdate expectedSarsaUpdate qLearningUpdate
   doubleQUpdateLeft doubleQUpdateRight)
end tabular

namespace value
export NN.API.rl.value
  (chosenActionValue maxQValue
   dqnTarget doubleDqnTarget
   dqnResidual dqnMSELoss dqnHuberLoss doubleDqnResidual
   ddpgActorObjective ddpgCriticTarget td3Target
   sacTarget sacActorObjective)
end value

namespace replay
export NN.API.rl.replay (Transition Buffer)
export _root_.Runtime.RL.Replay.Buffer
  (empty size isEmpty isFull push pushMany getModulo? sampleContiguous sampleRandom)
end replay

namespace dqn
export NN.API.rl.dqn
  (meanArray
   transitionMSELoss transitionHuberLoss transitionDoubleHuberLoss
   minibatchMSELoss minibatchHuberLoss minibatchDoubleHuberLoss
   softUpdateScalar)
end dqn

namespace policy
export NN.API.rl.policy
  (actionPolicy actionProbability actionLogProbability entropyBonus
   reinforceLoss actorLoss criticLoss actorCriticLoss
   a2cLoss
   importanceRatio categoricalKL categoricalKLFromLogits
   trpoSurrogateFromRatio klPenalizedPolicyLoss sacCategoricalActorLoss
   ppoClippedObjectiveFromRatio ppoClippedObjective ppoLoss
   sampleCategorical sampleActionFromLogits)

namespace autograd
export NN.API.rl.policy.autograd
  (actionLogProbOneHotBatch
   entropyMean
   ppoClippedObjectiveBatch
   ppoLossBatch
   ppoActorCriticScalarModuleDef)
end autograd
end policy

namespace ppo
export NN.API.rl.ppo
  (StateBatchShape LogitsBatchShape ScalarBatchShape ValueBatchShape
   Step Rollout
   instantiateActorCritic
   optimizerInputs
   params
   splitActorCriticParams
   collectRolloutSessionWith collectRolloutCheckedSessionWith collectRolloutWith)
export _root_.Runtime.RL.PPO.Rollout
  (toActorCriticSample)

/--
Build a single-observation actor policy from the parameter pack of a rollout-shaped actor-critic
module.

The compiled actor carries the single-observation parameter ABI and compiled forward artifact, so the
public API only needs that one value instead of a separate raw artifact.
-/
def actorPolicyFromParams
    {obsShape logitsShape rolloutStateShape rolloutLogitsShape rolloutValueShape : Shape}
    {actorParamShapes : List Shape}
    {α : Type} [Runtime.TensorScalar α]
    (actorCompiled : nn.Compiled actorParamShapes obsShape logitsShape α)
    (actorRollout : TorchLean.nn.Sequential rolloutStateShape rolloutLogitsShape)
    (criticRollout : TorchLean.nn.Sequential rolloutStateShape rolloutValueShape)
    (psAll : _root_.Runtime.Autograd.Torch.TList α
      (nn.paramShapes actorRollout ++ nn.paramShapes criticRollout))
    (sameActorParams :
      nn.paramShapes actorRollout =
        actorParamShapes := by rfl) :
    Tensor.T α obsShape → Tensor.T α logitsShape :=
  let (psActor, _psCritic) := NN.API.rl.ppo.splitActorCriticParams actorRollout criticRollout psAll
  let psActorObs : nn.ParamTensors α actorParamShapes :=
    Eq.mp (by rw [← sameActorParams]) psActor
  fun obs => actorCompiled.forward psActorObs obs

/--
Build a single-observation critic-value function from the parameter pack of a rollout-shaped
actor-critic module.
-/
def criticValueFromParams
    {obsShape rolloutStateShape rolloutLogitsShape rolloutValueShape : Shape}
    {criticParamShapes : List Shape}
    {α : Type} [Runtime.TensorScalar α]
    (criticCompiled : nn.Compiled criticParamShapes obsShape (.dim 1 .scalar) α)
    (actorRollout : TorchLean.nn.Sequential rolloutStateShape rolloutLogitsShape)
    (criticRollout : TorchLean.nn.Sequential rolloutStateShape rolloutValueShape)
    (psAll : _root_.Runtime.Autograd.Torch.TList α
      (nn.paramShapes actorRollout ++ nn.paramShapes criticRollout))
    (sameCriticParams :
      nn.paramShapes criticRollout =
        criticParamShapes := by rfl) :
    Tensor.T α obsShape → α :=
  let (_psActor, psCritic) := NN.API.rl.ppo.splitActorCriticParams actorRollout criticRollout psAll
  let psCriticObs : nn.ParamTensors α criticParamShapes :=
    Eq.mp (by rw [← sameCriticParams]) psCritic
  fun obs =>
    _root_.Spec.Tensor.vecGet
      (criticCompiled.forward psCriticObs obs)
      ⟨0, by decide⟩
end ppo

namespace eval
export NN.API.rl.eval
  (greedyActionFromLogits episodeTotalReward episodeSessPath averageEpisodeTotalReward)
end eval

namespace train
export NN.API.rl.train
  (Series TrainLog Curve ConfigEntry Artifact RunInfo ExperimentLog LogDestination)
export _root_.Runtime.Training.Curve (push toTrainLog)
export _root_.Runtime.Training.TrainLog (writeJson readJson)
export _root_.Runtime.Training.ExperimentLog (init log logRow addArtifact toTrainLog)
export _root_.Runtime.Training.LogDestination (disabled json isEnabled path? writeTrainLog)
end train

namespace boundary
export NN.API.rl.boundary
  (isFiniteFloat tensorAll tensorFinite tensorInClosedInterval
   Contract Transition
   checkAction
   checkObservation checkReward checkDoneFlags
   checkTransitionFin checkTransition
   parseTransitionJson loadRollout
   castObs castTransition castRollout loadRolloutCast)
export _root_.Runtime.RL.Boundary.Transition (done)
end boundary

namespace gym
export NN.API.rl.gym (Client Session)

namespace client
export NN.API.rl.gym.client (spawn reset close withClient)
end client

namespace session
export NN.API.rl.gym.session (start reset stepChecked close withSession)
end session

end gym

namespace session
export NN.API.rl.session
  (CheckedSession gymnasium ofEnv)
end session

namespace cli

export NN.API.rl.cli
  (PpoFlags parsePpoFlags)

end cli

end rl


end TorchLean
