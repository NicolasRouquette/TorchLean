/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Runtime
public import NN.Spec.RL.Core
public import NN.Spec.RL.Environment
public import NN.Spec.RL.MDP
public import NN.Spec.RL.MarkovMDP
public import NN.Spec.RL.FiniteStochasticMDP
public import NN.Runtime.RL.Algorithms
public import NN.Runtime.RL.Core
public import NN.Runtime.RL.Eval
public import NN.Runtime.RL.PolicyGradient.Autograd
public import NN.Runtime.RL.Replay
public import NN.Runtime.Training.Log

/-!
# Reinforcement-Learning Definitions

Mathematical definitions and executable algorithms exposed under `TorchLean.rl`.

References (background and terminology):
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed.):
  http://incompleteideas.net/book/the-book-2nd.html
- Puterman, *Markov Decision Processes* (finite discounted MDPs):
  https://doi.org/10.1002/9780470316887
- Gymnasium API reference (reset/step, `terminated` vs `truncated`):
  https://gymnasium.farama.org/
-/

@[expose] public section

namespace TorchLean
namespace rl

namespace env
export _root_.Spec.RL
  (StepResult ObservedTransition Env SafeEnv
   reset stepGym evolve evolveFrom
   states statesFrom rollout rolloutFrom)
export _root_.Spec.RL.StepResult (done)
export _root_.Spec.RL.SafeEnv (actionPathOk)
end env

namespace core
export _root_.Spec.RL
  (AdvantageStep
   continueMask discountedBackup tdTarget tdResidual
   discountedReturns discountedReturnsFrom discountedReturnsDone
   generalizedAdvantageEstimation returnsFromAdvantages)
export _root_.Runtime.RL.Core
  (Transition IndexedTransition
   discountedReturnsVecFrom discountedReturnsVec discountedReturnsVecDone
   generalizedAdvantageEstimationVec returnsFromAdvantagesVec
   squaredError huberLoss)
end core

namespace mdp
export _root_.Spec.RL
  (ValueFunction Policy FiniteMDP
   valueAt stateActionValue actionValues
   bellmanPolicy bellmanOptimality)
export _root_.Spec.RL.FiniteMDP (toEnv)
end mdp

namespace markov
export _root_.Spec.RL.Markov
  (ValueFunction Policy MDP Valid
   transitionMeasure
   expectedNextValue actionValue
   bellmanPolicy bellmanOptimality)
end markov

namespace finiteStochastic
export _root_.Spec.RL.FiniteStochastic
  (MDP Valid
   expectedNextValue actionValue actionValues
   bellmanPolicy bellmanOptimality)
end finiteStochastic

namespace bandits
export _root_.Runtime.RL.Bandits
  (ValueState PreferenceState
   greedyAction? epsilonGreedyAction?
   sampleAverageStep totalPulls
   ucb1Bonus ucb1Scores ucb1Action?
   gradientPolicy gradientBanditStep)
export _root_.Runtime.RL.Bandits.ValueState (init)
export _root_.Runtime.RL.Bandits.PreferenceState (init)
end bandits

namespace tabular
export _root_.Runtime.RL.Tabular
  (actionRow maxActionValue greedyAction? expectedActionValue
   td0Update
   sarsaTarget expectedSarsaTarget qLearningTarget doubleQTarget
   sarsaUpdate expectedSarsaUpdate qLearningUpdate
   doubleQUpdateLeft doubleQUpdateRight)
end tabular

namespace value
export _root_.Runtime.RL.ValueLearning
  (chosenActionValue maxQValue
   dqnTarget doubleDqnTarget
   dqnResidual dqnMSELoss dqnHuberLoss doubleDqnResidual
   ddpgActorObjective ddpgCriticTarget td3Target
   sacTarget sacActorObjective)
end value

namespace replay
export _root_.Runtime.RL.Replay (Transition Buffer)
export _root_.Runtime.RL.Replay.Buffer
  (empty size isEmpty isFull push pushMany getModulo? sampleContiguous sampleRandom)
end replay

namespace dqn
export _root_.Runtime.RL.DQN
  (meanArray
   transitionMSELoss transitionHuberLoss transitionDoubleHuberLoss
   minibatchMSELoss minibatchHuberLoss minibatchDoubleHuberLoss
   softUpdateScalar)
end dqn

namespace policy
export _root_.Runtime.RL.PolicyGradient
  (actionPolicy actionProbability actionLogProbability entropyBonus
   reinforceLoss actorLoss criticLoss actorCriticLoss
   a2cLoss
   importanceRatio categoricalKL categoricalKLFromLogits
   trpoSurrogateFromRatio klPenalizedPolicyLoss sacCategoricalActorLoss
   ppoClippedObjectiveFromRatio ppoClippedObjective ppoLoss)
export _root_.Runtime.RL.PolicyGradient
  (sampleCategorical sampleActionFromLogits)

namespace autograd
/-!
Differentiable policy-gradient losses over TorchLean backend references.

The pure exports above are algebra over concrete spec tensors. These helpers are the training-time
counterpart: they build scalar losses from backend refs, so the same formulas can run through eager
or compiled autograd.
-/
export _root_.Runtime.RL.PolicyGradient.Autograd
  (actionLogProbOneHotBatch
   entropyMean
   ppoClippedObjectiveBatch
   ppoLossBatch
   ppoActorCriticScalarModuleDef)
end autograd
end policy

namespace eval
export _root_.Runtime.RL.Eval
  (greedyActionFromLogits episodeTotalReward episodeSessPath averageEpisodeTotalReward)
end eval

namespace train
/-!
## Training Logs (Widgets and Examples)

TorchLean does not aim to be a full “trainer framework”, but many executable examples want to:

- evaluate a scalar metric every `N` updates,
- append it to a curve, and
- write a small JSON file for widgets (`#train_log_file_view`).

This namespace re-exports the small, stable log types and JSON IO helpers.
-/
export _root_.Runtime.Training
  (Series TrainLog Curve ConfigEntry Artifact RunInfo ExperimentLog LogDestination)
export _root_.Runtime.Training.Curve (push toTrainLog)
export _root_.Runtime.Training.TrainLog (writeJson readJson)
export _root_.Runtime.Training.ExperimentLog (init log logRow addArtifact toTrainLog)
export _root_.Runtime.Training.LogDestination (disabled json isEnabled path? writeTrainLog)
end train

end rl
end TorchLean
