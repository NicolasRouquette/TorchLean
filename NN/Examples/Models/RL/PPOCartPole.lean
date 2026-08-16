/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

End-to-end PPO example: train an actor-critic on Gymnasium `CartPole-v1` using TorchLean.
-/

module

public import NN.API
public import NN.Examples.ModelZoo
public import NN.Runtime.RL.Artifacts.DefaultPaths

/-!
# PPO on Gymnasium CartPole (Executable Example)

This example is small but complete:

- **Environment**: external Python Gymnasium (started as a subprocess).
- **Trust boundary**: every step is checked against a Lean side contract
  (`Runtime.RL.Boundary.Contract`) before being used for training data.
- **Algorithm**: PPO with GAE (all update math is Lean definitions; the PPO loss is a TorchLean
  autograd program).

More concretely:

- The policy is a categorical distribution over discrete actions parameterized by logits:
  $\pi_\theta(a\mid s)=\operatorname{softmax}(\mathrm{logits}_\theta(s))$.
- Advantages are computed using Generalized Advantage Estimation (GAE(λ)).
- PPO uses the clipped surrogate objective (plus a value-loss and optional entropy bonus, depending
  on the runtime configuration).

## CLI flags

- `--device cuda`: run the Torch backend on CUDA (requires building with `-K cuda=true`).
- `--seed <n>`: deterministic seed for TorchLean RNG streams (and evaluation seeding).
- `--updates <n>`: limit the number of PPO rollout/update cycles.
- `--log <path>`: write the widget log JSON to a custom path.

Run (from the repo root):

```bash
python3 -m pip install --user 'gymnasium>=1.0'
lake -R -K cuda=true exe torchlean ppo_cartpole --device cuda --updates 1 --eval-every 1 --eval-episodes 1 --eval-max-steps 8
```

Artifacts:
- The executable writes a widget-friendly training curve JSON to
  `data/rl/ppo_cartpole_trainlog.json` (override with `--log <path>`).
- Visualize it in the editor via `NN/Examples/RL/PPOCartPoleView.lean`.

## What this run does (and does not) guarantee

- The PPO/GAE math and the autograd loss program are Lean definitions, so they are suitable targets
  for formal reasoning.
- When Gymnasium is external, TorchLean cannot prove the environment satisfies Markov/measurability
  assumptions. The trust-boundary contract turns some common assumptions (finite tensors, reward
  bounds, done-flag semantics) into checked preconditions.
- The run favors readability, typed boundaries, and widget inspection over benchmark-specific PPO
  tuning.

References (primary):
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017): https://arxiv.org/abs/1707.06347
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2015):
  https://arxiv.org/abs/1506.02438
- Williams, "Simple statistical gradient-following algorithms for connectionist reinforcement learning"
  (REINFORCE, 1992): https://doi.org/10.1007/BF00992696
- Brockman et al., "OpenAI Gym" (2016): https://arxiv.org/abs/1606.01540
- Gymnasium API reference (reset/step, `terminated` vs `truncated`): https://gymnasium.farama.org/
- CartPole environment docs: https://gymnasium.farama.org/environments/classic_control/cart_pole/
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.RL.PPOCartPole

/-- Name of this executable target (used in CLI error messages and banners). -/
def exeName : String := "torchlean ppo_cartpole"

/-!
## Configuration

This stays with discrete-action CartPole so the native Lean executable remains easy to run and
inspect.
-/

/-- Gymnasium environment id passed to the Python subprocess (see Gymnasium docs for supported ids). -/
def envId : String := "CartPole-v1"

/-- Relative path to the Python Gymnasium bridge script (spawned as a subprocess). -/
def gymServerScript : String := "scripts/rl/gymnasium_server.py"

/-- Observation vector dimension for CartPole (`Gymnasium` reports 4 floats). -/
def stateDim : Nat := 4

/-- Number of discrete actions for CartPole (left/right). -/
def nActions : Nat := 2

/-- Width of the hidden layer in the actor and critic MLPs. -/
def hiddenDim : Nat := 32

/-- PPO rollout horizon (also the training batch size for this run). -/
def horizon : Nat := 64

/-- Discount factor used in returns / GAE. -/
def gamma : Float := 0.99

/-- GAE(λ) parameter controlling the bias/variance tradeoff of advantage estimates. -/
def lam : Float := 0.95

/-- Adam learning rate used for the CartPole actor-critic update. -/
def lr : Float := 3e-4

/-- Number of PPO optimization epochs per collected rollout batch. -/
def updateEpochs : Nat := 2

/-- Maximum number of PPO updates (training stops early if the "solved" criterion triggers). -/
def updatesMax : Nat := 1000

/-- Evaluate (greedy policy) every `evalEvery` PPO updates. -/
def evalEvery : Nat := 50

/-- Number of evaluation episodes per checkpoint. -/
def evalEpisodes : Nat := 5

/-- Stop early if average return meets/exceeds this threshold. -/
def solvedAvgReturn : Float := 475.0

instance : Fact (0 < horizon) := ⟨by decide⟩
instance : Fact (0 < nActions) := ⟨by decide⟩

/-- The observation tensor shape used by this run: `[..., stateDim]`. -/
def obsShape : Shape := shape![stateDim]

def pfxBatch : Shape := shape![horizon]
def sStateBatch : Shape := rl.ppo.StateBatchShape horizon obsShape
def sLogitsBatch : Shape := rl.ppo.LogitsBatchShape horizon nActions
def sScalarBatch : Shape := rl.ppo.ScalarBatchShape horizon
def sValueBatch : Shape := rl.ppo.ValueBatchShape horizon

def stateShape : Shape := obsShape
def logitsShape : Shape := shape![nActions]
def valueShape : Shape := shape![1]

/-!
## Model (Actor + Critic)

We use the public `TorchLean.nn` surface, which provides prefix-shape preserving layers:
if `x` has shape `[..., inDim]`, `nn.linear inDim outDim` maps it to `[..., outDim]`.
-/

def modelCfg : nn.models.PPOActorCriticConfig :=
  { obsDim := stateDim, hiddenDim := hiddenDim, nActions := nActions }

/-- Construct the actor network as an MLP mapping observations to action logits. -/
def actorMk (pfx : Shape) : nn.Builder (nn.Sequential (pfx.appendDim stateDim) (pfx.appendDim nActions)) :=
  nn.models.ppoActor modelCfg pfx

/-- Construct the critic network as an MLP mapping observations to a scalar value estimate. -/
def criticMk (pfx : Shape) : nn.Builder (nn.Sequential (pfx.appendDim stateDim) (pfx.appendDim 1)) :=
  nn.models.ppoCritic modelCfg pfx

/-!
## Gymnasium Bridge

We talk to a small Python service (`scripts/rl/gymnasium_server.py`) using the reusable runtime
bridge exposed as `rl.gym.*`.

The Lean side trust-boundary contract (`rl.boundary.Contract`) is enforced on every step.
-/

/-!
## Evaluation

Evaluation APIs live in `rl.eval`.
-/

/-!
## Main Training Loop
-/

/-- Entry point for `lake exe torchlean ppo_cartpole`.

This executable:
- launches a Python Gymnasium subprocess for `CartPole-v1`,
- collects checked rollouts under `rl.boundary.Contract`,
- performs PPO updates on the Torch backend (CUDA is the practical validation path for this command),
- writes a widget-friendly training curve JSON (default: `data/rl/ppo_cartpole_trainlog.json`).
-/
def main (args : List String) : IO UInt32 := do
  Module.Command.runFloat32 exeName args
    (banner := ModelZoo.bannerWithDeviceDetails
      exeName
      s!"PPO on {envId} (horizon={horizon})"
      "  env: Python Gymnasium subprocess (JSON-lines bridge) + Lean boundary contract")
    (k := fun opts rest => do
      let (ppo, rest) ← ModelZoo.orThrow exeName <|
        rl.cli.parsePpoFlags exeName rest Runtime.RL.Artifacts.DefaultPaths.ppoCartPoleTrainLog
          updatesMax evalEvery evalEpisodes 500
      ModelZoo.orThrow exeName <| CLI.checkNoArgs rest
      let updatesLimit : Nat := ppo.updates
      let evalEvery : Nat := ppo.evalEvery
      let evalEpisodes : Nat := ppo.evalEpisodes
      let evalMaxSteps : Nat := ppo.evalMaxSteps
      let contract : rl.boundary.Contract obsShape nActions :=
        { checkObsFinite := true
          checkRewardFinite := true
          obsRange? := none
          rewardRange? := none
          requireExclusiveDoneFlags := false }

      let gym ←
        rl.gym.client.spawn (obsShape := obsShape) (nActions := nActions) gymServerScript envId contract
      try
        -- Build actor/critic once from a single seed, then reuse them at both observation-time shapes
        -- (`pfx = scalar`) and rollout-time batched shapes (`pfx = horizon`).
        let seedActor ← nn.freshSeed
        let seedCritic ← nn.freshSeed
        let actorObs : nn.Sequential stateShape logitsShape :=
          nn.build seedActor (actorMk .scalar)
        let criticObs : nn.Sequential stateShape valueShape :=
          nn.build seedCritic (criticMk .scalar)
        let actorRollout : nn.Sequential sStateBatch sLogitsBatch :=
          nn.build seedActor (actorMk pfxBatch)
        let criticRollout : nn.Sequential sStateBatch sValueBatch :=
          nn.build seedCritic (criticMk pfxBatch)

        let actorGraph ← nn.lowerToTypedGraph actorObs
        let criticGraph ← nn.lowerToTypedGraph criticObs

        let m ← rl.ppo.instantiateActorCritic
          (α := Float) (opts := opts)
          (batch := horizon) (nActions := nActions)
          actorRollout criticRollout

        let stepSample ←
          rl.ppo.makeOptimizerStep m (.adam lr 0.9 0.999 1e-8 : optim.Optimizer)

        let mut rngSeed : Nat := opts.seed
        let mut rngCounter : Nat := 0

        -- Training curve: greedy-policy evaluation return before training, then at each
        -- `evalEvery` checkpoint. We keep this as a compact `Curve` (arrays) because it is
        -- destined for JSON/widget display; the actual learning data is stored as typed tensors.
        let mut curve : rl.train.Curve := {}

        let mkSession : Nat → rl.session.CheckedSession obsShape nActions :=
          fun seed =>
            rl.session.gymnasium (obsShape := obsShape) (nActions := nActions) gym
              (seed? := some seed) (resetOnDone := false)

        -- Evaluate the untrained policy once (step=0).
        do
          let psAll0 ← rl.ppo.state (α := Float) m
          let policyLogits0 : Tensor Float obsShape → Tensor Float logitsShape :=
            rl.ppo.actorPolicy actorGraph actorRollout criticRollout psAll0
          let avg0 ←
            rl.eval.averageEpisodeTotalReward (obsShape := obsShape) (nActions := nActions)
              mkSession policyLogits0 (baseSeed := 1000) (episodes := evalEpisodes)
              (maxSteps := evalMaxSteps)
          curve := curve.push 0 avg0
          IO.println s!"  eval(step=0) avg_return={avg0}"

        for update in [0:updatesLimit] do
          let psAll ← rl.ppo.state (α := Float) m
          let predictLogits : Tensor Float obsShape → Tensor Float logitsShape :=
            rl.ppo.actorPolicy actorGraph actorRollout criticRollout psAll
          let predictValue : Tensor Float obsShape → Float :=
            rl.ppo.criticValue criticGraph actorRollout criticRollout psAll
          let (rollout, rngCounter') ←
            rl.ppo.collectRolloutWith (α := Float) (obsShape := obsShape) (nActions := nActions)
              (horizon := horizon) (castObs := id) (castReward := id) gym predictLogits predictValue
              (rngSeed := rngSeed) (rngCounter := rngCounter) (resetSeed := update)
          rngCounter := rngCounter'

          let sample ←
            rollout.toActorCriticSample (α := Float) (obsShape := obsShape) (nActions := nActions)
              (horizon := horizon) gamma lam
          for _e in [0:updateEpochs] do
            stepSample sample

          if update % evalEvery == 0 then
            let psAll' ← rl.ppo.state (α := Float) m
            let policyLogits : Tensor Float obsShape → Tensor Float logitsShape :=
              rl.ppo.actorPolicy actorGraph actorRollout criticRollout psAll'
            let avg ←
              rl.eval.averageEpisodeTotalReward (obsShape := obsShape) (nActions := nActions)
                mkSession policyLogits (baseSeed := 1000 + update) (episodes := evalEpisodes)
                (maxSteps := evalMaxSteps)
            curve := curve.push (update + 1) avg
            IO.println s!"  update={update} avg_return={avg}"
            if avg ≥ solvedAvgReturn then
              IO.println s!"{exeName}: solved (avg_return ≥ {solvedAvgReturn})"
              break

          -- keep the RNG seed moving so repeated runs can explore different traces if desired
          rngSeed := rand.nextSeed rngSeed update

        ModelZoo.writeCurveTrainLog
          ppo.log
          s!"PPO {envId} (TorchLean)"
          curve
          "avg_return"
          "#4e79a7"
          #[
            s!"env_id={envId}",
            s!"horizon={horizon}",
            s!"gamma={gamma}",
            s!"lambda={lam}",
            s!"lr={lr}",
            s!"updates={updatesLimit}",
            s!"eval_every={evalEvery}",
            s!"eval_episodes={evalEpisodes}",
            s!"eval_max_steps={evalMaxSteps}",
            ModelZoo.deviceNote opts
          ]
        IO.println s!"{exeName}: done"
      finally
        gym.close
    )

end NN.Examples.Models.RL.PPOCartPole
