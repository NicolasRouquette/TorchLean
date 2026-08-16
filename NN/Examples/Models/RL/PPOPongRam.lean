/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

End-to-end PPO example: train an actor-critic on Atari Pong (ALE) using TorchLean.
-/

module

public import NN.API
public import NN.Examples.ModelZoo
public import NN.Runtime.RL.Artifacts.DefaultPaths

/-!
# PPO on Atari Pong (RAM Observations) (Executable Example)

This example mirrors `NN/Examples/Models/RL/PPOCartPole.lean`, but targets an Atari game via the
Arcade Learning Environment (ALE) registered into Gymnasium as `ALE/Pong-v5`.

Why "RAM" observations?
- Pixel-based Atari PPO is absolutely doable, but a JSON-lines subprocess bridge is not the right
  transport if you want millions of steps/hour. RAM observations (`obs_type="ram"`, shape `128`)
  keep the bridge compact and make this run viable as a native Lean executable.

The key TorchLean interface remains the same:

- **Algorithm math** (GAE, PPO clipped objective) is Lean definitions.
- **Autograd program** (PPO loss) is a TorchLean backend-generic program (CPU or CUDA).
- **Trust boundary** is explicit: every externally sampled transition is checked by
  `Runtime.RL.Boundary.Contract` before it can influence training.

## Dependencies

Atari/ALE environments require `ale-py` and a recent `gymnasium`:

```bash
python3 -m pip install --user 'gymnasium>=1.0' ale-py
```

## CLI flags

- `--device cuda`: run the Torch backend on CUDA (requires building with `-K cuda=true`).
- `--updates <n>`: number of PPO updates to run.
- `--eval-every <n>`: evaluate the greedy policy every `n` updates.
- `--eval-episodes <n>`: number of evaluation episodes per checkpoint.
- `--eval-max-steps <n>`: maximum steps per evaluation episode.
- `--log <path>`: write the widget log JSON to a custom path.

This module is optional. It depends on a compatible external ALE/Gymnasium installation and is not
part of the default `torchlean` runner quick-check list.

Dependency setup:

```bash
python3 -m pip install --user 'gymnasium>=1.0' ale-py
```

Artifacts:
- Writes `data/rl/ppo_pong_ram_trainlog.json` by default (override with `--log`).
- Visualize it in the editor via `NN/Examples/RL/PPOPongRamView.lean`.

References (primary):
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017): https://arxiv.org/abs/1707.06347
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2015):
  https://arxiv.org/abs/1506.02438
- Williams, "Simple statistical gradient-following algorithms for connectionist reinforcement learning"
  (REINFORCE, 1992): https://doi.org/10.1007/BF00992696
- Machado et al., "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems"
  (2018): https://arxiv.org/abs/1709.06009
- ALE docs (environment catalogue and versioned `ALE/...-v5` ids): https://ale.farama.org/
- Gymnasium API reference (reset/step, `terminated` vs `truncated`): https://gymnasium.farama.org/
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.RL.PPOPongRam

/-- Name used in CLI error messages and banners when the optional runner is wired in. -/
def exeName : String := "torchlean ppo_pong_ram"

/-- Help text for the optional ALE/Pong RAM PPO runner. -/
def usage : String :=
  String.intercalate "\n"
    [ "Usage:"
    , "  lake -R -K cuda=true exe torchlean ppo_pong_ram --device cuda [PPO flags]"
    , ""
    , "PPO flags:"
    , "  --updates N          number of PPO updates"
    , "  --eval-every N       evaluate every N updates"
    , "  --eval-episodes N    evaluation episodes per checkpoint"
    , "  --eval-max-steps N   maximum steps per evaluation episode"
    , "  --log PATH|off       training-curve JSON path, or disable logging"
    , "  --check-env-only     start ALE, reset once, take one checked step, then exit"
    , ""
    , "External dependency:"
    , "  python3 -m pip install --user 'gymnasium>=1.0' ale-py"
    ]

/-!
## Configuration
-/

/-- Atari environment id passed to the Python subprocess. -/
def envId : String := "ALE/Pong-v5"

/-- Relative path to the Python Gymnasium bridge script (spawned as a subprocess). -/
def gymServerScript : String := "scripts/rl/gymnasium_server.py"

/--
Pong RAM observation dimension.

Gymnasium exposes RAM as `Box(0, 255, (128,), uint8)` when `obs_type="ram"`.
-/
def stateDim : Nat := 128

/-- Number of discrete actions in Pong under ALE's reduced action set. -/
def nActions : Nat := 6

/-- Width of the hidden layer in the actor and critic MLPs. -/
def hiddenDim : Nat := 64

/-- PPO rollout horizon (also the training batch size for this run). -/
def horizon : Nat := 128

/-- Discount factor used in returns / GAE. -/
def gamma : Float := 0.99

/-- GAE(λ) parameter controlling the bias/variance tradeoff of advantage estimates. -/
def lam : Float := 0.95

/-- Adam learning rate used for the Pong RAM actor-critic update. -/
def lr : Float := 2.5e-4

/-- Number of PPO optimization epochs per collected rollout batch. -/
def updateEpochs : Nat := 1

/-- Default maximum number of PPO updates (override with `--updates`). -/
def updatesMax : Nat := 2000

/-- Default evaluation checkpoint interval (override with `--eval-every`). -/
def evalEvery : Nat := 100

/-- Default evaluation episodes per checkpoint (override with `--eval-episodes`). -/
def evalEpisodes : Nat := 5

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

We use MLPs over RAM. For pixel observations you would typically use a CNN (see
`NN.GraphSpec.Models.TorchLean.Cnn`) and wrap the environment with Atari preprocessing.
-/

def modelCfg : nn.models.PPOActorCriticConfig :=
  { obsDim := stateDim, hiddenDim := hiddenDim, nActions := nActions }

/-- Construct the actor network as an MLP mapping RAM observations to action logits. -/
def actorMk (pfx : Shape) : nn.Builder (nn.Sequential (pfx.appendDim stateDim) (pfx.appendDim nActions)) :=
  nn.models.ppoActor modelCfg pfx

/-- Construct the critic network as an MLP mapping RAM observations to a scalar value estimate. -/
def criticMk (pfx : Shape) : nn.Builder (nn.Sequential (pfx.appendDim stateDim) (pfx.appendDim 1)) :=
  nn.models.ppoCritic modelCfg pfx

/-!
## Gymnasium / ALE bridge

We request RAM observations by passing `{"obs_type": "ram"}` to `gym.make` through the bridge's
`--make-kwargs` option. The server also auto-registers `ale_py` when `envId` starts with `ALE/`.
-/

def makeKwargs : List (String × Lean.Json) :=
  [("obs_type", .str "ram")]

def contract : rl.boundary.Contract obsShape nActions :=
  { checkObsFinite := true
    checkRewardFinite := true
    -- RAM bytes live in `[0,255]` by construction. This range check guards
    -- against protocol bugs or unexpected adapters.
    obsRange? := some (0, 255)
    rewardRange? := none
    requireExclusiveDoneFlags := false }

/--
Start ALE, reset once, take one checked step, and close the subprocess.

This exercises the Gymnasium subprocess, ALE registration, RAM observation shape handshake, and
Lean side boundary contract as the full PPO runner, without collecting a 128-step rollout.
-/
def checkEnvOnly : IO Unit := do
  IO.eprintln s!"  starting env: {envId} (obs_type=ram)"
  let gym ←
    rl.gym.client.spawn (obsShape := obsShape) (nActions := nActions) gymServerScript envId contract
      (makeKwargs := makeKwargs)
  try
    let _obs ← rl.gym.client.reset gym (seed? := some 0)
    let (_obs', reward, terminated, truncated) ← Runtime.RL.Gymnasium.Client.stepRaw gym 0
    IO.println
      s!"{exeName}: env check ok reward={reward} terminated={terminated} truncated={truncated}"
  finally
    rl.gym.client.close gym

/-!
## Main Training Loop
-/

def main (args : List String) : IO UInt32 := do
  if args.contains "--help" || args.contains "-h" then
    IO.println usage
    return 0
  if args.contains "--check-env-only" then
    let args := args.erase "--check-env-only"
    return ←
      Module.Command.runFloat32 exeName args
        (banner := ModelZoo.bannerWithDeviceDetails
          exeName
          s!"PPO on {envId} (obs=ram, env check only)"
          "  env: Python Gymnasium subprocess (ALE) + Lean boundary contract")
        (k := fun _opts rest => do
          ModelZoo.orThrow exeName <| CLI.checkNoArgs rest
          checkEnvOnly)
  Module.Command.runFloat32 exeName args
    (banner := ModelZoo.bannerWithDeviceDetails
      exeName
      s!"PPO on {envId} (obs=ram, horizon={horizon})"
      "  env: Python Gymnasium subprocess (ALE) + Lean boundary contract")
    (k := fun opts rest => do
      let (ppo, rest) ← ModelZoo.orThrow exeName <|
        rl.cli.parsePpoFlags exeName rest Runtime.RL.Artifacts.DefaultPaths.ppoPongRamTrainLog
          updatesMax evalEvery evalEpisodes 10000
      ModelZoo.orThrow exeName <| CLI.checkNoArgs rest

      let updates : Nat := ppo.updates
      let evalEvery : Nat := ppo.evalEvery
      let evalEpisodes : Nat := ppo.evalEpisodes
      let evalMaxSteps : Nat := ppo.evalMaxSteps

      IO.eprintln s!"  starting env: {envId} (obs_type=ram)"
      let gym ←
        rl.gym.client.spawn (obsShape := obsShape) (nActions := nActions) gymServerScript envId contract
          (makeKwargs := makeKwargs)
      try
        IO.eprintln "  building actor/critic..."
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

        IO.eprintln "  lowering actor/critic to typed graphs..."
        let actorGraph ← nn.lowerToTypedGraph actorObs
        let criticGraph ← nn.lowerToTypedGraph criticObs

        IO.eprintln "  initializing module + optimizer..."
        let m ← rl.ppo.instantiateActorCritic
          (α := Float) (opts := opts)
          (batch := horizon) (nActions := nActions)
          actorRollout criticRollout
        IO.eprintln "  module ready"

        let stepSample ←
          rl.ppo.makeOptimizerStep m (.adam lr 0.9 0.999 1e-8 : optim.Optimizer)
        IO.eprintln "  optimizer ready"

        let mut rngSeed : Nat := opts.seed
        let mut rngCounter : Nat := 0

        let mut curve : rl.train.Curve := {}

        let mkSession : Nat → rl.session.CheckedSession obsShape nActions :=
          fun seed =>
            rl.session.gymnasium (obsShape := obsShape) (nActions := nActions) gym
              (seed? := some seed) (resetOnDone := false)

        -- Evaluate once before training (step=0).
        do
          IO.eprintln "  evaluating initial policy..."
          let psAll0 ← rl.ppo.state (α := Float) m
          let policy0 := rl.ppo.actorPolicy actorGraph actorRollout criticRollout psAll0
          let policyLogits0 : Tensor Float obsShape → Tensor Float logitsShape :=
            fun obs => policy0 (Tensor.map (fun x => x / 255.0) obs)
          let avg0 ←
            rl.eval.averageEpisodeTotalReward (obsShape := obsShape) (nActions := nActions)
              mkSession policyLogits0 (baseSeed := 9000) (episodes := evalEpisodes)
              (maxSteps := evalMaxSteps)
          curve := curve.push 0 avg0
          IO.eprintln s!"  eval(step=0) avg_return={avg0}"

        for update in [0:updates] do
          let psAll ← rl.ppo.state (α := Float) m
          let predictLogits : Tensor Float obsShape → Tensor Float logitsShape :=
            rl.ppo.actorPolicy actorGraph actorRollout criticRollout psAll
          let predictValue : Tensor Float obsShape → Float :=
            rl.ppo.criticValue criticGraph actorRollout criticRollout psAll

          let (rollout, rngCounter') ←
            rl.ppo.collectRolloutWith (α := Float) (obsShape := obsShape) (nActions := nActions)
              (horizon := horizon)
              (castObs := fun x => x / 255.0) (castReward := id)
              gym predictLogits predictValue
              (rngSeed := rngSeed) (rngCounter := rngCounter) (resetSeed := update)
          rngCounter := rngCounter'

          let sample ←
            rollout.toActorCriticSample (α := Float) (obsShape := obsShape) (nActions := nActions)
              (horizon := horizon) gamma lam
          for _e in [0:updateEpochs] do
            stepSample sample

          if update % evalEvery == 0 then
            let psAll' ← rl.ppo.state (α := Float) m
            let policy := rl.ppo.actorPolicy actorGraph actorRollout criticRollout psAll'
            let policyLogits : Tensor Float obsShape → Tensor Float logitsShape :=
              fun obs => policy (Tensor.map (fun x => x / 255.0) obs)
            let avg ←
              rl.eval.averageEpisodeTotalReward (obsShape := obsShape) (nActions := nActions)
                mkSession policyLogits (baseSeed := 9000 + update) (episodes := evalEpisodes)
                (maxSteps := evalMaxSteps)
            curve := curve.push (update + 1) avg
            IO.eprintln s!"  update={update} avg_return={avg}"

          rngSeed := rand.nextSeed rngSeed update

        ModelZoo.writeCurveTrainLog
          ppo.log
          s!"PPO {envId} (RAM, TorchLean)"
          curve
          "avg_return"
          "#f28e2b"
          #[
            s!"env_id={envId}",
            s!"obs_type=ram",
            s!"horizon={horizon}",
            s!"gamma={gamma}",
            s!"lambda={lam}",
            s!"lr={lr}",
            s!"eval_every={evalEvery}",
            s!"eval_episodes={evalEpisodes}",
            ModelZoo.deviceNote opts
          ]
        IO.eprintln s!"{exeName}: done"
      finally
        gym.close
    )

end NN.Examples.Models.RL.PPOPongRam
