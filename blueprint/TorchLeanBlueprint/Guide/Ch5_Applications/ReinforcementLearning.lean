import VersoManual

open Verso.Genre Manual

#doc (Manual) "Reinforcement Learning" =>
%%%
tag := "reinforcement-learning"
%%%

In supervised learning, the dataset is usually fixed before training starts. In reinforcement
learning, the current policy helps create its own future data. A complete application therefore
contains more than a neural network:

$$`\text{environment}
\longrightarrow\text{transition}
\longrightarrow\text{rollout or replay}
\longrightarrow\text{return and advantage}
\longrightarrow\text{policy/value update}.`

Each arrow carries assumptions. Is the observation shape correct? Is the action valid? Is the
reward finite? Does `done` mean termination, truncation, or either? Did the rollout keep its fields
aligned? TorchLean gives those questions explicit runtime and specification objects.

There are three complementary layers:

- [`NN.Spec.RL`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/RL.lean) defines
  environments, MDPs, Bellman operators, returns, and advantages;
- [`NN.Runtime.RL`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/RL.lean) implements
  checked transitions, replay, rollouts, PPO, and Gymnasium communication;
- [`NN.Proofs.RL`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RL/Core.lean) proves
  structural, numerical, and dynamic-programming facts about named objects from the first two
  layers.

The easiest way to see the pieces together is a small GridWorld run.

# A Complete PPO GridWorld Run

The environment is a `4 × 4` GridWorld defined in Lean. The actor and critic use a fixed rollout
horizon of 64. Run one update on CPU and send the artifacts to temporary files:

```
lake exe torchlean ppo_gridworld --device cpu \
  --updates 1 \
  --eval-every 1 --eval-episodes 1 --eval-max-steps 8 \
  --log /tmp/ppo-gridworld-trainlog.json \
  --policy /tmp/ppo-gridworld-policy.json \
  --path /tmp/ppo-gridworld-path.json
```

The current checkout produces:

```
[TorchLean] dtype: Float (Lean `Float`, trusted runtime semantics)
[TorchLean] backend: Runtime.Autograd.Torch.Backend.eager
[TorchLean] device: cpu
torchlean ppo_gridworld: PPO on Lean-native GridWorld (4x4, horizon=64) (device=cpu)
  env: pure Lean dynamics + boundary contract check + formal MDP validity proof available
  eval(step=0) avg_return=-0.400000
  update=0 avg_return=3.600000
  wrote TrainLog JSON: /tmp/ppo-gridworld-trainlog.json
torchlean ppo_gridworld: wrote policy snapshot to /tmp/ppo-gridworld-policy.json
torchlean ppo_gridworld: wrote path snapshot to /tmp/ppo-gridworld-path.json
torchlean ppo_gridworld: done
torchlean ppo_gridworld: ok
```

This trace contains three different results:

1. the PPO/autograd program completed one update;
2. a greedy-policy evaluation changed on this seeded run;
3. the command wrote a scalar curve, a policy table, and a decoded path.

Only the first is an optimizer execution fact. The second is an empirical observation from one
small run. The third gives inspectable evidence. The formal MDP facts described below are separate
theorems.

The implementation is
[`NN/Examples/Models/RL/PPOGridWorld.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/RL/PPOGridWorld.lean).
The pure environment is
[`NN.Spec.RL.Envs.GridWorld`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/RL/Envs/GridWorld.lean).

# One Actor, Two Views Of A Batch

The GridWorld command does not hide its actor and critic inside the training loop. It starts with a
small reusable configuration:

```
def modelCfg : nn.models.PPOActorCriticConfig :=
  { obsDim := 16, hiddenDim := 32, nActions := 4 }
```

[`NN.API.Models.PPO`](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Models/PPO.lean)
then builds a two-layer `tanh` MLP for each role. The actor ends in four action logits; the critic
ends in one value. Both constructors preserve a caller-supplied prefix shape, so the same model
description has two useful views:

:::table +header
*
  * Prefix
  * Actor shape
  * Critic shape
*
  * one observation
  * `16 -> 4`
  * `16 -> 1`
*
  * rollout of 64 observations
  * `64 × 16 -> 64 × 4`
  * `64 × 16 -> 64 × 1`
:::

The prefix belongs to the sequential model's input and output shapes rather than an implicit
broadcast convention. GridWorld, CartPole, and Pong RAM reuse these constructors with different
observation and action dimensions; environment collection and boundary checks remain outside the
model helper. This separation is why the same actor definition can score one state during rollout
and a full horizon during optimization without introducing a second, loosely related network.

# What Enters A Rollout

For a discrete action space of size `A`, one PPO step stores:

$$`(s_t,\;a_t,\;\log\pi_{\mathrm{old}}(a_t\mid s_t),\;
r_t,\;d_t,\;V(s_t),\;V(s_{t+1})).`

The Lean structure in
[`NN.Runtime.RL.PPO.Rollout`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/RL/PPO/Rollout.lean)
uses:

- `state : Tensor α obsShape`;
- `action : Fin nActions`;
- scalar old log probability, reward, value, and next value;
- a Boolean episode-boundary marker.

A `Rollout α obsShape nActions horizon` contains an array plus a proof that its size is exactly
`horizon`. This avoids the “parallel arrays drifted out of sync” failure common in hand-written
buffers.

Converting a rollout to an actor-critic minibatch produces:

$$`\begin{aligned}
\text{states}&:T\times\text{obsShape},\\
\text{actionsOneHot}&:T\times A,\\
\text{oldLogProb}&:T,\\
\text{advantages}&:T,\\
\text{valueTarget}&:T\times1.
\end{aligned}`

The conversion is total because the horizon invariant is already present.

# Returns And Generalized Advantage Estimation

Let `d_t` be one when timestep `t` ends an episode and zero otherwise. The one-step temporal
difference residual is

$$`\delta_t
=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t).`

Generalized Advantage Estimation runs backward:

$$`A_t
=\delta_t+\gamma\lambda(1-d_t)A_{t+1}.`

The corresponding value target is

$$`R_t=A_t+V(s_t).`

TorchLean keeps the list-level definitions in
[`NN.Spec.RL.Core`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/RL/Core.lean)
and tensor-shaped runtime siblings in
[`NN.Runtime.RL.Core`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/RL/Core.lean).
`Rollout.toActorCriticSample` computes value targets from the unnormalized advantages, then
z-score normalizes advantages for the policy term.

This is more than a documentation choice. Months after an experiment, the definition answers
questions that a log cannot: whether truncation stopped bootstrapping, whether normalization
changed value targets, and which direction the recurrence traversed the rollout.

# The PPO Objective

For the sampled action at timestep `t`, define

$$`r_t(\theta)
=\exp\!\left(
\log\pi_\theta(a_t\mid s_t)
-\log\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)
\right).`

The clipped surrogate is

$$`L^{\mathrm{clip}}_t(\theta)
=\min\!\left(
r_t(\theta)A_t,\;
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\right).`

The actor minimizes the negative mean surrogate. The critic receives a value loss, and the full
autograd program can add an entropy term. The categorical action log probability is implemented
from `logSoftmax` and a one-hot action tensor in
[`NN.Runtime.RL.PolicyGradient.Autograd`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/RL/PolicyGradient/Autograd.lean).

The ratio, clipping, return, and advantage computations also have checked binary32 helpers under
[`NN.Runtime.RL.Numerics.Float32`](https://github.com/lean-dojo/TorchLean/tree/main/NN/Runtime/RL/Numerics/Float32).
They use the executable `IEEE32Exec` semantics and return `Except String ...`, so a NaN, infinity,
or failed finite-path precondition is visible rather than entering the update silently.

These helpers establish properties of selected scalar recurrences; whole-loop CUDA refinement is a
larger claim.

The corresponding proof bridge is split between
[`Floats.IEEE32Exec`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RL/Floats/IEEE32Exec.lean)
and
[`Floats.CheckedRuntime`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RL/Floats/CheckedRuntime.lean).
The first states finite-path equalities for discounted backups and TD residuals; the second uses
the checked wrappers to discharge those finite-result premises. Together they cover those selected
recurrences, not a numerical error theorem for the whole PPO loop.

# Bellman Operators And Fixed Points

The same vocabulary supports classical MDP theorems. For a fixed policy `π`,

$$`(T^\pi V)(s)
=r(s,\pi(s))
+\gamma\mathbb E_{s'\sim P(\cdot\mid s,\pi(s))}[V(s')].`

The optimality operator is

$$`(T^\star V)(s)
=\max_a\left(
r(s,a)+\gamma\mathbb E_{s'\sim P(\cdot\mid s,a)}[V(s')]
\right).`

For `0 ≤ γ < 1`, both are contractions in the sup norm:

$$`\|TV-TW\|_\infty
\le\gamma\|V-W\|_\infty.`

The Markov-MDP proof file
[`NN.Proofs.RL.MarkovMDP`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RL/MarkovMDP.lean)
contains the named results `bellmanPolicy_contraction`,
`bellmanOptimality_contraction`, `bellmanPolicy_fixedPoint_unique`, and
`bellmanOptimality_fixedPoint_unique`.

These are real dynamic-programming theorems. PPO convergence concerns a different object: a
parameterized stochastic policy updated from sampled finite trajectories.

For explicitly finite state and action spaces,
[`FiniteStochasticMDP`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RL/FiniteStochasticMDP.lean)
also proves monotonicity, contraction, uniqueness of policy and optimality fixed points, and
geometric iterate-error bounds. The finiteness and discount hypotheses are part of those theorem
statements; they are not inferred from a simulator run.

# External Environments

GridWorld is fully represented in Lean. CartPole and Pong are not. Reimplementing every simulator
inside the theorem prover would make TorchLean isolated from the ecosystem, so the runtime can
start a Python Gymnasium process and communicate through one JSON request and response per line.

The bridge lives in
[`NN.Runtime.RL.Gymnasium`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/RL/Gymnasium.lean).
At startup, Lean asks the process for the observation shape and number of actions. On reset and
step, the returned values pass through
[`NN.Runtime.RL.Boundary`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/RL/Boundary/Core.lean).

A boundary contract may require:

- every observation entry to be finite;
- every reward to be finite;
- observations and rewards to lie in declared closed intervals;
- actions to satisfy `action < nActions`;
- `terminated` and `truncated` not to be simultaneously true.

After checking, an action is a `Fin nActions` and the transition is a
`Spec.RL.ObservedTransition`. A malformed Python dictionary never becomes training data merely
because it could be decoded as JSON.

This is a checked interface, not a proof of the simulator. TorchLean does not establish that
Gymnasium's CartPole dynamics are Markov, that ALE emulates the Atari hardware correctly, or that
the Python process honored its random seed.

# CartPole And Pong RAM

Install the optional external dependency:

```
python3 -m pip install --user 'gymnasium>=1.0'
```

Then a short CartPole run is:

```
lake exe torchlean ppo_cartpole --device cpu \
  --updates 1 --eval-every 1 \
  --eval-episodes 1 --eval-max-steps 8
```

The command expects a four-entry observation and two actions. Its actor and critic are MLPs with
hidden width 32, rollout horizon 64, discount `0.99`, GAE parameter `0.95`, and Adam learning rate
`3 × 10⁻⁴`. Those values are defined in
[`PPOCartPole.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/RL/PPOCartPole.lean),
not inferred from a generic PPO label.

The optional Pong path uses RAM observations to keep the JSON-lines boundary inspectable:

```
python3 -m pip install --user 'gymnasium>=1.0' ale-py

lake exe torchlean ppo_pong_ram --device cpu \
  --check-env-only
```

The environment is `ALE/Pong-v5`, each observation has 128 byte-valued entries, and the action
space has six elements. A pixel PPO implementation would require a much higher-throughput
transport; the RAM example is intentionally about the checked environment boundary.

# Off-Policy Data: DQN Replay

Off-policy algorithms reuse transitions. TorchLean's
[`Replay.Buffer`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/RL/Replay.lean)
is a bounded FIFO array indexed by scalar type, observation shape, and action count. Pushing into a
full buffer drops the oldest transition; sampling is deterministic from explicit seed and counter
values.

Run the self-contained example:

```
lake exe torchlean dqn_replay
```

Current output:

```
dqn_replay: begin
stored transitions: 2
sampled transitions: 4
DQN minibatch MSE loss:   0.917800
DQN minibatch Huber loss: 0.452500
soft target update example: 1.000000
dqn_replay: ok
```

The sample count can exceed the stored count because this compact demonstration samples with
replacement. The example computes DQN targets, MSE and Huber losses, and a soft target-network
update. It is not a full epsilon-greedy environment training run.

The proof module
[`NN.Proofs.RL.Replay`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RL/Replay.lean)
establishes structural facts about empty buffers, zero capacity, size growth, and eviction at
capacity. Those are small theorems with a useful job: later DQN reasoning need not assume that the
storage layer preserved its own capacity invariant.

The
[`DQN algebra proof`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RL/Algorithms/DQN.lean)
adds exact real identities for target-network soft updates, including the zero/one mixing endpoints
and `softUpdateScalar_sub_target_real`. It proves the scalar recurrence, not correctness of replay
sampling, Q-network optimization, or a complete DQN agent.

# Hands-On Checks

## Inspect The Artifacts

After the GridWorld command, compare the `before` and `after` policy arrays:

```
python3 -m json.tool /tmp/ppo-gridworld-policy.json
python3 -m json.tool /tmp/ppo-gridworld-path.json
```

The policy artifact records one greedy action per grid cell. The path artifact records decoded
`(row, column)` states. A rising scalar return is easier to interpret when these two objects agree
with it.

## Exercise Strict Parsing

The application runner rejects unconsumed flags. For example, `--rollout 8` is not a GridWorld
option:

```
lake exe torchlean ppo_gridworld --device cpu --updates 1 --rollout 8
```

The command fails with an `unexpected arguments` message instead of silently running horizon 64.
The horizon is a typed constant in this example; changing it is a source-level model change.

## Compare CPU And CUDA

With a CUDA build:

```
lake -R -K cuda=true exe torchlean ppo_gridworld --device cuda \
  --updates 1 --eval-every 1 --eval-episodes 1 --eval-max-steps 8 \
  --show-backend
```

The environment dynamics remain Lean-native. The actor, critic, and PPO autograd operations move to
the selected CUDA runtime, and `--show-backend` prints the chosen backend contracts. A successful
CUDA run is runtime evidence; kernel-level proof status is read from those contracts rather than
inferred from the device name.

# What Is Implemented Today

TorchLean currently provides:

- executable PPO applications over a Lean environment and external Gymnasium environments;
- typed fixed-horizon rollouts and bounded replay;
- explicit checks for external observations, actions, rewards, and episode flags;
- checked binary32 helpers for selected return, advantage, and PPO scalar formulas;
- Bellman contraction and fixed-point theorems for named MDP objects;
- finite-state policy/optimality iteration error bounds;
- finite-path IEEE32Exec and checked-runtime bridges for selected backup and TD formulas;
- real-algebra target-network update laws for DQN;
- structural proofs for environment and replay components.

The Bellman results cover the named MDP objects, while the PPO commands exercise the training
system. Connecting a particular simulator and native PPO run all the way to those mathematical
objects remains a larger end-to-end result.

# References

- Sutton and Barto,
  [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html),
  second edition.
- Schulman et al.,
  [*High-Dimensional Continuous Control Using Generalized Advantage Estimation*](https://arxiv.org/abs/1506.02438),
  2015.
- Schulman et al.,
  [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347), 2017.
- Mnih et al.,
  [*Human-level control through deep reinforcement learning*](https://www.nature.com/articles/nature14236),
  2015.
- Gymnasium, [environment API](https://gymnasium.farama.org/).
