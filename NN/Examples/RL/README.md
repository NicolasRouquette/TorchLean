# RL Examples

This folder is the companion layer for TorchLean's runnable RL examples.

The executable trainers are under `NN/Examples/Models` and are selected through the shared runner:

- `lake -R -K cuda=true exe torchlean ppo_gridworld --device cuda --updates 1 --eval-every 1 --eval-episodes 1 --eval-max-steps 8`
- `lake -R -K cuda=true exe torchlean ppo_cartpole --device cuda --updates 1 --eval-every 1 --eval-episodes 1 --eval-max-steps 8`
- `lake exe torchlean dqn_replay`

The Pong RAM files remain in the tree as an optional ALE/Gymnasium boundary example. They are not
part of the default runner quick-check list because they depend on a compatible external `ale-py` /
`gymnasium` installation.

The files here do three different jobs:

- `PPOGridWorldView.lean`, `PPOCartPoleView.lean`, `PPOPongRamView.lean`: editor widgets for logs,
  GridWorld policies, and episode paths written by the trainers.

The Python boundary helpers live under `scripts/rl/` so runtime code does not depend on
`Examples/` paths:

- `scripts/rl/gymnasium_server.py`: JSON lines bridge used by Lean to step external Gymnasium
  environments behind a checked boundary contract.
- `scripts/rl/export_gymnasium_rollout.py`: exporter for offline rollout JSON accepted by the Lean
  RL boundary loader.

## Recommended Workflow

1. Train or runtime check a Lean example:

```bash
lake -R -K cuda=true exe torchlean ppo_gridworld --device cuda --updates 1 --eval-every 1 --eval-episodes 1 --eval-max-steps 8
lake -R -K cuda=true exe torchlean ppo_cartpole --device cuda --updates 1 --eval-every 1 --eval-episodes 1 --eval-max-steps 8
```

2. Open the corresponding `*View.lean` file in the editor and put the cursor on the widget command.

3. For external Gymnasium environments, treat Python as an untrusted producer: TorchLean checks
   observations, rewards, actions, and done flags before consuming rollout data.

## Dependencies

For CartPole:

```bash
python3 -m pip install --user 'gymnasium>=1.0'
```

Optional ALE/Pong RAM path:

```bash
python3 -m pip install --user 'gymnasium>=1.0' ale-py
```

References:

- Schulman et al., "Proximal Policy Optimization Algorithms", 2017.
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation",
  2015.
- Machado et al., "Revisiting the Arcade Learning Environment", 2018.
- Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd ed.
