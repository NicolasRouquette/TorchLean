/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.RL.Core
public import NN.Spec.RL.Environment
public import NN.Spec.RL.Envs.GridWorld
public import NN.Spec.RL.MDP
public import NN.Spec.RL.FiniteStochasticMDP
public import NN.Spec.RL.MarkovMDP

/-!
# Spec reinforcement learning

Umbrella import for RL environments and MDP specifications.

The files here keep transition systems, rewards, and policy-facing environment contracts separate
from runtime training code, so RL examples can be checked against a small pure semantics.

Folder map:

- `Core`: Bellman/TD/return/GAE algebra over shape-indexed trajectories.
- `Environment`: a pure Gymnasium-style environment contract with explicit latent state.
- `MDP`: deterministic finite MDPs over `Fin n` states/actions and tensor value tables.
- `FiniteStochasticMDP`: finite stochastic MDPs with row-stochastic tensor transition kernels.
- `MarkovMDP`: measure-theoretic MDPs over measurable spaces using mathlib Markov kernels.
- `Envs/GridWorld`: a concrete finite environment plus deterministic/stochastic MDP views.

Deterministic finite MDPs live in `Spec.RL`, finite stochastic MDPs in
`Spec.RL.FiniteStochastic`, and measurable-space MDPs in `Spec.RL.Markov`. Each namespace supplies
the definitions its representation needs. Keeping these layers separate avoids forcing every RL
development into the heaviest probability-theory abstraction.
-/

@[expose] public section
