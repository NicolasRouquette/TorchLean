/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded

/-!
# PPO Actor-Critic Models

Reusable actor/critic MLP constructors for PPO examples.

These helpers cover the neural-network shape. Environment collection, trust-boundary checks,
advantage computation, and optimizer loops stay in the examples/runtime modules.
-/

@[expose] public section

namespace TorchLean


open Spec Tensor

namespace nn
namespace models

/-- Configuration for a simple PPO actor/critic pair over vector observations. -/
structure PPOActorCriticConfig where
  obsDim : Nat
  hiddenDim : Nat
  nActions : Nat
deriving Repr

/-- Actor input shape: observation vectors with a caller-chosen prefix shape. -/
abbrev ppoActorInShape (cfg : PPOActorCriticConfig) (pfx : Spec.Shape) : Spec.Shape :=
  pfx.appendDim cfg.obsDim

/-- Actor output shape: action logits with the same prefix shape. -/
abbrev ppoActorOutShape (cfg : PPOActorCriticConfig) (pfx : Spec.Shape) : Spec.Shape :=
  pfx.appendDim cfg.nActions

/-- Critic output shape: one scalar value per prefixed observation. -/
abbrev ppoCriticOutShape (_cfg : PPOActorCriticConfig) (pfx : Spec.Shape) : Spec.Shape :=
  pfx.appendDim 1

/-- Actor MLP mapping observations to action logits. -/
def ppoActor (cfg : PPOActorCriticConfig) (pfx : Spec.Shape) :
    nn.M (nn.Sequential (ppoActorInShape cfg pfx) (ppoActorOutShape cfg pfx)) :=
  nn.Sequential![
    linear cfg.obsDim cfg.hiddenDim (pfx := pfx),
    nn.tanh,
    linear cfg.hiddenDim cfg.nActions (pfx := pfx)
  ]

/-- Critic MLP mapping observations to a scalar value estimate. -/
def ppoCritic (cfg : PPOActorCriticConfig) (pfx : Spec.Shape) :
    nn.M (nn.Sequential (ppoActorInShape cfg pfx) (ppoCriticOutShape cfg pfx)) :=
  nn.Sequential![
    linear cfg.obsDim cfg.hiddenDim (pfx := pfx),
    nn.tanh,
    linear cfg.hiddenDim 1 (pfx := pfx)
  ]

end models
end nn

end TorchLean
