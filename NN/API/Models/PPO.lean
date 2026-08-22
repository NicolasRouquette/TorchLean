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
namespace PPO

/-- Configuration for a simple PPO actor/critic pair over vector observations. -/
structure Config where
  obsDim : Nat
  hiddenDim : Nat
  nActions : Nat
deriving Repr

/-- Observation shape with arbitrary leading axes. -/
abbrev inputShape (cfg : Config) (leading : Spec.Shape := .scalar) : Spec.Shape :=
  leading.appendDim cfg.obsDim

/-- Action-logit shape with the same leading axes as the observations. -/
abbrev actorOutputShape (cfg : Config) (leading : Spec.Shape := .scalar) : Spec.Shape :=
  leading.appendDim cfg.nActions

/-- Value-estimate shape with the same leading axes as the observations. -/
abbrev criticOutputShape (_cfg : Config) (leading : Spec.Shape := .scalar) : Spec.Shape :=
  leading.appendDim 1

/-- Actor MLP mapping observations to action logits. -/
def actor (cfg : Config) (leading : Spec.Shape := .scalar) :
    nn.Builder (nn.Sequential (inputShape cfg leading) (actorOutputShape cfg leading)) :=
  nn.Sequential![
    linear cfg.obsDim cfg.hiddenDim (leading := leading),
    nn.tanh,
    linear cfg.hiddenDim cfg.nActions (leading := leading)
  ]

/-- Critic MLP mapping observations to a scalar value estimate. -/
def critic (cfg : Config) (leading : Spec.Shape := .scalar) :
    nn.Builder (nn.Sequential (inputShape cfg leading) (criticOutputShape cfg leading)) :=
  nn.Sequential![
    linear cfg.obsDim cfg.hiddenDim (leading := leading),
    nn.tanh,
    linear cfg.hiddenDim 1 (leading := leading)
  ]

end PPO
end models
end nn

end TorchLean
