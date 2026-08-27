/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Trainer.Core

/-!
# Trainer Construction

Constructors for `TorchLean.Trainer`.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

universe u

/--
Values accepted by `Trainer.new`.

This class lets `Trainer.new` accept either a seedable model builder or an already-built checked
model:

```lean
Trainer.new modelBuilder ...
Trainer.new alreadyBuiltModel ...
```

The seed is consumed only by the builder case. Already-built models pass through unchanged.
-/
class ToModel (model : Type u)
    (inputShape outputShape : outParam (List Nat)) where
  /-- Materialize the model, using the seed only when the value still needs initialization. -/
  build : Nat → model → TorchLean.nn.Sequential inputShape outputShape

instance {inputShape outputShape : Shape} :
    ToModel (TorchLean.nn.Sequential inputShape outputShape)
      inputShape.toList outputShape.toList where
  build _ model := by
    simpa using model

instance {inputShape outputShape : List Nat} :
    ToModel (TorchLean.nn.Builder (TorchLean.nn.Sequential inputShape outputShape))
      inputShape outputShape where
  build seed model := TorchLean.nn.build seed model

/-- Build a trainer from a sequential model or seedable model builder. -/
def new {model : Type u} {inputShape outputShape : List Nat}
    [ToModel model inputShape outputShape] (m : model)
    (cfg : Config inputShape outputShape := {}) :
    TorchLean.Trainer inputShape outputShape :=
  let built := ToModel.build cfg.seed m
  { model := built
    task := cfg.task
    runtime :=
      { optimizer := cfg.optimizer
        scalar := cfg.scalar
        execution := cfg.execution
        device := cfg.device
        backendProfile? := cfg.backendProfile?
        showBackend := cfg.showBackend }
    seed := cfg.seed }

end Trainer

end TorchLean
