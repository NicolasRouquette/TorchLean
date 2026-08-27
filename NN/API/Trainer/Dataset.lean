/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Sample
public import NN.API.Scalar
public import NN.Data.SampleStream

/-!
# Trainer Datasets

`Trainer.Dataset inputShape targetShape` is the public supervised-data interface. It delays both
sample construction and scalar conversion until training begins. Consequently the same dataset can
be used with every scalar implementation supported by the trainer.

The materialized value is a finite `Data.SampleStream`: samples are requested by index and need not
be stored eagerly. Dataset constructors live in `NN.API.Data.Training`.
-/

@[expose] public section

namespace TorchLean.Trainer

/-- Supervised data with statically known input and target shapes. -/
structure Dataset (inputShape targetShape : List Nat) where
  /-- Construct the finite sample stream after the trainer selects its runtime scalar type. -/
  build :
    {α : Type} →
    [_root_.Context α] →
    [Runtime.FromFloat α] →
    IO (Data.SampleStream (Sample.Supervised α inputShape targetShape))

end TorchLean.Trainer
