/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Adapters
public import NN.API.CLI
public import NN.API.Checkpoint
public import NN.API.Scalar
public import NN.API.Data
public import NN.API.Json
public import NN.API.Loss
public import NN.API.Module
public import NN.API.Neural
public import NN.API.Optim
public import NN.API.RL
public import NN.API.Runtime
public import NN.API.SelfSupervised
public import NN.API.Tensor
public import NN.API.TensorPack
public import NN.API.Text
public import NN.API.Trainer
public import NN.API.Verification
public import NN.Spec.Models

/-!
# TorchLean

Neural-network construction, training, runtime execution, datasets, automatic differentiation,
verification, and mathematical model specifications.

Import `NN.API` for model code. Import `NN` when a file also uses the specification, proof,
floating-point, or backend libraries.
-/

@[expose] public section
