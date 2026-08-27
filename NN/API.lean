/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Adapters
public import NN.API.Autograd
public import NN.API.Checkpoint
public import NN.API.Scalar
public import NN.API.Data
public import NN.API.Loss
public import NN.API.Models
public import NN.API.Module
public import NN.API.Neural
public import NN.API.Optim
public import NN.API.RL
public import NN.API.Runtime
public import NN.API.SelfSupervised
public import NN.Tensor
public import NN.API.Sample
public import NN.API.Text
public import NN.API.Trainer

/-!
# TorchLean

Neural-network construction, training, runtime execution, datasets, reinforcement learning,
self-supervised learning, and automatic differentiation.

Import `NN.API` for model code. Import `NN` when a file also uses specification or proof internals.

Focused application surfaces are also available as `NN.API.RL`, `NN.API.SelfSupervised`, and
`NN.API.Verification`.
-/

@[expose] public section
