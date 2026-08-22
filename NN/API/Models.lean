/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Models.KAN
public import NN.API.Models.Cnn
public import NN.API.Models.ResNet
public import NN.API.Models.Vit
public import NN.API.Models.Recurrent
public import NN.API.Models.Transformer
public import NN.API.Models.CausalTransformer
public import NN.API.Models.Mamba
public import NN.API.Models.Generative
public import NN.API.Models.SelfSupervised
public import NN.API.Models.Diffusion
public import NN.API.Models.FNO
public import NN.API.Models.PPO
public import NN.Spec.Models

/-!
# Neural Architectures

Reusable model constructors, configuration records, and classical model specifications.

Individual files under `NN/API/Models/*` own the implementation of each architecture family.
Examples should import this API layer, then add only dataset loading, CLI parsing, and reporting.
-/
