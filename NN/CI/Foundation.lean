/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Scalar
public import NN.API.Data
public import NN.API.Data.Transforms
public import NN.API.Init
public import NN.API.Macros
public import NN.API.Neural
public import NN.API.Runtime
public import NN.API.Data.Bands
public import NN.API.Data.Synthetic
public import NN.API.Trainer.Scheduler
public import NN.GraphSpec
public import NN.Spec
public import NN.IR

public import NN.Spec.Autograd.AutogradSpec
public import NN.Spec.Autograd.Ops
public import NN.Spec.Core.Context
public import NN.Spec.Core.Scalar
public import NN.Spec.Core.Sequence
public import NN.Spec.Core.Shape
public import NN.Spec.Core.Tensor
public import NN.Spec.Core.Tensor.Constructors
public import NN.Spec.Core.Tensor.Core
public import NN.Spec.Core.Tensor.Linalg
public import NN.Spec.Core.Tensor.Vec
public import NN.Spec.Core.TensorArray
public import NN.Spec.Core.TensorBridge
public import NN.Spec.Core.TensorGrad
public import NN.Spec.Core.TensorOps
public import NN.Spec.Core.TensorReductionShape
public import NN.Spec.Core.Utils
public import NN.Spec.Dynamics.System
public import NN.Spec.Dynamics.StateSpace
public import NN.Spec.Layers.Activation
public import NN.Spec.Layers.Attention
public import NN.Spec.Layers.Conv
public import NN.Spec.Layers.Dropout
public import NN.Spec.Layers.Embedding
public import NN.Spec.Layers.FlashAttention
public import NN.Spec.Layers.Gnn
public import NN.Spec.Layers.Gru
public import NN.Spec.Layers.Linear
public import NN.Spec.Layers.Loss
public import NN.Spec.Layers.Lstm
public import NN.Spec.Layers.Normalization
public import NN.Spec.Layers.Pooling
public import NN.Spec.Layers.PositionalEncoding
public import NN.Spec.Layers.Rnn
public import NN.Spec.Layers.SelectiveScan
public import NN.Spec.Layers.Utils
public import NN.Spec.Models.Autoencoder
public import NN.Spec.Models.Cnn
public import NN.Spec.Models.CommonHelpers
public import NN.Spec.Models.Gmm
public import NN.Spec.Models.Gnn
public import NN.Spec.Models.GradientBoostedTrees
public import NN.Spec.Models.Hmm
public import NN.Spec.Models.Hopfield
public import NN.Spec.Models.Knn
public import NN.Spec.Models.LinearRegression
public import NN.Spec.Models.LogisticRegression
public import NN.Spec.Models.Mamba
public import NN.Spec.Models.Mlp
public import NN.Spec.Models.NaiveBayes
public import NN.Spec.Models.Pca
public import NN.Spec.Models.RandomForest
public import NN.Spec.Models.S4
public import NN.Spec.Models.Seq2seq
public import NN.Spec.Models.Svm
public import NN.Spec.Models.Transformer
public import NN.Spec.Models.Unet
public import NN.Spec.Models.Vit
public import NN.Spec.Module.Activation
public import NN.Spec.Module.Attention
public import NN.Spec.Module.Autoencoder
public import NN.Spec.Module.Conv
public import NN.Spec.Module.DecisionTree
public import NN.Spec.Module.Dropout
public import NN.Spec.Module.Embedding
public import NN.Spec.Module.Flatten
public import NN.Spec.Module.Gnn
public import NN.Spec.Module.GradientBoostedTrees
public import NN.Spec.Module.GruModels
public import NN.Spec.Module.Hmm
public import NN.Spec.Module.Linear
public import NN.Spec.Module.LinearRegression
public import NN.Spec.Module.LogisticRegression
public import NN.Spec.Module.LstmModels
public import NN.Spec.Module.Normalization
public import NN.Spec.Module.Pca
public import NN.Spec.Module.Pooling
public import NN.Spec.Module.PositionalEncoding
public import NN.Spec.Module.Rnn
public import NN.Spec.Module.RnnModels
public import NN.Spec.Module.Seq2seq
public import NN.Spec.Module.SpecModule
public import NN.Spec.Module.Svm
public import NN.Tensor

/-!
# Foundation CI Suite

Focused CI import suite. `NN.CI.All` combines every suite for exhaustive repository validation.

Local usage:

```bash
lake build NN.CI.All
```
-/

@[expose] public section
