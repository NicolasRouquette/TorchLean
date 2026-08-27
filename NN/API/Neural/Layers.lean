/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import NN.API.Neural.Layers.Attention
public import NN.API.Neural.Layers.Convolution
public import NN.API.Neural.Layers.Normalization
public import NN.API.Neural.Layers.Pooling

/-!
# Neural Layers

Named configurations and constructors for attention, normalization, convolution, and pooling.
Spatial operators state the trailing axes they consume, while `leading` records any axes mapped
pointwise by the layer.
-/
