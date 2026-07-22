/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.NN
public import NN.API.Public.TensorPack
public import NN.API.Public.Seeded
public import NN.API.Public.Autograd
public import NN.API.Data
public import NN.API.Data.Transforms
public import NN.API.Runtime
public import NN.API.Models
public import NN.API.Public.NN.Transformer
public import NN.API.RL
public import NN.API.Rand
public import NN.API.Samples.Bands
public import NN.API.Text.Bpe
public import NN.MLTheory.CROWN.Flatbox
public import NN.MLTheory.CROWN.Graph
public import NN.Verification.TorchLean.Compile

/-!
# TorchLean Public Names

Short root names and training-log types used by `import NN`.
-/

@[expose] public section

namespace TorchLean

/-- Public shape type used by tensors, layers, and verifier interfaces. -/
abbrev Shape := Spec.Shape

/-- One supervised-learning observation containing an input and its target. -/
abbrev SupervisedSample := TorchLean.Sample.Supervised

/-- Runtime options selecting TorchLean's dtype, backend, and device. -/
abbrev Options := NN.API.TorchLean.Options

/-- Heterogeneous, shape-indexed collection of tensors such as model parameters. -/
abbrev TensorPack := NN.API.TorchLean.TensorPack

namespace Training

/-- Named sequence of scalar measurements collected during training. -/
abbrev Curve := _root_.Runtime.Training.Curve

/-- Per-run training history, including losses and user-defined metrics. -/
abbrev TrainLog := _root_.Runtime.Training.TrainLog

/-- Collection of training runs recorded as one experiment. -/
abbrev ExperimentLog := _root_.Runtime.Training.ExperimentLog

/-- Destination to which a training run writes its structured log. -/
abbrev LogDestination := _root_.Runtime.Training.LogDestination

/-- Time series of named metric values accumulated during training. -/
abbrev MetricHistory := _root_.Runtime.Training.MetricHistory

/-- In-memory supervised dataset consumed by TorchLean training loops. -/
abbrev Dataset := _root_.Runtime.Autograd.Train.Dataset

/-- Batched dataset iterator with explicit ordering and shuffle state. -/
abbrev DataLoader := _root_.Runtime.Autograd.Train.DataLoader

namespace MetricHistory

export _root_.Runtime.Training.MetricHistory (empty)

end MetricHistory

end Training

end TorchLean
