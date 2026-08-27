/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Export.Core
public import NN.Spec.Models.Cnn

/-!
# Convolutional PyTorch Reference Export

PyTorch exporter for the two-block convolutional round-trip reference model.

The Lean configuration is rank-parametric. PyTorch itself exposes separate `Conv1d`, `Conv2d`, and
`Conv3d` classes, so that distinction is introduced only while rendering the external Python code.

The generated model has two convolution, ReLU, and max-pool blocks followed by `Flatten` and one
`Linear` head.
-/

@[expose] public section


namespace Export
namespace CNNPyTorch

open Spec
open Tensor
open Spec.Module
open Models
open Export.PyTorch

/-- Rank-parametric configuration for a PyTorch convolution layer. -/
structure ConvCfg (spatialRank : Nat) where
  /-- Input channels (`in_channels`). -/
  inChannels : Nat
  /-- Output channels (`out_channels`). -/
  outChannels : Nat
  /-- Kernel extent along each spatial axis. -/
  kernel : Spec.Tensor Nat [spatialRank]
  /-- Stride along each spatial axis. -/
  stride : Spec.Tensor Nat [spatialRank]
  /-- Zero-padding along each spatial axis. -/
  padding : Spec.Tensor Nat [spatialRank]

/-- Rank-parametric configuration for a PyTorch max-pooling layer. -/
structure MaxPoolCfg (spatialRank : Nat) where
  /-- Pooling-window extent along each spatial axis. -/
  kernel : Spec.Tensor Nat [spatialRank]
  /-- Stride along each spatial axis. -/
  stride : Spec.Tensor Nat [spatialRank]
  /-- Zero-padding along each spatial axis. -/
  padding : Spec.Tensor Nat [spatialRank]

/-- Configuration for the 2-block CNN exporter. -/
structure CnnStackConfig (spatialRank : Nat) where
  /-- Class name to use in the generated Python. -/
  className : String := "CNN"
  /-- Input channels. -/
  inputC : Nat
  /-- Input extent along each spatial axis. -/
  inputSpatial : Spec.Tensor Nat [spatialRank]
  /-- First convolution. -/
  conv1 : ConvCfg spatialRank
  /-- First pooling layer. -/
  pool1 : MaxPoolCfg spatialRank
  /-- Second convolution. -/
  conv2 : ConvCfg spatialRank
  /-- Second pooling layer. -/
  pool2 : MaxPoolCfg spatialRank
  /-- Flattened feature count consumed by the linear head. -/
  flatSize : Nat
  /-- Output width of the linear head. -/
  fcOut : Nat

/-- Render a list of dimensions as a Python tuple. -/
def natListToPyTuple (dims : List Nat) : String :=
  "(" ++ ", ".intercalate (dims.map toString) ++ (if dims.length = 1 then "," else "") ++ ")"

/-- Select the rank-specific class name required by PyTorch's public API. -/
def spatialClassName (base : String) (spatialRank : Nat) : Except String String :=
  match spatialRank with
  | 1 => .ok s!"{base}1d"
  | 2 => .ok s!"{base}2d"
  | 3 => .ok s!"{base}3d"
  | rank => .error s!"PyTorch provides {base} only for spatial ranks 1, 2, and 3; got {rank}"

/-- Render the two-block CNN as a Python `nn.Module` class definition. -/
def generateCnnStackPyTorchClass {spatialRank : Nat}
    (cfg : CnnStackConfig spatialRank) : Except String String := do
  let convClass ← spatialClassName "Conv" spatialRank
  let poolClass ← spatialClassName "MaxPool" spatialRank
  let className := cfg.className
  let tuple := fun (v : Spec.Tensor Nat [spatialRank]) => natListToPyTuple v.toList
  let inputShape := natListToPyTuple (cfg.inputC :: cfg.inputSpatial.toList)
  pure <| joinLines <|
    #[generatePyTorchImports, ""] ++
    #[
      s!"class {className}(nn.Module):",
      indentTwo s!"\"\"\"Two {convClass} / ReLU / {poolClass} blocks, then Flatten / Linear.\"\"\"",
      indentTwo "",
      indentTwo "def __init__(self):",
      indentFour "super().__init__()",
      indentFour (s!"self.conv1 = nn.{convClass}({cfg.conv1.inChannels}, " ++
        s!"{cfg.conv1.outChannels}, kernel_size={tuple cfg.conv1.kernel}, " ++
        s!"stride={tuple cfg.conv1.stride}, padding={tuple cfg.conv1.padding})"),
      indentFour "self.relu1 = nn.ReLU()",
      indentFour (s!"self.pool1 = nn.{poolClass}(kernel_size={tuple cfg.pool1.kernel}, " ++
        s!"stride={tuple cfg.pool1.stride}, padding={tuple cfg.pool1.padding})"),
      indentFour (s!"self.conv2 = nn.{convClass}({cfg.conv2.inChannels}, " ++
        s!"{cfg.conv2.outChannels}, kernel_size={tuple cfg.conv2.kernel}, " ++
        s!"stride={tuple cfg.conv2.stride}, padding={tuple cfg.conv2.padding})"),
      indentFour "self.relu2 = nn.ReLU()",
      indentFour (s!"self.pool2 = nn.{poolClass}(kernel_size={tuple cfg.pool2.kernel}, " ++
        s!"stride={tuple cfg.pool2.stride}, padding={tuple cfg.pool2.padding})"),
      indentFour "self.flatten = nn.Flatten()",
      indentFour s!"self.fc = nn.Linear({cfg.flatSize}, {cfg.fcOut})",
      indentTwo "",
      indentTwo "def forward(self, x):",
      indentFour "x = self.conv1(x)",
      indentFour "x = self.relu1(x)",
      indentFour "x = self.pool1(x)",
      indentFour "x = self.conv2(x)",
      indentFour "x = self.relu2(x)",
      indentFour "x = self.pool2(x)",
      indentFour "x = self.flatten(x)",
      indentFour "x = self.fc(x)",
      indentFour "return x",
      indentTwo "",
      indentTwo "@property",
      indentTwo "def input_shape(self):",
      indentFour s!"return {inputShape}",
      indentTwo "",
      indentTwo "@property",
      indentTwo "def output_shape(self):",
      indentFour s!"return ({cfg.fcOut},)",
      indentTwo "",
      indentTwo "@property",
      indentTwo "def layer_count(self):",
      indentFour "return 8",
      indentTwo "",
      indentTwo "@property",
      indentTwo "def operation_types(self):",
      indentFour (s!"return [\"{convClass}\", \"ReLU\", \"{poolClass}\", \"{convClass}\", " ++
        s!"\"ReLU\", \"{poolClass}\", \"Flatten\", \"Linear\"]"),
      indentTwo ""
    ]
    ++ generateGetModelInfoMethodLines className

/-- Generate a Python CNN module plus a helper that loads explicit weights from string literals.

This is mainly used for examples: you can paste JSON/Lean-rendered weight arrays into Python and run
the model without writing an extra serializer.
-/
def generateCNNWithWeights {spatialRank : Nat} (cfg : CnnStackConfig spatialRank)
    (convW1 convB1 convW2 convB2 linearW linearB : String) : Except String String := do
  let classCode ← generateCnnStackPyTorchClass cfg
  let batchInputShape := natListToPyTuple (1 :: cfg.inputC :: cfg.inputSpatial.toList)
  pure <| joinLines #[
    classCode,
    "",
    "# Weight initialization functions",
    "def get_cnn_state_dict():",
    indentTwo "state_dict = {}",
    indentTwo s!"state_dict['conv1.weight'] = torch.tensor({convW1})",
    indentTwo s!"state_dict['conv1.bias'] = torch.tensor({convB1})",
    indentTwo s!"state_dict['conv2.weight'] = torch.tensor({convW2})",
    indentTwo s!"state_dict['conv2.bias'] = torch.tensor({convB2})",
    indentTwo s!"state_dict['fc.weight'] = torch.tensor({linearW})",
    indentTwo s!"state_dict['fc.bias'] = torch.tensor({linearB})",
    indentTwo "return state_dict",
    indentTwo "",
    "def load_cnn_weights(model):",
    indentTwo "state_dict = get_cnn_state_dict()",
    indentTwo "model.load_state_dict(state_dict)",
    indentTwo "return model",
    indentTwo "",
    "# Usage example",
    "if __name__ == \"__main__\":",
    indentTwo s!"model = {cfg.className}()",
    indentTwo "model = load_cnn_weights(model)",
    indentTwo s!"x = torch.randn{batchInputShape}",
    indentTwo "y = model(x)",
    indentTwo "print(f\"Input shape: {x.shape}\")",
    indentTwo "print(f\"Output shape: {y.shape}\")",
    indentTwo "print(f\"Output: {y}\")",
    indentTwo "print(f\"Model info: {model.get_model_info()}\")"
  ]

end CNNPyTorch
end Export
