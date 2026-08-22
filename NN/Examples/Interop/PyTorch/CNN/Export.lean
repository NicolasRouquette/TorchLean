/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Export.Core
public import NN.Spec.Models.Cnn

/-!
# CNN PyTorch Reference Export

PyTorch exporter for the 2-block ConvNet round-trip reference model.

This exporter is meant to mirror the "reference CNN example" shape that shows up in many TorchLean
examples: two `Conv2d + ReLU + MaxPool2d` blocks, then `Flatten`, then a single `Linear` head.

Instead of taking a long positional list of naturals, we use small configuration records so call
sites stay readable and it's easy to extend the shape later.
-/

@[expose] public section


namespace Export
namespace CNNPyTorch

open Spec
open Tensor
open Spec.Module
open Models
open Export.PyTorch

/-- Configuration for a PyTorch `nn.Conv2d` layer. -/
structure Conv2dCfg where
  /-- Input channels (`in_channels`). -/
  inChannels : Nat
  /-- Output channels (`out_channels`). -/
  outChannels : Nat
  /-- Kernel height. -/
  kernelH : Nat
  /-- Kernel width. -/
  kernelW : Nat
  /-- Stride (applied to both spatial dims). -/
  stride : Nat
  /-- Zero-padding (applied to both spatial dims). -/
  padding : Nat

/-- Configuration for a PyTorch `nn.MaxPool2d` layer. -/
structure MaxPool2dCfg where
  /-- Pool kernel height. -/
  kernelH : Nat
  /-- Pool kernel width. -/
  kernelW : Nat
  /-- Pool stride (applied to both spatial dims). -/
  stride : Nat

/-- Configuration for the 2-block CNN exporter. -/
structure CnnStackConfig where
  /-- Class name to use in the generated Python. -/
  className : String := "CNN"
  /-- Input image channels. -/
  inputC : Nat
  /-- Input image height. -/
  inputH : Nat
  /-- Input image width. -/
  inputW : Nat
  /-- conv 1. -/
  conv1 : Conv2dCfg
  /-- pool 1. -/
  pool1 : MaxPool2dCfg
  /-- conv 2. -/
  conv2 : Conv2dCfg
  /-- pool 2. -/
  pool2 : MaxPool2dCfg
  /-- flat Size. -/
  flatSize : Nat
  /-- fc Out. -/
  fcOut : Nat

/-- Render the 2-block CNN as a Python `nn.Module` class definition. -/
def generateCnnStackPyTorchClass (cfg : CnnStackConfig) : String :=
  let className := cfg.className
  joinLines <|
    [generatePyTorchImports, ""] ++
    [
      s!"class {className}(nn.Module):",
      indentTwo "\"\"\"2-block ConvNet (Conv2d → ReLU → MaxPool2d) × 2, then Flatten → Linear.\"\"\"",
      indentTwo "",
      indentTwo "def __init__(self):",
      indentFour "super().__init__()",
      indentFour (s!"self.conv1 = nn.Conv2d({cfg.conv1.inChannels}, " ++
        s!"{cfg.conv1.outChannels}, kernel_size=({cfg.conv1.kernelH}, " ++
        s!"{cfg.conv1.kernelW}), stride={cfg.conv1.stride}, " ++
        s!"padding={cfg.conv1.padding})"),
      indentFour "self.relu1 = nn.ReLU()",
      indentFour (s!"self.pool1 = nn.MaxPool2d(kernel_size=({cfg.pool1.kernelH}, " ++
        s!"{cfg.pool1.kernelW}), stride={cfg.pool1.stride})"),
      indentFour (s!"self.conv2 = nn.Conv2d({cfg.conv2.inChannels}, " ++
        s!"{cfg.conv2.outChannels}, kernel_size=({cfg.conv2.kernelH}, " ++
        s!"{cfg.conv2.kernelW}), stride={cfg.conv2.stride}, " ++
        s!"padding={cfg.conv2.padding})"),
      indentFour "self.relu2 = nn.ReLU()",
      indentFour (s!"self.pool2 = nn.MaxPool2d(kernel_size=({cfg.pool2.kernelH}, " ++
        s!"{cfg.pool2.kernelW}), stride={cfg.pool2.stride})"),
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
      indentFour s!"return ({cfg.inputC}, {cfg.inputH}, {cfg.inputW})",
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
      indentFour ("return [\"Conv2d\", \"ReLU\", \"MaxPool2d\", \"Conv2d\", \"ReLU\", " ++
        "\"MaxPool2d\", \"Flatten\", \"Linear\"]"),
      indentTwo ""
    ]
    ++ generateGetModelInfoMethodLines className

/-- Generate a Python CNN module plus a helper that loads explicit weights from string literals.

This is mainly used for examples: you can paste JSON/Lean-rendered weight arrays into Python and run
the model without writing an extra serializer.
-/
def generateCNNWithWeights (convW1 convB1 convW2 convB2 linearW linearB : String)
    (inC outC inH inW kH kW stride1 padding1 stride2 padding2 poolKH poolKW poolstride1 poolstride2
      flatSize : Nat)
    (className : String := "CNN") : String :=
  let cfg : CnnStackConfig :=
    { className := className
      inputC := inC
      inputH := inH
      inputW := inW
      conv1 := { inChannels := inC, outChannels := outC, kernelH := kH, kernelW := kW, stride :=
        stride1, padding := padding1 }
      pool1 := { kernelH := poolKH, kernelW := poolKW, stride := poolstride1 }
      conv2 := { inChannels := outC, outChannels := outC, kernelH := kH, kernelW := kW, stride :=
        stride2, padding := padding2 }
      pool2 := { kernelH := poolKH, kernelW := poolKW, stride := poolstride2 }
      flatSize := flatSize
      fcOut := outC }
  joinLines [
    generateCnnStackPyTorchClass cfg,
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
    indentTwo s!"model = {className}()",
    indentTwo "model = load_cnn_weights(model)",
    indentTwo s!"x = torch.randn(1, {inC}, {inH}, {inW})  # batch_size=1, channels, height, width",
    indentTwo "y = model(x)",
    indentTwo "print(f\"Input shape: {x.shape}\")",
    indentTwo "print(f\"Output shape: {y.shape}\")",
    indentTwo "print(f\"Output: {y}\")",
    indentTwo "print(f\"Model info: {model.get_model_info()}\")"
  ]

end CNNPyTorch
end Export
