/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Export.Core
public import NN.Spec.Models.Mlp

/-!
# MLP PyTorch Reference Export

PyTorch code generator for the MLP round-trip reference model.

The generated Python mirrors the common `nn.Linear → ReLU → nn.Linear` pattern. We also support
embedding explicit weights into a `state_dict`-shaped dictionary for round-trip and regression
checks.
-/

@[expose] public section


namespace Export
namespace MLPPyTorch

open Spec
open Tensor
open Spec.Module
open Spec.Module.Chain
open Examples
open Export.PyTorch

/-- How to name `state_dict` keys when exporting weights. -/
inductive WeightKeyStyle where
  /-- Keys like `fc1.weight` / `fc2.bias` (matches PyTorch `nn.Linear` modules). -/
  | linear
  /-- Keys like `layers.0.weight` / `layers.2.bias` (common when exporting `nn.Sequential`). -/
  | sequential
  deriving DecidableEq, Repr

/-- Key name for the first layer's weight tensor in a PyTorch `state_dict`. -/
def firstWeightKey : WeightKeyStyle → String
  | .linear => "fc1.weight"
  | .sequential => "layers.0.weight"

/-- Key name for the first layer's bias tensor in a PyTorch `state_dict`. -/
def firstBiasKey : WeightKeyStyle → String
  | .linear => "fc1.bias"
  | .sequential => "layers.0.bias"

/-- Key name for the second layer's weight tensor in a PyTorch `state_dict`. -/
def secondWeightKey : WeightKeyStyle → String
  | .linear => "fc2.weight"
  | .sequential => "layers.2.weight"

/-- Key name for the second layer's bias tensor in a PyTorch `state_dict`. -/
def secondBiasKey : WeightKeyStyle → String
  | .linear => "fc2.bias"
  | .sequential => "layers.2.bias"

/--
Metadata for an exported MLP.

This is a small “record of facts” about the generated Python: shapes, dimensions, and optionally
embedded weights for round-trip tests.
-/
structure MLPExportMetadata (α : Type) (inDim hidDim outDim : Nat) where
  /-- Python class name used in the generated module. -/
  modelName : String
  /-- Input feature dimension. -/
  inputDim : Nat
  /-- Hidden layer dimension. -/
  hiddenDim : Nat
  /-- Output feature dimension. -/
  outputDim : Nat
  /-- Whether the generated Python should embed concrete weights. -/
  hasWeights : Bool
  /-- Optional two-layer MLP parameter payload: weights and biases for both linear layers. -/
  weights : Option (Tensor Float (.dim hidDim (.dim inDim .scalar)) ×
                   Tensor Float (.dim hidDim .scalar) ×
                   Tensor Float (.dim outDim (.dim hidDim .scalar)) ×
                   Tensor Float (.dim outDim .scalar))

/--
Emit the Python class body for a basic `Linear → ReLU → Linear` MLP.

This returns *lines* (not a single string) so callers can splice it into larger scripts.
-/
def generateMLPPyTorchClassLines (inDim hidDim outDim : Nat) (className : String) : List String :=
  [
    s!"class {className}(nn.Module):",
    indentTwo (s!"\"\"\"Multi-Layer Perceptron with {inDim} input, {hidDim} hidden, " ++
      s!"{outDim} output dimensions\"\"\""),
    indentTwo "",
    indentTwo (s!"def __init__(self, input_dim: int = {inDim}, hidden_dim: int = " ++
      s!"{hidDim}, output_dim: int = {outDim}):"),
    indentFour "super().__init__()",
    indentFour "self.input_dim = input_dim",
    indentFour "self.hidden_dim = hidden_dim",
    indentFour "self.output_dim = output_dim",
    indentFour "",
    indentFour "# Define layers",
    indentFour "self.fc1 = nn.Linear(input_dim, hidden_dim)",
    indentFour "self.relu = nn.ReLU()",
    indentFour "self.fc2 = nn.Linear(hidden_dim, output_dim)",
    indentFour "",
    indentTwo "def forward(self, x):",
    indentFour "x = self.fc1(x)",
    indentFour "x = self.relu(x)",
    indentFour "x = self.fc2(x)",
    indentFour "return x",
    indentFour "",
    indentTwo "@property",
    indentTwo "def input_shape(self):",
    indentFour s!"return ({inDim},)",
    indentFour "",
    indentTwo "@property",
    indentTwo "def output_shape(self):",
    indentFour s!"return ({outDim},)",
    indentFour "",
    indentTwo "@property",
    indentTwo "def layer_count(self):",
    indentFour "return 3",  -- fc1, relu, fc2
    indentFour "",
    indentTwo "@property",
    indentTwo "def operation_types(self):",
    indentFour "return [\"Linear\", \"ReLU\", \"Linear\"]",
    indentFour ""
  ] ++
    generateGetModelInfoMethodLines className
      [ ("input_dim", "self.input_dim")
      , ("hidden_dim", "self.hidden_dim")
      , ("output_dim", "self.output_dim")
      ]

/-- Generate a standalone Python file containing an `nn.Module` MLP class. -/
def generateMLPPyTorchClass (inDim hidDim outDim : Nat) (className : String := "MLP") : String :=
  joinLines <|
    [generatePyTorchImports, ""] ++ generateMLPPyTorchClassLines inDim hidDim outDim className

/--
Generate Python code for an MLP plus helper functions that embed concrete weights.

The output contains a `get_mlp_state_dict` function that returns a PyTorch-shaped dictionary
(`state_dict`) and a `load_mlp_weights` helper that calls `model.load_state_dict(...)`.
-/
def generateMLPWithWeights {inDim hidDim outDim : Nat}
  (w1 : Tensor Float (.dim hidDim (.dim inDim .scalar)))
  (b1 : Tensor Float (.dim hidDim .scalar))
  (w2 : Tensor Float (.dim outDim (.dim hidDim .scalar)))
  (b2 : Tensor Float (.dim outDim .scalar))
  (className : String := "MLP")
  (keyStyle : WeightKeyStyle := .linear) : String :=
  joinLines [
    generateMLPPyTorchClass inDim hidDim outDim className,
    "",
    "# Weight initialization functions",
    "def get_mlp_state_dict():",
    indentTwo "state_dict = {}",
    indentTwo s!"state_dict['{firstWeightKey keyStyle}'] = torch.tensor({matrixTensorToPy w1})",
    indentTwo s!"state_dict['{firstBiasKey keyStyle}'] = torch.tensor({vectorTensorToPy b1})",
    indentTwo s!"state_dict['{secondWeightKey keyStyle}'] = torch.tensor({matrixTensorToPy w2})",
    indentTwo s!"state_dict['{secondBiasKey keyStyle}'] = torch.tensor({vectorTensorToPy b2})",
    indentTwo "return state_dict",
    indentTwo "",
    "def load_mlp_weights(model):",
    indentTwo "state_dict = get_mlp_state_dict()",
    indentTwo "model.load_state_dict(state_dict)",
    indentTwo "return model",
    indentTwo "",
    "# Usage example",
    "if __name__ == \"__main__\":",
    indentTwo s!"model = {className}()",
    indentTwo "model = load_mlp_weights(model)",
    indentTwo s!"x = torch.randn(1, {inDim})  # batch_size=1, features={inDim}",
    indentTwo "y = model(x)",
    indentTwo "print(f\"Input shape: {x.shape}\")",
    indentTwo "print(f\"Output shape: {y.shape}\")",
    indentTwo "print(f\"Output: {y}\")",
    indentTwo "print(f\"Model info: {model.get_model_info()}\")"
  ]

/--
Generate a Python MLP class that ends with a Softmax (classification convenience).

This mirrors common compact PyTorch model code; full training pipelines usually use logits + a combined
loss (e.g. `CrossEntropyLoss`) instead of an explicit softmax.
-/
def generateMLPWithSoftmax {inDim hidDim outDim : Nat} (className : String := "MLPWithSoftmax") :
  String :=
  joinLines <|
  [generatePyTorchImports, ""] ++ [
    s!"class {className}(nn.Module):",
    indentTwo s!"\"\"\"Multi-Layer Perceptron with softmax output for classification\"\"\"",
    indentTwo "",
    indentTwo (s!"def __init__(self, input_dim: int = {inDim}, hidden_dim: int = " ++
      s!"{hidDim}, output_dim: int = {outDim}):"),
    indentFour "super().__init__()",
    indentFour "self.input_dim = input_dim",
    indentFour "self.hidden_dim = hidden_dim",
    indentFour "self.output_dim = output_dim",
    indentFour "",
    indentFour "# Define layers",
    indentFour "self.fc1 = nn.Linear(input_dim, hidden_dim)",
    indentFour "self.relu = nn.ReLU()",
    indentFour "self.fc2 = nn.Linear(hidden_dim, output_dim)",
    indentFour "self.softmax = nn.Softmax(dim=1)",
    indentFour "",
    indentTwo "def forward(self, x):",
    indentFour "x = self.fc1(x)",
    indentFour "x = self.relu(x)",
    indentFour "x = self.fc2(x)",
    indentFour "x = self.softmax(x)",
    indentFour "return x",
    indentFour "",
    indentTwo "@property",
    indentTwo "def input_shape(self):",
    indentFour s!"return ({inDim},)",
    indentFour "",
    indentTwo "@property",
    indentTwo "def output_shape(self):",
    indentFour s!"return ({outDim},)",
    indentFour "",
    indentTwo "@property",
    indentTwo "def layer_count(self):",
    indentFour "return 4",  -- fc1, relu, fc2, softmax
    indentFour "",
    indentTwo "@property",
    indentTwo "def operation_types(self):",
    indentFour "return [\"Linear\", \"ReLU\", \"Linear\", \"Softmax\"]"
  ]

/-- Line-based version of `generateMLPWithSoftmax` for script composition. -/
def generateMLPWithSoftmaxLines {inDim hidDim outDim : Nat} (className : String) : List String :=
  [
    s!"class {className}(nn.Module):",
    indentTwo s!"\"\"\"Multi-Layer Perceptron with softmax output for classification\"\"\"",
    indentTwo "",
    indentTwo (s!"def __init__(self, input_dim: int = {inDim}, hidden_dim: int = " ++
      s!"{hidDim}, output_dim: int = {outDim}):"),
    indentFour "super().__init__()",
    indentFour "self.input_dim = input_dim",
    indentFour "self.hidden_dim = hidden_dim",
    indentFour "self.output_dim = output_dim",
    indentFour "",
    indentFour "# Define layers",
    indentFour "self.fc1 = nn.Linear(input_dim, hidden_dim)",
    indentFour "self.relu = nn.ReLU()",
    indentFour "self.fc2 = nn.Linear(hidden_dim, output_dim)",
    indentFour "self.softmax = nn.Softmax(dim=1)",
    indentFour "",
    indentTwo "def forward(self, x):",
    indentFour "x = self.fc1(x)",
    indentFour "x = self.relu(x)",
    indentFour "x = self.fc2(x)",
    indentFour "x = self.softmax(x)",
    indentFour "return x",
    indentFour "",
    indentTwo "@property",
    indentTwo "def input_shape(self):",
    indentFour s!"return ({inDim},)",
    indentFour "",
    indentTwo "@property",
    indentTwo "def output_shape(self):",
    indentFour s!"return ({outDim},)",
    indentFour "",
    indentTwo "@property",
    indentTwo "def layer_count(self):",
    indentFour "return 4",  -- fc1, relu, fc2, softmax
    indentFour "",
    indentTwo "@property",
    indentTwo "def operation_types(self):",
    indentFour "return [\"Linear\", \"ReLU\", \"Linear\", \"Softmax\"]"
  ]

/--
Export metadata for an MLP described as a `Spec.Module.Chain`.

The chain is accepted to keep the API aligned with `Spec.Module.Chain`, while this exporter produces
metadata from the explicit dimensions supplied by the type parameters.
-/
def exportMLPFromChain {α : Type} {inDim hidDim outDim : Nat}
  (_chain : Spec.Module.Chain α (.dim inDim .scalar) (.dim outDim .scalar))
  (className : String := "ExportedMLP") : MLPExportMetadata α inDim hidDim outDim :=
  {
    modelName := className,
    inputDim := inDim,
    hiddenDim := hidDim,
    outputDim := outDim,
    hasWeights := false,
    weights := none
  }

/-- Like `exportMLPFromChain`, but include explicit weights in the metadata record. -/
def exportMLPWithWeights {α : Type} {inDim hidDim outDim : Nat}
  (_chain : Spec.Module.Chain α (.dim inDim .scalar) (.dim outDim .scalar))
  (w1 : Tensor Float (.dim hidDim (.dim inDim .scalar)))
  (b1 : Tensor Float (.dim hidDim .scalar))
  (w2 : Tensor Float (.dim outDim (.dim hidDim .scalar)))
  (b2 : Tensor Float (.dim outDim .scalar))
  (className : String := "ExportedMLP") : MLPExportMetadata α inDim hidDim outDim :=
  {
    modelName := className,
    inputDim := inDim,
    hiddenDim := hidDim,
    outputDim := outDim,
    hasWeights := true,
    weights := some (w1, b1, w2, b2)
  }

/--
Generate a complete Python script for MLP examples.

This includes:
- a base MLP class,
- a Softmax variant,
- shared helper modules from `NN/Runtime/PyTorch/Export/Core.lean`,
- and convenience helpers for construction and parameter counting.
-/
def generateCompleteMLPExport {inDim hidDim outDim : Nat}
  (className : String := "MLP") : String :=
  joinLines [
    generatePyTorchImports,
    "",
    joinLines (generateMLPPyTorchClassLines inDim hidDim outDim className),
    "",
    joinLines (generateMLPWithSoftmaxLines (inDim := inDim) (hidDim := hidDim) (outDim := outDim)
      s!"{className}WithSoftmax"),
    "",
    generateWeightLoadingUtils,
    "",
    generateTestingUtils,
    "",
    "# MLP-specific utilities",
    ("def create_mlp_from_spec(input_dim: int, hidden_dim: int, output_dim: " ++
      "int, use_softmax: bool = False):"),
    indentTwo "\"\"\"Create an MLP model from specifications.\"\"\"",
    indentTwo "if use_softmax:",
      indentFour s!"return {className}WithSoftmax(input_dim, hidden_dim, output_dim)",
    indentTwo "else:",
      indentFour s!"return {className}(input_dim, hidden_dim, output_dim)",
    indentTwo "",
    "def mlp_parameter_count(input_dim: int, hidden_dim: int, output_dim: int) -> int:",
    indentTwo "\"\"\"Calculate the number of parameters in an MLP.\"\"\"",
    indentTwo "return input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + output_dim"
  ]

end MLPPyTorch
end Export
