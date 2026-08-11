/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Module.Activation
public import NN.Spec.Module.Linear

/-!
# Export Core

PyTorch code generation helpers.

This module defines shared string-building utilities used by the PyTorch bridge and round-trip
examples. It emits readable Python `nn.Module` code (optionally with weights embedded) and centralizes
the common prelude used by the example MLP/CNN/Transformer exporters under
`NN.Examples.Interop.PyTorch.{MLP,CNN,Transformer}.Export`.

Design note (PyTorch export APIs, for context only):

PyTorch also has *graph capture* / *serialization* mechanisms such as ONNX export and
  `torch.export`.
Those APIs produce IR-like artifacts intended for execution in other runtimes. TorchLean's exporter
in this folder emits auditable Python source for parity checks and round-trip tests.

The public helpers are organized as follows:

- `generatePyTorchImports` / `generatePyTorchHelperModules` provide the shared Python prelude.
- `generateBasePyTorchModule` is the reusable class skeleton for the example exporters.
- `generatePyTorchModule` is the simplest end-to-end exporter for a `SpecChain`.
- `generateCompletePyTorchExport` combines the codegen pieces into a single script.
- `NN.Runtime.PyTorch.Export.StateDict` is the general checkpoint-to-JSON adapter for users who
  already have PyTorch weights.

## References

- PyTorch ONNX export: https://pytorch.org/docs/stable/onnx.html
- PyTorch `torch.export`: https://pytorch.org/docs/stable/export.html
-/

@[expose] public section


namespace Export
namespace PyTorch

open Spec
open Tensor
open ModSpec
open SpecChain

/--
Metadata for a generated PyTorch model snippet.

Most exporters in `NN/Runtime/PyTorch/Export/*` produce a Python string as their final artifact. We keep a
small structured record alongside that string so examples can report shapes/layer counts and decide
whether weights were embedded.
-/
structure PyTorchExportMetadata (α : Type) (s t : Shape) where
  /-- Human-friendly class/model name used in the emitted Python. -/
  modelName : String
  /-- Expected input shape (TorchLean spec `Shape`). -/
  inputShape : Shape
  /-- Expected output shape (TorchLean spec `Shape`). -/
  outputShape : Shape
  /-- Count of primitive layers/ops in the model (exporter-specific). -/
  layerCount : Nat
  /-- A list of operation names used for summary reporting in examples. -/
  operationTypes : List String
  /-- Whether the exporter embedded a `state_dict` literal in the emitted Python. -/
  hasWeights : Bool
  /-- The emitted Python source (usually a full script or class definition). -/
  pytorchCode : String

/-- Join a list of lines with newline separators. -/
def joinLines (xs : List String) : String := String.intercalate "\n" xs
/-- Render a Lean `Bool` as the corresponding Python literal. -/
def pyBool (b : Bool) : String :=
  if b then "True" else "False"
/-- Indent a line by `n` spaces. -/
def indent (n : Nat) (s : String) : String :=
  -- Use a computable definition (Lean's `String.replicate` is `meta` in some imports).
  String.ofList (List.replicate n ' ') ++ s
/-- Indent a line by 2 spaces (common for Python). -/
def indentTwo (s : String) : String := indent 2 s
/-- Indent a line by 4 spaces (common for Python block bodies). -/
def indentFour (s : String) : String := indent 4 s
/-- Indent a line by 6 spaces (used in nested Python blocks). -/
def indentSix (s : String) : String := indent 6 s
/-- Indent a line by 8 spaces (used for nested `nn.Sequential` strings). -/
def indentEight (s : String) : String := indent 8 s

/-!
## Common boilerplate fragments

Many exporters emit the same small pieces of Python: `@property` metadata and a `get_model_info`
dictionary. Shared boilerplate keeps the hand-written example exporters
and the more general IR exporter.
-/

/--
Emit a standard `get_model_info` method used by most TorchLean PyTorch example modules.

`extraFields` are inserted after the `"model_name"` entry. Each element is `(key, valueExpr)` where
`valueExpr` is emitted verbatim as Python code (e.g. `"self.input_shape"` or `"self.hidden_dim"`).
This is meant as a *formatting helper* only; it does not validate Python syntax.
-/
def generateGetModelInfoMethodLines (modelName : String)
    (extraFields : List (String × String) := []) : List String :=
  let items : List (String × String) :=
    ("model_name", s!"\"{modelName}\"") ::
      (extraFields ++
        [ ("input_shape", "self.input_shape")
        , ("output_shape", "self.output_shape")
        , ("layer_count", "self.layer_count")
        , ("operation_types", "self.operation_types")
        ])
  let rec renderItems : List (String × String) → List String
    | [] => []
    | [kv] =>
        let (k, v) := kv
        [indentSix s!"\"{k}\": {v}"]
    | kv :: rest =>
        let (k, v) := kv
        indentSix s!"\"{k}\": {v}," :: renderItems rest
  [ indentTwo "def get_model_info(self) -> dict:"
  , indentFour "return {"
  ] ++ renderItems items ++
  [ indentFour "}"
  ]

/-- Flatten a `Shape` into a list of dimension sizes (outermost-first). -/
def shapeDims : Shape → List Nat
| .scalar => []
| .dim n s => n :: shapeDims s

/--
Render a `Shape` as a Python tuple literal.

Examples:
- `.scalar` becomes `"()"`,
- a 1D shape becomes `"(n,)"` (note the trailing comma),
- higher-rank shapes become `"(d0, d1, ...)"`.
-/
def shapeToPyTupleString (s : Shape) : String :=
  let dims := shapeDims s
  match dims with
  | [] => "()"
  | [n] => s!"({n},)"
  | _ => "(" ++ String.intercalate ", " (dims.map (fun n => toString n)) ++ ")"

/-- Count the number of primitive layers in a `SpecChain`. -/
def countLayers {α : Type} {s t : Shape} : SpecChain α s t → Nat
| .single _ => 1
| .comp a b => countLayers a + countLayers b

/-- Convert a 1D float tensor into a Lean list (outermost dimension order). -/
def vectorTensorToList {n : Nat} (t : Tensor Float (.dim n .scalar)) : List Float :=
  match t with
  | .dim f =>
      (List.finRange n).map fun i =>
        match f i with
        | .scalar x => x

/-- Render a 1D float tensor as a Python list literal (e.g. `[1.0, 2.0]`). -/
def vectorTensorToPy {n : Nat} (t : Tensor Float (.dim n .scalar)) : String :=
  let elems := (vectorTensorToList (n := n) t).map toString
  "[" ++ String.intercalate ", " elems ++ "]"
/-- Render a 2D float tensor as a Python nested-list literal. -/
def matrixTensorToPy {rows cols : Nat} (t : Tensor Float (.dim rows (.dim cols .scalar))) : String :=
  let rowToStr (i : Fin rows) : String :=
    match t with
    | .dim f =>
      match f i with
      | .dim g =>
        let elems := (List.finRange cols).map (fun j =>
          match g j with
          | .scalar x => toString x)
        s!"[" ++ String.intercalate ", " elems ++ "]"
  let rowsStr := (List.finRange rows).map (fun i => rowToStr i)
  s!"[" ++ String.intercalate ", " rowsStr ++ "]"

/--
Render the transpose of a 2D float tensor as a Python nested-list literal.

TorchLean's matrix-valued specs often follow the mathematical convention where a feature matrix
`W` has shape `(in, out)` and is applied as `X * W`. PyTorch stores `nn.Linear` weights as
`(out, in)` and applies them as `X @ W.T + b`. This helper prints a TorchLean matrix in the
transposed orientation expected by PyTorch.
-/
def transposedMatrixTensorToPy {rows cols : Nat} (t : Tensor Float (.dim rows (.dim cols .scalar))) : String :=
  let colToStr (j : Fin cols) : String :=
    match t with
    | .dim f =>
      let elems := (List.finRange rows).map (fun i =>
        match f i with
        | .dim g =>
          match g j with
          | .scalar x => toString x)
      s!"[" ++ String.intercalate ", " elems ++ "]"
  let colsStr := (List.finRange cols).map colToStr
  s!"[" ++ String.intercalate ", " colsStr ++ "]"

/-- Render a 4D float tensor as a Python nested-list literal. -/
def rankFourTensorToPy {a b c d : Nat} (t : Tensor Float (.dim a (.dim b (.dim c (.dim d .scalar))))) :
  String :=
  let toStr (i : Fin a) : String :=
    let toStr2 (j : Fin b) : String :=
      let toStr3 (k : Fin c) : String :=
        let elems := (List.finRange d).map (fun l =>
          match t with
          | .dim f =>
            match f i with
            | .dim g =>
              match g j with
              | .dim h =>
                match h k with
                | .dim m =>
                  match m l with
                  | .scalar x => toString x)
        s!"[" ++ String.intercalate ", " elems ++ "]"
      let elems2 := (List.finRange c).map toStr3
      s!"[" ++ String.intercalate ", " elems2 ++ "]"
    let elems3 := (List.finRange b).map toStr2
    s!"[" ++ String.intercalate ", " elems3 ++ "]"
  let elems4 := (List.finRange a).map toStr
  s!"[" ++ String.intercalate ", " elems4 ++ "]"

/--
Best-effort conversion of a float tensor to a Python list literal.

This is a simple recursive printer used for examples and small regression tests; it is not intended
to be fast.
-/
def tensorToPyString {s : Shape} (t : Tensor Float s) : String :=
  match s with
  | .scalar => toString (toScalar t)
  | .dim n s' =>
    match t with
    | .dim f =>
      let elems := (List.finRange n).map (fun i =>
        match f i with
        | .scalar x => toString x
        | .dim _ => tensorToPyString (f i))
      s!"[" ++ String.intercalate ", " elems ++ "]"

/-- Standard imports used by the generated Python snippets. -/
def generatePyTorchImports : String :=
  joinLines [
    "import torch",
    "import torch.nn as nn",
    "import torch.nn.functional as F",
    "import numpy as np",
    "from typing import Optional, Tuple, List"
  ]

/--
Small helper modules used by some `toPyTorch` strings in the Lean specs.

These are small, dependency-free Python utilities (selectors, wrappers, a compact attention helper)
used so the generated model classes stay short and readable.
-/
def generatePyTorchHelperModules : String :=
  joinLines [
    "",
    "class SelectLast(nn.Module):",
    indentTwo "\"\"\"Select the last timestep from a (batch, seq, hidden) tensor.\"\"\"",
    indentTwo "def forward(self, x):",
    indentFour "return x[:, -1, :]",
    "",
    "class RNNOnlyOutput(nn.Module):",
    indentTwo "def __init__(self, input_size: int, hidden_size: int, **kwargs):",
    indentFour "super().__init__()",
    indentFour "self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, **kwargs)",
    indentTwo "def forward(self, x):",
    indentFour "y, _ = self.rnn(x)",
    indentFour "return y",
    "",
    "class GRUOnlyOutput(nn.Module):",
    indentTwo "def __init__(self, input_size: int, hidden_size: int, **kwargs):",
    indentFour "super().__init__()",
    indentFour "self.gru = nn.GRU(input_size, hidden_size, batch_first=True, **kwargs)",
    indentTwo "def forward(self, x):",
    indentFour "y, _ = self.gru(x)",
    indentFour "return y",
    "",
    "class LSTMOnlyOutput(nn.Module):",
    indentTwo "def __init__(self, input_size: int, hidden_size: int, **kwargs):",
    indentFour "super().__init__()",
    indentFour "self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, **kwargs)",
    indentTwo "def forward(self, x):",
    indentFour "y, _ = self.lstm(x)",
    indentFour "return y",
    "",
    "class RNNClassifier(nn.Module):",
    indentTwo "def __init__(self, input_size: int, hidden_size: int, num_classes: int):",
    indentFour "super().__init__()",
    indentFour "self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)",
    indentFour "self.select = SelectLast()",
    indentFour "self.fc = nn.Linear(hidden_size, num_classes)",
    indentTwo "def forward(self, x):",
    indentFour "y, _ = self.rnn(x)",
    indentFour "y = self.select(y)",
    indentFour "return self.fc(y)",
    "",
    "class GRUClassifier(nn.Module):",
    indentTwo "def __init__(self, input_size: int, hidden_size: int, num_classes: int):",
    indentFour "super().__init__()",
    indentFour "self.gru = nn.GRU(input_size, hidden_size, batch_first=True)",
    indentFour "self.select = SelectLast()",
    indentFour "self.fc = nn.Linear(hidden_size, num_classes)",
    indentTwo "def forward(self, x):",
    indentFour "y, _ = self.gru(x)",
    indentFour "y = self.select(y)",
    indentFour "return self.fc(y)",
    "",
    "class LSTMClassifier(nn.Module):",
    indentTwo "def __init__(self, input_size: int, hidden_size: int, num_classes: int):",
    indentFour "super().__init__()",
    indentFour "self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)",
    indentFour "self.select = SelectLast()",
    indentFour "self.fc = nn.Linear(hidden_size, num_classes)",
    indentTwo "def forward(self, x):",
    indentFour "y, _ = self.lstm(x)",
    indentFour "y = self.select(y)",
    indentFour "return self.fc(y)",
    "",
    "class UnsupportedLayer(nn.Module):",
    indentTwo "def __init__(self, kind: str, detail: str = \"\"):",
    indentFour "super().__init__()",
    indentFour "self.kind = kind",
    indentFour "self.detail = detail",
    indentTwo "def forward(self, x):",
    indentFour "raise NotImplementedError(f\"Unsupported layer: {self.kind} ({self.detail})\")",
    "",
    "class ScaledDotProductSelfAttention(nn.Module):",
    indentTwo "def __init__(self, d_model: int):",
    indentFour "super().__init__()",
    indentFour "self.d_model = d_model",
    indentTwo "def forward(self, x):",
    indentFour "# x: (batch, seq, d_model)",
    indentFour "scores = torch.matmul(x, x.transpose(-2, -1)) / (self.d_model ** 0.5)",
    indentFour "attn = torch.softmax(scores, dim=-1)",
    indentFour "return torch.matmul(attn, x)",
    "",
    "class SimpleRNN(nn.Module):",
    indentTwo ("def __init__(self, input_size: int, hidden_size: int, output_size: " ++
      "int, bidirectional: bool = False):"),
    indentFour "super().__init__()",
    indentFour
      "self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)",
    indentFour "out_dim = hidden_size * (2 if bidirectional else 1)",
    indentFour "self.fc = nn.Linear(out_dim, output_size)",
    indentTwo "def forward(self, x):",
    indentFour "y, _ = self.rnn(x)",
    indentFour "return self.fc(y)",
    "",
    "class SimpleGRU(nn.Module):",
    indentTwo ("def __init__(self, input_size: int, hidden_size: int, output_size: " ++
      "int, bidirectional: bool = False):"),
    indentFour "super().__init__()",
    indentFour
      "self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)",
    indentFour "out_dim = hidden_size * (2 if bidirectional else 1)",
    indentFour "self.fc = nn.Linear(out_dim, output_size)",
    indentTwo "def forward(self, x):",
    indentFour "y, _ = self.gru(x)",
    indentFour "return self.fc(y)",
    "",
    "class SimpleLSTM(nn.Module):",
    indentTwo ("def __init__(self, input_size: int, hidden_size: int, output_size: " ++
      "int, bidirectional: bool = False):"),
    indentFour "super().__init__()",
    indentFour
      "self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)",
    indentFour "out_dim = hidden_size * (2 if bidirectional else 1)",
    indentFour "self.fc = nn.Linear(out_dim, output_size)",
    indentTwo "def forward(self, x):",
    indentFour "y, _ = self.lstm(x)",
    indentFour "return self.fc(y)",
    "",
    "class GRULanguageModel(nn.Module):",
    indentTwo "def __init__(self, vocab_size: int, hidden_size: int):",
    indentFour "super().__init__()",
    indentFour "self.embed = nn.Linear(vocab_size, hidden_size)",
    indentFour "self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)",
    indentFour "self.proj = nn.Linear(hidden_size, vocab_size)",
    indentTwo "def forward(self, x):",
    indentFour "x = self.embed(x)",
    indentFour "y, _ = self.gru(x)",
    indentFour "return self.proj(y)",
    "",
    "class Seq2SeqInference(nn.Module):",
    indentTwo ("def __init__(self, src_vocab_size: int, tgt_vocab_size: int, " ++
      "embed_dim: int, hidden_dim: int, max_tgt_len: int, start_token: int = " ++
      "0):"),
    indentFour "super().__init__()",
    indentFour "self.src_embed = nn.Linear(src_vocab_size, embed_dim)",
    indentFour "self.tgt_embed = nn.Embedding(tgt_vocab_size, embed_dim)",
    indentFour "self.encoder = nn.RNN(embed_dim, hidden_dim, batch_first=True)",
    indentFour "self.decoder = nn.RNN(embed_dim, hidden_dim, batch_first=True)",
    indentFour "self.proj = nn.Linear(hidden_dim, tgt_vocab_size)",
    indentFour "self.max_tgt_len = max_tgt_len",
    indentFour "self.start_token = start_token",
    indentTwo "def forward(self, x):",
    indentFour "# x: (batch, src_len, src_vocab_size) one-hot/dists",
    indentFour "x = self.src_embed(x)",
    indentFour "_enc_out, h = self.encoder(x)",
    indentFour "batch = x.shape[0]",
    indentFour "token = torch.full((batch,), self.start_token, dtype=torch.long, device=x.device)",
    indentFour "inp = self.tgt_embed(token).unsqueeze(1)",
    indentFour "logits = []",
    indentFour "h_dec = h",
    indentFour "for _ in range(self.max_tgt_len):",
    indentSix "y, h_dec = self.decoder(inp, h_dec)",
    indentSix "step = self.proj(y.squeeze(1))",
    indentSix "logits.append(step)",
    indentSix "token = torch.argmax(step, dim=-1)",
    indentSix "inp = self.tgt_embed(token).unsqueeze(1)",
    indentFour "return torch.stack(logits, dim=1)",
  ]

/--
Generate a generic base `nn.Module` class skeleton.

This is used by exporters that want a "real" class with an explicit `_initialize_layers` hook,
instead of the simpler `nn.Sequential` emitter.
-/
def generateBasePyTorchModule (className : String) (docstring : String) : String :=
  joinLines <|
    [ s!"class {className}(nn.Module):"
    , indentTwo s!"\"\"\"{docstring}\"\"\""
    , indentTwo ""
    , indentTwo "def __init__(self):"
    , indentFour "super().__init__()"
    , indentFour "self._initialize_layers()"
    , indentFour ""
    , indentFour "def _initialize_layers(self):"
    , indentSix "raise NotImplementedError(\"Subclasses must implement _initialize_layers\")"
    , indentFour ""
    , indentTwo "def forward(self, x):"
    , indentFour "raise NotImplementedError(\"Subclasses must implement forward\")"
    , indentFour ""
    ] ++
      generateGetModelInfoMethodLines className ++
      [ indentFour ""
      , indentTwo "@property"
      , indentTwo "def input_shape(self):"
      , indentFour "raise NotImplementedError(\"Subclasses must implement input_shape\")"
      , indentFour ""
      , indentTwo "@property"
      , indentTwo "def output_shape(self):"
      , indentFour "raise NotImplementedError(\"Subclasses must implement output_shape\")"
      , indentFour ""
      , indentTwo "@property"
      , indentTwo "def layer_count(self):"
      , indentFour "raise NotImplementedError(\"Subclasses must implement layer_count\")"
      , indentFour ""
      , indentTwo "@property"
      , indentTwo "def operation_types(self):"
      , indentFour "raise NotImplementedError(\"Subclasses must implement operation_types\")"
      ]

/-- Emit Python helpers for saving/loading state dictionaries and JSON checkpoints. -/
def generateWeightLoadingUtils : String :=
  joinLines [
    "def load_weights_from_dict(model: nn.Module, state_dict: dict):",
    indentTwo "\"\"\"Load weights from a state dictionary into the model.\"\"\"",
    indentTwo "model.load_state_dict(state_dict)",
    indentTwo "return model",
    "",
    "def save_weights_to_dict(model: nn.Module) -> dict:",
    indentTwo "\"\"\"Save model weights to a state dictionary.\"\"\"",
    indentTwo "return model.state_dict()",
    "",
    "def save_model_to_file(model: nn.Module, filepath: str):",
    indentTwo "\"\"\"Save model weights to a file as a state_dict checkpoint.\"\"\"",
    indentTwo "torch.save(model.state_dict(), filepath)",
    "",
    "def load_model_from_file(filepath: str, model: Optional[nn.Module] = None):",
    indentTwo "\"\"\"Load a state_dict checkpoint; optionally materialize it into `model`.\"\"\"",
    indentTwo "state_dict = torch.load(filepath, weights_only=True)",
    indentTwo "if model is None:",
    indentTwo "    return state_dict",
    indentTwo "model.load_state_dict(state_dict)",
    indentTwo "return model"
  ]

/-- Emit Python helpers for validating exported models. -/
def generateTestingUtils : String :=
  joinLines [
    "def test_model_forward(model: nn.Module, input_shape: Tuple[int, ...], num_tests: int = 5):",
    indentTwo "\"\"\"Test model forward pass with random inputs.\"\"\"",
    indentTwo "model.eval()",
    indentTwo "with torch.no_grad():",
    indentFour "for i in range(num_tests):",
    indentSix "x = torch.randn(1, *input_shape)",
    indentSix "y = model(x)",
    indentSix "print(f\"Test {i+1}: Input shape: {x.shape}, Output shape: {y.shape}\")",
    indentSix "print(f\"Output range: [{y.min().item():.4f}, {y.max().item():.4f}]\")",
    "",
    "def count_parameters(model: nn.Module) -> int:",
    indentTwo "\"\"\"Count the number of trainable parameters in the model.\"\"\"",
    indentTwo "return sum(p.numel() for p in model.parameters() if p.requires_grad)",
    "",
    "def print_model_summary(model: nn.Module):",
    indentTwo "\"\"\"Print a summary of the model architecture.\"\"\"",
    indentTwo "print(f\"Model: {model.__class__.__name__}\")",
    indentTwo "print(f\"Total parameters: {count_parameters(model):,}\")",
    indentTwo "print(f\"Model info: {model.get_model_info()}\")"
  ]

/--
Generate a complete `nn.Sequential`-based Python module for a `SpecChain`.

This is the simplest exporter: we extract a list of `(opName, pythonLayerString)` pairs and drop
them into an `nn.Sequential(...)` in a new class.
-/
def generatePyTorchModule {α : Type} {s t : Shape}
  (chain : SpecChain α s t) (className : String := "ExportedModel") : String :=
  let inputShape := shapeToPyTupleString s
  let outputShape := shapeToPyTupleString t
  let layerCount := countLayers chain
  let layers := SpecChain.extractLayerInfo chain
  let layerStrings := layers.map (fun (_, pytorch) => indentEight pytorch)
  let opList := "[" ++ String.intercalate ", " (layers.map (fun (op, _) => s!"\"{op}\"")) ++ "]"

  joinLines <|
    [ generatePyTorchImports
    , generatePyTorchHelperModules
    , ""
    , s!"class {className}(nn.Module):"
    , indentTwo "def __init__(self):"
    , indentFour "super().__init__()"
    , indentFour s!"# Input shape: {inputShape}"
    , indentFour s!"# Output shape: {outputShape}"
    , indentFour s!"# Layer count: {layerCount}"
    , indentFour s!"# Operations: {String.intercalate ", " (layers.map (fun (op, _) => op))}"
    , indentFour ""
    , indentFour "self.layers = nn.Sequential("
    , String.intercalate ",\n" layerStrings
    , indentFour ")"
    , ""
    , indentTwo "def forward(self, x):"
    , indentFour "return self.layers(x)"
    , indentFour ""
    , indentTwo "@property"
    , indentTwo "def input_shape(self):"
    , indentFour s!"return {inputShape}"
    , indentFour ""
    , indentTwo "@property"
    , indentTwo "def output_shape(self):"
    , indentFour s!"return {outputShape}"
    , indentFour ""
    , indentTwo "@property"
    , indentTwo "def layer_count(self):"
    , indentFour s!"return {layerCount}"
    , indentFour ""
    , indentTwo "@property"
    , indentTwo "def operation_types(self):"
    , indentFour s!"return {opList}"
    , indentFour ""
    ] ++
      generateGetModelInfoMethodLines className

/-- Like `generateBasePyTorchModule`, but also include shared weight/testing helpers. -/
def generateCompletePyTorchExport (className : String) (docstring : String) : String :=
  joinLines [
    generatePyTorchImports,
    "",
    generateBasePyTorchModule className docstring,
    "",
    generateWeightLoadingUtils,
    "",
    generateTestingUtils,
    "",
    "# Usage example:",
    "# model = YourModelClass()",
    "# test_model_forward(model, input_shape=(your_input_dims))",
    "# print_model_summary(model)"
  ]

/--
Export a `SpecChain` to PyTorch and bundle the result in a metadata record.

This is the "one call" entrypoint used by some examples.
-/
def exportGeneralModel {α : Type} {s t : Shape}
  (chain : SpecChain α s t) (className : String := "GeneralModel") : PyTorchExportMetadata α s t :=
  {
    modelName := className,
    inputShape := s,
    outputShape := t,
    layerCount := countLayers chain,
    operationTypes := (extractLayerInfo chain).map (fun (op, _) => op),
    hasWeights := false,
    pytorchCode := generatePyTorchModule chain className
  }

end PyTorch
end Export
