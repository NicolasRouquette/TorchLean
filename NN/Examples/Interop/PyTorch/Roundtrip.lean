/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import NN.Examples.Interop.PyTorch.Export
import NN.Examples.Interop.PyTorch.Import
import NN.API
import NN.Examples.ModelZoo

/-!
# PyTorch Round-Trip Driver

This module assembles the state-dict round-trip examples.

It does **not** re-implement model math. Instead it wires together the existing:

- PyTorch exporters beside the reference examples (`MLP/Export`, `CNN/Export`, `Transformer/Export`)
- PyTorch JSON importers beside the reference examples (`MLP/Import`, `CNN/Import`, `Transformer/Import`)
- Spec models (`NN/Spec/Models/*`) for running a small forward pass in Lean

Run via the TorchLean example runner:

`lake exe torchlean pytorch_roundtrip --model mlp|cnn|transformer --action export|import`

Design goals:
- keep paths/dimensions centralized (no duplicated constants across examples),
- keep the example output deterministic and readable,
- keep this example import-safe (no root-level `main` that collides with other executables).
-/

namespace NN.Examples.Interop.PyTorch.Roundtrip

open Lean
open TorchLean

/-! ## CLI model/action selection -/

/-- Which example model the round-trip driver should export or import. -/
inductive Model where
  | mlp
  | cnn
  | transformer
  deriving Repr, DecidableEq

/-- Parse the `--model` CLI flag accepted by the round-trip example. -/
def Model.parse? (s : String) : Option Model :=
  match s.toLower with
  | "mlp" => some .mlp
  | "cnn" => some .cnn
  | "transformer" => some .transformer
  | _ => none

/-- Which round-trip action to run for the selected example model. -/
inductive Action where
  | export
  | import
  deriving Repr, DecidableEq

/-- Parse the `--action` CLI flag accepted by the round-trip example. -/
def Action.parse? (s : String) : Option Action :=
  match s.toLower with
  | "export" => some .export
  | "import" => some .import
  | _ => none

private def usage : String :=
  String.intercalate "\n"
    [ "PyTorch round-trip example (TorchLean)"
    , ""
    , "Usage:"
    , "  lake exe torchlean pytorch_roundtrip --model mlp|cnn|transformer --action export|import"
    , ""
    , "Notes:"
    , "  - `export` writes readable reference PyTorch modules under `NN/Examples/Interop/PyTorch/<Model>/`."
    , ("  - `import` reads the JSON weights under " ++
      "`NN/Examples/Interop/PyTorch/<Model>/` and runs a Lean forward pass.")
    ]

/-! ## Paths and fixed example dimensions -/

private def dirOf : Model → System.FilePath
  | .mlp => "NN/Examples/Interop/PyTorch/MLP"
  | .cnn => "NN/Examples/Interop/PyTorch/CNN"
  | .transformer => "NN/Examples/Interop/PyTorch/Transformer"

private def jsonOf : Model → System.FilePath
  | .mlp => "NN/Examples/Interop/PyTorch/MLP/mlp.json"
  | .cnn => "NN/Examples/Interop/PyTorch/CNN/cnn.json"
  | .transformer => "NN/Examples/Interop/PyTorch/Transformer/transformer_encoder.json"

-- MLP dims (matches `train_mlp.py` and `Import.MLPPyTorch` example)
private def mlpInDim : Nat := 2
private def mlpHidDim : Nat := 3
private def mlpOutDim : Nat := 1

-- CNN dims/hparams (matches `train_cnn.py` and `Import.CNNPyTorch` example)
private def cnnInC : Nat := 1
private def cnnOutC : Nat := 2
private def cnnInH : Nat := 8
private def cnnInW : Nat := 8
private def cnnKH : Nat := 3
private def cnnKW : Nat := 3
private def cnnStride1 : Nat := 1
private def cnnPadding1 : Nat := 1
private def cnnStride2 : Nat := 1
private def cnnPadding2 : Nat := 1
private def cnnPoolKH : Nat := 2
private def cnnPoolKW : Nat := 2
private def cnnPoolStride1 : Nat := 2
private def cnnPoolStride2 : Nat := 2

private def cnnFlatSize : Nat :=
  _root_.Models.Cnn.featureSize cnnOutC tensor! [cnnInH, cnnInW] tensor! [cnnKH, cnnKW]
    tensor! [cnnStride1, cnnStride1] tensor! [cnnPadding1, cnnPadding1]
    tensor! [cnnStride2, cnnStride2] tensor! [cnnPadding2, cnnPadding2]
    tensor! [cnnPoolKH, cnnPoolKW] tensor! [cnnPoolStride1, cnnPoolStride1] tensor! [0, 0]
    tensor! [cnnPoolStride2, cnnPoolStride2] tensor! [0, 0]

-- Transformer dims (matches `train_transformer.py` and `Import.TransformerPyTorch` example)
private def trSeqLen : Nat := 1
private def trEmbedDim : Nat := 2
private def trHeadCount : Nat := 1
private def trHiddenDim : Nat := 2
private def trNumLayers : Nat := 1

/-! ## Small IO helpers -/

private def writePy (dir : System.FilePath) (base : String) (content : String) : IO Unit := do
  IO.FS.createDirAll dir
  IO.FS.writeFile (dir / s!"{base}.py") content

/-! ## Export actions -/

private def exportMLP : IO Unit := do
  let dir := dirOf .mlp
  let stub := Export.MLPPyTorch.generateCompleteMLPExport (inDim := mlpInDim) (hidDim := mlpHidDim)
    (outDim := mlpOutDim) "TestMLP"
  writePy dir "TestMLP_PyTorch" stub
  -- If we have a JSON state_dict handy, also emit a runnable "with weights" helper.
  try
    let j ← TorchLean.Json.parseFile (jsonOf .mlp)
    let some sd := Import.MLPPyTorch.loadMlpStateDict mlpInDim mlpHidDim mlpOutDim j
      | throw <| IO.userError "MLP JSON present but failed to parse as an MLP state_dict"
    let codeW := Export.MLPPyTorch.generateMLPWithWeights sd.w1 sd.b1 sd.w2 sd.b2 "TestMLP"
    writePy dir "TestMLP_WithWeights" codeW
  catch _ =>
    pure ()
  IO.println "Exported MLP PyTorch files under NN/Examples/Interop/PyTorch/MLP/."

private def exportCNN : IO Unit := do
  let dir := dirOf .cnn
  let conv1 : Export.CNNPyTorch.ConvCfg 2 :=
    { inChannels := cnnInC
      outChannels := cnnOutC
      kernel := tensor! [cnnKH, cnnKW]
      stride := tensor! [cnnStride1, cnnStride1]
      padding := tensor! [cnnPadding1, cnnPadding1] }
  let pool1 : Export.CNNPyTorch.MaxPoolCfg 2 :=
    { kernel := tensor! [cnnPoolKH, cnnPoolKW]
      stride := tensor! [cnnPoolStride1, cnnPoolStride1]
      padding := tensor! [0, 0] }
  let conv2 : Export.CNNPyTorch.ConvCfg 2 :=
    { inChannels := cnnOutC
      outChannels := cnnOutC
      kernel := tensor! [cnnKH, cnnKW]
      stride := tensor! [cnnStride2, cnnStride2]
      padding := tensor! [cnnPadding2, cnnPadding2] }
  let pool2 : Export.CNNPyTorch.MaxPoolCfg 2 :=
    { kernel := tensor! [cnnPoolKH, cnnPoolKW]
      stride := tensor! [cnnPoolStride2, cnnPoolStride2]
      padding := tensor! [0, 0] }
  let cfg : Export.CNNPyTorch.CnnStackConfig 2 :=
    { className := "TestCNN"
      inputC := cnnInC
      inputSpatial := tensor! [cnnInH, cnnInW]
      conv1 := conv1
      pool1 := pool1
      conv2 := conv2
      pool2 := pool2
      flatSize := cnnFlatSize
      fcOut := cnnOutC }
  let stub ←
    match Export.CNNPyTorch.generateCnnStackPyTorchClass cfg with
    | .ok code => pure code
    | .error message => throw <| IO.userError message
  writePy dir "TestCNN_PyTorch" stub
  -- If we have a JSON state_dict handy, also emit a runnable "with weights" helper.
  try
    let j ← TorchLean.Json.parseFile (jsonOf .cnn)
    let some sd := Import.CNNPyTorch.loadCnnStateDict cnnInC cnnOutC cnnKH cnnKW cnnFlatSize j
      | throw <| IO.userError "CNN JSON present but failed to parse as a CNN state_dict"
    let codeW ←
      match Export.CNNPyTorch.generateCNNWithWeights cfg
        (Export.PyTorch.tensorToPyString sd.convW1) (Export.PyTorch.tensorToPyString sd.convB1)
        (Export.PyTorch.tensorToPyString sd.convW2) (Export.PyTorch.tensorToPyString sd.convB2)
        (Export.PyTorch.tensorToPyString sd.linearW) (Export.PyTorch.tensorToPyString sd.linearB)
      with
      | .ok code => pure code
      | .error message => throw <| IO.userError message
    writePy dir "TestCNN_WithWeights" codeW
  catch _ =>
    pure ()
  IO.println "Exported CNN PyTorch files under NN/Examples/Interop/PyTorch/CNN/."

private def exportTransformer : IO Unit := do
  let dir := dirOf .transformer
  let stub :=
    Export.TransformerPyTorch.generateTransformerEncoderPyTorchClass
      trSeqLen trEmbedDim trHeadCount trHiddenDim trNumLayers
      "TestTransformerEncoder"
  writePy dir "TestTransformer_Encoder" stub
  -- If we have a JSON state_dict handy, also emit a runnable "with weights" helper.
  try
    let j ← TorchLean.Json.parseFile (jsonOf .transformer)
    let some sd := Import.TransformerPyTorch.loadTransformerEncoderStateDict trEmbedDim trHeadCount
      trHiddenDim j
      | throw <| IO.userError "Transformer JSON present but failed to parse as a Transformer state_dict"
    let codeW :=
      Export.TransformerPyTorch.generateTransformerEncoderWithWeights
        trSeqLen trEmbedDim trHeadCount trHiddenDim
        sd.queryWeight sd.keyWeight sd.valueWeight sd.outputWeight
        sd.feedForwardInputWeight sd.feedForwardOutputWeight
        sd.feedForwardInputBias sd.feedForwardOutputBias
        sd.norm1Scale sd.norm1Bias sd.norm2Scale sd.norm2Bias
        "TestTransformerEncoder"
    writePy dir "TestTransformer_Encoder_WithWeights" codeW
  catch _ =>
    pure ()
  IO.println "Exported Transformer encoder PyTorch files under NN/Examples/Interop/PyTorch/Transformer/."

private def runExport (m : Model) : IO Unit := do
  match m with
  | .mlp => exportMLP
  | .cnn => exportCNN
  | .transformer => exportTransformer

/-! ## Import actions (Lean forward pass) -/

private def importMLP : IO Unit := do
  let j ← TorchLean.Json.parseFile (jsonOf .mlp)
  let some sd := Import.MLPPyTorch.loadMlpStateDict mlpInDim mlpHidDim mlpOutDim j
    | throw <| IO.userError "Failed to load MLP state dict"

  let x : Tensor Float [mlpInDim] := tensor! [0.5, 0.8]
  let y := Import.MLPPyTorch.forward sd x

  IO.println "== MLP import example =="
  IO.println s!"Loaded: {jsonOf .mlp}"
  IO.println "Output (Lean, Float):"
  TorchLean.Tensor.print y

private def importCNN : IO Unit := do
  let j ← TorchLean.Json.parseFile (jsonOf .cnn)
  let some sd := Import.CNNPyTorch.loadCnnStateDict cnnInC cnnOutC cnnKH cnnKW cnnFlatSize j
    | throw <| IO.userError "Failed to load CNN state dict"

  let hInC : cnnInC ≠ 0 := by decide
  let hOutC : cnnOutC ≠ 0 := by decide
  let hKH : cnnKH ≠ 0 := by decide
  let hKW : cnnKW ≠ 0 := by decide
  let hPoolH : cnnPoolKH ≠ 0 := by decide
  let hPoolW : cnnPoolKW ≠ 0 := by decide
  let hPoolStride1 : cnnPoolStride1 ≠ 0 := by decide
  let hPoolStride2 : cnnPoolStride2 ≠ 0 := by decide

  let conv1 : _root_.Spec.ConvSpec 2 cnnInC cnnOutC tensor! [cnnKH, cnnKW]
      tensor! [cnnStride1, cnnStride1] tensor! [cnnPadding1, cnnPadding1] Float :=
    { kernel := sd.convW1, bias := sd.convB1 }
  let conv2 : _root_.Spec.ConvSpec 2 cnnOutC cnnOutC tensor! [cnnKH, cnnKW]
      tensor! [cnnStride2, cnnStride2] tensor! [cnnPadding2, cnnPadding2] Float :=
    { kernel := sd.convW2, bias := sd.convB2 }
  let pool1 : _root_.Spec.MaxPoolSpec 2 tensor! [cnnPoolKH, cnnPoolKW]
      tensor! [cnnPoolStride1, cnnPoolStride1] tensor! [0, 0]
      (by intro i; fin_cases i <;> assumption)
      (by intro i; fin_cases i <;> assumption) :=
    {}
  let pool2 : _root_.Spec.MaxPoolSpec 2 tensor! [cnnPoolKH, cnnPoolKW]
      tensor! [cnnPoolStride2, cnnPoolStride2] tensor! [0, 0]
      (by intro i; fin_cases i <;> assumption)
      (by intro i; fin_cases i <;> assumption) :=
    {}
  let linear : _root_.Spec.LinearSpec Float cnnFlatSize cnnOutC := { weights := sd.linearW, bias :=
    sd.linearB }

  let spatial : Tensor Nat [2] := tensor! [cnnInH, cnnInW]
  let net :=
    _root_.Models.Cnn.withReluSpec (α := Float)
      (spatial := spatial)
      conv1 conv2 pool1 pool2 linear

  -- Deterministic input matching the Python training script: values 1..64 laid out row-major.
  let x : Tensor Float [cnnInC, cnnInH, cnnInW] :=
    TorchLean.Tensor.generate [cnnInC, cnnInH, cnnInW] fun
      | [_, i, j] => Float.ofNat (i * cnnInW + j + 1)
      | _ => 0.0

  let y := Spec.Module.Chain.forward (α := Float) net (by
    simpa [spatial] using x)

  IO.println "== CNN import example =="
  IO.println s!"Loaded: {jsonOf .cnn}"
  IO.println "Output (Lean, Float):"
  TorchLean.Tensor.print y

private def importTransformer : IO Unit := do
  let j ← TorchLean.Json.parseFile (jsonOf .transformer)
  let some sd := Import.TransformerPyTorch.loadTransformerEncoderStateDict trEmbedDim trHeadCount
    trHiddenDim j
    | throw <| IO.userError "Failed to load Transformer encoder state dict"

  let layer : _root_.Spec.TransformerEncoderLayer trHeadCount trEmbedDim trHiddenDim Float :=
    { mha :=
        { queryWeight := sd.queryWeight
          keyWeight := sd.keyWeight
          valueWeight := sd.valueWeight
          outputWeight := sd.outputWeight }
      ffn :=
        { inputWeight := sd.feedForwardInputWeight
          outputWeight := sd.feedForwardOutputWeight
          inputBias := sd.feedForwardInputBias
          outputBias := sd.feedForwardOutputBias }
      norm1Scale := sd.norm1Scale
      norm1Bias := sd.norm1Bias
      norm2Scale := sd.norm2Scale
      norm2Bias := sd.norm2Bias }
  let encoder : _root_.Spec.TransformerEncoder trNumLayers trHeadCount trEmbedDim trHiddenDim Float
    :=
    { layers := tensor! [layer] }

  let x : Tensor Float [trSeqLen, trEmbedDim] := tensor! [[1.5, 1.5]]
  let y := _root_.Spec.TransformerEncoder.forward (seqLen := trSeqLen) (embedDim := trEmbedDim)
    encoder x (by decide) (by decide)

  IO.println "== Transformer import example =="
  IO.println s!"Loaded: {jsonOf .transformer}"
  IO.println "Output (Lean, Float):"
  TorchLean.Tensor.print y

private def runImport (m : Model) : IO Unit := do
  match m with
  | .mlp => importMLP
  | .cnn => importCNN
  | .transformer => importTransformer

/-! ## Public entrypoint called from the examples zoo runner -/

public def main (args : List String) : IO Unit := do
  let args := _root_.TorchLean.CLI.dropDashDash args
  if _root_.TorchLean.CLI.hasHelp args then
    IO.println usage
    return
  let (modelArg?, args) ← _root_.NN.Examples.ModelZoo.orThrow "PyTorch.Roundtrip" <|
    _root_.TorchLean.CLI.takeFlagValueOnce args "model"
  let (actionArg?, args) ← _root_.NN.Examples.ModelZoo.orThrow "PyTorch.Roundtrip" <|
    _root_.TorchLean.CLI.takeFlagValueOnce args "action"
  _root_.TorchLean.CLI.requireNoArgs "PyTorch.Roundtrip" args

  let modelStr := modelArg?.getD "mlp"
  let actionStr := actionArg?.getD "export"
  let some model := Model.parse? modelStr
    | throw <| IO.userError s!"Unknown --model {modelStr}\n\n{usage}"
  let some action := Action.parse? actionStr
    | throw <| IO.userError s!"Unknown --action {actionStr}\n\n{usage}"

  match action with
  | .export => runExport model
  | .import => runImport model

end NN.Examples.Interop.PyTorch.Roundtrip
