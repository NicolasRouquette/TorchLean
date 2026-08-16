/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.Runtime.PyTorch.Export.IRPyTorch
public import NN.Verification.TorchLean.Lowering

/-!
# TorchLean IR to PyTorch

Tutorial: TorchLean → IR (`NN.IR.Graph`) → emitted PyTorch code.

Run:
  `lake exe torchlean torch_ir_pytorch --arch linear > exported_model.py`
  `lake exe torchlean torch_ir_pytorch --arch mlp > exported_model.py`
  `lake exe torchlean torch_ir_pytorch --arch sum > exported_model.py`
  `lake exe torchlean torch_ir_pytorch --arch autoencoder > exported_model.py`
  `lake exe torchlean torch_ir_pytorch --arch mha > exported_model.py`
  `lake exe torchlean torch_ir_pytorch --arch mha-mask > exported_model.py`
  `lake exe torchlean torch_ir_pytorch --arch transformer > exported_model.py`
Then:
  `python3 exported_model.py`
-/

@[expose] public section


namespace NN.Examples.DeepDives.TorchIRPyTorch

open TorchLean

/-! ## Architectures -/

def archLinear : nn.Builder (nn.Sequential (.dim 2 .scalar) (.dim 1 .scalar)) :=
  nn.linear 2 1

def archMLP : nn.Builder (nn.Sequential (.dim 2 .scalar) (.dim 1 .scalar)) :=
  nn.Sequential![
    nn.linear 2 3,
    nn.relu,
    nn.linear 3 1
  ]

def archSumReduce : nn.Builder (nn.Sequential (.dim 4 .scalar) Shape.scalar) :=
  nn.sum (s := .dim 4 .scalar)

def archAutoencoder : nn.Builder (nn.Sequential (.dim 3 .scalar) (.dim 3 .scalar)) :=
  nn.Sequential![
    nn.linear 3 2,
    nn.tanh,
    nn.linear 2 3
  ]

def archCNN : nn.Builder (nn.Sequential (.dim 1 (.dim 1 (.dim 4 (.dim 4 .scalar)))) (shape![1, 3])) :=
  let cfg : nn.models.CNNConfig 2 :=
    { batch := 1
      inChannels := 1
      spatial := #v[4, 4]
      outDim := 3
      conv :=
        { outChannels := 2
          kernel := #v[3, 3]
          kernelNonzero := by intro i; fin_cases i <;> decide
          strideNonzero := by intro i; fin_cases i <;> decide }
      pool :=
        { kernel := #v[1, 1]
          kernelNonzero := by intro i; fin_cases i <;> decide
          strideNonzero := by intro i; fin_cases i <;> decide } }
  by
    simpa [cfg, nn.models.cnnInShape, nn.models.cnnOutShape, Spec.Shape.ofList] using
      nn.models.cnn cfg

def archConvMLP :
    nn.Builder (nn.Sequential (.dim 1 (.dim 1 (.dim 3 (.dim 3 .scalar)))) (shape![1, 1])) :=
  let cfg : nn.models.CNNConfig 2 :=
    { batch := 1
      inChannels := 1
      spatial := #v[3, 3]
      outDim := 1
      conv :=
        { outChannels := 1
          kernel := #v[2, 2]
          kernelNonzero := by intro i; fin_cases i <;> decide
          strideNonzero := by intro i; fin_cases i <;> decide }
      pool :=
        { kernel := #v[1, 1]
          kernelNonzero := by intro i; fin_cases i <;> decide
          strideNonzero := by intro i; fin_cases i <;> decide } }
  by
    simpa [cfg, nn.models.cnnInShape, nn.models.cnnOutShape, Spec.Shape.ofList] using
      nn.models.cnn cfg

def archMHA :
    nn.Builder (nn.Sequential (shape![1, 4, 8]) (shape![1, 4, 8])) :=
  nn.multiheadAttention (batch := 1) (n := 4) (dModel := 8)
    { numHeads := 2, headDim := 4 }

def archMHAMask : Tensor Bool (.dim 4 (.dim 4 .scalar)) :=
  text.causalMask 4

def archMHAMasked :
    nn.Builder (nn.Sequential (shape![1, 4, 8]) (shape![1, 4, 8])) :=
  nn.multiheadAttention (batch := 1) (n := 4) (dModel := 8)
    { numHeads := 2, headDim := 4 } (mask := some archMHAMask)

def archTransformer :
    nn.Builder (nn.Sequential (shape![1, 2, 2]) (shape![1, 2, 2])) :=
  nn.transformerEncoderBlock (batch := 1) (n := 2) (dModel := 2)
    { numHeads := 1
    , headDim := 2
    , ffnHidden := 2 }

/-! ## CLI parsing -/

def usage : String :=
  String.intercalate "\n"
    [ "TorchLean → IR → PyTorch exporter"
    , ""
    , "Usage:"
    , "  lake exe torchlean torch_ir_pytorch --arch linear > exported_model.py"
    , "  lake exe torchlean torch_ir_pytorch --arch mlp > exported_model.py"
    , "  lake exe torchlean torch_ir_pytorch --arch mlp --seed 123 > exported_model.py"
    , "  lake exe torchlean torch_ir_pytorch --arch sum > exported_model.py"
    , "  lake exe torchlean torch_ir_pytorch --arch autoencoder > exported_model.py"
    , "  lake exe torchlean torch_ir_pytorch --arch mha > exported_model.py"
    , "  lake exe torchlean torch_ir_pytorch --arch mha-mask > exported_model.py"
    , "  lake exe torchlean torch_ir_pytorch --arch transformer > exported_model.py"
    , ""
    , "Then: python3 exported_model.py"
    ]

/-! ## Export driver -/

def emitSeq {σ τ : Shape} (className : String) (model : nn.Sequential σ τ) : IO Unit := do
  let ps := nn.stateShapes model
  let prog : _root_.Runtime.Autograd.TorchLean.Program Float (ps ++ [σ]) τ :=
    nn.forward model (α := Float)
  let params := nn.initState (m := model)
  let lowered ←
    match NN.Verification.TorchLean.lowerForwardToIR
        (α := Float) (paramShapes := ps) (inShape := σ) (outShape := τ)
        (model := prog) (params := params) with
    | .error e => throw <| IO.userError e
    | .ok c => pure c

  let code ←
    match Export.IRPyTorch.emit
        (g := lowered.graph) (ps := lowered.ps) (inputId := lowered.inputId) (outputId :=
          lowered.outputId)
        (opts := { className := className }) with
    | .error e => throw <| IO.userError e
    | .ok s => pure s

  IO.println code

def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  let help := args.contains "--help" || args.contains "-h"
  if help then
    IO.println usage
  else
    let (seed, args) ← CLI.orThrow "TorchIRPyTorch" <| CLI.takeSeed args 0
    let (arch, rest) ← CLI.orThrow "TorchIRPyTorch" <|
      CLI.takeFlagValueDefault args "arch" "mlp"
    CLI.requireNoArgs "TorchIRPyTorch" rest
    if arch == "linear" then
      emitSeq (className := "TorchLeanLinear") (nn.build seed archLinear)
    else if arch == "mlp" then
      emitSeq (className := "TorchLeanMLP") (nn.build seed archMLP)
    else if arch == "sum" then
      emitSeq (className := "TorchLeanSumReduce") (nn.build seed archSumReduce)
    else if arch == "autoencoder" then
      emitSeq (className := "TorchLeanAutoencoder") (nn.build seed archAutoencoder)
    else if arch == "cnn" then
      throw <| IO.userError
        "torch_ir_pytorch: --arch cnn is not in the supported exporter fragment yet (conv lowering uses scatter)"
    else if arch == "conv-mlp" then
      throw <| IO.userError
        "torch_ir_pytorch: --arch conv-mlp is not in the supported exporter fragment yet (conv lowering uses scatter)"
    else if arch == "mha" then
      emitSeq (className := "TorchLeanMHA") (nn.build seed archMHA)
    else if arch == "mha-mask" then
      emitSeq (className := "TorchLeanMHAMasked") (nn.build seed archMHAMasked)
    else if arch == "transformer" then
      emitSeq (className := "TorchLeanTransformerBlock") (nn.build seed archTransformer)
    else
      throw <| IO.userError
        (s!"unknown --arch {arch} (supported: linear | mlp | sum | autoencoder | " ++
          s!"mha | mha-mask | transformer)")

end NN.Examples.DeepDives.TorchIRPyTorch
