/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Lean.Data.Json
public import NN.API.Json
public import NN.Entrypoint.Tensor
public import NN.Examples.Interop.PyTorch.CNN.Import
public import NN.Examples.Interop.PyTorch.MLP.Import
public import NN.Examples.Interop.PyTorch.Transformer.Import
public import NN.Runtime.External.Process
public import NN.Spec.Models.Cnn
public import NN.Tests.Runtime.Floats.Utils

/-!
# PyTorch Roundtrip Parity Checks

Compares the checked-in PyTorch reference weights against TorchLean evaluation for the same fixed
inputs used by the round-trip examples.
-/

@[expose] public section

open Lean
open Spec
open Tensor
open Tests.Floats.Utils

namespace Tests
namespace Floats
namespace PyTorchRoundtripParity

def workDir : System.FilePath :=
  Runtime.External.Process.artifactWorkDir "pytorch_roundtrip_parity"

def parityScriptPath : System.FilePath :=
  workDir / "compute_parity.py"

def mlpJson : System.FilePath :=
  "NN/Examples/Interop/PyTorch/MLP/mlp.json"

def cnnJson : System.FilePath :=
  "NN/Examples/Interop/PyTorch/CNN/cnn.json"

def transformerJson : System.FilePath :=
  "NN/Examples/Interop/PyTorch/Transformer/transformer_encoder.json"

def parityScript : String :=
  String.intercalate "\n"
    [ "import json"
    , "import math"
    , "from pathlib import Path"
    , "import torch"
    , "import torch.nn as nn"
    , "import torch.nn.functional as F"
    , ""
    , "ROOT = Path('NN/Examples/Interop/PyTorch')"
    , ""
    , "def params(path):"
    , "    return json.loads(path.read_text())['params']"
    , ""
    , "def run_mlp():"
    , "    p = params(ROOT / 'MLP/mlp.json')"
    , "    fc1 = nn.Linear(2, 3)"
    , "    fc2 = nn.Linear(3, 1)"
    , "    with torch.no_grad():"
    , "        fc1.weight.copy_(torch.tensor(p['layers.0.weight'], dtype=torch.float32))"
    , "        fc1.bias.copy_(torch.tensor(p['layers.0.bias'], dtype=torch.float32))"
    , "        fc2.weight.copy_(torch.tensor(p['layers.2.weight'], dtype=torch.float32))"
    , "        fc2.bias.copy_(torch.tensor(p['layers.2.bias'], dtype=torch.float32))"
    , "        x = torch.tensor([[0.5, 0.8]], dtype=torch.float32)"
    , "        return fc2(F.relu(fc1(x))).flatten().tolist()"
    , ""
    , "def run_cnn():"
    , "    p = params(ROOT / 'CNN/cnn.json')"
    , "    conv1 = nn.Conv2d(1, 2, 3, padding=1)"
    , "    conv2 = nn.Conv2d(2, 2, 3, padding=1)"
    , "    fc = nn.Linear(8, 2)"
    , "    with torch.no_grad():"
    , "        conv1.weight.copy_(torch.tensor(p['conv1.weight'], dtype=torch.float32))"
    , "        conv1.bias.copy_(torch.tensor(p['conv1.bias'], dtype=torch.float32))"
    , "        conv2.weight.copy_(torch.tensor(p['conv2.weight'], dtype=torch.float32))"
    , "        conv2.bias.copy_(torch.tensor(p['conv2.bias'], dtype=torch.float32))"
    , "        fc.weight.copy_(torch.tensor(p['fc.weight'], dtype=torch.float32))"
    , "        fc.bias.copy_(torch.tensor(p['fc.bias'], dtype=torch.float32))"
    , "        x = torch.arange(1, 65, dtype=torch.float32).reshape(1, 1, 8, 8)"
    , "        x = F.max_pool2d(F.relu(conv1(x)), 2, 2)"
    , "        x = F.max_pool2d(F.relu(conv2(x)), 2, 2)"
    , "        return fc(x.reshape(x.shape[0], -1)).flatten().tolist()"
    , ""
    , "class TinyMHA(nn.Module):"
    , "    def __init__(self, p):"
    , "        super().__init__()"
    , "        self.head_dim = 2"
    , "        self.q = nn.Linear(2, 2, bias=False)"
    , "        self.k = nn.Linear(2, 2, bias=False)"
    , "        self.v = nn.Linear(2, 2, bias=False)"
    , "        self.o = nn.Linear(2, 2, bias=False)"
    , "        with torch.no_grad():"
    , "            self.q.weight.copy_(torch.tensor(p['Wq'], dtype=torch.float32).t())"
    , "            self.k.weight.copy_(torch.tensor(p['Wk'], dtype=torch.float32).t())"
    , "            self.v.weight.copy_(torch.tensor(p['Wv'], dtype=torch.float32).t())"
    , "            self.o.weight.copy_(torch.tensor(p['Wo'], dtype=torch.float32).t())"
    , "    def forward(self, x):"
    , "        q, k, v = self.q(x), self.k(x), self.v(x)"
    , "        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.head_dim), dim=-1)"
    , "        return self.o(attn @ v)"
    , ""
    , "def run_transformer():"
    , "    p = params(ROOT / 'Transformer/transformer_encoder.json')"
    , "    mha = TinyMHA(p)"
    , "    norm1 = nn.LayerNorm(2)"
    , "    fc1 = nn.Linear(2, 2)"
    , "    fc2 = nn.Linear(2, 2)"
    , "    norm2 = nn.LayerNorm(2)"
    , "    with torch.no_grad():"
    , "        fc1.weight.copy_(torch.tensor(p['W1'], dtype=torch.float32).t())"
    , "        fc1.bias.copy_(torch.tensor(p['b1'], dtype=torch.float32))"
    , "        fc2.weight.copy_(torch.tensor(p['W2'], dtype=torch.float32).t())"
    , "        fc2.bias.copy_(torch.tensor(p['b2'], dtype=torch.float32))"
    , "        norm1.weight.copy_(torch.tensor(p['norm1_gamma'], dtype=torch.float32))"
    , "        norm1.bias.copy_(torch.tensor(p['norm1_beta'], dtype=torch.float32))"
    , "        norm2.weight.copy_(torch.tensor(p['norm2_gamma'], dtype=torch.float32))"
    , "        norm2.bias.copy_(torch.tensor(p['norm2_beta'], dtype=torch.float32))"
    , "        x = torch.full((1, 1, 2), 1.5, dtype=torch.float32)"
    , "        x = norm1(x + mha(x))"
    , "        return norm2(x + fc2(F.relu(fc1(x)))).flatten().tolist()"
    , ""
    , "print(json.dumps({'mlp': run_mlp(), 'cnn': run_cnn(), 'transformer': run_transformer()}))"
    ]

def leanMlp : IO (Array Float) := do
  let j ← NN.API.Json.parseFile mlpJson
  let some sd := Import.MLPPyTorch.loadMlpStateDict 2 3 1 j
    | throw (IO.userError "pytorch_roundtrip_parity: failed to load MLP state dict")
  let x : Tensor Float (.dim 2 .scalar) := tensor! [0.5, 0.8]
  let y := Import.MLPPyTorch.forward sd x
  pure #[vecVal y ⟨0, by decide⟩]

def leanCnn : IO (Array Float) := do
  let j ← NN.API.Json.parseFile cnnJson
  let some sd := Import.CNNPyTorch.loadCnnStateDict 1 2 3 3 8 j
    | throw (IO.userError "pytorch_roundtrip_parity: failed to load CNN state dict")
  let conv1 : Conv2DSpec 1 2 3 3 1 1 Float (by decide) (by decide) (by decide) :=
    { kernel := sd.convW1, bias := sd.convB1 }
  let conv2 : Conv2DSpec 2 2 3 3 1 1 Float (by decide) (by decide) (by decide) :=
    { kernel := sd.convW2, bias := sd.convB2 }
  let pool1 : MaxPool2DSpec 2 2 2 (by decide) (by decide) (by decide) :=
    { kernelHeight := 2, kernelWidth := 2, stride := 2 }
  let pool2 : MaxPool2DSpec 2 2 2 (by decide) (by decide) (by decide) :=
    { kernelHeight := 2, kernelWidth := 2, stride := 2 }
  let linear : LinearSpec Float 8 2 := { weights := sd.linearW, bias := sd.linearB }
  let net := Models.cnnWithReluSpec (α := Float)
    (inH := 8) (inW := 8) conv1 conv2 pool1 pool2 linear
  let x : Tensor Float (.dim 1 (.dim 8 (.dim 8 .scalar))) :=
    Tensor.dim (fun _ =>
      Tensor.dim (fun i =>
        Tensor.dim (fun k =>
          Tensor.scalar (Float.ofNat (i.val * 8 + k.val + 1)))))
  let y := ModSpec.SpecChain.forward (α := Float) net x
  pure #[vecVal y ⟨0, by decide⟩, vecVal y ⟨1, by decide⟩]

def leanTransformer : IO (Array Float) := do
  let j ← NN.API.Json.parseFile transformerJson
  let some sd := Import.TransformerPyTorch.loadTransformerEncoderStateDict 2 1 2 j
    | throw (IO.userError "pytorch_roundtrip_parity: failed to load Transformer state dict")
  let layer : TransformerEncoderLayer 1 2 2 Float :=
    { mha := { Wq := sd.Wq, Wk := sd.Wk, Wv := sd.Wv, Wo := sd.Wo }
      ffn := { W1 := sd.W1, W2 := sd.W2, b1 := sd.b1, b2 := sd.b2 }
      norm1_gamma := sd.norm1_gamma
      norm1_beta := sd.norm1_beta
      norm2_gamma := sd.norm2_gamma
      norm2_beta := sd.norm2_beta }
  let encoder : TransformerEncoder 1 1 2 2 Float := { layers := [layer] }
  let x : Tensor Float (.dim 1 (.dim 2 .scalar)) := tensor! [[1.5, 1.5]]
  let y := TransformerEncoder.forward (seqLen := 1) (embedDim := 2)
    encoder x (by decide) (by decide)
  pure #[matVal y ⟨0, by decide⟩ ⟨0, by decide⟩,
    matVal y ⟨0, by decide⟩ ⟨1, by decide⟩]

def run : IO Unit := do
  IO.println "pytorch_roundtrip_parity: begin"
  if !(← pythonHasTorch) then
    IO.println "pytorch_roundtrip_parity: skipped (python package `torch` not installed)"
    return ()
  IO.FS.createDirAll workDir
  IO.FS.writeFile parityScriptPath parityScript
  let out ← Runtime.External.Process.runStdoutChecked
    (ctx := "pytorch_roundtrip_parity")
    (cmd := "python3")
    (args := #[parityScriptPath.toString])
    (cwd := some ".")
  let pyJson ←
    match Json.parse out with
    | .ok j => pure j
    | .error e => throw (IO.userError s!"pytorch_roundtrip_parity: bad Python JSON: {e}\n{out}")
  let pyMlp ←
    match jsonFloatArrayField pyJson "mlp" with
    | .ok xs => pure xs
    | .error e => throw (IO.userError e)
  let pyCnn ←
    match jsonFloatArrayField pyJson "cnn" with
    | .ok xs => pure xs
    | .error e => throw (IO.userError e)
  let pyTransformer ←
    match jsonFloatArrayField pyJson "transformer" with
    | .ok xs => pure xs
    | .error e => throw (IO.userError e)
  assertArrayApprox "pytorch_roundtrip_parity: mlp" (← leanMlp) pyMlp
  assertArrayApprox "pytorch_roundtrip_parity: cnn" (← leanCnn) pyCnn
  assertArrayApprox "pytorch_roundtrip_parity: transformer" (← leanTransformer) pyTransformer
  IO.println "pytorch_roundtrip_parity: ok"

end PyTorchRoundtripParity
end Floats
end Tests
