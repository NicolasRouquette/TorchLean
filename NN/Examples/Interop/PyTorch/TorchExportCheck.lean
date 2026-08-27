/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Lean.Data.Json
public import NN.Core.ExternalProcess
public import NN.IR.Semantics
public import NN.Runtime.PyTorch.Export.TorchExport
public import NN.Runtime.PyTorch.Import.TorchExport
public import NN.API

/-!
# PyTorch `nn.Module` → TorchLean IR Check

This executable example exercises the model-agnostic PyTorch graph bridge:

1. write the generated Python graph-capture adapter;
2. write several compact supported PyTorch models (MLP/CNN/CNN head/attention/MHA blocks);
3. capture them to `torchlean.ir.v1` JSON;
4. parse and validate each JSON artifact back into `NN.IR.Graph`;
5. compare affine normalization outputs against PyTorch;
6. exercise a real `torch.save(model.state_dict())` checkpoint reload path; and
7. run deliberate unsupported-op cases to confirm the bridge reports unsupported semantics clearly.

The check is scoped around the interop boundary rather than PyTorch performance. PyTorch may
produce an artifact, but TorchLean accepts it only after parsing and validating the explicit IR JSON.

Run:

```bash
lake exe torchlean pytorch_export_check
```
-/

@[expose] public section

namespace NN.Examples.Interop.PyTorch.TorchExportCheck

open Lean

/-- Command-line help for the PyTorch export bridge check. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean PyTorch export bridge check"
    , ""
    , "Usage:"
    , "  lake exe torchlean pytorch_export_check"
    , ""
    , "This command writes tiny PyTorch models, captures them with torch.export, validates the"
    , "TorchLean IR JSON, and checks a few deliberate unsupported-op failures."
    ]

def workDir : System.FilePath :=
  TorchLean.External.Process.artifactWorkDir "pytorch_export_check"

def bridgePath : System.FilePath :=
  workDir / "export_torchlean_graph.py"

def modelPath : System.FilePath :=
  workDir / "tiny_models.py"

def checkpointPath : System.FilePath :=
  workDir / "tiny_mlp_state.pt"

def supportedModelSource : String :=
  String.intercalate "\n"
    [ "import torch"
    , "import torch.nn as nn"
    , "import torch.nn.functional as F"
    , "from pathlib import Path"
    , ""
    , "class TinyAddRelu(nn.Module):"
    , "    def forward(self, x):"
    , "        return torch.relu(x + x)"
    , ""
    , "class TinyMLP(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.fc1 = nn.Linear(4, 3)"
    , "        self.fc2 = nn.Linear(3, 2)"
    , "    def forward(self, x):"
    , "        return self.fc2(torch.relu(self.fc1(x)))"
    , ""
    , "class TinyCheckpointMLP(TinyMLP):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        path = Path(__file__).with_name('tiny_mlp_state.pt')"
    , "        state = torch.load(path, map_location='cpu', weights_only=True)"
    , "        self.load_state_dict(state)"
    , ""
    , "class TinyCNN(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)"
    , "    def forward(self, x):"
    , "        return F.max_pool2d(torch.relu(self.conv(x)), kernel_size=2, stride=2)"
    , ""
    , "class TinyCNNHead(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)"
    , "        self.fc = nn.Linear(32, 5)"
    , "    def forward(self, x):"
    , "        x = F.max_pool2d(torch.relu(self.conv(x)), kernel_size=2, stride=2)"
    , "        x = torch.flatten(x)"
    , "        return self.fc(x)"
    , ""
    , "class TinyBatchNorm2d(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.bn = nn.BatchNorm2d(2)"
    , "    def forward(self, x):"
    , "        return self.bn(x)"
    , ""
    , "class TinyNormSoftmax(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.norm = nn.LayerNorm(4, elementwise_affine=False)"
    , "    def forward(self, x):"
    , "        return torch.softmax(self.norm(x), dim=-1)"
    , ""
    , "class TinySoftmaxModule(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.softmax = nn.Softmax(dim=-1)"
    , "    def forward(self, x):"
    , "        return self.softmax(x)"
    , ""
    , "class TinyLayerNormTail(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.norm = nn.LayerNorm((3, 4), elementwise_affine=False)"
    , "    def forward(self, x):"
    , "        return self.norm(x)"
    , ""
    , "class TinyPartialFlatten(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.flatten = nn.Flatten(start_dim=1, end_dim=2)"
    , "    def forward(self, x):"
    , "        return self.flatten(x)"
    , ""
    , "class TinyLastAxisSum(nn.Module):"
    , "    def forward(self, x):"
    , "        return x.sum(dim=-1)"
    , ""
    , "class TinyMiddleAxisMean(nn.Module):"
    , "    def forward(self, x):"
    , "        return x.mean(dim=1)"
    , ""
    , "class TinyNegativeConcat(nn.Module):"
    , "    def forward(self, x):"
    , "        return torch.cat((x, x), dim=-1)"
    , ""
    , "class TinyNegativePermute(nn.Module):"
    , "    def forward(self, x):"
    , "        return x.permute(0, -1, 1)"
    , ""
    , "class TinyReverseTranspose(nn.Module):"
    , "    def forward(self, x):"
    , "        return x.transpose(1, 0)"
    , ""
    , "class TinyTransformerishBlock(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.norm = nn.LayerNorm(4, elementwise_affine=False)"
    , "        self.fc1 = nn.Linear(4, 4)"
    , "        self.fc2 = nn.Linear(4, 4)"
    , "    def forward(self, x):"
    , "        y = self.fc2(torch.relu(self.fc1(self.norm(x))))"
    , "        return torch.softmax(x + y, dim=-1)"
    , ""
    , "class TinySelfAttentionOps(nn.Module):"
    , "    def forward(self, x):"
    , "        scores = torch.matmul(x, x.transpose(-2, -1))"
    , "        attn = torch.softmax(scores, dim=-1)"
    , "        return torch.matmul(attn, x)"
    , ""
    , "class UnsupportedSort(nn.Module):"
    , "    def forward(self, x):"
    , "        return torch.sort(x, dim=-1).values"
    , ""
    , "class TinyMultiAxisSum(nn.Module):"
    , "    def forward(self, x):"
    , "        return x.sum(dim=(1, 2))"
    , ""
    , "class TinyKeepdimSum(nn.Module):"
    , "    def forward(self, x):"
    , "        return x.sum(dim=-1, keepdim=True)"
    , ""
    , "class TinyFullMean(nn.Module):"
    , "    def forward(self, x):"
    , "        return x.mean()"
    , ""
    , "class UnsupportedBufferAdd(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.register_buffer('offset', torch.ones(4))"
    , "    def forward(self, x):"
    , "        return x + self.offset"
    , ""
    , "class TinyTwoOutputs(nn.Module):"
    , "    def forward(self, x):"
    , "        return torch.relu(x), torch.sigmoid(x)"
    , ""
    , "class TinyAffineLayerNorm(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.norm = nn.LayerNorm(4, eps=0.25)"
    , "        with torch.no_grad():"
    , "            self.norm.weight.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))"
    , "            self.norm.bias.copy_(torch.tensor([-1.0, -0.5, 0.5, 1.0]))"
    , "    def forward(self, x):"
    , "        return self.norm(x)"
    , ""
    , "class TinyLayerNormEpsilon(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.norm = nn.LayerNorm(4, eps=1e-4, elementwise_affine=False)"
    , "    def forward(self, x):"
    , "        return self.norm(x)"
    , ""
    , "class TinyGroupedConv(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.conv = nn.Conv2d(4, 4, 3, padding=1, groups=2)"
    , "    def forward(self, x):"
    , "        return self.conv(x)"
    , ""
    , "class TinyDilatedConv(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.conv = nn.Conv2d(1, 2, 3, padding=1, dilation=2)"
    , "    def forward(self, x):"
    , "        return self.conv(x)"
    , ""
    , "class TinyAsymmetricConvStride(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.conv = nn.Conv2d(1, 2, 3, stride=(1, 2), padding=1)"
    , "    def forward(self, x):"
    , "        return self.conv(x)"
    , ""
    , "class UnsupportedReflectConv(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.conv = nn.Conv2d(1, 2, 3, padding=1, padding_mode='reflect')"
    , "    def forward(self, x):"
    , "        return self.conv(x)"
    , ""
    , "class UnsupportedBatchNormNoStats(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.bn = nn.BatchNorm2d(2, track_running_stats=False)"
    , "    def forward(self, x):"
    , "        return self.bn(x)"
    , ""
    , "class TinyBatchNormEpsilon(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.bn = nn.BatchNorm2d(2, eps=1e-4)"
    , "        with torch.no_grad():"
    , "            self.bn.weight.copy_(torch.tensor([1.5, 0.5]))"
    , "            self.bn.bias.copy_(torch.tensor([-0.25, 0.75]))"
    , "            self.bn.running_mean.copy_(torch.tensor([0.5, -1.0]))"
    , "            self.bn.running_var.copy_(torch.tensor([2.0, 3.0]))"
    , "    def forward(self, x):"
    , "        return self.bn(x)"
    , ""
    , "class UnsupportedMaxPoolDilation(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.pool = nn.MaxPool2d(2, dilation=2)"
    , "    def forward(self, x):"
    , "        return self.pool(x)"
    , ""
    , "class UnsupportedMaxPoolCeil(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.pool = nn.MaxPool2d(2, ceil_mode=True)"
    , "    def forward(self, x):"
    , "        return self.pool(x)"
    , ""
    , "class TinyAsymmetricPoolStride(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.pool = nn.MaxPool2d(2, stride=(1, 2))"
    , "    def forward(self, x):"
    , "        return self.pool(x)"
    , ""
    , "class UnsupportedAvgPoolExcludePad(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.pool = nn.AvgPool2d(2, padding=1, count_include_pad=False)"
    , "    def forward(self, x):"
    , "        return self.pool(x)"
    , ""
    , "class UnsupportedAvgPoolDivisor(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.pool = nn.AvgPool2d(2, divisor_override=3)"
    , "    def forward(self, x):"
    , "        return self.pool(x)"
    , ""
    , "class UnsupportedSoftmaxDtype(nn.Module):"
    , "    def forward(self, x):"
    , "        return F.softmax(x, dim=-1, dtype=torch.float64)"
    , ""
    , "class UnsupportedReductionDtype(nn.Module):"
    , "    def forward(self, x):"
    , "        return torch.sum(x, dim=-1, dtype=torch.float64)"
    , ""
    , "class UnsupportedAddAlpha(nn.Module):"
    , "    def forward(self, x):"
    , "        return torch.add(x, x, alpha=2)"
    , ""
    , "class UnsupportedScalarAdd(nn.Module):"
    , "    def forward(self, x):"
    , "        return x + 1.0"
    , ""
    , "class TinySingleHeadMHA(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.mha = nn.MultiheadAttention(embed_dim=4, num_heads=1, batch_first=True)"
    , "    def forward(self, x):"
    , "        y, _ = self.mha(x, x, x)"
    , "        return y"
    , ""
    , "class UnsupportedMultiHeadMHA(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.mha = nn.MultiheadAttention(embed_dim=4, num_heads=2, batch_first=True)"
    , "    def forward(self, x):"
    , "        y, _ = self.mha(x, x, x)"
    , "        return y"
    , ""
    , "class UnsupportedAugmentedMHA(nn.Module):"
    , "    def __init__(self):"
    , "        super().__init__()"
    , "        self.mha = nn.MultiheadAttention(embed_dim=4, num_heads=1, batch_first=True, add_zero_attn=True)"
    , "    def forward(self, x):"
    , "        y, _ = self.mha(x, x, x)"
    , "        return y"
    , ""
    ]

def runCapture (ctor : String) (outPath : System.FilePath) (shape : String)
    (requireTorchExport : Bool := false) : IO String := do
  let args :=
    #[bridgePath.toString, modelPath.toString, ctor, outPath.toString, "--example-shape", shape] ++
      if requireTorchExport then #["--require-torch-export"] else #[]
  TorchLean.External.Process.runStdoutChecked
    (ctx := s!"PyTorch graph capture ({ctor})")
    (cmd := "python")
    (args := args)
    (cwd := some ".")

/--
Write a real PyTorch checkpoint used by `TinyCheckpointMLP`.

Python owns `.pt` loading/saving, and the graph bridge only sees the resulting initialized
`nn.Module`. That mirrors the intended user workflow:
load trained weights in Python, capture the model graph, then ask Lean to validate the exported IR.
-/
def writeCheckpoint : IO Unit := do
  let _stdout ← TorchLean.External.Process.runStdoutChecked
    (ctx := "PyTorch checkpoint reference")
    (cmd := "python")
    (args := #[
      "-c",
      "import importlib.util, torch; " ++
      s!"spec=importlib.util.spec_from_file_location('tiny_models', '{modelPath}'); " ++
      "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); " ++
      "torch.manual_seed(7); model=m.TinyMLP(); " ++
      s!"torch.save(model.state_dict(), '{checkpointPath}')"
    ])
    (cwd := some ".")
  pure ()

/--
Keep subprocess failures readable in the runtime-check output.

The full `IO.userError` still contains command, args, exit code, and stderr. The runtime check only prints
the first informative lines so unsupported-op tests explain the boundary without hiding the signal.
-/
def compactFailure (err : IO.Error) : String :=
  String.intercalate "\n" ((err.toString.splitOn "\n").take 24)

/-- Compare an imported normalization graph against PyTorch on a deterministic input. -/
def checkNormalizationParity
    (ctor shape : String) (artifactPath : System.FilePath) : IO Unit := do
  let json ← TorchLean.Json.parseFile artifactPath
  let captured ←
    match Import.PyTorch.TorchExport.parseGraph json with
    | .ok value => pure value
    | .error message => throw <| IO.userError message
  let payload ←
    match Import.PyTorch.TorchExport.parsePayload json with
    | .ok value => pure value
    | .error message => throw <| IO.userError message
  let python ← TorchLean.External.Process.runStdoutChecked
    (ctx := s!"PyTorch normalization parity ({ctor})")
    (cmd := "python")
    (args := #[
      "-c",
      "import importlib.util,json,math,torch; " ++
      s!"spec=importlib.util.spec_from_file_location('tiny_models','{modelPath}'); " ++
      "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); " ++
      s!"model=m.{ctor}().eval(); shape=tuple(int(x) for x in '{shape}'.split(',')); " ++
      "x=torch.arange(math.prod(shape),dtype=torch.float32).reshape(shape)/7.0; " ++
      "print(json.dumps({'input':x.tolist(),'output':model(x).detach().tolist()}))"
    ])
    (cwd := some ".")
  let reference ←
    match Lean.Json.parse python.trimAscii.toString with
    | .ok value => pure value
    | .error message => throw <| IO.userError message
  let fields ←
    match reference with
    | .obj value => pure value
    | _ => throw <| IO.userError "normalization parity reference was not a JSON object"
  let inputJson ←
    match fields.get? "input" with
    | some value => pure value
    | none => throw <| IO.userError "normalization parity reference omitted input"
  let outputJson ←
    match fields.get? "output" with
    | some value => pure value
    | none => throw <| IO.userError "normalization parity reference omitted output"
  let inputNode ←
    match captured.graph.getNode captured.inputId with
    | .ok value => pure value
    | .error message => throw <| IO.userError message
  let outputId := captured.outputIds[0]!
  let outputNode ←
    match captured.graph.getNode outputId with
    | .ok value => pure value
    | .error message => throw <| IO.userError message
  let some inputTensor := Import.PyTorch.parseTensor inputNode.outShape inputJson
    | throw <| IO.userError "normalization parity input shape mismatch"
  let some expectedTensor := Import.PyTorch.parseTensor outputNode.outShape outputJson
    | throw <| IO.userError "normalization parity output shape mismatch"
  let actual ←
    match NN.IR.Graph.denote (α := Float) captured.graph payload
        (Spec.SomeTensor.ofTensor inputTensor) outputId with
    | .ok value => pure value
    | .error message => throw <| IO.userError message
  unless actual.shape = outputNode.outShape do
    throw <| IO.userError "normalization parity produced the wrong output shape"
  for (value, expected) in actual.tensor.toArray.zip expectedTensor.toArray do
    unless Float.abs (value - expected) ≤ 2e-5 do
      throw <| IO.userError <|
        s!"normalization parity mismatch for {ctor}: expected {expected}, got {value}"
  IO.println s!"  numerical parity: ok ({ctor})"

/-- Run one supported capture path and parse the resulting graph in Lean. -/
def runSupportedCase (ctor shape : String) (expectedKind : Option String := none) : IO Unit := do
  let outPath := workDir / s!"{ctor}.graph.json"
  let _stdout ← runCapture ctor outPath shape
  IO.println s!"captured supported graph: {ctor} ({shape})"
  if ctor = "TinyBatchNorm2d" then
    let txt ← IO.FS.readFile outPath
    unless txt.contains "\"eps\"" do
      throw <| IO.userError "TinyBatchNorm2d export did not preserve BatchNorm epsilon metadata"
  let j ← TorchLean.Json.parseFile outPath
  match Import.PyTorch.TorchExport.parseGraph j with
  | .ok cg =>
      if let some expected := expectedKind then
        unless cg.graph.nodes.any (fun node => node.kind.describe.startsWith expected) do
          throw <| IO.userError
            s!"TorchLean graph `{ctor}` did not contain expected operation `{expected}`"
      IO.println <|
        s!"  accepted: nodes={cg.graph.nodes.size}, input={cg.inputId}, outputs={repr cg.outputIds}"
      IO.println "  guarantee: WellShaped via parseGraph_wellShaped"
      if ctor = "TinyAffineLayerNorm" || ctor = "TinyLayerNormEpsilon" ||
          ctor = "TinyBatchNormEpsilon" then
        checkNormalizationParity ctor shape outPath
  | .error e =>
      throw <| IO.userError s!"TorchLean rejected supported PyTorch graph `{ctor}`:\n{e}"

/-- Run the supported capture paths and parse the resulting graphs in Lean. -/
def runSupported : IO Unit := do
  runSupportedCase "TinyAddRelu" "1,4"
  runSupportedCase "TinyMLP" "4"
  runSupportedCase "TinyCheckpointMLP" "4"
  runSupportedCase "TinyCNN" "1,8,8"
  runSupportedCase "TinyCNNHead" "1,8,8"
  runSupportedCase "TinyAsymmetricConvStride" "1,8,8" (some "conv")
  runSupportedCase "TinyAsymmetricPoolStride" "1,8,8" (some "max_pool")
  runSupportedCase "TinyBatchNorm2d" "1,2,4,4"
  runSupportedCase "TinyNormSoftmax" "4"
  runSupportedCase "TinySoftmaxModule" "2,3,4" (some "softmax(axis=2)")
  runSupportedCase "TinyLayerNormTail" "2,3,4" (some "layernorm(axis=1)")
  runSupportedCase "TinyAffineLayerNorm" "2,4" (some "layernorm(axis=1)")
  runSupportedCase "TinyLayerNormEpsilon" "2,4" (some "layernorm(axis=1)")
  runSupportedCase "TinyBatchNormEpsilon" "1,2,4,4" (some "batch_norm_eval")
  runSupportedCase "TinyPartialFlatten" "2,3,4" (some "reshape")
  runSupportedCase "TinyLastAxisSum" "2,3,4" (some "reduce_sum(axis=2)")
  runSupportedCase "TinyMiddleAxisMean" "2,3,4" (some "reduce_mean(axis=1)")
  runSupportedCase "TinyMultiAxisSum" "2,3,4" (some "reduce_sum")
  runSupportedCase "TinyKeepdimSum" "2,3,4" (some "reshape")
  runSupportedCase "TinyFullMean" "2,3,4" (some "reduce_mean")
  runSupportedCase "TinyTwoOutputs" "2,4"
  runSupportedCase "TinyGroupedConv" "4,8,8" (some "conv")
  runSupportedCase "TinyDilatedConv" "1,8,8" (some "conv")
  runSupportedCase "TinyNegativeConcat" "2,3,4" (some "concat(axis=2)")
  runSupportedCase "TinyNegativePermute" "2,3,4" (some "permute(perm=#[0, 2, 1])")
  runSupportedCase "TinyReverseTranspose" "2,3,4" (some "transpose(axis1=1, axis2=0)")
  runSupportedCase "TinyTransformerishBlock" "4"
  runSupportedCase "TinySelfAttentionOps" "1,2,4"
  runSupportedCase "TinySingleHeadMHA" "1,2,4"
  if (← IO.getEnv "TORCHLEAN_REQUIRE_TORCH_EXPORT") = some "1" then
    let outPath := workDir / "TinyMLP.torch-export.graph.json"
    let _stdout ← runCapture "TinyMLP" outPath "4" true
    let j ← TorchLean.Json.parseFile outPath
    match Import.PyTorch.TorchExport.parseGraph j with
    | .ok _ => IO.println "required torch.export capture: ok (TinyMLP)"
    | .error e =>
        throw <| IO.userError s!"TorchLean rejected required torch.export graph `TinyMLP`:\n{e}"

/--
Require an unsupported PyTorch model to fail during capture or checked Lean import.

Capture strategy varies across PyTorch releases. Some unsupported operators fail in the Python
adapter; others survive as value-graph nodes and are rejected by Lean's tensor-IR importer. Either
boundary is valid, but successful tensor lowering is not.
-/
def runUnsupportedCase (ctor shape msg : String) (expectedDetails : List String) : IO Unit := do
  let outPath := workDir / s!"{ctor}.graph.json"
  let captured : Except IO.Error String ←
    try
      pure (.ok (← runCapture ctor outPath shape))
    catch err =>
      pure (.error err)
  match captured with
  | .error err =>
      let detail := compactFailure err
      unless expectedDetails.any (fun expected => detail.contains expected) do
        throw <| IO.userError <|
          s!"unsupported graph `{ctor}` failed for an unexpected reason:\n{detail}"
      IO.println s!"unsupported graph rejected during capture: ok ({ctor})"
      IO.println s!"  {msg}"
      IO.println s!"  failure detail:\n{detail}"
  | .ok _ =>
      let j ← TorchLean.Json.parseFile outPath
      match Import.PyTorch.TorchExport.parseGraph j with
      | .ok _ =>
          throw <| IO.userError s!"unsupported PyTorch graph unexpectedly lowered: {ctor}"
      | .error e =>
          unless expectedDetails.any (fun expected => e.contains expected) do
            throw <| IO.userError <|
              s!"unsupported graph `{ctor}` failed for an unexpected reason:\n{e}"
          IO.println s!"unsupported graph rejected during Lean import: ok ({ctor})"
          IO.println s!"  {msg}"
          IO.println s!"  failure detail:\n{e}"

/-- Run deliberate unsupported ops and require capture or checked import to reject them. -/
def runUnsupported : IO Unit := do
  runUnsupportedCase "UnsupportedSort" "1,4"
    "rejected `torch.sort`, which is not in the current TorchLean IR import subset"
    ["unsupported PyTorch attribute projection", "tuple producer has no tensor-lowering rule"]
  runUnsupportedCase "UnsupportedBufferAdd" "4"
    "rejected an arbitrary lifted buffer instead of silently erasing an add operand"
    ["bad parent count for add"]
  runUnsupportedCase "UnsupportedReflectConv" "1,8,8"
    "rejected reflect padding instead of treating it as zero padding"
    ["padding_mode='reflect'", "unsupported PyTorch op: pad"]
  runUnsupportedCase "UnsupportedBatchNormNoStats" "1,2,4,4"
    "rejected batch-dependent normalization instead of importing it as eval-mode BatchNorm"
    ["without running statistics", "training-mode batch_norm"]
  runUnsupportedCase "UnsupportedMaxPoolDilation" "1,8,8"
    "rejected dilated max pooling instead of importing a different window"
    ["dilated max pooling"]
  runUnsupportedCase "UnsupportedMaxPoolCeil" "1,8,8"
    "rejected ceil-mode pooling instead of changing output-window semantics"
    ["ceil_mode=True"]
  runUnsupportedCase "UnsupportedAvgPoolExcludePad" "1,8,8"
    "rejected padded averaging with a different divisor convention"
    ["count_include_pad=False"]
  runUnsupportedCase "UnsupportedAvgPoolDivisor" "1,8,8"
    "rejected an average-pooling divisor override that the current IR cannot record"
    ["divisor_override"]
  runUnsupportedCase "UnsupportedSoftmaxDtype" "2,3,4"
    "rejected an explicit softmax dtype cast that the scalar-generic graph does not represent"
    ["softmax with an explicit dtype/cast"]
  runUnsupportedCase "UnsupportedReductionDtype" "2,3,4"
    "rejected an explicit reduction dtype cast that the scalar-generic graph does not represent"
    ["explicit dtype changes scalar semantics"]
  runUnsupportedCase "UnsupportedAddAlpha" "2,4"
    "rejected scaled addition instead of importing it as ordinary addition"
    ["torch.add with alpha != 1", "unsupported PyTorch op"]
  runUnsupportedCase "UnsupportedScalarAdd" "2,4"
    "rejected scalar arithmetic until the scalar constant and broadcast are explicit in the graph"
    ["needs an explicit scalar constant/broadcast lowering", "unsupported PyTorch op",
      "bad parent count for add"]
  runUnsupportedCase "UnsupportedMultiHeadMHA" "1,2,4"
    "captured MultiheadAttention as a tuple-valued FX node, then rejected tensor lowering because multi-head splitting is not in the current tensor-IR lowering slice"
    ["supports only num_heads=1"]
  runUnsupportedCase "UnsupportedAugmentedMHA" "1,2,4"
    "rejected attention that appends a synthetic sequence entry instead of importing ordinary self-attention"
    ["added bias/zero sequence entries"]

/-- Build a compact `torchlean.ir.v1` artifact for parser-boundary regression cases. -/
def graphArtifact (nodes : String) (inputId : Nat := 0) (outputId : Nat := 0) : String :=
  "{\"format\":\"torchlean.ir.v1\",\"input_id\":" ++ toString inputId ++
    ",\"output_ids\":[" ++ toString outputId ++ "],\"nodes\":[" ++ nodes ++ "]}"

/-- Require malformed graph JSON to fail with a diagnostic from the intended validation boundary. -/
def expectImportRejection (name source expected : String) : IO Unit := do
  let json ←
    match Lean.Json.parse source with
    | .ok json => pure json
    | .error error =>
        throw <| IO.userError s!"malformed-artifact test `{name}` contains invalid JSON: {error}"
  match Import.PyTorch.TorchExport.parseGraph json with
  | .ok _ => throw <| IO.userError s!"malformed graph unexpectedly accepted: {name}"
  | .error error =>
      unless error.contains expected do
        throw <| IO.userError <|
          s!"malformed graph `{name}` failed for the wrong reason:\n{error}\n" ++
            s!"expected diagnostic containing: {expected}"
      IO.println s!"malformed graph rejected: ok ({name})"

/-- Exercise validation that must hold even for artifacts not produced by the Python adapter. -/
def runMalformedArtifacts : IO Unit := do
  expectImportRejection "missing format marker"
    ("{\"input_id\":0,\"output_ids\":[0],\"nodes\":[" ++
      "{\"id\":0,\"kind\":\"input\",\"parents\":[],\"shape\":[4]}]}")
    "missing field `format`"
  expectImportRejection "empty graph outputs"
    "{\"format\":\"torchlean.ir.v1\",\"input_id\":0,\"output_ids\":[],\"nodes\":[{\"id\":0,\"kind\":\"input\",\"parents\":[],\"shape\":[4]}]}"
    "must contain at least one output"
  expectImportRejection "raw id mismatch"
    (graphArtifact "{\"id\":7,\"kind\":\"input\",\"parents\":[],\"shape\":[4]}")
    "raw id discipline violated"
  expectImportRejection "raw forward reference"
    (graphArtifact
      ("{\"id\":0,\"kind\":\"input\",\"parents\":[],\"shape\":[4]}," ++
       "{\"id\":1,\"kind\":\"relu\",\"parents\":[2],\"shape\":[4]}," ++
       "{\"id\":2,\"kind\":\"relu\",\"parents\":[0],\"shape\":[4]}") 0 1)
    "parent id 2 is not < 1"
  expectImportRejection "multiple graph inputs"
    (graphArtifact
      ("{\"id\":0,\"kind\":\"input\",\"parents\":[],\"shape\":[4]}," ++
       "{\"id\":1,\"kind\":\"input\",\"parents\":[],\"shape\":[4]}") 0 1)
    "expected exactly one input node"
  expectImportRejection "misdesignated graph input"
    (graphArtifact
      ("{\"id\":0,\"kind\":\"input\",\"parents\":[],\"shape\":[4]}," ++
       "{\"id\":1,\"kind\":\"relu\",\"parents\":[0],\"shape\":[4]}") 1 1)
    "designates `relu`, not `input`"
  let tuplePrefix :=
    "{\"id\":0,\"kind\":\"input\",\"parents\":[],\"shape\":[1,2,4]}," ++
    "{\"id\":1,\"kind\":\"multihead_attention\",\"parents\":[0,0,0]," ++
    "\"value_kind\":\"tuple\",\"tuple_shapes\":[[1,2,4],[1,2,2]]," ++
    "\"embed_dim\":4,\"num_heads\":1,\"batch_first\":true," ++
    "\"dropout_zero\":true},"
  expectImportRejection "tuple index out of bounds"
    (graphArtifact
      (tuplePrefix ++
       "{\"id\":2,\"kind\":\"tuple_getitem\",\"parents\":[1]," ++
       "\"index\":2,\"shape\":[1,2,4]}") 0 2)
    "tuple index 2 is out of bounds"
  expectImportRejection "tuple projection shape mismatch"
    (graphArtifact
      (tuplePrefix ++
       "{\"id\":2,\"kind\":\"tuple_getitem\",\"parents\":[1]," ++
       "\"index\":0,\"shape\":[1,2,3]}") 0 2)
    "tuple component 0 has shape"
  expectImportRejection "axis outside tensor rank"
    (graphArtifact
      ("{\"id\":0,\"kind\":\"input\",\"parents\":[],\"shape\":[2,4]}," ++
       "{\"id\":1,\"kind\":\"softmax\",\"parents\":[0]," ++
       "\"axis\":2,\"shape\":[2,4]}") 0 1)
    "invalid axis 2 for rank 2"

/-- Main runtime-check body. -/
def run : IO Unit := do
  IO.FS.createDirAll workDir
  IO.FS.writeFile bridgePath (Export.PyTorch.TorchExport.generateGraphBridgeScript {})
  IO.FS.writeFile modelPath supportedModelSource
  writeCheckpoint
  IO.println "== PyTorch nn.Module → TorchLean IR runtime check =="
  runSupported
  runUnsupported
  runMalformedArtifacts
  IO.println "pytorch_export_check: ok"

/-- Entrypoint used by `lake exe torchlean pytorch_export_check`. -/
def main (args : List String) : IO UInt32 := do
  let args := _root_.TorchLean.CLI.dropDashDash args
  if _root_.TorchLean.CLI.hasHelp args then
    IO.println usage
    return 0
  _root_.TorchLean.CLI.requireNoArgs "pytorch_export_check" args
  run
  pure 0

end NN.Examples.Interop.PyTorch.TorchExportCheck
