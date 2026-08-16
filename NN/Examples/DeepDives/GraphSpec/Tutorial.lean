/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.GraphSpec.Models
public import NN.GraphSpec.ToSequential

/-!
# GraphSpec tutorial

GraphSpec is TorchLean's architecture-facing graph language. It is useful when you want a model to
exist first as a typed graph that can later be interpreted in several ways:

- as a pure specification (`Interp.spec`) for theorem statements,
- as an executable TorchLean program (`Chain.toProgram`) for direct execution,
- as a sequential `nn.Sequential` view when the graph is just a layer stack, and
- as a DAG model when the architecture has sharing, skip connections, or multi-input nodes.

That is why this folder is not a duplicate of `NN.Spec.Models` or `NN.Examples.Models`.

- `NN.Spec.Models` describes mathematical/reference model semantics.
- `NN.GraphSpec.Models` describes architecture graphs and their parameter ABI.
- `NN.Examples.Models` contains runnable training scripts.

This tutorial does two things:

1. It runs the smallest complete sequential-model path: `GraphSpec.Models.mlp → ToSequential.toSeq →
  Trainer.new → trainer.train`.
2. It typechecks the broader GraphSpec model ladder: MLP, CNN, and a residual linear block.

Only the MLP is trained here because it is the compact check path for `Seq` lowering. The CNN and
residual block remain architecture terms that graph passes, exporters, and proofs can consume.

Run:

```bash
lake exe torchlean graphspec --execution eager
lake exe torchlean graphspec --execution typed-graph
```

You can also pass the standard TorchLean runtime flags such as `--scalar ieee32-exec`, `--execution eager`,
or `--execution typed-graph`.
-/

@[expose] public section


namespace NN.Examples.DeepDives.GraphSpec.Tutorial

open _root_.Spec
open _root_.Spec.Tensor
open NN.Tensor
open _root_.TorchLean

/-- Command-line help for the GraphSpec tutorial. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean GraphSpec tutorial"
    , ""
    , "Usage:"
    , "  lake exe torchlean graphspec [options]"
    , ""
    , "Options:"
    , "  --scalar float32|ieee32-exec|complex64"
    , "  --execution eager|typed-graph"
    , "  --device auto|cpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external"
    , "  --show-backend                    print backend capsules as they execute"
    ]

/-! ## Small architecture terms that should typecheck -/

/--
The smallest sequential GraphSpec model.

Parameter ABI:
`[Mat 3 2, Vec 3, Mat 1 3, Vec 1]`.
-/
def tutorialMlp :
    NN.GraphSpec.Chain
      [ .dim 3 (.dim 2 .scalar), .dim 3 .scalar
      , .dim 1 (.dim 3 .scalar), .dim 1 .scalar ]
      (.dim 2 .scalar) (.dim 1 .scalar) :=
  NN.GraphSpec.Models.mlp (inDim := 2) (hidDim := 3) (outDim := 1)

/--
A small CNN graph, included here so the tutorial is visibly not “just MLP”.

This is still a sequential graph: convolution, ReLU, pooling, convolution, ReLU, pooling, flatten,
linear head. The ugly-looking type is the point: the parameter shapes and intermediate spatial
arithmetic are checked before the model can be used.
-/
def tutorialCnn :=
  NN.GraphSpec.Models.twoConvCnn
    (inC := 1) (c1 := 2) (c2 := 3) (outDim := 4)
    (inH := 8) (inW := 8) (kH := 3) (kW := 3)
    (stride1 := 1) (padding1 := 1) (stride2 := 1) (padding2 := 1)
    (poolKH := 2) (poolKW := 2) (poolStride1 := 2) (poolStride2 := 2)
    (h_inC := by decide) (h_c1 := by decide) (_h_c2 := by decide)
    (h_kH := by decide) (h_kW := by decide)
    (h_stride1 := by decide) (h_stride2 := by decide)
    (h_poolKH := by decide) (h_poolKW := by decide)
    (h_poolStride1 := by decide) (h_poolStride2 := by decide)

/--
The minimal DAG-native skip-connection example:

$$
x\mapsto\operatorname{ReLU}(Wx+b+x).
$$

This is not a plain chain: representing it that way would either duplicate the input path or hide
sharing in a special layer. That is the pedagogical reason `GraphSpec.DAG` exists.
-/
def tutorialResidual :=
  NN.GraphSpec.Models.residualLinear (d := 4)

/-- Print the architecture ladder this tutorial is checking. -/
def printCatalog : IO Unit := do
  IO.println "GraphSpec architecture ladder:"
  IO.println "  1. MLP: sequential layer stack; lowers to nn.Sequential and trains below."
  IO.println "  2. CNN: sequential vision graph with checked conv/pool shape arithmetic."
  IO.println "  3. residualLinear: minimal DAG-native skip connection."
  IO.println ""

/-- Tiny one-sample dataset for the lowered GraphSpec MLP training path. -/
def tutorialDataset : Trainer.DataSource (.dim 2 .scalar) (.dim 1 .scalar) :=
  let xF : Spec.Tensor Float (.dim 2 .scalar) := tensorF! id [2] [0.5, 0.8]
  let yF : Spec.Tensor Float (.dim 1 .scalar) := tensorF! id [1] [1.0]
  let XFloat : Spec.Tensor Float (shape![1, 2]) := Spec.Tensor.dim (fun _ => xF)
  let YFloat : Spec.Tensor Float (shape![1, 1]) := Spec.Tensor.dim (fun _ => yF)
  Data.tensorDataset XFloat YFloat

/-- Run the compact MLP lowering/training path. -/
def runMlpTrainingPath (args : List String) : IO Unit := do
  let inDim : Nat := 2
  let hidDim : Nat := 3
  let outDim : Nat := 1

  let xShape : Spec.Shape := .dim inDim .scalar
  let yShape : Spec.Shape := .dim outDim .scalar

  -- GraphSpec is the source architecture. This exact graph also has pure semantics and an
  -- executable program view; here we ask for the additional `nn.Sequential` training view.
  let g := NN.GraphSpec.Models.mlp (inDim := inDim) (hidDim := hidDim) (outDim := outDim)

  match NN.GraphSpec.ToSequential.toSeq (σ := xShape) (τ := yShape) g with
  | .error msg =>
      throw <| IO.userError s!"GraphSpec.ToSequential.toSeq failed: {msg}"
  | .ok seqR =>
      let seq : nn.Sequential xShape yShape := by
        -- `nn.Sequential` is the public API name for the same runtime `Seq` type.
        simpa using seqR
      let run ← Trainer.RunConfig.parseRuntimeArgsOrThrow "GraphSpecTutorial"
        (CLI.dropDashDash args)
        { optimizer := optim.sgd { lr := 0.1 } }
      let trainer := Trainer.new seq <|
        Trainer.Config.fromRunConfig run .regression
      trainer.printInfo
      let trained ← trainer.train tutorialDataset { steps := 3, title := "GraphSpec tutorial" }
      IO.println "forward: GraphSpec MLP lowered to TorchLean and executed"
      trained.printSummary

def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return
  IO.println "== GraphSpec tutorial =="
  printCatalog
  runMlpTrainingPath args

end NN.Examples.DeepDives.GraphSpec.Tutorial
