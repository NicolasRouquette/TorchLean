import VersoManual
import NN.Widgets
import NN.Spec.Core.Tensor.Core
import NN.IR.Graph
import NN.IR.Semantics
import NN.Floats.IEEEExec.Exec32
import NN.MLTheory.CROWN.Graph
import NN.Runtime.Autograd.Engine.Core
import NN.Runtime.Training.Log
import NN.Runtime.Context

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean

#doc (Manual) "Widgets" =>
%%%
tag := "widgets"
%%%

Widgets are the human inspection layer of TorchLean. They do not define semantics and they do not
prove theorems. They let a reader see the Lean objects that the semantics and theorems are about:
tensors, IR graphs, shape inference results, Float32 bit patterns, CROWN bounds, autograd tapes,
gradients, and training logs.

The [widgets source](https://github.com/lean-dojo/TorchLean/tree/main/NN/Widgets/) is collected by `NN.Widgets`. Import that module
when you want only the widget layer; `import NN` includes it as part of the broad main library
umbrella.

Formal artifacts are hard to debug because they are often large, nested, and invisible: graph
nodes, tensor shapes, interval boxes, affine forms, tape cotangents, Float32 bit patterns. Widgets
make those artifacts visible inside the same editor where the theorem or checker lives.

# What Widgets Are For

Widgets are for looking carefully at Lean objects while you work. They are not a separate semantic
layer, so TorchLean keeps the meaning of "execute this" explicit:

- Widgets that inspect tensors are read-only formatted views of concrete values (they do not change
  the underlying data).
- Widgets that *run* something (IR evaluation, backprop traces) are running a specific computation defined in Lean
  semantics (for example `NN.IR.Semantics` for the IR with named operations, or `Runtime.Autograd.Engine` for the
  tape engine).

This keeps assumptions visible:

- An execution trace widget is an evaluator defined in Lean stepping through the chosen semantics.
- A theorem still requires a proof about the same semantics the widget executes.

The intended workflow is simple: a widget renders a Lean object under a chosen Lean interpretation.
It should not invent a new meaning that the proof and checker layers cannot see.

So a widget can make a proof obligation visible, but it does not replace the proof. If a graph view
shows a bad shape, we fix the graph or the compiler. If a CROWN view shows loose bounds, we improve
the bound pass or certificate.

:::table +header
*
  * Widget
  * Object inspected
  * Main question
*
  * `#tensor_view`
  * typed tensor
  * what values and shape?
*
  * `#ir_view`
  * `NN.IR.Graph`
  * what nodes and parents?
*
  * `#shape_infer_view`
  * graph shape inference
  * where is the shape mismatch?
*
  * `#ir_exec_trace_view`
  * IR execution
  * what did each node compute?
*
  * `#float32_view`
  * `IEEE32Exec`
  * what are the bits?
*
  * `#crown_view`
  * verifier state
  * where are the bounds?
*
  * `#tape_trace_view`
  * autograd tape
  * how did gradients flow?
*
  * `#train_log_view`
  * training log
  * did metrics move?
*
  * `#train_log_file_view`
  * JSON training log
  * what did a command write?
*
  * `#gpt2_train_log_file_view`
  * GPT log with prompts/samples
  * what prompt and generated sample were recorded?
*
  * `#rl_boundary_rollout_file_view`
  * RL boundary artifact
  * which transition fields passed or failed the contract?
*
  * `#pytorch_translate_file`
  * PyTorch source snippet
  * what TorchLean sketch does the editor assistant infer?
:::

# How Widgets Fit Application Workflows

Application examples tend to produce artifacts rather than just values in memory:

- GPT examples write training logs with prompt and generated-text notes.
- PPO examples write reward curves, policy/path artifacts, and boundary traces.
- Verification examples produce graph and bound states.
- PyTorch interop examples produce imported/exported source or IR.

Widgets make those artifacts pleasant to inspect inside Lean:

```
#train_log_file_view "data/model_zoo/cnn_trainlog.json"
#gpt2_train_log_file_view "data/model_zoo/gpt2_trainlog.json"
#rl_boundary_rollout_file_view "data/rl/cartpole_rollout.json", contract, 12
#pytorch_translate_file "NN/Examples/Quickstart/pytorch_translator_mlp.py"
```

The command that produced the file remains part of the claim. A file-backed widget is only as
meaningful as the artifact path and producer command it is paired with.

# Tensor Viewer

```
open Spec
open TorchLean.Floats.IEEE754

def decimalTenth : Float :=
  Float.ofBits 0x3fb999999999999a

def oneThirdFloat : Float :=
  Float.ofBits 0x3fd5555555555555

def floatVector : Tensor Float (shape![4]) :=
  Tensor.dim (fun
    | ⟨0, _⟩ => Tensor.scalar (Float.ofNat 1)
    | ⟨1, _⟩ => Tensor.scalar (Float.ofNat 2)
    | ⟨2, _⟩ => Tensor.scalar decimalTenth
    | ⟨_, _⟩ => Tensor.scalar oneThirdFloat)

def ieeeVector : Tensor IEEE32Exec (shape![4]) :=
  Tensor.dim (fun
    | ⟨0, _⟩ => Tensor.scalar IEEE32Exec.posOne
    | ⟨1, _⟩ =>
        Tensor.scalar (IEEE32Exec.ofFloat (Float.ofNat 2))
    | ⟨2, _⟩ => Tensor.scalar (IEEE32Exec.ofFloat decimalTenth)
    | ⟨_, _⟩ => Tensor.scalar (IEEE32Exec.ofFloat oneThirdFloat))

def indexVector : Tensor Nat (shape![5]) :=
  Tensor.dim (fun i => Tensor.scalar i.1)

def rankThreeGrid :
    Tensor Nat (shape![2, 3, 4]) :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      Tensor.dim (fun k =>
        Tensor.scalar (i.1 * 100 + j.1 * 10 + k.1))))

def sampleMatrix : Tensor Int (shape![2, 4]) :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      Tensor.scalar (Int.ofNat (i.1 * 10 + j.1))))

#tensor_view indexVector
#tensor_view rankThreeGrid
#tensor_view floatVector
#tensor_view ieeeVector
#tensor_view sampleMatrix

-- Numeric summaries for the small tensors above:
#tensor_stats_view floatVector
```

# IR Graph Viewer

The IR widget family answers the three debugging questions that show up in practice:

1. *Structure*: which nodes, parents, and shapes are present?
2. *Invariants*: do declared node shapes match what the ops infer from parent shapes?
3. *Semantics*: when the graph is evaluated, which node fails first and what are the intermediate values?

```
open NN.IR
open Spec

def pairTensor (x y : Float) : Tensor Float (shape![2]) :=
  Tensor.dim (fun
    | ⟨0, _⟩ => Tensor.scalar x
    | ⟨_, _⟩ => Tensor.scalar y)

def sampleGraph : Graph :=
  { nodes := #[
      { id := 0, parents := [], kind := .input
        outShape := (shape![2]) },
      { id := 1, parents := []
        kind := .const (shape![2])
        outShape := (shape![2]) },
      { id := 2, parents := [0, 1], kind := .add
        outShape := (shape![2]) }
    ] }

def sampleGraphSub : Graph :=
  -- Same as `sampleGraph`, but uses `sub` instead of `add`.
  -- (Useful for rewrite/diff examples.)
  { nodes := #[
      { id := 0, parents := [], kind := .input
        outShape := (shape![2]) },
      { id := 1, parents := []
        kind := .const (shape![2])
        outShape := (shape![2]) },
      { id := 2, parents := [0, 1], kind := .sub
        outShape := (shape![2]) }
    ] }

#ir_view sampleGraph

-- 1) Invariant check:
-- declared shape tags vs inferred shapes.
#shape_infer_view sampleGraph

-- 2) Before/after view:
-- handy for compiler/optimizer passes.
#graph_rewrite_view sampleGraph, sampleGraphSub

-- 3) Evaluation trace: run the IR semantics step by step.
-- For `.const` nodes, a small external payload is supplied.
def sampleInput : Runtime.AnyTensor Float :=
  { s := (shape![2]), t := pairTensor 0.60 (-0.20) }

def samplePayload : NN.IR.Payload Float :=
  { const? := fun id =>
      if id = 1 then
        some { n := 2, v := pairTensor 0.25 0.25 }
      else
        none }

#ir_exec_trace_view sampleGraph, samplePayload, sampleInput
```

# Float32 Bit Layout Viewer

```
namespace Float32Demo

def one32 : IEEE32Exec :=
  IEEE32Exec.ofBits (0x3f800000 : UInt32)

def qnan32 : IEEE32Exec :=
  IEEE32Exec.ofBits (0x7fc00000 : UInt32)

#float32_view one32
#float32_view (1 : IEEE32Exec)
#float32_view qnan32
#float32_compare_view one32, qnan32

-- Compare a Float64 input to its Float32 rounding:
#float32_round_view decimalTenth
#float32_round_view oneThirdFloat

end Float32Demo
```

# Verification (IBP/CROWN State)

TorchLean's verification code includes executable bound propagation engines: IBP boxes and
CROWN affine forms. When debugging a verifier, inspect which nodes
have bounds and whether shapes and flattened dimensions match the intended layout.

```
open NN.IR
open Spec

def sampleGraphCROWN : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := [], kind := .input
        outShape := (shape![2]) },
      { id := 1, parents := []
        kind := .const (shape![2])
        outShape := (shape![2]) },
      { id := 2, parents := [0, 1], kind := .add
        outShape := (shape![2]) }
    ] }

def samplePropState :
    NN.MLTheory.CROWN.Graph.PropState Float :=
  let bIn : NN.MLTheory.CROWN.FlatBox Float :=
    { dim := 2
      lo := pairTensor (-1.0) (-1.0)
      hi := pairTensor (1.0) (1.0) }
  let bConst : NN.MLTheory.CROWN.FlatBox Float :=
    { dim := 2
      lo := pairTensor (0.25) (0.25)
      hi := pairTensor (0.25) (0.25) }
  let bOut : NN.MLTheory.CROWN.FlatBox Float :=
    { dim := 2
      lo := pairTensor (-0.75) (-0.75)
      hi := pairTensor (1.25) (1.25) }
  { inputId := 0
    inputDim := 2
    states := #[
      { shape := (shape![2])
        ibp? := some bIn
        aff? := none }
    , { shape := (shape![2])
        ibp? := some bConst
        aff? := none }
    , { shape := (shape![2])
        ibp? := some bOut
        aff? := none }
    ] }

#crown_view sampleGraphCROWN, samplePropState

-- Interval widths (`hi - lo`) are a fast
-- "where did bounds blow up?" diagnostic.
#bounds_tightness_view sampleGraphCROWN, samplePropState
```

# Autograd (Tape + Gradients)

TorchLean's eager autograd engine records a computation graph into a `Tape` and can run
reverse-mode to accumulate gradients. The widget below shows the recorded tape and the gradients
produced by scalar backprop (like `loss.backward()` in PyTorch).

```
open Runtime.Autograd
open Spec

def sampleTape : Tape Float :=
  let (t0, aId) :=
    Tape.leaf (α := Float) (t := Tape.empty)
      (value := Tensor.scalar 2.0) (name := some "a")
  let (t1, bId) :=
    Tape.leaf (α := Float) (t := t0)
      (value := Tensor.scalar 3.0) (name := some "b")
  let (t2, abId) :=
    match Tape.mul (α := Float) (t := t1)
        (s := Shape.scalar) aId bId with
    | .ok r => r
    | .error _ => (t1, 0)
  let (t3, outId) :=
    match Tape.add (α := Float) (t := t2)
        (s := Shape.scalar) abId bId with
    | .ok r => r
    | .error _ => (t2, 0)
  let _ := outId
  t3

#tape_grads_view sampleTape, 3

-- For a step by step explanation of why a grad exists (or is missing),
-- use the step by step reverse pass trace:
#tape_trace_view sampleTape, 3
```

# Training Dashboards

TorchLean's widget layer is not limited to semantic objects like tensors and tapes. It also
includes a small monitoring API for training and evaluation artifacts.

These logs are plain data structures, not a hidden runtime UI:

- `Runtime.Training.TrainLog` is a pure record of steps, metric series, and notes,
- `Runtime.Training.ConfusionMatrix` is a pure table of class counts,
- the widget layer renders them without changing their meaning.

That makes them suitable for pure Lean small runs, runtime/autograd training loops, and imported
metrics from external experiments, all rendered through the same viewer.

```
def sampleTrainLog : _root_.Runtime.Training.TrainLog :=
  { title := "Classifier training run"
    steps := #[0, 1, 2, 3, 4]
    series := #[
      { name := "loss", values := #[1.20, 0.84, 0.59, 0.41, 0.33], color := "#c44" }
    , { name := "val_acc", values := #[0.30, 0.48, 0.61, 0.73, 0.79], color := "#0a7" }
    , { name := "lr", values := #[0.05, 0.05, 0.01, 0.01, 0.01], color := "#06c" }
    ]
    notes := #[
      "optimizer: SGD"
    , "scheduler: StepLR(step_size=2, gamma=0.2)"
    , "dataset: synthetic 3-class classifier"
    ] }

def sampleLabels : Array String := #["cat", "dog", "owl"]

def sampleCM : _root_.Runtime.Training.ConfusionMatrix :=
  { counts := #[
      #[8, 1, 0]
    , #[2, 6, 1]
    , #[0, 1, 7]
    ] }

#train_log_view sampleTrainLog
#confusion_view sampleLabels, sampleCM
```

This widget family pairs particularly well with:

- the CSV loader training example,
- the NPY loader training example,
- the CNN and ViT model commands,
- and the callback/reporting helpers exposed through `Trainer` reports.

# Runtime Context Viewer

When debugging a failed training step, one of the first questions is often not "what is the graph?"
but "which values and gradients are registered now?"

The runtime-context widget answers that question directly.

```
def anyScalar (x : Float) : Runtime.AnyTensor Float :=
  { s := .scalar, t := Tensor.scalar x }

def sampleCtx : Runtime.RuntimeContext Float :=
  { var_registry := [
      ("w", anyScalar 3.0)
    , ("x", anyScalar 2.0)
    , ("wx", anyScalar 6.0)
    ]
    gradients := [
      ("w", anyScalar 2.0)
    , ("x", anyScalar 3.0)
    ]
    next_id := 3 }

#runtime_ctx_view sampleCtx
```

This view is good for comparing:

- the public training API,
- the eager autograd tape,
- and the actual runtime registry that stores values and accumulated gradients.

# GPT And Text-Model Logs

GPT-style examples write normal `TrainLog` artifacts, but prompt/sample notes benefit from a
specialized renderer:

```
#gpt2_train_log_file_view "data/model_zoo/gpt2_trainlog.json"
#gpt2_prompt_view "ROMEO:"
```

The file view is the one to use in documentation because it renders an artifact that already exists.
The prompt view can run a small command from the editor, which is convenient for demos but should be
described as execution, not passive inspection.

# RL Boundary And Policy Views

RL widgets are about the part of training that scalar reward curves hide:

- the checked transition boundary for Gymnasium rollouts,
- GridWorld policies and paths,
- PPO rollout curves derived from reward/value/advantage data.

The main entry files are:

- [GridWorld widget source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Widgets/RL/GridWorld.lean)
- [PPO widget source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Widgets/RL/PPO.lean)
- [RL boundary widget source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Widgets/RL/Boundary.lean)

Use them when the question is "what exactly entered the learner?" or "what policy/path artifact did
the command write?" Use the proof layer when the question is a theorem about the MDP or boundary.

# PyTorch Translator Widget

The PyTorch translator widget is an editor aid, not the checked importer:

```
#pytorch_translate_file "NN/Examples/Quickstart/pytorch_translator_mlp.py"
```

It helps readers see how a simple `torch.nn` snippet maps onto TorchLean constructors. Checked
interop claims should cite the explicit PyTorch roundtrip/export examples and the artifact bridge,
not the heuristic widget alone.

# Work Through The Maintained Example

Open
[NN/Examples/DeepDives/Widgets.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/DeepDives/Widgets.lean)
in VS Code with the Lean extension enabled. Put the cursor on each widget command and open the
Infoview. The file is deliberately small enough to elaborate interactively.

Work through four changes:

1. Change one entry of a tensor and confirm that `#tensor_view` and `#tensor_stats_view` update from
   the same Lean value.
2. Give an IR node the wrong declared output shape. `#shape_infer_view` should identify the first
   disagreement; restore the shape before continuing.
3. Replace `sampleGraph` by `sampleGraphSub` in the rewrite view and inspect the changed operation
   tag rather than comparing raw record syntax.
4. Compare `IEEE32Exec.posOne` with a quiet NaN. The bit viewer should expose the exponent,
   fraction, and classification that ordinary decimal printing hides.

The file can also be elaborated from a terminal:

```
lake env lean NN/Examples/DeepDives/Widgets.lean
```

Terminal elaboration checks the commands and definitions, while the VS Code Infoview provides the
interactive rendered panels. If a widget fails to elaborate, first distinguish a malformed Lean
object from a rendering problem: replace the widget command by `#check` on the same object, then
reintroduce the view after its type is correct.

## Other Widget Sources

The focused examples are useful when one artifact is the whole subject:

- [Float32 modes](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/DeepDives/Floats/Float32Modes.lean)
  pairs `#float32_view`, `#float32_compare_view`, and `#float32_round_view` with executable binary32
  values.
- [CROWN workflow](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/TorchLean/CrownOpsWorkflow.lean)
  and [IBP workflow](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/TorchLean/IBPWorkflow.lean)
  provide real verifier states for `#crown_view` and `#bounds_tightness_view`.
- [Autograd basics](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/AutogradBasics.lean)
  supplies a tape for `#tape_view`, `#tape_grads_view`, and `#tape_trace_view`.

## File-Backed Views

Training and application commands can write artifacts that outlive the Lean process. Use
`#train_log_file_view` or `#gpt2_train_log_file_view` for metric and text-generation logs, and
`#rl_boundary_rollout_file_view` for checked transition records. The
[CSV loader example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Csv.lean),
[GPT widget source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Widgets/Models/Sequence/Gpt2.lean),
and [RL rollout view](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/RL/GymnasiumRolloutView.lean)
show the corresponding producers.

`#runtime_ctx_view` is different: it renders the live runtime registry and accumulated gradients.
Use it when a training step failed before writing a log and the question is which variables reached
the runtime context.

# Keep The Object Close

A widget is most useful when it sits beside the definition that produced its input. Keep the graph
view beside the graph, the bounds view beside the verifier state, and the training plot beside the
file parser. Then a surprising picture has an immediate Lean object to inspect in the Infoview.
