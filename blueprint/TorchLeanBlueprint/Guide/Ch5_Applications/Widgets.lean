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

Suppose a verifier returns a much wider interval than expected. The value you need is already in
Lean, but it is buried in a graph record, a vector of node states, and several shape tags. Printing
the entire term would answer everything except the question you actually have: *at which node did
the bound become wide?*

That is the job of a widget. It renders an existing Lean object in the Infoview so that its relevant
structure is visible: tensor entries, graph parents, inferred shapes, Float32 fields, affine bounds,
tape gradients, or metric curves. It does not give that object a second meaning, and a convincing
picture is not a theorem.

The [widget modules](https://github.com/lean-dojo/TorchLean/tree/main/NN/Widgets/) are collected by
`NN.Widgets`. Import that module when a scratch file needs the inspection tools together; the
broader `import NN` also brings them into a larger development.

# An Inspection Loop

Start with the question, not with the widget name. If the question is “why is this output wrong?”,
inspect the graph and then its evaluation trace. If it is “why is verification inconclusive?”, look
at interval widths. If training stopped moving, compare the tape, accumulated gradients, and metric
log before changing the optimizer.

A productive session has four small moves:

1. Keep the value that matters as a named Lean definition or as an artifact at an explicit path.
2. Run the evaluator or checker whose semantics the eventual claim uses.
3. Put the corresponding widget beside that value and find the first surprising node, field, or
   step.
4. Change the definition or producer, elaborate again, and let the checker close the argument
   instead of trusting the picture.

Read-only views such as `#tensor_view` format a value without changing it. Trace views execute a
specific Lean computation: `#ir_exec_trace_view` steps through `NN.IR.Semantics`, while
`#tape_trace_view` follows the autograd engine's reverse pass. The interpretation is therefore
explicit. A theorem about that same interpretation is still a separate proof.

Here is the short lookup table; the sections below turn each row into a working example.

:::table +header
*
  * When the question is about…
  * Start with
  * What it exposes
*
  * tensor values or representation
  * `#tensor_view`, `#tensor_stats_view`, `#float32_view`
  * entries, shape, summaries, or binary32 fields
*
  * graph structure or execution
  * `#ir_view`, `#shape_infer_view`, `#ir_exec_trace_view`
  * parents, declared/inferred shapes, and intermediate values
*
  * a rewrite
  * `#graph_rewrite_view`
  * source and result graphs side by side
*
  * verifier precision
  * `#crown_view`, `#bounds_tightness_view`
  * node bounds, affine state, and interval widths
*
  * gradients
  * `#tape_grads_view`, `#tape_trace_view`, `#runtime_ctx_view`
  * tape structure, reverse steps, and accumulated gradients
*
  * a completed run
  * `#train_log_view` or a file-backed log view
  * metrics, notes, prompts, samples, policies, or transitions
*
  * a PyTorch sketch
  * `#pytorch_translate_file`
  * an editor-assistant translation, not an importer proof
:::

# How Widgets Fit Application Workflows

Some values live only while a file elaborates; others are written by a command and inspected later.
The distinction matters. A graph or tape definition can sit immediately above its view. A GPT or
PPO run instead writes a named artifact, which a later Lean file reads:

```
#train_log_file_view "data/model_zoo/cnn_trainlog.json"
#gpt2_train_log_file_view "data/model_zoo/gpt2_trainlog.json"
#rl_boundary_rollout_file_view "data/rl/cartpole_rollout.json", contract, 12
#pytorch_translate_file "NN/Examples/Quickstart/pytorch_translator_mlp.py"
```

Keep the producer command beside a file-backed view in a comment or experiment note. The renderer
can tell you what the file says, but it cannot recover where the file came from. When the artifact
has a checker, check it first and then inspect the accepted object.

# Tensor Viewer

Begin with values whose layout is unmistakable. `rankThreeGrid` writes its three indices into the
hundreds, tens, and units places, so a transposed or flattened axis is visible at a glance. The two
floating-point vectors then show a different problem: decimal printing may look the same even when
the underlying scalar semantics differ.

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

Read these three views in order. The graph view says node `2` depends on the input and constant. The
shape view checks that all three declared length-two vectors agree with inference. The trace should
then end at `[0.85, 0.05]`, because it adds `[0.25, 0.25]` to `[0.60, -0.20]`. If the final value is
wrong, the trace gives you the first intermediate value to compare; if elaboration never reaches
the trace, the shape view gives you the structural failure instead.

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

`one32` should show sign `0`, exponent field `127`, and a zero fraction. The quiet NaN has the
all-ones exponent and a nonzero fraction, so the classification remains visible even though it has
no ordinary real value. The round views answer a separate question: which binary32 bit pattern is
chosen when a binary64 `Float` such as decimal one tenth crosses the scalar boundary?

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

For this graph, adding the exact constant $`0.25` shifts $`[-1,1]` to $`[-0.75,1.25]` without
changing the interval width. The tightness view should therefore report width two at both the input
and output. In a larger network, the first unexpected jump in width is usually more informative
than the final loose bound.

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

The recorded scalar is $`ab+b` at $`a=2` and $`b=3`, so the value is nine and both derivatives are
three. That closed form gives the trace a human-sized oracle: if a gradient is absent or differs,
inspect the first reverse step where the contribution from multiplication or addition failed to
arrive.

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

- the training API,
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

File-backed views parse and render the artifact at the named path. A successful visualization does
not authenticate its producer or strengthen the artifact's checker claim. When validity matters,
run the corresponding checker first and use the widget to inspect the same accepted file.

# Keep The Object Close

A widget is most useful when it sits beside the definition that produced its input. Keep the graph
view beside the graph, the bounds view beside the verifier state, and the training plot beside the
file parser. Then a surprising picture has an immediate Lean object to inspect in the Infoview.
