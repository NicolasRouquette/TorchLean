/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import NN.Floats.IEEEExec.Exec32
import NN.IR.Graph
import NN.IR.Semantics
import NN.MLTheory.CROWN.Graph
import NN.Runtime.Autograd.Engine.Core
import NN.Spec.RL.Envs.GridWorld
import NN.Widgets
meta import NN.Spec.Core.TensorBridge

/-!
# Widget Gallery

The gallery is best explored in an editor:
- put the cursor on a `#tensor_view` / `#ir_view` / `#float32_view` line, and
- Lean will render a small interactive HTML panel in the infoview.

These widgets are inspection tools for teaching, debugging, and reviewing artifacts
without leaving Lean. They are available through the dedicated widget entrypoint
(`import NN.Widgets`) so ordinary runtime and proof imports stay focused.
-/

open Spec
open NN.IR
open TorchLean.Floats.IEEE754
open Runtime.Autograd
open TensorBridge TensorArray

/-!
## RL (GridWorld) widgets

These compact panels are useful when iterating on RL specs and proofs: they let you inspect
state encodings, policies, and rollout traces in the infoview.
-/

namespace GridWorldWidgets

open Spec.RL.Envs

/-- A small 4×4 GridWorld used by the widget gallery. -/
def galleryGridWorld : GridWorld 4 4 :=
  { start := (⟨0, by decide⟩, ⟨0, by decide⟩)
    goal := (⟨3, by decide⟩, ⟨3, by decide⟩)
    -- Discount isn't used by the widgets, so we pick a simple literal.
    discount := 0 }

/-- Example GridWorld position rendered by the state widget. -/
def pos : GridWorld.State 4 4 :=
  (⟨1, by decide⟩, ⟨2, by decide⟩)

/-- Constant policy for the policy-visualization demo. -/
def goRightPolicy : GridWorld.State 4 4 → GridWorld.Action :=
  fun _ => GridWorld.Action.right

/-- Example rollout path rendered by the GridWorld trace widget. -/
def samplePath : Array (GridWorld.State 4 4) :=
  #[
    (⟨0, by decide⟩, ⟨0, by decide⟩),
    (⟨0, by decide⟩, ⟨1, by decide⟩),
    (⟨0, by decide⟩, ⟨2, by decide⟩),
    (⟨0, by decide⟩, ⟨3, by decide⟩),
    (⟨1, by decide⟩, ⟨3, by decide⟩),
    (⟨2, by decide⟩, ⟨3, by decide⟩),
    (⟨3, by decide⟩, ⟨3, by decide⟩)
  ]

#gridworld_view galleryGridWorld, pos
#gridworld_policy_view galleryGridWorld, goRightPolicy
#gridworld_path_view galleryGridWorld, samplePath

end GridWorldWidgets

def sampleTrainLog : Runtime.Training.TrainLog :=
  let n : Nat := 40
  let steps : Array Nat := (Array.range n).map (fun i => i)
  let loss : Array Float :=
    (Array.range n).map (fun i =>
      let t : Float := Float.ofNat i
      -- A compact decreasing curve with a small oscillation, useful for checking rendering.
      (Float.exp (-0.08 * t)) + 0.03 * Float.sin (0.7 * t))
  let acc : Array Float :=
    (Array.range n).map (fun i =>
      let t : Float := Float.ofNat i
      -- A compact increasing curve that saturates, like a validation metric.
      0.4 + 0.6 * (1.0 - Float.exp (-0.12 * t)))
  { title := "Sample training loop"
    steps := steps
    series := #[
      { name := "loss", values := loss, color := "#c44" }
    , { name := "acc", values := acc, color := "#0a7" }
    ]
    notes := #[
      "sample curve for checking the widget layout"
    , "the same viewer renders JSON logs written by executable training loops"
    ] }

/-!
Actual training curve (SGD on a small regression dataset).

This is a real training loop computed in Lean. The model is compact (2→2→1 MLP), which keeps the
widget responsive inside the editor.

Note:
We keep this example **self-contained** (no Spec-tensor forward/backward), because the full
spec-side MLP backprop pipeline is large and would make this widget file slow to elaborate.
-/

abbrev FloatPair := Float × Float
abbrev PairMatrix := FloatPair × FloatPair

private def vsub (a b : FloatPair) : FloatPair := (a.1 - b.1, a.2 - b.2)
private def vscale (a : FloatPair) (c : Float) : FloatPair := (a.1 * c, a.2 * c)
private def dot (w x : FloatPair) : Float := w.1 * x.1 + w.2 * x.2

private def relu (x : Float) : Float := if x <= 0.0 then 0.0 else x
private def drelu (x : Float) : Float := if x <= 0.0 then 0.0 else 1.0
private def reluPair (x : FloatPair) : FloatPair := (relu x.1, relu x.2)

private def tinyData : Array (FloatPair × Float) :=
  -- y = x1 + 2*x2, sampled at a few points
  #[
    ((0.0, 0.0), 0.0)
  , ((1.0, -1.0), -1.0)
  , ((1.0, 0.0), 1.0)
  , ((0.0, 1.0), 2.0)
  ]

private theorem tinyData_nonempty : 0 < tinyData.size := by
  decide

private structure Params where
  hiddenWeights : PairMatrix
  hiddenBias : FloatPair
  outputWeights : FloatPair
  outputBias : Float

private def initParams : Params :=
  { hiddenWeights := ((0.12, -0.08), (0.05, 0.10))
    hiddenBias := (0.01, -0.02)
    outputWeights := (0.07, -0.04)
    outputBias := 0.0 }

private def lr : Float := 0.03

private def forward (p : Params) (x : FloatPair) : (Float × FloatPair × FloatPair) :=
  -- Returns the prediction, hidden pre-activation, and hidden activation.
  let hiddenPreactivation : FloatPair :=
    (dot p.hiddenWeights.1 x + p.hiddenBias.1, dot p.hiddenWeights.2 x + p.hiddenBias.2)
  let hiddenActivation : FloatPair := reluPair hiddenPreactivation
  let yhat : Float := dot p.outputWeights hiddenActivation + p.outputBias
  (yhat, hiddenPreactivation, hiddenActivation)

private def sgdStep (p : Params) (x : FloatPair) (y : Float) : (Params × Float × Float) :=
  let (yhat, hiddenPreactivation, hiddenActivation) := forward p x
  let err := yhat - y
  let loss := 0.5 * err * err
  -- Gradients.
  let dLdy : Float := err
  let outputWeightGrad : FloatPair := vscale hiddenActivation dLdy
  let outputBiasGrad : Float := dLdy
  let hiddenActivationGrad : FloatPair := vscale p.outputWeights dLdy
  let hiddenPreactivationGrad : FloatPair :=
    (hiddenActivationGrad.1 * drelu hiddenPreactivation.1,
      hiddenActivationGrad.2 * drelu hiddenPreactivation.2)
  let firstHiddenRowGrad : FloatPair := vscale x hiddenPreactivationGrad.1
  let secondHiddenRowGrad : FloatPair := vscale x hiddenPreactivationGrad.2
  let hiddenBiasGrad : FloatPair := hiddenPreactivationGrad
  -- SGD update.
  let p' : Params :=
    { hiddenWeights :=
        (vsub p.hiddenWeights.1 (vscale firstHiddenRowGrad lr),
          vsub p.hiddenWeights.2 (vscale secondHiddenRowGrad lr))
      hiddenBias := vsub p.hiddenBias (vscale hiddenBiasGrad lr)
      outputWeights := vsub p.outputWeights (vscale outputWeightGrad lr)
      outputBias := p.outputBias - lr * outputBiasGrad }
  let absErr := Float.abs err
  (p', loss, absErr)

private def trainLoop : Nat → Params → List Float → List Float → (List Float × List Float)
  | 0, _, losses, errs => (losses.reverse, errs.reverse)
  | Nat.succ k, p, losses, errs =>
      let idx : Fin tinyData.size := ⟨k % tinyData.size, Nat.mod_lt k tinyData_nonempty⟩
      let sample := tinyData[idx]
      let (x, y) := sample
      let (p', loss, absErr) := sgdStep p x y
      trainLoop k p' (loss :: losses) (absErr :: errs)

def mlpTrainLog : Runtime.Training.TrainLog :=
  let stepsN : Nat := 80
  let (losses, errs) := trainLoop stepsN initParams [] []
  { title := "MLP SGD (real run, pure Lean)"
    steps := (Array.range stepsN).map (fun i => i)
    series := #[
      { name := "mse_loss", values := losses.toArray, color := "#c44" }
    , { name := "abs_err", values := errs.toArray, color := "#0a7" }
    ]
    notes := #[
      "model: 2->2->1 ReLU MLP (scalar implementation for speed)"
    , "data: y = x1 + 2*x2 (4 points)"
    , s!"lr={lr}, steps={stepsN}"
    ] }

def sampleLabels : Array String := #["cat", "dog", "owl"]

def sampleConfusionMatrix : Runtime.Training.ConfusionMatrix :=
  { counts := #[
      #[8, 1, 0]
    , #[2, 6, 1]
    , #[0, 1, 7]
    ] }

def indexVector : Tensor Nat (shape![5]) :=
  Tensor.dim (fun i => Tensor.scalar i.1)

def rankThreeGrid : Tensor Nat (shape![2, 3, 4]) :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      Tensor.dim (fun k =>
        Tensor.scalar (i.1 * 100 + j.1 * 10 + k.1))))

def decimalTenth : Float :=
  -- 0.1 as a binary64 literal (exact via bit pattern).
  Float.ofBits 0x3fb999999999999a

def oneThirdFloat : Float :=
  -- 1/3 as a binary64 literal.
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
    | ⟨1, _⟩ => Tensor.scalar (IEEE32Exec.ofFloat (Float.ofNat 2))
    | ⟨2, _⟩ => Tensor.scalar (IEEE32Exec.ofFloat decimalTenth)
    | ⟨_, _⟩ => Tensor.scalar (IEEE32Exec.ofFloat oneThirdFloat))

def ieeeCube : Tensor IEEE32Exec (shape![2, 2, 3]) :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      Tensor.dim (fun k =>
        -- Small rank-3 tensor with values that make bit-patterns interesting.
        let base : Float := Float.ofNat (i.1 * 100 + j.1 * 10 + k.1)
        let x : Float := (base + decimalTenth) / 7.0
        Tensor.scalar (IEEE32Exec.ofFloat x))))

def sampleMatrix : Tensor Int (shape![2, 4]) :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      Tensor.scalar (Int.ofNat (i.1 * 10 + j.1))))

def anyMat : Spec.PackedTensor Int :=
  Spec.PackedTensor.ofTensor sampleMatrix

def anyF (x : Float) : Spec.PackedTensor Float :=
  Spec.PackedTensor.ofTensor (Tensor.scalar x)

def sampleRuntimeContext : Runtime.RuntimeContext Float :=
  { bindings := [
      ("w", anyF 3.0)
    , ("x", anyF 2.0)
    ]
    gradients := [
      ("w", anyF 0.1)
    , ("x", anyF 0.0)
    ]
    nextId := 2 }

def sampleGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := [], kind := .input
        outShape := (shape![2]) },
      { id := 1, parents := []
        kind := .const (shape![2])
        outShape := (shape![2]) },
      { id := 2, parents := [0, 1], kind := .add
        outShape := (shape![2]) }
    ] }

def sampleGraphSub : NN.IR.Graph :=
  -- Same as `sampleGraph` but with `sub` instead of `add` at the output.
  { nodes := #[
      { id := 0, parents := [], kind := .input
        outShape := (shape![2]) },
      { id := 1, parents := []
        kind := .const (shape![2])
        outShape := (shape![2]) },
      { id := 2, parents := [0, 1], kind := .sub
        outShape := (shape![2]) }
    ] }

def pairTensor (x y : Float) : Tensor Float (shape![2]) :=
  Tensor.dim (fun
    | ⟨0, _⟩ => Tensor.scalar x
    | ⟨_, _⟩ => Tensor.scalar y)

def sampleInput : Spec.PackedTensor Float :=
  Spec.PackedTensor.ofTensor (pairTensor 0.60 (-0.20))

def samplePayload : NN.IR.Payload Float :=
  { const? := fun id =>
      if id = 1 then
        -- Node 1 is the `.const` in `sampleGraph` (a fixed vector).
        some { n := 2, v := pairTensor 0.25 0.25 }
      else
        none }

def one : IEEE32Exec :=
  IEEE32Exec.ofBits (0x3f800000 : UInt32)

def qnan : IEEE32Exec :=
  IEEE32Exec.ofBits (0x7fc00000 : UInt32)

def samplePropState : NN.MLTheory.CROWN.Graph.PropState Float :=
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
      { shape := (shape![2]), ibp? := some bIn, aff? := none }
    , { shape := (shape![2]), ibp? := some bConst, aff? := none }
    , { shape := (shape![2]), ibp? := some bOut, aff? := none }
    ] }

def sampleTape : Tape Float :=
  let (t0, aId) := Tape.leaf (α := Float) (t := Tape.empty) (value := Tensor.scalar 2.0) (name :=
    some "a")
  let (t1, bId) := Tape.leaf (α := Float) (t := t0) (value := Tensor.scalar 3.0) (name := some "b")
  let (t2, abId) :=
    match Tape.mul (α := Float) (t := t1) (s := Shape.scalar) aId bId with
    | .ok r => r
    | .error _ => (t1, 0)
  let (t3, outId) :=
    match Tape.add (α := Float) (t := t2) (s := Shape.scalar) abId bId with
    | .ok r => r
    | .error _ => (t2, 0)
  -- In this construction we expect ids [0=a,1=b,2=mul,3=add].
  -- We keep the final tape even if a future implementation changes ids.
  let _ := outId
  t3

/-!
Tensor basics / bridge example:

Widgets are defined for `Spec.Tensor` (shape-indexed, spec-level tensors). If you have an
array-backed `TensorArray.Tensor` (common at IO boundaries), you can convert it with
`TensorBridge.to_tensor` and then use the same `#tensor_view` UI.
-/

meta def taMat23 : TensorArray.Tensor Float [2, 3] :=
  TensorArray.ofArray #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0] [2, 3] (by simp)

meta def taMat23_spec : Tensor Float (Shape.ofList [2, 3]) :=
  toTensor taMat23

-- Try hovering/cursoring on these commands in the editor.
#tensor_view indexVector
#tensor_view rankThreeGrid
#tensor_view floatVector
#tensor_view ieeeVector
#tensor_view ieeeCube
#tensor_view sampleMatrix
#tensor_view taMat23_spec
#tensor_stats_view floatVector
#tensor_stats_view (pairTensor 0.60 (-0.20))
#ir_view sampleGraph
#shape_infer_view sampleGraph
#graph_rewrite_view sampleGraph, sampleGraphSub
#float32_view one
#float32_view (1 : IEEE32Exec)
#float32_view qnan
#float32_compare_view one, qnan
#anytensor_view anyMat
#runtime_ctx_view sampleRuntimeContext
#ir_exec_trace_view sampleGraph, samplePayload, sampleInput
#train_log_view mlpTrainLog
#train_log_view sampleTrainLog
#confusion_view sampleLabels, sampleConfusionMatrix

-- Compare Float64 input to its Float32 rounding.
#float32_round_view decimalTenth
#float32_round_view oneThirdFloat

-- Verification: show a small CROWN/IBP state aligned with `sampleGraph`.
#crown_view sampleGraph, samplePropState
#bounds_tightness_view sampleGraph, samplePropState

-- Autograd: show a compact tape and its scalar backprop (like `loss.backward()`).
#tape_view sampleTape
#tape_grads_view sampleTape, 3
#tape_trace_view sampleTape, 3
