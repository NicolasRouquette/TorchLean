/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TypedGraph.GraphM
public import NN.Runtime.Autograd.Torch.Core.TypedGraph
public import NN.API.Neural.Execution
public import NN.API.Optim
public import NN.Runtime.Autograd.Train
public import NN.Spec.Core.Tensor
public import NN.Spec.Models.Mlp
public import NN.Tensor

/-!
# Consolidated Float Runtime Autograd Tests

This file collects runtime tests that exercise the *dynamic autograd tape*.
-/

@[expose] public section


/-! ## autograd_engine_test.lean -/

/-!
Regression tests for `Runtime.Autograd` dynamic tape.

We check that for a simple 2-layer MLP, the tape-based gradients match the existing
hand-derived `Examples.mlp_backward`.
-/

open _root_.Spec
open _root_.Spec.Tensor
open Examples

namespace Tests
namespace Floats
namespace AutogradEngine

open Runtime.Autograd

abbrev inDim  := 2
abbrev hidDim := 3
abbrev outDim := 1

-- Small tag used for readable error messages.
abbrev tag : String := "autograd_engine_test"

-- Parameter node ids we want to read gradients for.
structure ParamIds where
  /-- w 1 Id. -/
  hiddenWeightId : Nat
  /-- b 1 Id. -/
  hiddenBiasId : Nat
  /-- w 2 Id. -/
  outputWeightId : Nat
  /-- b 2 Id. -/
  outputBiasId : Nat

/-!
## Fixed inputs and parameters

We use a small deterministic 2-layer MLP so the gradients are stable.
-/
def hiddenWeight : Tensor Float [hidDim, inDim] :=
  tensorOfArray! [hidDim, inDim] #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

def hiddenBias : Tensor Float [hidDim] :=
  tensorOfArray! [hidDim] #[0.1, 0.2, 0.3]

def outputWeight : Tensor Float [outDim, hidDim] :=
  tensorOfArray! [outDim, hidDim] #[0.7, 0.8, 0.9]

def outputBias : Tensor Float [outDim] :=
  tensorOfArray! [outDim] #[0.4]

def x : Tensor Float [inDim] :=
  tensorOfArray! [inDim] #[0.5, 0.8]

def dLdy : Tensor Float [outDim] :=
  tensorOfArray! [outDim] #[1.0]

def hiddenLayer : Spec.LinearSpec Float inDim hidDim := { weights := hiddenWeight, bias := hiddenBias }
def outputLayer : Spec.LinearSpec Float hidDim outDim := { weights := outputWeight, bias := outputBias }

def expected :=
  Examples.mlpBackward hiddenLayer outputLayer x dLdy

/-!
## Test: dynamic tape gradients vs. reference

We compare the autograd tape gradients against the hand-derived MLP backward pass.
-/
def checkMlpGrads :
  Runtime.Autograd.Result Bool := do
  let t0 : Tape Float := Tape.empty

  -- Build the graph in TapeM for readability.
  let m : TapeM Float _ := do
    let hiddenWeightId ← Train.TapeM.param hiddenWeight (name := some "hiddenWeight")
    let hiddenBiasId ← Train.TapeM.param hiddenBias (name := some "hiddenBias")
    let outputWeightId ← Train.TapeM.param outputWeight (name := some "outputWeight")
    let outputBiasId ← Train.TapeM.param outputBias (name := some "outputBias")
    let xId ← Train.TapeM.const x (name := some "x")

    -- Forward pass: linear -> relu -> linear
    let z1Id ← TapeM.linear (inDim:=inDim) (outDim:=hidDim) hiddenWeightId hiddenBiasId xId
    let a1Id ← TapeM.relu (s := [hidDim]) z1Id
    let yId ← TapeM.linear (inDim:=hidDim) (outDim:=outDim) outputWeightId outputBiasId a1Id

    let t ← TapeM.getTape
    let grads ← liftM (Tape.backward (t:=t) yId (Spec.SomeTensor.ofTensor dLdy))

    let ids : ParamIds := { hiddenWeightId := hiddenWeightId, hiddenBiasId := hiddenBiasId, outputWeightId := outputWeightId, outputBiasId := outputBiasId }
    pure (ids, grads)

  let ((ids, grads), _) ← TapeM.run t0 m

  let (dW1_exp, db1_exp, dW2_exp, db2_exp, _dX_exp) := expected

  let dW1_dyn ← Train.requireGradTensor (tag := tag)
    (s := [hidDim, inDim]) grads ids.hiddenWeightId
  let db1_dyn ← Train.requireGradTensor (tag := tag)
    (s := [hidDim]) grads ids.hiddenBiasId
  let dW2_dyn ← Train.requireGradTensor (tag := tag)
    (s := [outDim, hidDim]) grads ids.outputWeightId
  let db2_dyn ← Train.requireGradTensor (tag := tag)
    (s := [outDim]) grads ids.outputBiasId

  let ok1 := decide (pretty dW1_dyn = pretty dW1_exp)
  let ok2 := decide (pretty db1_dyn = pretty db1_exp)
  let ok3 := decide (pretty dW2_dyn = pretty dW2_exp)
  let ok4 := decide (pretty db2_dyn = pretty db2_exp)
  pure (ok1 && ok2 && ok3 && ok4)

def run : IO Unit := do
  match checkMlpGrads with
  | .ok true => IO.println "autograd_engine_test (Float): OK"
  | .ok false => throw <| IO.userError "autograd_engine_test (Float): FAILED"
  | .error msg => throw <| IO.userError s!"autograd_engine_test (Float): {msg}"

end AutogradEngine
end Floats
end Tests

/-! ## autograd_linear_regression_test.lean -/

/-!
# Autograd linear regression (Float)

This file is a small, end-to-end training regression test for the *dynamic autograd tape*.

We fit a 1D linear model to a small dataset:

$$
y=2x+1
$$

using SGD on the mean-squared-error (MSE) loss.

Key things to notice when reading the code:
* The forward pass is written in the `TapeM` style, so the tape is threaded implicitly.
* Dataset inputs/targets are created with `requiresGrad = false` (they are constants).
* After the forward pass, we call `backwardScalar` to get a gradient map `id -> grad`.
* Parameters are updated with a simple tensor-level SGD update rule.
-/

open _root_.Spec
open _root_.Spec.Tensor

namespace Tests
namespace Floats
namespace AutogradLinearRegression

open Runtime.Autograd

-- A short tag used for readable error messages.
abbrev tag : String := "autograd_linear_regression_test"

abbrev inDim := 1
abbrev outDim := 1

-- One training example: (x, y)
abbrev Sample := Prod Float Float

-- A small dataset: y = 2x + 1
def dataset : Array Sample :=
  #[ (0.0, 1.0)
   , (1.0, 3.0)
   , (2.0, 5.0)
   , (3.0, 7.0)
   ]

-- Expose the examples through the reusable finite stream abstraction.
def testDataset : TorchLean.Data.SampleStream Sample :=
  TorchLean.Data.SampleStream.ofArray dataset

-- Model parameters (W, b) for y = W * x + b
structure Params where
  /-- W. -/
  W : Tensor Float [outDim, inDim]
  /-- b. -/
  b : Tensor Float [outDim]

-- Initial parameters (not too close to the target).
def initParams : Params :=
  { W := fill (0.5 : Float) [outDim, inDim]
  , b := fill (0.0 : Float) [outDim]
  }

-- Optimizer config: ids are stable because we create W then b each step.
def lrScheduler : Train.LRScheduler Float :=
  .linearWarmup (Optim.linearWarmupScheduler (initialLr := 0.2) (warmupSteps := 2)
    (startLr := 0.05))

def initialOptimizerState : Train.OptimizerState Float :=
  { kind := .adamw
  , groups :=
      #[{ params := #[0, 1]
        , lr := 0.2
        , weightDecay := 0.0
        , scheduler := some lrScheduler
        }]
  }

-- Training state for the trainer API.
structure TrainState where
  /-- params. -/
  params : Params
  /-- opt. -/
  opt : Train.OptimizerState Float

def initState : TrainState := { params := initParams, opt := initialOptimizerState }

-- Single-sample loss using the tape.
def sampleLoss (WId bId : Nat) (sample : Sample) :
  Runtime.Autograd.TapeM Float Nat := do
  let xVal : Tensor Float [inDim] := fill sample.fst [inDim]
  let yVal : Tensor Float [outDim] := fill sample.snd [outDim]
  let xId ← Train.TapeM.const xVal (name := some "x")
  let yId ← Train.TapeM.const yVal (name := some "y")
  let yHatId ← TapeM.linear (inDim:=inDim) (outDim:=outDim) WId bId xId
  let lossId ← TapeM.mseLoss (s := [outDim]) yHatId yId
  pure lossId

-- One optimizer-backed training step over a batch of samples.
def trainStep
  (s : TrainState) (batch : Array Sample) :
  Runtime.Autograd.Result (Prod TrainState Float) := do
  let t0 : Tape Float := Tape.empty
  let m : TapeM Float _ := do
    let wId ← Train.TapeM.param s.params.W (name := some "W")
    let bId ← Train.TapeM.param s.params.b (name := some "b")
    let lossId ← Train.TapeM.meanScalarOver (tag := tag) batch (fun sample => sampleLoss wId bId
      sample)
    let t ← TapeM.getTape
    let lossVal ← liftM (Train.requireScalarValue (tag := tag) t lossId)
    let grads ← liftM (Tape.backwardScalar (t:=t) lossId)
    pure (wId, bId, lossVal, grads)

  let ((wId, bId, lossVal, grads), _) ← TapeM.run t0 m

  let paramTable : Train.ParamTable Float :=
    #[Train.ParamEntry.ofTensor wId s.params.W (name := some "W")
    , Train.ParamEntry.ofTensor bId s.params.b (name := some "b")
    ]

  let (opt', paramTable') ← Train.Optim.step s.opt paramTable grads

  let newW ← Train.ParamTable.getTensor (tag := tag)
    (s := [outDim, inDim]) paramTable' wId
  let newb ← Train.ParamTable.getTensor (tag := tag)
    (s := [outDim]) paramTable' bId

  let newParams : Params := { W := newW, b := newb }
  pure ({ params := newParams, opt := opt' }, lossVal)

-- One trainer step over the fixed dataset.
def step (s : TrainState) :
  Runtime.Autograd.Result (Prod TrainState (Train.StepReport Float)) := do
  let (s', loss) ← trainStep s dataset
  pure (s', { loss := loss, metrics := #[] })

def trainer : Train.Trainer Runtime.Autograd.Result TrainState Float :=
  Train.Trainer.noLog initState step

/-!
## Training

Run a small number of epochs and report the per-epoch loss.
-/
def train (epochs : Nat) :
  Runtime.Autograd.Result (Prod Params (Array Float)) := do
  let (s, losses) ← Train.Trainer.runLosses (steps := epochs) trainer
  pure (s.params, losses)

/-!
## Evaluation

This runs a forward pass only (no backprop) and averages loss over a dataset.
-/
def evalSample (p : Params) : Sample -> Runtime.Autograd.Result (Train.StepReport Float)
  | sample => do
      let t0 : Tape Float := Tape.empty
      let m : TapeM Float _ := do
        let wId ← Train.TapeM.const p.W (name := some "W")
        let bId ← Train.TapeM.const p.b (name := some "b")
        let lossId ← sampleLoss wId bId sample
        let t ← TapeM.getTape
        let lossVal ← liftM (Train.requireScalarValue (tag := tag) t lossId)
        pure lossVal
      let (lossVal, _) ← TapeM.run t0 m
      pure { loss := lossVal, metrics := #[] }

def evalDataset (p : Params) : Runtime.Autograd.Result (Train.StepReport Float) :=
  Train.Eval.evalDataset (tag := tag) testDataset (evalSample p)

def run : IO Unit := do
  let res :=
    (Train.Trainer.run (steps := 5) trainer) >>= fun (s, reports) => do
      let evalReport ← evalDataset s.params
      pure (Train.renderReports reports, Train.renderReport 0 evalReport,
        pretty s.params.W, pretty s.params.b)
  match res with
  | .error msg => throw <| IO.userError s!"autograd_linear_regression_test (Float): {msg}"
  | .ok (reports, evalReport, wStr, bStr) =>
    IO.println "=== Autograd linear regression (Float) ==="
    for line in reports do
      IO.println line
    IO.println evalReport
    IO.println s!"W: {wStr}"
    IO.println s!"b: {bStr}"

end AutogradLinearRegression
end Floats
end Tests

/-! ## autograd_train_test.lean -/

/-!
End-to-end training runtime check using the dynamic autograd tape.

This mirrors the hand-written SGD loop in `mlp_test.lean`, but gradients are produced
via the dynamic tape (`Runtime.Autograd.Tape.backwardScalar`).
-/

open _root_.Spec
open _root_.Spec.Tensor

namespace Tests
namespace Floats
namespace AutogradTrain

open Runtime.Autograd

abbrev inDim  := 2
abbrev hidDim := 3
abbrev outDim := 1

-- Small tag used for readable error messages.
abbrev tag : String := "autograd_train_test"

/-!
## Parameter record

We keep parameters in a small structure so we can update them together after each step.
-/
structure Params where
  /-- Weight matrix for layer 1. -/
  hiddenWeight : Tensor Float [hidDim, inDim]
  /-- Bias for layer 1. -/
  hiddenBias : Tensor Float [hidDim]
  /-- Weight matrix for layer 2. -/
  outputWeight : Tensor Float [outDim, hidDim]
  /-- Bias for layer 2. -/
  outputBias : Tensor Float [outDim]

-- A fixed initialization so the test is deterministic.
def initParams : Params :=
  {
    hiddenWeight := tensorOfArray! [hidDim, inDim] #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    hiddenBias := tensorOfArray! [hidDim] #[0.1, 0.2, 0.3],
    outputWeight := tensorOfArray! [outDim, hidDim] #[0.7, 0.8, 0.9],
    outputBias := tensorOfArray! [outDim] #[0.4]
  }

def x : Tensor Float [inDim] :=
  tensorOfArray! [inDim] #[0.5, 0.8]

def yTarget : Tensor Float [outDim] :=
  tensorOfArray! [outDim] #[1.0]

/-!
## One training step

We explicitly build the tape with `Tape.*` operations, then call `backwardScalar`.
-/
def trainStep (p : Params) (lr : Float := 0.1) : Runtime.Autograd.Result (Prod Params Float) := do
  let t0 : Tape Float := Tape.empty
  let (t1, hiddenWeightId) := Tape.leaf (t:=t0) p.hiddenWeight (name := some "hiddenWeight")
  let (t2, hiddenBiasId) := Tape.leaf (t:=t1) p.hiddenBias (name := some "hiddenBias")
  let (t3, outputWeightId) := Tape.leaf (t:=t2) p.outputWeight (name := some "outputWeight")
  let (t4, outputBiasId) := Tape.leaf (t:=t3) p.outputBias (name := some "outputBias")
  let (t5, xId)  := Tape.leaf (t:=t4) x (name := some "x") (requiresGrad := false)
  let (t6, yId)  := Tape.leaf (t:=t5) yTarget (name := some "y") (requiresGrad := false)

  -- Forward pass: linear -> relu -> linear -> mse_loss
  let (t7, z1Id) ← Tape.linear (t:=t6) (inDim:=inDim) (outDim:=hidDim) hiddenWeightId hiddenBiasId xId
  let (t8, a1Id) ← Tape.relu (t := t7) (s := [hidDim]) z1Id
  let (t9, yhatId) ← Tape.linear (t:=t8) (inDim:=hidDim) (outDim:=outDim) outputWeightId outputBiasId a1Id
  let (t10, lossId) ← Tape.mseLoss (t := t9) (s := [outDim]) yhatId yId

  -- Read loss and backpropagate from the scalar loss node.
  let lossVal ← Train.requireScalarValue (tag := tag) t10 lossId
  let grads ← Tape.backwardScalar (t:=t10) lossId

  -- Extract typed gradients and apply SGD updates.
  let hiddenWeightGrad ← Train.requireGradTensor (tag := tag)
    (s := [hidDim, inDim]) grads hiddenWeightId
  let hiddenBiasGrad ← Train.requireGradTensor (tag := tag)
    (s := [hidDim]) grads hiddenBiasId
  let outputWeightGrad ← Train.requireGradTensor (tag := tag)
    (s := [outDim, hidDim]) grads outputWeightId
  let outputBiasGrad ← Train.requireGradTensor (tag := tag)
    (s := [outDim]) grads outputBiasId

  let updatedHiddenWeight := Optim.SGD.update { lr := lr } p.hiddenWeight hiddenWeightGrad
  let updatedHiddenBias := Optim.SGD.update { lr := lr } p.hiddenBias hiddenBiasGrad
  let updatedOutputWeight := Optim.SGD.update { lr := lr } p.outputWeight outputWeightGrad
  let updatedOutputBias := Optim.SGD.update { lr := lr } p.outputBias outputBiasGrad

  pure ({ hiddenWeight := updatedHiddenWeight, hiddenBias := updatedHiddenBias, outputWeight := updatedOutputWeight, outputBias := updatedOutputBias }, lossVal)

/-!
## Training loop

Run a small fixed number of epochs and collect the per-epoch loss.
-/
def train (epochs : Nat) (lr : Float := 0.1) :
  Runtime.Autograd.Result (Array Float) := do
  let (_, losses) ← Train.runStepsM (m := Runtime.Autograd.Result) epochs initParams
    (fun p => trainStep p lr)
  pure losses

def run : IO Unit := do
  match train 6 0.1 with
  | .ok losses =>
    IO.println "=== Autograd train runtime check (Float) ==="
    IO.println s!"losses: {losses}"
  | .error msg => throw <| IO.userError s!"autograd_train_test (Float): {msg}"

end AutogradTrain
end Floats
end Tests

/-! ## autograd_layernorm_test.lean -/

/-!
Small layer norm gradient runtime check using the dynamic tape.
-/

open _root_.Spec
open _root_.Spec.Tensor

namespace Tests
namespace Floats
namespace AutogradLayerNorm

open Runtime.Autograd

abbrev seqLen := 2
abbrev embedDim := 3

def x : Tensor Float [seqLen, embedDim] :=
  tensorOfArray! [seqLen, embedDim] #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

def gamma : Tensor Float [embedDim] :=
  tensorOfArray! [embedDim] #[1.0, 0.9, 1.1]

def beta : Tensor Float [embedDim] :=
  tensorOfArray! [embedDim] #[0.0, 0.1, -0.1]

def checkLayerNormGrads :
  Runtime.Autograd.Result (String × String × String) := do
  let t0 : Tape Float := Tape.empty
  let m : TapeM Float _ := do
    let xId ← Train.TapeM.param x (name := some "x")
    let gammaId ← Train.TapeM.param gamma (name := some "gamma")
    let betaId ← Train.TapeM.param beta (name := some "beta")
    let yId ← TapeM.layerNorm (seqLen := seqLen) (embedDim := embedDim) (by decide) (by decide) xId
      gammaId betaId
    let lossId ← TapeM.sum (s := [seqLen, embedDim]) yId
    let t ← TapeM.getTape
    let lossVal ← liftM (Train.requireScalarValue (tag := "layer_norm") t lossId)
    let grads ← liftM (Tape.backwardScalar (t := t) lossId)
    pure (xId, gammaId, betaId, lossVal, grads)

  let ((xId, gammaId, betaId, lossVal, grads), _) ← TapeM.run t0 m

  let dX ← Train.requireGradTensor (tag := "layer_norm")
    (s := [seqLen, embedDim]) grads xId
  let dGamma ← Train.requireGradTensor (tag := "layer_norm")
    (s := [embedDim]) grads gammaId
  let dBeta ← Train.requireGradTensor (tag := "layer_norm")
    (s := [embedDim]) grads betaId

  pure (s!"loss={lossVal}", pretty dGamma, pretty dBeta)

def run : IO Unit := do
  match checkLayerNormGrads with
  | .error msg => throw <| IO.userError s!"autograd_layernorm_test (Float): {msg}"
  | .ok (lossStr, dGammaStr, dBetaStr) =>
    IO.println "=== Autograd layer norm grad runtime check (Float) ==="
    IO.println lossStr
    IO.println s!"dGamma: {dGammaStr}"
    IO.println s!"dBeta: {dBetaStr}"

end AutogradLayerNorm
end Floats
end Tests

/-! ## Convolution gradient check -/

/-!
Gradient runtime check for the rank-polymorphic convolution operation, instantiated here with two
spatial axes.
-/

open _root_.Spec
open _root_.Spec.Tensor

namespace Tests
namespace Floats
namespace AutogradConv

open Runtime.Autograd

abbrev inC := 1
abbrev outC := 1
abbrev kH := 2
abbrev kW := 2
abbrev stride := 1
abbrev padding := 0
abbrev inH := 2
abbrev inW := 2

theorem h1 : inC ≠ 0 := by decide
theorem h2 : kH ≠ 0 := by decide
theorem h3 : kW ≠ 0 := by decide

def outH : Nat := Spec.Shape.slidingWindowOutDim inH kH stride padding
def outW : Nat := Spec.Shape.slidingWindowOutDim inW kW stride padding

def kernel : Tensor Float [outC, inC, kH, kW] :=
  tensorOfArray! [outC, inC, kH, kW] #[0.2, -0.1, 0.3, 0.4]

def bias : Tensor Float [outC] :=
  tensorOfArray! [outC] #[0.05]

def input : Tensor Float [inC, inH, inW] :=
  tensorOfArray! [inC, inH, inW] #[1.0, 2.0, 3.0, 4.0]

def checkConvGrads :
  Runtime.Autograd.Result (String × String) := do
  let t0 : Tape Float := Tape.empty
  let m : TapeM Float _ := do
    let kId ← Train.TapeM.param kernel (name := some "kernel")
    let bId ← Train.TapeM.param bias (name := some "bias")
    let xId ← Train.TapeM.const input (name := some "input")
    let yId ← TapeM.conv (d := 2) (inC := inC) (outC := outC)
      (kernel := tensor! [kH, kW]) (stride := tensor! [stride, stride])
      (padding := tensor! [padding, padding]) (inSpatial := tensor! [inH, inW]) kId bId xId
    let lossId ← TapeM.sum (s := [outC, outH, outW]) yId
    let t ← TapeM.getTape
    let grads ← liftM (Tape.backwardScalar (t := t) lossId)
    pure (kId, bId, grads)

  let ((kId, bId, grads), _) ← TapeM.run t0 m
  let dK ← Train.requireGradTensor (tag := "conv")
    (s := [outC, inC, kH, kW]) grads kId
  let dB ← Train.requireGradTensor (tag := "conv")
    (s := [outC]) grads bId
  pure (pretty dK, pretty dB)

def run : IO Unit := do
  match checkConvGrads with
  | .error msg => throw <| IO.userError s!"autograd_conv_test (Float): {msg}"
  | .ok (dKStr, dBStr) =>
    IO.println "=== Autograd convolution gradient runtime check (Float) ==="
    IO.println s!"dK: {dKStr}"
    IO.println s!"dB: {dBStr}"

end AutogradConv
end Floats
end Tests

/-! ## Typed graph log-softmax JVP -/

namespace Tests
namespace Floats
namespace TypedGraphLogSoftmaxJvp

open _root_.Spec
open _root_.Spec.Tensor

/-- Check that typed graph log-softmax uses its JVP rather than its distinct reverse-mode VJP. -/
def run : IO Unit := do
  let vectorShape : Shape := [2]
  let build :
      Runtime.Autograd.TypedGraph.GraphM.M Float [vectorShape]
        (Runtime.Autograd.TypedGraph.GraphM.Var vectorShape) := do
    let x ← Runtime.Autograd.TypedGraph.GraphM.arg
      (α := Float) (Γ := [vectorShape]) 0 vectorShape
    Runtime.Autograd.TypedGraph.GraphM.logSoftmax 0 x
  let graph ←
    match Runtime.Autograd.Torch.lowerToTypedGraph
        (α := Float) (Γ := [vectorShape]) (τ := vectorShape) build with
    | .ok c => pure c
    | .error e => throw <| IO.userError s!"typed graph log-softmax JVP: lowering failed: {e}"
  let logits : Tensor Float vectorShape := tensorOfArray! [2] #[0.0, Float.log 2.0]
  let tangent : Tensor Float vectorShape := tensorOfArray! [2] #[1.0, 0.0]
  let inputs : TorchLean.TensorPack Float [vectorShape] := .cons logits .nil
  let tangents : TorchLean.TensorPack Float [vectorShape] := .cons tangent .nil
  let got := Runtime.Autograd.Torch.TypedGraph.jvp graph inputs tangents
  let got0 := Tensor.getScalar got ⟨0, by decide⟩
  let got1 := Tensor.getScalar got ⟨1, by decide⟩
  unless Float.abs (got0 - 2.0 / 3.0) ≤ 1e-5 &&
      Float.abs (got1 - (-1.0 / 3.0)) ≤ 1e-5 do
    throw <| IO.userError s!"typed graph log-softmax JVP: got {pretty got}, expected [2/3, -1/3]"
  IO.println "typed_graph_log_softmax_jvp_test (Float): OK"

end TypedGraphLogSoftmaxJvp
end Floats
end Tests

/-! ## Typed graph output references -/

namespace Tests
namespace Floats
namespace TypedGraphOutputReference

open _root_.Spec
open _root_.Spec.Tensor

/--
Typed graph lowering accepts an input as the output, even when no node is recorded or later nodes
are not selected as the result. Forward, JVP, and VJP must all follow that same output reference.
-/
def run : IO Unit := do
  let identityBuild :
      Runtime.Autograd.TypedGraph.GraphM.M Float [Shape.scalar]
        (Runtime.Autograd.TypedGraph.GraphM.Var Shape.scalar) := do
    Runtime.Autograd.TypedGraph.GraphM.arg
      (α := Float) (Γ := [Shape.scalar]) 0 Shape.scalar
  let identity ←
    match Runtime.Autograd.Torch.lowerToTypedGraph
        (α := Float) (Γ := [Shape.scalar]) (τ := Shape.scalar) identityBuild with
    | .ok graph => pure graph
    | .error e => throw <| IO.userError s!"typed graph identity lowering failed: {e}"
  unless identity.nodeShapes.isEmpty do
    throw <| IO.userError "typed graph identity lowering unexpectedly recorded a node"

  let earlierOutputBuild :
      Runtime.Autograd.TypedGraph.GraphM.M Float [Shape.scalar]
        (Runtime.Autograd.TypedGraph.GraphM.Var Shape.scalar) := do
    let x ← Runtime.Autograd.TypedGraph.GraphM.arg
      (α := Float) (Γ := [Shape.scalar]) 0 Shape.scalar
    let _unused ← Runtime.Autograd.TypedGraph.GraphM.add x x
    pure x
  let earlierOutput ←
    match Runtime.Autograd.Torch.lowerToTypedGraph
        (α := Float) (Γ := [Shape.scalar]) (τ := Shape.scalar) earlierOutputBuild with
    | .ok graph => pure graph
    | .error e => throw <| IO.userError s!"typed graph earlier-output lowering failed: {e}"
  unless earlierOutput.nodeShapes.length == 1 do
    throw <| IO.userError "typed graph earlier-output lowering lost the unused recorded node"

  let inputs : TorchLean.TensorPack Float [Shape.scalar] :=
    .cons (Tensor.scalar 3.0) .nil
  let tangents : TorchLean.TensorPack Float [Shape.scalar] :=
    .cons (Tensor.scalar 2.0) .nil
  let checkGraph (label : String)
      (graph : Runtime.Autograd.Torch.TypedGraph Float [Shape.scalar] Shape.scalar) : IO Unit := do
    let output := Tensor.item (Runtime.Autograd.Torch.TypedGraph.forward graph inputs)
    let tangent := Tensor.item (Runtime.Autograd.Torch.TypedGraph.jvp graph inputs tangents)
    let gradients := Runtime.Autograd.Torch.TypedGraph.vjpWithSeed
      graph inputs (Tensor.scalar 5.0)
    let gradient := match gradients with
      | .cons grad .nil => Tensor.item grad
    unless output == 3.0 && tangent == 2.0 && gradient == 5.0 do
      throw <| IO.userError
        s!"{label}: got forward={output}, jvp={tangent}, vjp={gradient}; expected 3, 2, 5"
  checkGraph "typed graph identity output" identity
  checkGraph "typed graph earlier output" earlierOutput

  let publicModel : TorchLean.nn.TypedGraphModel [] Shape.scalar Shape.scalar Float := identity
  let noParams : TorchLean.TensorPack Float [] := .nil
  let publicOutput := Tensor.item <|
    TorchLean.nn.TypedGraphModel.forward publicModel noParams (Tensor.scalar 3.0)
  let publicTangent := Tensor.item <|
    TorchLean.nn.TypedGraphModel.jvp publicModel noParams noParams
      (Tensor.scalar 3.0) (Tensor.scalar 2.0)
  let (publicParamGrads, publicInputGrad) :=
    TorchLean.nn.TypedGraphModel.vjpWithSeed publicModel noParams
      (Tensor.scalar 3.0) (Tensor.scalar 5.0)
  let publicInputGradient := Tensor.item publicInputGrad
  let noPublicParamGrads := match publicParamGrads with
    | .nil => true
  unless noPublicParamGrads && publicOutput == 3.0 && publicTangent == 2.0 &&
      publicInputGradient == 5.0 do
    throw <| IO.userError <|
      s!"typed graph public API: got forward={publicOutput}, jvp={publicTangent}, " ++
      s!"vjp={publicInputGradient}; expected 3, 2, 5"
  IO.println "typed_graph_output_reference_test (Float): OK"

end TypedGraphOutputReference
end Floats
end Tests

/-! ## Typed graph smooth-max parameter checks -/

namespace Tests
namespace Floats
namespace TypedGraphSmoothMaxDomain

open _root_.Spec

/-- Typed `GraphM` rejects an undefined zero inverse temperature while building the graph. -/
def run : IO Unit := do
  let inputShape : Shape := [1, 1, 2]
  let spatial : Spec.Tensor Nat [2] := tensor! [1, 2]
  let kernel : Spec.Tensor Nat [2] := tensor! [1, 2]
  let stride : Spec.Tensor Nat [2] := tensor! [1, 1]
  let padding : Spec.Tensor Nat [2] := tensor! [0, 0]
  let outputShape : Shape :=
    Shape.ofList (1 :: (Spec.poolOutSpatialPad spatial kernel stride padding).toList)
  let build :
      Runtime.Autograd.TypedGraph.GraphM.M Float [inputShape]
        (Runtime.Autograd.TypedGraph.GraphM.Var outputShape) := do
    let x ← Runtime.Autograd.TypedGraph.GraphM.arg
      (α := Float) (Γ := [inputShape]) 0 inputShape
    Runtime.Autograd.TypedGraph.GraphM.smoothMaxPool
      (d := 2) (C := 1) (inSpatial := spatial) (kernel := kernel)
      (stride := stride) (padding := padding)
      (hKernel := by intro i; fin_cases i <;> simp [kernel]) x 0.0
  match Runtime.Autograd.Torch.lowerToTypedGraph
      (α := Float) (Γ := [inputShape]) (τ := outputShape) build with
  | .error _ => IO.println "typed_graph_smooth_max_domain_test (Float): OK"
  | .ok _ => throw <| IO.userError "typed graph smooth-max accepted zero beta"

end TypedGraphSmoothMaxDomain
end Floats
end Tests

/-! ## Dense gradients for disconnected nodes -/

namespace Tests
namespace Floats
namespace DisconnectedDenseGradient

open _root_.Spec
open _root_.Spec.Tensor
open Runtime.Autograd

/-- A disconnected reciprocal at zero must not turn an unrelated leaf gradient into `NaN`. -/
def run : IO Unit := do
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) (Tensor.scalar 0.0) (name := some "x")
  let (t2, outId) := Tape.leaf (t := t1) (Tensor.scalar 3.0) (name := some "output")
  let (t3, invId) ← okOrThrow <|
    Tape.inv (α := Float) (t := t2) (s := Shape.scalar) xId
  let grads ← okOrThrow <|
    Tape.backwardDenseAll (t := t3) outId (Spec.SomeTensor.ofTensor (Tensor.scalar 1.0))
  unless grads.size = t3.nodes.size do
    throw <| IO.userError "disconnected dense gradient: result length mismatch"
  let checkFiniteZero (label : String) (id : Nat) : IO Unit := do
    let grad ← match grads[id]? with
      | some grad => pure grad
      | none => throw <| IO.userError s!"{label}: gradient id out of bounds"
    if h : grad.shape = Shape.scalar then
      let value := Tensor.item (grad.cast h)
      unless value.isFinite && value == 0.0 do
        throw <| IO.userError s!"{label}: expected finite zero, got {value}"
    else
      throw <| IO.userError s!"{label}: expected a scalar gradient"
  checkFiniteZero "disconnected reciprocal input gradient" xId
  checkFiniteZero "disconnected reciprocal output gradient" invId
  IO.println "disconnected_dense_gradient_test (Float): OK"

end DisconnectedDenseGradient
end Floats
end Tests

/-! ## Optimizer and scheduler edge-case regressions -/

namespace Tests
namespace Floats
namespace OptimizerNumerics

open Runtime.Autograd

/-- Finite approximate equality used by the optimizer numerical regressions. -/
def close (x y : Float) (tol : Float := 1e-5) : Bool :=
  x.isFinite && y.isFinite && (x - y).abs ≤ tol

/-- Construct one scalar parameter entry for a compact optimizer test. -/
def scalarParam (id : Nat) (x : Float) : Train.ParamEntry Float :=
  Train.ParamEntry.ofTensor id (Tensor.scalar x)

/-- Construct a one-entry scalar gradient map. -/
def scalarGrad (id : Nat) (x : Float) : Std.HashMap Nat (Spec.SomeTensor Float) :=
  ({} : Std.HashMap Nat (Spec.SomeTensor Float)).insert id
    (Spec.SomeTensor.ofTensor (Tensor.scalar x))

/-- Read a scalar parameter while preserving the runtime's error reporting. -/
def getScalar (tag : String) (params : Train.ParamTable Float) (id : Nat) :
    Runtime.Autograd.Result Float := do
  let value ← Train.ParamTable.getTensor (tag := tag) (s := .scalar) params id
  match value with
  | .scalar x => pure x

/-- Adam bias correction advances only when that particular parameter receives a gradient. -/
def checkSparseAdamSteps : Runtime.Autograd.Result Bool := do
  let opt0 : Train.OptimizerState Float :=
    { kind := .adam
      groups := #[{ params := #[0, 1], lr := 0.1, beta1 := 0.9, beta2 := 0.999,
                    epsilon := 1e-8 }] }
  let params0 : Train.ParamTable Float := #[scalarParam 0 1.0, scalarParam 1 1.0]
  let (opt1, params1) ← Train.Optim.step opt0 params0 (scalarGrad 0 1.0)
  let (opt2, params2) ← Train.Optim.step opt1 params1 (scalarGrad 1 1.0)
  let p0 ← getScalar "sparse Adam" params2 0
  let p1 ← getScalar "sparse Adam" params2 1
  let restored := Train.OptimizerState.ofStateDict opt2.toStateDict
  pure <|
    close p0 0.9 && close p1 0.9 && opt2.step == 2 &&
      opt2.parameterSteps.get? 0 == some 1 && opt2.parameterSteps.get? 1 == some 1 &&
      restored.parameterSteps.get? 0 == some 1 && restored.parameterSteps.get? 1 == some 1

/-- Momentum dampening does not scale the first buffer, matching the standard SGD convention. -/
def checkMomentumInitialization : Runtime.Autograd.Result Bool := do
  let opt0 : Train.OptimizerState Float :=
    { kind := .momentum
      groups := #[{ params := #[0], lr := 0.1, momentum := 0.9, dampening := 0.5 }] }
  let params0 : Train.ParamTable Float := #[scalarParam 0 1.0]
  let (opt1, params1) ← Train.Optim.step opt0 params0 (scalarGrad 0 2.0)
  let first ← getScalar "momentum initialization" params1 0
  let (_, params2) ← Train.Optim.step opt1 params1 (scalarGrad 0 2.0)
  let second ← getScalar "momentum initialization" params2 0
  pure (close first 0.8 && close second 0.52)

/-- Adadelta's update accumulator stores the unscaled update, independently of the learning rate. -/
def checkAdadeltaAccumulator : Bool :=
  let params : Tensor Float .scalar := Tensor.scalar 10.0
  let grads : Tensor Float .scalar := Tensor.scalar 2.0
  let state := _root_.Optim.Adadelta.init 0.5 0.0 1.0 params
  let (nextState, nextParams) := _root_.Optim.Adadelta.update state params grads
  match nextState.u, nextParams with
  | .scalar updateSquare, .scalar parameter =>
      close updateSquare 0.8 && close parameter (10.0 - 1.0 / Float.sqrt 5.0)

/-- Warmup-cosine decay remains at zero after its finite schedule has ended. -/
def checkWarmupCosineStops : Bool :=
  let base : _root_.Optim.WarmupCosineScheduler Float :=
    _root_.Optim.warmupCosineScheduler 1.0 2 10
  let atEnd := { base with currentStep := 10 }
  let afterEnd := { base with currentStep := 20 }
  _root_.Optim.WarmupCosineScheduler.getLr atEnd == 0.0 &&
    _root_.Optim.WarmupCosineScheduler.getLr afterEnd == 0.0

/-- Public optimizer configurations reject domains that make their updates undefined. -/
def checkPublicOptimizerValidation : Bool :=
  let rejected : Except String Unit -> Bool
    | .error _ => true
    | .ok () => false
  (TorchLean.optim.adam { lr := 1e-3 }).validate.isOk &&
    rejected (TorchLean.optim.adam { lr := 1e-3, beta1 := 1.0 }).validate &&
    rejected (TorchLean.optim.adamW { lr := 1e-3, weightDecay := -0.1 }).validate &&
    rejected (TorchLean.optim.rmsprop { lr := 1e-3, epsilon := 0.0 }).validate &&
    rejected (TorchLean.optim.sgd
      { lr := 0.1, momentum := Float.ofBits 0x7ff8000000000000 }).validate

/-- Run the optimizer and scheduler edge-case regressions. -/
def run : IO Unit := do
  match checkSparseAdamSteps with
  | .error msg => throw <| IO.userError s!"optimizer numerics (sparse Adam): {msg}"
  | .ok false => throw <| IO.userError "optimizer numerics (sparse Adam): FAILED"
  | .ok true => pure ()
  match checkMomentumInitialization with
  | .error msg => throw <| IO.userError s!"optimizer numerics (momentum): {msg}"
  | .ok false => throw <| IO.userError "optimizer numerics (momentum): FAILED"
  | .ok true => pure ()
  unless checkAdadeltaAccumulator do
    throw <| IO.userError "optimizer numerics (Adadelta accumulator): FAILED"
  unless checkWarmupCosineStops do
    throw <| IO.userError "optimizer numerics (warmup cosine): FAILED"
  unless checkPublicOptimizerValidation do
    throw <| IO.userError "optimizer configuration validation: FAILED"
  IO.println "optimizer and scheduler edge cases (Float): OK"

end OptimizerNumerics
end Floats
end Tests

namespace Tests
namespace Floats

def runAllAutogradTests : IO Unit := do
  IO.println "=== Runtime autograd test suite (Float) ==="
  AutogradEngine.run
  AutogradLinearRegression.run
  AutogradTrain.run
  AutogradLayerNorm.run
  AutogradConv.run
  TypedGraphLogSoftmaxJvp.run
  TypedGraphOutputReference.run
  TypedGraphSmoothMaxDomain.run
  DisconnectedDenseGradient.run
  OptimizerNumerics.run
  IO.println "=== Autograd test suite completed ==="

end Floats
end Tests
