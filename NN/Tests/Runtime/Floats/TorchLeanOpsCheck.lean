/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Lean.Data.Json
public import NN.API
public import NN.Core.ExternalProcess
public import NN.Runtime.Autograd.TorchLean.Norm
public import NN.Runtime.RL.Core
public import NN.Spec.Generative.Diffusion.PFODE
public import NN.Spec.Layers.Loss
public import NN.Spec.Models.Gmm
public import NN.Spec.Models.Hmm
public import NN.Tests.Runtime.Floats.Utils
public import Std

public import NN.IR.OpContracts

/-!
# TorchLeanOpsCheck

Runtime checks for TorchLean operator wrappers over the float runtime.

The file is intentionally fixture-driven: each helper builds one small tensor example, runs it
through the relevant execution path or external backend boundary, and compares the result against
another TorchLean path, a closed form, or PyTorch when it is available.
-/

@[expose] public section

open Lean
open Spec
open Tensor
open Tests.Floats.Utils

namespace Tests
namespace Floats
namespace TorchLeanOpsCheck

/-! ## Shared fixtures -/

/-- BatchNorm parity fixture: batch size. -/
abbrev bnN : Nat := 2
/-- BatchNorm parity fixture: channel count. -/
abbrev bnC : Nat := 2
/-- BatchNorm parity fixture: image height. -/
abbrev bnH : Nat := 2
/-- BatchNorm parity fixture: image width. -/
abbrev bnW : Nat := 2
/-- NCHW shape used by the BatchNorm runtime and PyTorch parity checks. -/
abbrev bnShape : Shape := [bnN, bnC, bnH, bnW]

/-- Scratch directory for small Python parity scripts emitted by this test module. -/
def workDir : System.FilePath :=
  TorchLean.External.Process.artifactWorkDir "ops_check"

/-- Path for the generated BatchNorm parity script. -/
def batchNormParityScriptPath : System.FilePath :=
  workDir / "batchnorm_parity.py"

/-- Evaluate softmax along a statically checked tensor dimension in the eager runtime. -/
def evalSoftmaxAxis {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (x : Tensor Float s) : IO (Tensor Float s) := do
  let sess ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float)
  let action : Runtime.Autograd.Torch.Internal.EagerM Float (Tensor Float s) := do
    let xRef ← Runtime.Autograd.Torch.Ops.const
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float) (s := s) x
    let yRef ← Runtime.Autograd.TorchLean.F.softmax
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float) (s := s) axis xRef
    let sess ← read
    liftM <| Runtime.Autograd.Torch.Internal.EagerSession.getValue
      (α := Float) (sh := s) sess yRef
  action sess

/-- Evaluate softmax and its VJP along a statically checked tensor dimension. -/
def evalSoftmaxAxisWithGradient {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (x upstream : Tensor Float s) : IO (Tensor Float s × Tensor Float s) := do
  let sess ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float)
  let xRef ← Runtime.Autograd.Torch.Internal.EagerSession.input
    (α := Float) (sh := s) sess x (name := some "softmax_input") (requiresGrad := true)
  let action : Runtime.Autograd.Torch.Internal.EagerM Float
      (Runtime.Autograd.Torch.TensorRef Float s) :=
    Runtime.Autograd.TorchLean.F.softmax
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float) (s := s) axis xRef
  let yRef ← action sess
  let y ← Runtime.Autograd.Torch.Internal.EagerSession.getValue
    (α := Float) (sh := s) sess yRef
  let grads ← Runtime.Autograd.Torch.Internal.EagerSession.backwardDenseAll
    (α := Float) (sh := s) sess yRef upstream
  let dx ← Runtime.Autograd.Torch.Internal.EagerSession.grad grads xRef
  pure (y, dx)

/-- Evaluate log-softmax and its VJP along a statically checked tensor dimension. -/
def evalLogSoftmaxAxisWithGradient {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (x upstream : Tensor Float s) : IO (Tensor Float s × Tensor Float s) := do
  let sess ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float)
  let xRef ← Runtime.Autograd.Torch.Internal.EagerSession.input
    (α := Float) (sh := s) sess x (name := some "log_softmax_input") (requiresGrad := true)
  let action : Runtime.Autograd.Torch.Internal.EagerM Float
      (Runtime.Autograd.Torch.TensorRef Float s) :=
    Runtime.Autograd.TorchLean.F.logSoftmax
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float) (s := s) axis xRef
  let yRef ← action sess
  let y ← Runtime.Autograd.Torch.Internal.EagerSession.getValue
    (α := Float) (sh := s) sess yRef
  let grads ← Runtime.Autograd.Torch.Internal.EagerSession.backwardDenseAll
    (α := Float) (sh := s) sess yRef upstream
  let dx ← Runtime.Autograd.Torch.Internal.EagerSession.grad grads xRef
  pure (y, dx)

/-- Check outer, final, and interior softmax dimensions against explicit numerical fixtures. -/
def checkSoftmaxDimensions : IO Unit := do
  let matrix : Tensor Float [2, 3] :=
    tensorOfArray! [2, 3] #[1, 2, 3, 4, 5, 6]
  let alongRows ← evalSoftmaxAxis 0 matrix
  let alongColumns ← evalSoftmaxAxis 1 matrix
  let rowExpected := #[0.047425873, 0.047425873, 0.047425873,
    0.952574127, 0.952574127, 0.952574127]
  let columnExpected := #[0.090030573, 0.244728471, 0.665240956,
    0.090030573, 0.244728471, 0.665240956]
  for (actual, expected) in alongRows.toArray.zip rowExpected do
    assertApprox "softmax axis 0" actual expected 1e-5
  for (actual, expected) in alongColumns.toArray.zip columnExpected do
    assertApprox "softmax axis 1" actual expected 1e-5

  let cube : Tensor Float [2, 2, 2] :=
    tensorOfArray! [2, 2, 2] #[0, 2, 1, 4, 3, 8, 7, 9]
  let alongMiddle ← evalSoftmaxAxis 1 cube
  let middleExpected := #[0.268941421, 0.119202922, 0.731058579, 0.880797078,
    0.017986210, 0.268941421, 0.982013790, 0.731058579]
  for (actual, expected) in alongMiddle.toArray.zip middleExpected do
    assertApprox "softmax interior axis" actual expected 1e-5

  let upstream : Tensor Float [2, 2, 2] :=
    tensorOfArray! [2, 2, 2] #[1, -2, 3, 4, -1, 2, 5, -3]
  let (_, gradient) ← evalSoftmaxAxisWithGradient 1 cube upstream
  let expectedGradient :=
    Activation.softmaxBackwardSpec (α := Float) (s := [2, 2, 2])
      1 cube upstream
  for (actual, expected) in gradient.toArray.zip expectedGradient.toArray do
    assertApprox "softmax interior-axis gradient" actual expected 1e-5

  let (logProbabilities, logGradient) ←
    evalLogSoftmaxAxisWithGradient 1 cube upstream
  let expectedLogProbabilities :=
    Activation.logSoftmaxSpec (α := Float) (s := [2, 2, 2])
      1 cube
  let expectedLogGradient :=
    Activation.logSoftmaxBackwardSpec
      (α := Float) (s := [2, 2, 2])
      1 expectedLogProbabilities upstream
  for (actual, expected) in logProbabilities.toArray.zip expectedLogProbabilities.toArray do
    assertApprox "log-softmax interior axis" actual expected 1e-5
  for (actual, expected) in logGradient.toArray.zip expectedLogGradient.toArray do
    assertApprox "log-softmax interior-axis gradient" actual expected 1e-5

/-- Check that classification metrics support outer and inner class axes. -/
def checkClassificationAxes : IO Unit := do
  let logits : Tensor Float [2, 3] :=
    tensorOfArray! [2, 3] #[1, 5, 2, 9, 4, 3]
  unless TorchLean.Metrics.argmaxAxis? 1 logits = #[some 1, some 0] do
    throw <| IO.userError "argmax along the inner class axis returned incorrect indices"
  unless TorchLean.Metrics.argmaxAxis? 0 logits = #[some 1, some 0, some 1] do
    throw <| IO.userError "argmax along the outer class axis returned incorrect indices"

  let rowTargets : Tensor Float [2, 3] :=
    tensorOfArray! [2, 3] #[0, 1, 0, 1, 0, 0]
  unless TorchLean.Metrics.accuracyOneHotAxis 1 logits rowTargets = (2, 2) do
    throw <| IO.userError "one-hot accuracy along the inner class axis was incorrect"

  let columnTargets : Tensor Float [2, 3] :=
    tensorOfArray! [2, 3] #[0, 1, 0, 1, 0, 1]
  unless TorchLean.Metrics.accuracyOneHotAxis 0 logits columnTargets = (3, 3) do
    throw <| IO.userError "one-hot accuracy along the outer class axis was incorrect"

/-- Evaluate a two-row weighted integer-label cross entropy through the eager runtime. -/
def evalWeightedRowCrossEntropy
    (logits : Tensor Float [2, 2])
    (weights : Tensor Float [2]) : IO Float := do
  let targets : Tensor (Fin 2) [2] :=
    Tensor.ofFn fun i => if i = 0 then ⟨0, by decide⟩ else ⟨1, by decide⟩
  let sess ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float)
  let action : Runtime.Autograd.Torch.Internal.EagerM Float (Tensor Float .scalar) := do
    let logitsRef ← Runtime.Autograd.Torch.Ops.const
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float)
      (s := [2, 2]) logits
    let weightsRef ← Runtime.Autograd.Torch.Ops.const
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float)
      (s := [2]) weights
    let logitsIndexed : Runtime.Autograd.TorchLean.RefTy
        (Runtime.Autograd.Torch.Internal.EagerM Float) Float
        (([2] : Shape).concat [2]) := by simpa using logitsRef
    let targetsIndexed : Tensor (Fin 2) (([2] : Shape).concat Shape.scalar) := by
      simpa using targets
    let weightsIndexed : Runtime.Autograd.TorchLean.RefTy
        (Runtime.Autograd.Torch.Internal.EagerM Float) Float
        (([2] : Shape).concat Shape.scalar) := by simpa using weightsRef
    let lossRef ← _root_.TorchLean.Loss.crossEntropyWeighted
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float)
      1 rfl logitsIndexed targetsIndexed weightsIndexed
    let sess ← read
    liftM <| Runtime.Autograd.Torch.Internal.EagerSession.getValue
      (α := Float) (sh := .scalar) sess lossRef
  pure <| Tensor.item (← action sess)

/--
Check weighted cross entropy as a runtime operation.

The first assertion changes only a zero-weight row and therefore must leave the loss unchanged. The
second checks linearity in the explicit row weights. Together they catch accidental mean reduction,
mask misalignment, and a weights tensor that is ignored by the runtime.
-/
def checkWeightedRowCrossEntropy : IO Unit := do
  let logitsA : Tensor Float [2, 2] :=
    tensorOfArray! [2, 2] #[0, 0, 20, -20]
  let logitsB : Tensor Float [2, 2] :=
    tensorOfArray! [2, 2] #[0, 0, -20, 20]
  let firstOnly : Tensor Float [2] := tensorOfArray! [2] #[1, 0]
  let secondOnly : Tensor Float [2] := tensorOfArray! [2] #[0, 1]
  let mixture : Tensor Float [2] := tensorOfArray! [2] #[0.25, 0.75]
  let firstA ← evalWeightedRowCrossEntropy logitsA firstOnly
  let firstB ← evalWeightedRowCrossEntropy logitsB firstOnly
  assertApprox "zero-weight row is excluded" firstA firstB 1e-5
  let second ← evalWeightedRowCrossEntropy logitsA secondOnly
  let mixed ← evalWeightedRowCrossEntropy logitsA mixture
  assertApprox "weighted row loss is linear in row weights"
    mixed (0.25 * firstA + 0.75 * second) 1e-4

/-- Check indexed cross entropy with both leading and trailing coordinates around the class axis. -/
def checkMiddleAxisCrossEntropy : IO Unit := do
  let logits : Tensor Float [2, 2, 2] :=
    tensorOfArray! [2, 2, 2] #[10, -10, -10, 10, -10, 10, 10, -10]
  let targets : Tensor (Fin 2) [2, 2] :=
    TorchLean.Tensor.stack 0 fun outer => Tensor.ofFn fun inner =>
      if outer.val = inner.val then ⟨0, by decide⟩ else ⟨1, by decide⟩
  let sess ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float)
  let action : Runtime.Autograd.Torch.Internal.EagerM Float (Tensor Float .scalar) := do
    let logitsRef ← Runtime.Autograd.Torch.Ops.const
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float)
      (s := [2, 2, 2]) logits
    let logitsIndexed : Runtime.Autograd.TorchLean.RefTy
        (Runtime.Autograd.Torch.Internal.EagerM Float) Float
        (([2] : Shape).concat ([2, 2] : Shape)) := by simpa using logitsRef
    let targetsIndexed : Tensor (Fin 2) (([2] : Shape).concat [2]) := by
      simpa using targets
    let lossRef ← _root_.TorchLean.Loss.crossEntropy
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float)
      1 rfl logitsIndexed targetsIndexed
    let sess ← read
    liftM <| Runtime.Autograd.Torch.Internal.EagerSession.getValue
      (α := Float) (sh := .scalar) sess lossRef
  assertApprox "cross entropy preserves a suffix after the class axis"
    (Tensor.item (← action sess)) 0 1e-5

/-- Check that tied token lookup removes exactly one independent affine vocabulary head. -/
def checkTiedTokenEmbeddingParameterCount : IO Unit := do
  let countElements (shapes : List Shape) : Nat :=
    shapes.foldl (fun count shape => count + Spec.Shape.size shape) 0
  let cfg : TorchLean.nn.models.CausalTransformer.Config :=
    { seqLen := 2
      vocab := 7
      numHeads := 1
      headDim := 4
      ffnHidden := 8
      layers := 1 }
  let untiedBody := TorchLean.nn.build 11 <|
    TorchLean.nn.models.CausalTransformer.fromEmbeddings cfg [1]
  let tiedBody := TorchLean.nn.build 11 <|
    TorchLean.nn.models.CausalTransformer.hidden cfg [1]
  let untiedCount := countElements <|
    (TorchLean.nn.models.CausalTransformer.Indexed.model cfg untiedBody).stateShapes
  let tiedCount := countElements <|
    TorchLean.nn.models.CausalTransformer.Tied.stateShapes cfg tiedBody
  let independentHeadCount := cfg.vocab * cfg.dModel + cfg.vocab
  unless untiedCount = tiedCount + independentHeadCount do
    throw <| IO.userError <|
      s!"tied token embedding: untied={untiedCount}, tied={tiedCount}, " ++
        s!"expected difference={independentHeadCount}"

/-- Run a tied-token model through one loss and backward pass. -/
def checkTiedTokenEmbeddingBackward : IO Unit := do
  let cfg : TorchLean.nn.models.CausalTransformer.Config :=
    { seqLen := 2
      vocab := 7
      numHeads := 1
      headDim := 4
      ffnHidden := 8
      layers := 1 }
  let body := TorchLean.nn.build 19 <|
    TorchLean.nn.models.CausalTransformer.hidden cfg [1]
  let definition := TorchLean.nn.models.CausalTransformer.Tied.objective cfg body
  let module ← _root_.Runtime.Autograd.TorchLean.Module.ObjectiveDef.instantiateFloat64
    definition { execution := .eager }
  let inputs : Tensor (Fin cfg.vocab) [1, 2] :=
    TorchLean.Tensor.stack 0 fun _ => Tensor.ofFn fun i =>
      if i.val = 0 then ⟨0, by decide⟩ else ⟨1, by decide⟩
  let targets : Tensor (Fin cfg.vocab) [1, 2] :=
    TorchLean.Tensor.stack 0 fun _ => Tensor.ofFn fun i =>
      if i.val = 0 then ⟨1, by decide⟩ else ⟨2, by decide⟩
  let (loss, gradients) ←
    _root_.Runtime.Autograd.TorchLean.Module.Objective.lossAndGradState module
      .nil (.cons inputs (.cons targets .nil))
  assertFinite "tied token embedding loss" (Tensor.item loss)
  let sharedGradient := TorchLean.TensorPack.get gradients ⟨0, by simp⟩
  let values := sharedGradient.toArray
  for value in values do
    assertFinite "tied token embedding gradient" value
  unless values.any (fun value => Float.abs value > 1.0e-8) do
    throw <| IO.userError
      "tied token embedding produced a zero gradient for its shared lookup/projection matrix"

/--
Check that self-attention can initialize its residual output projection independently of Q/K/V.

Deep Transformers commonly use a smaller initializer for the projection written back to the
residual stream. Distinct constant schemes make this a direct wiring check: Q/K/V must contain
ones, while the output projection must contain zeros.
-/
def checkAttentionOutputProjectionInitializer : IO Unit := do
  let layer :=
    _root_.Runtime.Autograd.TorchLean.NN.multiHeadAttention
      1 1 2 1 2 (h1 := by decide)
      (weightInit? := some .ones)
      (outputWeightInit? := some .zeros)
  let (wq, wk, wv, wo) :=
    match layer.initState with
    | .cons wq (.cons wk (.cons wv (.cons wo .nil))) => (wq, wk, wv, wo)
  for i in List.finRange 2 do
    for j in List.finRange 2 do
      assertApprox s!"attention Q initializer[{i.val},{j.val}]" (matVal wq i j) 1 1e-7
      assertApprox s!"attention K initializer[{i.val},{j.val}]" (matVal wk i j) 1 1e-7
      assertApprox s!"attention V initializer[{i.val},{j.val}]" (matVal wv i j) 1 1e-7
      assertApprox s!"attention output initializer[{i.val},{j.val}]" (matVal wo i j) 0 1e-7

/-- Typed graph execution preserves leaf gradient flags and leaves frozen parameters unchanged. -/
def checkTypedGraphLeafMetadata : IO Unit := do
  let sess ← _root_.Runtime.Autograd.TorchLean.Session.new (α := Float)
    (opts := { execution := .typedGraph })
  let frozen ← _root_.Runtime.Autograd.TorchLean.Session.param sess
    (Tensor.scalar 2.0) (name := some "frozen") (requiresGrad := some false)
  let trainable ← _root_.Runtime.Autograd.TorchLean.Session.param sess
    (Tensor.scalar 3.0) (name := some "trainable") (requiresGrad := some true)
  let frozenRef ← _root_.Runtime.Autograd.TorchLean.Session.use sess frozen
  let trainableRef ← _root_.Runtime.Autograd.TorchLean.Session.use sess trainable
  let sum ← _root_.Runtime.Autograd.TorchLean.Session.add sess frozenRef trainableRef
  let loss ← _root_.Runtime.Autograd.TorchLean.Session.mul sess sum sum
  let grads ← _root_.Runtime.Autograd.TorchLean.Session.backwardScalarDenseAll sess loss
  let frozenGrad ← _root_.Runtime.Autograd.TorchLean.Session.grad sess grads frozenRef
  let trainableGrad ← _root_.Runtime.Autograd.TorchLean.Session.grad sess grads trainableRef
  assertApprox "typed graph frozen gradient" (Tensor.item frozenGrad) 0 1e-7
  assertApprox "typed graph trainable gradient" (Tensor.item trainableGrad) 10 1e-7
  _root_.Runtime.Autograd.TorchLean.Session.sgdStepAll sess 0.1 grads
  assertApprox "typed graph frozen parameter" (Tensor.item (← frozen.value.get)) 2 1e-7
  assertApprox "typed graph updated parameter" (Tensor.item (← trainable.value.get)) 2 1e-7

/-! ## Eager/typed graph operator parity -/

/-- Evaluate public session softmax and log-softmax along a three-axis tensor's middle axis. -/
def evalSessionSoftmaxFixture (execution : _root_.Runtime.Autograd.Torch.ExecutionMode) :
    IO (Tensor Float [2, 2, 2] ×
      Tensor Float [2, 2, 2]) := do
  let shape : Shape := [2, 2, 2]
  let input : Tensor Float shape := tensorOfArray! [2, 2, 2] #[0, 2, 1, 4, 3, 8, 7, 9]
  let sess ← _root_.Runtime.Autograd.TorchLean.Session.new (α := Float)
    (opts := { execution := execution })
  let inputRef ← _root_.Runtime.Autograd.TorchLean.Session.const sess (sh := shape) input
  let probabilitiesRef ←
    _root_.Runtime.Autograd.TorchLean.Session.softmax sess (sh := shape) 1 inputRef
  let logProbabilitiesRef ←
    _root_.Runtime.Autograd.TorchLean.Session.logSoftmax sess (sh := shape) 1 inputRef
  let probabilities ←
    _root_.Runtime.Autograd.TorchLean.Session.getValue sess (sh := shape) probabilitiesRef
  let logProbabilities ←
    _root_.Runtime.Autograd.TorchLean.Session.getValue sess (sh := shape) logProbabilitiesRef
  pure (probabilities, logProbabilities)

/-- Evaluate a fixed 2x3 by 3x2 matrix product in the selected execution mode. -/
def evalMatmulFixture (execution : _root_.Runtime.Autograd.Torch.ExecutionMode) :
    IO (Tensor Float [2, 2]) := do
  let sess ← _root_.Runtime.Autograd.TorchLean.Session.new (α := Float)
    (opts := { execution := execution })
  let a : Tensor Float [2, 3] :=
    Tensor.generate [2, 3] fun coordinate =>
      Float.ofNat (coordinate.getD 0 0 + 2 * coordinate.getD 1 0 + 1)
  let b : Tensor Float [3, 2] :=
    Tensor.generate [3, 2] fun coordinate =>
      Float.ofNat (3 * coordinate.getD 0 0 + coordinate.getD 1 0 + 1)
  let aR ← _root_.Runtime.Autograd.TorchLean.Session.const sess (sh := [2, 3]) a
  let bR ← _root_.Runtime.Autograd.TorchLean.Session.const sess (sh := [3, 2]) b
  let cR ← _root_.Runtime.Autograd.TorchLean.Session.matmul sess
    (batchA := Shape.scalar) (batchB := Shape.scalar) (batch := Shape.scalar)
    (m := 2) (n := 3) (p := 2) aR bR
  _root_.Runtime.Autograd.TorchLean.Session.getValue sess (sh := [2, 2]) cR

/-- Evaluate a fixed length-2 plus length-3 vector concatenation in the selected execution mode. -/
def evalConcatFixture (execution : _root_.Runtime.Autograd.Torch.ExecutionMode) :
    IO (Tensor Float [5]) := do
  let sess ← _root_.Runtime.Autograd.TorchLean.Session.new (α := Float)
    (opts := { execution := execution })
  let a : Tensor Float [2] := Tensor.ofFn fun i => Float.ofNat (i.val + 1)
  let b : Tensor Float [3] := Tensor.ofFn fun i => 10.0 + Float.ofNat i.val
  let aR ← _root_.Runtime.Autograd.TorchLean.Session.const sess (sh := [2]) a
  let bR ← _root_.Runtime.Autograd.TorchLean.Session.const sess (sh := [3]) b
  let cR ← _root_.Runtime.Autograd.TorchLean.Session.concatLeadingAxis sess
    (n := 2) (m := 3) (sh := .scalar) aR bR
  _root_.Runtime.Autograd.TorchLean.Session.getValue sess (sh := [5]) cR

/-- Evaluate a two-spatial-axis max-pooling example through the generic pooling API. -/
def evalMaxPoolFixture (execution : _root_.Runtime.Autograd.Torch.ExecutionMode) :
    IO (Tensor Float [1, 2, 2]) := do
  let sess ← _root_.Runtime.Autograd.TorchLean.Session.new (α := Float)
    (opts := { execution := execution })
  let spatial : Spec.Tensor Nat [2] := tensor! [4, 4]
  let kernel : Spec.Tensor Nat [2] := tensor! [2, 2]
  let stride : Spec.Tensor Nat [2] := tensor! [2, 2]
  let padding : Spec.Tensor Nat [2] := tensor! [0, 0]
  have hKernel : ∀ i : Fin 2, kernel.getScalar i ≠ 0 := by
    intro i
    fin_cases i <;> simp [kernel]
  let x : Tensor Float [1, 4, 4] :=
    Tensor.generate [1, 4, 4] fun coordinate =>
      Float.ofNat (coordinate.getD 1 0 * 10 + coordinate.getD 2 0)
  have hInputShape : Shape.ofList (1 :: spatial.toList) = [1, 4, 4] := by
    simp [spatial]
  let x' : Tensor Float (Shape.ofList (1 :: spatial.toList)) := hInputShape.symm ▸ x
  let xR ← _root_.Runtime.Autograd.TorchLean.Session.const sess x'
  let yR ← _root_.Runtime.Autograd.TorchLean.Session.maxPool sess
    (d := 2) (channels := 1) (spatial := spatial)
    (kernel := kernel) (stride := stride) (padding := padding) (hKernel := hKernel) xR
  let y ← _root_.Runtime.Autograd.TorchLean.Session.getValue sess yR
  have hOutputSpatial :
      Spec.poolOutSpatialPad spatial kernel stride padding = tensor! [2, 2] := by
    apply Spec.Tensor.ext_vector
    intro i
    fin_cases i <;>
      norm_num [spatial, kernel, stride, padding, Spec.poolOutSpatialPad, Spec.poolOutDim,
        Shape.slidingWindowOutDim]
  have hShape :
      Shape.ofList
          (1 :: (Spec.poolOutSpatialPad spatial kernel stride padding).toList) =
        [1, 2, 2] := by
    rw [hOutputSpatial]
    rfl
  return cast (congrArg (Tensor Float) hShape) y

/-- Evaluate a two-spatial-axis average-pooling example through the generic pooling API. -/
def evalAvgPoolFixture (execution : _root_.Runtime.Autograd.Torch.ExecutionMode) :
    IO (Tensor Float [1, 2, 2]) := do
  let sess ← _root_.Runtime.Autograd.TorchLean.Session.new (α := Float)
    (opts := { execution := execution })
  let spatial : Spec.Tensor Nat [2] := tensor! [4, 4]
  let kernel : Spec.Tensor Nat [2] := tensor! [2, 2]
  let stride : Spec.Tensor Nat [2] := tensor! [2, 2]
  let padding : Spec.Tensor Nat [2] := tensor! [0, 0]
  have hKernel : ∀ i : Fin 2, kernel.getScalar i ≠ 0 := by
    intro i
    fin_cases i <;> simp [kernel]
  let x : Tensor Float [1, 4, 4] :=
    Tensor.generate [1, 4, 4] fun coordinate =>
      Float.ofNat (coordinate.getD 1 0 * 10 + coordinate.getD 2 0)
  have hInputShape : Shape.ofList (1 :: spatial.toList) = [1, 4, 4] := by
    simp [spatial]
  let x' : Tensor Float (Shape.ofList (1 :: spatial.toList)) := hInputShape.symm ▸ x
  let xR ← _root_.Runtime.Autograd.TorchLean.Session.const sess x'
  let yR ← _root_.Runtime.Autograd.TorchLean.Session.avgPool sess
    (d := 2) (channels := 1) (spatial := spatial)
    (kernel := kernel) (stride := stride) (padding := padding) hKernel xR
  let y ← _root_.Runtime.Autograd.TorchLean.Session.getValue sess yR
  have hOutputSpatial :
      Spec.poolOutSpatialPad spatial kernel stride padding = tensor! [2, 2] := by
    apply Spec.Tensor.ext_vector
    intro i
    fin_cases i <;>
      norm_num [spatial, kernel, stride, padding, Spec.poolOutSpatialPad, Spec.poolOutDim,
        Shape.slidingWindowOutDim]
  have hShape :
      Shape.ofList
          (1 :: (Spec.poolOutSpatialPad spatial kernel stride padding).toList) =
        [1, 2, 2] := by
    rw [hOutputSpatial]
    rfl
  return cast (congrArg (Tensor Float) hShape) y

/-! ## BatchNorm fixture and PyTorch parity -/

/-- NCHW input with different signs per channel, chosen so mean/variance are easy to inspect. -/
def bnInput : Tensor Float bnShape :=
  Tensor.generate [bnN, bnC, bnH, bnW] fun coordinate =>
    let n := coordinate.getD 0 0
    let c := coordinate.getD 1 0
    let h := coordinate.getD 2 0
    let w := coordinate.getD 3 0
    let base := Float.ofNat (n * 8 + c * 4 + h * 2 + w + 1)
    if c = 0 then base else -base

/-- BatchNorm scale parameter for the two channels. -/
def bnGamma : Tensor Float [bnC] :=
  tensor! [1.0, 0.5]

/-- BatchNorm shift parameter for the two channels. -/
def bnBeta : Tensor Float [bnC] :=
  tensor! [0.0, 0.1]

/-- Running mean used by the eval-mode BatchNorm fixture. -/
def bnMean : Tensor Float [bnC] :=
  tensor! [2.0, -3.0]

/-- Running variance used by the eval-mode BatchNorm fixture. -/
def bnVar : Tensor Float [bnC] :=
  tensor! [4.0, 9.0]

/--
Run training-mode BatchNorm and return the output together with the computed channel statistics.

This checks the TorchLean runtime wrapper directly, not only the exported graph path.
-/
def evalBatchNormTrain :
    IO (Tensor Float bnShape × Tensor Float [bnC] × Tensor Float [bnC]) :=
    do
  let sess ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float)
  let action : Runtime.Autograd.Torch.Internal.EagerM Float
      (Tensor Float bnShape × Tensor Float [bnC] × Tensor Float [bnC]) :=
    do
      let xR ← Runtime.Autograd.Torch.Ops.const
        (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float) (s := bnShape) bnInput
      let gR ← Runtime.Autograd.Torch.Ops.const
        (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float) (s := [bnC])
        bnGamma
      let bR ← Runtime.Autograd.Torch.Ops.const
        (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float) (s := [bnC])
        bnBeta
      let (yR, meanR, varR) ← Runtime.Autograd.TorchLean.Norm.batchNormTrainStats
        (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float)
        (batch := bnN) (channels := bnC) (spatial := [bnH, bnW])
        (by decide) xR gR bR
      let sess ← read
      let y ← liftM <| Runtime.Autograd.Torch.Internal.EagerSession.getValue
        (α := Float) (sh := bnShape) sess yR
      let mean ← liftM <| Runtime.Autograd.Torch.Internal.EagerSession.getValue
        (α := Float) (sh := [bnC]) sess meanR
      let var ← liftM <| Runtime.Autograd.Torch.Internal.EagerSession.getValue
        (α := Float) (sh := [bnC]) sess varR
      pure (y, mean, var)
  action sess

/-- Run eval-mode BatchNorm using fixed running statistics. -/
def evalBatchNormEval :
    IO (Tensor Float bnShape) := do
  let sess ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float)
  let action : Runtime.Autograd.Torch.Internal.EagerM Float (Tensor Float bnShape) := do
    let xR ← Runtime.Autograd.Torch.Ops.const
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float) (s := bnShape) bnInput
    let gR ← Runtime.Autograd.Torch.Ops.const
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float) (s := [bnC])
      bnGamma
    let bR ← Runtime.Autograd.Torch.Ops.const
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float) (s := [bnC])
      bnBeta
    let mR ← Runtime.Autograd.Torch.Ops.const
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float) (s := [bnC])
      bnMean
    let vR ← Runtime.Autograd.Torch.Ops.const
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float) (s := [bnC])
      bnVar
    let yR ← Runtime.Autograd.TorchLean.Norm.batchNormEval
      (m := Runtime.Autograd.Torch.Internal.EagerM Float) (α := Float)
      (batch := bnN) (channels := bnC) (spatial := [bnH, bnW])
      (by decide) xR gR bR mR vR
    let sess ← read
    liftM <| Runtime.Autograd.Torch.Internal.EagerSession.getValue
      (α := Float) (sh := bnShape) sess yR
  action sess

/-- Closed-form BatchNorm expression when the mean and variance come from the current batch. -/
def expectedBatchNormFromBatchStats (x gamma beta mean var : Float) : Float :=
  ((x - mean) / Float.sqrt (var + Numbers.normalizationEpsilon)) * gamma + beta

/-- Closed-form BatchNorm expression when the mean and variance are fixed running statistics. -/
def expectedBatchNormFromRunningStats (x gamma beta mean var : Float) : Float :=
  ((x - mean) / Float.sqrt (var + Numbers.normalizationEpsilon)) * gamma + beta

/-- Flatten a tensor in TorchLean's canonical row-major order for parity comparisons. -/
def flattenRowMajor {s : Shape} (t : Tensor Float s) : Array Float :=
  t.toArray

/-- Small Python program used to compare TorchLean BatchNorm against PyTorch. -/
def batchNormParityScript : String :=
  String.intercalate "\n"
    [ "import json"
    , "import torch"
    , "import torch.nn.functional as F"
    , ""
    , "x = torch.tensor(["
    , "    [[[1., 2.], [3., 4.]], [[-5., -6.], [-7., -8.]]],"
    , "    [[[9., 10.], [11., 12.]], [[-13., -14.], [-15., -16.]]],"
    , "], dtype=torch.float32)"
    , "gamma = torch.tensor([1.0, 0.5], dtype=torch.float32)"
    , "beta = torch.tensor([0.0, 0.1], dtype=torch.float32)"
    , "running_mean = torch.tensor([2.0, -3.0], dtype=torch.float32)"
    , "running_var = torch.tensor([4.0, 9.0], dtype=torch.float32)"
    , "eps = 1e-5"
    , "mean = x.mean(dim=(0, 2, 3))"
    , "var = x.var(dim=(0, 2, 3), unbiased=False)"
    , "train = F.batch_norm(x, None, None, gamma, beta, training=True, eps=eps)"
    , "eval = F.batch_norm(x, running_mean, running_var, gamma, beta, training=False, eps=eps)"
    , "print(json.dumps({"
    , "    'mean': mean.flatten().tolist(),"
    , "    'var': var.flatten().tolist(),"
    , "    'train': train.flatten().tolist(),"
    , "    'eval': eval.flatten().tolist(),"
    , "}))"
    ]

/--
Compare TorchLean BatchNorm output and statistics against PyTorch, when PyTorch is installed.

The fallback skip is deliberate: the Lean-side closed-form checks still run on machines without a
Python/PyTorch environment.
-/
def checkBatchNormAgainstPyTorch
    (trainY evalY : Tensor Float bnShape)
    (mean var : Tensor Float [bnC]) : IO Unit := do
  if !(← pythonHasTorch) then
    IO.println "torchlean_ops_check: PyTorch BatchNorm parity skipped (`torch` not installed)"
    return ()
  IO.FS.createDirAll workDir
  IO.FS.writeFile batchNormParityScriptPath batchNormParityScript
  let out ← TorchLean.External.Process.runStdoutChecked
    (ctx := "torchlean_ops_check: batchnorm pytorch parity")
    (cmd := "python3")
    (args := #[batchNormParityScriptPath.toString])
    (cwd := some ".")
  let pyJson ←
    match Json.parse out with
    | .ok j => pure j
    | .error e => throw (IO.userError s!"torchlean_ops_check: bad BatchNorm parity JSON: {e}\n{out}")
  let readField key := do
    match jsonFloatArrayField pyJson key with
    | .ok xs => pure xs
    | .error e => throw (IO.userError s!"torchlean_ops_check: {e}")
  assertArrayApprox "batchnorm_nchw pytorch mean"
    #[vecVal mean ⟨0, by decide⟩, vecVal mean ⟨1, by decide⟩] (← readField "mean")
  assertArrayApprox "batchnorm_nchw pytorch var"
    #[vecVal var ⟨0, by decide⟩, vecVal var ⟨1, by decide⟩] (← readField "var")
  assertArrayApprox "batchnorm pytorch train" (flattenRowMajor trainY) (← readField "train")
  assertArrayApprox "batchnorm pytorch eval" (flattenRowMajor evalY) (← readField "eval")

/--
Run the full BatchNorm check: closed-form expectations first, then optional PyTorch parity.
-/
def checkBatchNorm : IO Unit := do
  let (trainY, mean, var) ← evalBatchNormTrain
  let evalY ← evalBatchNormEval

  assertApprox "batchnorm_nchw mean[0] expected" (vecVal mean ⟨0, by decide⟩) 6.5
  assertApprox "batchnorm_nchw mean[1] expected" (vecVal mean ⟨1, by decide⟩) (-10.5)
  assertApprox "batchnorm_nchw var[0] expected" (vecVal var ⟨0, by decide⟩) 17.25
  assertApprox "batchnorm_nchw var[1] expected" (vecVal var ⟨1, by decide⟩) 17.25

  for n in List.finRange bnN do
    for c in List.finRange bnC do
      for h in List.finRange bnH do
        for w in List.finRange bnW do
          let x := tensorVal bnInput [n, c, h, w]
          let gamma := vecVal bnGamma c
          let beta := vecVal bnBeta c
          let trainExpected :=
            expectedBatchNormFromBatchStats x gamma beta (vecVal mean c) (vecVal var c)
          let evalExpected :=
            expectedBatchNormFromRunningStats x gamma beta (vecVal bnMean c) (vecVal bnVar c)
          assertApprox s!"batchnorm_nchw train[{n.val},{c.val},{h.val},{w.val}] expected"
            (tensorVal trainY [n, c, h, w]) trainExpected 1e-5
          assertApprox s!"batchnorm_nchw eval[{n.val},{c.val},{h.val},{w.val}] expected"
            (tensorVal evalY [n, c, h, w]) evalExpected 1e-5

  checkBatchNormAgainstPyTorch trainY evalY mean var

/-! ## Loss and attention specification regressions -/

/-- Check reduction axes and the selected derivatives at clipped loss branches. -/
def checkLossSemantics : IO Unit := do
  let logits : Tensor Float [2, 3] :=
    tensorOfArray! [2, 3] #[0, 0, 0, 0, 0, 0]
  let target : Tensor Float [2, 3] :=
    tensorOfArray! [2, 3] #[1, 0, 0, 0, 1, 0]
  let logitsLoss := Spec.crossEntropyLogitsSpec 1 logits target
  assertApprox "cross entropy averages samples, not classes" logitsLoss (Float.log 3) 1e-5

  let probabilities : Tensor Float [3] :=
    tensorOfArray! [3] #[0.05, 0.5, 0.95]
  let distribution : Tensor Float [3] :=
    tensorOfArray! [3] #[1, 1, 1]
  let probabilityGrad :=
    Spec.crossEntropyDerivSpec 0 probabilities distribution (epsilon := 0.1)
  assertApprox "cross entropy lower clipped branch" (vecVal probabilityGrad ⟨0, by decide⟩) 0
  assertApprox "cross entropy interior branch" (vecVal probabilityGrad ⟨1, by decide⟩) (-2)
  assertApprox "cross entropy upper clipped branch" (vecVal probabilityGrad ⟨2, by decide⟩) 0

  assertApprox "BCE lower clipped branch"
    (Spec.binaryCrossEntropyDerivSpec 0.05 1 (epsilon := 0.1)) 0
  assertApprox "BCE interior branch"
    (Spec.binaryCrossEntropyDerivSpec 0.5 1 (epsilon := 0.1)) (-2)
  assertApprox "BCE upper clipped branch"
    (Spec.binaryCrossEntropyDerivSpec 0.95 1 (epsilon := 0.1)) 0

  let huberPrediction : Tensor Float [2] := tensorOfArray! [2] #[1, 3]
  let huberTarget : Tensor Float [2] := tensorOfArray! [2] #[0, 0]
  assertApprox "Huber delta=2 forward"
    (Spec.huberSpec huberPrediction huberTarget (delta := 2)) 2.25
  let huberGrad := Spec.huberDerivSpec huberPrediction huberTarget (delta := 2)
  assertApprox "Huber delta=2 quadratic gradient"
    (vecVal huberGrad ⟨0, by decide⟩) 0.5
  assertApprox "Huber delta=2 linear gradient"
    (vecVal huberGrad ⟨1, by decide⟩) 1
  assertApprox "RL Huber uses the same delta convention"
    (Runtime.RL.Core.huberLoss (α := Float) 3 0 2) 4

  let shortPrediction : Tensor Float [2] := tensorOfArray! [2] #[0.05, 0]
  let unitTarget : Tensor Float [2] := tensorOfArray! [2] #[1, 0]
  let cosineGrad :=
    Spec.cosineSimilarityDerivSpec shortPrediction unitTarget (epsilon := 0.1)
  assertApprox "cosine epsilon branch[0]" (vecVal cosineGrad ⟨0, by decide⟩) (-10)
  assertApprox "cosine epsilon branch[1]" (vecVal cosineGrad ⟨1, by decide⟩) 0

  assertApprox "zero-feature attention scale"
    (Spec.attentionScaleDenom (α := Float) 0) 1

def checkCorrectedMathematicalSpecs : IO Unit := do
  let matrix : Tensor Float [2, 2] :=
    tensorOfArray! [2, 2] #[0, 2, 4, 6]
  assertApprox "tensor-wide population variance"
    (Spec.Tensor.varianceSpec matrix) 5

  -- Each GroupNorm group contains one channel and both spatial positions. This fixture checks that
  -- the generic spatial suffix is reduced independently for each channel group.
  let groupNormInput : Tensor Float [1, 2, 1, 2] :=
    tensorOfArray! [1, 2, 1, 2] #[1, 10, 3, 14]
  let groupNormScale : Tensor Float [2] := tensorOfArray! [2] #[1, 1]
  let groupNormBias : Tensor Float [2] := tensorOfArray! [2] #[0, 0]
  let groupNormOutput := Spec.groupNorm (groups := 2)
    groupNormInput groupNormScale groupNormBias (hGroupsLe := by decide) (hDiv := by decide)
  assertArrayApprox "GroupNorm channel grouping" groupNormOutput.toArray
    #[-0.9999998, 0.9999998, -0.99999994, 0.99999994] 1e-5

  let clustered : Tensor Float [2] :=
    tensorOfArray! [2] #[1.0e12, 1.0e12 + 1]
  let clusteredVariance := Spec.Tensor.reduceVar 0 clustered Spec.Shape.NonemptyAxis.zero
  assertApprox "centered population variance"
    (scalarVal clusteredVariance) 0.25 1e-8

  assertApprox "softplus large positive input"
    (Activation.Math.softplusSpec (1000 : Float)) 1000 1e-10

  let dropoutInput : Tensor Float [2] := tensorOfArray! [2] #[3, -4]
  let dropoutOutput := Spec.dropoutInferenceSpec (p := 0.75) dropoutInput
  assertApprox "evaluation dropout is identity[0]"
    (vecVal dropoutOutput ⟨0, by decide⟩) 3
  assertApprox "evaluation dropout is identity[1]"
    (vecVal dropoutOutput ⟨1, by decide⟩) (-4)

  let emptySpatial : Spec.Tensor Nat [1] := tensor! [0]
  let unitKernel : Spec.Tensor Nat [1] := tensor! [1]
  let unitStride : Spec.Tensor Nat [1] := tensor! [1]
  let zeroPadding : Spec.Tensor Nat [1] := tensor! [0]
  unless (Spec.poolOutSpatialPad emptySpatial unitKernel unitStride zeroPadding).getScalar 0 = 0 do
    throw <| IO.userError "pooling emitted a fully padded window for an empty input axis"
  let emptyPoolShape ← match NN.IR.OpContracts.inferPoolOutShape "max_pool"
      unitKernel unitStride zeroPadding (Shape.ofList [1, 0]) with
    | .ok shape => pure shape
    | .error message => throw <| IO.userError message
  unless emptyPoolShape = Shape.ofList [1, 0] do
    throw <| IO.userError "IR pooling rejected the specification's empty-axis semantics"
  let oversizedKernel : Spec.Tensor Nat [1] := tensor! [3]
  let oversizedPoolShape ← match NN.IR.OpContracts.inferPoolOutShape "max_pool"
      oversizedKernel unitStride zeroPadding (Shape.ofList [1, 1]) with
    | .ok shape => pure shape
    | .error message => throw <| IO.userError message
  unless oversizedPoolShape = Shape.ofList [1, 0] do
    throw <| IO.userError "IR pooling rejected the specification's oversized-window semantics"

  let schedule : Generative.Diffusion.VPLinearSchedule Float :=
    { beta0 := 1, beta1 := 2 }
  let epsModel : Generative.Diffusion.EpsModel Float [1] :=
    { eps := fun _ _ => tensorOfArray! [1] #[1] }
  let state : Tensor Float [1] := tensorOfArray! [1] #[2]
  let t : Float := 0.75
  let rhs := Generative.Diffusion.pfOdeRhs schedule epsModel state t
  let beta := schedule.beta t
  let sigma := schedule.sigma t
  let expectedRhs :=
    Numbers.negHalf * beta * 2 +
      Numbers.half * Generative.Diffusion.safeDiv beta sigma
  assertApprox "probability-flow ODE coefficient"
    (vecVal rhs ⟨0, by decide⟩) expectedRhs

  let dt : Float := -0.5
  let afterT1 := Generative.Diffusion.eulerStep
    (Generative.Diffusion.pfOdeRhs schedule epsModel) state 1 dt
  let expectedSample := Generative.Diffusion.eulerStep
    (Generative.Diffusion.pfOdeRhs schedule epsModel) afterT1 0.5 dt
  let sampled := Generative.Diffusion.pfOdeSampleEuler schedule epsModel 2 state
  assertApprox "probability-flow Euler time order"
    (vecVal sampled ⟨0, by decide⟩) (vecVal expectedSample ⟨0, by decide⟩)

  let impossibleHmm : Spec.HMMSpec Float 1 1 :=
    { initial := tensorOfArray! [1] #[1]
      transition := tensorOfArray! [1, 1] #[1]
      emission := tensorOfArray! [1, 1] #[0] }
  let observations : Spec.ObservationSeq 1 1 :=
    Spec.Tensor.ofFn fun _ => ⟨0, by decide⟩
  assertApprox "impossible HMM observation has zero likelihood"
    (Spec.hmmForwardSpec impossibleHmm observations) 0
  unless (Spec.hmmLogLikelihoodSpec impossibleHmm observations).isNone do
    throw <| IO.userError "impossible HMM observation received a finite log-likelihood"

  let singular : Tensor Float [1, 1] := tensorOfArray! [1, 1] #[0]
  unless (Spec.inverseSpec? singular).isNone do
    throw <| IO.userError "singular matrix inverse returned a value"
  let invalidGmm : Spec.GMMSpec Float 1 1 :=
    { weights := tensorOfArray! [1] #[1]
      means := tensorOfArray! [1, 1] #[0]
      covariances := tensorOfArray! [1, 1, 1] #[0] }
  unless (Spec.gmmForwardSpec invalidGmm (tensorOfArray! [1] #[0])).isNone do
    throw <| IO.userError "GMM accepted a singular covariance"
  let unnormalizedGmm : Spec.GMMSpec Float 2 1 :=
    { weights := tensorOfArray! [2] #[0.6, 0.6]
      means := tensorOfArray! [2, 1] #[0, 1]
      covariances := tensorOfArray! [2, 1, 1] #[1, 1] }
  unless (Spec.gmmForwardSpec unnormalizedGmm (tensorOfArray! [1] #[0])).isNone do
    throw <| IO.userError "GMM accepted mixture weights that do not sum to one"
  let negativeDefiniteGmm : Spec.GMMSpec Float 1 2 :=
    { weights := tensorOfArray! [1] #[1]
      means := tensorOfArray! [1, 2] #[0, 0]
      covariances := tensorOfArray! [1, 2, 2] #[-1, 0, 0, -1] }
  unless (Spec.gmmForwardSpec negativeDefiniteGmm (tensorOfArray! [2] #[0, 0])).isNone do
    throw <| IO.userError "GMM accepted a negative-definite covariance with positive determinant"
  let validGmm : Spec.GMMSpec Float 1 2 :=
    { weights := tensorOfArray! [1] #[1]
      means := tensorOfArray! [1, 2] #[0, 0]
      covariances := tensorOfArray! [1, 2, 2] #[2, 0.5, 0.5, 1] }
  unless (Spec.gmmForwardSpec validGmm (tensorOfArray! [2] #[0, 0])).isSome do
    throw <| IO.userError "GMM rejected a symmetric positive-definite covariance"

/-- Entrypoint called by the curated float runtime suite. -/
def run : IO Unit := do
  IO.println "torchlean_ops_check: begin"
  checkSoftmaxDimensions
  checkClassificationAxes
  checkWeightedRowCrossEntropy
  checkMiddleAxisCrossEntropy
  checkTiedTokenEmbeddingParameterCount
  checkTiedTokenEmbeddingBackward
  checkAttentionOutputProjectionInitializer
  checkTypedGraphLeafMetadata

  let softmaxInput : Tensor Float [2, 2, 2] :=
    tensorOfArray! [2, 2, 2] #[0, 2, 1, 4, 3, 8, 7, 9]
  let expectedSoftmax := Activation.softmaxSpec (α := Float) 1 softmaxInput
  let expectedLogSoftmax := Activation.logSoftmaxSpec (α := Float) 1 softmaxInput
  for execution in [.eager, .typedGraph] do
    let executionName := match execution with
      | .eager => "eager"
      | .typedGraph => "typed_graph"
    let (actualSoftmax, actualLogSoftmax) ← evalSessionSoftmaxFixture execution
    for (actual, expected) in actualSoftmax.toArray.zip expectedSoftmax.toArray do
      assertApprox s!"session softmax axis 1 ({executionName})" actual expected 1e-5
    for (actual, expected) in actualLogSoftmax.toArray.zip expectedLogSoftmax.toArray do
      assertApprox s!"session log-softmax axis 1 ({executionName})" actual expected 1e-5

  let mmE ← evalMatmulFixture .eager
  let mmC ← evalMatmulFixture .typedGraph
  for i in List.finRange 2 do
    for j in List.finRange 2 do
      assertApprox s!"matmul[{i.val},{j.val}] eager/typed-graph" (matVal mmE i j) (matVal mmC i j) 1e-5

  let cvE ← evalConcatFixture .eager
  let cvC ← evalConcatFixture .typedGraph
  for i in List.finRange 5 do
    assertApprox s!"concat[{i.val}] eager/typed-graph" (vecVal cvE i) (vecVal cvC i) 1e-5

  let mpE ← evalMaxPoolFixture .eager
  let mpC ← evalMaxPoolFixture .typedGraph
  for hi in List.finRange 2 do
    for wi in List.finRange 2 do
      assertApprox s!"max_pool[{hi.val},{wi.val}] eager/typed-graph"
        (tensorVal mpE [0, hi, wi])
        (tensorVal mpC [0, hi, wi])
        1e-5

  let apE ← evalAvgPoolFixture .eager
  let apC ← evalAvgPoolFixture .typedGraph
  for hi in List.finRange 2 do
    for wi in List.finRange 2 do
      assertApprox s!"avg_pool[{hi.val},{wi.val}] eager/typed-graph"
        (tensorVal apE [0, hi, wi])
        (tensorVal apC [0, hi, wi])
        1e-5

  checkBatchNorm
  checkLossSemantics
  checkCorrectedMathematicalSpecs

  IO.println "torchlean_ops_check: ok"

end TorchLeanOpsCheck
end Floats
end Tests
