/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec
public import NN.API.Public.TensorPack
public import NN.MLTheory.CROWN.Graph
public import NN.MLTheory.CROWN.Lyapunov.TwoStage.Core
public import NN.MLTheory.CROWN.Lyapunov.TwoStage.ExecUtils
public import NN.Runtime.Autograd.TorchLean.Autodiff
public import NN.Verification.TorchLean.Compile

/-!
# Compiled Loss Analysis for Two-Stage Lyapunov Workflows

The executable operations shared by the hybrid and all-in-Lean pipelines: projected gradient
ascent on the compiled loss and an IBP/CROWN check of that loss over an input box.
-/

@[expose] public section

open Spec
open Tensor

namespace NN.MLTheory.CROWN.Lyapunov.TwoStage.CompiledLossAnalysis

open _root_.TorchLean.Floats.IEEE754
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN.Lyapunov.TwoStage

/-- Executable float32 semantics used by both compiled two-stage pipelines. -/
abbrev Scalar : Type := IEEE32Exec

/-- Shapes supplied to the compiled scalar loss: parameters followed by one state vector. -/
abbrev LossInputs (width : Nat) : List Shape := Core.paramShapes width ++ [Core.xShape]

/--
Take one projected gradient-ascent step on the state input of a compiled scalar loss.

Only the state gradient is retained; parameters remain fixed. The result is projected back into
the coordinate box `[-radius, radius]`.
-/
def projectedGradientStep
    (width : Nat)
    (compiledLoss : _root_.Runtime.Autograd.Torch.CompiledScalar Scalar (LossInputs width))
    (parameters : NN.API.TorchLean.TensorPack Scalar (Core.paramShapes width))
    (state : Tensor Scalar Core.xShape)
    (stepSize radius : Scalar) : Tensor Scalar Core.xShape :=
  let arguments : NN.API.TorchLean.TensorPack Scalar (LossInputs width) :=
    _root_.Proofs.Autograd.Algebra.TList.append (α := Scalar)
      (ss₁ := Core.paramShapes width) (ss₂ := [Core.xShape]) parameters (.cons state .nil)
  let allGradients : NN.API.TorchLean.TensorPack Scalar (LossInputs width) :=
    _root_.Runtime.Autograd.Torch.CompiledScalar.backward
      (α := Scalar) (Γ := LossInputs width) compiledLoss arguments
  let stateGradients : NN.API.TorchLean.TensorPack Scalar [Core.xShape] :=
    (_root_.Proofs.Autograd.Algebra.TList.splitAppend (α := Scalar)
      (ss₁ := Core.paramShapes width) (ss₂ := [Core.xShape]) allGradients).2
  let .cons gradient .nil := stateGradients
  let updated := Tensor.addSpec state (Tensor.scaleSpec gradient stepSize)
  ExecUtils.clampStateVector (-radius) radius updated

-- Lean 4.32's native-code optimizer needs more than the default budget for this compiled graph.
set_option maxHeartbeats 1000000 in
/--
Compile the shared scalar loss and report its IBP and CROWN bounds on an origin-centered box.
-/
def checkLossBox
    (width : Nat)
    (parameters : NN.API.TorchLean.TensorPack Scalar (Core.paramShapes width))
    (epsilon : Scalar) : IO Unit := do
  IO.println "Stage 2 check: IBP + CROWN on the scalar loss over a small box"
  let compiled ←
    match NN.Verification.TorchLean.compileForward
          (α := Scalar) (paramShapes := Core.paramShapes width)
          (inShape := Core.xShape) (outShape := Shape.scalar)
          (Core.lossProgram width (β := Scalar)) parameters with
    | .ok result => pure result
    | .error error => throw <| IO.userError error

  IO.println s!"compiled IR nodes: {compiled.graph.nodes.size}"

  let origin : Tensor Scalar Core.xShape := Spec.zeros (α := Scalar) Core.xShape
  let inputBox : FlatBox Scalar := FlatBox.lInfBall (α := Scalar) origin epsilon
  let parameterStore : ParamStore Scalar := compiled.seedInputBox inputBox

  let intervalBounds := runIBP (α := Scalar) compiled.graph parameterStore
  let outputBounds ← compiled.outputBoxOrThrow intervalBounds
  IO.println s!"[IBP] scalar loss box dim={outputBounds.dim}"

  match outputBoxCROWN? compiled.graph parameterStore inputBox
      compiled.inputId compiled.outputId Core.xDim with
  | .ok bounds =>
      IO.println s!"[CROWN] loss lo = {pretty bounds.lo}"
      IO.println s!"[CROWN] loss hi = {pretty bounds.hi}"
  | .error message =>
      IO.println s!"[CROWN] {message}"

end NN.MLTheory.CROWN.Lyapunov.TwoStage.CompiledLossAnalysis
