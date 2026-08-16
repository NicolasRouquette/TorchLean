/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec
public import NN.API.TensorPack
public import NN.MLTheory.CROWN.Graph
public import NN.MLTheory.CROWN.Lyapunov.TwoStage.Core
public import NN.MLTheory.CROWN.Lyapunov.TwoStage.ExecUtils
public import NN.Runtime.Autograd.TorchLean.Autodiff
public import NN.Verification.TorchLean.Lowering

/-!
# Lowered Loss Analysis for Two-Stage Lyapunov Workflows

The executable operations shared by the hybrid and all-in-Lean pipelines: projected gradient
ascent on the lowered loss and an IBP/CROWN check of that loss over an input box.
-/

@[expose] public section

open Spec
open Tensor

namespace NN.MLTheory.CROWN.Lyapunov.TwoStage.LossAnalysis

open _root_.TorchLean.Floats.IEEE754
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN.Lyapunov.TwoStage

/-- Executable float32 semantics used by both lowered two-stage pipelines. -/
abbrev Scalar : Type := IEEE32Exec

/-- Shapes supplied to the lowered scalar loss: parameters followed by one state vector. -/
abbrev LossInputs (width : Nat) : List Shape := Core.paramShapes width ++ [Core.xShape]

/--
Take one projected gradient-ascent step on the state input of a lowered scalar loss.

Only the state gradient is retained; parameters remain fixed. The result is projected back into
the coordinate box `[-radius, radius]`.
-/
def projectedGradientStep
    (width : Nat)
    (lossGraph : _root_.Runtime.Autograd.Torch.TypedScalarGraph Scalar (LossInputs width))
    (parameters : _root_.TorchLean.TensorPack Scalar (Core.paramShapes width))
    (state : Tensor Scalar Core.xShape)
    (stepSize radius : Scalar) : Tensor Scalar Core.xShape :=
  let arguments : _root_.TorchLean.TensorPack Scalar (LossInputs width) :=
    _root_.Proofs.Autograd.Algebra.TList.append (α := Scalar)
      (ss₁ := Core.paramShapes width) (ss₂ := [Core.xShape]) parameters (.cons state .nil)
  let allGradients : _root_.TorchLean.TensorPack Scalar (LossInputs width) :=
    _root_.Runtime.Autograd.Torch.TypedScalarGraph.backward
      (α := Scalar) (Γ := LossInputs width) lossGraph arguments
  let stateGradients : _root_.TorchLean.TensorPack Scalar [Core.xShape] :=
    (_root_.Proofs.Autograd.Algebra.TList.splitAppend (α := Scalar)
      (ss₁ := Core.paramShapes width) (ss₂ := [Core.xShape]) allGradients).2
  let .cons gradient .nil := stateGradients
  let updated := Tensor.addSpec state (Tensor.scaleSpec gradient stepSize)
  ExecUtils.clampStateVector (-radius) radius updated

/-- Lower the Lyapunov loss while keeping its large polymorphic program out of callers' code. -/
@[noinline] def lowerLossToIR
    (width : Nat)
    (parameters : _root_.TorchLean.TensorPack Scalar (Core.paramShapes width)) :
    Except String (NN.Verification.TorchLean.LoweredIR Scalar) :=
  NN.Verification.TorchLean.lowerForwardToIR
    (α := Scalar) (paramShapes := Core.paramShapes width)
    (inShape := Core.xShape) (outShape := Shape.scalar)
    (Core.lossProgram width (β := Scalar)) parameters

/-- Run and print the IBP bound for an already lowered loss graph. -/
@[noinline] def reportIBP
    (lowered : NN.Verification.TorchLean.LoweredIR Scalar)
    (parameterStore : ParamStore Scalar) : IO Unit := do
  let intervalBounds := runIBP (α := Scalar) lowered.graph parameterStore
  let outputBounds ← lowered.outputBoxOrThrow intervalBounds
  IO.println s!"[IBP] scalar loss box dim={outputBounds.dim}"

/-- Run and print the CROWN bound for an already lowered loss graph. -/
@[noinline] def reportCROWN
    (lowered : NN.Verification.TorchLean.LoweredIR Scalar)
    (inputBox : FlatBox Scalar) (parameterStore : ParamStore Scalar) : IO Unit := do
  match outputBoxCROWN? lowered.graph parameterStore inputBox
      lowered.inputId lowered.outputId Core.xDim with
  | .ok bounds =>
      IO.println s!"[CROWN] loss lo = {pretty bounds.lo}"
      IO.println s!"[CROWN] loss hi = {pretty bounds.hi}"
  | .error message =>
      IO.println s!"[CROWN] {message}"

/--
Lower the shared scalar loss and report its IBP and CROWN bounds on an origin-centered box.
-/
def checkLossBox
    (width : Nat)
    (parameters : _root_.TorchLean.TensorPack Scalar (Core.paramShapes width))
    (epsilon : Scalar) : IO Unit := do
  IO.println "Stage 2 check: IBP + CROWN on the scalar loss over a small box"
  let lowered ←
    match lowerLossToIR width parameters with
    | .ok result => pure result
    | .error error => throw <| IO.userError error

  IO.println s!"lowered IR nodes: {lowered.graph.nodes.size}"

  let origin : Tensor Scalar Core.xShape := Spec.zeros (α := Scalar) Core.xShape
  let inputBox : FlatBox Scalar := FlatBox.lInfBall (α := Scalar) origin epsilon
  let parameterStore : ParamStore Scalar := lowered.seedInputBox inputBox

  reportIBP lowered parameterStore
  reportCROWN lowered inputBox parameterStore

end NN.MLTheory.CROWN.Lyapunov.TwoStage.LossAnalysis
