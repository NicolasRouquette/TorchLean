/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Neural.Execution
public import NN.API.Tensor
public import NN.MLTheory.CROWN.Flatbox
public import NN.MLTheory.CROWN.Graph
public import NN.Verification.TorchLean.Lowering

/-!
# Verification

Convenience names for lowering TorchLean models into IBP/CROWN checks.
-/

@[expose] public section

namespace TorchLean

namespace Verification

export NN.Verification.TorchLean (LoweredIR)
export NN.MLTheory.CROWN (FlatBox)
export NN.MLTheory.CROWN.Graph (ParamStore AffineCtx FlatAffine FlatAffineBounds)

/--
Lower a sequential TorchLean model into verifier IR with one distinguished input.

Usual "train a model, then run IBP/CROWN on its forward pass" path.

This is the broad executable lowering. It checks the resulting graph but does not attach the
source-evaluation theorem from `NN.Verification.TorchLean.Proved`.
-/
def lowerForwardToIR {α : Type} [_root_.Context α] [DecidableEq Shape]
    {σ τ : Shape}
    (model : TorchLean.nn.Sequential σ τ)
    (params : TensorPack α (TorchLean.nn.stateShapes model)) :
    Except String (LoweredIR α) :=
  NN.Verification.TorchLean.lowerForwardToIR
    (α := α) (paramShapes := TorchLean.nn.stateShapes model)
    (inShape := σ) (outShape := τ)
    (TorchLean.nn.forward model (α := α)) params

/--
Lower a custom TorchLean forward program into verifier IR with one distinguished input.

Use this when the target is not a plain `TorchLean.nn.Sequential`, for example a hand-written loss program or
an attention fragment built directly from `TorchLean.Ops`.

Successful lowering validates the produced IR. It is not an end-to-end proof that every accepted
custom program agrees with IR denotation.
-/
def lowerProgramToIR {α : Type} [_root_.Context α] [DecidableEq Shape]
    {paramShapes : List Shape} {σ τ : Shape}
    (forwardProgram : _root_.Runtime.Autograd.TorchLean.Program α (paramShapes ++ [σ]) τ)
    (params : TensorPack α paramShapes) :
    Except String (LoweredIR α) :=
  NN.Verification.TorchLean.lowerForwardToIR
    (α := α) (paramShapes := paramShapes)
    (inShape := σ) (outShape := τ)
    forwardProgram params

/--
Seed the verifier input with an explicit input box.

Call this after `lowerForwardToIR`, then hand the returned store to IBP/CROWN passes.
-/
def seedInputBox {α : Type} [_root_.Context α]
    (lowered : LoweredIR α) (xB : FlatBox α) : ParamStore α :=
  lowered.seedInputBox xB

/--
Flatten a center tensor and radius tensor into the `FlatBox` expected by IBP/CROWN.

Use this for a shaped TorchLean input with a shaped perturbation radius.
-/
def lInfBox {α : Type} [_root_.Context α] {s : Shape}
    (center radius : _root_.Spec.Tensor α s) : FlatBox α :=
  NN.Verification.TorchLean.lInfBox (α := α) center radius

/--
Build a uniform $\ell^\infty$ box around a shaped TorchLean input tensor.

This fills the input shape with the scalar radius `eps`, then flattens it into a verifier box.
-/
def lInfBall {α : Type} [_root_.Context α] {s : Shape}
    (center : _root_.Spec.Tensor α s) (eps : α) : FlatBox α :=
  NN.Verification.TorchLean.lInfBall (α := α) center eps

/--
Seed the lowered verifier input with a uniform $\ell^\infty$ box around a shaped TorchLean input
tensor.
-/
def seedLInfBall {α : Type} [_root_.Context α] {s : Shape}
    (lowered : LoweredIR α) (center : _root_.Spec.Tensor α s) (eps : α) : ParamStore α :=
  lowered.seedLInfBall center eps

/-- Shape of the distinguished verifier input node. -/
def inputShape? {α : Type} [Context α] (lowered : LoweredIR α) : Except String Shape :=
  lowered.inputShape?

/-- Flattened dimension of the distinguished verifier input node. -/
def inputDim? {α : Type} [Context α] (lowered : LoweredIR α) : Except String Nat :=
  lowered.inputDim?

/-- Affine context for the distinguished verifier input. -/
def affineCtx? {α : Type} [Context α] (lowered : LoweredIR α) :
    Except String AffineCtx :=
  lowered.affineCtx?

/-- Run IBP on a lowered verifier graph. -/
def runIBP {α : Type} [Context α] [NN.MLTheory.CROWN.BoundOps α]
    [NN.MLTheory.CROWN.NonlinearBoundOps α]
    (lowered : LoweredIR α) (ps : ParamStore α) : Array (Option (FlatBox α)) :=
  lowered.runIBP ps

/-- Read the verifier output box from an IBP result array. -/
def outputBox? {α : Type} [Context α] (lowered : LoweredIR α)
    (boxes : Array (Option (FlatBox α))) : Except String (FlatBox α) := do
  lowered.outputBox? boxes

/-- Read the verifier output box, throwing an `IO.userError` if it is missing. -/
def outputBoxOrThrow {α : Type} [Context α] (lowered : LoweredIR α)
    (boxes : Array (Option (FlatBox α))) : IO (FlatBox α) :=
  lowered.outputBoxOrThrow boxes

/-- Run the forward affine pass after validating the lowered verifier input. -/
def runAffine {α : Type} [Context α] [NN.MLTheory.CROWN.BoundOps α]
    (lowered : LoweredIR α) (ps : ParamStore α) (ibp : Array (Option (FlatBox α))) :
    Except String (Array (Option (FlatAffine α))) := do
  let ctx ← affineCtx? lowered
  pure <| NN.MLTheory.CROWN.Graph.runAffine (α := α) lowered.graph ps ctx ibp

/-- Read the verifier output affine form from a forward affine result array. -/
def outputAffine? {α : Type} [Context α] (lowered : LoweredIR α)
    (affs : Array (Option (FlatAffine α))) : Except String (FlatAffine α) := do
  lowered.outputAffine? affs

/-- Run CROWN after validating the lowered verifier input. -/
def runCROWN {α : Type} [Context α] [NN.MLTheory.CROWN.BoundOps α]
    [NN.MLTheory.CROWN.NonlinearBoundOps α]
    (lowered : LoweredIR α) (ps : ParamStore α) (ibp : Array (Option (FlatBox α))) :
    Except String (Array (Option (FlatAffineBounds α))) := do
  let ctx ← affineCtx? lowered
  pure <| NN.MLTheory.CROWN.Graph.runCROWN (α := α) lowered.graph ps ctx ibp

/-- Read the verifier output CROWN bounds from a CROWN result array. -/
def outputCROWN? {α : Type} [Context α] (lowered : LoweredIR α)
    (bounds : Array (Option (FlatAffineBounds α))) : Except String (FlatAffineBounds α) := do
  lowered.outputCROWN? bounds

/-- Run forward CROWN and evaluate the verifier output bounds on the lowered input box. -/
def outputBoxCROWN? {α : Type} [Context α] [NN.MLTheory.CROWN.BoundOps α]
    [NN.MLTheory.CROWN.NonlinearBoundOps α]
    (lowered : LoweredIR α) (ps : ParamStore α) (xB : FlatBox α) :
    Except String (FlatBox α) :=
  lowered.outputBoxCROWN? ps xB

/-- Run forward CROWN and return the evaluated verifier output box, throwing on failure. -/
def outputBoxCROWNOrThrow {α : Type} [Context α] [NN.MLTheory.CROWN.BoundOps α]
    [NN.MLTheory.CROWN.NonlinearBoundOps α]
    (lowered : LoweredIR α) (ps : ParamStore α) (xB : FlatBox α) : IO (FlatBox α) :=
  lowered.outputBoxCROWNOrThrow ps xB

/-- Run backward CROWN for a scalar objective and evaluate it on the lowered input box. -/
def backwardObjectiveBox? {α : Type} [Context α] [NN.MLTheory.CROWN.BoundOps α]
    (lowered : LoweredIR α) (ps : ParamStore α) (ibp : Array (Option (FlatBox α)))
    (xB : FlatBox α) (obj : NN.MLTheory.CROWN.Graph.FlatVec α) :
    Except String (FlatBox α) :=
  lowered.backwardObjectiveBox? ps ibp xB obj

/-- `IO` version of `backwardObjectiveBox?`. -/
def backwardObjectiveBoxOrThrow {α : Type} [Context α] [NN.MLTheory.CROWN.BoundOps α]
    (lowered : LoweredIR α) (ps : ParamStore α) (ibp : Array (Option (FlatBox α)))
    (xB : FlatBox α) (obj : NN.MLTheory.CROWN.Graph.FlatVec α) : IO (FlatBox α) :=
  lowered.backwardObjectiveBoxOrThrow ps ibp xB obj

/--
Compute the conservative two-class margin lower bound
$\mathrm{lo}[\mathrm{class0}]-\mathrm{hi}[\mathrm{class1}]$.

If this is positive, class `class0` is certified against `class1` over the input box.
-/
def twoClassMarginLowerBound {α : Type} [Context α] {n : Nat}
    (lo hi : _root_.Spec.Tensor α (.dim n .scalar)) (class0 class1 : Fin n) : α :=
  Spec.Tensor.vecGet lo class0 - Spec.Tensor.vecGet hi class1

/-- Decide whether the two-class margin lower bound is strictly positive. -/
def certifiesTwoClassMargin {α : Type} [Context α] {n : Nat}
    (lo hi : _root_.Spec.Tensor α (.dim n .scalar)) (class0 class1 : Fin n) : Bool :=
  Context.gtBool
    (twoClassMarginLowerBound (α := α) lo hi class0 class1) (0 : α)

end Verification


end TorchLean
