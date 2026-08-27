/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Neural.Execution
public import NN.Tensor
public import NN.MLTheory.CROWN.Flatbox
public import NN.MLTheory.CROWN.Graph
public import NN.Verification.TorchLean.Lowering

/-!
# Verification

This module lowers public TorchLean models into the verifier graph and runs the affine bound
passes on that graph. `FlatBox`, `ParamStore`, and the affine result types below are the canonical
CROWN values. They are re-exported here so applications can construct verification problems
without importing the CROWN modules directly.
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
    (params : _root_.TorchLean.TensorPack α (TorchLean.nn.stateShapes model)) :
    Except String (LoweredIR α) :=
  NN.Verification.TorchLean.lowerForwardToIR
    (α := α) (paramShapes := TorchLean.nn.stateShapes model)
    (inShape := σ) (outShape := τ)
    (TorchLean.nn.forward model (α := α)) params

/--
Lower a custom TorchLean forward program into verifier IR with one distinguished input.

Use this when the target is not a plain `TorchLean.nn.Sequential`, for example a hand-written loss
program or an attention fragment built directly from `TorchLean.Runtime` operations.

Successful lowering validates the produced IR. It is not an end-to-end proof that every accepted
custom program agrees with IR denotation.
-/
def lowerProgramToIR {α : Type} [_root_.Context α] [DecidableEq Shape]
    {paramShapes : List Shape} {σ τ : Shape}
    (forwardProgram : _root_.Runtime.Autograd.TorchLean.Program α (paramShapes ++ [σ]) τ)
    (params : _root_.TorchLean.TensorPack α paramShapes) :
    Except String (LoweredIR α) :=
  NN.Verification.TorchLean.lowerForwardToIR
    (α := α) (paramShapes := paramShapes)
    (inShape := σ) (outShape := τ)
    forwardProgram params

/-- Dimensions of the distinguished verifier input node. -/
def inputShape? {α : Type} [Context α] (lowered : LoweredIR α) : Except String (List Nat) :=
  return (← lowered.inputShape?).toList

/-- Run the forward affine pass after validating the lowered verifier input. -/
def runAffine {α : Type} [Context α] [NN.MLTheory.CROWN.BoundOps α]
    (lowered : LoweredIR α) (ps : ParamStore α) (ibp : Array (Option (FlatBox α))) :
    Except String (Array (Option (FlatAffine α))) := do
  let ctx ← lowered.affineCtx?
  pure <| NN.MLTheory.CROWN.Graph.runAffine (α := α) lowered.graph ps ctx ibp

/-- Run CROWN after validating the lowered verifier input. -/
def runCROWN {α : Type} [Context α] [NN.MLTheory.CROWN.BoundOps α]
    [NN.MLTheory.CROWN.NonlinearBoundOps α]
    (lowered : LoweredIR α) (ps : ParamStore α) (ibp : Array (Option (FlatBox α))) :
    Except String (Array (Option (FlatAffineBounds α))) := do
  let ctx ← lowered.affineCtx?
  pure <| NN.MLTheory.CROWN.Graph.runCROWN (α := α) lowered.graph ps ctx ibp

/--
Compute the conservative two-class margin lower bound
$\mathrm{lo}[\mathrm{class0}]-\mathrm{hi}[\mathrm{class1}]$.

If this is positive, class `class0` is certified against `class1` over the input box.
-/
def twoClassMarginLowerBound {α : Type} [Context α] {n : Nat}
    (lo hi : Tensor α [n]) (class0 class1 : Fin n) : α :=
  Tensor.item (Tensor.get lo class0) - Tensor.item (Tensor.get hi class1)

/-- Decide whether the two-class margin lower bound is strictly positive. -/
def certifiesTwoClassMargin {α : Type} [Context α] {n : Nat}
    (lo hi : Tensor α [n]) (class0 class1 : Fin n) : Bool :=
  Context.gtBool
    (twoClassMarginLowerBound (α := α) lo hi class0 class1) (0 : α)

end Verification


end TorchLean
