/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor
public import NN.Spec.Models.Mlp

/-!
# Closed MLP Model Facts

Small rational MLP examples belong in the proof layer. The forward composition law is already
proved generally by `Examples.mlp_spec_forward_eq`; the facts below state a deterministic backward
calculation as coordinate-level Lean theorems.
-/

open scoped NN.Spec.RationalAlgebraic

@[expose] public section

namespace NN.Proofs.Models.Mlp

open _root_.NN.Spec
open Spec.Tensor
open Examples
open ModSpec

abbrev exInDim  := 2
abbrev exHidDim := 3
abbrev exOutDim := 1

def exampleHiddenWeight : Spec.Tensor ℚ (.dim exHidDim (.dim exInDim .scalar)) :=
  Spec.matrixTensor (fun i j =>
    match i.val, j.val with
    | 0, 0 => 1 / 10
    | 0, 1 => 1 / 5
    | 1, 0 => 3 / 10
    | 1, 1 => 2 / 5
    | 2, 0 => 1 / 2
    | 2, 1 => 3 / 5
    | _, _ => 0)

def exampleHiddenBias : Spec.Tensor ℚ (.dim exHidDim .scalar) :=
  Spec.vectorTensor (fun i =>
    match i.val with
    | 0 => 1 / 10
    | 1 => 1 / 5
    | 2 => 3 / 10
    | _ => 0)

def exampleOutputWeight : Spec.Tensor ℚ (.dim exOutDim (.dim exHidDim .scalar)) :=
  Spec.matrixTensor (fun _ j =>
    match j.val with
    | 0 => 7 / 10
    | 1 => 4 / 5
    | 2 => 9 / 10
    | _ => 0)

def exampleOutputBias : Spec.Tensor ℚ (.dim exOutDim .scalar) :=
  Spec.vectorTensor (fun _ => 2 / 5)

def exampleHiddenLayer : Spec.LinearSpec ℚ exInDim exHidDim :=
  { weights := exampleHiddenWeight, bias := exampleHiddenBias }

def exampleOutputLayer : Spec.LinearSpec ℚ exHidDim exOutDim :=
  { weights := exampleOutputWeight, bias := exampleOutputBias }

def exInput : Spec.Tensor ℚ (.dim exInDim .scalar) :=
  Spec.vectorTensor (fun i =>
    match i.val with
    | 0 => 1 / 2
    | 1 => 4 / 5
    | _ => 0)

def exNet : SpecChain ℚ (.dim exInDim .scalar) (.dim exOutDim .scalar) :=
  Examples.mlpSpec (α := ℚ) exampleHiddenLayer exampleOutputLayer

def exOutput : Spec.Tensor ℚ (.dim exOutDim .scalar) :=
  SpecChain.forward (α := ℚ) exNet exInput

def exExpected : Spec.Tensor ℚ (.dim exOutDim .scalar) :=
  let z1 := Spec.linearSpec (α := ℚ) exampleHiddenLayer exInput
  let a1 := Activation.reluSpec z1
  Spec.linearSpec (α := ℚ) exampleOutputLayer a1

def exDLdy : Spec.Tensor ℚ (.dim exOutDim .scalar) :=
  Spec.vectorTensor (fun _ => 1)

def exGrad :=
  Examples.mlpBackward (α := ℚ) exampleHiddenLayer exampleOutputLayer exInput exDLdy

def exDXOpspec :=
  Examples.mlpOpspecBackward (α := ℚ) exampleHiddenLayer exampleOutputLayer exInput exDLdy

def dXHand : Spec.Tensor ℚ (.dim exInDim .scalar) :=
  match exGrad with
  | (_, _, _, _, dX) => dX

example :
    exOutput = exExpected := by
  simpa [exOutput, exExpected, exNet] using
    (Examples.mlp_spec_forward_eq (α := ℚ) exampleHiddenLayer exampleOutputLayer exInput)

example :
    dXHand = exDXOpspec := by
  simp [dXHand, exGrad, exDXOpspec, Examples.mlpBackward, Examples.mlpOpspecBackward,
    Examples.mlpOpspec, Spec.OpSpec.compose, Spec.linearOp, Spec.reluOp,
    Spec.liftElementwiseBackward, Spec.liftElementwise, Activation.reluDerivSpec]

end NN.Proofs.Models.Mlp
