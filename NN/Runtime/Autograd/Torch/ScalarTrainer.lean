/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core

/-!
# Scalar Trainer Operations

Packed operations and small training loops for `Runtime.Autograd.Torch.ScalarTrainer`.

The trainer keeps differentiable tensors of scalar type `α` separate from non-differentiable data
tensors of type `δ`. The latter may contain bounded token identifiers, class labels, masks, or any
other values consumed by an `Ops.DataRef`; they are not encoded through natural numbers.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec
open Tensor

namespace ScalarTrainer

/-- Evaluate the scalar loss on packed differentiable and non-differentiable inputs. -/
def runLoss {α δ : Type} {paramShapes inputShapes dataInputShapes : List Shape}
    (trainer : ScalarTrainer α δ paramShapes inputShapes dataInputShapes)
    (inputs : _root_.TorchLean.TensorPack α inputShapes)
    (dataInputs : _root_.TorchLean.TensorPack δ dataInputShapes) : IO (Tensor α .scalar) :=
  let withData := Curried.uncurry (α := α) (ss := inputShapes)
    (β := Curried.Fn δ dataInputShapes (IO (Tensor α .scalar))) trainer.loss inputs
  Curried.uncurry (α := δ) (ss := dataInputShapes)
    (β := IO (Tensor α .scalar)) withData dataInputs

/-- Evaluate one loss and its parameter gradients from the same tape. -/
def runLossAndGradState {α δ : Type}
    {paramShapes inputShapes dataInputShapes : List Shape}
    (trainer : ScalarTrainer α δ paramShapes inputShapes dataInputShapes)
    (inputs : _root_.TorchLean.TensorPack α inputShapes)
    (dataInputs : _root_.TorchLean.TensorPack δ dataInputShapes) :
    IO (Tensor α .scalar × _root_.TorchLean.TensorPack α paramShapes) :=
  let withData := Curried.uncurry (α := α) (ss := inputShapes)
    (β := Curried.Fn δ dataInputShapes
      (IO (Tensor α .scalar × _root_.TorchLean.TensorPack α paramShapes)))
    trainer.lossAndGradState inputs
  Curried.uncurry (α := δ) (ss := dataInputShapes)
    (β := IO (Tensor α .scalar × _root_.TorchLean.TensorPack α paramShapes))
    withData dataInputs

/-- Evaluate parameter gradients on packed inputs. -/
def runGradState {α δ : Type} {paramShapes inputShapes dataInputShapes : List Shape}
    (trainer : ScalarTrainer α δ paramShapes inputShapes dataInputShapes)
    (inputs : _root_.TorchLean.TensorPack α inputShapes)
    (dataInputs : _root_.TorchLean.TensorPack δ dataInputShapes) :
    IO (_root_.TorchLean.TensorPack α paramShapes) :=
  let withData := Curried.uncurry (α := α) (ss := inputShapes)
    (β := Curried.Fn δ dataInputShapes
      (IO (_root_.TorchLean.TensorPack α paramShapes))) trainer.gradState inputs
  Curried.uncurry (α := δ) (ss := dataInputShapes)
    (β := IO (_root_.TorchLean.TensorPack α paramShapes)) withData dataInputs

/-- Apply the trainer's SGD update to packed inputs. -/
def runStep {α δ : Type} {paramShapes inputShapes dataInputShapes : List Shape}
    (trainer : ScalarTrainer α δ paramShapes inputShapes dataInputShapes)
    (learningRate : α) (inputs : _root_.TorchLean.TensorPack α inputShapes)
    (dataInputs : _root_.TorchLean.TensorPack δ dataInputShapes) : IO Unit :=
  let withData := Curried.uncurry (α := α) (ss := inputShapes)
    (β := Curried.Fn δ dataInputShapes (IO Unit)) (trainer.step learningRate) inputs
  Curried.uncurry (α := δ) (ss := dataInputShapes) (β := IO Unit) withData dataInputs

/-- Apply the trainer's SGD update and return the loss used for the update. -/
def runStepWithLoss {α δ : Type}
    {paramShapes inputShapes dataInputShapes : List Shape}
    (trainer : ScalarTrainer α δ paramShapes inputShapes dataInputShapes)
    (learningRate : α) (inputs : _root_.TorchLean.TensorPack α inputShapes)
    (dataInputs : _root_.TorchLean.TensorPack δ dataInputShapes) : IO (Tensor α .scalar) :=
  let withData := Curried.uncurry (α := α) (ss := inputShapes)
    (β := Curried.Fn δ dataInputShapes (IO (Tensor α .scalar)))
    (trainer.stepWithLoss learningRate) inputs
  Curried.uncurry (α := δ) (ss := dataInputShapes)
    (β := IO (Tensor α .scalar)) withData dataInputs

end ScalarTrainer

/-- Apply `steps` SGD updates while cycling through samples without auxiliary data tensors. -/
def trainCycleSGD {α : Type} [ToString α] {paramShapes inputShapes : List Shape}
    (trainer : ScalarTrainer α Unit paramShapes inputShapes)
    (learningRate : α) (steps : Nat)
    (samples : List (_root_.TorchLean.TensorPack α inputShapes))
    (logEvery : Nat := 1) : IO Unit := do
  match samples with
  | [] => throw <| IO.userError "trainCycleSGD: empty dataset"
  | first :: _ =>
      for step in [0:steps] do
        let inputs := samples.getD (step % samples.length) first
        let loss ← ScalarTrainer.runLoss trainer inputs .nil
        if logEvery != 0 && step % logEvery = 0 then
          IO.println s!"step {step}: loss={loss.item}"
        ScalarTrainer.runStep trainer learningRate inputs .nil

/-- Evaluate the arithmetic mean loss over samples without auxiliary data tensors. -/
def meanLoss {α : Type} [ToString α] [Add α] [Div α] [Zero α] [Coe Nat α]
    {paramShapes inputShapes : List Shape}
    (trainer : ScalarTrainer α Unit paramShapes inputShapes)
    (samples : List (_root_.TorchLean.TensorPack α inputShapes)) : IO α := do
  if samples.isEmpty then
    throw <| IO.userError "meanLoss: empty dataset"
  let mut total : α := 0
  for inputs in samples do
    let loss ← ScalarTrainer.runLoss trainer inputs .nil
    total := total + loss.item
  pure (total / (samples.length : α))

end Torch
end Autograd
end Runtime
