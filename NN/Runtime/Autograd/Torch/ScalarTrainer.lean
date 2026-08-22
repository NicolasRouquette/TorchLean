/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core

/-!
# Scalar trainer operations

Adapters and small training loops for `Runtime.Autograd.Torch.ScalarTrainer`. A scalar trainer
stores curried functions indexed by the input shapes. The `*Packed` operations accept the same
inputs as a heterogeneous `TList`, which is more convenient for runtime callers.

Stateful optimizers live under `Runtime.Autograd.TorchLean.Optim`; this module only exposes the SGD
step already carried by `ScalarTrainer`.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec
open Tensor
open Proofs.Autograd.Algebra

namespace ScalarTrainer

/-- Evaluate the curried scalar loss on packed tensor and natural-number inputs. -/
def lossPacked {α : Type} {paramShapes inputShapes natInputShapes : List Shape}
    (trainer : ScalarTrainer α paramShapes inputShapes natInputShapes)
    (inputs : TList α inputShapes) (natInputs : TList Nat natInputShapes) :
    IO (Tensor α Shape.scalar) :=
  let withNat := Curried.uncurry (α := α) (ss := inputShapes)
    (β := Curried.Fn Nat natInputShapes (IO (Tensor α Shape.scalar))) trainer.loss inputs
  Curried.uncurry (α := Nat) (ss := natInputShapes)
    (β := IO (Tensor α Shape.scalar)) withNat natInputs

/-- Evaluate the loss and parameter gradients from one tape traversal. -/
def lossAndGradStatePacked {α : Type} {paramShapes inputShapes natInputShapes : List Shape}
    (trainer : ScalarTrainer α paramShapes inputShapes natInputShapes)
    (inputs : TList α inputShapes) (natInputs : TList Nat natInputShapes) :
    IO (Tensor α Shape.scalar × TList α paramShapes) :=
  let withNat := Curried.uncurry (α := α) (ss := inputShapes)
    (β := Curried.Fn Nat natInputShapes
      (IO (Tensor α Shape.scalar × TList α paramShapes))) trainer.lossAndGradState inputs
  Curried.uncurry (α := Nat) (ss := natInputShapes)
    (β := IO (Tensor α Shape.scalar × TList α paramShapes)) withNat natInputs

/-- Evaluate the parameter gradients on packed inputs. -/
def gradStatePacked {α : Type} {paramShapes inputShapes natInputShapes : List Shape}
    (trainer : ScalarTrainer α paramShapes inputShapes natInputShapes)
    (inputs : TList α inputShapes) (natInputs : TList Nat natInputShapes) :
    IO (TList α paramShapes) :=
  let withNat := Curried.uncurry (α := α) (ss := inputShapes)
    (β := Curried.Fn Nat natInputShapes (IO (TList α paramShapes))) trainer.gradState inputs
  Curried.uncurry (α := Nat) (ss := natInputShapes)
    (β := IO (TList α paramShapes)) withNat natInputs

/-- Apply the trainer's SGD update to packed inputs. -/
def stepPacked {α : Type} {paramShapes inputShapes natInputShapes : List Shape}
    (trainer : ScalarTrainer α paramShapes inputShapes natInputShapes)
    (learningRate : α) (inputs : TList α inputShapes) (natInputs : TList Nat natInputShapes) :
    IO Unit :=
  let withNat := Curried.uncurry (α := α) (ss := inputShapes)
    (β := Curried.Fn Nat natInputShapes (IO Unit)) (trainer.step learningRate) inputs
  Curried.uncurry (α := Nat) (ss := natInputShapes) (β := IO Unit) withNat natInputs

/-- Apply the trainer's SGD update and return the loss used for the update. -/
def stepWithLossPacked {α : Type} {paramShapes inputShapes natInputShapes : List Shape}
    (trainer : ScalarTrainer α paramShapes inputShapes natInputShapes)
    (learningRate : α) (inputs : TList α inputShapes) (natInputs : TList Nat natInputShapes) :
    IO (Tensor α Shape.scalar) :=
  let withNat := Curried.uncurry (α := α) (ss := inputShapes)
    (β := Curried.Fn Nat natInputShapes (IO (Tensor α Shape.scalar)))
    (trainer.stepWithLoss learningRate) inputs
  Curried.uncurry (α := Nat) (ss := natInputShapes)
    (β := IO (Tensor α Shape.scalar)) withNat natInputs

end ScalarTrainer

/-- Apply `steps` SGD updates while cycling through `samples`. -/
def trainCycleSGD
    {α : Type} [ToString α] {paramShapes inputShapes : List Shape}
    (trainer : ScalarTrainer α paramShapes inputShapes)
    (learningRate : α) (steps : Nat) (samples : List (TList α inputShapes))
    (logEvery : Nat := 1) : IO Unit := do
  match samples with
  | [] =>
      throw <| IO.userError "trainCycleSGD: empty dataset"
  | first :: _ =>
      for step in [0:steps] do
        let inputs := samples.getD (step % samples.length) first
        let loss ← ScalarTrainer.lossPacked trainer inputs .nil
        if logEvery != 0 && step % logEvery = 0 then
          IO.println s!"step {step}: loss={loss.item}"
        ScalarTrainer.stepPacked trainer learningRate inputs .nil

/-- Evaluate the arithmetic mean of the scalar losses over `samples`. -/
def meanLoss
    {α : Type} [ToString α] [Add α] [Div α] [Zero α] [Coe Nat α]
    {paramShapes inputShapes : List Shape}
    (trainer : ScalarTrainer α paramShapes inputShapes)
    (samples : List (TList α inputShapes)) : IO α := do
  if samples.isEmpty then
    throw <| IO.userError "meanLoss: empty dataset"
  let mut total : α := 0
  for inputs in samples do
    let loss ← ScalarTrainer.lossPacked trainer inputs .nil
    total := total + loss.item
  pure (total / (samples.length : α))

end Torch
end Autograd
end Runtime
