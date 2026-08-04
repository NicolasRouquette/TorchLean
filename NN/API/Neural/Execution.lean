/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Neural.Builders
public import NN.API.Runtime
public import NN.API.Tensor
public import NN.API.TensorPack

/-!
# Model Execution

Forward, prediction, compiled inference, and scalar-module operations for checked sequential models.
-/

@[expose] public section

namespace TorchLean

namespace nn

export _root_.Runtime.Autograd.TorchLean.NN.Seq (forward predict)

/--
A compiled sequential model.

The object returned by `model.compile` stores the compiled artifact and carries the
parameter-shape ABI in its type, so callers can run `compiled.forward params x` without passing the
source model again.
-/
structure Compiled (paramShapes : List Shape) (σ τ : Shape) (α : Type) where
  artifact : _root_.Runtime.Autograd.Torch.CompiledGraph α (paramShapes ++ [σ]) τ

namespace Compiled

/-- Run a compiled model forward with explicit parameter tensors. -/
def forward {σ τ : Shape} {α : Type}
    [_root_.Context α] [DecidableEq Shape]
    {paramShapes : List Shape}
    (compiled : Compiled paramShapes σ τ α)
    (params : TensorPack α paramShapes)
    (x : Tensor.T α σ) : Tensor.T α τ :=
  let args : _root_.Runtime.Autograd.Torch.TList α (paramShapes ++ [σ]) :=
    _root_.Proofs.Autograd.Algebra.TList.append (α := α)
      (ss₁ := paramShapes) (ss₂ := [σ]) params (.cons x .nil)
  _root_.Runtime.Autograd.Torch.CompiledGraph.forward compiled.artifact args

end Compiled

end nn

end TorchLean

namespace Runtime
namespace Autograd
namespace TorchLean
namespace NN
namespace Seq

/-- Compile a sequential model into a reusable callable object. -/
def compile {σ τ : _root_.Spec.Shape}
    (model : _root_.Runtime.Autograd.TorchLean.NN.Seq σ τ)
    {α : Type} [_root_.Context α] [DecidableEq _root_.Spec.Shape] :
    IO (_root_.TorchLean.nn.Compiled
      (_root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes model) σ τ α) := do
  let artifact ← _root_.Runtime.Autograd.TorchLean.NN.Seq.compileForward (α := α) model
  pure { artifact := artifact }

/-- Compile a sequential model under an explicit layer mode. -/
def compileWithMode {σ τ : _root_.Spec.Shape}
    (model : _root_.Runtime.Autograd.TorchLean.NN.Seq σ τ)
    (mode : _root_.Runtime.Autograd.TorchLean.NN.Mode)
    {α : Type} [_root_.Context α] [DecidableEq _root_.Spec.Shape] :
    IO (_root_.TorchLean.nn.Compiled
      (_root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes model) σ τ α) := do
  let artifact ← _root_.Runtime.Autograd.TorchLean.NN.Seq.compileForwardWithMode (α := α) mode model
  pure { artifact := artifact }

end Seq
end NN
end TorchLean
end Autograd
end Runtime
