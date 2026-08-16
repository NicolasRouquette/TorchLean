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

Forward, prediction, typed-graph inference, and scalar-module operations for checked sequential
models.
-/

@[expose] public section

namespace TorchLean

namespace nn

export _root_.Runtime.Autograd.TorchLean.NN.Seq (forward)

/--
A reusable typed graph for a model with explicit state tensors and one input tensor.

This is the runtime `TypedGraph` itself, with the leaf context factored as `stateShapes ++ [σ]` so
the model-facing `forward` operation can pack model state and input for the caller. No second graph
representation or wrapper is introduced.
-/
abbrev TypedGraphModel (stateShapes : List Shape) (σ τ : Shape) (α : Type) :=
  _root_.Runtime.Autograd.Torch.TypedGraph α (stateShapes ++ [σ]) τ

namespace TypedGraphModel

/-- Evaluate a lowered model with explicit model state and one input tensor. -/
def forward {σ τ : Shape} {α : Type}
    {stateShapes : List Shape}
    (model : TypedGraphModel stateShapes σ τ α)
    (state : TensorPack α stateShapes)
    (x : Tensor α σ) : Tensor α τ :=
  let args : _root_.Runtime.Autograd.Torch.TList α (stateShapes ++ [σ]) :=
    _root_.Proofs.Autograd.Algebra.TList.append (α := α)
      (ss₁ := stateShapes) (ss₂ := [σ]) state (.cons x .nil)
  _root_.Runtime.Autograd.Torch.TypedGraph.forward model args

/--
Evaluate a Jacobian-vector product for simultaneous state and input tangents.

The state and input contexts remain separate at the API boundary and are packed only for the
underlying typed graph call.
-/
def jvp {σ τ : Shape} {α : Type}
    {stateShapes : List Shape}
    (model : TypedGraphModel stateShapes σ τ α)
    (state dState : TensorPack α stateShapes)
    (x dx : Tensor α σ) : Tensor α τ :=
  let args : _root_.Runtime.Autograd.Torch.TList α (stateShapes ++ [σ]) :=
    _root_.Proofs.Autograd.Algebra.TList.append (α := α)
      (ss₁ := stateShapes) (ss₂ := [σ]) state (.cons x .nil)
  let dArgs : _root_.Runtime.Autograd.Torch.TList α (stateShapes ++ [σ]) :=
    _root_.Proofs.Autograd.Algebra.TList.append (α := α)
      (ss₁ := stateShapes) (ss₂ := [σ]) dState (.cons dx .nil)
  _root_.Runtime.Autograd.Torch.TypedGraph.jvp model args dArgs

/--
Evaluate a vector-Jacobian product and return state and input cotangents separately.

`seedOut` is the cotangent for the model output. The result follows the model-facing split rather
than exposing the runtime graph's concatenated leaf context.
-/
def vjpWithSeed {σ τ : Shape} {α : Type} [Add α] [Zero α]
    {stateShapes : List Shape}
    (model : TypedGraphModel stateShapes σ τ α)
    (state : TensorPack α stateShapes)
    (x : Tensor α σ) (seedOut : Tensor α τ) :
    TensorPack α stateShapes × Tensor α σ :=
  let args : _root_.Runtime.Autograd.Torch.TList α (stateShapes ++ [σ]) :=
    _root_.Proofs.Autograd.Algebra.TList.append (α := α)
      (ss₁ := stateShapes) (ss₂ := [σ]) state (.cons x .nil)
  let grads := _root_.Runtime.Autograd.Torch.TypedGraph.vjpWithSeed model args seedOut
  let split := _root_.Proofs.Autograd.Algebra.TList.splitAppend
    (α := α) (ss₁ := stateShapes) (ss₂ := [σ]) grads
  match split.2 with
  | .cons dx .nil => (split.1, dx)

end TypedGraphModel

/--
Lower a sequential model into a reusable typed graph.

The optional mode fixes training-sensitive layer behavior, such as dropout and batch normalization,
when the graph is built. The default is evaluation mode.
-/
def lowerToTypedGraph {σ τ : _root_.Spec.Shape}
    (model : _root_.Runtime.Autograd.TorchLean.NN.Seq σ τ)
    (mode : _root_.Runtime.Autograd.TorchLean.NN.Mode := .eval)
    {α : Type} [_root_.Context α] [DecidableEq _root_.Spec.Shape] :
    IO (_root_.TorchLean.nn.TypedGraphModel
      (_root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes model) σ τ α) :=
  _root_.Runtime.Autograd.TorchLean.NN.Seq.lowerToTypedGraph
    (α := α) model (mode := mode)

end nn

end TorchLean
