/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Neural.Builders
public import NN.API.Runtime
public import NN.Tensor

/-!
# Executing Sequential Models

Execution operations for shape-checked sequential models and the typed graphs obtained by lowering
them.
-/

@[expose] public section

namespace TorchLean

namespace nn

export _root_.Runtime.Autograd.TorchLean.NN.Seq (forward lowerToTypedGraph)

/-- A typed graph whose inputs are model state followed by one model input. -/
abbrev TypedGraphModel (stateShapes : List Shape) (σ τ : Shape) (α : Type) :=
  _root_.Runtime.Autograd.Torch.TypedGraph α (stateShapes ++ [σ]) τ

namespace TypedGraphModel

/-- Evaluate a lowered model with explicit model state and one input tensor. -/
def forward {σ τ : Shape} {α : Type}
    {stateShapes : List Shape}
    (model : TypedGraphModel stateShapes σ τ α)
    (state : TensorPack α stateShapes)
    (x : Tensor α σ) : Tensor α τ :=
  _root_.Runtime.Autograd.Torch.TypedGraph.forward model <|
    TensorPack.append (ss₁ := stateShapes) (ss₂ := [σ]) state (.cons x .nil)

/--
Evaluate a Jacobian-vector product with separate tangents for the model state and input.
-/
def jvp {σ τ : Shape} {α : Type}
    {stateShapes : List Shape}
    (model : TypedGraphModel stateShapes σ τ α)
    (state stateTangent : TensorPack α stateShapes)
    (x inputTangent : Tensor α σ) : Tensor α τ :=
  _root_.Runtime.Autograd.Torch.TypedGraph.jvp model
    (TensorPack.append (ss₁ := stateShapes) (ss₂ := [σ]) state (.cons x .nil))
    (TensorPack.append (ss₁ := stateShapes) (ss₂ := [σ]) stateTangent
      (.cons inputTangent .nil))

/--
Evaluate a vector-Jacobian product, returning the state and input cotangents separately.
-/
def vjpWithSeed {σ τ : Shape} {α : Type} [Add α] [Zero α]
    {stateShapes : List Shape}
    (model : TypedGraphModel stateShapes σ τ α)
    (state : TensorPack α stateShapes)
    (x : Tensor α σ) (outputCotangent : Tensor α τ) :
    TensorPack α stateShapes × Tensor α σ :=
  let cotangents := _root_.Runtime.Autograd.Torch.TypedGraph.vjpWithSeed model
    (TensorPack.append (ss₁ := stateShapes) (ss₂ := [σ]) state (.cons x .nil))
    outputCotangent
  let split := TensorPack.split
    (α := α) (ss₁ := stateShapes) (ss₂ := [σ]) cotangents
  match split.2 with
  | .cons dx .nil => (split.1, dx)

end TypedGraphModel

end nn

end TorchLean
