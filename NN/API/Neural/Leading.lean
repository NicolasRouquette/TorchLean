/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Neural.Builders

/-!
# Leading-Shape Model Operations

Adapters for applying a layer or sequential model independently over arbitrary leading dimensions.
A conventional batch is the special case `leading = .dim batch .scalar`; the same definitions also
support several leading axes without introducing another tensor or model type.
-/

@[expose] public section

namespace TorchLean
namespace nn
namespace Internal

/--
Expose a runtime layer over an arbitrary leading shape.

The runtime layer sees one flattened leading dimension. This adapter changes only the input and
output views; parameters, buffer updates, and the underlying forward program are preserved.
-/
def adaptLeadingShape (leading : Spec.Shape) {σ τ : Spec.Shape}
    (layer : Layer (.dim (Spec.Shape.size leading) σ) (.dim (Spec.Shape.size leading) τ)) :
    Layer (leading.concat σ) (leading.concat τ) :=
  { kind := layer.kind
    stateShapes := layer.stateShapes
    initState := layer.initState
    runtimeInit := layer.runtimeInit
    requiresGrad := layer.requiresGrad
    updateBuffers := layer.updateBuffers.map fun update mode {α} _ _ ps x =>
      update mode ps <| Spec.Tensor.reshapeSpec x (by
        simp [Spec.Shape.size_concat, Spec.Shape.size])
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        _root_.Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := fun shape => _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) shape)
          (ss := layer.stateShapes ++ [leading.concat σ])
          (β := m (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) (leading.concat τ)))
          (fun args => do
            let (ps, x) :=
              _root_.Runtime.Autograd.Torch.RefList.splitLast
                (Ref := fun shape =>
                  _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) shape)
                (ss := layer.stateShapes) (τ := leading.concat σ) args
            let xBatch ←
              _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
                (s₁ := leading.concat σ) (s₂ := .dim (Spec.Shape.size leading) σ)
                x (by simp [Spec.Shape.size_concat, Spec.Shape.size])
            let yBatch ←
              _root_.Runtime.Autograd.Torch.CurriedRef.uncurry
                (Ref := fun shape =>
                  _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) shape)
                (ss := layer.stateShapes ++ [.dim (Spec.Shape.size leading) σ])
                (β := m (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α)
                  (.dim (Spec.Shape.size leading) τ)))
                (layer.forward mode (α := α) (m := m))
                (_root_.Runtime.Autograd.Torch.RefList.append ps (.cons xBatch .nil))
            _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
              (s₁ := .dim (Spec.Shape.size leading) τ) (s₂ := leading.concat τ)
              yBatch (by simp [Spec.Shape.size_concat, Spec.Shape.size])) }

namespace Implementation

/-- Apply one layer independently at every position of a leading axis. -/
def mapLayerOverAxis (n : Nat) {σ τ : Spec.Shape} (layer : Layer σ τ) :
    Layer (.dim n σ) (.dim n τ) :=
  { kind := layer.kind
    stateShapes := layer.stateShapes
    initState := layer.initState
    runtimeInit := layer.runtimeInit
    requiresGrad := layer.requiresGrad
    updateBuffers := layer.updateBuffers.map fun update mode {_α} _ _ ps x =>
      match x with
      | Spec.Tensor.dim rows =>
          (List.finRange n).foldlM (init := ps) fun state i => update mode state (rows i)
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        _root_.Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := fun shape => _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) shape)
          (ss := layer.stateShapes ++ [.dim n σ])
          (β := m (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) (.dim n τ)))
          (fun args => do
            let (ps, xBatch) :=
              _root_.Runtime.Autograd.Torch.RefList.splitLast
                (Ref := fun shape =>
                  _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) shape)
                (ss := layer.stateShapes) (τ := .dim n σ) args
            _root_.Runtime.Autograd.Torch.mapLeadingAxis (m := m) (α := α)
              (fun x => layer.forwardRef (α := α) (m := m) mode ps x)
              xBatch) }

/-- Apply a sequential model independently at every position of a leading axis. -/
def mapModelOverAxis (n : Nat) {σ τ : Spec.Shape} :
    Sequential σ τ → Sequential (.dim n σ) (.dim n τ)
  | .id shape => .id (.dim n shape)
  | .cons layer rest => .cons (mapLayerOverAxis n layer) (mapModelOverAxis n rest)

end Implementation

/-- Apply a model independently over arbitrary leading dimensions. -/
def mapLeading (leading : Spec.Shape) {σ τ : Spec.Shape} :
    Sequential σ τ → Sequential (leading.concat σ) (leading.concat τ)
  | model =>
      match leading with
      | .scalar => model
      | .dim n rest => Implementation.mapModelOverAxis n (mapLeading rest model)

end Internal
end nn
end TorchLean
