/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Neural.Builders
public import NN.API.Runtime

/-!
# Models over Leading Dimensions

This module lifts layers and sequential models over any number of leading tensor dimensions. A
batch is the common case `leading = [batch]`; shapes such as `[batch, time]` use the same machinery.

There are two distinct operations. `adaptLeadingShape` flattens the leading dimensions for a layer
that already accepts one outer dimension. `mapEach` applies a model separately at every leading
index. Keeping that distinction explicit matters for stateful layers, whose buffer updates may
depend on whether the leading positions are processed together or one at a time.
-/

@[expose] public section

namespace TorchLean
namespace nn
namespace Internal

/--
Reshape arbitrary leading dimensions into the single outer dimension expected by `layer`.

For an input of shape `leading.concat σ`, the layer receives shape
`[leading.size].concat σ`; its output is then reshaped from `[leading.size].concat τ` to
`leading.concat τ`. The adapter reuses the layer's parameters and buffer-update function.
-/
def adaptLeadingShape (leading : Spec.Shape) {σ τ : Spec.Shape}
    (layer : Layer (σ.prependDim leading.size) (τ.prependDim leading.size)) :
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
          (Ref := fun shape => _root_.TorchLean.Runtime.ValueRef (m := m) (α := α) shape)
          (ss := layer.stateShapes ++ [leading.concat σ])
          (β := m (_root_.TorchLean.Runtime.ValueRef (m := m) (α := α) (leading.concat τ)))
          (fun args => do
            let (ps, x) :=
              _root_.Runtime.Autograd.Torch.RefList.splitLast
                (Ref := fun shape =>
                  _root_.TorchLean.Runtime.ValueRef (m := m) (α := α) shape)
                (ss := layer.stateShapes) (τ := leading.concat σ) args
            let xBatch ←
              _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
                (s₁ := leading.concat σ) (s₂ := σ.prependDim leading.size)
                x (by simp [Spec.Shape.size_concat, Spec.Shape.size])
            let yBatch ←
              _root_.Runtime.Autograd.Torch.CurriedRef.uncurry
                (Ref := fun shape =>
                  _root_.TorchLean.Runtime.ValueRef (m := m) (α := α) shape)
                (ss := layer.stateShapes ++ [σ.prependDim leading.size])
                (β := m (_root_.TorchLean.Runtime.ValueRef (m := m) (α := α)
                  (τ.prependDim leading.size)))
                (layer.forward mode (α := α) (m := m))
                (_root_.Runtime.Autograd.Torch.RefList.append ps (.cons xBatch .nil))
            _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
              (s₁ := τ.prependDim leading.size) (s₂ := leading.concat τ)
              yBatch (by simp [Spec.Shape.size_concat, Spec.Shape.size])) }

/-- Apply `layer` separately at every position of one new leading dimension. -/
private def mapLayerOverAxis (n : Nat) {σ τ : Spec.Shape} (layer : Layer σ τ) :
    Layer (σ.prependDim n) (τ.prependDim n) :=
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
          (Ref := fun shape => _root_.TorchLean.Runtime.ValueRef (m := m) (α := α) shape)
          (ss := layer.stateShapes ++ [σ.prependDim n])
          (β := m (_root_.TorchLean.Runtime.ValueRef (m := m) (α := α)
            (τ.prependDim n)))
          (fun args => do
            let (ps, xBatch) :=
              _root_.Runtime.Autograd.Torch.RefList.splitLast
                (Ref := fun shape =>
                  _root_.TorchLean.Runtime.ValueRef (m := m) (α := α) shape)
                (ss := layer.stateShapes) (τ := σ.prependDim n) args
            _root_.Runtime.Autograd.Torch.mapOuterAxis (m := m) (α := α)
              (fun x => layer.forwardRef (α := α) (m := m) mode ps x)
              xBatch) }

/-- Apply every layer of `model` over one new leading dimension. -/
private def mapModelOverAxis (n : Nat) {σ τ : Spec.Shape} :
    Sequential σ τ → Sequential (σ.prependDim n) (τ.prependDim n)
  | .id shape => .id (shape.prependDim n)
  | .cons layer rest => .cons (mapLayerOverAxis n layer) (mapModelOverAxis n rest)

/--
Apply a sequential model separately at every index of `leading`.

All positions use the same model parameters. Buffer updates are evaluated in lexicographic order
over the leading indices.
-/
opaque mapEach (leading : Spec.Shape) {σ τ : Spec.Shape} :
    Sequential σ τ → Sequential (leading.concat σ) (leading.concat τ) :=
  fun model =>
    match leading with
    | .scalar => model
    | .dim n rest => mapModelOverAxis n (mapEach rest model)

end Internal
end nn
end TorchLean
