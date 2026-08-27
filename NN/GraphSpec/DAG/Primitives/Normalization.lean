/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.DAG.Primitives.Shape
public import NN.Spec.Layers.Normalization
public import NN.Spec.Core.Tensor.Numerics

/-!
# DAG Normalization Primitives

RMS normalization and regularized L2 normalization with explicit shape and positivity contracts.
-/

@[expose] public section

namespace NN
namespace GraphSpec
namespace DAG

open _root_.Spec
open Spec.Tensor
open _root_.TorchLean.Tensor

namespace PrimOp

namespace Internal

def rmsNormVectorSemantics {α : Type} [Context α] {width : Nat}
    (hWidth : 0 < width) (input gamma : _root_.Spec.Tensor α [width]) :
    _root_.Spec.Tensor α [width] :=
  _root_.Spec.get
    (_root_.Spec.rmsNorm (α := α) (.dim fun _ => input) gamma
      (Nat.zero_lt_succ 0) hWidth)
    ⟨0, Nat.zero_lt_succ 0⟩

end Internal

/-- Apply RMS normalization with a shared final-axis scale over an arbitrary leading shape. -/
def rmsNormSemantics {α : Type} [Context α] (leading : Shape) {width : Nat}
    (hWidth : 0 < width) (gamma : _root_.Spec.Tensor α [width]) :
    _root_.Spec.Tensor α (leading.appendDim width) →
      _root_.Spec.Tensor α (leading.appendDim width)
  | input =>
      match leading, input with
      | .scalar, input => Internal.rmsNormVectorSemantics hWidth input gamma
      | .dim _ rest, .dim values =>
          .dim fun i => rmsNormSemantics rest hWidth gamma (values i)

/-- Apply RMS normalization with a pointwise scale over an arbitrary leading shape. -/
def rmsNormElementwiseSemantics {α : Type} [Context α] (leading : Shape) {width : Nat}
    (hWidth : 0 < width) :
    _root_.Spec.Tensor α (leading.appendDim width) →
      _root_.Spec.Tensor α (leading.appendDim width) →
      _root_.Spec.Tensor α (leading.appendDim width)
  | input, gamma =>
      match leading, input, gamma with
      | .scalar, input, gamma => Internal.rmsNormVectorSemantics hWidth input gamma
      | .dim _ rest, .dim inputs, .dim gammas =>
          .dim fun i => rmsNormElementwiseSemantics rest hWidth (inputs i) (gammas i)

@[simp] theorem rmsNormElementwiseSemantics_dim {α : Type} [Context α]
    {count width : Nat} {rest : Shape} (hWidth : 0 < width)
    (inputs gammas : Fin count → _root_.Spec.Tensor α (rest.appendDim width)) :
    rmsNormElementwiseSemantics (.dim count rest) hWidth (.dim inputs) (.dim gammas) =
      .dim (fun index => rmsNormElementwiseSemantics rest hWidth
        (inputs index) (gammas index)) := by
  rfl

/-- Root-mean-square normalization along the final axis of an arbitrary tensor. -/
def rmsNorm (leading : Shape) (width : Nat) (hWidth : 0 < width) :
    PrimOp [leading.appendDim width, [width]] (leading.appendDim width) :=
  { name := s!"rmsNorm({width})"
    specFwd := fun {_} _ xs =>
      match xs with
      | .cons input (.cons gamma .nil) => rmsNormSemantics leading hWidth gamma input
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input gamma =>
        (Runtime.Autograd.TorchLean.Norm.rmsNorm (m := m) (α := α)
          (leading := leading) (width := width) hWidth input gamma :
          m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α)
            (leading.appendDim width))) }

/-- Pure evaluation of final-axis RMS normalization with a shared scale. -/
@[simp] theorem rmsNorm_specFwd {leading : Shape} {width : Nat}
    (hWidth : 0 < width) {α : Type} [Context α]
    (input : _root_.Spec.Tensor α (leading.appendDim width))
    (gamma : _root_.Spec.Tensor α [width]) :
    (rmsNorm leading width hWidth).specFwd (.cons input (.cons gamma .nil)) =
      rmsNormSemantics leading hWidth gamma input := by
  rfl

/-- RMS normalization with a separate final-axis scale at every leading coordinate. -/
def rmsNormElementwise (leading : Shape) (width : Nat) (hWidth : 0 < width) :
    PrimOp [leading.appendDim width, leading.appendDim width] (leading.appendDim width) :=
  { name := s!"rmsNormElementwise({width})"
    specFwd := fun {_} _ xs =>
      match xs with
      | .cons input (.cons gamma .nil) =>
          rmsNormElementwiseSemantics leading hWidth input gamma
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input gamma =>
        (do
          let ones ← Runtime.Autograd.TorchLean.const (m := m) (α := α)
            (_root_.Spec.fill 1 ([width] : Shape))
          let normalized ← Runtime.Autograd.TorchLean.Norm.rmsNorm (m := m) (α := α)
            (leading := leading) (width := width) hWidth input ones
          Runtime.Autograd.TorchLean.mul (m := m) (α := α)
            (s := leading.appendDim width) normalized gamma :
          m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α)
            (leading.appendDim width))) }

/-- Pure evaluation of final-axis RMS normalization with pointwise scale. -/
@[simp] theorem rmsNormElementwise_specFwd {leading : Shape} {width : Nat}
    (hWidth : 0 < width) {α : Type} [Context α]
    (input gamma : _root_.Spec.Tensor α (leading.appendDim width)) :
    (rmsNormElementwise leading width hWidth).specFwd (.cons input (.cons gamma .nil)) =
      rmsNormElementwiseSemantics leading hWidth input gamma := by
  rfl

/-- Apply regularized L2 normalization over the final axis of an arbitrary tensor. -/
def l2NormalizeSemantics {α : Type} [Context α] (leading : Shape) {width : Nat}
    (epsilon : α) : _root_.Spec.Tensor α (leading.appendDim width) →
      _root_.Spec.Tensor α (leading.appendDim width) :=
  match leading with
  | .scalar => fun input => _root_.Spec.normalizeL2RegularizedSpec input epsilon
  | .dim _ rest => fun input =>
      match input with
      | .dim values => .dim fun i => l2NormalizeSemantics rest epsilon (values i)

@[simp] theorem l2NormalizeSemantics_scalar {α : Type} [Context α]
    {width : Nat} (epsilon : α) (values : _root_.Spec.Tensor α [width]) :
    l2NormalizeSemantics .scalar epsilon values =
      _root_.Spec.normalizeL2RegularizedSpec values epsilon := by
  rfl

@[simp] theorem l2NormalizeSemantics_dim {α : Type} [Context α]
    {count width : Nat} {rest : Shape} (epsilon : α)
    (values : Fin count → _root_.Spec.Tensor α (rest.appendDim width)) :
    l2NormalizeSemantics (.dim count rest) epsilon (.dim values) =
      .dim (fun index => l2NormalizeSemantics rest epsilon (values index)) := by
  rfl

/-- Normalize the final axis by `sqrt(sum (x * x) + epsilon)`.

The stabilizer is a scalar graph input rather than a declaration parameter. This keeps the exact
numerical convention visible in captured graphs and permits architectures to select their own
positive epsilon without adding another primitive.
-/
def l2Normalize (leading : Shape) (width : Nat) (hWidth : 0 < width) :
    PrimOp [leading.appendDim width, .scalar] (leading.appendDim width) :=
  { name := s!"l2Normalize({width})"
    specFwd := fun {_α} _ xs =>
      match xs with
      | .cons input (.cons (.scalar epsilon) .nil) =>
          l2NormalizeSemantics leading epsilon input
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input epsilon =>
        (Runtime.Autograd.TorchLean.Norm.l2Normalize (m := m) (α := α)
          (leading := leading) (width := width) hWidth input epsilon :
          m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α)
            (leading.appendDim width))) }

/-- Pure evaluation of final-axis regularized L2 normalization. -/
@[simp] theorem l2Normalize_specFwd {leading : Shape} {width : Nat}
    (hWidth : 0 < width) {α : Type} [Context α]
    (input : _root_.Spec.Tensor α (leading.appendDim width)) (epsilon : α) :
    (l2Normalize leading width hWidth).specFwd
        (.cons input (.cons (.scalar epsilon) .nil)) =
      l2NormalizeSemantics leading epsilon input := by
  rfl


end PrimOp

end DAG
end GraphSpec
end NN
