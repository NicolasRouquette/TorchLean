/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape

/-!
# Linear layer (spec layer)

This file defines a fully‑connected layer and its gradients:

- forward: `y = W x + b`
- backward: ∂L/∂W, ∂L/∂b, ∂L/∂x

Definitions are purely functional and shape‑indexed, suitable for both proofs and reuse by
autograd wrappers in `NN/Spec/Autograd`.
-/

@[expose] public section


namespace Spec

open Tensor

variable {α : Type} [Add α] [Mul α] [Zero α]

/--
Linear layer specification (pure, shape-indexed).

This is the spec-level analogue of PyTorch `torch.nn.Linear` / `torch.nn.functional.linear`:
- `weights` has shape `[outDim, inDim]`,
- `bias` has shape `[outDim]`.
-/
structure LinearSpec (α : Type) (inDim outDim : Nat) where
  /-- Weight matrix with rows indexed by output features. -/
  weights : Tensor α [outDim, inDim]
  /-- Bias vector added to each output feature. -/
  bias    : Tensor α [outDim]

/--
Unbatched forward pass: `y = W x + b`.

PyTorch analogue: `torch.nn.functional.linear`.
-/
def linearSpec {inDim outDim : Nat}
  (m : LinearSpec α inDim outDim)
  (input : Tensor α [inDim]) :
  Tensor α [outDim] :=
  addSpec (matVecMulSpec m.weights input) m.bias

/--
Gradient w.r.t. weights: `∂L/∂W = (∂L/∂y) ⊗ x` (outer product).

This is the standard linear-layer backward formula for `y = W x + b`.
-/
def linearWeightsDerivSpec {inDim outDim : Nat}
  (input : Tensor α [inDim])
  (grad_output : Tensor α [outDim]) :
  Tensor α [outDim, inDim] :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      match grad_output, input with
      | Tensor.dim g_vals, Tensor.dim x_vals =>
        match g_vals i, x_vals j with
        | Tensor.scalar g, Tensor.scalar x => Tensor.scalar (g * x)
    ))

/--
Gradient w.r.t. bias: `∂L/∂b = ∂L/∂y`.

Since `y = W x + b`, the Jacobian of `y` w.r.t. `b` is the identity.
-/
def linearBiasDerivSpec {inDim outDim : Nat}
  (_dW : Tensor α [outDim, inDim])
  (grad_output : Tensor α [outDim])
  (_input : Tensor α [inDim]) :
  Tensor α [outDim] := grad_output

/--
Gradient w.r.t. input: `∂L/∂x = Wᵀ (∂L/∂y)`.

This is the standard "matmul by the transpose" rule for `y = W x + b`.
-/
def linearInputDerivSpec {inDim outDim : Nat}
  (weights : Tensor α [outDim, inDim])
  (grad_output : Tensor α [outDim]) :
  Tensor α [inDim] :=
  vecMatMulSpec grad_output weights

/--
Linear derivatives `(∂L/∂W, ∂L/∂b, ∂L/∂x)` over any nonempty leading shape.

The leading axes are flattened only while accumulating the parameter gradients:
- `d_weights = (grad_outputᵀ) · input`,
- `d_bias = sum(grad_output)` over every leading coordinate,
- `d_input = grad_output · weights`.
-/
def linearDerivSpec [Inhabited α] {leading : Shape} {inDim outDim : Nat}
  (hLeading : 0 < Shape.size leading)
  (weights : Tensor α [outDim, inDim])
  (input : Tensor α (leading.appendDim inDim))
  (gradOutput : Tensor α (leading.appendDim outDim)) :
  (Tensor α [outDim, inDim] ×
   Tensor α [outDim] ×
   Tensor α (leading.appendDim inDim)) :=
  let inputFlat : Tensor α [Shape.size leading, inDim] :=
    reshapeSpec input (by simp [Shape.size_appendDim, Shape.size])
  let gradOutputFlat : Tensor α [Shape.size leading, outDim] :=
    reshapeSpec gradOutput (by simp [Shape.size_appendDim, Shape.size])
  let hSamples : Shape.NonemptyAxis 0 [Shape.size leading, outDim] := by
    obtain ⟨sampleCount, hSampleCount⟩ :=
      Nat.exists_eq_succ_of_ne_zero (Nat.ne_of_gt hLeading)
    rw [hSampleCount]
    exact .zero
  let dWeights := matMulSpec (swapAdjacentAxes gradOutputFlat 0) inputFlat
  let dBias := reduceSum 0 gradOutputFlat hSamples
  let dInputFlat := matMulSpec gradOutputFlat weights
  let dInput := reshapeSpec dInputFlat (by simp [Shape.size_appendDim, Shape.size])
  (dWeights, dBias, dInput)

/--
Complete unbatched backward pass for a linear layer.

Returns `(∂L/∂W, ∂L/∂b, ∂L/∂x)` given the layer params, input `x`, and output gradient `∂L/∂y`.
-/
def linearBackwardSpec {inDim outDim : Nat}
  (layer : LinearSpec α inDim outDim)
  (input : Tensor α [inDim])
  (grad_output : Tensor α [outDim]) :
  (Tensor α [outDim, inDim] ×
   Tensor α [outDim] ×
   Tensor α [inDim]) :=
  let d_weights := linearWeightsDerivSpec input grad_output
  let d_bias := linearBiasDerivSpec d_weights grad_output input
  let d_input := linearInputDerivSpec layer.weights grad_output
  (d_weights, d_bias, d_input)

/--
Accumulate two weight gradients by addition.

This is a small helper used by batching/training code.
-/
def linearGradientAccumulateSpec {inDim outDim : Nat}
  (grad1 : Tensor α [outDim, inDim])
  (grad2 : Tensor α [outDim, inDim]) :
  Tensor α [outDim, inDim] :=
  addSpec grad1 grad2

/-- Scale a weight gradient by a scalar factor (e.g. learning-rate adjustment). -/
def linearGradientScaleSpec {inDim outDim : Nat}
  (grad : Tensor α [outDim, inDim])
  (scale_factor : α) :
  Tensor α [outDim, inDim] :=
  scaleSpec grad scale_factor

end Spec
