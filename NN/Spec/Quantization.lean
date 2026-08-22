/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.Quantization
public import NN.Spec.Core.TensorOps

/-!
# Tensor Quantization

This module lifts the scalar affine quantizer from `NN.Floats.Quantization` pointwise over
TorchLean's shape-indexed tensors. The numerical definition remains usable without importing the
tensor library; only this adapter depends on `NN.Spec`.
-/

@[expose] public section

namespace TorchLean.Floats.Quantization

open Spec

namespace AffineQuantizer

/-- Apply the affine quantizer independently at every coordinate of an arbitrary-rank tensor. -/
noncomputable def quantizeTensor (q : AffineQuantizer) (rnd : ℝ → ℤ) {s : Shape}
    (x : Tensor ℝ s) : Tensor ℤ s :=
  mapTensor (q.quantize rnd) x

/-- Reconstruct every code in an arbitrary-rank tensor on the quantizer's real grid. -/
noncomputable def dequantizeTensor (q : AffineQuantizer) {s : Shape}
    (codes : Tensor ℤ s) : Tensor ℝ s :=
  mapTensor q.dequantize codes

/-- Pointwise condition saying that quantization does not clip any coordinate of `x`. -/
def SaturationInactive (q : AffineQuantizer) (rnd : ℝ → ℤ) {s : Shape}
    (x : Tensor ℝ s) : Prop :=
  Tensor.Forall (fun a => q.qmin ≤ q.rawCode rnd a ∧ q.rawCode rnd a ≤ q.qmax) x

/-- Pointwise condition saying that every stored code belongs to the quantizer's code set. -/
def CodesInRange (q : AffineQuantizer) {s : Shape} (codes : Tensor ℤ s) : Prop :=
  Tensor.Forall (fun code => q.qmin ≤ code ∧ code ≤ q.qmax) codes

/-- Every coordinate produced by tensor quantization lies in the declared code interval. -/
theorem quantizeTensor_inRange (q : AffineQuantizer) (rnd : ℝ → ℤ)
    {s : Shape} (x : Tensor ℝ s) :
    q.CodesInRange (q.quantizeTensor rnd x) := by
  apply Tensor.forall_mapTensor (Tensor.forall_true x)
  intro a _
  exact q.quantize_mem rnd a

/-- Pointwise order is preserved by tensor quantization. -/
theorem quantizeTensor_mono (q : AffineQuantizer) (rnd : ℝ → ℤ) [NeuralValidRnd rnd]
    {s : Shape} {x y : Tensor ℝ s}
    (hxy : Tensor.Forall₂ (· ≤ ·) x y) :
    Tensor.Forall₂ (· ≤ ·) (q.quantizeTensor rnd x) (q.quantizeTensor rnd y) := by
  induction s with
  | scalar =>
      cases x with
      | scalar a =>
          cases y with
          | scalar b => exact q.quantize_mono rnd hxy
  | dim n inner ih =>
      cases x with
      | dim xs =>
          cases y with
          | dim ys =>
              intro i
              exact ih (hxy i)

/-- An in-range code tensor survives pointwise dequantization and requantization exactly. -/
theorem quantizeTensor_dequantizeTensor (q : AffineQuantizer) (rnd : ℝ → ℤ)
    [NeuralValidRnd rnd] {s : Shape} {codes : Tensor ℤ s} (hcodes : q.CodesInRange codes) :
    q.quantizeTensor rnd (q.dequantizeTensor codes) = codes := by
  induction s with
  | scalar =>
      cases codes with
      | scalar code =>
          simp only [quantizeTensor, dequantizeTensor, mapTensor]
          rw [q.quantize_dequantize rnd hcodes.1 hcodes.2]
  | dim n inner ih =>
      cases codes with
      | dim values =>
          simp only [quantizeTensor, dequantizeTensor, mapTensor]
          congr 1
          funext i
          exact ih (hcodes i)

/-- If no coordinate clips, every tensor reconstruction error is at most half a step. -/
theorem dequantizeTensor_quantizeTensor_error_le (q : AffineQuantizer) (rnd : ℝ → ℤ)
    [NeuralValidRndToNearest rnd] {s : Shape} {x : Tensor ℝ s}
    (hinactive : q.SaturationInactive rnd x) :
    Tensor.Forall (fun e : ℝ => abs e ≤ q.scale / 2)
      ((q.dequantizeTensor (q.quantizeTensor rnd x)).subSpec x) := by
  induction s with
  | scalar =>
      cases x with
      | scalar a =>
          exact q.dequantize_quantize_error_le rnd a hinactive.1 hinactive.2
  | dim n inner ih =>
      cases x with
      | dim values =>
          intro i
          exact ih (hinactive i)

end AffineQuantizer
end TorchLean.Floats.Quantization
