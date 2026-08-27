/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Neural.Leading

/-!
# Pooling

Arbitrary-rank local and global pooling layer constructors.
-/

@[expose] public section

namespace TorchLean
namespace nn
namespace Internal

/-- Configuration shared by arbitrary-dimensional pooling layers. -/
structure Pool (d : Nat) where
  /-- Window extent along each spatial axis. -/
  kernel : Tensor Nat [d]
  /-- Step along each spatial axis. -/
  stride : Tensor Nat [d] := Spec.fill 1 [d]
  /-- Symmetric padding along each spatial axis. -/
  padding : Tensor Nat [d] := Spec.fill 0 [d]
  /-- Every window extent is positive. -/
  kernelNonzero : ∀ i : Fin d, kernel.getScalar i ≠ 0
  /-- Every stride is positive. -/
  strideNonzero : ∀ i : Fin d, stride.getScalar i ≠ 0

/-- Apply max pooling to the channel and spatial suffix of a tensor. -/
def maxPool (leading : List Nat := []) {d channels : Nat} (spatial : Tensor Nat [d])
    (cfg : Pool d) :
    Sequential
      (leading ++ channels :: spatial.toList)
      (leading ++ channels ::
        (Spec.poolOutSpatialPad spatial cfg.kernel cfg.stride cfg.padding).toList) := by
  simpa only [Spec.Shape.ofList_append] using
    (nn.of <| adaptLeadingShape (Spec.Shape.ofList leading) <|
      _root_.Runtime.Autograd.TorchLean.NN.maxPool
        (Spec.Shape.size (Spec.Shape.ofList leading)) d channels
        cfg.kernel cfg.stride cfg.padding spatial
        (hKernel := cfg.kernelNonzero) (hStride := cfg.strideNonzero))

/-- Apply average pooling to the channel and spatial suffix of a tensor. -/
def avgPool (leading : List Nat := []) {d channels : Nat} (spatial : Tensor Nat [d])
    (cfg : Pool d) :
    Sequential
      (leading ++ channels :: spatial.toList)
      (leading ++ channels ::
        (Spec.poolOutSpatialPad spatial cfg.kernel cfg.stride cfg.padding).toList) := by
  simpa only [Spec.Shape.ofList_append] using
    (nn.of <| adaptLeadingShape (Spec.Shape.ofList leading) <|
      _root_.Runtime.Autograd.TorchLean.NN.avgPool
        (Spec.Shape.size (Spec.Shape.ofList leading)) d channels
        cfg.kernel cfg.stride cfg.padding spatial cfg.kernelNonzero cfg.strideNonzero)

/--
Global average pooling over every spatial axis, preserving the leading axes and channels.
-/
def globalAvgPool (leading : List Nat := []) {d channels : Nat}
    (spatial : Tensor Nat [d]) (spatialNonzero : ∀ i : Fin d, spatial.getScalar i ≠ 0) :
    Sequential
      (leading ++ channels :: spatial.toList)
      (leading ++ [channels]) := by
  let leadingShape := Spec.Shape.ofList leading
  let config : Pool d :=
    { kernel := spatial
      stride := Spec.fill 1 [d]
      padding := Spec.fill 0 [d]
      kernelNonzero := spatialNonzero
      strideNonzero := by intro i; simp }
  let pooledShape := leadingShape.concat (Spec.Shape.ofList
    (channels :: (Spec.poolOutSpatialPad spatial spatial
      (Spec.fill 1 [d]) (Spec.fill 0 [d])).toList))
  let pooled : Sequential
      (leadingShape.concat (Spec.Shape.ofList (channels :: spatial.toList))) pooledShape :=
    nn.of <| adaptLeadingShape leadingShape <|
      _root_.Runtime.Autograd.TorchLean.NN.avgPool
        (Spec.Shape.size leadingShape) d channels config.kernel config.stride config.padding spatial
        config.kernelNonzero config.strideNonzero
  let outputShape := leadingShape.appendDim channels
  let removeSingletons : Layer pooledShape outputShape :=
    { kind := "GlobalAvgPool"
      stateShapes := []
      initState := .nil
      requiresGrad := #[]
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun x =>
            _root_.Runtime.Autograd.Torch.reshape
              (m := m) (α := α) (s₁ := pooledShape) (s₂ := outputShape)
              x (by
                dsimp [pooledShape, outputShape]
                rw [Spec.poolOutSpatialPad_global spatial spatialNonzero]
                simp [Spec.Shape.size_concat, Spec.Shape.ofList,
                  Spec.Shape.size, Spec.Shape.size_appendDim]) }
  simpa [leadingShape, outputShape, Spec.Shape.ofList_append,
    Spec.Shape.appendDim_eq_concat] using
    (seq! pooled, nn.of removeSingletons)
