/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Pooling.ND

@[expose] public section

namespace Spec

open Tensor
open Spec (getValueAtPosition)

variable {α : Type} [Context α]

/-!
# Two-Dimensional Pooling Adapters

The fixed-window operations in this file are two-dimensional views of the rank-polymorphic
definitions in `NN.Spec.Layers.Pooling.ND`. They retain the conventional `maxPool2d` and
`avgPool2d` names used by model APIs and PyTorch interchange, but do not define a second pooling
semantics.

Adaptive pooling remains genuinely two-dimensional here because its variable-size binning is a
different operation from fixed-kernel N-D pooling.
-/

/-- Witness for a nonempty two-dimensional max-pooling kernel and nonzero uniform stride. -/
structure MaxPool2DSpec (kH kW stride : Nat) (_hH : kH ≠ 0) (_hW : kW ≠ 0)
    (_hStride : stride ≠ 0) where

/-- Witness for a nonempty two-dimensional average-pooling kernel and nonzero uniform stride. -/
structure AvgPool2DSpec (kH kW stride : Nat) (_hH : kH ≠ 0) (_hW : kW ≠ 0)
    (_hStride : stride ≠ 0) where

/-- Output shape for unpadded single-channel two-dimensional pooling. -/
def pool2dOutShape (inH inW kH kW stride : Nat) : Shape :=
  .dim (poolOutDim inH kH stride 0)
    (.dim (poolOutDim inW kW stride 0) .scalar)

/-- Output shape for unpadded channel-first two-dimensional pooling. -/
def pool2dMultiOutShape (inC inH inW kH kW stride : Nat) : Shape :=
  .dim inC (pool2dOutShape inH inW kH kW stride)

/-- Output shape for single-channel two-dimensional pooling with symmetric padding. -/
def pool2dOutShapePad (inH inW kH kW stride padding : Nat) : Shape :=
  .dim (poolOutDim inH kH stride padding)
    (.dim (poolOutDim inW kW stride padding) .scalar)

/-- Output shape for channel-first two-dimensional pooling with symmetric padding. -/
def pool2dMultiOutShapePad (inC inH inW kH kW stride padding : Nat) : Shape :=
  .dim inC (pool2dOutShapePad inH inW kH kW stride padding)

namespace Private

theorem pool2d_kernel_ne {kH kW : Nat} (hH : kH ≠ 0) (hW : kW ≠ 0) :
    ∀ i : Fin 2, (#v[kH, kW]).get i ≠ 0 := by
  intro i
  fin_cases i <;> simp [Vector.get, hH, hW]

theorem pool2d_stride_ne {stride : Nat} (hStride : stride ≠ 0) :
    ∀ i : Fin 2, (#v[stride, stride]).get i ≠ 0 := by
  intro i
  fin_cases i <;> simp [Vector.get, hStride]

def maxPool2DConfig {kH kW stride : Nat} (padding : Nat)
    (hH : kH ≠ 0) (hW : kW ≠ 0) (hStride : stride ≠ 0) :
    MaxPoolSpec 2 (#v[kH, kW]) (#v[stride, stride]) (#v[padding, padding])
      (pool2d_kernel_ne hH hW) (pool2d_stride_ne hStride) := {}

def avgPool2DConfig {kH kW stride : Nat} (padding : Nat)
    (hH : kH ≠ 0) (hW : kW ≠ 0) (hStride : stride ≠ 0) :
    AvgPoolSpec 2 (#v[kH, kW]) (#v[stride, stride]) (#v[padding, padding])
      (pool2d_kernel_ne hH hW) (pool2d_stride_ne hStride) := {}

theorem pool2d_spatial_out_shape_eq (inH inW kH kW stride padding : Nat) :
    Shape.ofList
        (poolOutSpatialPad (#v[inH, inW]) (#v[kH, kW]) (#v[stride, stride])
          (#v[padding, padding])).toList =
      pool2dOutShapePad inH inW kH kW stride padding := by
  simp [poolOutSpatialPad, poolOutDim, pool2dOutShapePad, Shape.slidingWindowOutDim, Vector.get,
    Vector.toList, Shape.ofList]

theorem pool2d_multi_out_shape_eq (inC inH inW kH kW stride padding : Nat) :
    Shape.ofList
        (inC ::
          (poolOutSpatialPad (#v[inH, inW]) (#v[kH, kW]) (#v[stride, stride])
            (#v[padding, padding])).toList) =
      pool2dMultiOutShapePad inC inH inW kH kW stride padding := by
  simp [poolOutSpatialPad, poolOutDim, pool2dMultiOutShapePad, pool2dOutShapePad,
    Shape.slidingWindowOutDim, Vector.get, Vector.toList, Shape.ofList]

end Private

/-! ## Fixed-window adapters -/

/-- Single-channel two-dimensional max pooling, specialized from `maxPoolSpatialSpec`. -/
def maxPool2dSpec {kH kW inH inW stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
    {hStride : stride ≠ 0} (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
    (input : Tensor α (.dim inH (.dim inW .scalar))) :
    Tensor α (pool2dOutShape inH inW kH kW stride) := by
  let _ := layer
  let config := Private.maxPool2DConfig 0 h1 h2 hStride
  let output := maxPoolSpatialSpec (α := α) (inSpatial := #v[inH, inW])
    (kernel := #v[kH, kW]) (stride := #v[stride, stride]) (padding := #v[0, 0]) config input
  exact tensorCast _ (Private.pool2d_spatial_out_shape_eq inH inW kH kW stride 0) output

/-- Channel-first two-dimensional max pooling, specialized from `maxPoolSpec`. -/
def maxPool2dMultiSpec {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
    {hStride : stride ≠ 0} (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
    (input : Tensor α (.dim inC (.dim inH (.dim inW .scalar)))) :
    Tensor α (pool2dMultiOutShape inC inH inW kH kW stride) := by
  let _ := layer
  let config := Private.maxPool2DConfig 0 h1 h2 hStride
  let output := maxPoolSpec (α := α) (C := inC) (inSpatial := #v[inH, inW])
    (kernel := #v[kH, kW]) (stride := #v[stride, stride]) (padding := #v[0, 0]) config input
  exact tensorCast _ (Private.pool2d_multi_out_shape_eq inC inH inW kH kW stride 0) output

/-- Selected-branch JVP for single-channel two-dimensional hard max pooling. -/
def maxPool2dLinearizationSpec {kH kW inH inW stride : Nat} {h1 : kH ≠ 0}
    {h2 : kW ≠ 0} {hStride : stride ≠ 0}
    (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
    (input tangent : Tensor α (.dim inH (.dim inW .scalar))) :
    Tensor α (pool2dOutShape inH inW kH kW stride) := by
  let _ := layer
  let config := Private.maxPool2DConfig 0 h1 h2 hStride
  let output := maxPoolSpatialLinearizationSpec (α := α) (inSpatial := #v[inH, inW])
    (kernel := #v[kH, kW]) (stride := #v[stride, stride]) (padding := #v[0, 0])
    config input tangent
  exact tensorCast _ (Private.pool2d_spatial_out_shape_eq inH inW kH kW stride 0) output

/-- Selected-branch JVP for channel-first two-dimensional hard max pooling. -/
def maxPool2dMultiLinearizationSpec {kH kW inH inW inC stride : Nat}
    {h1 : kH ≠ 0} {h2 : kW ≠ 0} {hStride : stride ≠ 0}
    (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
    (input tangent : Tensor α (.dim inC (.dim inH (.dim inW .scalar)))) :
    Tensor α (pool2dMultiOutShape inC inH inW kH kW stride) := by
  let _ := layer
  let config := Private.maxPool2DConfig 0 h1 h2 hStride
  let output := maxPoolLinearizationSpec (α := α) (C := inC) (inSpatial := #v[inH, inW])
    (kernel := #v[kH, kW]) (stride := #v[stride, stride]) (padding := #v[0, 0])
    config input tangent
  exact tensorCast _ (Private.pool2d_multi_out_shape_eq inC inH inW kH kW stride 0) output

/-- Single-channel two-dimensional average pooling, specialized from `avgPoolSpatialSpec`. -/
def avgPool2dSpec {kH kW inH inW stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
    {hStride : stride ≠ 0} (layer : AvgPool2DSpec kH kW stride h1 h2 hStride)
    (input : Tensor α (.dim inH (.dim inW .scalar))) :
    Tensor α (pool2dOutShape inH inW kH kW stride) := by
  let _ := layer
  let config := Private.avgPool2DConfig 0 h1 h2 hStride
  let output := avgPoolSpatialSpec (α := α) (inSpatial := #v[inH, inW])
    (kernel := #v[kH, kW]) (stride := #v[stride, stride]) (padding := #v[0, 0]) config input
  exact tensorCast _ (Private.pool2d_spatial_out_shape_eq inH inW kH kW stride 0) output

/-- Channel-first two-dimensional average pooling, specialized from `avgPoolSpec`. -/
def avgPool2dMultiSpec {kH kW inH inW inC stride : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
    {hStride : stride ≠ 0} (layer : AvgPool2DSpec kH kW stride h1 h2 hStride)
    (input : Tensor α (.dim inC (.dim inH (.dim inW .scalar)))) :
    Tensor α (pool2dMultiOutShape inC inH inW kH kW stride) := by
  let _ := layer
  let config := Private.avgPool2DConfig 0 h1 h2 hStride
  let output := avgPoolSpec (α := α) (C := inC) (inSpatial := #v[inH, inW])
    (kernel := #v[kH, kW]) (stride := #v[stride, stride]) (padding := #v[0, 0]) config input
  exact tensorCast _ (Private.pool2d_multi_out_shape_eq inC inH inW kH kW stride 0) output

/-- VJP for single-channel two-dimensional max pooling. -/
def maxPool2dBackwardSpec {kH kW inH inW stride : Nat} {h1 : kH ≠ 0}
    {h2 : kW ≠ 0} {hStride : stride ≠ 0}
    (_layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
    (input : Tensor α (.dim inH (.dim inW .scalar)))
    (grad_output : Tensor α (pool2dOutShape inH inW kH kW stride)) :
    Tensor α (.dim inH (.dim inW .scalar)) := by
  let config := Private.maxPool2DConfig 0 h1 h2 hStride
  let gradOutput' := tensorCast _
    (Private.pool2d_spatial_out_shape_eq inH inW kH kW stride 0).symm grad_output
  exact maxPoolSpatialBackwardSpec (α := α) (inSpatial := #v[inH, inW])
    (kernel := #v[kH, kW]) (stride := #v[stride, stride]) (padding := #v[0, 0])
    config input gradOutput'

/-- VJP for channel-first two-dimensional max pooling. -/
def maxPool2dMultiBackwardSpec {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0}
    {h2 : kW ≠ 0} {hStride : stride ≠ 0}
    (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
    (input : Tensor α (.dim inC (.dim inH (.dim inW .scalar))))
    (grad_output : Tensor α (pool2dMultiOutShape inC inH inW kH kW stride)) :
    Tensor α (.dim inC (.dim inH (.dim inW .scalar))) := by
  let _ := layer
  let config := Private.maxPool2DConfig 0 h1 h2 hStride
  let gradOutput' := tensorCast _
    (Private.pool2d_multi_out_shape_eq inC inH inW kH kW stride 0).symm grad_output
  exact maxPoolBackwardSpec (α := α) (C := inC) (inSpatial := #v[inH, inW])
    (kernel := #v[kH, kW]) (stride := #v[stride, stride]) (padding := #v[0, 0])
    config input gradOutput'

/-- VJP for single-channel two-dimensional average pooling. -/
def avgPool2dBackwardSpec {kH kW inH inW stride : Nat} (_h1 : kH ≠ 0) (_h2 : kW ≠ 0)
    {hStride : stride ≠ 0} (_layer : AvgPool2DSpec kH kW stride _h1 _h2 hStride)
    (grad_output : Tensor α (pool2dOutShape inH inW kH kW stride)) :
    Tensor α (.dim inH (.dim inW .scalar)) := by
  let config := Private.avgPool2DConfig 0 _h1 _h2 hStride
  let gradOutput' := tensorCast _
    (Private.pool2d_spatial_out_shape_eq inH inW kH kW stride 0).symm grad_output
  exact avgPoolSpatialBackwardSpec (α := α) (inSpatial := #v[inH, inW])
    (kernel := #v[kH, kW]) (stride := #v[stride, stride]) (padding := #v[0, 0])
    config gradOutput'

/-! ## Smooth max pooling -/

/-- Single-channel smooth max pooling using the N-D log-sum-exp specification. -/
def smoothMaxPool2dSpec {kH kW inH inW stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
    {hStride : stride ≠ 0} (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
    (beta : α) (input : Tensor α (.dim inH (.dim inW .scalar))) :
    Tensor α (pool2dOutShape inH inW kH kW stride) := by
  let _ := layer
  let config := Private.maxPool2DConfig 0 h1 h2 hStride
  let output := smoothMaxPoolSpatialSpec (α := α) (inSpatial := #v[inH, inW])
    (kernel := #v[kH, kW]) (stride := #v[stride, stride]) (padding := #v[0, 0])
    config beta input
  exact tensorCast _ (Private.pool2d_spatial_out_shape_eq inH inW kH kW stride 0) output

/-- Channel-first smooth max pooling using the N-D log-sum-exp specification. -/
def smoothMaxPool2dMultiSpec {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0}
    {h2 : kW ≠ 0} {hStride : stride ≠ 0}
    (layer : MaxPool2DSpec kH kW stride h1 h2 hStride) (beta : α)
    (input : Tensor α (.dim inC (.dim inH (.dim inW .scalar)))) :
    Tensor α (pool2dMultiOutShape inC inH inW kH kW stride) := by
  let _ := layer
  let config := Private.maxPool2DConfig 0 h1 h2 hStride
  let output := smoothMaxPoolSpec (α := α) (C := inC) (inSpatial := #v[inH, inW])
    (kernel := #v[kH, kW]) (stride := #v[stride, stride]) (padding := #v[0, 0])
    config beta input
  exact tensorCast _ (Private.pool2d_multi_out_shape_eq inC inH inW kH kW stride 0) output

/-- JVP for single-channel smooth max pooling. -/
def smoothMaxPool2dJvpSpec {kH kW inH inW stride : Nat} {h1 : kH ≠ 0}
    {h2 : kW ≠ 0} {hStride : stride ≠ 0}
    (layer : MaxPool2DSpec kH kW stride h1 h2 hStride) (beta : α)
    (input tangent : Tensor α (.dim inH (.dim inW .scalar))) :
    Tensor α (pool2dOutShape inH inW kH kW stride) := by
  let _ := layer
  let config := Private.maxPool2DConfig 0 h1 h2 hStride
  let output := smoothMaxPoolSpatialJvpSpec (α := α) (inSpatial := #v[inH, inW])
    (kernel := #v[kH, kW]) (stride := #v[stride, stride]) (padding := #v[0, 0])
    config beta input tangent
  exact tensorCast _ (Private.pool2d_spatial_out_shape_eq inH inW kH kW stride 0) output

/-- JVP for channel-first smooth max pooling. -/
def smoothMaxPool2dMultiJvpSpec {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0}
    {h2 : kW ≠ 0} {hStride : stride ≠ 0}
    (layer : MaxPool2DSpec kH kW stride h1 h2 hStride) (beta : α)
    (input tangent : Tensor α (.dim inC (.dim inH (.dim inW .scalar)))) :
    Tensor α (pool2dMultiOutShape inC inH inW kH kW stride) := by
  let _ := layer
  let config := Private.maxPool2DConfig 0 h1 h2 hStride
  let output := smoothMaxPoolJvpSpec (α := α) (C := inC) (inSpatial := #v[inH, inW])
    (kernel := #v[kH, kW]) (stride := #v[stride, stride]) (padding := #v[0, 0])
    config beta input tangent
  exact tensorCast _ (Private.pool2d_multi_out_shape_eq inC inH inW kH kW stride 0) output

/-! ## Adaptive pooling -/

/-- Witness for adaptive average pooling to a fixed two-dimensional output shape. -/
structure AdaptiveAvgPool2DSpec (outH outW : Nat) where

/-- Witness for adaptive max pooling to a fixed two-dimensional output shape. -/
structure AdaptiveMaxPool2DSpec (outH outW : Nat) where

/-- Start of adaptive-pooling bin `i`: `floor(i * input / output)`. -/
def adaptiveStart (inSize outSize i : Nat) : Nat :=
  (i * inSize) / outSize

/-- End of adaptive-pooling bin `i`: `ceil((i + 1) * input / output)`. -/
def adaptiveEnd (inSize outSize i : Nat) : Nat :=
  ((i + 1) * inSize + outSize - 1) / outSize

/-- Two-dimensional adaptive average pooling with PyTorch-compatible bins. -/
def adaptiveAvgPool2dSpec {inH inW inC : Nat} (outH outW : Nat)
    (_layer : AdaptiveAvgPool2DSpec outH outW)
    (input : Tensor α (.dim inC (.dim inH (.dim inW .scalar))))
    (_hInH : inH > 0 := by norm_num) (_hInW : inW > 0 := by norm_num)
    (_hOutH : outH > 0 := by norm_num) (_hOutW : outW > 0 := by norm_num) :
    Tensor α (.dim inC (.dim outH (.dim outW .scalar))) :=
  Tensor.dim (fun c =>
    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        let startI := adaptiveStart inH outH i.val
        let startJ := adaptiveStart inW outW j.val
        let endI := adaptiveEnd inH outH i.val
        let endJ := adaptiveEnd inW outW j.val
        let height := endI - startI
        let width := endJ - startJ
        let sum := (List.range height).foldl (fun rowSum di =>
          (List.range width).foldl (fun acc dj =>
            let row := startI + di
            let col := startJ + dj
            if hRow : row < inH then
              if hCol : col < inW then
                acc + Tensor.toScalar
                  (getAtSpec (getAtSpec (getAtSpec input c) ⟨row, hRow⟩) ⟨col, hCol⟩)
              else acc
            else acc) rowSum) (0 : α)
        Tensor.scalar (sum / (height * width : Nat)))))

/-- Two-dimensional adaptive max pooling with PyTorch-compatible bins. -/
def adaptiveMaxPool2dSpec {inH inW inC : Nat} (outH outW : Nat)
    (_layer : AdaptiveMaxPool2DSpec outH outW)
    (input : Tensor α (.dim inC (.dim inH (.dim inW .scalar))))
    (_hInH : inH > 0 := by norm_num) (_hInW : inW > 0 := by norm_num)
    (_hOutH : outH > 0 := by norm_num) (_hOutW : outW > 0 := by norm_num) :
    Tensor α (.dim inC (.dim outH (.dim outW .scalar))) :=
  Tensor.dim (fun c =>
    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        let startI := adaptiveStart inH outH i.val
        let startJ := adaptiveStart inW outW j.val
        let endI := adaptiveEnd inH outH i.val
        let endJ := adaptiveEnd inW outW j.val
        let height := endI - startI
        let width := endJ - startJ
        let initial := Tensor.toScalar <| getValueAtPosition (getAtSpec input c) startI startJ
        let maximum := (List.range height).foldl (fun rowMax di =>
          (List.range width).foldl (fun current dj =>
            let row := startI + di
            let col := startJ + dj
            if hRow : row < inH then
              if hCol : col < inW then
                max current <| Tensor.toScalar
                  (getAtSpec (getAtSpec (getAtSpec input c) ⟨row, hRow⟩) ⟨col, hCol⟩)
              else current
            else current) rowMax) initial
        Tensor.scalar maximum)))

end Spec
