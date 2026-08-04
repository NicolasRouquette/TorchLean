/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Pooling.TwoD

@[expose] public section

namespace Spec

open Tensor

variable {α : Type} [Context α]

/-!
# Padded Two-Dimensional Pooling

These definitions expose the conventional two-dimensional, channel-first API for symmetric
padding. The numerical semantics come from `NN.Spec.Layers.Pooling.ND`: max pooling ignores
padded cells, while average pooling includes padded zeros in the divisor. This file contains only
dependent-shape adapters and therefore cannot drift from the N-dimensional implementation.
-/

/-- Channel-first two-dimensional max pooling with symmetric padding. -/
def maxPool2dMultiSpecPad {kH kW inH inW inC stride padding : Nat}
    {h1 : kH ≠ 0} {h2 : kW ≠ 0} {hStride : stride ≠ 0}
    (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
    (input : Tensor α (.dim inC (.dim inH (.dim inW .scalar)))) :
    Tensor α (pool2dMultiOutShapePad inC inH inW kH kW stride padding) := by
  let _ := layer
  let config := Private.maxPool2DConfig padding h1 h2 hStride
  let output := maxPoolSpec (α := α) (C := inC) (inSpatial := #v[inH, inW])
    (kernel := #v[kH, kW]) (stride := #v[stride, stride])
    (padding := #v[padding, padding]) config input
  exact tensorCast _
    (Private.pool2d_multi_out_shape_eq inC inH inW kH kW stride padding) output

/-- Selected-branch JVP for padded channel-first two-dimensional max pooling. -/
def maxPool2dMultiLinearizationSpecPad {kH kW inH inW inC stride padding : Nat}
    {h1 : kH ≠ 0} {h2 : kW ≠ 0} {hStride : stride ≠ 0}
    (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
    (input tangent : Tensor α (.dim inC (.dim inH (.dim inW .scalar)))) :
    Tensor α (pool2dMultiOutShapePad inC inH inW kH kW stride padding) := by
  let _ := layer
  let config := Private.maxPool2DConfig padding h1 h2 hStride
  let output := maxPoolLinearizationSpec (α := α) (C := inC)
    (inSpatial := #v[inH, inW]) (kernel := #v[kH, kW])
    (stride := #v[stride, stride]) (padding := #v[padding, padding]) config input tangent
  exact tensorCast _
    (Private.pool2d_multi_out_shape_eq inC inH inW kH kW stride padding) output

/-- Channel-first average pooling with symmetric zero padding and `count_include_pad = true`. -/
def avgPool2dMultiSpecPad {kH kW inH inW inC stride padding : Nat}
    (h1 : kH ≠ 0) (h2 : kW ≠ 0) {hStride : stride ≠ 0}
    (layer : AvgPool2DSpec kH kW stride h1 h2 hStride)
    (input : Tensor α (.dim inC (.dim inH (.dim inW .scalar)))) :
    Tensor α (pool2dMultiOutShapePad inC inH inW kH kW stride padding) := by
  let _ := layer
  let config := Private.avgPool2DConfig padding h1 h2 hStride
  let output := avgPoolSpec (α := α) (C := inC) (inSpatial := #v[inH, inW])
    (kernel := #v[kH, kW]) (stride := #v[stride, stride])
    (padding := #v[padding, padding]) config input
  exact tensorCast _
    (Private.pool2d_multi_out_shape_eq inC inH inW kH kW stride padding) output

/-- VJP for padded channel-first two-dimensional max pooling. -/
def maxPool2dMultiBackwardSpecPad {kH kW inH inW inC stride padding : Nat}
    {h1 : kH ≠ 0} {h2 : kW ≠ 0} {hStride : stride ≠ 0}
    (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
    (input : Tensor α (.dim inC (.dim inH (.dim inW .scalar))))
    (grad_output : Tensor α (pool2dMultiOutShapePad inC inH inW kH kW stride padding)) :
    Tensor α (.dim inC (.dim inH (.dim inW .scalar))) := by
  let _ := layer
  let config := Private.maxPool2DConfig padding h1 h2 hStride
  let gradOutput' := tensorCast _
    (Private.pool2d_multi_out_shape_eq inC inH inW kH kW stride padding).symm grad_output
  exact maxPoolBackwardSpec (α := α) (C := inC) (inSpatial := #v[inH, inW])
    (kernel := #v[kH, kW]) (stride := #v[stride, stride])
    (padding := #v[padding, padding]) config input gradOutput'

/-- VJP for padded channel-first average pooling with `count_include_pad = true`. -/
def avgPool2dMultiBackwardSpecPad {kH kW inH inW inC stride padding : Nat}
    (h1 : kH ≠ 0) (h2 : kW ≠ 0) {hStride : stride ≠ 0}
    (layer : AvgPool2DSpec kH kW stride h1 h2 hStride)
    (grad_output : Tensor α (pool2dMultiOutShapePad inC inH inW kH kW stride padding)) :
    Tensor α (.dim inC (.dim inH (.dim inW .scalar))) := by
  let _ := layer
  let config := Private.avgPool2DConfig padding h1 h2 hStride
  let gradOutput' := tensorCast _
    (Private.pool2d_multi_out_shape_eq inC inH inW kH kW stride padding).symm grad_output
  exact avgPoolBackwardSpec (α := α) (C := inC) (inSpatial := #v[inH, inW])
    (kernel := #v[kH, kW]) (stride := #v[stride, stride])
    (padding := #v[padding, padding]) config gradOutput'

/-! ## Smooth max VJPs -/

/-- VJP for unpadded single-channel smooth max pooling. -/
def smoothMaxPool2dBackwardSpec {kH kW inH inW stride : Nat}
    {h1 : kH ≠ 0} {h2 : kW ≠ 0} {hStride : stride ≠ 0}
    (_layer : MaxPool2DSpec kH kW stride h1 h2 hStride) (beta : α)
    (input : Tensor α (.dim inH (.dim inW .scalar)))
    (grad_output : Tensor α (pool2dOutShape inH inW kH kW stride)) :
    Tensor α (.dim inH (.dim inW .scalar)) := by
  let config := Private.maxPool2DConfig 0 h1 h2 hStride
  let gradOutput' := tensorCast _
    (Private.pool2d_spatial_out_shape_eq inH inW kH kW stride 0).symm grad_output
  exact smoothMaxPoolSpatialBackwardSpec (α := α) (inSpatial := #v[inH, inW])
    (kernel := #v[kH, kW]) (stride := #v[stride, stride]) (padding := #v[0, 0])
    config beta input gradOutput'

/-- VJP for unpadded channel-first smooth max pooling. -/
def smoothMaxPool2dMultiBackwardSpec {kH kW inH inW inC stride : Nat}
    {h1 : kH ≠ 0} {h2 : kW ≠ 0} {hStride : stride ≠ 0}
    (layer : MaxPool2DSpec kH kW stride h1 h2 hStride) (beta : α)
    (input : Tensor α (.dim inC (.dim inH (.dim inW .scalar))))
    (grad_output : Tensor α (pool2dMultiOutShape inC inH inW kH kW stride)) :
    Tensor α (.dim inC (.dim inH (.dim inW .scalar))) := by
  let _ := layer
  let config := Private.maxPool2DConfig 0 h1 h2 hStride
  let gradOutput' := tensorCast _
    (Private.pool2d_multi_out_shape_eq inC inH inW kH kW stride 0).symm grad_output
  exact smoothMaxPoolBackwardSpec (α := α) (C := inC) (inSpatial := #v[inH, inW])
    (kernel := #v[kH, kW]) (stride := #v[stride, stride]) (padding := #v[0, 0])
    config beta input gradOutput'

end Spec
