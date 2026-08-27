/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorOps

/-!
# Convolution Specifications

This file defines channels-first convolution and transpose convolution over an arbitrary spatial
rank. The core specification handles one sample; batched interfaces map it over their leading
dimensions.

PyTorch analogy: the grouped, dilated core corresponds to `torch.nn.Conv{d}d` with:

- arbitrary positive `groups`,
- per-axis `dilation` and `stride`,
- independent zero padding before and after each spatial axis,
- and the usual output-size formula (floor division, like PyTorch):

For each axis `a : Fin d`:

`out[a] = (in[a] + 2*padding[a] - kernel[a]) / stride[a] + 1`

The weight tensor has shape `(outC × inC × kernel[0] × ... × kernel[d-1])` and the bias has
shape `(outC)`.

Implementation notes:

- Convolution uses natural nested loops (outer axes first) and one `foldl` accumulator. This fixes
  an explicit evaluation order for executable scalar models.
- Padding semantics are implemented via `get_at_or_zero` plus an explicit guard for the
  left/top/front padding region (to avoid negative indices, which `Nat` cannot represent).
-/

@[expose] public section

namespace Spec
open Tensor

variable {α : Type} [Context α]

/-! ## Index helpers -/

namespace Conv
namespace Internal

def foldlIndices {β : Type} (dims : List Nat) (init : β) (f : β → List Nat → β) : β :=
  match dims with
  | [] => f init []
  | n :: ns =>
      (List.finRange n).foldl
        (fun acc i => foldlIndices ns acc (fun acc is => f acc (i.val :: is))) init

/--
Given:
- an output index tuple `outIdx`,
- a kernel index tuple `kIdx`,
- per-axis `stride` and `padding`,
compute the corresponding *input* index tuple (into the unpadded input),
or return `none` if we are in the left/top/front padding region on some axis.

Right/bottom/back padding is handled by `get_at_or_zero` when the computed index is out of bounds.
-/
def mkInputIdx?
    (outIdx kIdx stride padding : List Nat) : Option (List Nat) :=
  match outIdx, kIdx, stride, padding with
  | [], [], [], [] => some []
  | o :: os, k :: ks, s :: ss, p :: ps =>
      let q := o * s + k
      if _h : q < p then
        none
      else
        match mkInputIdx? os ks ss ps with
        | none => none
        | some rest => some ((q - p) :: rest)
  | _, _, _, _ => none

/-- Convolution input-index map with per-axis dilation. -/
def mkDilatedInputIdx?
    (outIdx kIdx stride dilation paddingBefore : List Nat) : Option (List Nat) :=
  match outIdx, kIdx, stride, dilation, paddingBefore with
  | [], [], [], [], [] => some []
  | o :: os, k :: ks, s :: ss, d :: ds, p :: ps =>
      let q := o * s + k * d
      if _h : q < p then
        none
      else
        match mkDilatedInputIdx? os ks ss ds ps with
        | none => none
        | some rest => some ((q - p) :: rest)
  | _, _, _, _, _ => none

/-- Unit dilation reduces the dilated convolution index map to the dense index map. -/
theorem mkDilatedInputIdx?_replicate_one
    (outIdx kernelIdx stride padding : List Nat) :
    mkDilatedInputIdx? outIdx kernelIdx stride (List.replicate stride.length 1) padding =
      mkInputIdx? outIdx kernelIdx stride padding := by
  induction outIdx generalizing kernelIdx stride padding with
  | nil =>
      cases kernelIdx <;> cases stride <;> cases padding <;>
        simp [mkDilatedInputIdx?, mkInputIdx?]
  | cons out outIdx ih =>
      cases kernelIdx with
      | nil => cases stride <;> cases padding <;> simp [mkDilatedInputIdx?, mkInputIdx?]
      | cons kernel kernelIdx =>
          cases stride with
          | nil => cases padding <;> simp [mkDilatedInputIdx?, mkInputIdx?]
          | cons step stride =>
              cases padding with
              | nil => simp [mkDilatedInputIdx?, mkInputIdx?]
              | cons pad padding =>
                  simp only [List.length_cons, List.replicate_succ, mkDilatedInputIdx?,
                    mkInputIdx?, mul_one]
                  split <;> simp_all

/--
Given:
- an output index tuple `outIdx`,
- a kernel index tuple `kIdx`,
- per-axis `stride` and `padding`,
compute the corresponding *input* index tuple for transpose convolution, or `none` if
the equality `out + padding = in * stride + k` cannot be satisfied on some axis.

Implementation detail: for each axis we solve

`in = (out + padding - k) / stride`

and require divisibility (`% stride = 0`) plus `out + padding ≥ k`.
Out-of-bounds input indices are handled by `get_at_or_zero` at the call site.
-/
def mkTransposeInputIdx?
    (outIdx kIdx stride padding : List Nat) : Option (List Nat) :=
  match outIdx, kIdx, stride, padding with
  | [], [], [], [] => some []
  | o :: os, k :: ks, s :: ss, p :: ps =>
      if s = 0 then
        none
      else
        let q := o + p
        if _h : q < k then
          none
        else
          let r := q - k
          if _hs : r % s = 0 then
            match mkTransposeInputIdx? os ks ss ps with
            | none => none
            | some rest => some ((r / s) :: rest)
          else
            none
  | _, _, _, _ => none

def matchesInputPos
    (outIdx kIdx stride padding inIdx : List Nat) : Bool :=
  match outIdx, kIdx, stride, padding, inIdx with
  | [], [], [], [], [] => true
  | o :: os, k :: ks, s :: ss, p :: ps, i :: is =>
      decide (o * s + k = i + p) && matchesInputPos os ks ss ps is
  | _, _, _, _, _ => false

end Internal
end Conv

/-! ## Spec definition -/

/-- Parameters for an arbitrary-rank dense convolution, in channels-first layout. -/
structure ConvSpec (d inC outC : Nat) (kernel stride padding : Tensor Nat [d]) (α : Type) where
  /-- Kernel weights, shape `(outC, inC, kernel[0], ..., kernel[d-1])`. -/
  kernel : Tensor α (Shape.ofList (outC :: inC :: kernel.toList))
  /-- Bias, shape `(outC)`. -/
  bias   : Tensor α [outC]

/-- Effective extent of a dilated kernel along one axis. -/
def convEffectiveKernel (kernel dilation : Nat) : Nat :=
  if kernel = 0 then 0 else dilation * (kernel - 1) + 1

/-- Output extent of a dilated window with independent two-sided padding. -/
def convOutDimDilated
    (input kernel stride dilation paddingBefore paddingAfter : Nat) : Nat :=
  let effective := convEffectiveKernel kernel dilation
  let padded := input + paddingBefore + paddingAfter
  if effective = 0 || stride = 0 || padded < effective then
    0
  else
    (padded - effective) / stride + 1

/-- Output spatial sizes for grouped/dilated convolution with asymmetric zero padding. -/
def convOutSpatialDilated {d : Nat}
    (inSpatial kernel stride dilation paddingBefore paddingAfter : Tensor Nat [d]) :
    Tensor Nat [d] :=
  Tensor.ofFn fun axis =>
    convOutDimDilated (inSpatial.getScalar axis) (kernel.getScalar axis)
      (stride.getScalar axis) (dilation.getScalar axis) (paddingBefore.getScalar axis)
      (paddingAfter.getScalar axis)

/-- Output spatial sizes, one extent for each spatial axis. -/
def convOutSpatial {d : Nat} (inSpatial kernel stride padding : Tensor Nat [d]) : Tensor Nat [d] :=
  Tensor.ofFn (fun a =>
    Shape.slidingWindowOutDim
      (inSpatial.getScalar a) (kernel.getScalar a) (stride.getScalar a) (padding.getScalar a))

/-- Dilated convolution geometry reduces to the symmetric, unit-dilation case. -/
@[simp]
theorem convOutSpatialDilated_one_symmetric {d : Nat}
    (input kernel stride padding : Tensor Nat [d]) :
    convOutSpatialDilated input kernel stride (fill 1 [d]) padding padding =
      convOutSpatial input kernel stride padding := by
  apply Tensor.ext_vector
  intro i
  simp only [convOutSpatialDilated, convOutSpatial, Tensor.getScalar_ofFn]
  by_cases hk : kernel.getScalar i = 0
  · simp [convOutDimDilated, convEffectiveKernel, Shape.slidingWindowOutDim, hk]
  · have hkpos : 1 ≤ kernel.getScalar i := Nat.one_le_iff_ne_zero.mpr hk
    simp [convOutDimDilated, convEffectiveKernel, Shape.slidingWindowOutDim, hk,
      Nat.sub_add_cancel hkpos]
    grind

/-- A unit kernel with unit stride and no padding preserves every positive spatial extent. -/
theorem convOutSpatial_unit {d : Nat} (spatial : Tensor Nat [d])
    (hSpatial : ∀ i : Fin d, spatial.getScalar i ≠ 0) :
    convOutSpatial spatial (fill 1 [d]) (fill 1 [d])
      (fill 0 [d]) = spatial := by
  apply Tensor.ext_vector
  intro i
  have hPos : 1 ≤ spatial.getScalar i :=
    Nat.one_le_iff_ne_zero.mpr (hSpatial i)
  have hNonzero : spatial.getScalar i ≠ 0 := hSpatial i
  simpa [convOutSpatial, Shape.slidingWindowOutDim, hNonzero,
    Nat.not_lt.mpr hPos] using Nat.sub_add_cancel hPos

/--
Unit-stride convolution with an odd kernel and padding equal to the kernel radius preserves every
positive spatial extent.
-/
theorem convOutSpatial_same {d : Nat} (spatial radius : Tensor Nat [d])
    (hSpatial : ∀ i : Fin d, spatial.getScalar i ≠ 0) :
    convOutSpatial spatial (radius.map fun p => 2 * p + 1) (fill 1 [d]) radius =
      spatial := by
  apply Tensor.ext_vector
  intro i
  have hNonzero : spatial.getScalar i ≠ 0 := hSpatial i
  simp [convOutSpatial, Shape.slidingWindowOutDim, hNonzero]
  grind

/-- Output spatial shape `Shape.ofList [out0, ..., out(d-1)]`. -/
def convOutShape {d : Nat} (inSpatial kernel stride padding : Tensor Nat [d]) : Shape :=
  Shape.ofList (convOutSpatial inSpatial kernel stride padding).toList

/-- Output shape including channels: `Shape.ofList (outC :: [out0, ..., out(d-1)])`. -/
def convMultiOutShape {d : Nat} (_inC outC : Nat) (inSpatial kernel stride padding : Tensor Nat [d])
    : Shape :=
  Shape.ofList (outC :: (convOutSpatial inSpatial kernel stride padding).toList)

/-- The grouped bilinear contraction shared by arbitrary-rank convolutions. -/
def Conv.Internal.convCoreWith
    {d inC outC : Nat} {kernel inSpatial outSpatial : Tensor Nat [d]}
    (channelsPerOutput : Nat)
    (inputChannel : Fin outC → Fin channelsPerOutput → Nat)
    (inputIndex? : List Nat → List Nat → Option (List Nat))
    (weights : Tensor α (Shape.ofList (outC :: inC :: kernel.toList)))
    (input : Tensor α (Shape.ofList (inC :: inSpatial.toList))) :
    Tensor α (Shape.ofList (outC :: outSpatial.toList)) :=
  Tensor.dim fun outChannel =>
    Tensor.generate outSpatial.toList fun outIndex =>
      (List.finRange channelsPerOutput).foldl (fun acc localChannel =>
        let inChannel := inputChannel outChannel localChannel
        Conv.Internal.foldlIndices kernel.toList acc fun acc kernelIndex =>
          let inputValue : α :=
            match inputIndex? outIndex kernelIndex with
            | none => 0
            | some inputIndex => getAtOrZero input (inChannel :: inputIndex)
          let kernelValue : α :=
            getAtOrZero weights (outChannel.val :: inChannel :: kernelIndex)
          acc + inputValue * kernelValue) 0

/-- The grouped, dilated contraction for an arbitrary-rank convolution. -/
def groupedConvCoreSpec
    {d inC outC : Nat}
    {kernel stride dilation paddingBefore paddingAfter : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (groups : Nat)
    (weights : Tensor α (Shape.ofList (outC :: inC :: kernel.toList)))
    (input : Tensor α (Shape.ofList (inC :: inSpatial.toList))) :
    Tensor α (Shape.ofList (outC ::
      (convOutSpatialDilated inSpatial kernel stride dilation paddingBefore paddingAfter).toList)) :=

  let inChannelsPerGroup := inC / groups
  let outChannelsPerGroup := outC / groups
  Conv.Internal.convCoreWith inChannelsPerGroup
    (fun outChannel localChannel =>
      (outChannel.val / outChannelsPerGroup) * inChannelsPerGroup + localChannel.val)
    (fun outIndex kernelIndex =>
      Conv.Internal.mkDilatedInputIdx? outIndex kernelIndex stride.toList dilation.toList
        paddingBefore.toList)
    weights input

/-- The bilinear kernel/input contraction underlying an arbitrary-rank dense convolution. -/
def convCoreSpec
    {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (weights : Tensor α (Shape.ofList (outC :: inC :: kernel.toList)))
    (input : Tensor α (Shape.ofList (inC :: inSpatial.toList))) :
    Tensor α (Shape.ofList (outC :: (convOutSpatial inSpatial kernel stride padding).toList)) :=
  Conv.Internal.convCoreWith inC (fun _ inChannel => inChannel.val)
    (fun outIndex kernelIndex =>
      Conv.Internal.mkInputIdx? outIndex kernelIndex stride.toList padding.toList)
    weights input

/--
The grouped convolution contraction reduces to dense convolution for one group, unit dilation,
and symmetric padding. The explicit cast transports the dilated output shape across the geometric
specialization theorem.
 -/
theorem castShape_groupedConvCoreSpec_one_symmetric
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (weights : Tensor α (Shape.ofList (outC :: inC :: kernel.toList)))
    (input : Tensor α (Shape.ofList (inC :: inSpatial.toList))) :
    Tensor.castShape
        (groupedConvCoreSpec (dilation := fill 1 [d]) (paddingBefore := padding)
          (paddingAfter := padding) (stride := stride) 1 weights input)
        (congrArg (fun spatial => Shape.ofList (outC :: spatial.toList))
          (convOutSpatialDilated_one_symmetric inSpatial kernel stride padding)) =
      convCoreSpec (stride := stride) (padding := padding) weights input := by
  unfold groupedConvCoreSpec convCoreSpec
  simp only [Nat.div_one]
  have hStrideLength : stride.toList.length = d := by
    simp [Shape.size]
  have hFillStride : (fill 1 [d]).toList =
      List.replicate stride.toList.length 1 := by
    simp [hStrideLength, Shape.size]
  have hInputIndex :
      (fun outIndex kernelIndex =>
          Conv.Internal.mkDilatedInputIdx? outIndex kernelIndex stride.toList
            (fill 1 [d]).toList padding.toList) =
        fun outIndex kernelIndex =>
          Conv.Internal.mkInputIdx? outIndex kernelIndex stride.toList padding.toList := by
    funext outIndex kernelIndex
    rw [hFillStride]
    exact Conv.Internal.mkDilatedInputIdx?_replicate_one
      outIndex kernelIndex stride.toList padding.toList
  have hInputChannel :
      (fun (outChannel : Fin outC) (localChannel : Fin inC) =>
          (outChannel.val / outC) * inC + localChannel.val) =
        fun _ localChannel => localChannel.val := by
    funext outChannel localChannel
    simp [Nat.div_eq_of_lt outChannel.isLt]
  apply Tensor.ext_getSpec
  intro index
  simp only [get_spec_castShape]
  rw [convOutSpatialDilated_one_symmetric]
  rw [Nat.div_one inC]
  rw [hInputChannel, hInputIndex]

/-- Broadcast one channel value over a supplied spatial shape. -/
def Conv.Internal.convBiasBroadcastWith
    {d outC : Nat} (outSpatial : Tensor Nat [d]) (bias : Tensor α [outC]) :
    Tensor α (Shape.ofList (outC :: outSpatial.toList)) :=
  Tensor.dim fun outChannel =>
    Tensor.generate outSpatial.toList fun _ => getAtOrZero bias [outChannel.val]

/-- Broadcast a convolution bias over every output spatial position. -/
def convBiasBroadcastSpec
    {d outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (bias : Tensor α [outC]) :
    Tensor α (Shape.ofList (outC :: (convOutSpatial inSpatial kernel stride padding).toList)) :=
  Conv.Internal.convBiasBroadcastWith (convOutSpatial inSpatial kernel stride padding) bias

/-- Add a channel bias to a dilated convolution output. -/
def convBiasBroadcastDilatedSpec
    {d outC : Nat}
    {kernel stride dilation paddingBefore paddingAfter inSpatial : Tensor Nat [d]}
    (bias : Tensor α [outC]) :
    Tensor α (Shape.ofList (outC ::
      (convOutSpatialDilated inSpatial kernel stride dilation paddingBefore paddingAfter).toList)) :=
  Conv.Internal.convBiasBroadcastWith
    (convOutSpatialDilated inSpatial kernel stride dilation paddingBefore paddingAfter) bias

/-- Dilated-output bias broadcasting reduces to dense bias broadcasting in the symmetric case. -/
theorem castShape_convBiasBroadcastDilatedSpec_one_symmetric
    {d outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (bias : Tensor α [outC]) :
    Tensor.castShape
        (convBiasBroadcastDilatedSpec (kernel := kernel) (stride := stride)
          (dilation := fill 1 [d]) (paddingBefore := padding) (paddingAfter := padding)
          (inSpatial := inSpatial) bias)
        (congrArg (fun spatial => Shape.ofList (outC :: spatial.toList))
          (convOutSpatialDilated_one_symmetric inSpatial kernel stride padding)) =
      convBiasBroadcastSpec (kernel := kernel) (stride := stride) (padding := padding)
        (inSpatial := inSpatial) bias := by
  apply Tensor.ext_getSpec
  intro index
  simp only [get_spec_castShape]
  unfold convBiasBroadcastDilatedSpec convBiasBroadcastSpec
  rw [convOutSpatialDilated_one_symmetric]

/-- Numerical semantics for grouped, dilated convolution with asymmetric zero padding. -/
def groupedConvSpec
    {d inC outC : Nat}
    {kernel stride dilation paddingBefore paddingAfter : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (groups : Nat)
    (weights : Tensor α (Shape.ofList (outC :: inC :: kernel.toList)))
    (bias : Tensor α [outC])
    (input : Tensor α (Shape.ofList (inC :: inSpatial.toList))) :
    Tensor α (Shape.ofList (outC ::
      (convOutSpatialDilated inSpatial kernel stride dilation paddingBefore paddingAfter).toList)) :=
  addSpec (groupedConvCoreSpec groups weights input)
    (convBiasBroadcastDilatedSpec bias)

/--
Arbitrary-rank dense convolution on a single channels-first input (no batch dimension).

Mathematically, for output channel `oc` and output spatial index `o : Tensor Nat [d]`:

`y[oc,o] = Σ_{ic, k} x_pad[ic, o*stride + k] * W[oc,ic,k] + b[oc]`

where `k` ranges over the kernel window and `x_pad` is `input` with zero-padding.
-/
def convSpec
    {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding α)
    (input : Tensor α (Shape.ofList (inC :: inSpatial.toList))) :
    Tensor α (Shape.ofList (outC :: (convOutSpatial inSpatial kernel stride padding).toList)) :=
  addSpec (convCoreSpec layer.kernel input) (convBiasBroadcastSpec layer.bias)

/-- Grouped convolution reduces to dense convolution for its canonical dense configuration. -/
theorem castShape_groupedConvSpec_one_symmetric
    {d inC outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (weights : Tensor α (Shape.ofList (outC :: inC :: kernel.toList)))
    (bias : Tensor α [outC])
    (input : Tensor α (Shape.ofList (inC :: inSpatial.toList))) :
    Tensor.castShape
        (groupedConvSpec (dilation := fill 1 [d]) (paddingBefore := padding)
          (paddingAfter := padding) (stride := stride) 1 weights bias input)
        (congrArg (fun spatial => Shape.ofList (outC :: spatial.toList))
          (convOutSpatialDilated_one_symmetric inSpatial kernel stride padding)) =
      convSpec (stride := stride) (padding := padding) { kernel := weights, bias := bias } input := by
  unfold groupedConvSpec convSpec
  rw [Tensor.castShape_addSpec]
  rw [castShape_groupedConvCoreSpec_one_symmetric]
  rw [castShape_convBiasBroadcastDilatedSpec_one_symmetric]

/-- Directional derivative formula for convolution in its kernel, bias, and input arguments. -/
def convJvpSpec
    {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (layer tangentLayer : ConvSpec d inC outC kernel stride padding α)
    (input tangentInput : Tensor α (Shape.ofList (inC :: inSpatial.toList))) :
    Tensor α (Shape.ofList (outC :: (convOutSpatial inSpatial kernel stride padding).toList)) :=
  addSpec
    (addSpec (convCoreSpec tangentLayer.kernel input)
      (convCoreSpec layer.kernel tangentInput))
    (convBiasBroadcastSpec tangentLayer.bias)


/-- Gradient of convolution output w.r.t. the kernel weights (given `grad_output`). -/
def convKernelDerivSpec
    {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (_layer : ConvSpec d inC outC kernel stride padding α)
    (input : Tensor α (Shape.ofList (inC :: inSpatial.toList)))
    (grad_output :
      Tensor α
        (Shape.ofList (outC :: (convOutSpatial inSpatial kernel stride padding).toList)))
    : Tensor α (Shape.ofList (outC :: inC :: kernel.toList)) :=

  let outSpatial := convOutSpatial inSpatial kernel stride padding
  let outDims : List Nat := outSpatial.toList
  let kDims : List Nat := kernel.toList
  let strideDims : List Nat := stride.toList
  let padDims : List Nat := padding.toList

  Tensor.dim (fun out_ch =>
    Tensor.dim (fun in_ch =>
      Tensor.generate kDims (fun kIdx =>
        let total_sum : α :=
          Conv.Internal.foldlIndices outDims 0 (fun acc outIdx =>
            let input_val : α :=
              match Conv.Internal.mkInputIdx? outIdx kIdx strideDims padDims with
              | none => 0
              | some inIdx => getAtOrZero input (in_ch.val :: inIdx)
            let grad_val : α :=
              getAtOrZero grad_output (out_ch.val :: outIdx)
            acc + input_val * grad_val
          )
        total_sum
      )
    )
  )

/-- Gradient of convolution output w.r.t. the bias (sum over spatial positions). -/
def convBiasDerivSpec
    {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (_layer : ConvSpec d inC outC kernel stride padding α)
    (_input : Tensor α (Shape.ofList (inC :: inSpatial.toList)))
    (grad_output :
      Tensor α
        (Shape.ofList (outC :: (convOutSpatial inSpatial kernel stride padding).toList)))
    : Tensor α [outC] :=

  let outSpatial := convOutSpatial inSpatial kernel stride padding
  let outDims : List Nat := outSpatial.toList

  Tensor.dim (fun out_ch =>
    let total_sum : α :=
      Conv.Internal.foldlIndices outDims 0 (fun acc outIdx =>
        acc + getAtOrZero grad_output (out_ch.val :: outIdx)
      )
    Tensor.scalar total_sum
  )


/--
Gradient of convolution output w.r.t. the input (the "input-gradient" / transpose-convolution map).

This mirrors `conv{1,2,3}d_input_deriv_spec` but for arbitrary spatial rank `d`.
-/
def convInputDerivSpec
    {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding α)
    (_input : Tensor α (Shape.ofList (inC :: inSpatial.toList)))
    (grad_output :
      Tensor α
        (Shape.ofList (outC :: (convOutSpatial inSpatial kernel stride padding).toList)))
    : Tensor α (Shape.ofList (inC :: inSpatial.toList)) :=

  let kDims : List Nat := kernel.toList
  let strideDims : List Nat := stride.toList
  let padDims : List Nat := padding.toList
  let inDims : List Nat := inSpatial.toList

  Tensor.dim (fun in_ch =>
    Tensor.generate inDims (fun inIdx =>
      let total_sum : α :=
        (List.finRange outC).foldl (fun acc out_ch =>
          Conv.Internal.foldlIndices kDims acc (fun acc kIdx =>
            let contrib : α :=
              match Conv.Internal.mkTransposeInputIdx? inIdx kIdx strideDims padDims with
              | none => 0
              | some outIdx =>
                  let grad_val := getAtOrZero grad_output (out_ch.val :: outIdx)
                  let kernel_val := getAtOrZero layer.kernel (out_ch.val :: in_ch.val :: kIdx)
                  grad_val * kernel_val
            acc + contrib
          )
        ) 0
      total_sum
    )
  )


/-- Convolution backward pass: returns `(dKernel, dBias, dInput)`. -/
def convBackwardSpec
    {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding α)
    (input : Tensor α (Shape.ofList (inC :: inSpatial.toList)))
    (grad_output :
      Tensor α
        (Shape.ofList (outC :: (convOutSpatial inSpatial kernel stride padding).toList)))
    :
    (Tensor α (Shape.ofList (outC :: inC :: kernel.toList)) ×
     Tensor α [outC] ×
     Tensor α (Shape.ofList (inC :: inSpatial.toList))) :=
  let d_kernel := convKernelDerivSpec layer input grad_output
  let d_bias := convBiasDerivSpec layer input grad_output
  let d_input := convInputDerivSpec layer input grad_output
  (d_kernel, d_bias, d_input)

/-! ## Transpose convolution -/

/--
Parameters for an arbitrary-rank transpose convolution, in channels-first layout.

PyTorch analogy: this is `torch.nn.ConvTranspose{d}d` with:

- `output_padding = 0`,
- `dilation = 1`,
- `groups = 1`,
- per-axis `stride` and `padding`,
- and weight layout `(inC, outC, k0, ..., k(d-1))`.
-/
structure ConvTransposeSpec (d inC outC : Nat) (kernel stride padding : Tensor Nat [d]) (α : Type) where
  /-- Kernel weights, shape `(inC, outC, kernel[0], ..., kernel[d-1])`. -/
  kernel : Tensor α (Shape.ofList (inC :: outC :: kernel.toList))
  /-- Bias, shape `(outC)`. -/
  bias   : Tensor α [outC]

/--
Output size along one transpose-convolution axis with `output_padding = 0`.

For positive input, kernel, and stride this is
`(input - 1) * stride + kernel - 2 * padding`. A zero input, kernel, or stride is treated as an
invalid axis and has size zero; excessive padding also saturates the final subtraction at zero.
The addition precedes subtraction intentionally: Nat subtraction in
`(input - 1) * stride - 2 * padding + kernel` does not represent the integer formula.
-/
def convTransposeOutDim (inDim kDim stride padding : Nat) : Nat :=
  if inDim = 0 || kDim = 0 || stride = 0 then
    0
  else
    (inDim - 1) * stride + kDim - 2 * padding

/-- Output spatial sizes (`Tensor Nat [d]`) for transpose convolution (`output_padding = 0`). -/
def convTransposeOutSpatial {d : Nat} (inSpatial kernel stride padding : Tensor Nat [d]) :
    Tensor Nat [d] :=
  Tensor.ofFn (fun a =>
    convTransposeOutDim (inSpatial.getScalar a) (kernel.getScalar a) (stride.getScalar a) (padding.getScalar a))

/-- Output spatial shape `Shape.ofList [out0, ..., out(d-1)]` (transpose convolution). -/
def convTransposeOutShape {d : Nat} (inSpatial kernel stride padding : Tensor Nat [d]) : Shape :=
  Shape.ofList (convTransposeOutSpatial inSpatial kernel stride padding).toList

/-- Output shape including channels: `Shape.ofList (outC :: [out0, ..., out(d-1)])`. -/
def convTransposeMultiOutShape {d : Nat} (_inC outC : Nat)
    (inSpatial kernel stride padding : Tensor Nat [d]) : Shape :=
  Shape.ofList (outC :: (convTransposeOutSpatial inSpatial kernel stride padding).toList)

/--
Arbitrary-rank transpose convolution on a single channels-first input (no batch dimension).

 For output channel `oc` and output spatial index `o : Tensor Nat [d]` we define:

`y[oc,o] = Σ_{ic, k} x[ic, (o + padding - k) / stride] * W[ic,oc,k] + b[oc]`

where each axis must satisfy `out + padding ≥ k` and divisibility by `stride` (`% stride = 0`).
-/
def convTransposeSpec
    {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (layer : ConvTransposeSpec d inC outC kernel stride padding α)
    (input : Tensor α (Shape.ofList (inC :: inSpatial.toList))) :
    Tensor α
      (Shape.ofList (outC :: (convTransposeOutSpatial inSpatial kernel stride padding).toList))
    :=

  let outSpatial := convTransposeOutSpatial inSpatial kernel stride padding
  let outDims : List Nat := outSpatial.toList
  let kDims : List Nat := kernel.toList
  let strideDims : List Nat := stride.toList
  let padDims : List Nat := padding.toList

  Tensor.dim (fun out_ch =>
    Tensor.generate outDims (fun outIdx =>
      let total_sum : α :=
        (List.finRange inC).foldl (fun acc in_ch =>
          Conv.Internal.foldlIndices kDims acc (fun acc kIdx =>
            let input_val : α :=
              match Conv.Internal.mkTransposeInputIdx? outIdx kIdx strideDims padDims with
              | none => 0
              | some inIdx => getAtOrZero input (in_ch.val :: inIdx)
            let kernel_val : α :=
              getAtOrZero layer.kernel (in_ch.val :: out_ch.val :: kIdx)
            acc + input_val * kernel_val
          )
        ) 0
      total_sum + getAtOrZero layer.bias [out_ch.val]
    )
  )


/-- Gradient of transpose convolution output w.r.t. the kernel weights (given `grad_output`). -/
def convTransposeKernelDerivSpec
    {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (input : Tensor α (Shape.ofList (inC :: inSpatial.toList)))
    (grad_output :
      Tensor α
        (Shape.ofList
          (outC :: (convTransposeOutSpatial inSpatial kernel stride padding).toList)))
    : Tensor α (Shape.ofList (inC :: outC :: kernel.toList)) :=

  let inDims : List Nat := inSpatial.toList
  let kDims : List Nat := kernel.toList
  let strideDims : List Nat := stride.toList
  let padDims : List Nat := padding.toList

  Tensor.dim (fun in_ch =>
    Tensor.dim (fun out_ch =>
      Tensor.generate kDims (fun kIdx =>
        let total_sum : α :=
          Conv.Internal.foldlIndices inDims 0 (fun acc inIdx =>
            match Conv.Internal.mkInputIdx? inIdx kIdx strideDims padDims with
            | none => acc
            | some outIdx =>
                let x : α := getAtOrZero input (in_ch.val :: inIdx)
                let g : α := getAtOrZero grad_output (out_ch.val :: outIdx)
                acc + x * g
          )
        total_sum
      )
    )
  )


/-- Gradient of transpose convolution output w.r.t. the bias (sum over spatial positions). -/
def convTransposeBiasDerivSpec
    {d outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (grad_output :
      Tensor α
        (Shape.ofList
          (outC :: (convTransposeOutSpatial inSpatial kernel stride padding).toList)))
    : Tensor α [outC] :=

  let outSpatial := convTransposeOutSpatial inSpatial kernel stride padding
  let outDims : List Nat := outSpatial.toList

  Tensor.dim (fun out_ch =>
    let total_sum : α :=
      Conv.Internal.foldlIndices outDims 0 (fun acc outIdx =>
        acc + getAtOrZero grad_output (out_ch.val :: outIdx)
      )
    Tensor.scalar total_sum
  )


/-- Gradient of transpose convolution output w.r.t. the input (given `grad_output`). -/
def convTransposeInputDerivSpec
    {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (weights : Tensor α (Shape.ofList (inC :: outC :: kernel.toList)))
    (grad_output :
      Tensor α
        (Shape.ofList
          (outC :: (convTransposeOutSpatial inSpatial kernel stride padding).toList)))
    : Tensor α (Shape.ofList (inC :: inSpatial.toList)) :=

  let inDims : List Nat := inSpatial.toList
  let kDims : List Nat := kernel.toList
  let strideDims : List Nat := stride.toList
  let padDims : List Nat := padding.toList

  Tensor.dim (fun in_ch =>
    Tensor.generate inDims (fun inIdx =>
      let total_sum : α :=
        (List.finRange outC).foldl (fun acc out_ch =>
          Conv.Internal.foldlIndices kDims acc (fun acc kIdx =>
            match Conv.Internal.mkInputIdx? inIdx kIdx strideDims padDims with
            | none => acc
            | some outIdx =>
                let w : α := getAtOrZero weights (in_ch.val :: out_ch.val :: kIdx)
                let g : α := getAtOrZero grad_output (out_ch.val :: outIdx)
                acc + w * g
          )
        ) 0
      total_sum
    )
  )


/-- Transpose convolution backward pass: returns `(dKernel, dBias, dInput)`. -/
def convTransposeBackwardSpec
    {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (layer : ConvTransposeSpec d inC outC kernel stride padding α)
    (input : Tensor α (Shape.ofList (inC :: inSpatial.toList)))
    (grad_output :
      Tensor α
        (Shape.ofList
          (outC :: (convTransposeOutSpatial inSpatial kernel stride padding).toList)))
    :
    (Tensor α (Shape.ofList (inC :: outC :: kernel.toList)) ×
     Tensor α [outC] ×
     Tensor α (Shape.ofList (inC :: inSpatial.toList))) :=
  let d_kernel := convTransposeKernelDerivSpec input grad_output
  let d_bias := convTransposeBiasDerivSpec grad_output
  let d_input := convTransposeInputDerivSpec layer.kernel grad_output
  (d_kernel, d_bias, d_input)

end Spec
