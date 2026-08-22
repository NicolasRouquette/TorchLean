/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape

@[expose] public section


namespace Spec
open Tensor

variable {α : Type} [Context α]

/-!
# N-D Pooling

Dimension-polymorphic pooling specs for spatial tensors and channels-first tensors.
-/

/-!
## Generic N-D pooling (channels-first, no batch)

These operators generalize the existing 2D pooling specs to an arbitrary spatial rank `d`.

Conventions:
- Input is channels-first: shape `[C] ++ spatialDims`.
- Pooling is applied independently per channel (like the existing 2D specs).
- `kernel`, `stride`, and `padding` are per-axis vectors (`Vector Nat d`).
- Padding is symmetric. Average pooling counts padded positions as zeros. Max pooling ignores
  padded positions; a window with no input position is outside PyTorch's valid max-pool domain and
  is totalized to zero by the scalar-polymorphic TorchLean spec.

PyTorch comparisons (conceptual, without batch axis):
- `max_pool_spec` corresponds to `torch.nn.functional.max_poolNd`.
- `avg_pool_spec` corresponds to `torch.nn.functional.avg_poolNd`.
-/

/-!
### Layer configs + output shapes
-/

/--
Witness that an N-D max-pooling configuration has nonzero kernel and stride on every axis.

The vectors are indices of the type rather than duplicate structure fields, so a value cannot
advertise one configuration while its type describes another.
-/
structure MaxPoolSpec (d : Nat)
    (kernel stride padding : Vector Nat d)
    (hKernel : ∀ i : Fin d, kernel.get i ≠ 0)
    (hStride : ∀ i : Fin d, stride.get i ≠ 0) where

/-- Witness that an N-D average-pooling configuration has nonzero kernel and stride on every axis. -/
structure AvgPoolSpec (d : Nat)
    (kernel stride padding : Vector Nat d)
    (hKernel : ∀ i : Fin d, kernel.get i ≠ 0)
    (hStride : ∀ i : Fin d, stride.get i ≠ 0) where

/--
Output spatial sizes with symmetric padding.

Pooling follows the usual floor-mode sliding-window formula, but an empty input axis, empty kernel,
or padding larger than half the kernel gives an empty output axis. The last condition is part of the
pooling contract used by PyTorch and by TorchLean's native implementations; it is not a restriction
on convolution.
-/
def poolOutDim (input kernel stride padding : Nat) : Nat :=
  if input = 0 || kernel = 0 || padding > kernel / 2 then
    0
  else
    Shape.slidingWindowOutDim input kernel stride padding

/--
Output spatial sizes without padding.

An invalid axis (empty input, zero kernel, zero stride, or a kernel larger than the input) has size
zero.
-/
def poolOutSpatial {d : Nat} (inSpatial kernel stride : Vector Nat d) : Vector Nat d :=
  Vector.ofFn (fun i =>
    poolOutDim (inSpatial.get i) (kernel.get i) (stride.get i) 0)

/-- Apply `poolOutDim` independently to each spatial axis. -/
def poolOutSpatialPad {d : Nat} (inSpatial kernel stride padding : Vector Nat d) : Vector Nat d :=
  Vector.ofFn (fun i =>
    poolOutDim (inSpatial.get i) (kernel.get i) (stride.get i) (padding.get i))

/-- Pooling over the complete spatial extent produces one value on every spatial axis. -/
theorem poolOutSpatialPad_global {d : Nat} (spatial : Vector Nat d)
    (hSpatial : ∀ i : Fin d, spatial.get i ≠ 0) :
    poolOutSpatialPad spatial spatial (Vector.replicate d 1) (Vector.replicate d 0) =
      Vector.replicate d 1 := by
  apply Vector.ext
  intro i hi
  have hNonzero : spatial[i] ≠ 0 := by
    simpa [Vector.get] using hSpatial ⟨i, hi⟩
  simp [poolOutSpatialPad, poolOutDim, Shape.slidingWindowOutDim, Vector.get, hNonzero]

/-- Output shape for single-channel N-D pooling (no padding). -/
def poolOutShape {d : Nat} (inSpatial kernel stride : Vector Nat d) : Shape :=
  Shape.ofList (poolOutSpatial inSpatial kernel stride).toList

/-- Output shape for channels-first N-D pooling (no padding; channels preserved). -/
def poolMultiOutShape {d : Nat} (inC : Nat) (inSpatial kernel stride : Vector Nat d) : Shape :=
  Shape.ofList (inC :: (poolOutSpatial inSpatial kernel stride).toList)

/-- Output shape for single-channel N-D pooling with symmetric padding. -/
def poolOutShapePad {d : Nat} (inSpatial kernel stride padding : Vector Nat d) : Shape :=
  Shape.ofList (poolOutSpatialPad inSpatial kernel stride padding).toList

/-- Output shape for channels-first N-D pooling with symmetric padding (channels preserved). -/
def poolMultiOutShapePad {d : Nat} (inC : Nat) (inSpatial kernel stride padding : Vector Nat d)
    : Shape :=
  Shape.ofList (inC :: (poolOutSpatialPad inSpatial kernel stride padding).toList)

namespace Pooling
namespace Internal

/-- Choose the input-space pivot whose scaled value is maximal. -/
def smoothMaxPivotStep (beta current candidate : α) : α :=
  if beta > 0 then Max.max current candidate else Min.min current candidate

def tensorOfDims (dims : List Nat) (f : List Nat → α) : Tensor α (Shape.ofList dims) :=
  match dims with
  | [] => Tensor.scalar (f [])
  | _n :: ns =>
      Tensor.dim (fun i =>
        tensorOfDims ns (fun is => f (i.val :: is)))

def foldlIndices' {β : Type} (dims : List Nat) (init : β) (f : β → List Nat → β) : β :=
  match dims with
  | [] => f init []
  | n :: ns =>
      (List.range n).foldl (fun acc i =>
        foldlIndices' ns acc (fun acc' is => f acc' (i :: is))) init

def paddedCoords? (outIdxs winIdxs stride : List Nat) : Option (List Nat) :=
  match outIdxs, winIdxs, stride with
  | [], [], [] => some []
  | o :: os, w :: ws, s :: ss =>
      match paddedCoords? os ws ss with
      | some rest => some ((o * s + w) :: rest)
      | none => none
  | _, _, _ => none

def unpadCoords? (padded padding : List Nat) : Option (List Nat) :=
  match padded, padding with
  | [], [] => some []
  | x :: xs, p :: ps =>
      if _h : x < p then
        none
      else
        match unpadCoords? xs ps with
        | some rest => some ((x - p) :: rest)
        | none => none
  | _, _ => none

def coordsInBounds (idx dims : List Nat) : Bool :=
  match idx, dims with
  | [], [] => true
  | i :: is, d :: ds => decide (i < d) && coordsInBounds is ds
  | _, _ => false

/--
Input lookup for average/smooth pooling.

For average-style pooling, padded cells contribute numeric zero and are still counted by the
denominator chosen by the surrounding pooling spec. We keep this separate from
`getPaddedMaxInputVal?`, where padded cells must be ignored rather than treated as zero.
-/
def getPaddedAverageInputVal
    {d : Nat} {inSpatial : Vector Nat d}
    (input : Tensor α (Shape.ofList inSpatial.toList))
    (outIdxs winIdxs : List Nat)
    (stride padding : List Nat) : α :=
  match paddedCoords? outIdxs winIdxs stride with
  | none => 0
  | some padded =>
      match unpadCoords? padded padding with
      | none => 0
      | some orig => getAtOrZero input orig

/--
Input lookup for hard max-pooling.

Unlike average pooling, max pooling does not insert numeric zero for an individual padded cell:
PyTorch's valid max-pool configurations behave as though those cells were `-∞`. TorchLean keeps
the spec scalar-polymorphic by returning `none` for padded coordinates and ignoring them in the
max fold. `poolOutSpatialPad` rejects empty input axes, empty kernels, and padding beyond PyTorch's
half-kernel restriction, so every emitted output window contains at least one input coordinate.
-/
def getPaddedMaxInputVal?
    {d : Nat} {inSpatial : Vector Nat d}
    (input : Tensor α (Shape.ofList inSpatial.toList))
    (outIdxs winIdxs : List Nat)
    (stride padding : List Nat) : Option α :=
  match paddedCoords? outIdxs winIdxs stride with
  | none => none
  | some padded =>
      match unpadCoords? padded padding with
      | none => none
      | some orig =>
          if coordsInBounds orig inSpatial.toList then
            some (getAtOrZero input orig)
          else
            none

def kernelProd (kernel : List Nat) : Nat :=
  kernel.foldl (fun acc k => acc * k) 1

def maxPoolValue
    {d : Nat} {inSpatial : Vector Nat d}
    (input : Tensor α (Shape.ofList inSpatial.toList))
    (outIdxs : List Nat)
    (kernel stride padding : List Nat) : α :=
  let best? := foldlIndices' kernel none (fun best winIdxs =>
    match getPaddedMaxInputVal? (d := d) (inSpatial := inSpatial)
      (input := input) (outIdxs := outIdxs) (winIdxs := winIdxs) (stride := stride)
      (padding := padding), best with
    | none, _ => best
    | some v, none => some v
    | some v, some b => if v > b then some v else best)
  -- The default makes this helper total; valid pooling shapes always select an input value.
  best?.getD 0

/--
Selected-branch tangent for one hard max-pooling window.

The tangent follows the same winner selected by `maxPoolValue`. At a tie this is a deterministic
generalized-derivative convention, not the mathematical directional derivative of `max`.
-/
def maxPoolSelectedTangentValue
    {d : Nat} {inSpatial : Vector Nat d}
    (input tangent : Tensor α (Shape.ofList inSpatial.toList))
    (outIdxs : List Nat)
    (kernel stride padding : List Nat) : α :=
  let best? := foldlIndices' kernel none (fun best winIdxs =>
    match getPaddedMaxInputVal? (d := d) (inSpatial := inSpatial)
      (input := input) (outIdxs := outIdxs) (winIdxs := winIdxs) (stride := stride)
      (padding := padding), best with
    | none, _ => best
    | some v, none => some (winIdxs, v)
    | some v, some (_, b) => if v > b then some (winIdxs, v) else best)
  match best? with
  | none => 0
  | some (bestWin, _) =>
      match paddedCoords? outIdxs bestWin stride with
      | none => 0
      | some padded =>
          match unpadCoords? padded padding with
          | none => 0
          | some orig =>
              if coordsInBounds orig inSpatial.toList then
                getAtOrZero tangent orig
              else
                0

def avgPoolValue
    {d : Nat} {inSpatial : Vector Nat d}
    (input : Tensor α (Shape.ofList inSpatial.toList))
    (outIdxs : List Nat)
    (kernel stride padding : List Nat) : α :=
  let sum := foldlIndices' kernel (0 : α) (fun acc winIdxs =>
    acc + getPaddedAverageInputVal (d := d) (inSpatial := inSpatial)
      (input := input) (outIdxs := outIdxs) (winIdxs := winIdxs) (stride := stride)
      (padding := padding))
  sum / (kernelProd kernel : α)

/-- Input-space pivot whose scaled value is maximal over a nonempty N-D pooling window. -/
def smoothMaxPoolPivot
    {d : Nat} {inSpatial : Vector Nat d}
    (beta : α)
    (input : Tensor α (Shape.ofList inSpatial.toList))
    (outIdxs : List Nat)
    (kernel stride padding : List Nat) : α :=
  let firstIdx := List.replicate d 0
  let first := getPaddedAverageInputVal (d := d) (inSpatial := inSpatial)
    (input := input) (outIdxs := outIdxs) (winIdxs := firstIdx) (stride := stride)
    (padding := padding)
  foldlIndices' kernel first (fun pivot winIdxs =>
    let x := getPaddedAverageInputVal (d := d) (inSpatial := inSpatial)
      (input := input) (outIdxs := outIdxs) (winIdxs := winIdxs) (stride := stride)
      (padding := padding)
    smoothMaxPivotStep beta pivot x)

/-- Evaluate one N-D smooth-max window with the sign-aware input-space pivot. -/
def smoothMaxPoolValue
    {d : Nat} {inSpatial : Vector Nat d}
    (beta : α)
    (input : Tensor α (Shape.ofList inSpatial.toList))
    (outIdxs : List Nat)
    (kernel stride padding : List Nat) : α :=
  let pivot := smoothMaxPoolPivot (d := d) (inSpatial := inSpatial) beta input outIdxs
    kernel stride padding
  let sumExp := foldlIndices' kernel (0 : α) (fun acc winIdxs =>
    let x := getPaddedAverageInputVal (d := d) (inSpatial := inSpatial)
      (input := input) (outIdxs := outIdxs) (winIdxs := winIdxs) (stride := stride)
      (padding := padding)
    acc + MathFunctions.exp (beta * (x - pivot)))
  let invTemp : α := 1 / beta
  pivot + MathFunctions.log sumExp * invTemp

/--
Directional derivative of the smooth log-sum-exp pooling value.

For `y = beta⁻¹ log Σ exp(beta*xᵢ)`, the directional derivative is
`Σ softmax(beta*xᵢ) * dxᵢ`, using the same zero-padding and stable input-pivot convention as
`smoothMaxPoolValue`.
-/
def smoothMaxPoolJvpValue
    {d : Nat} {inSpatial : Vector Nat d}
    (beta : α)
    (input tangent : Tensor α (Shape.ofList inSpatial.toList))
    (outIdxs : List Nat)
    (kernel stride padding : List Nat) : α :=
  let pivot := smoothMaxPoolPivot (d := d) (inSpatial := inSpatial) beta input outIdxs
    kernel stride padding
  let sumExp := foldlIndices' kernel (0 : α) (fun acc winIdxs =>
    let x := getPaddedAverageInputVal (d := d) (inSpatial := inSpatial)
      (input := input) (outIdxs := outIdxs) (winIdxs := winIdxs) (stride := stride)
      (padding := padding)
    acc + MathFunctions.exp (beta * (x - pivot)))
  foldlIndices' kernel (0 : α) (fun acc winIdxs =>
    let x := getPaddedAverageInputVal (d := d) (inSpatial := inSpatial)
      (input := input) (outIdxs := outIdxs) (winIdxs := winIdxs) (stride := stride)
      (padding := padding)
    let dx := getPaddedAverageInputVal (d := d) (inSpatial := inSpatial)
      (input := tangent) (outIdxs := outIdxs) (winIdxs := winIdxs) (stride := stride)
      (padding := padding)
    acc + (MathFunctions.exp (beta * (x - pivot)) / sumExp) * dx)

end Internal
end Pooling

/-!
### Forward (single-channel spatial tensor)
-/

/-- N-D max pooling on a spatial tensor (no explicit channel axis). -/
def maxPoolSpatialSpec
    {d : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (_layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (input : Tensor α (Shape.ofList inSpatial.toList)) :
    Tensor α (Shape.ofList (poolOutSpatialPad inSpatial kernel stride padding).toList) :=

  let outSpatial := poolOutSpatialPad inSpatial kernel stride padding
  let kernelL := kernel.toList
  let strideL := stride.toList
  let paddingL := padding.toList

  Pooling.Internal.tensorOfDims outSpatial.toList (fun outIdxs =>
    Pooling.Internal.maxPoolValue (d := d) (inSpatial := inSpatial)
      (input := input) (outIdxs := outIdxs)
      (kernel := kernelL) (stride := strideL) (padding := paddingL))

/--
Selected-branch linearization for N-D hard max-pooling on a spatial tensor.

Away from ties this is the ordinary JVP. At ties it follows the first row-major primal maximizer,
matching the VJP convention but not claiming an analytic directional derivative.
-/
def maxPoolSpatialLinearizationSpec
    {d : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (_layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (input tangent : Tensor α (Shape.ofList inSpatial.toList)) :
    Tensor α (Shape.ofList (poolOutSpatialPad inSpatial kernel stride padding).toList) :=

  let outSpatial := poolOutSpatialPad inSpatial kernel stride padding
  let kernelL := kernel.toList
  let strideL := stride.toList
  let paddingL := padding.toList

  Pooling.Internal.tensorOfDims outSpatial.toList (fun outIdxs =>
    Pooling.Internal.maxPoolSelectedTangentValue (d := d) (inSpatial := inSpatial)
      (input := input) (tangent := tangent) (outIdxs := outIdxs)
      (kernel := kernelL) (stride := strideL) (padding := paddingL))

/-- N-D average pooling on a spatial tensor (no explicit channel axis). -/
def avgPoolSpatialSpec
    {d : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (_layer : AvgPoolSpec d kernel stride padding hKernel hStride)
    (input : Tensor α (Shape.ofList inSpatial.toList)) :
    Tensor α (Shape.ofList (poolOutSpatialPad inSpatial kernel stride padding).toList) :=

  let outSpatial := poolOutSpatialPad inSpatial kernel stride padding
  let kernelL := kernel.toList
  let strideL := stride.toList
  let paddingL := padding.toList

  Pooling.Internal.tensorOfDims outSpatial.toList (fun outIdxs =>
    Pooling.Internal.avgPoolValue (d := d) (inSpatial := inSpatial)
      (input := input) (outIdxs := outIdxs)
      (kernel := kernelL) (stride := strideL) (padding := paddingL))

/-!
### Backward (single-channel spatial tensor)

These are the VJPs of the forward pooling specs above.

Conventions:
- For max pooling, ties are broken by **first occurrence** in row-major order (same as the 2D spec).
- For max pooling, padded cells are ignored, modeling PyTorch's `-∞` padding without requiring a
  scalar-polymorphic infinity constant.
- For average pooling, gradients are evenly distributed across the full kernel window
  (`count_include_pad=true` behavior when padding is present).
-/

/--
Backward/VJP for `max_pool_spatial_spec`.

Each output gradient is propagated to the argmax location in the corresponding input window.
Ties keep the first position in row-major order.
-/
def maxPoolSpatialBackwardSpec
    {d : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (input : Tensor α (Shape.ofList inSpatial.toList))
    (grad_output :
      Tensor α (Shape.ofList (poolOutSpatialPad inSpatial kernel stride padding).toList)) :
    Tensor α (Shape.ofList inSpatial.toList) :=

  let _ := layer
  let outSpatial := poolOutSpatialPad inSpatial kernel stride padding
  let outDims := outSpatial.toList
  let kernelL := kernel.toList
  let strideL := stride.toList
  let paddingL := padding.toList

  let grad_init : Tensor α (Shape.ofList inSpatial.toList) :=
    Pooling.Internal.tensorOfDims inSpatial.toList (fun _ => 0)

  Pooling.Internal.foldlIndices' outDims grad_init (fun acc_grad outIdxs =>
    let best? : Option (List Nat × α) :=
      Pooling.Internal.foldlIndices' kernelL none (fun best winIdxs =>
        match Pooling.Internal.getPaddedMaxInputVal? (d := d) (inSpatial := inSpatial)
          (input := input) (outIdxs := outIdxs) (winIdxs := winIdxs) (stride := strideL)
          (padding := paddingL), best with
        | none, _ => best
        | some curr, none => some (winIdxs, curr)
        | some curr, some (_, bestVal) =>
            if curr > bestVal then some (winIdxs, curr) else best)
    let gOut : α := getAtOrZero grad_output outIdxs
    match best? with
    | none => acc_grad
    | some (bestWin, _) =>
        match Pooling.Internal.paddedCoords? outIdxs bestWin strideL with
        | none => acc_grad
        | some padded =>
            match Pooling.Internal.unpadCoords? padded paddingL with
            | none => acc_grad
            | some orig =>
                if Pooling.Internal.coordsInBounds orig inSpatial.toList then
                  let current : α := getAtOrZero acc_grad orig
                  updateTensorSpec acc_grad orig (current + gOut)
                else
                  acc_grad)

/--
Backward/VJP for `avg_pool_spatial_spec` (single-channel).

Each output gradient is evenly distributed across its kernel window.
-/
def avgPoolSpatialBackwardSpec
    {d : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (_layer : AvgPoolSpec d kernel stride padding hKernel hStride)
    (grad_output :
      Tensor α (Shape.ofList (poolOutSpatialPad inSpatial kernel stride padding).toList)) :
    Tensor α (Shape.ofList inSpatial.toList) :=

  let outSpatial := poolOutSpatialPad inSpatial kernel stride padding
  let outDims := outSpatial.toList
  let kernelL := kernel.toList
  let strideL := stride.toList
  let paddingL := padding.toList
  let poolSize : α := (Pooling.Internal.kernelProd kernelL : Nat)

  let grad_init : Tensor α (Shape.ofList inSpatial.toList) :=
    Pooling.Internal.tensorOfDims inSpatial.toList (fun _ => 0)

  Pooling.Internal.foldlIndices' outDims grad_init (fun acc_grad outIdxs =>
    let gOut : α := getAtOrZero grad_output outIdxs
    Pooling.Internal.foldlIndices' kernelL acc_grad (fun acc winIdxs =>
      match Pooling.Internal.paddedCoords? outIdxs winIdxs strideL with
      | none => acc
      | some padded =>
          match Pooling.Internal.unpadCoords? padded paddingL with
          | none => acc
          | some orig =>
              let current : α := getAtOrZero acc orig
              updateTensorSpec acc orig (current + gOut / poolSize)))

/-!
### Forward (channels-first: `C × spatial...`)
-/

/-- N-D max pooling on a channels-first tensor: shape `[C] ++ spatial`. -/
def maxPoolSpec
    {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (input : Tensor α (Shape.ofList (C :: inSpatial.toList))) :
    Tensor α (Shape.ofList (C :: (poolOutSpatialPad inSpatial kernel stride padding).toList)) :=

  Tensor.dim (fun c =>
    maxPoolSpatialSpec (α := α) (d := d) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding)
      layer (getAtSpec input c))

/-- N-D hard max-pool selected-branch linearization, applied channel-wise. -/
def maxPoolLinearizationSpec
    {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (input tangent : Tensor α (Shape.ofList (C :: inSpatial.toList))) :
    Tensor α (Shape.ofList (C :: (poolOutSpatialPad inSpatial kernel stride padding).toList)) :=

  Tensor.dim (fun c =>
    maxPoolSpatialLinearizationSpec (α := α) (d := d) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding)
      layer (getAtSpec input c) (getAtSpec tangent c))

/-- N-D average pooling on a channels-first tensor: shape `[C] ++ spatial`. -/
def avgPoolSpec
    {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (layer : AvgPoolSpec d kernel stride padding hKernel hStride)
    (input : Tensor α (Shape.ofList (C :: inSpatial.toList))) :
    Tensor α (Shape.ofList (C :: (poolOutSpatialPad inSpatial kernel stride padding).toList)) :=

  Tensor.dim (fun c =>
    avgPoolSpatialSpec (α := α) (d := d) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding)
      layer (getAtSpec input c))

/-!
### Backward (channels-first: `C × spatial...`)
-/

/-- Multi-channel VJP for `max_pool_spec` (apply spatial backward per channel). -/
def maxPoolBackwardSpec
    {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (input : Tensor α (Shape.ofList (C :: inSpatial.toList)))
    (grad_output :
      Tensor α (Shape.ofList (C :: (poolOutSpatialPad inSpatial kernel stride padding).toList))) :
    Tensor α (Shape.ofList (C :: inSpatial.toList)) :=

  Tensor.dim (fun c =>
    maxPoolSpatialBackwardSpec (α := α) (d := d) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding)
      layer (getAtSpec input c) (getAtSpec grad_output c))

/-- Multi-channel VJP for `avg_pool_spec` (apply spatial backward per channel). -/
def avgPoolBackwardSpec
    {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (layer : AvgPoolSpec d kernel stride padding hKernel hStride)
    (grad_output :
      Tensor α (Shape.ofList (C :: (poolOutSpatialPad inSpatial kernel stride padding).toList))) :
    Tensor α (Shape.ofList (C :: inSpatial.toList)) :=

  Tensor.dim (fun _c =>
    avgPoolSpatialBackwardSpec (α := α) (d := d) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding)
      layer (getAtSpec grad_output _c))

/-!
### Smooth max pooling (log-sum-exp surrogate)
-/

/-- Smooth log-sum-exp max pooling on a spatial tensor (no explicit channel axis). -/
def smoothMaxPoolSpatialSpec
    {d : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (_layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (beta : α)
    (input : Tensor α (Shape.ofList inSpatial.toList)) :
    Tensor α (Shape.ofList (poolOutSpatialPad inSpatial kernel stride padding).toList) :=

  let outSpatial := poolOutSpatialPad inSpatial kernel stride padding
  let kernelL := kernel.toList
  let strideL := stride.toList
  let paddingL := padding.toList

  Pooling.Internal.tensorOfDims outSpatial.toList (fun outIdxs =>
    Pooling.Internal.smoothMaxPoolValue (d := d) (inSpatial := inSpatial) (beta := beta)
      (input := input) (outIdxs := outIdxs)
      (kernel := kernelL) (stride := strideL) (padding := paddingL))

/--
Forward-mode JVP for N-D smooth max-pooling on a spatial tensor.

For the log-sum-exp surrogate this is the softmax-weighted sum of the input tangent over each
window. It is the forward-mode counterpart of `smoothMaxPoolSpatialBackwardSpec`.
-/
def smoothMaxPoolSpatialJvpSpec
    {d : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (_layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (beta : α)
    (input tangent : Tensor α (Shape.ofList inSpatial.toList)) :
    Tensor α (Shape.ofList (poolOutSpatialPad inSpatial kernel stride padding).toList) :=

  let outSpatial := poolOutSpatialPad inSpatial kernel stride padding
  let kernelL := kernel.toList
  let strideL := stride.toList
  let paddingL := padding.toList

  Pooling.Internal.tensorOfDims outSpatial.toList (fun outIdxs =>
    Pooling.Internal.smoothMaxPoolJvpValue (d := d) (inSpatial := inSpatial) (beta := beta)
      (input := input) (tangent := tangent) (outIdxs := outIdxs)
      (kernel := kernelL) (stride := strideL) (padding := paddingL))

/-- Smooth log-sum-exp max pooling on a channels-first tensor (channel-wise application). -/
def smoothMaxPoolSpec
    {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (beta : α)
    (input : Tensor α (Shape.ofList (C :: inSpatial.toList))) :
    Tensor α (Shape.ofList (C :: (poolOutSpatialPad inSpatial kernel stride padding).toList)) :=

  Tensor.dim (fun c =>
    smoothMaxPoolSpatialSpec (α := α) (d := d) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding)
      layer beta (getAtSpec input c))

/-- N-D smooth max-pool JVP on a channels-first tensor (channel-wise application). -/
def smoothMaxPoolJvpSpec
    {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (beta : α)
    (input tangent : Tensor α (Shape.ofList (C :: inSpatial.toList))) :
    Tensor α (Shape.ofList (C :: (poolOutSpatialPad inSpatial kernel stride padding).toList)) :=

  Tensor.dim (fun c =>
    smoothMaxPoolSpatialJvpSpec (α := α) (d := d) (inSpatial := inSpatial)
      (kernel := kernel) (stride := stride) (padding := padding)
      layer (beta := beta)
      (input := getAtSpec input c) (tangent := getAtSpec tangent c))

/-!
### Smooth max pooling backward
-/

/--
Backward/VJP for `smooth_max_pool_spatial_spec` (log-sum-exp surrogate).

For a window `x₁,…,xₙ`, the surrogate is:

`y = (1/beta) * log(∑ exp(beta*xᵢ))`

and the VJP distributes upstream gradient proportionally to `exp(beta*xᵢ)`.
The implementation evaluates the equivalent max/min-shifted weights so large finite inputs do not
overflow before normalization.
-/
def smoothMaxPoolSpatialBackwardSpec
    {d : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (_layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (beta : α)
    (input : Tensor α (Shape.ofList inSpatial.toList))
    (grad_output :
      Tensor α (Shape.ofList (poolOutSpatialPad inSpatial kernel stride padding).toList)) :
    Tensor α (Shape.ofList inSpatial.toList) :=

  let outSpatial := poolOutSpatialPad inSpatial kernel stride padding
  let outDims := outSpatial.toList
  let kernelL := kernel.toList
  let strideL := stride.toList
  let paddingL := padding.toList
  let coeff : α := 1

  let grad_init : Tensor α (Shape.ofList inSpatial.toList) :=
    Pooling.Internal.tensorOfDims inSpatial.toList (fun _ => 0)

  Pooling.Internal.foldlIndices' outDims grad_init (fun acc_grad outIdxs =>
    let pivot : α :=
      Pooling.Internal.smoothMaxPoolPivot (d := d) (inSpatial := inSpatial) beta input outIdxs
        kernelL strideL paddingL
    let sumExp : α :=
      Pooling.Internal.foldlIndices' kernelL (0 : α) (fun acc winIdxs =>
        let x :=
          Pooling.Internal.getPaddedAverageInputVal (d := d) (inSpatial := inSpatial)
            (input := input) (outIdxs := outIdxs) (winIdxs := winIdxs)
            (stride := strideL) (padding := paddingL)
        acc + MathFunctions.exp (beta * (x - pivot)))
    let gOut : α := getAtOrZero grad_output outIdxs
    Pooling.Internal.foldlIndices' kernelL acc_grad (fun acc winIdxs =>
      match Pooling.Internal.paddedCoords? outIdxs winIdxs strideL with
      | none => acc
      | some padded =>
          match Pooling.Internal.unpadCoords? padded paddingL with
          | none => acc
          | some orig =>
              let x :=
                Pooling.Internal.getPaddedAverageInputVal (d := d) (inSpatial := inSpatial)
                  (input := input) (outIdxs := outIdxs) (winIdxs := winIdxs)
                  (stride := strideL) (padding := paddingL)
              let expVal := MathFunctions.exp (beta * (x - pivot))
              let w : α := coeff * (expVal / sumExp)
              let current : α := getAtOrZero acc orig
              updateTensorSpec acc orig (current + gOut * w)))

/-- Multi-channel VJP for `smooth_max_pool_spec` (apply spatial backward per channel). -/
def smoothMaxPoolBackwardSpec
    {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (layer : MaxPoolSpec d kernel stride padding hKernel hStride)
    (beta : α)
    (input : Tensor α (Shape.ofList (C :: inSpatial.toList)))
    (grad_output :
      Tensor α (Shape.ofList (C :: (poolOutSpatialPad inSpatial kernel stride padding).toList))) :
    Tensor α (Shape.ofList (C :: inSpatial.toList)) :=

  Tensor.dim (fun c =>
    smoothMaxPoolSpatialBackwardSpec (α := α) (d := d) (inSpatial := inSpatial)
      (kernel := kernel) (stride := stride) (padding := padding)
      layer (beta := beta)
      (input := getAtSpec input c) (grad_output := getAtSpec grad_output c))
end Spec
