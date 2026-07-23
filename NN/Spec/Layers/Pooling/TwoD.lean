/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape
public import NN.Spec.Layers.Utils

@[expose] public section


namespace Spec
open Tensor
open Spec (Image MultiChannelImage getValueAtPosition extractWindow)

variable {α : Type} [Context α]

/-!
# 2D Pooling

Single-channel and channels-first 2D max, average, adaptive, and smooth-max pooling specs.
-/


/--
MaxPool2d configuration.

The spec uses a fixed kernel `(kH,kW)` and a single stride value (applied to both height and width).
We require `kH ≠ 0`, `kW ≠ 0`, and `stride ≠ 0` so windows are nonempty and the output-shape
arithmetic is well-defined.

PyTorch analogy: `F.max_pool2d(x, kernel_size=(kH,kW), stride=stride)`.
-/
structure MaxPool2DSpec (kH kW stride: ℕ) (h1 : kH ≠ 0) (h2 : kW ≠ 0) (hStride : stride ≠ 0) where
  /-- kernel Height. -/
  kernelHeight : ℕ := kH
  /-- kernel Width. -/
  kernelWidth : ℕ := kW
  /-- Stride. -/
  stride : ℕ := stride

/--
AvgPool2d configuration.

We treat the pooling window as `kH*kW` elements and divide by that count.
This corresponds to PyTorch's default behavior when no padding is present.
-/
structure AvgPool2DSpec (kH kW stride : ℕ) (h1 : kH ≠ 0) (h2 : kW ≠ 0) (hStride : stride ≠ 0) where
  /-- kernel Height. -/
  kernelHeight : ℕ := kH
  /-- kernel Width. -/
  kernelWidth : ℕ := kW
  /-- Stride. -/
  stride : ℕ := stride

/--
Output shape for a 2D pooling op (single-channel) with no padding.

This uses the standard "valid" pooling formula:

`outH = floor((inH - kH)/stride) + 1`, `outW = floor((inW - kW)/stride) + 1`.

PyTorch analogy: `ceil_mode=false` with no padding.
-/
def pool2dOutShape (inH inW kH kW stride : ℕ) : Shape :=
  let outH := Shape.slidingWindowOutDim inH kH stride 0
  let outW := Shape.slidingWindowOutDim inW kW stride 0
  .dim outH (.dim outW .scalar)

/-- Output shape for multi-channel 2D pooling (channels preserved). -/
def pool2dMultiOutShape (inC inH inW kH kW stride : ℕ) : Shape :=
  let outH := Shape.slidingWindowOutDim inH kH stride 0
  let outW := Shape.slidingWindowOutDim inW kW stride 0
  .dim inC (.dim outH (.dim outW .scalar))

/--
Output shape for a 2D pooling op (single-channel) with symmetric padding.

`padding` means we use the usual PyTorch output-size formula for an input extended by `padding`
cells on each side. Hard max-pooling ignores padded cells (the PyTorch `-∞` convention), while
average-pooling below explicitly includes padded zeros.
-/
def pool2dOutShapePad (inH inW kH kW stride padding : ℕ) : Shape :=
  let outH := Shape.slidingWindowOutDim inH kH stride padding
  let outW := Shape.slidingWindowOutDim inW kW stride padding
  .dim outH (.dim outW .scalar)

/-- Output shape for multi-channel 2D pooling with symmetric padding (channels preserved). -/
def pool2dMultiOutShapePad (inC inH inW kH kW stride padding : ℕ) : Shape :=
  let outH := Shape.slidingWindowOutDim inH kH stride padding
  let outW := Shape.slidingWindowOutDim inW kW stride padding
  .dim inC (.dim outH (.dim outW .scalar))

/-!
## Smooth max pooling

`max_pool2d_spec` uses `max` and is non-differentiable (ties and kink points).

For proofs that need everywhere differentiability, we provide a smooth surrogate
based on log-sum-exp over each pooling window:

  `smooth_max(x₁,…,xₙ) = (1 / β) * log (∑ exp (β * xᵢ))`

This is the standard log-sum-exp surrogate and is intended for `β ≠ 0`.
-/

namespace Private

/--
Choose the input-space pivot whose scaled value `beta * pivot` is maximal.

Selecting a maximum for positive `beta` and a minimum for negative `beta` avoids forming
`beta * x` before the log-sum-exp shift, which matters when both operands are large finite
floating-point values.
-/
def smoothMaxPivotStep (beta current candidate : α) : α :=
  if beta > 0 then Max.max current candidate else Min.min current candidate

/-- Input-space pivot for a nonempty two-dimensional smooth-max window. -/
def smoothMaxPool2dPivot {kH kW : Nat} (hH : kH ≠ 0) (hW : kW ≠ 0)
    (beta : α) (window : Tensor α (Shape.dim kH (Shape.dim kW Shape.scalar))) : α :=
  let i0 : Fin kH := ⟨0, Nat.pos_of_ne_zero hH⟩
  let j0 : Fin kW := ⟨0, Nat.pos_of_ne_zero hW⟩
  let first := Tensor.toScalar (getAtSpec (getAtSpec window i0) j0)
  (List.finRange kH).foldl (fun rowPivot i =>
    (List.finRange kW).foldl (fun pivot j =>
      smoothMaxPivotStep beta pivot (Tensor.toScalar (getAtSpec (getAtSpec window i) j)))
      rowPivot) first

end Private

/--
Smooth max-pooling (single-channel) using a log-sum-exp surrogate.

This is useful in proof settings that want a differentiable alternative to `max_pool2d_spec`.
For large positive `beta`, the output approaches hard max pooling; for large negative `beta`, it
approaches hard min pooling. The implementation selects an unscaled input-space pivot—a maximum
for positive `beta` and a minimum for negative `beta`—so it need not form `beta * x` before the
log-sum-exp shift.
-/
def smoothMaxPool2dSpec {kH kW inH inW stride : ℕ} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  {hStride : stride ≠ 0}
  (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
  (beta : α)
  (input : Image inH inW α) :
  Image (Shape.slidingWindowOutDim inH kH stride 0) (Shape.slidingWindowOutDim inW kW stride 0) α :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      let window := extractWindow kW kH input (i.val * layer.stride) (j.val * layer.stride)
      let pivot := Private.smoothMaxPool2dPivot h1 h2 beta window
      let expWindow :=
        mapSpec (s := Shape.dim kH (Shape.dim kW Shape.scalar))
          (fun x => MathFunctions.exp (beta * (x - pivot))) window
      have instH : Shape.valid_axis_inst 0 (Shape.dim kH (Shape.dim kW Shape.scalar)) :=
        Shape.validAxisInstZeroAlt h1
      have instW : Shape.valid_axis_inst 0 (Shape.dim kW Shape.scalar) :=
        Shape.validAxisInstZeroAlt h2
      let sumH := reduceSumAuto 0 expWindow
      have hcast : (Shape.dim kW Shape.scalar) = shapeAfterSum (Shape.dim kH (Shape.dim kW
        Shape.scalar)) 0 := by
        simp [shapeAfterSum]
      let sumH' := tensorCast (Shape.dim kW Shape.scalar) hcast.symm sumH
      let sumAll := reduceSumAuto 0 sumH'
      match sumAll with
      | Tensor.scalar s =>
          let invTemp : α := 1 / beta
          Tensor.scalar (pivot + MathFunctions.log s * invTemp)))

/-- Smooth max-pooling (multi-channel): apply `smooth_max_pool2d_spec` per channel. -/
def smoothMaxPool2dMultiSpec {kH kW inH inW inC stride : ℕ} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  {hStride : stride ≠ 0}
  (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
  (beta : α)
  (input : MultiChannelImage inC inH inW α) :
  MultiChannelImage inC (Shape.slidingWindowOutDim inH kH stride 0) (Shape.slidingWindowOutDim inW kW stride 0) α :=
  Tensor.dim (fun c => smoothMaxPool2dSpec (layer := layer) (beta := beta) (getAtSpec input c))

/--
Forward-mode JVP for smooth max-pooling (single-channel).

For each pooling window this is the differential of the log-sum-exp surrogate,
`Σᵢ softmax(beta*xᵢ) * dxᵢ`. This mirrors the VJP weights below but pushes an input tangent
forward instead of pulling an output cotangent backward.
-/
def smoothMaxPool2dJvpSpec {kH kW inH inW stride : ℕ} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  {hStride : stride ≠ 0}
  (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
  (beta : α)
  (input tangent : Image inH inW α) :
  Image (Shape.slidingWindowOutDim inH kH stride 0) (Shape.slidingWindowOutDim inW kW stride 0) α :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      let window := extractWindow kW kH input (i.val * layer.stride) (j.val * layer.stride)
      let tangentWindow := extractWindow kW kH tangent (i.val * layer.stride) (j.val * layer.stride)
      let pivot := Private.smoothMaxPool2dPivot h1 h2 beta window
      let expWindow :=
        mapSpec (s := Shape.dim kH (Shape.dim kW Shape.scalar))
          (fun x => MathFunctions.exp (beta * (x - pivot))) window
      let weighted :=
        map2Spec (fun e dx => e * dx) expWindow tangentWindow
      have instH : Shape.valid_axis_inst 0 (Shape.dim kH (Shape.dim kW Shape.scalar)) :=
        Shape.validAxisInstZeroAlt h1
      have instW : Shape.valid_axis_inst 0 (Shape.dim kW Shape.scalar) :=
        Shape.validAxisInstZeroAlt h2
      let sumExpH := reduceSumAuto 0 expWindow
      let sumWeightedH := reduceSumAuto 0 weighted
      have hcast : (Shape.dim kW Shape.scalar) = shapeAfterSum
          (Shape.dim kH (Shape.dim kW Shape.scalar)) 0 := by
        simp [shapeAfterSum]
      let sumExpH' := tensorCast (Shape.dim kW Shape.scalar) hcast.symm sumExpH
      let sumWeightedH' := tensorCast (Shape.dim kW Shape.scalar) hcast.symm sumWeightedH
      let sumExp := reduceSumAuto 0 sumExpH'
      let sumWeighted := reduceSumAuto 0 sumWeightedH'
      match sumExp, sumWeighted with
      | Tensor.scalar denom, Tensor.scalar num => Tensor.scalar (num / denom)))

/-- Multi-channel JVP for smooth max-pooling (channel-wise application). -/
def smoothMaxPool2dMultiJvpSpec {kH kW inH inW inC stride : ℕ} {h1 : kH ≠ 0}
  {h2 : kW ≠ 0} {hStride : stride ≠ 0}
  (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
  (beta : α)
  (input tangent : MultiChannelImage inC inH inW α) :
  MultiChannelImage inC (Shape.slidingWindowOutDim inH kH stride 0) (Shape.slidingWindowOutDim inW kW stride 0) α :=
  Tensor.dim (fun c =>
    smoothMaxPool2dJvpSpec (layer := layer) (beta := beta)
      (input := getAtSpec input c) (tangent := getAtSpec tangent c))

/--
MaxPool2d forward pass (single-channel).

This takes the maximum over each `kH×kW` window sampled with the given stride.
The return type encodes the standard output spatial size formula.
-/
def maxPool2dSpec {kH kW inH inW stride : ℕ} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  {hStride : stride ≠ 0}
  (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
  (input : Image inH inW α) :
  Image (Shape.slidingWindowOutDim inH kH stride 0) (Shape.slidingWindowOutDim inW kW stride 0) α :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      let window := extractWindow kW kH input (i.val * layer.stride) (j.val * layer.stride)
      have inst : Shape.valid_axis_inst 0 (Shape.dim kH (Shape.dim kW Shape.scalar)) :=
              Shape.validAxisInstZeroAlt h1
      let window_max := reduceMaxAuto 0 window
      have h1_eq : ((Shape.dim kW Shape.scalar)) =
        (shapeAfterSum (Shape.dim kH (Shape.dim kW Shape.scalar)) 0) := by
        simp [shapeAfterSum]
      have inst2 : Shape.valid_axis_inst 0 (Shape.dim kW Shape.scalar) :=
        Shape.validAxisInstZeroAlt h2
      let window_max' := tensorCast ((Shape.dim kW Shape.scalar)) h1_eq.symm window_max
      let window_max_2 := reduceMaxAuto 0 window_max'
      window_max_2))

-- Multi-channel max pooling forward pass
/-- MaxPool2d forward pass (multi-channel): apply `max_pool2d_spec` per channel. -/
def maxPool2dMultiSpec {kH kW inH inW inC stride : ℕ} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  {hStride : stride ≠ 0}
  (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
  (input : MultiChannelImage inC inH inW α) :
  MultiChannelImage inC (Shape.slidingWindowOutDim inH kH stride 0) (Shape.slidingWindowOutDim inW kW stride 0) α :=
  Tensor.dim (fun c => maxPool2dSpec layer (getAtSpec input c))

/--
Selected-branch linearization for hard max-pooling (single-channel).

Away from ties this is the JVP. At ties it reads the tangent at the first row-major primal
maximizer, matching `maxPool2dBackwardSpec` and PyTorch's index convention.
-/
def maxPool2dLinearizationSpec {kH kW inH inW stride : ℕ} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  {hStride : stride ≠ 0}
  (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
  (input tangent : Image inH inW α) :
  Image (Shape.slidingWindowOutDim inH kH stride 0) (Shape.slidingWindowOutDim inW kW stride 0) α :=
  Tensor.dim (fun out_i =>
    Tensor.dim (fun out_j =>
      let window := extractWindow kW kH input (out_i.val * layer.stride) (out_j.val * layer.stride)
      let max_pos : Fin kH × Fin kW :=
        (List.finRange kH).foldl (fun best_pos (di : Fin kH) =>
          (List.finRange kW).foldl (fun best_pos_inner (dj : Fin kW) =>
            let current_val := getAtSpec (getAtSpec window di) dj
            let best_val := getAtSpec (getAtSpec window best_pos.1) best_pos.2
            match current_val, best_val with
            | Tensor.scalar curr, Tensor.scalar best =>
                if curr > best then (di, dj) else best_pos_inner
          ) best_pos
        ) (⟨0, Nat.zero_lt_of_ne_zero h1⟩, ⟨0, Nat.zero_lt_of_ne_zero h2⟩)
      let inp_i := out_i.val * layer.stride + max_pos.1.val
      let inp_j := out_j.val * layer.stride + max_pos.2.val
      if h_inp_i : inp_i < inH then
        if h_inp_j : inp_j < inW then
          getAtSpec (getAtSpec tangent ⟨inp_i, h_inp_i⟩) ⟨inp_j, h_inp_j⟩
        else
          Tensor.scalar 0
      else
        Tensor.scalar 0))

/-- Multi-channel selected-branch linearization for hard max-pooling. -/
def maxPool2dMultiLinearizationSpec {kH kW inH inW inC stride : ℕ} {h1 : kH ≠ 0}
  {h2 : kW ≠ 0}
  {hStride : stride ≠ 0}
  (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
  (input tangent : MultiChannelImage inC inH inW α) :
  MultiChannelImage inC (Shape.slidingWindowOutDim inH kH stride 0) (Shape.slidingWindowOutDim inW kW stride 0) α :=
  Tensor.dim (fun c =>
    maxPool2dLinearizationSpec (layer := layer)
      (input := getAtSpec input c) (tangent := getAtSpec tangent c))

/--
AvgPool2d forward pass (single-channel).

We sum all values in the window and divide by `kH*kW`.
PyTorch analogy: `avg_pool2d` with `count_include_pad=true` only matters for *padded* pooling;
for the unpadded case it matches the usual definition.
-/
def avgPool2dSpec {kH kW inH inW stride : ℕ} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  {hStride : stride ≠ 0}
  (layer : AvgPool2DSpec kH kW stride h1 h2 hStride)
  (input : Image inH inW α) :
  Image (Shape.slidingWindowOutDim inH kH stride 0)
        (Shape.slidingWindowOutDim inW kW stride 0) α :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      let window := extractWindow kW kH input (i.val * layer.stride) (j.val * layer.stride)
      have inst : Shape.valid_axis_inst 0 (Shape.dim kH (Shape.dim kW Shape.scalar)) :=
              Shape.validAxisInstZeroAlt h1
      have inst2 : Shape.valid_axis_inst 0 (Shape.dim kW Shape.scalar) :=
              Shape.validAxisInstZeroAlt h2
      let sumHeight := reduceSumAuto 0 window
      have h1_eq : (Shape.dim kW Shape.scalar) =
        shapeAfterSum (Shape.dim kH (Shape.dim kW Shape.scalar)) 0 := by
        simp [shapeAfterSum]
      let sumHeight' := tensorCast (Shape.dim kW Shape.scalar) h1_eq.symm sumHeight
      let sumTotal := reduceSumAuto 0 sumHeight'
      divSpec sumTotal (Tensor.scalar (kH * kW))))

-- Multi-channel average pooling forward pass
/-- AvgPool2d forward pass (multi-channel): apply `avg_pool2d_spec` per channel. -/
def avgPool2dMultiSpec {kH kW inH inW inC stride : ℕ} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  {hStride : stride ≠ 0}
  (layer : AvgPool2DSpec kH kW stride h1 h2 hStride)
  (input : MultiChannelImage inC inH inW α) :
  MultiChannelImage inC (Shape.slidingWindowOutDim inH kH stride 0) (Shape.slidingWindowOutDim inW kW stride 0) α :=
  Tensor.dim (fun c => avgPool2dSpec (α := α) (h1 := h1) (h2 := h2) (layer := layer)
    (input := getAtSpec input c))

-- Adaptive pooling that pools to a specific output size
/-- Spec record for adaptive average pooling to a fixed output size. -/
structure AdaptiveAvgPool2DSpec (outH outW : ℕ) where
  /-- output Height. -/
  outputHeight : ℕ := outH
  /-- output Width. -/
  outputWidth : ℕ := outW

/-- Spec record for adaptive max pooling to a fixed output size. -/
structure AdaptiveMaxPool2DSpec (outH outW : ℕ) where
  /-- output Height. -/
  outputHeight : ℕ := outH
  /-- output Width. -/
  outputWidth : ℕ := outW

/-!
## Adaptive pooling

PyTorch defines adaptive pooling by partitioning the input into `out` bins.
For output index `i`, the pooling region is:

- `start = floor(i * in / out)`
- `end   = ceil((i+1) * in / out)`

This matters when `in` is not divisible by `out`: region sizes vary by at most 1.
-/

/-- Adaptive-pooling region start index: `floor(i * in / out)` (PyTorch definition). -/
def adaptiveStart (inSize outSize i : Nat) : Nat :=
  (i * inSize) / outSize

/-- Adaptive-pooling region end index: `ceil((i+1) * in / out)` (PyTorch definition). -/
def adaptiveEnd (inSize outSize i : Nat) : Nat :=
  -- `ceil(a/b) = (a + b - 1) / b` for naturals.
  ((i + 1) * inSize + outSize - 1) / outSize

-- Adaptive average pooling forward pass
/--
AdaptiveAvgPool2d forward pass.

Unlike fixed-kernel pooling, adaptive pooling chooses a window for each output position so that
the whole input is covered by `outH×outW` bins.
This follows the PyTorch start/end formula (see the section comment above).
-/
def adaptiveAvgPool2dSpec {inH inW inC : ℕ} (outH outW : ℕ)
  (_layer : AdaptiveAvgPool2DSpec outH outW)
  (input : MultiChannelImage inC inH inW α)
  (_h_inH : inH > 0 := by norm_num)
  (_h_inW : inW > 0 := by norm_num)
  (_h_outH : outH > 0 := by norm_num)
  (_h_outW : outW > 0 := by norm_num) :
  MultiChannelImage inC outH outW α :=

  Tensor.dim (fun c =>
    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        -- Pooling region for this output position (PyTorch definition).
        let start_i := adaptiveStart inH outH i.val
        let start_j := adaptiveStart inW outW j.val
        let end_i := adaptiveEnd inH outH i.val
        let end_j := adaptiveEnd inW outW j.val
        let actual_kH := end_i - start_i
        let actual_kW := end_j - start_j

        -- Extract and sum the region
        let region_sum :=
          (List.range actual_kH).foldl (fun acc_i di =>
            (List.range actual_kW).foldl (fun acc_j dj =>
              let pos_i := start_i + di
              let pos_j := start_j + dj
              if h_i : pos_i < inH then
                if h_j : pos_j < inW then
                  let val := getAtSpec (getAtSpec (getAtSpec input c) ⟨pos_i, h_i⟩) ⟨pos_j,
                    h_j⟩
                  match acc_j, val with
                  | Tensor.scalar acc, Tensor.scalar v => Tensor.scalar (acc + v)
                else acc_j
              else acc_j
            ) acc_i
          ) (Tensor.scalar 0)

        -- Divide by the actual region size
        divSpec region_sum (Tensor.scalar (actual_kH * actual_kW)))))

-- Adaptive max pooling forward pass
/--
AdaptiveMaxPool2d forward pass (same binning as adaptive avg, but with `max` instead of `mean`).

We intentionally do not use a numeric sentinel value to seed the max fold; we seed from the first
element of the region via `getValueAtPosition`. That keeps the spec meaningful across different
scalar backends.
-/
def adaptiveMaxPool2dSpec {inH inW inC : ℕ} (outH outW : ℕ)
  (_layer : AdaptiveMaxPool2DSpec outH outW)
  (input : MultiChannelImage inC inH inW α)
  (_h_inH : inH > 0 := by norm_num)
  (_h_inW : inW > 0 := by norm_num)
  (_h_outH : outH > 0 := by norm_num)
  (_h_outW : outW > 0 := by norm_num) :
  MultiChannelImage inC outH outW α :=

  Tensor.dim (fun c =>
    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        -- Pooling region for this output position (PyTorch definition).
        let start_i := adaptiveStart inH outH i.val
        let start_j := adaptiveStart inW outW j.val
        let end_i := adaptiveEnd inH outH i.val
        let end_j := adaptiveEnd inW outW j.val
        let actual_kH := end_i - start_i
        let actual_kW := end_j - start_j

        -- Find max in the region
        -- We seed the fold with the first element instead of using a finite sentinel.
        -- That choice works for arbitrary scalar types and scales.
        let init : Tensor α .scalar :=
          -- `getValueAtPosition` performs the bounds check for us, so we don't have to thread a
          -- proof that `start_i < inH` and `start_j < inW` through the code.
          -- Under the stated positivity assumptions this is always in-bounds.
          getValueAtPosition (getAtSpec input c) start_i start_j

        (List.range actual_kH).foldl (fun acc_i di =>
          (List.range actual_kW).foldl (fun acc_j dj =>
            let pos_i := start_i + di
            let pos_j := start_j + dj
            if h_i : pos_i < inH then
              if h_j : pos_j < inW then
                let val := getAtSpec (getAtSpec (getAtSpec input c) ⟨pos_i, h_i⟩) ⟨pos_j, h_j⟩
                match acc_j, val with
                | Tensor.scalar acc, Tensor.scalar v => Tensor.scalar (max acc v)
              else acc_j
            else acc_j
          ) acc_i
        ) init)))

/--
Backward/VJP for `max_pool2d_spec`.

This propagates each output gradient to the argmax location inside the corresponding window.
Tie-breaking: if multiple values in the window are equal to the maximum, we keep the *first*
position in row-major order (same convention as PyTorch's max-pool indices).
-/
def maxPool2dBackwardSpec {kH kW inH inW stride : ℕ} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  {hStride : stride ≠ 0}
  (_layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
  (input : Image inH inW α)
  (grad_output : Image (Shape.slidingWindowOutDim inH kH stride 0) (Shape.slidingWindowOutDim inW kW stride 0) α) :
  Image inH inW α :=

  -- Initialize input gradient to zero
  let input_grad_init : Image inH inW α := createZeroImage inH inW

  let outH := Shape.slidingWindowOutDim inH kH stride 0
  let outW := Shape.slidingWindowOutDim inW kW stride 0

  -- For each output position, find max position in input and propagate gradient
  (List.finRange outH).foldl (fun acc_grad (out_i : Fin outH) =>
    (List.finRange outW).foldl (fun acc_grad_inner (out_j : Fin outW) =>

      -- Extract window starting at (out_i * stride, out_j * stride)
      let window := extractWindow kW kH input (out_i.val * stride) (out_j.val * stride)

      -- Find position of maximum in the window
      let max_pos : Fin kH × Fin kW :=
        (List.finRange kH).foldl (fun best_pos (di : Fin kH) =>
          (List.finRange kW).foldl (fun best_pos_inner (dj : Fin kW) =>
            let current_val := getAtSpec (getAtSpec window di) dj
            let best_val    := getAtSpec (getAtSpec window best_pos.1) best_pos.2
            match current_val, best_val with
            | Tensor.scalar curr, Tensor.scalar best =>
              if curr > best then (di, dj) else best_pos_inner
          ) best_pos
        ) (⟨0, Nat.zero_lt_of_ne_zero h1⟩, ⟨0, Nat.zero_lt_of_ne_zero h2⟩)

      -- Gradient value to propagate
      let grad_val := getAtSpec (getAtSpec grad_output out_i) out_j

      -- Compute absolute input indices of the maximum
      let inp_i := out_i.val * stride + max_pos.1.val
      let inp_j := out_j.val * stride + max_pos.2.val

      -- Bounds checks for inH, inW
      if h_inp_i : inp_i < inH then
        if h_inp_j : inp_j < inW then
          let current_grad := getAtSpec (getAtSpec acc_grad_inner ⟨inp_i, h_inp_i⟩) ⟨inp_j,
            h_inp_j⟩
          let new_grad     := addSpec current_grad grad_val
          updateTensorSpec acc_grad_inner [inp_i, inp_j] (Tensor.toScalar new_grad)
        else acc_grad_inner
      else acc_grad_inner

    ) acc_grad
  ) input_grad_init

/-- Multi-channel max-pooling backward (channel-wise application of `max_pool2d_backward_spec`). -/
def maxPool2dMultiBackwardSpec {kH kW inH inW inC stride : ℕ} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  {hStride : stride ≠ 0}
  (layer : MaxPool2DSpec kH kW stride h1 h2 hStride)
  (input : MultiChannelImage inC inH inW α)
  (grad_output :
    MultiChannelImage inC (Shape.slidingWindowOutDim inH kH stride 0) (Shape.slidingWindowOutDim inW kW stride 0) α) :
  MultiChannelImage inC inH inW α :=
  Tensor.dim (fun c =>
    maxPool2dBackwardSpec (α := α) (_layer := layer)
      (input := getAtSpec input c) (grad_output := getAtSpec grad_output c))


-- Backward pass for average pooling
/--
Backward/VJP for `avg_pool2d_spec` (single-channel).

Each output gradient is evenly distributed across its corresponding input window.
-/
def avgPool2dBackwardSpec {kH kW inH inW stride : ℕ} (_h1 : kH ≠ 0) (_h2 : kW ≠ 0)
  {hStride : stride ≠ 0}
  (_layer : AvgPool2DSpec kH kW stride _h1 _h2 hStride)
  (grad_output : Image (Shape.slidingWindowOutDim inH kH stride 0) (Shape.slidingWindowOutDim inW kW stride 0) α) :
  Image inH inW α :=

  -- Initialize input gradient to zero
  let input_grad_init : Image inH inW α := createZeroImage inH inW

  let outH := Shape.slidingWindowOutDim inH kH stride 0
  let outW := Shape.slidingWindowOutDim inW kW stride 0
  let pool_size := kH * kW

  -- For each output position, distribute its gradient evenly across the corresponding input window.
  (List.finRange outH).foldl (fun acc_grad (out_i : Fin outH) =>
    (List.finRange outW).foldl (fun acc_grad_inner (out_j : Fin outW) =>
      let grad_val := getAtSpec (getAtSpec grad_output out_i) out_j
      let distributed_grad := divSpec grad_val (Tensor.scalar pool_size)

      (List.finRange kH).foldl (fun acc_di (di : Fin kH) =>
        (List.finRange kW).foldl (fun acc_dj (dj : Fin kW) =>
          let inp_i := out_i.val * stride + di.val
          let inp_j := out_j.val * stride + dj.val
          if h_inp_i : inp_i < inH then
            if h_inp_j : inp_j < inW then
              let current_grad := getAtSpec (getAtSpec acc_dj ⟨inp_i, h_inp_i⟩) ⟨inp_j, h_inp_j⟩
              let new_grad := addSpec current_grad distributed_grad
              updateTensorSpec acc_dj [inp_i, inp_j] (Tensor.toScalar new_grad)
            else acc_dj
          else acc_dj
        ) acc_di
      ) acc_grad_inner
    ) acc_grad
  ) input_grad_init
end Spec
