/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import Init.Data.Vector.Lemmas

public import NN.Spec.Core.TensorOps

/-!
# Conv (generic N-D, spec layer)

This file defines channels-first convolution and transpose convolution over an arbitrary spatial
rank. The core specification handles one sample; batched interfaces map it over their leading
dimensions. The familiar two-dimensional declarations near the end are adapters to this generic
definition, not separate convolution semantics.

PyTorch analogy: this corresponds to `torch.nn.Conv{d}d` with:

- `groups = 1`,
- `dilation = 1`,
- per-axis `stride` and `padding`,
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

/-! ## Small generic helpers -/

namespace Conv
namespace Internal

def tensorOfFnList {α : Type} (dims : List Nat) (f : List Nat → α) : Tensor α (Shape.ofList dims) :=
  match dims with
  | [] => Tensor.scalar (f [])
  | _ :: ns => Tensor.dim (fun i => tensorOfFnList ns (fun is => f (i.val :: is)))

def foldlIndices {β : Type} (dims : List Nat) (init : β) (f : β → List Nat → β) : β :=
  let rec go (dims : List Nat) (prefixRev : List Nat) (acc : β) : β :=
    match dims with
    | [] => f acc prefixRev.reverse
    | n :: ns =>
        (List.finRange n).foldl (fun acc i => go ns (i.val :: prefixRev) acc) acc
  go dims [] init

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

/-- Parameters for a generic N-D convolution (weights + bias), channels-first. -/
structure ConvSpec (d inC outC : Nat) (kernel stride padding : Vector Nat d) (α : Type) where
  /-- Kernel weights, shape `(outC, inC, kernel[0], ..., kernel[d-1])`. -/
  kernel : Tensor α (Shape.ofList (outC :: inC :: kernel.toList))
  /-- Bias, shape `(outC)`. -/
  bias   : Tensor α (.dim outC .scalar)

/-- Output spatial sizes (`Vector Nat d`). -/
def convOutSpatial {d : Nat} (inSpatial kernel stride padding : Vector Nat d) : Vector Nat d :=
  Vector.ofFn (fun a =>
    Shape.slidingWindowOutDim
      (inSpatial.get a) (kernel.get a) (stride.get a) (padding.get a))

/-- A unit kernel with unit stride and no padding preserves every positive spatial extent. -/
theorem convOutSpatial_unit {d : Nat} (spatial : Vector Nat d)
    (hSpatial : ∀ i : Fin d, spatial.get i ≠ 0) :
    convOutSpatial spatial (Vector.replicate d 1) (Vector.replicate d 1)
      (Vector.replicate d 0) = spatial := by
  apply Vector.ext
  intro i hi
  have hPos : 1 ≤ spatial.get ⟨i, hi⟩ :=
    Nat.one_le_iff_ne_zero.mpr (hSpatial ⟨i, hi⟩)
  have hZero : (Vector.replicate d 0).get ⟨i, hi⟩ = 0 := by
    change (Vector.replicate d 0)[i] = 0
    simp
  have hOne : (Vector.replicate d 1).get ⟨i, hi⟩ = 1 := by
    change (Vector.replicate d 1)[i] = 1
    simp
  have hPosElem : 1 ≤ spatial[i] := by simpa [Vector.get] using hPos
  have hNonzero : spatial[i] ≠ 0 := Nat.ne_of_gt (Nat.lt_of_lt_of_le Nat.zero_lt_one hPosElem)
  simpa [convOutSpatial, Shape.slidingWindowOutDim, Vector.get, hNonzero,
    Nat.not_lt.mpr hPosElem] using
    Nat.sub_add_cancel hPosElem

/-- Output spatial shape `Shape.ofList [out0, ..., out(d-1)]`. -/
def convOutShape {d : Nat} (inSpatial kernel stride padding : Vector Nat d) : Shape :=
  Shape.ofList (convOutSpatial inSpatial kernel stride padding).toList

/-- Output shape including channels: `Shape.ofList (outC :: [out0, ..., out(d-1)])`. -/
def convMultiOutShape {d : Nat} (_inC outC : Nat) (inSpatial kernel stride padding : Vector Nat d)
    : Shape :=
  Shape.ofList (outC :: (convOutSpatial inSpatial kernel stride padding).toList)

/--
Generic N-D convolution forward pass on a single channels-first input (no batch dimension).

 Mathematically, for output channel `oc` and output spatial index `o : Vector Nat d`:

`y[oc,o] = Σ_{ic, k} x_pad[ic, o*stride + k] * W[oc,ic,k] + b[oc]`

where `k` ranges over the kernel window and `x_pad` is `input` with zero-padding.
-/
def convSpec
    {d inC outC : Nat}
    {kernel stride padding : Vector Nat d}
    {inSpatial : Vector Nat d}
    (layer : ConvSpec d inC outC kernel stride padding α)
    (input : Tensor α (Shape.ofList (inC :: inSpatial.toList))) :
    Tensor α (Shape.ofList (outC :: (convOutSpatial inSpatial kernel stride padding).toList)) :=

  let outSpatial := convOutSpatial inSpatial kernel stride padding
  let outDims : List Nat := outSpatial.toList
  let kDims : List Nat := kernel.toList
  let strideDims : List Nat := stride.toList
  let padDims : List Nat := padding.toList

  Tensor.dim (fun out_ch =>
    Conv.Internal.tensorOfFnList outDims (fun outIdx =>
      let total_sum : α :=
        (List.finRange inC).foldl (fun acc in_ch =>
          Conv.Internal.foldlIndices kDims acc (fun acc kIdx =>
            let input_val : α :=
              match Conv.Internal.mkInputIdx? outIdx kIdx strideDims padDims with
              | none => 0
              | some inIdx => getAtOrZero input (in_ch.val :: inIdx)
            let kernel_val : α :=
              getAtOrZero layer.kernel (out_ch.val :: in_ch.val :: kIdx)
            acc + input_val * kernel_val
          )
        ) 0
      total_sum + getAtOrZero layer.bias [out_ch.val]
    )
  )


/-- Gradient of convolution output w.r.t. the kernel weights (given `grad_output`). -/
def convKernelDerivSpec
    {d inC outC : Nat}
    {kernel stride padding : Vector Nat d}
    {inSpatial : Vector Nat d}
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
      Conv.Internal.tensorOfFnList kDims (fun kIdx =>
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
    {kernel stride padding : Vector Nat d}
    {inSpatial : Vector Nat d}
    (_layer : ConvSpec d inC outC kernel stride padding α)
    (_input : Tensor α (Shape.ofList (inC :: inSpatial.toList)))
    (grad_output :
      Tensor α
        (Shape.ofList (outC :: (convOutSpatial inSpatial kernel stride padding).toList)))
    : Tensor α (.dim outC .scalar) :=

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
    {kernel stride padding : Vector Nat d}
    {inSpatial : Vector Nat d}
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
    Conv.Internal.tensorOfFnList inDims (fun inIdx =>
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
    {kernel stride padding : Vector Nat d}
    {inSpatial : Vector Nat d}
    (layer : ConvSpec d inC outC kernel stride padding α)
    (input : Tensor α (Shape.ofList (inC :: inSpatial.toList)))
    (grad_output :
      Tensor α
        (Shape.ofList (outC :: (convOutSpatial inSpatial kernel stride padding).toList)))
    :
    (Tensor α (Shape.ofList (outC :: inC :: kernel.toList)) ×
     Tensor α (.dim outC .scalar) ×
     Tensor α (Shape.ofList (inC :: inSpatial.toList))) :=
  let d_kernel := convKernelDerivSpec layer input grad_output
  let d_bias := convBiasDerivSpec layer input grad_output
  let d_input := convInputDerivSpec layer input grad_output
  (d_kernel, d_bias, d_input)

/-! ## ConvTranspose (generic N-D) -/

/--
Parameters for a generic N-D transpose convolution (weights + bias), channels-first.

PyTorch analogy: this is `torch.nn.ConvTranspose{d}d` with:

- `output_padding = 0`,
- `dilation = 1`,
- `groups = 1`,
- per-axis `stride` and `padding`,
- and weight layout `(inC, outC, k0, ..., k(d-1))`.
-/
structure ConvTransposeSpec (d inC outC : Nat) (kernel stride padding : Vector Nat d) (α : Type) where
  /-- Kernel weights, shape `(inC, outC, kernel[0], ..., kernel[d-1])`. -/
  kernel : Tensor α (Shape.ofList (inC :: outC :: kernel.toList))
  /-- Bias, shape `(outC)`. -/
  bias   : Tensor α (.dim outC .scalar)

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

/-- Output spatial sizes (`Vector Nat d`) for transpose convolution (`output_padding = 0`). -/
def convTransposeOutSpatial {d : Nat} (inSpatial kernel stride padding : Vector Nat d) :
    Vector Nat d :=
  Vector.ofFn (fun a =>
    convTransposeOutDim (inSpatial.get a) (kernel.get a) (stride.get a) (padding.get a))

/-- Output spatial shape `Shape.ofList [out0, ..., out(d-1)]` (transpose convolution). -/
def convTransposeOutShape {d : Nat} (inSpatial kernel stride padding : Vector Nat d) : Shape :=
  Shape.ofList (convTransposeOutSpatial inSpatial kernel stride padding).toList

/-- Output shape including channels: `Shape.ofList (outC :: [out0, ..., out(d-1)])`. -/
def convTransposeMultiOutShape {d : Nat} (_inC outC : Nat)
    (inSpatial kernel stride padding : Vector Nat d) : Shape :=
  Shape.ofList (outC :: (convTransposeOutSpatial inSpatial kernel stride padding).toList)

/--
Generic N-D transpose convolution forward pass on a single channels-first input (no batch dim).

 For output channel `oc` and output spatial index `o : Vector Nat d` we define:

`y[oc,o] = Σ_{ic, k} x[ic, (o + padding - k) / stride] * W[ic,oc,k] + b[oc]`

where each axis must satisfy `out + padding ≥ k` and divisibility by `stride` (`% stride = 0`).
-/
def convTransposeSpec
    {d inC outC : Nat}
    {kernel stride padding : Vector Nat d}
    {inSpatial : Vector Nat d}
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
    Conv.Internal.tensorOfFnList outDims (fun outIdx =>
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
    {kernel stride padding : Vector Nat d}
    {inSpatial : Vector Nat d}
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
      Conv.Internal.tensorOfFnList kDims (fun kIdx =>
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
    {kernel stride padding : Vector Nat d}
    {inSpatial : Vector Nat d}
    (grad_output :
      Tensor α
        (Shape.ofList
          (outC :: (convTransposeOutSpatial inSpatial kernel stride padding).toList)))
    : Tensor α (.dim outC .scalar) :=

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
    {kernel stride padding : Vector Nat d}
    {inSpatial : Vector Nat d}
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
    Conv.Internal.tensorOfFnList inDims (fun inIdx =>
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
    {kernel stride padding : Vector Nat d}
    {inSpatial : Vector Nat d}
    (layer : ConvTransposeSpec d inC outC kernel stride padding α)
    (input : Tensor α (Shape.ofList (inC :: inSpatial.toList)))
    (grad_output :
      Tensor α
        (Shape.ofList
          (outC :: (convTransposeOutSpatial inSpatial kernel stride padding).toList)))
    :
    (Tensor α (Shape.ofList (inC :: outC :: kernel.toList)) ×
     Tensor α (.dim outC .scalar) ×
     Tensor α (Shape.ofList (inC :: inSpatial.toList))) :=
  let d_kernel := convTransposeKernelDerivSpec input grad_output
  let d_bias := convTransposeBiasDerivSpec grad_output
  let d_input := convTransposeInputDerivSpec layer.kernel grad_output
  (d_kernel, d_bias, d_input)

/-!
## 2D specializations

TorchLean exposes 2D convolution specs as first-class names because the model and proof layers use
these shapes directly. They share the same indexing conventions as the generic N-D convolution spec
above.
-/

/-- Parameters for a 2D convolution: this is `ConvSpec` specialized to `d = 2`. -/
abbrev Conv2dSpec (inC outC kH kW stride padding : Nat) (α : Type)
    (h1 : inC ≠ 0) (h2 : kH ≠ 0) (h3 : kW ≠ 0) :=
  (let _ := h1; let _ := h2; let _ := h3
   ConvSpec (d := 2) (inC := inC) (outC := outC)
     (kernel := ⟨#[kH, kW], by simp⟩)
     (stride := ⟨#[stride, stride], by simp⟩)
     (padding := ⟨#[padding, padding], by simp⟩) α)

/-- Output spatial shape `(outH,outW)` for a Conv2d with given hyperparameters. -/
def conv2dOutShape (inH inW kH kW stride padding : Nat) : Shape :=
  let outH := Shape.slidingWindowOutDim inH kH stride padding
  let outW := Shape.slidingWindowOutDim inW kW stride padding
  .dim outH (.dim outW .scalar)

/-- Output shape including channels: `(outC,outH,outW)`. -/
def conv2dMultiOutShape (_inC outC inH inW kH kW stride padding : Nat) : Shape :=
  let outH := Shape.slidingWindowOutDim inH kH stride padding
  let outW := Shape.slidingWindowOutDim inW kW stride padding
  .dim outC (.dim outH (.dim outW .scalar))

theorem Conv.Internal.conv2d_multi_out_shape_eq
    (outC inH inW kH kW stride padding : Nat) :
    Shape.ofList
        (outC ::
          (convOutSpatial (d := 2) (#v[inH, inW]) (#v[kH, kW]) (#v[stride, stride])
                (#v[padding, padding])).toList) =
      .dim outC
        (.dim (Shape.slidingWindowOutDim inH kH stride padding)
          (.dim (Shape.slidingWindowOutDim inW kW stride padding) .scalar)) := by
  simp [convOutSpatial, Shape.slidingWindowOutDim, Vector.get, Vector.toList, Shape.ofList]

/--
Conv2d forward pass on a single image `C×H×W` (no batch dimension).

PyTorch note: this matches the usual `Conv2d` definition (cross-correlation form, i.e. the kernel
is not spatially flipped).
-/
def conv2dSpec {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (layer : Conv2dSpec inC outC kH kW stride padding α h1 h2 h3)
    (input : Tensor α (.dim inC (.dim inH (.dim inW .scalar)))) :
    Tensor α (.dim outC (.dim (Shape.slidingWindowOutDim inH kH stride padding) (.dim (Shape.slidingWindowOutDim inW kW stride padding) .scalar))) := by
  have _ := h1
  have _ := h2
  have _ := h3
  let _ := layer
  -- We *do not* define this as a `tensorCast` of the generic N-D `convSpec` output.
  --
  -- Reason: the generic shape is expressed via `Vector.toList`, which elaborates through
  -- `Array.ofFn`; the cast proof `conv2d_multi_out_shape_eq` is therefore not definitionally
  -- transparent. Downstream proofs pattern-match directly on `conv2dSpec`, so the specialized
  -- tensor constructor keeps those matches at the expected shape.
  --
  -- Instead, we replay the same `mkInputIdx?`-based definition with explicit 2D dimension lists,
  -- which makes the output tensor constructor-built at the expected `(outC,outH,outW)` shape.
  let outH : Nat := Shape.slidingWindowOutDim inH kH stride padding
  let outW : Nat := Shape.slidingWindowOutDim inW kW stride padding
  let strideDims : List Nat := [stride, stride]
  let padDims : List Nat := [padding, padding]
  -- We keep the loop structure explicit (`inC × kH × kW`) so it matches the Conv2d pointwise
  -- reasoning lemmas directly.
  exact Tensor.dim (fun out_ch =>
    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        let total_sum : α :=
          (List.finRange inC).foldl (fun acc in_ch =>
            (List.finRange kH).foldl (fun acc di =>
              (List.finRange kW).foldl (fun acc dj =>
                let input_val : α :=
                  match Conv.Internal.mkInputIdx? [i.val, j.val] [di.val, dj.val] strideDims padDims with
                  | none => 0
                  | some inIdx => getAtOrZero input (in_ch.val :: inIdx)
                let kernel_val : α :=
                  getAtOrZero layer.kernel [out_ch.val, in_ch.val, di.val, dj.val]
                acc + input_val * kernel_val
              ) acc
            ) acc
          ) 0
        Tensor.scalar (total_sum + getAtOrZero layer.bias [out_ch.val]))))

/-- Gradient of Conv2d output w.r.t. the kernel weights (given `grad_output`). -/
def conv2dKernelDerivSpec
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (layer : Conv2dSpec inC outC kH kW stride padding α h1 h2 h3)
    (input : Tensor α (.dim inC (.dim inH (.dim inW .scalar))))
    (grad_output :
      Tensor α (.dim outC (.dim (Shape.slidingWindowOutDim inH kH stride padding) (.dim (Shape.slidingWindowOutDim inW kW stride padding) .scalar)))) :
    Tensor α (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))) := by
  have _ := h1
  have _ := h2
  have _ := h3
  let _ := layer
  let _ := input
  let outH : Nat := Shape.slidingWindowOutDim inH kH stride padding
  let outW : Nat := Shape.slidingWindowOutDim inW kW stride padding
  let strideDims : List Nat := [stride, stride]
  let padDims : List Nat := [padding, padding]
  exact Tensor.dim (fun out_ch =>
    Tensor.dim (fun in_ch =>
      Tensor.dim (fun di =>
        Tensor.dim (fun dj =>
          let total_sum : α :=
            (List.finRange outH).foldl (fun acc i =>
              (List.finRange outW).foldl (fun acc j =>
                let input_val : α :=
                  match Conv.Internal.mkInputIdx? [i.val, j.val] [di.val, dj.val] strideDims padDims with
                  | none => 0
                  | some inIdx => getAtOrZero input (in_ch.val :: inIdx)
                let grad_val : α := getAtOrZero grad_output [out_ch.val, i.val, j.val]
                acc + input_val * grad_val) acc) 0
          Tensor.scalar total_sum))))

/-- Gradient of Conv2d output w.r.t. the bias (sum over spatial positions). -/
def conv2dBiasDerivSpec
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (layer : Conv2dSpec inC outC kH kW stride padding α h1 h2 h3)
    (input : Tensor α (.dim inC (.dim inH (.dim inW .scalar))))
    (grad_output :
      Tensor α (.dim outC (.dim (Shape.slidingWindowOutDim inH kH stride padding) (.dim (Shape.slidingWindowOutDim inW kW stride padding) .scalar)))) :
    Tensor α (.dim outC .scalar) := by
  have _ := h1
  have _ := h2
  have _ := h3
  let _ := layer
  let _ := input
  let outH : Nat := Shape.slidingWindowOutDim inH kH stride padding
  let outW : Nat := Shape.slidingWindowOutDim inW kW stride padding
  exact Tensor.dim (fun out_ch =>
    let total_sum : α :=
      (List.finRange outH).foldl (fun acc i =>
        (List.finRange outW).foldl (fun acc j =>
          acc + getAtOrZero grad_output [out_ch.val, i.val, j.val]) acc) 0
    Tensor.scalar total_sum)

/-- Gradient of Conv2d output w.r.t. the input image. -/
def conv2dInputDerivSpec
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (layer : Conv2dSpec inC outC kH kW stride padding α h1 h2 h3)
    (input : Tensor α (.dim inC (.dim inH (.dim inW .scalar))))
    (grad_output :
      Tensor α (.dim outC (.dim (Shape.slidingWindowOutDim inH kH stride padding) (.dim (Shape.slidingWindowOutDim inW kW stride padding) .scalar)))) :
    Tensor α (.dim inC (.dim inH (.dim inW .scalar))) := by
  have _ := h1
  have _ := h2
  have _ := h3
  let _ := input
  let outH : Nat := Shape.slidingWindowOutDim inH kH stride padding
  let outW : Nat := Shape.slidingWindowOutDim inW kW stride padding
  exact Tensor.dim (fun in_ch =>
    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        let total_sum : α :=
          (List.finRange outC).foldl (fun acc out_ch =>
            (List.finRange outH).foldl (fun acc out_i =>
              (List.finRange outW).foldl (fun acc out_j =>
                (List.finRange kH).foldl (fun acc di =>
                  (List.finRange kW).foldl (fun acc dj =>
                    let contrib : α :=
                      if out_i.val * stride + di.val = i.val + padding ∧
                          out_j.val * stride + dj.val = j.val + padding then
                        let grad_val := getAtOrZero grad_output [out_ch.val, out_i.val, out_j.val]
                        let kernel_val :=
                          getAtOrZero layer.kernel [out_ch.val, in_ch.val, di.val, dj.val]
                        grad_val * kernel_val
                      else
                        0
                    acc + contrib) acc) acc) acc) acc) 0
        Tensor.scalar total_sum)))

/-- Conv2d backward pass: returns `(dKernel, dBias, dInput)`. -/
def conv2dBackwardSpec
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (layer : Conv2dSpec inC outC kH kW stride padding α h1 h2 h3)
    (input : Tensor α (.dim inC (.dim inH (.dim inW .scalar))))
    (grad_output :
      Tensor α (.dim outC (.dim (Shape.slidingWindowOutDim inH kH stride padding) (.dim (Shape.slidingWindowOutDim inW kW stride padding) .scalar)))) :
    (Tensor α (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))) ×
     Tensor α (.dim outC .scalar) ×
       Tensor α (.dim inC (.dim inH (.dim inW .scalar)))) := by
    have _ := h1
    have _ := h2
    have _ := h3
    exact
      (conv2dKernelDerivSpec (α := α) (layer := layer) (input := input) (grad_output := grad_output),
        conv2dBiasDerivSpec (α := α) (layer := layer) (input := input) (grad_output := grad_output),
        conv2dInputDerivSpec (α := α) (layer := layer) (input := input) (grad_output := grad_output))

/-!
## ConvTranspose2d specializations

The transpose-convolution definitions below are the 2D specialization of the generic N-D transpose
convolution spec above.
-/

/-- Kernel layout for transpose-convolution: `(inC, outC, kH, kW)`. -/
abbrev ConvTransposeKernel (outC inC kH kW : Nat) (α : Type) :=
  Tensor α (.dim inC (.dim outC (.dim kH (.dim kW .scalar))))

/-- Parameters for a 2D transpose convolution. -/
structure ConvTranspose2dSpec (inC outC kH kW stride padding : Nat) (α : Type)
    (h1 : inC > 0) (h2 : kH ≠ 0) (h3 : kW ≠ 0) where
  /-- Transposed-convolution kernel tensor. -/
  kernel : ConvTransposeKernel outC inC kH kW α
  /-- Per-output-channel bias. -/
  bias   : Tensor α (.dim outC .scalar)

/-- Output spatial shape `(outH,outW)` for `ConvTranspose2d` (with `output_padding = 0`). -/
def convTranspose2dOutShape (inH inW kH kW stride padding : Nat) : Shape :=
  let outH := convTransposeOutDim inH kH stride padding
  let outW := convTransposeOutDim inW kW stride padding
  .dim outH (.dim outW .scalar)

/-- Output shape including channels: `(outC,outH,outW)`. -/
def convTranspose2dMultiOutShape (_inC outC inH inW kH kW stride padding : Nat) : Shape :=
  let outH := convTransposeOutDim inH kH stride padding
  let outW := convTransposeOutDim inW kW stride padding
  .dim outC (.dim outH (.dim outW .scalar))

theorem Conv.Internal.conv_transpose2d_multi_out_shape_eq
    (outC inH inW kH kW stride padding : Nat) :
    Shape.ofList
        (outC ::
          (convTransposeOutSpatial (d := 2) (#v[inH, inW]) (#v[kH, kW]) (#v[stride, stride])
                (#v[padding, padding])).toList) =
      .dim outC
        (.dim (convTransposeOutDim inH kH stride padding)
          (.dim (convTransposeOutDim inW kW stride padding) .scalar)) := by
  simp [convTransposeOutSpatial, convTransposeOutDim, Vector.get, Vector.toList, Shape.ofList]

/--
ConvTranspose2d forward pass on a single image `C×H×W` (no batch dimension).

This is written as an output-indexed sum (no in-place updates), matching the standard definition.
-/
def convTranspose2dSpec {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC > 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (layer : ConvTranspose2dSpec inC outC kH kW stride padding α h1 h2 h3)
    (input : Tensor α (.dim inC (.dim inH (.dim inW .scalar)))) :
    Tensor α (.dim outC (.dim (convTransposeOutDim inH kH stride padding) (.dim (convTransposeOutDim inW kW stride padding) .scalar))) := by
  have _ := h1
  have _ := h2
  have _ := h3
  let layer' :
      ConvTransposeSpec (d := 2) (inC := inC) (outC := outC)
        (kernel := ⟨#[kH, kW], by simp⟩)
        (stride := ⟨#[stride, stride], by simp⟩)
        (padding := ⟨#[padding, padding], by simp⟩) α :=
    { kernel := layer.kernel, bias := layer.bias }
  let y :=
    convTransposeSpec (α := α) (d := 2)
      (kernel := #v[kH, kW])
      (stride := #v[stride, stride])
      (padding := #v[padding, padding])
      (inSpatial := #v[inH, inW])
      layer' input
  exact
    tensorCast _ (Conv.Internal.conv_transpose2d_multi_out_shape_eq (outC := outC) (inH := inH) (inW := inW)
      (kH := kH) (kW := kW) (stride := stride) (padding := padding)) y

/-- Gradient of ConvTranspose2d output w.r.t. the kernel weights. -/
def convTranspose2dWeightsDerivSpec {inC outC kH kW stride padding inH inW : Nat}
    {_h1 : inC > 0} {_h2 : kH ≠ 0} {_h3 : kW ≠ 0}
    (input : Tensor α (.dim inC (.dim inH (.dim inW .scalar))))
    (grad_output :
      Tensor α (.dim outC (.dim (convTransposeOutDim inH kH stride padding) (.dim (convTransposeOutDim inW kW stride padding) .scalar)))) :
    ConvTransposeKernel outC inC kH kW α := by
  let grad_output' :
      Tensor α
        (Shape.ofList
          (outC ::
            (convTransposeOutSpatial (d := 2) (#v[inH, inW]) (#v[kH, kW]) (#v[stride, stride])
                  (#v[padding, padding])).toList)) :=
    tensorCast _ (Conv.Internal.conv_transpose2d_multi_out_shape_eq (outC := outC) (inH := inH) (inW := inW)
      (kH := kH) (kW := kW) (stride := stride) (padding := padding)).symm grad_output
  exact
    convTransposeKernelDerivSpec (α := α) (d := 2)
      (kernel := #v[kH, kW])
      (stride := #v[stride, stride])
      (padding := #v[padding, padding])
      (inSpatial := #v[inH, inW])
      input grad_output'

/-- Gradient of ConvTranspose2d output w.r.t. the bias (sum over spatial positions). -/
def convTranspose2dBiasDerivSpec {_inC outC kH kW stride padding inH inW : Nat}
    (grad_output :
      Tensor α (.dim outC (.dim (convTransposeOutDim inH kH stride padding) (.dim (convTransposeOutDim inW kW stride padding) .scalar)))) :
    Tensor α (.dim outC .scalar) := by
  let grad_output' :
      Tensor α
        (Shape.ofList
          (outC ::
            (convTransposeOutSpatial (d := 2) (#v[inH, inW]) (#v[kH, kW]) (#v[stride, stride])
                  (#v[padding, padding])).toList)) :=
    tensorCast _ (Conv.Internal.conv_transpose2d_multi_out_shape_eq (outC := outC) (inH := inH) (inW := inW)
      (kH := kH) (kW := kW) (stride := stride) (padding := padding)).symm grad_output
  exact
    convTransposeBiasDerivSpec (α := α) (d := 2)
      (kernel := #v[kH, kW])
      (stride := #v[stride, stride])
      (padding := #v[padding, padding])
      (inSpatial := #v[inH, inW])
      grad_output'

/-- Gradient of ConvTranspose2d output w.r.t. the input image. -/
def convTranspose2dInputDerivSpec {inC outC kH kW stride padding inH inW : Nat}
    {_h1 : inC > 0} {_h2 : kH ≠ 0} {_h3 : kW ≠ 0}
    (weights : ConvTransposeKernel outC inC kH kW α)
    (grad_output :
      Tensor α (.dim outC (.dim (convTransposeOutDim inH kH stride padding) (.dim (convTransposeOutDim inW kW stride padding) .scalar)))) :
    Tensor α (.dim inC (.dim inH (.dim inW .scalar))) := by
  let grad_output' :
      Tensor α
        (Shape.ofList
          (outC ::
            (convTransposeOutSpatial (d := 2) (#v[inH, inW]) (#v[kH, kW]) (#v[stride, stride])
                  (#v[padding, padding])).toList)) :=
    tensorCast _ (Conv.Internal.conv_transpose2d_multi_out_shape_eq (outC := outC) (inH := inH) (inW := inW)
      (kH := kH) (kW := kW) (stride := stride) (padding := padding)).symm grad_output
  exact
    convTransposeInputDerivSpec (α := α) (d := 2)
      (kernel := #v[kH, kW])
      (stride := #v[stride, stride])
      (padding := #v[padding, padding])
      (inSpatial := #v[inH, inW])
      weights grad_output'

/-- ConvTranspose2d backward pass: returns `(dKernel, dBias, dInput)`. -/
def convTranspose2dBackwardSpec {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC > 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (layer : ConvTranspose2dSpec inC outC kH kW stride padding α h1 h2 h3)
    (input : Tensor α (.dim inC (.dim inH (.dim inW .scalar))))
    (grad_output :
      Tensor α (.dim outC (.dim (convTransposeOutDim inH kH stride padding) (.dim (convTransposeOutDim inW kW stride padding) .scalar)))) :
    (ConvTransposeKernel outC inC kH kW α ×
     Tensor α (.dim outC .scalar) ×
     Tensor α (.dim inC (.dim inH (.dim inW .scalar)))) := by
  have _ := h1
  have _ := h2
  have _ := h3
  let layer' :
      ConvTransposeSpec (d := 2) (inC := inC) (outC := outC)
        (kernel := ⟨#[kH, kW], by simp⟩)
        (stride := ⟨#[stride, stride], by simp⟩)
        (padding := ⟨#[padding, padding], by simp⟩) α :=
    { kernel := layer.kernel, bias := layer.bias }
  let grad_output' :
      Tensor α
        (Shape.ofList
          (outC ::
            (convTransposeOutSpatial (d := 2) (#v[inH, inW]) (#v[kH, kW]) (#v[stride, stride])
                  (#v[padding, padding])).toList)) :=
    tensorCast _ (Conv.Internal.conv_transpose2d_multi_out_shape_eq (outC := outC) (inH := inH) (inW := inW)
      (kH := kH) (kW := kW) (stride := stride) (padding := padding)).symm grad_output
  exact
    convTransposeBackwardSpec (α := α) (d := 2)
      (kernel := #v[kH, kW])
      (stride := #v[stride, stride])
      (padding := #v[padding, padding])
      (inSpatial := #v[inH, inW])
      layer' input grad_output'

end Spec
