/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Activation
public import NN.Spec.Layers.Conv
public import NN.Spec.Layers.Pooling

/-!
# U-Net Specification

This file specifies a two-level U-Net with one downsampling and one upsampling stage:

- down path: two `Conv2d(3x3, stride=1, padding=1) + ReLU` blocks,
- downsample: `MaxPool2d(kernel=2, stride=2)`,
- bottleneck: two more conv blocks,
- upsample: `ConvTranspose2d(kernel=2, stride=2)`,
- skip connection: concatenate channels and run two conv blocks,
- output head: `Conv2d(1x1)` to map the base channels to the output channels.

The tensors have shape `(C, H, W)`. The skip connection therefore concatenates axis `0` with
`concatLeadingAxisSpec`; a batched runtime implementation concatenates the corresponding channel
axis instead.

The convolution blocks preserve spatial dimensions with the default configuration. For odd input
sizes, pooling followed by transposed convolution can differ by one cell; `Model.forward` therefore
requires explicit equalities connecting the computed and requested dimensions.

References:
- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (MICCAI
  2015).

PyTorch documentation for the corresponding executable layers:
- `torch.nn.Conv2d`: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
- `torch.nn.MaxPool2d`: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
- `torch.nn.ConvTranspose2d`:
  https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
-/

@[expose] public section


namespace Models

open Spec
open Tensor
open Activation

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

namespace TwoLevelUnet

/-!
## Configuration

`Config` contains the kernel geometry and channel width used throughout the specification.
-/

/-- Architectural parameters for the two-level U-Net. -/
structure Config where
  /-- `kernel_size` for the max-pool layer (typical: `2`). -/
  poolKernel : Nat := 2
  /-- `stride` for the max-pool layer (typical: `2`). -/
  poolStride : Nat := 2

  /-- `kernel_size` for the 2D conv blocks (typical: `3`). -/
  convKernel : Nat := 3
  /-- `stride` for the 2D conv blocks (typical: `1`). -/
  convStride : Nat := 1
  /-- symmetric zero `padding` for the 2D conv blocks (typical: `1`). -/
  convPadding : Nat := 1

  /-- `kernel_size` for the transposed-convolution upsampler (typical: `2`). -/
  upKernel : Nat := 2
  /-- `stride` for the transposed-convolution upsampler (typical: `2`). -/
  upStride : Nat := 2
  /-- `padding` for the transposed-convolution upsampler (typical: `0`). -/
  upPadding : Nat := 0

  /-- `kernel_size` for the final output head conv (typical: `1`). -/
  headKernel : Nat := 1
  /-- `stride` for the final output head conv (typical: `1`). -/
  headStride : Nat := 1
  /-- `padding` for the final output head conv (typical: `0`). -/
  headPadding : Nat := 0

  /-- Number of channels in the full-resolution blocks. -/
  baseChannels : Nat := 64

/-- Conditions required by the convolution and pooling specifications. -/
structure Config.WF (cfg : Config) : Prop where
  poolKernel_ne_zero : cfg.poolKernel ≠ 0
  poolStride_ne_zero : cfg.poolStride ≠ 0
  convKernel_ne_zero : cfg.convKernel ≠ 0
  upKernel_ne_zero : cfg.upKernel ≠ 0
  upStride_ne_zero : cfg.upStride ≠ 0
  headKernel_ne_zero : cfg.headKernel ≠ 0
  baseChannels_pos : cfg.baseChannels > 0

/-- Standard kernel geometry with 64 base channels. -/
def defaultConfig : Config := {}

/-- `defaultConfig` satisfies the conditions required by the layer specifications. -/
theorem defaultConfig_wf : defaultConfig.WF := by
  refine
    { poolKernel_ne_zero := by decide
      poolStride_ne_zero := by decide
      convKernel_ne_zero := by decide
      upKernel_ne_zero := by decide
      upStride_ne_zero := by decide
      headKernel_ne_zero := by decide
      baseChannels_pos := by decide }

/-- Height after the downsampling pool. -/
abbrev downHeight (cfg : Config) (inH : Nat) : Nat :=
  poolOutDim inH cfg.poolKernel cfg.poolStride 0

/-- Width after the downsampling pool. -/
abbrev downWidth (cfg : Config) (inW : Nat) : Nat :=
  poolOutDim inW cfg.poolKernel cfg.poolStride 0

/-- Height after downsampling and transposed-convolution upsampling. -/
abbrev upHeight (cfg : Config) (inH : Nat) : Nat :=
  convTransposeOutDim (downHeight cfg inH) cfg.upKernel cfg.upStride cfg.upPadding

/-- Width after downsampling and transposed-convolution upsampling. -/
abbrev upWidth (cfg : Config) (inW : Nat) : Nat :=
  convTransposeOutDim (downWidth cfg inW) cfg.upKernel cfg.upStride cfg.upPadding

/--
Parameters of the two-level U-Net.

This is a compact U-Net with one downsample and one upsample step:
- two conv + ReLU blocks at full resolution (with a skip),
- max-pooling, then two conv + ReLU blocks at the lower resolution,
- a transposed-conv upsampler,
- channel concatenation with the skip feature map,
- two more conv + ReLU blocks,
- a final `1×1` conv head.

Tensors have shape `(C, H, W)`; this mathematical model does not include a batch axis.
-/
structure Model (cfg : Config) (inC outC inH inW : Nat) (α : Type)
  [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  (hInC : inC ≠ 0) (hCfg : cfg.WF) where
  /-- First convolution in the full-resolution down block. -/
  down1Conv1 :
    Conv2dSpec inC cfg.baseChannels cfg.convKernel cfg.convKernel cfg.convStride cfg.convPadding α hInC
      hCfg.convKernel_ne_zero hCfg.convKernel_ne_zero
  /-- Second convolution in the full-resolution down block. -/
  down1Conv2 :
    Conv2dSpec cfg.baseChannels cfg.baseChannels cfg.convKernel cfg.convKernel cfg.convStride cfg.convPadding α
      (Nat.ne_of_gt hCfg.baseChannels_pos) hCfg.convKernel_ne_zero hCfg.convKernel_ne_zero

  /-- First convolution in the bottleneck block. -/
  down2Conv1 :
    Conv2dSpec cfg.baseChannels (2 * cfg.baseChannels) cfg.convKernel cfg.convKernel cfg.convStride cfg.convPadding α
      (Nat.ne_of_gt hCfg.baseChannels_pos) hCfg.convKernel_ne_zero hCfg.convKernel_ne_zero
  /-- Second convolution in the bottleneck block. -/
  down2Conv2 :
    Conv2dSpec (2 * cfg.baseChannels) (2 * cfg.baseChannels) cfg.convKernel cfg.convKernel cfg.convStride cfg.convPadding α
      (Nat.ne_of_gt (Nat.mul_pos (by decide : 0 < 2) hCfg.baseChannels_pos)) hCfg.convKernel_ne_zero hCfg.convKernel_ne_zero

  /-- Transposed convolution that restores the spatial resolution. -/
  upsample :
    ConvTranspose2dSpec (2 * cfg.baseChannels) cfg.baseChannels cfg.upKernel cfg.upKernel cfg.upStride cfg.upPadding α
      (Nat.mul_pos (by decide : 0 < 2) hCfg.baseChannels_pos) hCfg.upKernel_ne_zero hCfg.upKernel_ne_zero

  /-- First convolution after concatenating the skip connection. -/
  up1Conv1 :
    Conv2dSpec (cfg.baseChannels + cfg.baseChannels) cfg.baseChannels cfg.convKernel cfg.convKernel cfg.convStride cfg.convPadding α
    (by
      have : 0 < cfg.baseChannels + cfg.baseChannels :=
        Nat.lt_of_lt_of_le hCfg.baseChannels_pos (Nat.le_add_right cfg.baseChannels cfg.baseChannels)
      exact Nat.ne_of_gt this)
    hCfg.convKernel_ne_zero hCfg.convKernel_ne_zero
  /-- Second convolution after the skip connection. -/
  up1Conv2 :
    Conv2dSpec cfg.baseChannels cfg.baseChannels cfg.convKernel cfg.convKernel cfg.convStride cfg.convPadding α
      (Nat.ne_of_gt hCfg.baseChannels_pos) hCfg.convKernel_ne_zero hCfg.convKernel_ne_zero

  /-- Final convolution that produces the requested output channels. -/
  output :
    Conv2dSpec cfg.baseChannels outC cfg.headKernel cfg.headKernel cfg.headStride cfg.headPadding α
      (Nat.ne_of_gt hCfg.baseChannels_pos) hCfg.headKernel_ne_zero hCfg.headKernel_ne_zero

/-!
## Gradients

The backward specification reconstructs the forward intermediates and applies the corresponding
layer-level vector-Jacobian products in reverse order.

Key details:
- `concatLeadingAxisSpec` is split via `concatLeadingAxisBackwardSpec`,
- pooling backward uses `maxPool2dMultiBackwardSpec`,
- ReLU is handled via elementwise gating `dZ = dY ⊙ ReLU'(Z)`.
-/

/--
Parameter-gradient container for `Model`.

The fields follow the parameter layout of `Model`.
-/
structure Grads (cfg : Config) (inC outC inH inW : Nat) (α : Type) where
  /-- Kernel gradient for `Model.down1Conv1`. -/
  down1Conv1Kernel :
    Tensor α (.dim cfg.baseChannels (.dim inC (.dim cfg.convKernel (.dim cfg.convKernel .scalar))))
  /-- Bias gradient for `Model.down1Conv1`. -/
  down1Conv1Bias : Tensor α (.dim cfg.baseChannels .scalar)
  /-- Kernel gradient for `Model.down1Conv2`. -/
  down1Conv2Kernel :
    Tensor α (.dim cfg.baseChannels (.dim cfg.baseChannels (.dim cfg.convKernel (.dim cfg.convKernel .scalar))))
  /-- Bias gradient for `Model.down1Conv2`. -/
  down1Conv2Bias : Tensor α (.dim cfg.baseChannels .scalar)
  /-- Kernel gradient for `Model.down2Conv1`. -/
  down2Conv1Kernel :
    Tensor α (.dim (2 * cfg.baseChannels) (.dim cfg.baseChannels (.dim cfg.convKernel (.dim cfg.convKernel .scalar))))
  /-- Bias gradient for `Model.down2Conv1`. -/
  down2Conv1Bias : Tensor α (.dim (2 * cfg.baseChannels) .scalar)
  /-- Kernel gradient for `Model.down2Conv2`. -/
  down2Conv2Kernel :
    Tensor α (.dim (2 * cfg.baseChannels) (.dim (2 * cfg.baseChannels) (.dim cfg.convKernel (.dim cfg.convKernel .scalar))))
  /-- Bias gradient for `Model.down2Conv2`. -/
  down2Conv2Bias : Tensor α (.dim (2 * cfg.baseChannels) .scalar)
  /-- Kernel gradient for `Model.upsample`. -/
  upsampleKernel :
    Spec.ConvTransposeKernel cfg.baseChannels (2 * cfg.baseChannels) cfg.upKernel cfg.upKernel α
  /-- Bias gradient for `Model.upsample`. -/
  upsampleBias : Tensor α (.dim cfg.baseChannels .scalar)
  /-- Kernel gradient for `Model.up1Conv1`. -/
  up1Conv1Kernel :
    Tensor α (.dim cfg.baseChannels (.dim (cfg.baseChannels + cfg.baseChannels) (.dim cfg.convKernel (.dim cfg.convKernel .scalar))))
  /-- Bias gradient for `Model.up1Conv1`. -/
  up1Conv1Bias : Tensor α (.dim cfg.baseChannels .scalar)
  /-- Kernel gradient for `Model.up1Conv2`. -/
  up1Conv2Kernel :
    Tensor α (.dim cfg.baseChannels (.dim cfg.baseChannels (.dim cfg.convKernel (.dim cfg.convKernel .scalar))))
  /-- Bias gradient for `Model.up1Conv2`. -/
  up1Conv2Bias : Tensor α (.dim cfg.baseChannels .scalar)
  /-- Kernel gradient for `Model.output`. -/
  outputKernel :
    Tensor α (.dim outC (.dim cfg.baseChannels (.dim cfg.headKernel (.dim cfg.headKernel .scalar))))
  /-- Bias gradient for `Model.output`. -/
  outputBias : Tensor α (.dim outC .scalar)

/--
Forward pass for `Model`.

Inputs and outputs are tensors of shape `(C, H, W)` with no batch axis.

The equality arguments identify the dimensions computed by the layer formulas with the dimensions
in the result type.
-/
def Model.forward
  {cfg : Config} {inC outC inH inW : Nat}
  {hInC : inC ≠ 0} {hCfg : cfg.WF}
  (m : Model (α := α) cfg inC outC inH inW hInC hCfg)
  (x : Tensor α (.dim inC (.dim inH (.dim inW .scalar))))
  (hConvHeight :
    (Shape.slidingWindowOutDim inH cfg.convKernel cfg.convStride cfg.convPadding) = inH)
  (hConvWidth :
    (Shape.slidingWindowOutDim inW cfg.convKernel cfg.convStride cfg.convPadding) = inW)
  (hDownConvHeight :
    (Shape.slidingWindowOutDim (downHeight cfg inH) cfg.convKernel cfg.convStride cfg.convPadding) = downHeight cfg inH)
  (hDownConvWidth :
    (Shape.slidingWindowOutDim (downWidth cfg inW) cfg.convKernel cfg.convStride cfg.convPadding) = downWidth cfg inW)
  (hUpsampleHeight : upHeight cfg inH = inH)
  (hUpsampleWidth : upWidth cfg inW = inW)
  (hOutputHeight : (Shape.slidingWindowOutDim inH cfg.headKernel cfg.headStride cfg.headPadding) = inH)
  (hOutputWidth : (Shape.slidingWindowOutDim inW cfg.headKernel cfg.headStride cfg.headPadding) = inW) :
  Tensor α (.dim outC (.dim inH (.dim inW .scalar))) :=

  -- Down block 1 (spatial preserved because conv is 3x3, stride=1, padding=1).
  let s1Raw :=
    reluSpec (conv2dSpec (α := α) m.down1Conv1 x)
  let s1 : Tensor α (.dim cfg.baseChannels (.dim inH (.dim inW .scalar))) :=
    Tensor.castShape s1Raw (by simp only [hConvHeight, hConvWidth])

  let skip1Raw :=
    reluSpec (conv2dSpec (α := α) m.down1Conv2 s1)
  let skip1 : Tensor α (.dim cfg.baseChannels (.dim inH (.dim inW .scalar))) :=
    Tensor.castShape skip1Raw (by simp only [hConvHeight, hConvWidth])

  -- Downsample (PyTorch analogy: `nn.MaxPool2d(kernel_size=2, stride=2)`).
  let pool : MaxPool2dSpec cfg.poolKernel cfg.poolKernel cfg.poolStride hCfg.poolKernel_ne_zero hCfg.poolKernel_ne_zero
      hCfg.poolStride_ne_zero :=
    {}

  let downH := downHeight cfg inH
  let downW := downWidth cfg inW

  let pooled : Tensor α (.dim cfg.baseChannels (.dim downH (.dim downW .scalar))) :=
    maxPool2dMultiSpec (α := α) (layer := pool) skip1

  -- Down block 2
  let b1Raw :=
    reluSpec (conv2dSpec (α := α) m.down2Conv1 pooled)
  let b1 : Tensor α (.dim (2 * cfg.baseChannels) (.dim downH (.dim downW .scalar))) :=
    Tensor.castShape b1Raw (by
      simp only [downH, downW, hDownConvHeight, hDownConvWidth])

  let bottleneckRaw :=
    reluSpec (conv2dSpec (α := α) m.down2Conv2 b1)
  let bottleneck : Tensor α (.dim (2 * cfg.baseChannels) (.dim downH (.dim downW .scalar))) :=
    Tensor.castShape bottleneckRaw (by
      simp only [downH, downW, hDownConvHeight, hDownConvWidth])

  -- Upsample (PyTorch analogy: `nn.ConvTranspose2d(kernel_size=2, stride=2, padding=0)`).
  let upRaw : Tensor α (.dim cfg.baseChannels (.dim (upHeight cfg inH) (.dim (upWidth cfg inW) .scalar))) :=
    convTranspose2dSpec (inC := 2 * cfg.baseChannels) (outC := cfg.baseChannels)
      (kH := cfg.upKernel) (kW := cfg.upKernel) (stride := cfg.upStride) (padding := cfg.upPadding)
      (inH := downH) (inW := downW) m.upsample bottleneck

  let up : Tensor α (.dim cfg.baseChannels (.dim inH (.dim inW .scalar))) :=
    Tensor.castShape upRaw (by simp only [hUpsampleHeight, hUpsampleWidth])

  -- Skip connection: concatenate channels (no batch axis in this file, so channels are axis 0).
  let merged : Tensor α (.dim (cfg.baseChannels + cfg.baseChannels) (.dim inH (.dim inW .scalar))) :=
    concatLeadingAxisSpec (t1 := skip1) (t2 := up)

  -- Up block
  let u1Raw :=
    reluSpec (conv2dSpec (α := α) m.up1Conv1 merged)
  let u1 : Tensor α (.dim cfg.baseChannels (.dim inH (.dim inW .scalar))) :=
    Tensor.castShape u1Raw (by simp only [hConvHeight, hConvWidth])

  let u2Raw :=
    reluSpec (conv2dSpec (α := α) m.up1Conv2 u1)
  let u2 : Tensor α (.dim cfg.baseChannels (.dim inH (.dim inW .scalar))) :=
    Tensor.castShape u2Raw (by simp only [hConvHeight, hConvWidth])

  let outputRaw :=
    conv2dSpec (α := α) m.output u2
  Tensor.castShape outputRaw (by simp only [hOutputHeight, hOutputWidth])

/--
Backward pass for `Model.forward`.

It returns the parameter gradients and the gradient with respect to the input. Intermediates are
recomputed from `m` and `x`; the mathematical specification does not use a mutable autograd tape.
-/
def Model.backward
  {cfg : Config} {inC outC inH inW : Nat}
  {hInC : inC ≠ 0} {hCfg : cfg.WF}
  (m : Model (α := α) cfg inC outC inH inW hInC hCfg)
  (x : Tensor α (.dim inC (.dim inH (.dim inW .scalar))))
  (gradOutput : Tensor α (.dim outC (.dim inH (.dim inW .scalar))))
  (hConvHeight :
    (Shape.slidingWindowOutDim inH cfg.convKernel cfg.convStride cfg.convPadding) = inH)
  (hConvWidth :
    (Shape.slidingWindowOutDim inW cfg.convKernel cfg.convStride cfg.convPadding) = inW)
  (hDownConvHeight :
    (Shape.slidingWindowOutDim (downHeight cfg inH) cfg.convKernel cfg.convStride cfg.convPadding) = downHeight cfg inH)
  (hDownConvWidth :
    (Shape.slidingWindowOutDim (downWidth cfg inW) cfg.convKernel cfg.convStride cfg.convPadding) = downWidth cfg inW)
  (hUpsampleHeight : upHeight cfg inH = inH)
  (hUpsampleWidth : upWidth cfg inW = inW)
  (hOutputHeight : (Shape.slidingWindowOutDim inH cfg.headKernel cfg.headStride cfg.headPadding) = inH)
  (hOutputWidth : (Shape.slidingWindowOutDim inW cfg.headKernel cfg.headStride cfg.headPadding) = inW) :
  (Grads cfg inC outC inH inW α × Tensor α (.dim inC (.dim inH (.dim inW .scalar)))) :=

  -- Forward reconstruction (mirrors `Model.forward`).
  -- We reconstruct intermediates because the backward rules (pooling / ReLU / conv) need the
  -- forward inputs (and in the case of max-pool, the values to determine which entries "won").
  let down1Conv1Output := conv2dSpec (α := α) m.down1Conv1 x
  let s1Raw := reluSpec down1Conv1Output
  let s1 : Tensor α (.dim cfg.baseChannels (.dim inH (.dim inW .scalar))) :=
    Tensor.castShape s1Raw (by simp only [hConvHeight, hConvWidth])

  let down1Conv2Output := conv2dSpec (α := α) m.down1Conv2 s1
  let skip1Raw := reluSpec down1Conv2Output
  let skip1 : Tensor α (.dim cfg.baseChannels (.dim inH (.dim inW .scalar))) :=
    Tensor.castShape skip1Raw (by simp only [hConvHeight, hConvWidth])

  let pool : MaxPool2dSpec cfg.poolKernel cfg.poolKernel cfg.poolStride hCfg.poolKernel_ne_zero hCfg.poolKernel_ne_zero
      hCfg.poolStride_ne_zero :=
    {}

  let downH := downHeight cfg inH
  let downW := downWidth cfg inW

  let pooled : Tensor α (.dim cfg.baseChannels (.dim downH (.dim downW .scalar))) :=
    maxPool2dMultiSpec (α := α) (layer := pool) skip1

  let down2Conv1Output := conv2dSpec (α := α) m.down2Conv1 pooled
  let b1Raw := reluSpec down2Conv1Output
  let b1 : Tensor α (.dim (2 * cfg.baseChannels) (.dim downH (.dim downW .scalar))) :=
    Tensor.castShape b1Raw (by
      simp only [downH, downW, hDownConvHeight, hDownConvWidth])

  let down2Conv2Output := conv2dSpec (α := α) m.down2Conv2 b1
  let bottleneckRaw := reluSpec down2Conv2Output
  let bottleneck : Tensor α (.dim (2 * cfg.baseChannels) (.dim downH (.dim downW .scalar))) :=
    Tensor.castShape bottleneckRaw (by
      simp only [downH, downW, hDownConvHeight, hDownConvWidth])

  let upRaw : Tensor α (.dim cfg.baseChannels (.dim (upHeight cfg inH) (.dim (upWidth cfg inW) .scalar))) :=
    convTranspose2dSpec (inC := 2 * cfg.baseChannels) (outC := cfg.baseChannels)
      (kH := cfg.upKernel) (kW := cfg.upKernel) (stride := cfg.upStride) (padding := cfg.upPadding)
      (inH := downH) (inW := downW) m.upsample bottleneck

  let up : Tensor α (.dim cfg.baseChannels (.dim inH (.dim inW .scalar))) :=
    Tensor.castShape upRaw (by simp only [hUpsampleHeight, hUpsampleWidth])

  let merged : Tensor α (.dim (cfg.baseChannels + cfg.baseChannels) (.dim inH (.dim inW .scalar))) :=
    concatLeadingAxisSpec (t1 := skip1) (t2 := up)

  let up1Conv1Output := conv2dSpec (α := α) m.up1Conv1 merged
  let u1Raw := reluSpec up1Conv1Output
  let u1 : Tensor α (.dim cfg.baseChannels (.dim inH (.dim inW .scalar))) :=
    Tensor.castShape u1Raw (by simp only [hConvHeight, hConvWidth])

  let up1Conv2Output := conv2dSpec (α := α) m.up1Conv2 u1
  let u2Raw := reluSpec up1Conv2Output
  let u2 : Tensor α (.dim cfg.baseChannels (.dim inH (.dim inW .scalar))) :=
    Tensor.castShape u2Raw (by simp only [hConvHeight, hConvWidth])

  -- Backward starts here.
  -- For each ReLU, we backprop through it using the standard gate:
  -- `dZ = dY ⊙ ReLU'(Z)` where `Z` is the pre-activation tensor.
  let outputRawGrad :
      Tensor α (.dim outC (.dim (Shape.slidingWindowOutDim inH cfg.headKernel cfg.headStride cfg.headPadding) (.dim (Shape.slidingWindowOutDim inW cfg.headKernel cfg.headStride cfg.headPadding) .scalar))) :=
    Tensor.castShape gradOutput (by simp only [hOutputHeight, hOutputWidth])

  let (outputKernel, outputBias, u2Grad) :=
    conv2dBackwardSpec (α := α)
      (inC := cfg.baseChannels) (outC := outC) (kH := cfg.headKernel) (kW := cfg.headKernel)
      (stride := cfg.headStride) (padding := cfg.headPadding)
      (inH := inH) (inW := inW)
      (h1 := Nat.ne_of_gt hCfg.baseChannels_pos) (h2 := hCfg.headKernel_ne_zero) (h3 := hCfg.headKernel_ne_zero)
      m.output u2 outputRawGrad

  let u2RawGrad :
      Tensor α (.dim cfg.baseChannels (.dim (Shape.slidingWindowOutDim inH cfg.convKernel cfg.convStride cfg.convPadding) (.dim (Shape.slidingWindowOutDim inW cfg.convKernel cfg.convStride cfg.convPadding) .scalar))) :=
    Tensor.castShape u2Grad (by simp only [hConvHeight, hConvWidth])

  let up1Conv2Grad := mulSpec u2RawGrad (reluDerivSpec up1Conv2Output)

  let (up1Conv2Kernel, up1Conv2Bias, u1Grad) :=
    conv2dBackwardSpec (α := α)
      (inC := cfg.baseChannels) (outC := cfg.baseChannels) (kH := cfg.convKernel) (kW := cfg.convKernel)
      (stride := cfg.convStride) (padding := cfg.convPadding)
      (inH := inH) (inW := inW)
      (h1 := Nat.ne_of_gt hCfg.baseChannels_pos) (h2 := hCfg.convKernel_ne_zero) (h3 := hCfg.convKernel_ne_zero)
      m.up1Conv2 u1 up1Conv2Grad

  let u1RawGrad :
      Tensor α (.dim cfg.baseChannels (.dim (Shape.slidingWindowOutDim inH cfg.convKernel cfg.convStride cfg.convPadding) (.dim (Shape.slidingWindowOutDim inW cfg.convKernel cfg.convStride cfg.convPadding) .scalar))) :=
    Tensor.castShape u1Grad (by simp only [hConvHeight, hConvWidth])

  let up1Conv1Grad := mulSpec u1RawGrad (reluDerivSpec up1Conv1Output)

  let (up1Conv1Kernel, up1Conv1Bias, mergedGrad) :=
    conv2dBackwardSpec (α := α)
      (inC := cfg.baseChannels + cfg.baseChannels) (outC := cfg.baseChannels) (kH := cfg.convKernel) (kW := cfg.convKernel)
      (stride := cfg.convStride) (padding := cfg.convPadding)
      (inH := inH) (inW := inW)
      (h1 := by
        have : 0 < cfg.baseChannels + cfg.baseChannels :=
          Nat.lt_of_lt_of_le hCfg.baseChannels_pos (Nat.le_add_right cfg.baseChannels cfg.baseChannels)
        exact Nat.ne_of_gt this)
      (h2 := hCfg.convKernel_ne_zero) (h3 := hCfg.convKernel_ne_zero)
      m.up1Conv1 merged up1Conv1Grad

  -- Split concat backward: merged = concat(skip1, up).
  -- Channel-concat is linear, so its backward just splits the incoming gradient into the two
  -- channel ranges.
  let (skip1MergeGrad, upGrad) :=
    concatLeadingAxisBackwardSpec (α := α) (n := cfg.baseChannels) (m := cfg.baseChannels)
      (s := .dim inH (.dim inW .scalar))
      mergedGrad

  let upRawGrad : Tensor α (.dim cfg.baseChannels (.dim (upHeight cfg inH) (.dim (upWidth cfg inW) .scalar))) :=
    Tensor.castShape upGrad (by simp only [hUpsampleHeight, hUpsampleWidth])

  let (upsampleKernel, upsampleBias, bottleneckGrad) :=
    convTranspose2dBackwardSpec
      (inC := 2 * cfg.baseChannels) (outC := cfg.baseChannels) (kH := cfg.upKernel) (kW := cfg.upKernel)
      (stride := cfg.upStride) (padding := cfg.upPadding)
      (inH := downH) (inW := downW)
      (h1 := Nat.mul_pos (by decide : 0 < 2) hCfg.baseChannels_pos) (h2 := hCfg.upKernel_ne_zero) (h3 := hCfg.upKernel_ne_zero)
      m.upsample bottleneck upRawGrad

  let bottleneckRawGrad : Tensor α (.dim (2 * cfg.baseChannels) (.dim (Shape.slidingWindowOutDim (downHeight cfg inH) cfg.convKernel cfg.convStride cfg.convPadding) (.dim (Shape.slidingWindowOutDim (downWidth cfg inW) cfg.convKernel cfg.convStride cfg.convPadding) .scalar))) :=
    Tensor.castShape bottleneckGrad (by
      simp only [downH, downW, hDownConvHeight, hDownConvWidth])

  let down2Conv2Grad := mulSpec bottleneckRawGrad (reluDerivSpec down2Conv2Output)

  let (down2Conv2Kernel, down2Conv2Bias, b1Grad) :=
    conv2dBackwardSpec (α := α)
      (inC := 2 * cfg.baseChannels) (outC := 2 * cfg.baseChannels) (kH := cfg.convKernel) (kW := cfg.convKernel)
      (stride := cfg.convStride) (padding := cfg.convPadding)
      (inH := downH) (inW := downW)
      (h1 := Nat.ne_of_gt (Nat.mul_pos (by decide : 0 < 2) hCfg.baseChannels_pos)) (h2 := hCfg.convKernel_ne_zero)
      (h3 := hCfg.convKernel_ne_zero)
      m.down2Conv2 b1 down2Conv2Grad

  let b1RawGrad : Tensor α (.dim (2 * cfg.baseChannels) (.dim (Shape.slidingWindowOutDim (downHeight cfg inH) cfg.convKernel cfg.convStride cfg.convPadding) (.dim (Shape.slidingWindowOutDim (downWidth cfg inW) cfg.convKernel cfg.convStride cfg.convPadding) .scalar))) :=
    Tensor.castShape b1Grad (by
      simp only [downH, downW, hDownConvHeight, hDownConvWidth])

  let down2Conv1Grad := mulSpec b1RawGrad (reluDerivSpec down2Conv1Output)

  let (down2Conv1Kernel, down2Conv1Bias, pooledGrad) :=
    conv2dBackwardSpec (α := α)
      (inC := cfg.baseChannels) (outC := 2 * cfg.baseChannels) (kH := cfg.convKernel) (kW := cfg.convKernel)
      (stride := cfg.convStride) (padding := cfg.convPadding)
      (inH := downH) (inW := downW)
      (h1 := Nat.ne_of_gt hCfg.baseChannels_pos) (h2 := hCfg.convKernel_ne_zero) (h3 := hCfg.convKernel_ne_zero)
      m.down2Conv1 pooled down2Conv1Grad

  -- Pool backward.
  -- MaxPool backward routes gradient back to the (per-window) argmax location.
  let skip1PoolGrad :=
    maxPool2dMultiBackwardSpec (α := α) (layer := pool) (input := skip1) (grad_output :=
      pooledGrad)

  let skip1Grad := addSpec skip1PoolGrad skip1MergeGrad
  let skip1RawGrad : Tensor α (.dim cfg.baseChannels (.dim (Shape.slidingWindowOutDim inH cfg.convKernel cfg.convStride cfg.convPadding) (.dim (Shape.slidingWindowOutDim inW cfg.convKernel cfg.convStride cfg.convPadding) .scalar))) :=
    Tensor.castShape skip1Grad (by simp only [hConvHeight, hConvWidth])

  let down1Conv2Grad := mulSpec skip1RawGrad (reluDerivSpec down1Conv2Output)

  let (down1Conv2Kernel, down1Conv2Bias, s1Grad) :=
    conv2dBackwardSpec (α := α)
      (inC := cfg.baseChannels) (outC := cfg.baseChannels) (kH := cfg.convKernel) (kW := cfg.convKernel)
      (stride := cfg.convStride) (padding := cfg.convPadding)
      (inH := inH) (inW := inW)
      (h1 := Nat.ne_of_gt hCfg.baseChannels_pos) (h2 := hCfg.convKernel_ne_zero) (h3 := hCfg.convKernel_ne_zero)
      m.down1Conv2 s1 down1Conv2Grad

  let s1RawGrad : Tensor α (.dim cfg.baseChannels (.dim (Shape.slidingWindowOutDim inH cfg.convKernel cfg.convStride cfg.convPadding) (.dim (Shape.slidingWindowOutDim inW cfg.convKernel cfg.convStride cfg.convPadding) .scalar))) :=
    Tensor.castShape s1Grad (by simp only [hConvHeight, hConvWidth])

  let down1Conv1Grad := mulSpec s1RawGrad (reluDerivSpec down1Conv1Output)

  let (down1Conv1Kernel, down1Conv1Bias, inputGrad) :=
    conv2dBackwardSpec (α := α)
      (inC := inC) (outC := cfg.baseChannels) (kH := cfg.convKernel) (kW := cfg.convKernel)
      (stride := cfg.convStride) (padding := cfg.convPadding)
      (inH := inH) (inW := inW)
      (h1 := hInC) (h2 := hCfg.convKernel_ne_zero) (h3 := hCfg.convKernel_ne_zero)
      m.down1Conv1 x down1Conv1Grad

  let grads : Grads cfg inC outC inH inW α :=
    { down1Conv1Kernel := down1Conv1Kernel
      down1Conv1Bias := down1Conv1Bias
      down1Conv2Kernel := down1Conv2Kernel
      down1Conv2Bias := down1Conv2Bias
      down2Conv1Kernel := down2Conv1Kernel
      down2Conv1Bias := down2Conv1Bias
      down2Conv2Kernel := down2Conv2Kernel
      down2Conv2Bias := down2Conv2Bias
      upsampleKernel := upsampleKernel
      upsampleBias := upsampleBias
      up1Conv1Kernel := up1Conv1Kernel
      up1Conv1Bias := up1Conv1Bias
      up1Conv2Kernel := up1Conv2Kernel
      up1Conv2Bias := up1Conv2Bias
      outputKernel := outputKernel
      outputBias := outputBias }

  (grads, inputGrad)

end TwoLevelUnet

end Models
