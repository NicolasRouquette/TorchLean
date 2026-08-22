/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Module.Activation
public import NN.Spec.Module.Conv
public import NN.Spec.Module.Flatten
public import NN.Spec.Module.Linear
public import NN.Spec.Module.Pooling

/-!
# Convolutional Network Specifications

This file wires together a small CNN in two styles:

1. A compositional `Spec.Module.Chain` description:

- `Models.Cnn.spec`: `Conv2d → MaxPool2d → Conv2d → MaxPool2d → Flatten → Linear`
- `Models.Cnn.withReluSpec`: the same architecture with ReLU after each convolution

2. A fully explicit forward/backward pair (`Models.TwoBlockCnn.Model`) for the classic training setup:

`Conv → ReLU → MaxPool → Conv → ReLU → MaxPool → Flatten → Linear`

PyTorch analogue (single image, no batch):

```python
nn.Sequential(
  nn.Conv2d(inC, c1, (kH,kW), stride=stride1, padding=padding1),
  nn.ReLU(),
  nn.MaxPool2d((poolKH,poolKW), stride=poolStride1),
  nn.Conv2d(c1, c2, (kH,kW), stride=stride2, padding=padding2),
  nn.ReLU(),
  nn.MaxPool2d((poolKH,poolKW), stride=poolStride2),
  nn.Flatten(),
  nn.Linear(c2 * H2 * W2, outDim),
)
```

All shapes are tracked at the type level; the feature dimension for the final `LinearSpec` is
computed as `Spec.Shape.size` of the post-pooling feature map.

Both descriptions are mathematical specifications. Runtime implementations live under `NN.Runtime`.
-/

@[expose] public section


namespace Models

open Spec.Module
open Spec
open Tensor
open Activation

namespace Cnn

/-- Output size for a convolution along one spatial axis.

The shared shape helper agrees with the usual floor formula when the kernel and stride are valid,
and returns zero for invalid geometry instead of manufacturing a one-cell output through truncated
natural-number subtraction.
-/
abbrev convOut (input kernel stride padding : Nat) : Nat :=
  Spec.Shape.slidingWindowOutDim input kernel stride padding

/-- Output size for a pooling op along one spatial axis (no padding).

Matches the standard formula:

`out = (in - k) / stride + 1`.
-/
abbrev poolOut (input kernel stride : Nat) : Nat :=
  Spec.poolOutDim input kernel stride 0

/-- Output height after the first convolution stage. -/
abbrev firstConvOutHeight (inH kH stride1 padding1 : Nat) : Nat :=
  convOut inH kH stride1 padding1

/-- Output width after the first convolution stage. -/
abbrev firstConvOutWidth (inW kW stride1 padding1 : Nat) : Nat :=
  convOut inW kW stride1 padding1

/-- Output height after the first pooling stage. -/
abbrev firstPoolOutHeight (inH kH stride1 padding1 poolKH poolStride1 : Nat) : Nat :=
  poolOut (firstConvOutHeight inH kH stride1 padding1) poolKH poolStride1

/-- Output width after the first pooling stage. -/
abbrev firstPoolOutWidth (inW kW stride1 padding1 poolKW poolStride1 : Nat) : Nat :=
  poolOut (firstConvOutWidth inW kW stride1 padding1) poolKW poolStride1

/-- Output height after the second convolution stage (after pool1). -/
abbrev secondConvOutHeight (inH kH stride1 padding1 stride2 padding2 poolKH poolStride1 : Nat) : Nat :=
  convOut (firstPoolOutHeight inH kH stride1 padding1 poolKH poolStride1) kH stride2 padding2

/-- Output width after the second convolution stage (after pool1). -/
abbrev secondConvOutWidth (inW kW stride1 padding1 stride2 padding2 poolKW poolStride1 : Nat) : Nat :=
  convOut (firstPoolOutWidth inW kW stride1 padding1 poolKW poolStride1) kW stride2 padding2

/-- Output height after the second pooling stage. -/
abbrev secondPoolOutHeight (inH kH stride1 padding1 stride2 padding2 poolKH poolStride1 poolStride2 : Nat) : Nat
  :=
  poolOut (secondConvOutHeight inH kH stride1 padding1 stride2 padding2 poolKH poolStride1) poolKH poolStride2

/-- Output width after the second pooling stage. -/
abbrev secondPoolOutWidth (inW kW stride1 padding1 stride2 padding2 poolKW poolStride1 poolStride2 : Nat) : Nat
  :=
  poolOut (secondConvOutWidth inW kW stride1 padding1 stride2 padding2 poolKW poolStride1) poolKW poolStride2

/-- Feature-map shape after the second pooling stage: `(c2, H2, W2)`. -/
abbrev featShape
  (c2 inH inW kH kW stride1 padding1 stride2 padding2 poolKH poolKW poolStride1 poolStride2 : Nat) :
    Shape :=
  Shape.dim c2
    (.dim (secondPoolOutHeight inH kH stride1 padding1 stride2 padding2 poolKH poolStride1 poolStride2)
      (.dim (secondPoolOutWidth inW kW stride1 padding1 stride2 padding2 poolKW poolStride1 poolStride2)
        .scalar))

/-- Flattened feature size after the second pooling stage: `c2 * H2 * W2`. -/
abbrev featSize
  (c2 inH inW kH kW stride1 padding1 stride2 padding2 poolKH poolKW poolStride1 poolStride2 : Nat) :
    Nat :=
  Spec.Shape.size (featShape c2 inH inW kH kW stride1 padding1 stride2 padding2 poolKH poolKW poolStride1
    poolStride2)

/--
CNN `Spec.Module.Chain` wiring (no activations): `Conv2d -> MaxPool2d -> Conv2d -> MaxPool2d -> Flatten ->
  Linear`.

Use `withReluSpec` for the ReLU-after-convolution variant.
-/
def spec
  {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {inC outC inH inW kH kW stride1 padding1 stride2 padding2 poolKH poolKW poolStride1 poolStride2 :
    Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0} {h4 : outC ≠ 0} {h5 : poolKH ≠ 0} {h6 : poolKW ≠ 0}
  {hPoolStride1 : poolStride1 ≠ 0} {hPoolStride2 : poolStride2 ≠ 0}
  (conv1Spec : Conv2dSpec inC outC kH kW stride1 padding1 α h1 h2 h3)
  (conv2Spec : Conv2dSpec outC outC kH kW stride2 padding2 α h4 h2 h3)
  (pool1Spec : MaxPool2dSpec poolKH poolKW poolStride1 h5 h6 hPoolStride1)
  (pool2Spec : MaxPool2dSpec poolKH poolKW poolStride2 h5 h6 hPoolStride2)
  (linearSpec :
    LinearSpec α
      (Cnn.featSize outC inH inW kH kW stride1 padding1 stride2 padding2 poolKH poolKW poolStride1
        poolStride2)
      outC) :
  Spec.Module.Chain α (.dim inC (.dim inH (.dim inW .scalar))) (.dim outC .scalar) :=

  -- Create module specs
  let conv1Module := Spec.Module.conv2d conv1Spec
  let pool1Module := Spec.Module.maxPool2d pool1Spec
  let conv2Module := Spec.Module.conv2d conv2Spec
  let pool2Module := Spec.Module.maxPool2d pool2Spec
  let flattenModule :=
    Spec.Module.flatten α (Cnn.featShape outC inH inW kH kW stride1 padding1 stride2 padding2 poolKH
      poolKW poolStride1 poolStride2)
  let linearModule := Spec.Module.linear linearSpec

  -- Compose the chain: Conv1 → Pool1 → Conv2 → Pool2 → Flatten → Linear
  Spec.Module.Chain.single conv1Module
    |>.append pool1Module
    |>.append conv2Module
    |>.append pool2Module
    |>.append flattenModule
    |>.append linearModule

/-- A CNN `Spec.Module.Chain` with ReLU after each convolution.

PyTorch analogy: insert `nn.ReLU()` between the conv and pool blocks.
-/
def withReluSpec
  {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {inC outC inH inW kH kW stride1 padding1 stride2 padding2 poolKH poolKW poolStride1 poolStride2 :
    Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0} {h4 : outC ≠ 0} {h5 : poolKH ≠ 0} {h6 : poolKW ≠ 0}
  {hPoolStride1 : poolStride1 ≠ 0} {hPoolStride2 : poolStride2 ≠ 0}
  (conv1Spec : Conv2dSpec inC outC kH kW stride1 padding1 α h1 h2 h3)
  (conv2Spec : Conv2dSpec outC outC kH kW stride2 padding2 α h4 h2 h3)
  (pool1Spec : MaxPool2dSpec poolKH poolKW poolStride1 h5 h6 hPoolStride1)
  (pool2Spec : MaxPool2dSpec poolKH poolKW poolStride2 h5 h6 hPoolStride2)
  (linearSpec :
    LinearSpec α
      (Cnn.featSize outC inH inW kH kW stride1 padding1 stride2 padding2 poolKH poolKW poolStride1
        poolStride2)
      outC) :
  Spec.Module.Chain α (.dim inC (.dim inH (.dim inW .scalar))) (.dim outC .scalar) :=

  -- Create module specs
  let conv1Module := Spec.Module.conv2d conv1Spec
  let relu1Module :=
    Spec.Module.relu (α := α)
      (.dim outC
        (.dim (Cnn.firstConvOutHeight inH kH stride1 padding1)
          (.dim (Cnn.firstConvOutWidth inW kW stride1 padding1) .scalar)))
  let pool1Module := Spec.Module.maxPool2d pool1Spec
  let conv2Module := Spec.Module.conv2d conv2Spec
  let relu2Module :=
    Spec.Module.relu (α := α)
      (.dim outC
        (.dim (Cnn.secondConvOutHeight inH kH stride1 padding1 stride2 padding2 poolKH poolStride1)
          (.dim (Cnn.secondConvOutWidth inW kW stride1 padding1 stride2 padding2 poolKW poolStride1) .scalar)))
  let pool2Module := Spec.Module.maxPool2d pool2Spec
  let flattenModule :=
    Spec.Module.flatten α (Cnn.featShape outC inH inW kH kW stride1 padding1 stride2 padding2 poolKH
      poolKW poolStride1 poolStride2)
  let linearModule := Spec.Module.linear linearSpec

  -- Compose the chain: Conv1 → ReLU1 → Pool1 → Conv2 → ReLU2 → Pool2 → Flatten → Linear
  Spec.Module.Chain.single conv1Module
    |>.append relu1Module
    |>.append pool1Module
    |>.append conv2Module
    |>.append relu2Module
    |>.append pool2Module
    |>.append flattenModule
    |>.append linearModule

/-- Evaluate `spec` on one input tensor. -/
def forward
  {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {inC outC inH inW kH kW stride1 padding1 stride2 padding2 poolKH poolKW poolStride1 poolStride2 :
    Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0} {h4 : outC ≠ 0} {h5 : poolKH ≠ 0} {h6 : poolKW ≠ 0}
  {hPoolStride1 : poolStride1 ≠ 0} {hPoolStride2 : poolStride2 ≠ 0}
  (conv1Spec : Conv2dSpec inC outC kH kW stride1 padding1 α h1 h2 h3)
  (conv2Spec : Conv2dSpec outC outC kH kW stride2 padding2 α h4 h2 h3)
  (pool1Spec : MaxPool2dSpec poolKH poolKW poolStride1 h5 h6 hPoolStride1)
  (pool2Spec : MaxPool2dSpec poolKH poolKW poolStride2 h5 h6 hPoolStride2)
  (linearSpec :
    LinearSpec α
      (Cnn.featSize outC inH inW kH kW stride1 padding1 stride2 padding2 poolKH poolKW poolStride1
        poolStride2)
      outC)
  (x : Tensor α (.dim inC (.dim inH (.dim inW .scalar)))) :
  Tensor α (.dim outC .scalar) :=
  let net := spec conv1Spec conv2Spec pool1Spec pool2Spec linearSpec
  Spec.Module.Chain.forward (α := α) net x

end Cnn

/-!
## A fully explicit CNN spec with backward pass

The `Spec.Module.Chain` wiring above describes composition. The model below also gives an explicit
reverse-mode specification that returns parameter gradients.

The code below defines a small CNN in the classic:

`Conv → ReLU → MaxPool → Conv → ReLU → MaxPool → Flatten → Linear`

form, with a complete backward pass using the per-layer backward specifications:
- `conv2dBackwardSpec`,
- `maxPool2dMultiBackwardSpec`,
- `linearBackwardSpec`,
- elementwise gating for ReLU.
-/

namespace TwoBlockCnn

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-!
### Configuration

`Config` records the architectural choices shared by the model, its forward pass, and its backward
pass.
-/

/-- Hyperparameters for the small 2-layer CNN (`Model`). -/
structure Config where
  /-- Number of channels produced by the first convolution. -/
  conv1Channels : Nat := 32
  /-- Number of channels produced by the second convolution. -/
  conv2Channels : Nat := 64
  /-- Width of the linear output. -/
  outputSize : Nat := 10

  /-- Convolution kernel height. -/
  kernelHeight : Nat := 3
  /-- Convolution kernel width. -/
  kernelWidth : Nat := 3
  /-- Stride of the first convolution. -/
  conv1Stride : Nat := 1
  /-- Padding of the first convolution. -/
  conv1Padding : Nat := 1
  /-- Stride of the second convolution. -/
  conv2Stride : Nat := 1
  /-- Padding of the second convolution. -/
  conv2Padding : Nat := 1

  /-- Pooling kernel height. -/
  poolKernelHeight : Nat := 2
  /-- Pooling kernel width. -/
  poolKernelWidth : Nat := 2
  /-- Stride of the first pooling op. -/
  poolStride1 : Nat := 2
  /-- Stride of the second pooling op. -/
  poolStride2 : Nat := 2

/-- Conditions required by the convolution and pooling specifications. -/
structure Config.WF (cfg : Config) : Prop where
  conv1Channels_ne_zero : cfg.conv1Channels ≠ 0
  conv2Channels_ne_zero : cfg.conv2Channels ≠ 0
  outputSize_ne_zero : cfg.outputSize ≠ 0
  kernelHeight_ne_zero : cfg.kernelHeight ≠ 0
  kernelWidth_ne_zero : cfg.kernelWidth ≠ 0
  poolKernelHeight_ne_zero : cfg.poolKernelHeight ≠ 0
  poolKernelWidth_ne_zero : cfg.poolKernelWidth ≠ 0
  poolStride1_ne_zero : cfg.poolStride1 ≠ 0
  poolStride2_ne_zero : cfg.poolStride2 ≠ 0

/-- A small two-block CNN configuration. -/
def defaultConfig : Config := {}

/-- `defaultConfig` is well-formed. -/
theorem defaultConfig_wf : defaultConfig.WF := by
  refine
    { conv1Channels_ne_zero := by decide
      conv2Channels_ne_zero := by decide
      outputSize_ne_zero := by decide
      kernelHeight_ne_zero := by decide
      kernelWidth_ne_zero := by decide
      poolKernelHeight_ne_zero := by decide
      poolKernelWidth_ne_zero := by decide
      poolStride1_ne_zero := by decide
      poolStride2_ne_zero := by decide }

/-- A two-block CNN with an explicit linear head.

Parameters:

- `conv1 : Conv2d(inC -> conv1Channels)`
- `conv2 : Conv2d(conv1Channels -> conv2Channels)`
- `pool1`, `pool2 : MaxPool2d` (no padding in this spec)
- `head : Linear(conv2Channels * H2 * W2 -> outputSize)`

PyTorch analogue:

`Conv → ReLU → MaxPool → Conv → ReLU → MaxPool → Flatten → Linear`.

This structure is used by `forward` and `backward` below.
-/
structure Model
  (cfg : Config) (inC inH inW : Nat)
  (α : Type)
  (hInC : inC ≠ 0) (hCfg : cfg.WF) where
  conv1 :
    Spec.Conv2dSpec inC cfg.conv1Channels cfg.kernelHeight cfg.kernelWidth cfg.conv1Stride cfg.conv1Padding α hInC hCfg.kernelHeight_ne_zero hCfg.kernelWidth_ne_zero
  conv2 :
    Spec.Conv2dSpec cfg.conv1Channels cfg.conv2Channels cfg.kernelHeight cfg.kernelWidth cfg.conv2Stride cfg.conv2Padding α hCfg.conv1Channels_ne_zero hCfg.kernelHeight_ne_zero
      hCfg.kernelWidth_ne_zero
  pool1 :
    Spec.MaxPool2dSpec cfg.poolKernelHeight cfg.poolKernelWidth cfg.poolStride1 hCfg.poolKernelHeight_ne_zero hCfg.poolKernelWidth_ne_zero
      hCfg.poolStride1_ne_zero
  pool2 :
    Spec.MaxPool2dSpec cfg.poolKernelHeight cfg.poolKernelWidth cfg.poolStride2 hCfg.poolKernelHeight_ne_zero hCfg.poolKernelWidth_ne_zero
      hCfg.poolStride2_ne_zero
  head :
    Spec.LinearSpec α
      (Cnn.featSize cfg.conv2Channels inH inW cfg.kernelHeight cfg.kernelWidth cfg.conv1Stride cfg.conv1Padding cfg.conv2Stride cfg.conv2Padding
        cfg.poolKernelHeight cfg.poolKernelWidth cfg.poolStride1 cfg.poolStride2)
      cfg.outputSize

/-- Gradients for `Model` parameters (returned by `Model.backward`). -/
structure Grads
  (cfg : Config) (inC inH inW : Nat)
  (α : Type) where
  conv1Kernel : Tensor α (.dim cfg.conv1Channels (.dim inC (.dim cfg.kernelHeight (.dim cfg.kernelWidth .scalar))))
  conv1Bias : Tensor α (.dim cfg.conv1Channels .scalar)
  conv2Kernel : Tensor α (.dim cfg.conv2Channels (.dim cfg.conv1Channels (.dim cfg.kernelHeight (.dim cfg.kernelWidth .scalar))))
  conv2Bias : Tensor α (.dim cfg.conv2Channels .scalar)
  headWeight :
    Tensor α (.dim cfg.outputSize (.dim (Cnn.featSize cfg.conv2Channels inH inW cfg.kernelHeight cfg.kernelWidth cfg.conv1Stride cfg.conv1Padding
      cfg.conv2Stride cfg.conv2Padding cfg.poolKernelHeight cfg.poolKernelWidth cfg.poolStride1 cfg.poolStride2) .scalar))
  headBias : Tensor α (.dim cfg.outputSize .scalar)

/-- Forward pass for `Model`.

This definition evaluates the layer specifications directly rather than through `Spec.Module.Chain`.
-/
def Model.forward
  {cfg : Config} {inC inH inW : Nat}
  {hInC : inC ≠ 0} {hCfg : cfg.WF}
  (m : Model (α := α) cfg inC inH inW hInC hCfg)
  (x : Spec.Tensor α (.dim inC (.dim inH (.dim inW .scalar)))) :
  Tensor α (.dim cfg.outputSize .scalar) :=
  let y1 := Spec.conv2dSpec (α := α) m.conv1 x
  let r1 := Activation.reluSpec y1
  let p1 := Spec.maxPool2dMultiSpec (α := α) (layer := m.pool1) (input := r1)
  let y2 := Spec.conv2dSpec (α := α) m.conv2 p1
  let r2 := Activation.reluSpec y2
  let p2 := Spec.maxPool2dMultiSpec (α := α) (layer := m.pool2) (input := r2)
  let flat := Tensor.flattenSpec p2
  Spec.linearSpec (α := α) m.head flat

/-- Backward pass (reverse-mode / VJP) for `Model`.

Returns parameter gradients plus the input gradient.

Each step uses the corresponding layer backward spec:

- `linearBackwardSpec`
- `maxPool2dMultiBackwardSpec`
- `conv2dBackwardSpec`

and ReLU uses the standard pointwise gate `dY = dR ⊙ relu'(preact)`.
-/
def Model.backward
  {cfg : Config} {inC inH inW : Nat}
  {hInC : inC ≠ 0} {hCfg : cfg.WF}
  (m : Model (α := α) cfg inC inH inW hInC hCfg)
  (x : Spec.Tensor α (.dim inC (.dim inH (.dim inW .scalar))))
  (gradOutput : Tensor α (.dim cfg.outputSize .scalar)) :
  (Grads cfg inC inH inW α ×
   Spec.Tensor α (.dim inC (.dim inH (.dim inW .scalar)))) :=

  -- Forward reconstruction.
  let y1 := Spec.conv2dSpec (α := α) m.conv1 x
  let r1 := Activation.reluSpec y1
  let p1 := Spec.maxPool2dMultiSpec (α := α) (layer := m.pool1) (input := r1)
  let y2 := Spec.conv2dSpec (α := α) m.conv2 p1
  let r2 := Activation.reluSpec y2
  let p2 := Spec.maxPool2dMultiSpec (α := α) (layer := m.pool2) (input := r2)
  let flat := Tensor.flattenSpec p2

  -- Linear head backward.
  let (dW_head, db_head, d_flat) := Spec.linearBackwardSpec (α := α) m.head flat gradOutput

  -- Unflatten back to the pooled feature map.
  let featShape :=
    Cnn.featShape cfg.conv2Channels inH inW cfg.kernelHeight cfg.kernelWidth cfg.conv1Stride cfg.conv1Padding cfg.conv2Stride cfg.conv2Padding
      cfg.poolKernelHeight cfg.poolKernelWidth cfg.poolStride1 cfg.poolStride2

  let d_p2 : Spec.Tensor α (.dim cfg.conv2Channels (.dim (Cnn.secondPoolOutHeight inH cfg.kernelHeight cfg.conv1Stride cfg.conv1Padding cfg.conv2Stride cfg.conv2Padding cfg.poolKernelHeight cfg.poolStride1 cfg.poolStride2) (.dim (Cnn.secondPoolOutWidth inW cfg.kernelWidth cfg.conv1Stride cfg.conv1Padding cfg.conv2Stride cfg.conv2Padding cfg.poolKernelWidth cfg.poolStride1 cfg.poolStride2) .scalar))) :=
    Tensor.unflattenSpec featShape d_flat

  -- Pool2 backward.
  let d_r2 := Spec.maxPool2dMultiBackwardSpec (α := α) (layer := m.pool2) (input := r2)
    (grad_output := d_p2)

  -- ReLU2 backward.
  let d_y2 := mulSpec d_r2 (Activation.reluDerivSpec y2)

  -- Conv2 backward.
  let (conv2Kernel, conv2Bias, d_p1) :=
    Spec.conv2dBackwardSpec (α := α)
      (inC := cfg.conv1Channels) (outC := cfg.conv2Channels) (kH := cfg.kernelHeight) (kW := cfg.kernelWidth)
      (stride := cfg.conv2Stride) (padding := cfg.conv2Padding)
      (inH := (Cnn.firstPoolOutHeight inH cfg.kernelHeight cfg.conv1Stride cfg.conv1Padding cfg.poolKernelHeight cfg.poolStride1))
      (inW := (Cnn.firstPoolOutWidth inW cfg.kernelWidth cfg.conv1Stride cfg.conv1Padding cfg.poolKernelWidth cfg.poolStride1))
      (h1 := hCfg.conv1Channels_ne_zero) (h2 := hCfg.kernelHeight_ne_zero) (h3 := hCfg.kernelWidth_ne_zero)
      m.conv2 p1 d_y2

  -- Pool1 backward.
  let d_r1 := Spec.maxPool2dMultiBackwardSpec (α := α) (layer := m.pool1) (input := r1)
    (grad_output := d_p1)

  -- ReLU1 backward.
  let d_y1 := mulSpec d_r1 (Activation.reluDerivSpec y1)

  -- Conv1 backward.
  let (conv1Kernel, conv1Bias, d_x) :=
    Spec.conv2dBackwardSpec (α := α)
      (inC := inC) (outC := cfg.conv1Channels) (kH := cfg.kernelHeight) (kW := cfg.kernelWidth)
      (stride := cfg.conv1Stride) (padding := cfg.conv1Padding)
      (inH := inH) (inW := inW)
      (h1 := hInC) (h2 := hCfg.kernelHeight_ne_zero) (h3 := hCfg.kernelWidth_ne_zero)
      m.conv1 x d_y1

  let grads : Grads cfg inC inH inW α :=
    { conv1Kernel := conv1Kernel
      conv1Bias := conv1Bias
      conv2Kernel := conv2Kernel
      conv2Bias := conv2Bias
      headWeight := dW_head
      headBias := db_head }

  (grads, d_x)

end TwoBlockCnn

end Models
