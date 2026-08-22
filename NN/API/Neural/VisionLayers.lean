/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import NN.API.Neural.Leading

/-!
# Vision Layers

This file provides named-field layer records for spatial operators. Tensors remain ordinary
arbitrary-rank tensors; each operator states the trailing axes it consumes, while `leading` records
any axes mapped pointwise by the layer.
-/

@[expose] public section

namespace TorchLean
namespace nn
namespace Internal

/-! ## Spatial layers -/

/-- Configuration shared by arbitrary-dimensional convolution layers. -/
structure Conv (d : Nat) where
  /-- Number of output channels. -/
  outChannels : Nat
  /-- Kernel extent along each spatial axis. -/
  kernel : Vector Nat d
  /-- Step along each spatial axis. -/
  stride : Vector Nat d := Vector.replicate d 1
  /-- Symmetric zero-padding along each spatial axis. -/
  padding : Vector Nat d := Vector.replicate d 0
  /-- Every kernel extent is positive. -/
  kernelNonzero : ∀ i : Fin d, kernel.get i ≠ 0
  /-- Every stride is positive. -/
  strideNonzero : ∀ i : Fin d, stride.get i ≠ 0
  /-- Seed for deterministic kernel initialization. -/
  seedKernel : Nat := 0
  /-- Seed for deterministic bias initialization. -/
  seedBias : Nat := 0
  /-- Initialization scheme for the kernel weights. -/
  kernelInit : _root_.Runtime.Autograd.Torch.Init.Scheme := .uniform (-0.1) 0.1

/--
Apply an arbitrary-dimensional convolution to the channel and spatial suffix of a tensor.

The input suffix is `(inChannels, spatial...)`. Any axes in `leading` are preserved; internally
they are flattened into one runtime batch and restored after the convolution.
-/
def conv (leading : Spec.Shape := .scalar) {d inChannels : Nat} (spatial : Vector Nat d)
    (cfg : Conv d) [NeZero inChannels] :
    Sequential
      (leading.concat (Spec.Shape.ofList (inChannels :: spatial.toList)))
      (leading.concat (Spec.Shape.ofList
        (cfg.outChannels :: (Spec.convOutSpatial spatial cfg.kernel cfg.stride cfg.padding).toList))) :=
  nn.of <| adaptLeadingShape leading <|
    _root_.Runtime.Autograd.TorchLean.NN.conv
      (Spec.Shape.size leading) d inChannels cfg.outChannels
      cfg.kernel cfg.stride cfg.padding spatial
      (hInC := NeZero.ne _) (hKernel := cfg.kernelNonzero) (hStride := cfg.strideNonzero)
      cfg.seedKernel cfg.seedBias cfg.kernelInit

/-- Configuration shared by arbitrary-dimensional pooling layers. -/
structure Pool (d : Nat) where
  /-- Window extent along each spatial axis. -/
  kernel : Vector Nat d
  /-- Step along each spatial axis. -/
  stride : Vector Nat d := Vector.replicate d 1
  /-- Symmetric padding along each spatial axis. -/
  padding : Vector Nat d := Vector.replicate d 0
  /-- Every window extent is positive. -/
  kernelNonzero : ∀ i : Fin d, kernel.get i ≠ 0
  /-- Every stride is positive. -/
  strideNonzero : ∀ i : Fin d, stride.get i ≠ 0

/-- Apply max pooling to the channel and spatial suffix of a tensor. -/
def maxPool (leading : Spec.Shape := .scalar) {d channels : Nat} (spatial : Vector Nat d)
    (cfg : Pool d) :
    Sequential
      (leading.concat (Spec.Shape.ofList (channels :: spatial.toList)))
      (leading.concat (Spec.Shape.ofList
        (channels :: (Spec.poolOutSpatialPad spatial cfg.kernel cfg.stride cfg.padding).toList))) :=
  nn.of <| adaptLeadingShape leading <|
    _root_.Runtime.Autograd.TorchLean.NN.maxPool (Spec.Shape.size leading) d channels
      cfg.kernel cfg.stride cfg.padding spatial
      (hKernel := cfg.kernelNonzero) (hStride := cfg.strideNonzero)

/-- Apply average pooling to the channel and spatial suffix of a tensor. -/
def avgPool (leading : Spec.Shape := .scalar) {d channels : Nat} (spatial : Vector Nat d)
    (cfg : Pool d) :
    Sequential
      (leading.concat (Spec.Shape.ofList (channels :: spatial.toList)))
      (leading.concat (Spec.Shape.ofList
        (channels :: (Spec.poolOutSpatialPad spatial cfg.kernel cfg.stride cfg.padding).toList))) :=
  nn.of <| adaptLeadingShape leading <|
    _root_.Runtime.Autograd.TorchLean.NN.avgPool (Spec.Shape.size leading) d channels
      cfg.kernel cfg.stride cfg.padding spatial cfg.kernelNonzero cfg.strideNonzero

/--
Global average pooling over every spatial axis, preserving the leading axes and channels.
-/
def globalAvgPool (leading : Spec.Shape := .scalar) {d channels : Nat}
    (spatial : Vector Nat d) (spatialNonzero : ∀ i : Fin d, spatial.get i ≠ 0) :
    Sequential
      (leading.concat (Spec.Shape.ofList (channels :: spatial.toList)))
      (leading.appendDim channels) :=
  let config : Pool d :=
    { kernel := spatial
      stride := Vector.replicate d 1
      padding := Vector.replicate d 0
      kernelNonzero := spatialNonzero
      strideNonzero := by intro i; simp [Vector.get] }
  let pooled := avgPool leading spatial config
  let pooledShape := leading.concat (Spec.Shape.ofList
    (channels :: (Spec.poolOutSpatialPad spatial spatial
      (Vector.replicate d 1) (Vector.replicate d 0)).toList))
  let outputShape := leading.appendDim channels
  let removeSingletons : Layer pooledShape outputShape :=
    { kind := "GlobalAvgPool"
      stateShapes := []
      initState := .nil
      requiresGrad := []
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun x =>
            _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α) (s₁ := pooledShape) (s₂ := outputShape)
              x (by
                dsimp [pooledShape, outputShape]
                rw [Spec.poolOutSpatialPad_global spatial spatialNonzero]
                simp [Spec.Shape.size_concat, Spec.Shape.ofList,
                  Spec.Shape.size, Spec.Shape.size_appendDim]) }
  seq! pooled, nn.of removeSingletons

/--
LayerNorm parameter initialization.

PyTorch analogue: `torch.nn.LayerNorm`.
See `https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html`.
-/
structure LayerNormConfig where
  /-- Seed for deterministic initialization of `gamma` (scale). -/
  seedGamma : Nat := 0
  /-- Seed for deterministic initialization of `beta` (shift). -/
  seedBeta : Nat := 0

/-- Layer normalization over the final axis, with an explicit nonempty-width proof. -/
def layerNormWith (leading : Spec.Shape := .scalar) {width : Nat} (cfg : LayerNormConfig)
    (hWidth : width > 0) :
    Sequential (leading.appendDim width) (leading.appendDim width) :=
  of <| _root_.Runtime.Autograd.TorchLean.NN.layerNorm leading width (hWidth := hWidth)
    (seedGamma := cfg.seedGamma) (seedBeta := cfg.seedBeta)

/--
Layer normalization over the final axis of a tensor.

Every index in `leading` selects one width-`width` vector. Empty leading axes are allowed; only the
normalized axis must be nonempty.
-/
def layerNorm (leading : Spec.Shape := .scalar) {width : Nat} (cfg : LayerNormConfig := {})
    [NeZero width] : Sequential (leading.appendDim width) (leading.appendDim width) :=
  layerNormWith leading (width := width) cfg (Nat.pos_of_ne_zero (NeZero.ne (n := width)))

/-- RMSNorm parameter initialization. -/
structure RmsNormConfig where
  /-- Seed for deterministic initialization of `gamma` (scale). -/
  seedGamma : Nat := 0

/-- RMS normalization over the final axis, with an explicit nonempty-width proof. -/
def rmsNormWith (leading : Spec.Shape := .scalar) {width : Nat} (cfg : RmsNormConfig)
    (hWidth : width > 0) :
    Sequential (leading.appendDim width) (leading.appendDim width) :=
  of <| _root_.Runtime.Autograd.TorchLean.NN.rmsNorm leading width (hWidth := hWidth)
    (seedGamma := cfg.seedGamma)

/-- RMS normalization over the final axis of a tensor. -/
def rmsNorm (leading : Spec.Shape := .scalar) {width : Nat} (cfg : RmsNormConfig := {})
    [NeZero width] : Sequential (leading.appendDim width) (leading.appendDim width) :=
  rmsNormWith leading (width := width) cfg (Nat.pos_of_ne_zero (NeZero.ne (n := width)))

/-- Parameter initialization for affine channel normalization. -/
structure ChannelNormConfig where
  seedGamma : Nat := 0
  seedBeta : Nat := 0

namespace Implementation

/-- Apply a channel-first normalization kernel after adding its singleton trailing axis. -/
def channelFirstKernel (leadingSize channels spatialSize : Nat)
    (kernel : Layer
      (.dim leadingSize (.dim channels (.dim spatialSize (.dim 1 .scalar))))
      (.dim leadingSize (.dim channels (.dim spatialSize (.dim 1 .scalar))))) :
    Sequential
      (.dim leadingSize (.dim channels (.dim spatialSize .scalar)))
      (.dim leadingSize (.dim channels (.dim spatialSize .scalar))) :=
  let source := .dim leadingSize (.dim channels (.dim spatialSize .scalar))
  let target := .dim leadingSize (.dim channels (.dim spatialSize (.dim 1 .scalar)))
  seq!
    reshape source target (by simp [source, target, Spec.Shape.size]),
    of kernel,
    reshape target source (by simp [source, target, Spec.Shape.size])

/-- Flatten arbitrary spatial axes to the channel-first kernel representation. -/
def spatialReshape {d channels : Nat} (leading : Spec.Shape)
    (spatial : Vector Nat d) :
    Sequential
      (leading.concat (Spec.Shape.ofList (channels :: spatial.toList)))
      (.dim (Spec.Shape.size leading)
        (.dim channels (.dim (Spec.Shape.size (Spec.Shape.ofList spatial.toList)) .scalar))) :=
  reshape _ _ (by simp [Spec.Shape.size_concat, Spec.Shape.ofList, Spec.Shape.size])

/-- Restore the original spatial axes after channel normalization. -/
def spatialRestore {d channels : Nat} (leading : Spec.Shape)
    (spatial : Vector Nat d) :
    Sequential
      (.dim (Spec.Shape.size leading)
        (.dim channels (.dim (Spec.Shape.size (Spec.Shape.ofList spatial.toList)) .scalar)))
      (leading.concat (Spec.Shape.ofList (channels :: spatial.toList))) :=
  reshape _ _ (by simp [Spec.Shape.size_concat, Spec.Shape.ofList, Spec.Shape.size])

end Implementation

/-- Batch normalization over `(leading..., channels, spatial...)` for any spatial rank. -/
def batchNorm (leading : Spec.Shape := .scalar) {d channels : Nat}
    (spatial : Vector Nat d) (cfg : ChannelNormConfig := {})
    [NeZero (Spec.Shape.size leading)] [NeZero channels]
    [NeZero (Spec.Shape.size (Spec.Shape.ofList spatial.toList))] :
    Sequential
      (leading.concat (Spec.Shape.ofList (channels :: spatial.toList)))
      (leading.concat (Spec.Shape.ofList (channels :: spatial.toList))) :=
  let n := Spec.Shape.size leading
  let extent := Spec.Shape.size (Spec.Shape.ofList spatial.toList)
  seq!
    Implementation.spatialReshape (channels := channels) leading spatial,
    Implementation.channelFirstKernel n channels extent <|
      _root_.Runtime.Autograd.TorchLean.NN.batchNorm2dNchwMode n channels extent 1
        (h_n_pos := Nat.pos_of_ne_zero (NeZero.ne n))
        (h_c_pos := Nat.pos_of_ne_zero (NeZero.ne channels))
        (h_h_pos := Nat.pos_of_ne_zero (NeZero.ne extent))
        (h_w_pos := by decide)
        cfg.seedGamma cfg.seedBeta,
    Implementation.spatialRestore (channels := channels) leading spatial

/-- Instance normalization over `(leading..., channels, spatial...)` for any spatial rank. -/
def instanceNorm (leading : Spec.Shape := .scalar) {d channels : Nat}
    (spatial : Vector Nat d) (cfg : ChannelNormConfig := {})
    [NeZero (Spec.Shape.size leading)] [NeZero channels]
    [NeZero (Spec.Shape.size (Spec.Shape.ofList spatial.toList))] :
    Sequential
      (leading.concat (Spec.Shape.ofList (channels :: spatial.toList)))
      (leading.concat (Spec.Shape.ofList (channels :: spatial.toList))) :=
  let n := Spec.Shape.size leading
  let extent := Spec.Shape.size (Spec.Shape.ofList spatial.toList)
  seq!
    Implementation.spatialReshape (channels := channels) leading spatial,
    Implementation.channelFirstKernel n channels extent <|
      _root_.Runtime.Autograd.TorchLean.NN.instanceNorm2dNchw n channels extent 1
        (h_n_pos := Nat.pos_of_ne_zero (NeZero.ne n))
        (h_c_pos := Nat.pos_of_ne_zero (NeZero.ne channels))
        (h_h_pos := Nat.pos_of_ne_zero (NeZero.ne extent))
        (h_w_pos := by decide)
        cfg.seedGamma cfg.seedBeta,
    Implementation.spatialRestore (channels := channels) leading spatial

/-- Group normalization over `(leading..., channels, spatial...)` for any spatial rank. -/
def groupNorm (leading : Spec.Shape := .scalar) {d channels : Nat}
    (spatial : Vector Nat d) (groups : Nat) (hGroups : groups > 0)
    (hGroupsLe : channels ≥ groups) (hDiv : channels % groups = 0)
    (cfg : ChannelNormConfig := {}) [NeZero (Spec.Shape.size leading)] [NeZero channels]
    [NeZero (Spec.Shape.size (Spec.Shape.ofList spatial.toList))] :
    Sequential
      (leading.concat (Spec.Shape.ofList (channels :: spatial.toList)))
      (leading.concat (Spec.Shape.ofList (channels :: spatial.toList))) :=
  let n := Spec.Shape.size leading
  let extent := Spec.Shape.size (Spec.Shape.ofList spatial.toList)
  seq!
    Implementation.spatialReshape (channels := channels) leading spatial,
    Implementation.channelFirstKernel n channels extent <|
      _root_.Runtime.Autograd.TorchLean.NN.groupNorm2dNchw n channels extent 1 groups
        (h_n_pos := Nat.pos_of_ne_zero (NeZero.ne n))
        (h_c_pos := Nat.pos_of_ne_zero (NeZero.ne channels))
        (h_h_pos := Nat.pos_of_ne_zero (NeZero.ne extent))
        (h_w_pos := by decide) (h_g_pos := hGroups)
        hGroupsLe hDiv cfg.seedGamma cfg.seedBeta,
    Implementation.spatialRestore (channels := channels) leading spatial

/--
Multi-head self-attention configuration.

PyTorch analogue: `torch.nn.MultiheadAttention` (conceptually).
See `https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html`.
-/
structure MultiHeadAttention where
  /-- Number of attention heads. -/
  numHeads : Nat
  /-- Per-head embedding dimension. -/
  headDim : Nat
  /-- Base seed for deterministic parameter initialization. -/
  seedW : Nat := 0
  /-- Projection-weight initialization. `none` retains Xavier-uniform initialization. -/
  weightInit? : Option _root_.Runtime.Autograd.Torch.Init.Scheme := none
  /--
  Optional initializer for the output projection.

  This is separate because deep residual stacks commonly scale the projection that writes back to
  the residual stream. When omitted, `weightInit?` is used.
  -/
  outputWeightInit? : Option _root_.Runtime.Autograd.Torch.Init.Scheme := none
  /-- Add a trainable bias after the output projection. -/
  outputBias : Bool := false

/--
Multi-head self-attention with an explicit nonzero sequence length proof.

If `mask` is provided, it is a boolean attention mask of shape `(n × n)` (e.g. causal masking).
-/
def multiHeadAttentionWith (leading : Spec.Shape := .scalar) {n dModel : Nat}
    (cfg : MultiHeadAttention) (hN : n ≠ 0)
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    Sequential ((leading.concat (.dim n .scalar)).appendDim dModel)
      ((leading.concat (.dim n .scalar)).appendDim dModel) := by
  simpa only [Spec.Shape.concat_appendDim, Spec.Shape.appendDim] using
    (if cfg.outputBias then
      of <| adaptLeadingShape leading <|
        _root_.Runtime.Autograd.TorchLean.NN.multiHeadAttentionOutputBias
        (Spec.Shape.size leading) n dModel cfg.numHeads cfg.headDim
        (h1 := hN) (seedW := cfg.seedW) (weightInit? := cfg.weightInit?)
        (outputWeightInit? := cfg.outputWeightInit?) (mask := mask)
    else
      of <| adaptLeadingShape leading <| _root_.Runtime.Autograd.TorchLean.NN.multiHeadAttention
        (Spec.Shape.size leading) n dModel cfg.numHeads cfg.headDim
        (h1 := hN) (seedW := cfg.seedW) (weightInit? := cfg.weightInit?)
        (outputWeightInit? := cfg.outputWeightInit?) (mask := mask))

/--
Multi-head self-attention using `NeZero` to hide the nonzero sequence length proof.

If `mask` is provided, it is a boolean attention mask of shape `(n × n)` (e.g. causal masking).
-/
def multiHeadAttention (leading : Spec.Shape := .scalar) {n dModel : Nat}
    (cfg : MultiHeadAttention) [NeZero n]
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    Sequential ((leading.concat (.dim n .scalar)).appendDim dModel)
      ((leading.concat (.dim n .scalar)).appendDim dModel) :=
  multiHeadAttentionWith leading (n := n) (dModel := dModel) cfg (NeZero.ne (n := n))
    (mask := mask)
