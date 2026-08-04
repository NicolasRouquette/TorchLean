/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import NN.API.Neural.FunctionalBatch

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
  nn.of <| Implementation.adaptFlatBatch leading <|
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
  nn.of <| Implementation.adaptFlatBatch leading <|
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
  nn.of <| Implementation.adaptFlatBatch leading <|
    _root_.Runtime.Autograd.TorchLean.NN.avgPool (Spec.Shape.size leading) d channels
      cfg.kernel cfg.stride cfg.padding spatial cfg.kernelNonzero cfg.strideNonzero

/--
Global average pooling over every spatial axis, preserving the leading axes and channels.
-/
def globalAvgPool (leading : Spec.Shape := .scalar) {d channels : Nat}
    (spatial : Vector Nat d) (spatialNonzero : ∀ i : Fin d, spatial.get i ≠ 0) :
    Sequential
      (leading.concat (Spec.Shape.ofList (channels :: spatial.toList)))
      (leading.concat (.dim channels .scalar)) :=
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
  let outputShape := leading.concat (.dim channels .scalar)
  let removeSingletons : LayerDef pooledShape outputShape :=
    { kind := "GlobalAvgPool"
      paramShapes := []
      initParams := .nil
      paramRequiresGrad := []
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun x =>
            _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α) (s₁ := pooledShape) (s₂ := outputShape)
              x (by
                dsimp [pooledShape, outputShape]
                rw [Spec.poolOutSpatialPad_global spatial spatialNonzero]
                simp [Spec.Shape.size_concat, Spec.Shape.ofList,
                  Spec.Shape.size]) }
  seq! pooled, nn.of removeSingletons

/--
LayerNorm configuration for batched `(batch x seqLen x embedDim)` tensors.

PyTorch analogue: `torch.nn.LayerNorm`.
See `https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html`.
-/
structure LayerNorm where
  /-- Seed for deterministic initialization of `gamma` (scale). -/
  seedGamma : Nat := 0
  /-- Seed for deterministic initialization of `beta` (shift). -/
  seedBeta : Nat := 0

/--
Layer normalization over `(batch × seqLen × embedDim)` tensors, with explicit positivity proofs.

This matches the common Transformer usage: normalize each token’s `embedDim`-vector independently,
with learnable scale/shift parameters `gamma` and `beta`.

PyTorch analogue: `torch.nn.LayerNorm(embedDim)` applied to a tensor of shape
`(batch, seqLen, embedDim)`.

Call `nn.layerNorm` when `NeZero` can discharge the positivity proofs automatically.
-/
def layerNormWith {batch seqLen embedDim : Nat} (cfg : LayerNorm)
    (hSeq : seqLen > 0) (hEmbed : embedDim > 0) :
    Sequential (.dim batch (.dim seqLen (.dim embedDim .scalar)))
      (.dim batch (.dim seqLen (.dim embedDim .scalar))) :=
  of <| _root_.Runtime.Autograd.TorchLean.NN.layerNorm
    (batch := batch) (seqLen := seqLen) (embedDim := embedDim)
    (h_seq_pos := hSeq) (h_embed_pos := hEmbed)
    (seedGamma := cfg.seedGamma) (seedBeta := cfg.seedBeta)

/--
Layer normalization over `(batch × seqLen × embedDim)` tensors.

This normalizes each `embedDim`-vector (per batch element, per sequence position), and applies
learned affine parameters `gamma` and `beta`.

PyTorch analogue: `torch.nn.LayerNorm(embedDim)` on a tensor shaped `(batch, seqLen, embedDim)`.

Implementation note:
TorchLean uses `NeZero` to ensure `seqLen` and `embedDim` are positive, avoiding degenerate shapes.
-/
def layerNorm {batch seqLen embedDim : Nat} (cfg : LayerNorm := {})
    [NeZero seqLen] [NeZero embedDim] :
    Sequential (.dim batch (.dim seqLen (.dim embedDim .scalar)))
      (.dim batch (.dim seqLen (.dim embedDim .scalar))) :=
  layerNormWith (batch := batch) (seqLen := seqLen) (embedDim := embedDim) cfg
    (Nat.pos_of_ne_zero (NeZero.ne (n := seqLen)))
    (Nat.pos_of_ne_zero (NeZero.ne (n := embedDim)))

/--
RMSNorm configuration for batched `(batch x seqLen x embedDim)` tensors.

This is a common alternative to LayerNorm in modern transformer architectures.
-/
structure RMSNorm where
  /-- Seed for deterministic initialization of `gamma` (scale). -/
  seedGamma : Nat := 0

/--
RMS normalization over `(batch × seqLen × embedDim)` tensors, with explicit positivity proofs.

This is like LayerNorm but without mean subtraction: we scale by the root-mean-square over the
`embedDim` axis, and apply a learned scale `gamma`.

PyTorch analogue: many libraries provide an `RMSNorm(embedDim)` module; conceptually it is applied
to tensors shaped `(batch, seqLen, embedDim)`.

Call `nn.rmsNorm` when `NeZero` can discharge the positivity proofs automatically.
-/
def rmsNormWith {batch seqLen embedDim : Nat} (cfg : RMSNorm)
    (hSeq : seqLen > 0) (hEmbed : embedDim > 0) :
    Sequential (.dim batch (.dim seqLen (.dim embedDim .scalar)))
      (.dim batch (.dim seqLen (.dim embedDim .scalar))) :=
  of <| _root_.Runtime.Autograd.TorchLean.NN.rmsNorm
    (batch := batch) (seqLen := seqLen) (embedDim := embedDim)
    (h_seq_pos := hSeq) (h_embed_pos := hEmbed)
    (seedGamma := cfg.seedGamma)

/--
RMS normalization over `(batch × seqLen × embedDim)` tensors.

This normalizes by the root-mean-square over the `embedDim` axis (per batch element, per position),
then applies a learned scale `gamma`.

Implementation note:
TorchLean uses `NeZero` to ensure `seqLen` and `embedDim` are positive, avoiding degenerate shapes.
-/
def rmsNorm {batch seqLen embedDim : Nat} (cfg : RMSNorm := {})
    [NeZero seqLen] [NeZero embedDim] :
    Sequential (.dim batch (.dim seqLen (.dim embedDim .scalar)))
      (.dim batch (.dim seqLen (.dim embedDim .scalar))) :=
  rmsNormWith (batch := batch) (seqLen := seqLen) (embedDim := embedDim) cfg
    (Nat.pos_of_ne_zero (NeZero.ne (n := seqLen)))
    (Nat.pos_of_ne_zero (NeZero.ne (n := embedDim)))

/-- Parameter initialization for affine channel normalization. -/
structure ChannelNorm where
  seedGamma : Nat := 0
  seedBeta : Nat := 0

namespace Implementation

/-- A checked reshape layer used internally to flatten and restore spatial axes. -/
def reshapeLayer (source target : Spec.Shape)
    (sameSize : Spec.Shape.size source = Spec.Shape.size target) :
    LayerDef source target :=
  { kind := "Reshape"
    paramShapes := []
    initParams := .nil
    paramRequiresGrad := []
    forward := fun _ {α} _ _ => fun {m} _ _ => fun x =>
      _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α) (s₁ := source) (s₂ := target) x sameSize }

/-- Apply a channel-first normalization kernel after adding its singleton trailing axis. -/
def channelFirstKernel (leadingSize channels spatialSize : Nat)
    (kernel : LayerDef
      (.dim leadingSize (.dim channels (.dim spatialSize (.dim 1 .scalar))))
      (.dim leadingSize (.dim channels (.dim spatialSize (.dim 1 .scalar))))) :
    Sequential
      (.dim leadingSize (.dim channels (.dim spatialSize .scalar)))
      (.dim leadingSize (.dim channels (.dim spatialSize .scalar))) :=
  let source := .dim leadingSize (.dim channels (.dim spatialSize .scalar))
  let target := .dim leadingSize (.dim channels (.dim spatialSize (.dim 1 .scalar)))
  seq!
    of (reshapeLayer source target (by simp [source, target, Spec.Shape.size])),
    of kernel,
    of (reshapeLayer target source (by simp [source, target, Spec.Shape.size]))

/-- Flatten arbitrary spatial axes to the channel-first kernel representation. -/
def spatialReshape {d channels : Nat} (leading : Spec.Shape)
    (spatial : Vector Nat d) :
    Sequential
      (leading.concat (Spec.Shape.ofList (channels :: spatial.toList)))
      (.dim (Spec.Shape.size leading)
        (.dim channels (.dim (Spec.Shape.size (Spec.Shape.ofList spatial.toList)) .scalar))) :=
  of <| reshapeLayer _ _ (by simp [Spec.Shape.size_concat, Spec.Shape.ofList, Spec.Shape.size])

/-- Restore the original spatial axes after channel normalization. -/
def spatialRestore {d channels : Nat} (leading : Spec.Shape)
    (spatial : Vector Nat d) :
    Sequential
      (.dim (Spec.Shape.size leading)
        (.dim channels (.dim (Spec.Shape.size (Spec.Shape.ofList spatial.toList)) .scalar)))
      (leading.concat (Spec.Shape.ofList (channels :: spatial.toList))) :=
  of <| reshapeLayer _ _ (by simp [Spec.Shape.size_concat, Spec.Shape.ofList, Spec.Shape.size])

end Implementation

/-- Batch normalization over `(leading..., channels, spatial...)` for any spatial rank. -/
def batchNorm (leading : Spec.Shape := .scalar) {d channels : Nat}
    (spatial : Vector Nat d) (cfg : ChannelNorm := {})
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
    (spatial : Vector Nat d) (cfg : ChannelNorm := {})
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
    (cfg : ChannelNorm := {}) [NeZero (Spec.Shape.size leading)] [NeZero channels]
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
structure MultiheadAttention where
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
def multiheadAttentionWith {batch n dModel : Nat} (cfg : MultiheadAttention) (hN : n ≠ 0)
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    Sequential (.dim batch (.dim n (.dim dModel .scalar))) (.dim batch (.dim n (.dim dModel .scalar)))
      :=
  if cfg.outputBias then
    of <| _root_.Runtime.Autograd.TorchLean.NN.multiHeadAttentionOutputBias
      batch n dModel cfg.numHeads cfg.headDim
      (h1 := hN) (seedW := cfg.seedW) (weightInit? := cfg.weightInit?)
      (outputWeightInit? := cfg.outputWeightInit?) (mask := mask)
  else
    of <| _root_.Runtime.Autograd.TorchLean.NN.multiHeadAttention
      batch n dModel cfg.numHeads cfg.headDim
      (h1 := hN) (seedW := cfg.seedW) (weightInit? := cfg.weightInit?)
      (outputWeightInit? := cfg.outputWeightInit?) (mask := mask)

/--
Multi-head self-attention using `NeZero` to hide the nonzero sequence length proof.

If `mask` is provided, it is a boolean attention mask of shape `(n × n)` (e.g. causal masking).
-/
def multiheadAttention {batch n dModel : Nat} (cfg : MultiheadAttention) [NeZero n]
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    Sequential (.dim batch (.dim n (.dim dModel .scalar))) (.dim batch (.dim n (.dim dModel .scalar)))
      :=
  multiheadAttentionWith (batch := batch) (n := n) (dModel := dModel) cfg (NeZero.ne (n := n))
    (mask := mask)
