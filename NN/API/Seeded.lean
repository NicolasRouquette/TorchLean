/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Rand
public import NN.Tensor
public import NN.API.Neural.Heads
public import NN.API.Neural.Indexed
public import NN.API.Neural.Positional
public import NN.API.Neural.Layers

@[expose] public section

namespace TorchLean

/-!
# Seeded model builders

Layer constructors draw deterministic initialization seeds from an explicit seed stream.
-/

namespace nn

universe u

/-- Deterministic model builder that threads an explicit initialization seed stream. -/
abbrev Builder := rand.SeedM

/-!
## Model Builders and Seeding

TorchLean keeps initialization randomness explicit so examples are reproducible.

Layer constructors return `nn.Builder`, a deterministic state computation over the initialization
seed stream. Call `nn.build seed` to construct a model reproducibly.

Note: `nn.Sequential` lives in `Type 2`, so it cannot be returned directly from `IO`. We keep
model building pure by drawing a base seed in `IO` and then calling `nn.build`.
-/

/-
Configuration records are available at `nn.*`; constructors with explicit seed arguments remain
under `nn.Internal`.
-/

export Internal
  (Linear LearnedPositionalEmbedding SinusoidalPositionalEncoding RotaryEmbeddingConfig
   ConvGeometry Conv ConvTranspose Pool LayerNormConfig RmsNormConfig ChannelNormConfig
   MultiHeadAttention)

namespace ConvGeometry
export Internal.ConvGeometry (outSpatial samePadding outSpatial_samePadding toConv)
end ConvGeometry

/-- Build global average pooling over the supplied nonempty spatial dimensions without consuming a seed. -/
def globalAvgPool (leading : List Nat := []) {d channels : Nat}
    (spatial : Tensor Nat [d]) (spatialNonzero : ∀ i : Fin d, spatial.getScalar i ≠ 0) :
    Builder (Sequential
      (leading ++ channels :: spatial.toList)
      (leading ++ [channels])) :=
  pure <| Internal.globalAvgPool leading (channels := channels) spatial spatialNonzero

namespace functional
export _root_.Runtime.Autograd.TorchLean.F
  (square checkpoint
   exp log scale shift affine
   detach
   addB mulB
   embedding mean
   dropoutSeeded)
end functional

namespace blocks
export Internal.blocks
  (activation MlpConfig
   ConvAct convAct ConvActPool convActPool
   residualBlock
   TransformerEncoderBlock transformerEncoderBlock
   TransformerEncoderStack transformerEncoderStack
   transformerEncoderClassifier
   residual residualLayer
   addBranches addBranchesLayer
   concatBranches concatBranchesLayer)
end blocks

namespace heads
export Internal.heads (classifier regressor)
end heads

/-!
## Default Builders

The `nn.*` constructors allocate initialization seeds through `nn.Builder`.

-/

open Spec

/-- Build a value from a deterministic initialization seed. -/
def build {α : Type u} (seed : Nat) (x : Builder α) : α :=
  (x (rand.SeedStream.init seed)).1

/-- Apply a model independently over an arbitrary list of leading dimensions. -/
def mapEach (leading : List Nat) {source target : List Nat}
    (model : Sequential source target) :
    Builder (Sequential (leading ++ source) (leading ++ target)) := by
  simpa only [Shape.ofList_append] using
    (pure <| Internal.mapEach (Shape.ofList leading) model)

/-- Consume one initialization seed and continue building in the same result universe. -/
def withSeed {α : Type u} (k : Nat → Builder α) : Builder α :=
  fun state =>
    let (seed, state') := rand.SeedStream.next state
    k seed state'

/-- Build an elementwise ReLU layer without consuming an initialization seed. -/
def relu {shape : List Nat} : Builder (Sequential shape shape) :=
  pure (Internal.relu (s := Shape.ofList shape))

/-- Build an elementwise SiLU layer without consuming an initialization seed. -/
def silu {shape : List Nat} : Builder (Sequential shape shape) :=
  pure (Internal.silu (s := Shape.ofList shape))

/-- Build an elementwise GELU layer without consuming an initialization seed. -/
def gelu {shape : List Nat} : Builder (Sequential shape shape) :=
  pure (Internal.gelu (s := Shape.ofList shape))

/-- Build an elementwise sigmoid layer without consuming an initialization seed. -/
def sigmoid {shape : List Nat} : Builder (Sequential shape shape) :=
  pure (Internal.sigmoid (s := Shape.ofList shape))

/-- Build an elementwise hyperbolic-tangent layer without consuming an initialization seed. -/
def tanh {shape : List Nat} : Builder (Sequential shape shape) :=
  pure (Internal.tanh (s := Shape.ofList shape))

/-- Build a softmax layer along any valid tensor dimension without consuming a seed. -/
def softmax {shape : List Nat} (axis : Nat)
    [Shape.AxisInBounds axis (Shape.ofList shape)] :
    Builder (Sequential shape shape) :=
  pure (Internal.softmax (s := Shape.ofList shape) axis)

/-- Build a stable log-softmax layer along any valid tensor dimension without consuming a seed. -/
def logSoftmax {shape : List Nat} (axis : Nat)
    [Shape.AxisInBounds axis (Shape.ofList shape)] :
    Builder (Sequential shape shape) :=
  pure (Internal.logSoftmax (s := Shape.ofList shape) axis)

/-- Build a reduction that sums every tensor entry to a scalar. -/
def sum {shape : List Nat} : Builder (Sequential shape ([] : List Nat)) :=
  pure (Internal.sum (s := Shape.ofList shape))

/-- Build a layer that flattens the entire input shape into one vector. -/
def flatten {shape : List Nat} : Builder (Sequential shape [shape.prod]) := by
  simpa only [Shape.size_ofList] using
    (pure <| Internal.flatten (s := Shape.ofList shape))

/-- Build a checked reshape between two ordinary dimension lists. -/
def reshape (source target : List Nat) (sameSize : source.prod = target.prod) :
    Builder (Sequential source target) :=
  pure (Internal.reshape source target (by simpa using sameSize))

/-- Flatten each tensor after an arbitrary collection of leading dimensions. -/
def flattenAfter (leading : List Nat := []) {s : List Nat} :
    Builder (Sequential (leading ++ s) (leading ++ [s.prod])) :=
  pure (Internal.flattenAfter leading (shape := s))

/-- Build max pooling over arbitrary spatial rank using the supplied pooling configuration. -/
def maxPool (leading : List Nat := []) {d channels : Nat} (spatial : Tensor Nat [d])
    (cfg : Pool d) :
    Builder (Sequential
      (leading ++ channels :: spatial.toList)
      (leading ++ channels ::
        (Spec.poolOutSpatialPad spatial cfg.kernel cfg.stride cfg.padding).toList)) :=
  pure (Internal.maxPool leading (channels := channels) spatial cfg)

/-- Build average pooling over arbitrary spatial rank using the supplied pooling configuration. -/
def avgPool (leading : List Nat := []) {d channels : Nat} (spatial : Tensor Nat [d])
    (cfg : Pool d) :
    Builder (Sequential
      (leading ++ channels :: spatial.toList)
      (leading ++ channels ::
        (Spec.poolOutSpatialPad spatial cfg.kernel cfg.stride cfg.padding).toList)) :=
  pure (Internal.avgPool leading (channels := channels) spatial cfg)

/-- Build a transpose convolution over arbitrary spatial rank. -/
def convTranspose (leading : List Nat := []) {d inChannels : Nat}
    (spatial : Tensor Nat [d]) (cfg : ConvTranspose d) [NeZero inChannels] :
    Builder (Sequential
      (leading ++ inChannels :: spatial.toList)
      (leading ++ cfg.outChannels ::
        (Spec.convTransposeOutSpatial spatial cfg.kernel cfg.stride cfg.padding).toList)) :=
  withSeed fun seedKernel =>
    withSeed fun seedBias =>
      pure <| Internal.convTranspose leading spatial cfg seedKernel seedBias

/-- Build an affine layer, consuming independent seeds for its weight and bias initializers. -/
def linear (inDim outDim : Nat) (leading : List Nat := []) :
    Builder (Sequential (leading ++ [inDim]) (leading ++ [outDim])) :=
  withSeed fun seedW =>
    withSeed fun seedB =>
      pure <| Internal.linear inDim outDim seedW seedB (leading := leading)

/-- Seeded affine layer with an explicit initialization policy. -/
def linearWith (inDim outDim : Nat) (cfg : Linear) (leading : List Nat := []) :
    Builder (Sequential (leading ++ [inDim]) (leading ++ [outDim])) :=
  withSeed fun seedW =>
    withSeed fun seedB =>
      pure <| Internal.linearWith inDim outDim cfg seedW seedB (leading := leading)

/--
Build a seeded recurrent neural network over a fixed sequence length.

The recurrent computation acts independently over every index in `leading`; its parameters are
shared across those indices. The scalar default is a single sequence, while a shape such as
`[batch]` gives the usual batched model.
-/
def rnn (seqLen inputSize hiddenSize : Nat) (leading : List Nat := []) :
    Builder (Sequential
      (leading ++ [seqLen, inputSize])
      (leading ++ [seqLen, hiddenSize])) :=
  withSeed fun seedW =>
    withSeed fun seedB =>
      mapEach leading <| Internal.rnn seqLen inputSize hiddenSize seedW seedB

/-- Build a seeded gated recurrent unit, shared over every index in `leading`. -/
def gru (seqLen inputSize hiddenSize : Nat) (leading : List Nat := []) :
    Builder (Sequential
      (leading ++ [seqLen, inputSize])
      (leading ++ [seqLen, hiddenSize])) :=
  withSeed fun seedW =>
    withSeed fun seedB =>
      mapEach leading <| Internal.gru seqLen inputSize hiddenSize seedW seedB

/-- Build a seeded Mamba-style state-space layer, shared over every index in `leading`. -/
def mamba (seqLen inputSize hiddenSize : Nat) (leading : List Nat := []) :
    Builder (Sequential
      (leading ++ [seqLen, inputSize])
      (leading ++ [seqLen, hiddenSize])) :=
  withSeed fun seedW =>
    withSeed fun seedB =>
      mapEach leading <| Internal.mamba seqLen inputSize hiddenSize seedW seedB

/-- Build a seeded long short-term memory layer, shared over every index in `leading`. -/
def lstm (seqLen inputSize hiddenSize : Nat) (leading : List Nat := []) :
    Builder (Sequential
      (leading ++ [seqLen, inputSize])
      (leading ++ [seqLen, hiddenSize])) :=
  withSeed fun seedW =>
    withSeed fun seedB =>
      mapEach leading <| Internal.lstm seqLen inputSize hiddenSize seedW seedB

/-- Build an arbitrary-rank convolution, allocating separate kernel and bias seeds. -/
def conv (leading : List Nat := []) {d inChannels : Nat} (spatial : Tensor Nat [d])
    (cfg : Conv d) [NeZero inChannels] :
    Builder (Sequential
      (leading ++ inChannels :: spatial.toList)
      (leading ++ cfg.outChannels ::
        (Spec.convOutSpatial spatial cfg.kernel cfg.stride cfg.padding).toList)) :=
  withSeed fun seedKernel =>
    withSeed fun seedBias =>
      pure <| Internal.conv leading (inChannels := inChannels) spatial cfg seedKernel seedBias

/--
Build a pointwise convolution over any spatial rank.

The unit kernel, unit stride, and zero padding preserve every nonempty spatial axis. This is the
common channel-projection operation used by residual, diffusion, and encoder-decoder models.
-/
def pointwiseConv (leading : List Nat := []) {d inChannels : Nat}
    (spatial : Tensor Nat [d]) (spatialNonzero : ∀ i : Fin d, spatial.getScalar i ≠ 0)
    (outChannels : Nat) [NeZero inChannels] :
    Builder (Sequential
      (leading ++ inChannels :: spatial.toList)
      (leading ++ outChannels :: spatial.toList)) := by
  simpa [Spec.convOutSpatial_unit spatial spatialNonzero] using
    (conv (leading := leading) (inChannels := inChannels) spatial
      { outChannels := outChannels
        kernel := Spec.fill 1 [d]
        stride := Spec.fill 1 [d]
        padding := Spec.fill 0 [d]
        kernelNonzero := by intro i; simp [Spec.Tensor.getScalar, Spec.fill]
        strideNonzero := by intro i; simp [Spec.Tensor.getScalar, Spec.fill] })

/-- Build batch normalization with seeded scale and offset parameters. -/
def batchNorm (leading : List Nat := []) {d channels : Nat}
    (spatial : Tensor Nat [d])
    [NeZero leading.prod] [NeZero channels] [NeZero spatial.prod] :
    Builder (Sequential
      (leading ++ channels :: spatial.toList)
      (leading ++ channels :: spatial.toList)) :=
  withSeed fun seedGamma =>
    withSeed fun seedBeta =>
      pure <| Internal.batchNorm leading spatial
        { seedGamma := seedGamma, seedBeta := seedBeta }

/-- Build instance normalization with seeded scale and offset parameters. -/
def instanceNorm (leading : List Nat := []) {d channels : Nat}
    (spatial : Tensor Nat [d])
    [NeZero leading.prod] [NeZero channels] [NeZero spatial.prod] :
    Builder (Sequential
      (leading ++ channels :: spatial.toList)
      (leading ++ channels :: spatial.toList)) :=
  withSeed fun seedGamma =>
    withSeed fun seedBeta =>
      pure <| Internal.instanceNorm leading spatial
        { seedGamma := seedGamma, seedBeta := seedBeta }

/-- Build group normalization after checking the positive group count and channel divisibility. -/
def groupNorm (leading : List Nat := []) {d channels : Nat}
    (spatial : Tensor Nat [d]) (groups : Nat) (hGroups : groups > 0)
    (hGroupsLe : channels >= groups) (hDiv : channels % groups = 0)
    [NeZero leading.prod] [NeZero channels] [NeZero spatial.prod] :
    Builder (Sequential
      (leading ++ channels :: spatial.toList)
      (leading ++ channels :: spatial.toList)) :=
  withSeed fun seedGamma =>
    withSeed fun seedBeta =>
      pure <| Internal.groupNorm leading spatial groups hGroups hGroupsLe hDiv
        { seedGamma := seedGamma, seedBeta := seedBeta }

/-- Build an embedding lookup layer from a freshly seeded embedding table. -/
def oneHotEmbedding (vocab embedDim : Nat) (cfg : Embedding.Config := {})
    {leading : List Nat} :
    Builder (Sequential (leading ++ [vocab]) (leading ++ [embedDim])) :=
  withSeed fun seed =>
    pure <| Internal.oneHotEmbedding vocab embedDim { cfg with seed := seed }
      (leading := leading)

/-- Build a trainable lookup table for a tensor of natural-number indices. -/
def embedding (vocab embedDim : Nat) (cfg : Embedding.Config := {}) :
    Builder (Embedding vocab embedDim) :=
  withSeed fun seed =>
    pure <| Internal.embedding vocab embedDim { cfg with seed := seed }

/-- Build deterministic sinusoidal positional encoding over a sequence suffix. -/
def sinusoidalPositionalEncoding (leading : List Nat := []) {seqLen embedDim : Nat}
    (cfg : SinusoidalPositionalEncoding := {}) :
    Builder (Sequential
      (leading ++ [seqLen, embedDim])
      (leading ++ [seqLen, embedDim])) :=
  pure <| Internal.sinusoidalPositionalEncoding leading
    (seqLen := seqLen) (embedDim := embedDim) cfg

/-- Build deterministic rotary positional encoding for multi-head sequence features. -/
def rope (leading : List Nat := []) {seqLen headDim : Nat}
    (cfg : RotaryEmbeddingConfig := {}) :
    Builder (Sequential
      (leading ++ [seqLen, headDim])
      (leading ++ [seqLen, headDim])) :=
  pure <| Internal.rope leading (seqLen := seqLen) (headDim := headDim) cfg

/-- Build learned positional embeddings from a freshly allocated parameter seed. -/
def learnedPositionalEmbedding (leading : List Nat := []) {seqLen embedDim : Nat}
    (cfg : LearnedPositionalEmbedding := {}) :
    Builder (Sequential
      (leading ++ [seqLen, embedDim])
      (leading ++ [seqLen, embedDim])) :=
  withSeed fun seedPos =>
    pure <| Internal.learnedPositionalEmbedding leading
      (seqLen := seqLen) (embedDim := embedDim) { cfg with seedPos := seedPos }

/-- Build final-axis layer normalization with independently seeded scale and offset parameters. -/
def layerNorm (leading : List Nat := []) {width : Nat} [NeZero width] :
    Builder (Sequential (leading ++ [width]) (leading ++ [width])) :=
  withSeed fun seedGamma =>
    withSeed fun seedBeta =>
      pure <| Internal.layerNorm leading (width := width)
        { seedGamma := seedGamma, seedBeta := seedBeta }

/-- Build final-axis RMS normalization with a freshly seeded scale parameter. -/
def rmsNorm (leading : List Nat := []) {width : Nat} [NeZero width] :
    Builder (Sequential (leading ++ [width]) (leading ++ [width])) :=
  withSeed fun seedGamma =>
    pure <| Internal.rmsNorm leading (width := width) { seedGamma := seedGamma }

/-- Build seeded multi-head self-attention with an optional fixed attention mask. -/
def multiHeadAttention (leading : List Nat := []) {n dModel : Nat} [NeZero n]
    (cfg : MultiHeadAttention)
    (mask : Option (Tensor Bool [n, n]) := none) :
    Builder (Sequential
      (leading ++ [n, dModel])
      (leading ++ [n, dModel])) :=
  withSeed fun seedW =>
    pure <| Internal.multiHeadAttention leading (n := n) (dModel := dModel)
      { cfg with seedW := seedW } (mask := mask)

/-- Build one seeded transformer encoder block, optionally applying a fixed attention mask. -/
def transformerEncoderBlock (leading : List Nat := []) {n dModel : Nat}
    [NeZero n] [NeZero dModel]
    (cfg : blocks.TransformerEncoderBlock)
    (mask : Option (Tensor Bool [n, n]) := none) :
    Builder (Sequential
      (leading ++ [n, dModel])
      (leading ++ [n, dModel])) :=
  withSeed fun seedBase =>
    pure <| Internal.blocks.transformerEncoderBlock leading
      (n := n) (dModel := dModel) { cfg with seedBase := seedBase }
      (mask := mask)

/-- Build a seeded stack of transformer encoder blocks with an optional attention mask. -/
def transformerEncoderStack (leading : List Nat := []) {n dModel : Nat}
    [NeZero n] [NeZero dModel]
    (cfg : blocks.TransformerEncoderStack)
    (mask : Option (Tensor Bool [n, n]) := none) :
    Builder (Sequential
      (leading ++ [n, dModel])
      (leading ++ [n, dModel])) :=
  withSeed fun seedBase =>
    pure <| Internal.blocks.transformerEncoderStack leading
      (n := n) (dModel := dModel) { cfg with seedBase := seedBase }
      (mask := mask)

/-- Build dropout with a fresh deterministic mask seed from the builder stream. -/
def dropout {shape : List Nat} (p : Float) : Builder (Sequential shape shape) :=
  withSeed fun seed =>
    pure <| Internal.dropout (s := Shape.ofList shape) p (seed := seed)

namespace blocks

/--
Build a multilayer perceptron over any collection of leading dimensions.

Each hidden width contributes a linear layer followed by the configured activation and optional
dropout. Initialization seeds come from the surrounding `Builder` seed stream.
-/
def mlp (inDim outDim : Nat) (cfg : MlpConfig := {})
    (leading : List Nat := []) :
    Builder (Sequential (leading ++ [inDim]) (leading ++ [outDim])) :=
  withSeed fun seed =>
    pure <| Internal.blocks.mlpWithSeed inDim outDim seed cfg leading

end blocks

/--
Build a model using the next global seed, then run a continuation.

`nn.Sequential` lives in `Type 2`, so executable code passes the model to a continuation rather than
returning it directly from `IO`.
-/
def withModel {inputShape outputShape : List Nat} {β : Type}
    (mk : Builder (Sequential inputShape outputShape))
    (k : Sequential inputShape outputShape → IO β) : IO β := do
  let seed ← rand.nextSeedGlobal
  let model := build seed mk
  k model

end nn

end TorchLean
