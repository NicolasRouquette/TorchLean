/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Rand
public import NN.API.Neural.Heads

import Mathlib.Algebra.Order.Algebra

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
   Conv Pool LayerNormConfig RmsNormConfig ChannelNormConfig MultiHeadAttention)

/-- Build global average pooling over the supplied nonempty spatial dimensions without consuming a seed. -/
def globalAvgPool (leading : Spec.Shape := .scalar) {d channels : Nat}
    (spatial : Vector Nat d) (spatialNonzero : ∀ i : Fin d, spatial.get i ≠ 0) :
    Builder (Sequential
      (leading.concat (Spec.Shape.ofList (channels :: spatial.toList)))
      (leading.appendDim channels)) :=
  fun state =>
    (Internal.globalAvgPool leading (channels := channels) spatial spatialNonzero, state)

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
   addBranches addBranchesLayer)
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

/-- Lift a pure value into the seeded builder (consumes no seeds). -/
def lift {α : Type u} (x : α) : Builder α :=
  pure x

/-- Apply a model independently over an arbitrary collection of leading dimensions. -/
def mapLeading (leading : Spec.Shape) {σ τ : Spec.Shape} (model : Sequential σ τ) :
    Builder (Sequential (leading.concat σ) (leading.concat τ)) :=
  lift (Internal.mapLeading leading model)

/-- Consume one initialization seed and continue building in the same result universe. -/
def withSeed {α : Type u} (k : Nat → Builder α) : Builder α :=
  fun state =>
    let (seed, state') := rand.SeedStream.next state
    k seed state'

/-- Build an elementwise ReLU layer without consuming an initialization seed. -/
def relu {s : Spec.Shape} : Builder (Sequential s s) :=
  lift (Internal.relu (s := s))

/-- Build an elementwise SiLU layer without consuming an initialization seed. -/
def silu {s : Spec.Shape} : Builder (Sequential s s) :=
  lift (Internal.silu (s := s))

/-- Build an elementwise GELU layer without consuming an initialization seed. -/
def gelu {s : Spec.Shape} : Builder (Sequential s s) :=
  lift (Internal.gelu (s := s))

/-- Build an elementwise sigmoid layer without consuming an initialization seed. -/
def sigmoid {s : Spec.Shape} : Builder (Sequential s s) :=
  lift (Internal.sigmoid (s := s))

/-- Build an elementwise hyperbolic-tangent layer without consuming an initialization seed. -/
def tanh {s : Spec.Shape} : Builder (Sequential s s) :=
  lift (Internal.tanh (s := s))

/-- Build a softmax layer along any valid tensor dimension without consuming a seed. -/
def softmax {s : Spec.Shape} (axis : Nat) [Spec.Shape.AxisInBounds axis s] :
    Builder (Sequential s s) :=
  lift (Internal.softmax (s := s) axis)

/-- Build a stable log-softmax layer along any valid tensor dimension without consuming a seed. -/
def logSoftmax {s : Spec.Shape} (axis : Nat) [Spec.Shape.AxisInBounds axis s] :
    Builder (Sequential s s) :=
  lift (Internal.logSoftmax (s := s) axis)

/-- Build a reduction that sums every tensor entry to a scalar. -/
def sum {s : Spec.Shape} : Builder (Sequential s Spec.Shape.scalar) :=
  lift (Internal.sum (s := s))

/-- Build a layer that flattens the entire input shape into one vector. -/
def flatten {s : Spec.Shape} : Builder (Sequential s (.dim (Spec.Shape.size s) .scalar)) :=
  lift (Internal.flatten (s := s))

/-- Build a checked reshape without consuming an initialization seed. -/
def reshape (source target : Spec.Shape)
    (sameSize : Spec.Shape.size source = Spec.Shape.size target) :
    Builder (Sequential source target) :=
  lift (Internal.reshape source target sameSize)

/-- Flatten each tensor after an arbitrary collection of leading dimensions. -/
def flattenLeading (leading : Spec.Shape := .scalar) {s : Spec.Shape} :
    Builder (Sequential (leading.concat s) (leading.appendDim (Spec.Shape.size s))) :=
  lift (Internal.flattenLeading leading (s := s))

/-- Build max pooling over arbitrary spatial rank using the supplied pooling configuration. -/
def maxPool (leading : Spec.Shape := .scalar) {d channels : Nat} (spatial : Vector Nat d)
    (cfg : Pool d) :=
  lift (Internal.maxPool leading (channels := channels) spatial cfg)

/-- Build average pooling over arbitrary spatial rank using the supplied pooling configuration. -/
def avgPool (leading : Spec.Shape := .scalar) {d channels : Nat} (spatial : Vector Nat d)
    (cfg : Pool d) :=
  lift (Internal.avgPool leading (channels := channels) spatial cfg)

/-- Build an affine layer, consuming independent seeds for its weight and bias initializers. -/
def linear (inDim outDim : Nat) (leading : Spec.Shape := Spec.Shape.scalar) :
    Builder (Sequential (leading.appendDim inDim) (leading.appendDim outDim)) :=
  withSeed fun seedW =>
    withSeed fun seedB =>
      pure (Internal.linear inDim outDim seedW seedB (leading := leading))

/-- Seeded affine layer with an explicit initialization policy. -/
def linearWith (inDim outDim : Nat) (cfg : Linear) (leading : Spec.Shape := Spec.Shape.scalar) :
    Builder (Sequential (leading.appendDim inDim) (leading.appendDim outDim)) :=
  withSeed fun seedW =>
    withSeed fun seedB =>
      pure (Internal.linearWith inDim outDim cfg seedW seedB (leading := leading))

namespace deterministic

/-- Construct a linear layer with explicit parameter-initialization seeds. -/
def linear (inDim outDim : Nat) (seedWeight seedBias : Nat)
    (leading : Spec.Shape := .scalar) :
    Sequential (leading.appendDim inDim) (leading.appendDim outDim) :=
  Internal.linear inDim outDim seedWeight seedBias (leading := leading)

end deterministic

/--
Build a seeded recurrent neural network over a fixed sequence length.

The recurrent computation acts independently over every index in `leading`; its parameters are
shared across those indices. The scalar default is a single sequence, while a shape such as
`shape![batch]` gives the usual batched model.
-/
def rnn (seqLen inputSize hiddenSize : Nat) (leading : Spec.Shape := .scalar) :
    Builder (Sequential
      (leading.concat (.dim seqLen (.dim inputSize .scalar)))
      (leading.concat (.dim seqLen (.dim hiddenSize .scalar)))) :=
  withSeed fun seedW =>
    withSeed fun seedB =>
      pure <| Internal.mapLeading leading <| Internal.rnn seqLen inputSize hiddenSize seedW seedB

/-- Build a seeded gated recurrent unit, shared over every index in `leading`. -/
def gru (seqLen inputSize hiddenSize : Nat) (leading : Spec.Shape := .scalar) :
    Builder (Sequential
      (leading.concat (.dim seqLen (.dim inputSize .scalar)))
      (leading.concat (.dim seqLen (.dim hiddenSize .scalar)))) :=
  withSeed fun seedW =>
    withSeed fun seedB =>
      pure <| Internal.mapLeading leading <| Internal.gru seqLen inputSize hiddenSize seedW seedB

/-- Build a seeded Mamba-style state-space layer, shared over every index in `leading`. -/
def mamba (seqLen inputSize hiddenSize : Nat) (leading : Spec.Shape := .scalar) :
    Builder (Sequential
      (leading.concat (.dim seqLen (.dim inputSize .scalar)))
      (leading.concat (.dim seqLen (.dim hiddenSize .scalar)))) :=
  withSeed fun seedW =>
    withSeed fun seedB =>
      pure <| Internal.mapLeading leading <| Internal.mamba seqLen inputSize hiddenSize seedW seedB

/-- Build a seeded long short-term memory layer, shared over every index in `leading`. -/
def lstm (seqLen inputSize hiddenSize : Nat) (leading : Spec.Shape := .scalar) :
    Builder (Sequential
      (leading.concat (.dim seqLen (.dim inputSize .scalar)))
      (leading.concat (.dim seqLen (.dim hiddenSize .scalar)))) :=
  withSeed fun seedW =>
    withSeed fun seedB =>
      pure <| Internal.mapLeading leading <| Internal.lstm seqLen inputSize hiddenSize seedW seedB

/-- Build an arbitrary-rank convolution, allocating separate kernel and bias seeds. -/
def conv (leading : Spec.Shape := .scalar) {d inChannels : Nat} (spatial : Vector Nat d)
    (cfg : Conv d) [NeZero inChannels] :
    Builder (Sequential
      (leading.concat (Spec.Shape.ofList (inChannels :: spatial.toList)))
      (leading.concat (Spec.Shape.ofList
        (cfg.outChannels ::
          (Spec.convOutSpatial spatial cfg.kernel cfg.stride cfg.padding).toList)))) :=
  withSeed fun seedKernel =>
    withSeed fun seedBias =>
      let cfg' : Conv d := { cfg with seedKernel := seedKernel, seedBias := seedBias }
      pure <| by
        simpa [cfg'] using Internal.conv leading (inChannels := inChannels) spatial cfg'

/-- Build batch normalization with seeded scale and offset parameters. -/
def batchNorm (leading : Spec.Shape := .scalar) {d channels : Nat}
    (spatial : Vector Nat d) (cfg : ChannelNormConfig := {})
    [NeZero (Spec.Shape.size leading)] [NeZero channels]
    [NeZero (Spec.Shape.size (Spec.Shape.ofList spatial.toList))] :
    Builder (Sequential
      (leading.concat (Spec.Shape.ofList (channels :: spatial.toList)))
      (leading.concat (Spec.Shape.ofList (channels :: spatial.toList)))) :=
  withSeed fun seedGamma =>
    withSeed fun seedBeta =>
      pure <| Internal.batchNorm leading spatial
        { cfg with seedGamma := seedGamma, seedBeta := seedBeta }

/-- Build instance normalization with seeded scale and offset parameters. -/
def instanceNorm (leading : Spec.Shape := .scalar) {d channels : Nat}
    (spatial : Vector Nat d) (cfg : ChannelNormConfig := {})
    [NeZero (Spec.Shape.size leading)] [NeZero channels]
    [NeZero (Spec.Shape.size (Spec.Shape.ofList spatial.toList))] :
    Builder (Sequential
      (leading.concat (Spec.Shape.ofList (channels :: spatial.toList)))
      (leading.concat (Spec.Shape.ofList (channels :: spatial.toList)))) :=
  withSeed fun seedGamma =>
    withSeed fun seedBeta =>
      pure <| Internal.instanceNorm leading spatial
        { cfg with seedGamma := seedGamma, seedBeta := seedBeta }

/-- Build group normalization after checking the positive group count and channel divisibility. -/
def groupNorm (leading : Spec.Shape := .scalar) {d channels : Nat}
    (spatial : Vector Nat d) (groups : Nat) (hGroups : groups > 0)
    (hGroupsLe : channels >= groups) (hDiv : channels % groups = 0)
    (cfg : ChannelNormConfig := {}) [NeZero (Spec.Shape.size leading)] [NeZero channels]
    [NeZero (Spec.Shape.size (Spec.Shape.ofList spatial.toList))] :
    Builder (Sequential
      (leading.concat (Spec.Shape.ofList (channels :: spatial.toList)))
      (leading.concat (Spec.Shape.ofList (channels :: spatial.toList)))) :=
  withSeed fun seedGamma =>
    withSeed fun seedBeta =>
      pure <| Internal.groupNorm leading spatial groups hGroups hGroupsLe hDiv
        { cfg with seedGamma := seedGamma, seedBeta := seedBeta }

/-- Build an embedding lookup layer from a freshly seeded embedding table. -/
def oneHotEmbedding (vocab embedDim : Nat) (cfg : Embedding.Config := {}) {leading : Spec.Shape} :
    Builder (Sequential (leading.appendDim vocab) (leading.appendDim embedDim)) :=
  withSeed fun seed =>
    pure (Internal.oneHotEmbedding vocab embedDim { cfg with seed := seed } (leading := leading))

/-- Build a trainable lookup table for a tensor of natural-number indices. -/
def embedding (vocab embedDim : Nat) (cfg : Embedding.Config := {}) :
    Builder (Embedding vocab embedDim) :=
  withSeed fun seed =>
    pure (Internal.embedding vocab embedDim { cfg with seed := seed })

/-- Build deterministic sinusoidal positional encoding over a sequence suffix. -/
def sinusoidalPositionalEncoding (leading : Spec.Shape := .scalar) {seqLen embedDim : Nat}
    (cfg : SinusoidalPositionalEncoding := {}) :
    Builder (Sequential
      (leading.concat (.dim seqLen (.dim embedDim .scalar)))
      (leading.concat (.dim seqLen (.dim embedDim .scalar)))) :=
  lift <|
    Internal.sinusoidalPositionalEncoding leading (seqLen := seqLen) (embedDim := embedDim) cfg

/-- Build deterministic rotary positional encoding for multi-head sequence features. -/
def rope (leading : Spec.Shape := .scalar) {seqLen headDim : Nat}
    (cfg : RotaryEmbeddingConfig := {}) :
    Builder (Sequential
      (leading.concat (.dim seqLen (.dim headDim .scalar)))
      (leading.concat (.dim seqLen (.dim headDim .scalar)))) :=
  lift <|
    Internal.rope leading (seqLen := seqLen) (headDim := headDim) cfg

/-- Build learned positional embeddings from a freshly allocated parameter seed. -/
def learnedPositionalEmbedding (leading : Spec.Shape := .scalar) {seqLen embedDim : Nat}
    (cfg : LearnedPositionalEmbedding := {}) :
    Builder (Sequential
      (leading.concat (.dim seqLen (.dim embedDim .scalar)))
      (leading.concat (.dim seqLen (.dim embedDim .scalar)))) :=
  withSeed fun seedPos =>
    pure <| Internal.learnedPositionalEmbedding leading
      (seqLen := seqLen) (embedDim := embedDim) { cfg with seedPos := seedPos }

/-- Build final-axis layer normalization with independently seeded scale and offset parameters. -/
def layerNorm (leading : Spec.Shape := .scalar) {width : Nat} [NeZero width] :
    Builder (Sequential (leading.appendDim width) (leading.appendDim width)) :=
  withSeed fun seedGamma =>
    withSeed fun seedBeta =>
      pure <| Internal.layerNorm leading (width := width)
        { seedGamma := seedGamma, seedBeta := seedBeta }

/-- Build seeded multi-head self-attention with an optional fixed attention mask. -/
def multiHeadAttention (leading : Spec.Shape := .scalar) {n dModel : Nat} [NeZero n]
    (cfg : MultiHeadAttention)
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    Builder (Sequential
      ((leading.concat (.dim n .scalar)).appendDim dModel)
      ((leading.concat (.dim n .scalar)).appendDim dModel)) :=
  withSeed fun seedW =>
    pure <| Internal.multiHeadAttention leading (n := n) (dModel := dModel)
      { cfg with seedW := seedW } (mask := mask)

/-- Build one seeded transformer encoder block, optionally applying a fixed attention mask. -/
def transformerEncoderBlock (leading : Spec.Shape := .scalar) {n dModel : Nat}
    [NeZero n] [NeZero dModel]
    (cfg : blocks.TransformerEncoderBlock)
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    Builder (Sequential
      ((leading.concat (.dim n .scalar)).appendDim dModel)
      ((leading.concat (.dim n .scalar)).appendDim dModel)) :=
  withSeed fun seedBase =>
    let cfg' : blocks.TransformerEncoderBlock := { cfg with seedBase := seedBase }
    pure <| blocks.transformerEncoderBlock leading (n := n) (dModel := dModel) cfg'
      (mask := mask)

/-- Build a seeded stack of transformer encoder blocks with an optional attention mask. -/
def transformerEncoderStack (leading : Spec.Shape := .scalar) {n dModel : Nat}
    [NeZero n] [NeZero dModel]
    (cfg : blocks.TransformerEncoderStack)
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    Builder (Sequential
      ((leading.concat (.dim n .scalar)).appendDim dModel)
      ((leading.concat (.dim n .scalar)).appendDim dModel)) :=
  withSeed fun seedBase =>
    let cfg' : blocks.TransformerEncoderStack := { cfg with seedBase := seedBase }
    pure <| blocks.transformerEncoderStack leading (n := n) (dModel := dModel) cfg'
      (mask := mask)

/-- Build dropout with a fresh deterministic mask seed from the builder stream. -/
def dropout {s : Spec.Shape} (p : Float) : Builder (Sequential s s) :=
  withSeed fun seed =>
    pure (Internal.dropout (s := s) p (seed := seed))

namespace blocks

/--
Build a multilayer perceptron over any collection of leading dimensions.

Each hidden width contributes a linear layer followed by the configured activation and optional
dropout. Initialization seeds come from the surrounding `Builder` seed stream.
-/
def mlp (inDim outDim : Nat) (cfg : MlpConfig := {})
    (leading : Spec.Shape := .scalar) :
    Builder (Sequential (leading.appendDim inDim) (leading.appendDim outDim)) :=
  withSeed fun seed =>
    pure (Internal.blocks.mlpWithSeed inDim outDim seed cfg leading)

end blocks

/--
Build a model using the next global seed, then run a continuation.

`nn.Sequential` lives in `Type 2`, so executable code passes the model to a continuation rather than
returning it directly from `IO`.
-/
def withModel {σ τ : Spec.Shape} {β : Type}
    (mk : Builder (Sequential σ τ)) (k : Sequential σ τ → IO β) : IO β := do
  let seed ← rand.nextSeedGlobal
  let model := build seed mk
  k model

end nn

end TorchLean
