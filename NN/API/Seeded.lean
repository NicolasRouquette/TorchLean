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

/--
Set the global seed used by `nn.runGlobal` and `nn.freshSeed`.

Prefer `nn.build seed` when the seed belongs in the model definition itself.
-/
def manualSeed (seed : Nat) : IO Unit :=
  rand.manualSeed seed

/-
Configuration records are available at `nn.*`; constructors with explicit seed arguments remain
under `nn.Internal`.
-/

export Internal
  (Linear LearnedPositionalEmbedding SinusoidalPositionalEncoding RoPE
   Conv Pool LayerNorm RMSNorm ChannelNorm MultiheadAttention)

/-- Build global average pooling over the supplied nonempty spatial dimensions without consuming a seed. -/
def globalAvgPool (leading : Spec.Shape := .scalar) {d channels : Nat}
    (spatial : Vector Nat d) (spatialNonzero : ∀ i : Fin d, spatial.get i ≠ 0) :
    Builder (Sequential
      (leading.concat (Spec.Shape.ofList (channels :: spatial.toList)))
      (leading.concat (.dim channels .scalar))) :=
  fun state =>
    (Internal.globalAvgPool leading (channels := channels) spatial spatialNonzero, state)

namespace functional
export Internal.functional
  (square checkpoint
   exp log scale shift affine
   detach
   addB mulB
   embedding mean
   dropoutSeeded)
end functional

namespace blocks
export Internal.blocks
  (Activation activation
   MLP mlp
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

/-- Consume one fresh seed and pass it to `k`. -/
def withSeed {α : Type u} (k : Nat → α) : Builder α :=
  fun st =>
    let (seed, st') := rand.SeedStream.next st
    (k seed, st')

/-- Consume two fresh seeds and pass them to `k` (in order). -/
def withSeedPair {α : Type u} (k : Nat → Nat → α) : Builder α :=
  fun st =>
    let (a, st') := rand.SeedStream.next st
    let (b, st'') := rand.SeedStream.next st'
    (k a b, st'')

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

/-- Build a final-axis softmax layer without consuming an initialization seed. -/
def softmaxLast {s : Spec.Shape} : Builder (Sequential s s) :=
  lift (Internal.softmaxLast (s := s))

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
def linear (inDim outDim : Nat) (pfx : Spec.Shape := Spec.Shape.scalar) :
    Builder (Sequential (pfx.appendDim inDim) (pfx.appendDim outDim)) :=
  withSeedPair (fun seedW seedB =>
    Internal.linear inDim outDim seedW seedB (pfx := pfx))

/-- Seeded affine layer with an explicit initialization policy. -/
def linearWith (inDim outDim : Nat) (cfg : Linear) (pfx : Spec.Shape := Spec.Shape.scalar) :
    Builder (Sequential (pfx.appendDim inDim) (pfx.appendDim outDim)) :=
  withSeedPair (fun seedW seedB =>
    Internal.linearWith inDim outDim cfg seedW seedB (pfx := pfx))

namespace deterministic

/-- Construct a linear layer with explicit parameter-initialization seeds. -/
def linear (inDim outDim : Nat) (seedWeight seedBias : Nat)
    (leading : Spec.Shape := .scalar) :
    Sequential (leading.appendDim inDim) (leading.appendDim outDim) :=
  Internal.linear inDim outDim seedWeight seedBias (pfx := leading)

end deterministic

/-- Build a seeded recurrent neural network over a fixed sequence length. -/
def rnn (seqLen inputSize hiddenSize : Nat) :
    Builder (Sequential
      (.dim seqLen (.dim inputSize .scalar))
      (.dim seqLen (.dim hiddenSize .scalar))) :=
  withSeedPair (fun seedW seedB =>
    Internal.rnn seqLen inputSize hiddenSize seedW seedB)

/-- Build a seeded gated recurrent unit over a fixed sequence length. -/
def gru (seqLen inputSize hiddenSize : Nat) :
    Builder (Sequential
      (.dim seqLen (.dim inputSize .scalar))
      (.dim seqLen (.dim hiddenSize .scalar))) :=
  withSeedPair (fun seedW seedB =>
    Internal.gru seqLen inputSize hiddenSize seedW seedB)

/-- Build a seeded Mamba-style state-space sequence layer. -/
def mamba (seqLen inputSize hiddenSize : Nat) :
    Builder (Sequential
      (.dim seqLen (.dim inputSize .scalar))
      (.dim seqLen (.dim hiddenSize .scalar))) :=
  withSeedPair (fun seedW seedB =>
    Internal.mamba seqLen inputSize hiddenSize seedW seedB)

/-- Build a seeded long short-term memory layer over a fixed sequence length. -/
def lstm (seqLen inputSize hiddenSize : Nat) :
    Builder (Sequential
      (.dim seqLen (.dim inputSize .scalar))
      (.dim seqLen (.dim hiddenSize .scalar))) :=
  withSeedPair (fun seedW seedB =>
    Internal.lstm seqLen inputSize hiddenSize seedW seedB)

/-- Build an arbitrary-rank convolution, allocating separate kernel and bias seeds. -/
def conv (leading : Spec.Shape := .scalar) {d inChannels : Nat} (spatial : Vector Nat d)
    (cfg : Conv d) [NeZero inChannels] :
    Builder (Sequential
      (leading.concat (Spec.Shape.ofList (inChannels :: spatial.toList)))
      (leading.concat (Spec.Shape.ofList
        (cfg.outChannels ::
          (Spec.convOutSpatial spatial cfg.kernel cfg.stride cfg.padding).toList)))) :=
  withSeedPair (fun seedKernel seedBias =>
    let cfg' : Conv d := { cfg with seedKernel := seedKernel, seedBias := seedBias }
    by
      simpa [cfg'] using Internal.conv leading (inChannels := inChannels) spatial cfg')

/-- Build batch normalization with seeded scale and offset parameters. -/
def batchNorm (leading : Spec.Shape := .scalar) {d channels : Nat}
    (spatial : Vector Nat d) (cfg : ChannelNorm := {})
    [NeZero (Spec.Shape.size leading)] [NeZero channels]
    [NeZero (Spec.Shape.size (Spec.Shape.ofList spatial.toList))] :
    Builder (Sequential
      (leading.concat (Spec.Shape.ofList (channels :: spatial.toList)))
      (leading.concat (Spec.Shape.ofList (channels :: spatial.toList)))) :=
  withSeedPair (fun seedGamma seedBeta =>
    Internal.batchNorm leading spatial
      { cfg with seedGamma := seedGamma, seedBeta := seedBeta })

/-- Build instance normalization with seeded scale and offset parameters. -/
def instanceNorm (leading : Spec.Shape := .scalar) {d channels : Nat}
    (spatial : Vector Nat d) (cfg : ChannelNorm := {})
    [NeZero (Spec.Shape.size leading)] [NeZero channels]
    [NeZero (Spec.Shape.size (Spec.Shape.ofList spatial.toList))] :
    Builder (Sequential
      (leading.concat (Spec.Shape.ofList (channels :: spatial.toList)))
      (leading.concat (Spec.Shape.ofList (channels :: spatial.toList)))) :=
  withSeedPair (fun seedGamma seedBeta =>
    Internal.instanceNorm leading spatial
      { cfg with seedGamma := seedGamma, seedBeta := seedBeta })

/-- Build group normalization after checking the positive group count and channel divisibility. -/
def groupNorm (leading : Spec.Shape := .scalar) {d channels : Nat}
    (spatial : Vector Nat d) (groups : Nat) (hGroups : groups > 0)
    (hGroupsLe : channels >= groups) (hDiv : channels % groups = 0)
    (cfg : ChannelNorm := {}) [NeZero (Spec.Shape.size leading)] [NeZero channels]
    [NeZero (Spec.Shape.size (Spec.Shape.ofList spatial.toList))] :
    Builder (Sequential
      (leading.concat (Spec.Shape.ofList (channels :: spatial.toList)))
      (leading.concat (Spec.Shape.ofList (channels :: spatial.toList)))) :=
  withSeedPair (fun seedGamma seedBeta =>
    Internal.groupNorm leading spatial groups hGroups hGroupsLe hDiv
      { cfg with seedGamma := seedGamma, seedBeta := seedBeta })

/-- Build an embedding lookup layer from a freshly seeded embedding table. -/
def oneHotEmbedding (vocab embedDim : Nat) (cfg : Embedding.Config := {}) {pfx : Spec.Shape} :
    Builder (Sequential (pfx.appendDim vocab) (pfx.appendDim embedDim)) :=
  withSeed (fun seed =>
    Internal.oneHotEmbedding vocab embedDim { cfg with seed := seed } (pfx := pfx))

/-- Build a trainable lookup table for a tensor of natural-number indices. -/
def embedding (vocab embedDim : Nat) (cfg : Embedding.Config := {}) :
    Builder (Embedding vocab embedDim) :=
  withSeed (fun seed =>
    Internal.embedding vocab embedDim { cfg with seed := seed })

/-- Build deterministic sinusoidal positional encoding for a batched sequence. -/
def sinusoidalPositionalEncoding {batch seqLen embedDim : Nat}
    (cfg : SinusoidalPositionalEncoding := {}) :
    Builder (Sequential
      (.dim batch (.dim seqLen (.dim embedDim .scalar)))
      (.dim batch (.dim seqLen (.dim embedDim .scalar)))) :=
  lift <|
    Internal.sinusoidalPositionalEncoding (batch := batch) (seqLen := seqLen) (embedDim := embedDim) cfg

/-- Build deterministic rotary positional encoding for multi-head sequence features. -/
def rope {batch numHeads seqLen headDim : Nat} (cfg : RoPE := {}) :
    Builder (Sequential
      (.dim batch (.dim numHeads (.dim seqLen (.dim headDim .scalar))))
      (.dim batch (.dim numHeads (.dim seqLen (.dim headDim .scalar))))) :=
  lift <|
    Internal.rope (batch := batch) (numHeads := numHeads) (seqLen := seqLen) (headDim := headDim) cfg

/-- Build learned positional embeddings from a freshly allocated parameter seed. -/
def learnedPositionalEmbedding {batch seqLen embedDim : Nat} (cfg : LearnedPositionalEmbedding := {}) :
    Builder (Sequential
      (.dim batch (.dim seqLen (.dim embedDim .scalar)))
      (.dim batch (.dim seqLen (.dim embedDim .scalar)))) :=
  withSeed (fun seedPos =>
    Internal.learnedPositionalEmbedding (batch := batch) (seqLen := seqLen) (embedDim := embedDim)
      { cfg with seedPos := seedPos })

/-- Build layer normalization with independently seeded scale and offset parameters. -/
def layerNorm {batch seqLen embedDim : Nat} [NeZero seqLen] [NeZero embedDim] :
    Builder (Sequential
      (.dim batch (.dim seqLen (.dim embedDim .scalar)))
      (.dim batch (.dim seqLen (.dim embedDim .scalar)))) :=
  withSeedPair (fun seedGamma seedBeta =>
    Internal.layerNorm (batch := batch) (seqLen := seqLen) (embedDim := embedDim)
      { seedGamma := seedGamma, seedBeta := seedBeta })

/-- Build seeded multi-head self-attention with an optional fixed attention mask. -/
def multiheadAttention {batch n dModel : Nat} [NeZero n]
    (cfg : MultiheadAttention)
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    Builder (Sequential
      (.dim batch (.dim n (.dim dModel .scalar)))
      (.dim batch (.dim n (.dim dModel .scalar)))) :=
  withSeed (fun seedW =>
    Internal.multiheadAttention (batch := batch) (n := n) (dModel := dModel)
      { cfg with seedW := seedW } (mask := mask))

/-- Build one seeded transformer encoder block, optionally applying a fixed attention mask. -/
def transformerEncoderBlock {batch n dModel : Nat} [NeZero n] [NeZero dModel]
    (cfg : blocks.TransformerEncoderBlock)
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    Builder (Sequential
      (.dim batch (.dim n (.dim dModel .scalar)))
      (.dim batch (.dim n (.dim dModel .scalar)))) :=
  withSeed (fun seedBase =>
    let cfg' : blocks.TransformerEncoderBlock := { cfg with seedBase := seedBase }
    blocks.transformerEncoderBlock (batch := batch) (n := n) (dModel := dModel) cfg'
      (mask := mask))

/-- Build a seeded stack of transformer encoder blocks with an optional attention mask. -/
def transformerEncoderStack {batch n dModel : Nat} [NeZero n] [NeZero dModel]
    (cfg : blocks.TransformerEncoderStack)
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    Builder (Sequential
      (.dim batch (.dim n (.dim dModel .scalar)))
      (.dim batch (.dim n (.dim dModel .scalar)))) :=
  withSeed (fun seedBase =>
    let cfg' : blocks.TransformerEncoderStack := { cfg with seedBase := seedBase }
    blocks.transformerEncoderStack (batch := batch) (n := n) (dModel := dModel) cfg'
      (mask := mask))

/-- Build dropout with a fresh deterministic mask seed from the builder stream. -/
def dropout {s : Spec.Shape} (p : Float) : Builder (Sequential s s) :=
  withSeed (fun seed =>
    Internal.dropout (s := s) p (seed := seed))

/--
Run a seeded builder using the global seed stream set by `nn.manualSeed` (results in `Type`).

Note: model values like `nn.Sequential` live in `Type 2`, so they cannot be returned from `IO`.
For models, use `nn.build` with an explicit base seed (obtained from `nn.freshSeed`).
-/
def runGlobal {α : Type} (x : Builder α) : IO α :=
  rand.runGlobal x

/-- Draw a fresh base seed from the global seed stream set by `nn.manualSeed`. -/
def freshSeed : IO Nat :=
  rand.nextSeedGlobal

/-- Draw `n` fresh base seeds from the global seed stream. -/
def freshSeeds (n : Nat) : IO (List Nat) :=
  rand.nextSeedsGlobal n

/--
Build a model using the next global seed, then run a continuation.

`nn.Sequential` lives in `Type 2`, so executable code passes the model to a continuation rather than
returning it directly from `IO`.
-/
def withModel {σ τ : Spec.Shape} {β : Type}
    (mk : Builder (Sequential σ τ)) (k : Sequential σ τ → IO β) : IO β := do
  let seed ← freshSeed
  let model := build seed mk
  k model

end nn

end TorchLean
