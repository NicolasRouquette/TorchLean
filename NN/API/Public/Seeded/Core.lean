/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Adapters
public import NN.API.Common
public import NN.API.Data
public import NN.API.Macros
public import NN.API.Rand
public import NN.API.RL
public import NN.API.Runtime
public import NN.API.Samples
public import NN.API.SelfSupervised
public import NN.API.Text
public import NN.API.Text.Bpe
public import NN.Spec.Layers.PositionalEncoding
public import NN.API.Public.NN

import Mathlib.Algebra.Order.Algebra

@[expose] public section

namespace NN
namespace API

/-!
# Seeded model builders

This module reopens `NN.API.nn` with the PyTorch-style seeded builder API.  It sits on top of the
pure builders from `NN.API.Public.NN` and allocates deterministic initialization seeds for users.
-/

namespace nn

/-!
## Model Builders and Seeding

TorchLean keeps initialization randomness explicit so examples are reproducible.

- `nn.*` is the default *seeded builder* API: layer constructors allocate initialization seeds
  via `nn.M` (a deterministic seed stream).
- `nn.pure.*` contains the explicit-seed constructors (proof/reproducibility-friendly).

Typical patterns:

1. Explicit seeds (best for proofs / reproducibility-sensitive code):
   - build with `nn.pure.linear ... (seedW := ...) (seedB := ...)` etc
   - compose with `seq! ...` / `>>>`

2. Script-style “manual seed once”:
   - `nn.manualSeed seed`
   - `let seed ← nn.nextSeed`
   - `let model := nn.run seed <| nn.Sequential![ ... ]`

Note: `nn.Sequential` lives in `Type 2`, so it cannot be returned directly from `IO`. We keep
model building pure by drawing a base seed in `IO` and then calling `nn.run`.
-/

/--
PyTorch-like global seeding convenience for seeded model builders.

This sets the global seed stream used by `nn.runGlobal` / `nn.nextSeed`.
-/
def manualSeed (seed : Nat) : IO Unit :=
  rand.manualSeed seed

/-
`nn.pure.*` holds the explicit-seed constructors (proof/reproducibility-friendly).

For user ergonomics, we re-export the *config records* and the pure construction namespaces
(`functional`, `blocks`, `heads`) at `nn.*`, while keeping the explicit layer constructors under
`nn.pure.*`.
-/

@[inherit_doc pure.Embedding]
abbrev Embedding := pure.Embedding

@[inherit_doc pure.LearnedPositionalEmbedding]
abbrev LearnedPositionalEmbedding := pure.LearnedPositionalEmbedding

@[inherit_doc pure.SinusoidalPositionalEncoding]
abbrev SinusoidalPositionalEncoding := pure.SinusoidalPositionalEncoding

@[inherit_doc pure.RoPE]
abbrev RoPE := pure.RoPE

@[inherit_doc pure.Conv2d]
abbrev Conv2d := pure.Conv2d

@[inherit_doc Conv2d]
abbrev Conv := Conv2d

@[inherit_doc pure.MaxPool2d]
abbrev MaxPool2d := pure.MaxPool2d

@[inherit_doc MaxPool2d]
abbrev MaxPool := MaxPool2d

@[inherit_doc pure.AvgPool2d]
abbrev AvgPool2d := pure.AvgPool2d

@[inherit_doc AvgPool2d]
abbrev AvgPool := AvgPool2d

@[inherit_doc pure.LayerNorm]
abbrev LayerNorm := pure.LayerNorm

@[inherit_doc pure.RMSNorm]
abbrev RMSNorm := pure.RMSNorm

@[inherit_doc pure.BatchNorm2d]
abbrev BatchNorm2d := pure.BatchNorm2d

@[inherit_doc pure.InstanceNorm2d]
abbrev InstanceNorm2d := pure.InstanceNorm2d

@[inherit_doc pure.MultiheadAttention]
abbrev MultiheadAttention := pure.MultiheadAttention

@[inherit_doc pure.globalAvgPoolCHW]
def globalAvgPoolCHW := pure.globalAvgPoolCHW

@[inherit_doc pure.globalAvgPoolNCHW]
def globalAvgPoolNCHW := pure.globalAvgPoolNCHW

namespace functional
export pure.functional
  (square checkpoint
   exp log scale shift affine
   detach stopGrad
   addB mulB
   embedding mean
   dropoutSeeded)
end functional

namespace blocks
export pure.blocks
  (Activation activation
   MLP mlp
   Conv2dAct conv2dAct
   Conv2dNormAct conv2dNormActCHW conv2dNormAct
   Conv2dNormActPool conv2dNormActPoolCHW conv2dNormActPool
   down2 down2_pos
   conv3x3Same conv3x3Down conv1x1Same conv1x1Down
   conv3x3SameImages conv3x3DownImages conv1x1SameImages conv1x1DownImages
   ResNetBasicBlock resnetBasicBlock
   TransformerEncoderBlock transformerEncoderBlockWithMask transformerEncoderBlock
   TransformerEncoderStack transformerEncoderStackWithMask transformerEncoderStack
   transformerEncoderClassifier
   residual residualLayer
   addBranches addBranchesLayer)
end blocks

namespace heads
export pure.heads (classifier regressor classifierBatch regressorBatch)
end heads

/-!
## Seeded Builders (Default `nn.*`)

For end-user code, the default `nn.*` layer constructors allocate initialization seeds
automatically via `nn.M` (a deterministic seed-stream builder).

Use `nn.pure.*` when you want to pass explicit seeds (proof-friendly / fully reproducible).
-/

open Spec

/-- Seeded builder monad: a state monad over `API.rand.SeedStream`. -/
abbrev M := rand.SeedM

/-- Run a seeded builder starting from a base seed. -/
def run {α : Type 2} (seed : Nat) (x : M α) : α :=
  (x (rand.SeedStream.init seed)).1

/-- Lift a pure value into the seeded builder (consumes no seeds). -/
def lift {α : Type 2} (x : α) : M α :=
  pure x

/-- Consume one fresh seed and pass it to `k`. -/
def withSeed {α : Type 2} (k : Nat → α) : M α :=
  fun st =>
    let (seed, st') := rand.SeedStream.next st
    (k seed, st')

/-- Consume two fresh seeds and pass them to `k` (in order). -/
def withSeeds2 {α : Type 2} (k : Nat → Nat → α) : M α :=
  fun st =>
    let (a, st') := rand.SeedStream.next st
    let (b, st'') := rand.SeedStream.next st'
    (k a b, st'')

@[inherit_doc pure.relu]
def relu {s : Spec.Shape} : M (Sequential s s) :=
  lift (pure.relu (s := s))

@[inherit_doc relu]
abbrev ReLU {s : Spec.Shape} : M (Sequential s s) := relu (s := s)

@[inherit_doc pure.silu]
def silu {s : Spec.Shape} : M (Sequential s s) :=
  lift (pure.silu (s := s))

@[inherit_doc silu]
abbrev SiLU {s : Spec.Shape} : M (Sequential s s) := silu (s := s)

@[inherit_doc pure.gelu]
def gelu {s : Spec.Shape} : M (Sequential s s) :=
  lift (pure.gelu (s := s))

@[inherit_doc gelu]
abbrev GELU {s : Spec.Shape} : M (Sequential s s) := gelu (s := s)

@[inherit_doc pure.sigmoid]
def sigmoid {s : Spec.Shape} : M (Sequential s s) :=
  lift (pure.sigmoid (s := s))

@[inherit_doc sigmoid]
abbrev Sigmoid {s : Spec.Shape} : M (Sequential s s) := sigmoid (s := s)

@[inherit_doc pure.tanh]
def tanh {s : Spec.Shape} : M (Sequential s s) :=
  lift (pure.tanh (s := s))

@[inherit_doc tanh]
abbrev Tanh {s : Spec.Shape} : M (Sequential s s) := tanh (s := s)

@[inherit_doc pure.softmax]
def softmax {s : Spec.Shape} : M (Sequential s s) :=
  lift (pure.softmax (s := s))

@[inherit_doc softmax]
abbrev Softmax {s : Spec.Shape} : M (Sequential s s) := softmax (s := s)

@[inherit_doc pure.sum]
def sum {s : Spec.Shape} : M (Sequential s Spec.Shape.scalar) :=
  lift (pure.sum (s := s))

@[inherit_doc pure.flatten]
def flatten {s : Spec.Shape} : M (Sequential s (.dim (Spec.Shape.size s) .scalar)) :=
  lift (pure.flatten (s := s))

@[inherit_doc flatten]
abbrev Flatten {s : Spec.Shape} : M (Sequential s (.dim (Spec.Shape.size s) .scalar)) :=
  flatten (s := s)

@[inherit_doc pure.flattenBatch]
def flattenBatch {n : Nat} {s : Spec.Shape} :
    M (Sequential (.dim n s) (NN.Tensor.Shape.Mat n (Spec.Shape.size s))) :=
  lift (pure.flattenBatch (n := n) (s := s))

@[inherit_doc flattenBatch]
abbrev FlattenBatch {n : Nat} {s : Spec.Shape} :
    M (Sequential (.dim n s) (NN.Tensor.Shape.Mat n (Spec.Shape.size s))) :=
  flattenBatch (n := n) (s := s)

@[inherit_doc pure.flattenStart1]
def flattenStart1 {n : Nat} {s : Spec.Shape} :
    M (Sequential (.dim n s) (NN.Tensor.Shape.Mat n (Spec.Shape.size s))) :=
  lift (pure.flattenStart1 (n := n) (s := s))

@[inherit_doc pure.maxPool2d]
def maxPool2d {n inC inH inW : Nat} (cfg : MaxPool2d) [NeZero cfg.kH] [NeZero cfg.kW] :
    M (Sequential
      (NN.Tensor.Shape.Images n inC inH inW)
      (NN.Tensor.Shape.Images n inC
        ((inH - cfg.kH) / cfg.stride + 1)
        ((inW - cfg.kW) / cfg.stride + 1))) :=
  lift (pure.maxPool2d (n := n) (inC := inC) (inH := inH) (inW := inW) cfg)

/--
Max pooling over batched CHW images, allocating any required initialization seeds automatically.

Shorthand for `maxPool2d`.
-/
def maxPool {n inC inH inW : Nat} (cfg : MaxPool) [NeZero cfg.kH] [NeZero cfg.kW] :=
  maxPool2d (n := n) (inC := inC) (inH := inH) (inW := inW) cfg

@[inherit_doc pure.avgPool2d]
def avgPool2d {n inC inH inW : Nat} (cfg : AvgPool2d) [NeZero cfg.kH] [NeZero cfg.kW] :
    M (Sequential
      (NN.Tensor.Shape.Images n inC inH inW)
      (NN.Tensor.Shape.Images n inC
        ((inH - cfg.kH) / cfg.stride + 1)
        ((inW - cfg.kW) / cfg.stride + 1))) :=
  lift (pure.avgPool2d (n := n) (inC := inC) (inH := inH) (inW := inW) cfg)

/--
Average pooling over batched CHW images, allocating any required initialization seeds automatically.

Shorthand for `avgPool2d`.
-/
def avgPool {n inC inH inW : Nat} (cfg : AvgPool) [NeZero cfg.kH] [NeZero cfg.kW] :=
  avgPool2d (n := n) (inC := inC) (inH := inH) (inW := inW) cfg

@[inherit_doc pure.linear]
def linear (inDim outDim : Nat) (pfx : Spec.Shape := Spec.Shape.scalar) :
    M (Sequential (pfx.appendDim inDim) (pfx.appendDim outDim)) :=
  withSeeds2 (fun seedW seedB =>
    pure.linear inDim outDim seedW seedB (pfx := pfx))

@[inherit_doc linear]
abbrev Linear (inDim outDim : Nat) (pfx : Spec.Shape := Spec.Shape.scalar) :
    M (Sequential (pfx.appendDim inDim) (pfx.appendDim outDim)) :=
  linear inDim outDim (pfx := pfx)

/-- Vector-only linear layer, specialized to the scalar prefix shape. -/
def linearV (inDim outDim : Nat) : M (Sequential (NN.Tensor.Shape.Vec inDim)
    (NN.Tensor.Shape.Vec outDim)) :=
  linear inDim outDim

@[inherit_doc linearV]
abbrev LinearV (inDim outDim : Nat) :
    M (Sequential (NN.Tensor.Shape.Vec inDim) (NN.Tensor.Shape.Vec outDim)) :=
  linearV inDim outDim

@[inherit_doc pure.rnn]
def rnn (seqLen inputSize hiddenSize : Nat) :
    M (Sequential
      (NN.Tensor.Shape.Mat seqLen inputSize)
      (NN.Tensor.Shape.Mat seqLen hiddenSize)) :=
  withSeeds2 (fun seedW seedB =>
    pure.rnn seqLen inputSize hiddenSize seedW seedB)

@[inherit_doc pure.gru]
def gru (seqLen inputSize hiddenSize : Nat) :
    M (Sequential
      (NN.Tensor.Shape.Mat seqLen inputSize)
      (NN.Tensor.Shape.Mat seqLen hiddenSize)) :=
  withSeeds2 (fun seedW seedB =>
    pure.gru seqLen inputSize hiddenSize seedW seedB)

@[inherit_doc pure.mamba]
def mamba (seqLen inputSize hiddenSize : Nat) :
    M (Sequential
      (NN.Tensor.Shape.Mat seqLen inputSize)
      (NN.Tensor.Shape.Mat seqLen hiddenSize)) :=
  withSeeds2 (fun seedW seedB =>
    pure.mamba seqLen inputSize hiddenSize seedW seedB)

@[inherit_doc pure.lstm]
def lstm (seqLen inputSize hiddenSize : Nat) :
    M (Sequential
      (NN.Tensor.Shape.Mat seqLen inputSize)
      (NN.Tensor.Shape.Mat seqLen hiddenSize)) :=
  withSeeds2 (fun seedW seedB =>
    pure.lstm seqLen inputSize hiddenSize seedW seedB)

@[inherit_doc pure.conv2d]
def conv2d {n inC inH inW : Nat} (cfg : Conv2d) [NeZero inC] [NeZero cfg.kH] [NeZero cfg.kW] :
    M (Sequential
      (NN.Tensor.Shape.Images n inC inH inW)
      (NN.Tensor.Shape.Images n cfg.outC
        ((inH + 2 * cfg.padding - cfg.kH) / cfg.stride + 1)
        ((inW + 2 * cfg.padding - cfg.kW) / cfg.stride + 1))) :=
  withSeeds2 (fun seedK seedB =>
    let cfg' : Conv2d := { cfg with seedK := seedK, seedB := seedB }
    pure.conv2d (n := n) (inC := inC) (inH := inH) (inW := inW) cfg')

/--
Convolution over batched CHW images, allocating initialization seeds automatically.

Shorthand for `conv2d`.
-/
def conv {n inC inH inW : Nat} (cfg : Conv) [NeZero inC] [NeZero cfg.kH] [NeZero cfg.kW] :=
  conv2d (n := n) (inC := inC) (inH := inH) (inW := inW) cfg

@[inherit_doc pure.batchNorm2d]
def batchNorm2d {n c h w : Nat} (cfg : BatchNorm2d := {})
    [NeZero n] [NeZero c] [NeZero h] [NeZero w] :
    M (Sequential (NN.Tensor.Shape.Images n c h w) (NN.Tensor.Shape.Images n c h w)) :=
  withSeeds2 (fun seedGamma seedBeta =>
    let cfg' : BatchNorm2d := { cfg with seedGamma := seedGamma, seedBeta := seedBeta }
    pure.batchNorm2d (n := n) (c := c) (h := h) (w := w) cfg')

/--
BatchNorm over batched CHW images, allocating initialization seeds automatically.

Shorthand for `batchNorm2d`.
-/
def batchNorm {n c h w : Nat} (cfg : BatchNorm2d := {}) [NeZero n] [NeZero c] [NeZero h] [NeZero w] :=
  batchNorm2d (n := n) (c := c) (h := h) (w := w) cfg

@[inherit_doc pure.embedding]
def embedding (vocab embedDim : Nat) (cfg : Embedding := {}) {pfx : Spec.Shape} :
    M (Sequential (pfx.appendDim vocab) (pfx.appendDim embedDim)) :=
  withSeed (fun seedW =>
    pure.embedding vocab embedDim { cfg with seedW := seedW } (pfx := pfx))

@[inherit_doc pure.sinusoidalPositionalEncoding]
def sinusoidalPositionalEncoding {batch seqLen embedDim : Nat}
    (cfg : SinusoidalPositionalEncoding := {}) :
    M (Sequential
      (.dim batch (NN.Tensor.Shape.Mat seqLen embedDim))
      (.dim batch (NN.Tensor.Shape.Mat seqLen embedDim))) :=
  lift <|
    pure.sinusoidalPositionalEncoding (batch := batch) (seqLen := seqLen) (embedDim := embedDim) cfg

@[inherit_doc pure.rope]
def rope {batch numHeads seqLen headDim : Nat} (cfg : RoPE := {}) :
    M (Sequential
      (.dim batch (.dim numHeads (NN.Tensor.Shape.Mat seqLen headDim)))
      (.dim batch (.dim numHeads (NN.Tensor.Shape.Mat seqLen headDim)))) :=
  lift <|
    pure.rope (batch := batch) (numHeads := numHeads) (seqLen := seqLen) (headDim := headDim) cfg

@[inherit_doc pure.learnedPositionalEmbedding]
def learnedPositionalEmbedding {batch seqLen embedDim : Nat} (cfg : LearnedPositionalEmbedding := {}) :
    M (Sequential
      (.dim batch (NN.Tensor.Shape.Mat seqLen embedDim))
      (.dim batch (NN.Tensor.Shape.Mat seqLen embedDim))) :=
  withSeed (fun seedPos =>
    pure.learnedPositionalEmbedding (batch := batch) (seqLen := seqLen) (embedDim := embedDim)
      { cfg with seedPos := seedPos })

@[inherit_doc pure.layerNorm]
def layerNorm {batch seqLen embedDim : Nat} [NeZero seqLen] [NeZero embedDim] :
    M (Sequential
      (.dim batch (NN.Tensor.Shape.Mat seqLen embedDim))
      (.dim batch (NN.Tensor.Shape.Mat seqLen embedDim))) :=
  withSeeds2 (fun seedGamma seedBeta =>
    pure.layerNorm (batch := batch) (seqLen := seqLen) (embedDim := embedDim)
      { seedGamma := seedGamma, seedBeta := seedBeta })

@[inherit_doc pure.multiheadAttention]
def multiheadAttention {batch n dModel : Nat} [NeZero n]
    (cfg : MultiheadAttention)
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    M (Sequential
      (.dim batch (NN.Tensor.Shape.Mat n dModel))
      (.dim batch (NN.Tensor.Shape.Mat n dModel))) :=
  withSeed (fun seedW =>
    pure.multiheadAttention (batch := batch) (n := n) (dModel := dModel)
      { cfg with seedW := seedW } (mask := mask))

@[inherit_doc blocks.transformerEncoderBlock]
def transformerEncoderBlock {batch n dModel : Nat} [NeZero n] [NeZero dModel]
    (cfg : blocks.TransformerEncoderBlock)
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    M (Sequential
      (.dim batch (NN.Tensor.Shape.Mat n dModel))
      (.dim batch (NN.Tensor.Shape.Mat n dModel))) :=
  withSeed (fun seedBase =>
    let cfg' : blocks.TransformerEncoderBlock := { cfg with seedBase := seedBase }
    blocks.transformerEncoderBlock (batch := batch) (n := n) (dModel := dModel) cfg'
      (mask := mask))

@[inherit_doc blocks.transformerEncoderStack]
def transformerEncoderStack {batch n dModel : Nat} [NeZero n] [NeZero dModel]
    (cfg : blocks.TransformerEncoderStack)
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    M (Sequential
      (.dim batch (NN.Tensor.Shape.Mat n dModel))
      (.dim batch (NN.Tensor.Shape.Mat n dModel))) :=
  withSeed (fun seedBase =>
    let cfg' : blocks.TransformerEncoderStack := { cfg with seedBase := seedBase }
    blocks.transformerEncoderStack (batch := batch) (n := n) (dModel := dModel) cfg'
      (mask := mask))

@[inherit_doc blocks.resnetBasicBlock]
def resnetBasicBlock {n inC h w : Nat} (cfg : blocks.ResNetBasicBlock)
    [NeZero n] [NeZero inC] [NeZero h] [NeZero w] [NeZero cfg.outC] :
    M (Sequential
      (NN.Tensor.Shape.Images n inC h w)
      (NN.Tensor.Shape.Images n cfg.outC
        (if cfg.downsample then blocks.down2 h else h)
        (if cfg.downsample then blocks.down2 w else w))) :=
  withSeed (fun seedBase =>
    let cfg' : blocks.ResNetBasicBlock := { cfg with seedBase := seedBase }
    blocks.resnetBasicBlock (n := n) (inC := inC) (h := h) (w := w) cfg')

@[inherit_doc pure.dropout]
def dropout {s : Spec.Shape} (p : Float) : M (Sequential s s) :=
  withSeed (fun seed =>
    pure.dropout (s := s) p (seed := seed))

/--
Run a seeded builder using the global seed stream set by `nn.manualSeed` (results in `Type`).

Note: model values like `nn.Sequential` live in `Type 2`, so they cannot be returned from `IO`.
For models, use `nn.run` with an explicit base seed (obtained from `nn.nextSeed`).
-/
def runGlobal {α : Type} (x : M α) : IO α :=
  rand.runGlobal x

/-- Draw a fresh base seed from the global seed stream set by `nn.manualSeed`. -/
def nextSeed : IO Nat :=
  rand.nextSeedGlobal

/-- Draw `n` fresh base seeds from the global seed stream. -/
def nextSeeds (n : Nat) : IO (List Nat) :=
  rand.nextSeedsGlobal n

/-!
### Naming Convenience

`nn.run` / `nn.nextSeed` are the core primitives, but in user code it is often clearer to read:
- “build a model from this seed” (`nn.run`)
- “draw a fresh init seed” (`nn.freshSeed`)
- “build a model using the next global init seed” (`nn.withModel`)
-/

/-- Draw a fresh base seed from the global seed stream. -/
def freshSeed : IO Nat :=
  nextSeed

/--
Build a model using the next global seed, then run a continuation.

`nn.Sequential` lives in `Type 2`, so executable code passes the model to a continuation rather than
returning it directly from `IO`.
-/
def withModel {σ τ : Spec.Shape} {β : Type}
    (mk : M (Sequential σ τ)) (k : Sequential σ τ → IO β) : IO β := do
  let seed ← nextSeed
  let model := run seed mk
  k model

end nn

end API
end NN
