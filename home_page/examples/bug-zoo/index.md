---
title: Bug Zoo
---

# Bug Zoo

Bug Zoo collects mistakes that are easy to miss in ordinary machine-learning tests. The program
still runs and returns a tensor, loss, or token sequence, but the result no longer has the meaning
the caller assumed.

Each example is deliberately small. It states the intended behavior as a Lean definition or
theorem, shows where an implementation can depart from it, and identifies any runtime assumption
that remains outside the proof. Together they cover attention, decoding, data boundaries,
normalization, losses, compilation, floating point, and geometry.

## Attention and Autoregressive Decoding

A causal mask should exclude future keys exactly. Replacing $-\infty$ with a large finite negative
number only approximates that behavior and can fail when logits leave the expected range.
[`AttentionMask.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/AttentionMask.lean)
uses hard-mask semantics and proves that every strict-future attention weight is zero.

Incremental decoding introduces a different problem. A key/value cache must contain the same keys
and values that full-sequence attention would have seen, in the same positions.
[`KVCache.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/KVCache.lean)
checks the append operation, while
[`RoPEPosition.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/RoPEPosition.lean)
records the position assigned to the new token. These contracts isolate the two common off-by-one
errors instead of hiding them inside a generation loop.

## Data, Batches, and Normalization State

Tokenizer errors often appear much earlier than the model. A checkpoint may expect one vocabulary
or special-token convention while the data loader supplies another.
[`TokenizerBoundary.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/TokenizerBoundary.lean)
requires imported token ids to inhabit `Fin vocabSize`, making the vocabulary bound part of the
object passed to the network.

Batching should normally change throughput, not the prediction for an individual sample.
[`BatchInvariance.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/BatchInvariance.lean)
states that selecting a row from a batched reference run agrees with evaluating that row alone.
This catches accidental reductions across the batch axis and state that leaks between samples.

Normalization has its own hidden state. Batch normalization uses learned affine parameters and
running statistics at inference time; layer normalization has a degenerate one-feature case that
is easy to mishandle. The normalization examples make the axes, epsilon placement, running state,
and zero-gradient corner cases explicit.

## Losses, Floating Point, and Compilation

Several examples concern computations that are mathematically familiar but numerically unsafe.
Masking a quotient after division does not repair a division by zero, and a direct implementation
of a logit loss can overflow even when its stable form is finite. `AutogradDomain.lean` and
`StableLoss.lean` put the domain restriction and stable formula into the contract before reverse
mode is considered.

Real-number proofs do not automatically describe a binary32 run.
[`FloatBoundary.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/FloatBoundary.lean)
uses Lean's logical `Float32.Model` and TorchLean's independent bit-level executor to state the
connection explicitly. `CompilerBoundary.lean` does the analogous job for optimized graphs: an
accepted rewrite must preserve operations, shapes, dtypes, weights, and buffers.

The final geometry case starts from tensors exported by a detector. Lean recomputes camera
projection and positive depth, then checks that the reported two-dimensional box encloses every
projected corner. The detector remains an external producer; the enclosure claim does not.

## Run the Examples

Build the complete collection with:

```bash
lake build NN.Examples.BugZoo.All
```

The cases with registered certificate checkers can also be run from the verification command:

```bash
lake exe verify -- camera-box3d-cert
lake exe verify -- all
```

The source files and the contracts they expose are listed below.

| Source file | Bug family | Contract exposed |
| --- | --- | --- |
| `AttentionMask.lean` | Causal masks, mask polarity, finite sentinels standing in for $-\infty$ | Future positions receive exactly zero attention weight under hard-mask semantics. |
| `KVCache.lean` | Shifted or malformed key/value caches in autoregressive decoding | The appended key/value vector is exactly the final cache entry. |
| `RoPEPosition.lean` | Off-by-one or mismatched rotary/absolute positions | Appending a token assigns the next sequence position. |
| `TokenizerBoundary.lean` | Vocabulary-size and special-token mismatches | Imported token ids inhabit `Fin vocabSize`. |
| `BatchInvariance.lean` | Dynamic batching changing per-sample outputs | Selecting one row from a batched reference run equals evaluating that row alone. |
| `NormalizationState.lean` | BatchNorm formula/state mistakes | Epsilon placement and eval-time running statistics are explicit objects. |
| `LayerNormDegenerateAxis.lean` | One-feature LayerNorm corner cases | The output is the bias, with zero input and scale-gradient contribution. |
| `ConstantNormalizationSlice.lean` | Cancellation in normalization kernels on constant slices | Affine normalization returns the bias and contributes zero scale gradient. |
| `IgnoredLabelLoss.lean` | All-ignored cross-entropy reductions | Ignored labels and the empty-reduction policy are named. |
| `AutogradDomain.lean` | Masking after undefined division | The safe graph records epsilon-protected division before masking. |
| `StableLoss.lean` | Numerically unstable losses and domain-sensitive ops | Logit losses use the stable log-softmax path. |
| `ShapeAndBroadcast.lean` | Missing axes and silent broadcasts | Dimension changes are explicit terms with shape evidence. |
| `CompilerBoundary.lean` | Optimized graphs silently changing semantics | Backend acceptance is a preservation obligation over ops, shapes, dtypes, weights, and buffers. |
| `FloatBoundary.lean` | Real-valued reasoning applied to Float32 runs | Lean's logical Float32 model and TorchLean's independent executor are connected by a named equivalence obligation. |
| `Geometry3DProjection.lean` | Camera convention, depth, layout, and projection-box errors | The checker recomputes projection, positive depth, and 2D box enclosure. |

## Two Checked Statements

Under hard-mask semantics, every strict-future key receives exactly zero attention weight:

```lean
theorem trueInfinityMask_future_attention_weight_zero :
  Spec.get2 (Spec.hardMaskedSoftmaxSpec scores (Spec.causalMask n)) i j = 0
```

Lean 4.33 defines core `Float32` operations through `Float32.Model`. TorchLean proves that the model
agrees with its independently implemented `IEEE32Exec` arithmetic:

```lean
theorem Float32Bridge.float32_isFinite_eq_ieee32 (a : _root_.Float32) :
    Float32.isFinite a = IEEE32Exec.isFinite (toIEEE32Exec a)

theorem Float32Bridge.toIEEE32Exec_add
    (a b : _root_.Float32) :
    toIEEE32Exec (a + b) =
      canonicalize (IEEE32Exec.add (toIEEE32Exec a) (toIEEE32Exec b))
```

Classification and addition need no finiteness premise. The addition theorem covers finite values,
signed zeros, infinities, NaNs, underflow, overflow, and nearest-even rounding. NaNs are
canonicalized before comparing bits because Lean stores one NaN representation while
`IEEE32Exec` retains payload and sign information.
