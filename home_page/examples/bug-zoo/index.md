---
title: Bug Zoo Walkthrough
usemathjax: true
---

Bug Zoo is TorchLean’s collection of small case studies for semantic bugs that can pass ordinary
runtime checks. These examples focus on cases where code still returns tensors, losses, or tokens,
but the returned value no longer satisfies the intended contract.

Bug Zoo shows the motivation for TorchLean in miniature: many ML failures are not type errors or
crashes, but silent changes in meaning.

Each card starts with a bug pattern, then isolates the small mathematical contract that would have
made the intended behavior explicit. Some cards make a mistake unrepresentable in the checked
fragment. Others turn the mistake into a theorem obligation or a runtime agreement that has to be
named.

## A Few Case Studies

<div class="showcase-grid bug-zoo-grid">
  <a class="showcase-card showcase-image-card" href="#what-a-checked-claim-looks-like-here">
    <img class="showcase-media" src="{{ '/assets/media/examples/bug-zoo/attention-masks.png' | relative_url }}" alt="Attention mask Bug Zoo case study"/>
    <span class="showcase-body">
      <span class="showcase-title">Attention Masks</span>
      <span class="showcase-text">Checks that masked future positions receive exactly zero attention weight under the stated causal mask semantics.</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="#what-a-checked-claim-looks-like-here">
    <img class="showcase-media" src="{{ '/assets/media/examples/bug-zoo/kv-cache-rope.png' | relative_url }}" alt="KV cache and RoPE Bug Zoo case study"/>
    <span class="showcase-body">
      <span class="showcase-title">KV Cache and RoPE</span>
      <span class="showcase-text">Makes the cache-position contract explicit so incremental decoding agrees with the intended full-sequence computation.</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="#what-a-checked-claim-looks-like-here">
    <img class="showcase-media" src="{{ '/assets/media/examples/bug-zoo/tokenizer-boundaries.png' | relative_url }}" alt="Tokenizer boundary Bug Zoo case study"/>
    <span class="showcase-body">
      <span class="showcase-title">Tokenizer Boundaries</span>
      <span class="showcase-text">Separates byte/token assumptions from model assumptions, so text preprocessing cannot silently change the checked claim.</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="#what-a-checked-claim-looks-like-here">
    <img class="showcase-media" src="{{ '/assets/media/examples/bug-zoo/normalization-state.png' | relative_url }}" alt="Normalization state Bug Zoo case study"/>
    <span class="showcase-body">
      <span class="showcase-title">Normalization State</span>
      <span class="showcase-text">Tracks which statistics are training state, inference state, or explicit inputs rather than treating normalization as a black box.</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="#what-a-checked-claim-looks-like-here">
    <img class="showcase-media" src="{{ '/assets/media/examples/bug-zoo/batch-invariance.png' | relative_url }}" alt="Batch invariance Bug Zoo case study"/>
    <span class="showcase-body">
      <span class="showcase-title">Batch Invariance</span>
      <span class="showcase-text">States when processing one sample alone should agree with processing it as part of a batch.</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="#what-a-checked-claim-looks-like-here">
    <img class="showcase-media" src="{{ '/assets/media/examples/bug-zoo/float-autograd-boundaries.png' | relative_url }}" alt="Float and autograd boundaries Bug Zoo case study"/>
    <span class="showcase-body">
      <span class="showcase-title">Float and Autograd Boundaries</span>
      <span class="showcase-text">Shows how runtime <code>Float32</code> and reverse-mode claims are connected through named assumptions and proof statements.</span>
    </span>
  </a>
</div>

## How To Read The Zoo

Each Bug Zoo file is small enough to read directly. Read the whole chain: the bug
family, the bad pattern, the TorchLean object that names the intended behavior, and the theorem or
checker condition that makes the contract explicit.

Build the whole set:

```bash
lake build NN.Examples.BugZoo.All
```

For the cases that also have runnable checker commands, use:

```bash
lake exe verify -- camera-box3d-cert
lake exe verify -- all
```

The Lean files are the primary artifacts:

| Source file | Bug family | Contract exposed |
| --- | --- | --- |
| `AttentionMask.lean` | Causal masks, mask polarity, finite sentinels standing in for `-∞` | Future positions receive exactly zero attention weight under hard-mask semantics. |
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
| `FloatBoundary.lean` | Real-valued reasoning applied to Float32 runs | Runtime Float32 claims pass through a named IEEE-style bridge assumption. |
| `Geometry3DProjection.lean` | Camera convention, depth, layout, and projection-box errors | The checker recomputes projection, positive depth, and 2D box enclosure. |

## What “A Checked Claim” Looks Like Here

Each case study should end in a precise statement.

Here is the attention-mask claim in one line: under the hard-mask semantics, strict-future keys get
exactly zero attention weight.

```lean
theorem trueInfinityMask_future_attention_weight_zero :
  Spec.get2 (Spec.hardMaskedSoftmaxSpec scores (Spec.causalMask n)) i j = 0
```

And here is the Float32 boundary claim: runtime arithmetic rewrites to the explicit `IEEE32Exec`
model only under a named assumption.

```lean
theorem runtimeFloat32_add_rewrites_to_ieee32
    [RuntimeFloat32FiniteMatchesIEEE32Exec] (a b : _root_.Float32)
    (ha : Float32.isFinite a = true) (hb : Float32.isFinite b = true)
    (hr : Float32.isFinite (Float32.add a b) = true) :
    toIEEE32Exec (Float32.add a b) = IEEE32Exec.add (toIEEE32Exec a) (toIEEE32Exec b)
```

Those are the statement shapes Bug Zoo makes routine.
