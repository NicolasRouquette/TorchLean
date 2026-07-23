import VersoManual

open Verso.Genre Manual

#doc (Manual) "BugZoo Catalog" =>
%%%
tag := "bugzoo-catalog"
%%%

Imagine reviewing a causal-attention optimization. The output shape is right, the benchmark is
faster, and several sampled prompts look normal. The missing question is semantic: can any future
token receive attention weight? BugZoo begins at moments like this. It takes failures seen in
frameworks, compilers, deployment tools, and serving systems and asks what small object or
proposition would have made the intended behavior explicit before the failure reached production.

The [BugZoo catalog API](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/BugZoo/) and
[BugZoo overview](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/README.md) focus on the TorchLean fragment itself:
once a computation enters a typed TorchLean spec, shape changes, masks, token bounds, finite domain
choices, stateful normalization parameters, and backend semantics become named objects that can be
checked.

Every file is small for a reason. A theorem such as “strict-future attention weight is zero” is
easier to reuse and harder to overstate than a large demo that merely happens to produce plausible
tokens. The motivating incident explains why the contract matters; the Lean declaration says
exactly what was checked.

# Compile The Catalog

All entries are imported by `NN/Examples/BugZoo/All.lean`. Compile them together with:

```
lake env lean NN/Examples/BugZoo/All.lean
```

A successful command is silent. It means every definition and theorem in the catalog elaborated;
it does not mean that every external framework implementation satisfies those contracts.

`All.lean` is also the completeness boundary for the maintained catalog: an example file not
imported there is not covered by this compile command. The contracts describe TorchLean reference
objects; external framework conformance requires a separate importer, refinement theorem, or
explicit assumption.

For a more interactive pass, create `BugZooAudit.lean`:

```
import NN.Examples.BugZoo.AttentionMask
import NN.Examples.BugZoo.ShapeAndBroadcast

#check NN.Examples.BugZoo.AttentionMask.exactMaskedLogit_blocked_exp_zero
#check NN.Examples.BugZoo.AttentionMask.trueInfinityMask_future_attention_weight_zero
#check NN.Examples.BugZoo.ShapeAndBroadcast.addSingletonBatch
#check NN.Examples.BugZoo.ShapeAndBroadcast.broadcastRowToMatrix_firstRow
```

Open the file in the Lean Infoview. The attention theorem quantifies over every strict-future
position `j > i`; the shape theorem exposes the singleton batch insertion and the proof-carrying
broadcast as different operations. These are the contracts. The motivating PyTorch snippets in the
source comments explain the bug family, but are not imported as trusted evidence.

# Example Anatomy

A good BugZoo example has four parts:

- the pattern in the framework that goes wrong;
- the TorchLean object that names the intended behavior;
- the theorem, structure, or definition that marks the checked boundary;
- the external conformance obligation or unsupported scope that remains outside the checked claim.

Most ML bugs here are semantic rather than syntactic. The program often still returns a tensor. The
loss may still be a scalar. A compiled graph may still run. An LLM server may still emit tokens.
BugZoo asks whether those tensors and tokens still mean what the user thought they meant.

The common contract shape is:

$$`\text{bug pattern}
\;\leadsto\;
\text{TorchLean object}
\;\leadsto\;
\text{checked claim}`

The examples stay short so the semantic invariant is visible. Each case isolates the condition that
would have made the original kind of bug harder to miss.

Some representative contract shapes:

:::table +header
*
  * Example
  * Contract shape
*
  * Attention mask
  * $`j>i\Rightarrow A_{ij}=0`
*
  * Batch invariance
  * $`\operatorname{select}(\operatorname{mapBatch}(f,X),i)=f(X_i)`
*
  * Tokenizer boundary
  * token ids inhabit `Fin vocabSize`
*
  * KV cache
  * appended key/value appears at the final slot
*
  * Float boundary
  * runtime Float32 agrees with `IEEE32Exec` under a named agreement
*
  * Compiler boundary
  * target output equals source output
*
  * Stable loss
  * logits path uses log-softmax semantics
*
  * Ignored labels
  * inactive labels contribute zero
*
  * LayerNorm degenerate axis
  * zero-variance normalization follows an explicit epsilon policy
*
  * 3D projection
  * camera projection exposes depth and denominator preconditions
:::

# The Examples

## Shape And Broadcast

[NN.Examples.BugZoo.ShapeAndBroadcast source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/ShapeAndBroadcast.lean)
records one of TorchLean's core design choices: shapes belong in the object, not as an
afterthought. The example uses a missing batch dimension and a reduce then broadcast pattern as its
running examples. In NumPy style tensor libraries, a reduced vector can silently expand back across a
matrix and still produce a plausible loss. In TorchLean, ordinary elementwise operations require the
same shape, and explicit broadcasting carries `Shape.CanBroadcastTo` evidence.

The local contract is concrete: `addSingletonBatch` names the batch insertion,
`reduceRows` names the reduction, and `broadcastRowToMatrix` names the expansion. The theorem
`broadcastRowToMatrix_firstRow` shows the preferred style: if a dimension is added, removed, or
reintroduced, the documentation and proof script can point at a term that did it.

The empirical motivation is tensor shape fault work such as
[SFData](https://doi.org/10.1145/3533767.3534383), plus
[numerical bug studies](https://conf.researchr.org/details/ase-2022/ase-2022-nier-track/18/An-Empirical-Study-on-Numerical-Bugs-in-Deep-Learning-Programs)
that found bad reductions and accidental broadcasting in real DL programs.

The contract shape is:

$$`\operatorname{add} :
\operatorname{Tensor}(\alpha,s)\to
\operatorname{Tensor}(\alpha,s)\to
\operatorname{Tensor}(\alpha,s)`

and explicit broadcast operations carry evidence that the source shape can be broadcast to the
target shape.

## Stable Loss

[NN.Examples.BugZoo.StableLoss source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/StableLoss.lean) is about losses that
look mathematically harmless but fail numerically. The classic sketch is `softmax` followed by
`log`: if a probability rounds to zero, the log path can produce infinities and downstream NaNs.
TorchLean keeps two APIs separate. Logits should use `crossEntropyLogitsSpec`, which unfolds through
`logSoftmaxSpec`; probability inputs use the clipped probability form `crossEntropySpec`.

The checked hooks are small. `crossEntropyLogits_uses_logSoftmax` says that the logits
loss really takes the stable logits path. `crossEntropyProbabilities_clips_before_log` says that the
probability path clamps before `log`. `safeDivSpec_unfold` gives division with domain assumptions an
explicit node protected by epsilon rather than hiding it in an optimizer or backend.

This example is motivated by [TensorFuzz](https://proceedings.mlr.press/v97/odena19a.html), which
targeted rare numerical failures, and by
[empirical studies of numerical bugs](https://conf.researchr.org/details/ase-2022/ase-2022-nier-track/18/An-Empirical-Study-on-Numerical-Bugs-in-Deep-Learning-Programs)
involving `log`, `sqrt`, division, `exp`, and reductions.

## Ignored Labels

[NN.Examples.BugZoo.IgnoredLabelLoss source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/IgnoredLabelLoss.lean) turns a
corner case that is easy to dismiss into a reduction contract.
[PyTorch issue #75181](https://github.com/pytorch/pytorch/issues/75181) reported an `ignore_index`
case where all labels were ignored and the result was `nan`.

TorchLean exposes the policy instead of copying every branch of a framework kernel.
`labelContribution false loss = 0` states that ignored labels contribute no scalar loss, and the
example's helper for empty reductions names one policy for the all-ignored case. The bug appears
when an empty active set reaches an unnamed backend reduction.

The policy is the definition:

$$`\operatorname{labelContribution}(active,loss)
=
\begin{cases}
loss, & active\\
0, & \neg active
\end{cases}`

The all ignored case is then a declared reduction policy, not an accidental division by zero.

## Autograd Domain

[NN.Examples.BugZoo.AutogradDomain source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/AutogradDomain.lean) follows
[PyTorch's own autograd note about division by zero](https://docs.pytorch.org/docs/main/notes/autograd.html#division-by-zero-in-autograd).
If a graph computes `x / 0` and masks the bad value afterward, the forward result may look hidden
while the backward graph still contains the undefined operation.

TorchLean's example names the difference between "divide first, mask later" and "safe divide, then
mask." `maskAfterSafeDiv` records `safedivSpec` before the mask, and
`maskAfterSafeDiv_uses_epsilon_denominator` unfolds to division by `denominator + epsilon`. The
contrast definition, `unsafeDivThenMask`, stays in the file so importers and reviewers can see the
risky graph shape rather than treating every masked expression as safe.

## Attention Mask

[NN.Examples.BugZoo.AttentionMask source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/AttentionMask.lean) is the catalog
entry for causal mask semantics. Attention masks fail by polarity, layout, fake negative infinity,
fully masked rows, and interactions with API flags. PyTorch has had relevant reports for
`MultiheadAttention`, including
[`is_causal=True` being ignored when `need_weights=True`](https://github.com/pytorch/pytorch/issues/99282)
and [fully masked heads producing NaNs](https://github.com/pytorch/pytorch/issues/160064) when
weights are requested.

The example connects the runtime convention to the math. Lean's real numbers do not contain literal
negative infinity, so the file first uses `EReal` to state the exact fact:
`exp(-infinity) = 0`. TorchLean's ordinary attention spec then uses the equivalent hard masked
softmax numerator. The theorem `trueInfinityMask_future_attention_weight_zero` says that strict
future positions receive exactly zero attention mass under the causal mask. That property is the one
needed when reasoning about autoregressive output causality.

The spec also makes fully blocked rows total: if every mask entry in a row is `false`, the complete
weight row is zero. This is distinct from first forming a vector of negative infinities and then
applying an implementation whose normalization may produce `NaN` on that row.

In formula form:

$$`j>i \quad\Longrightarrow\quad
\operatorname{attentionWeight}_{causal}(i,j)=0`

This example also points back to the original transformer paper,
[*Attention Is All You Need*](https://arxiv.org/abs/1706.03762).

## Compiler Boundary

[NN.Examples.BugZoo.CompilerBoundary source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/CompilerBoundary.lean) is the
wrong-code example. It is about optimized graphs that run, return tensors, and are nevertheless not
the same computation as the source graph. [NNSmith](https://arxiv.org/abs/2207.13066) found
compiler bugs across TVM, TensorRT, ONNXRuntime, and PyTorch,
[FreeFuzz](https://arxiv.org/abs/2201.06589) found framework/API bugs by mining real snippets, and
a recent [PyTorch compiler correctness study](https://arxiv.org/abs/2604.08720) focuses directly
on silent `torch.compile` wrong outputs.

The local object is the `SemanticBoundary` structure. It has a source evaluator, a target
evaluator, an implementation relation, and a preservation field. This compact shape is the
reusable claim behind heavier IR compiler correctness theorems: accepted target code must agree with
the source semantics on every input. A runtime check can make us more confident, but it is not the same
kind of artifact as a semantic boundary.

The reusable statement is:

$$`\operatorname{implements}(source,target)
\quad\Longrightarrow\quad
\forall x,\; target(x)=source(x)`

## Float Boundary

[NN.Examples.BugZoo.FloatBoundary source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/FloatBoundary.lean) is where we
refuse to let proofs over real numbers silently masquerade as float32 deployment guarantees.
Floating point verification attacks show why that matters: a property proved over reals can be
invalidated by finite precision, exceptional values, fused operations, denorm behavior, or changed
reduction order. The example cites
[Jia and Rinard's warning paper](https://doi.org/10.1109/SPW50608.2020.00058).

TorchLean's answer is explicit modeling. `IEEE32Exec` is the executable bit level float32 model.
Runtime `Float32` primitives are not transparent to the Lean kernel, so the theorem
`runtimeFloat32_add_rewrites_to_ieee32` requires the named assumption
`RuntimeFloat32FiniteMatchesIEEE32Exec`, together with finite-input and finite-result hypotheses.
We built that friction on purpose. If a proof uses the
float32 model, the boundary says where runtime conformance entered.

The boundary is therefore visible in the theorem shape:

$$`\operatorname{RuntimeFloat32FiniteMatchesIEEE32Exec}
\quad\Longrightarrow\quad
\operatorname{runtimeAdd}(x,y)
=
\operatorname{IEEE32Exec.add}(x,y)`

## Normalization State

[NN.Examples.BugZoo.NormalizationState source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/NormalizationState.lean)
covers BatchNorm style bugs where the formula or state is wrong but the layer still emits a tensor.
[CRADLE](https://www.cs.purdue.edu/homes/lintan/publications/cradle-icse19.pdf) reported a
BatchNorm epsilon placement issue across backends, while
[LEMON](https://lingming.cs.illinois.edu/courses/cs598ast-f20/paper-dnn-lib-testing.pdf) found
BatchNormalization moving stat bugs and BatchNorm layers that produce NaNs.

The example splits the concern in two. First, `normalizeCore_scalar_uses_variance_plus_epsilon` shows
that TorchLean's scalar normalization puts epsilon inside the variance term before square root.
Second, `RunningStats` packages inference time mean and variance as explicit inputs. Train/eval mode
bugs are often state bugs. If running statistics are ambient mutable framework state, the proof
cannot see them. If they are arguments to `batchNormEvalWithStats`, the state boundary is visible.

## LayerNorm Degenerate Axis

[NN.Examples.BugZoo.LayerNormDegenerateAxis source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/LayerNormDegenerateAxis.lean)
records a related but different normalization bug family. LayerNorm can receive an axis with one
element or a constant slice. The tensor output still has the expected shape, but the variance term is
zero and the implementation's epsilon convention determines whether the result is finite and
meaningful.

TorchLean's contract is to expose the denominator policy instead of burying it in a fused kernel.
The invariant is small:

$$`\operatorname{denom}=\sqrt{\operatorname{variance}+\epsilon}`

and the degenerate case should be governed by the same explicit formula. That gives later kernels a
reference behavior to match.

## Constant Normalization Slice

[NN.Examples.BugZoo.ConstantNormalizationSlice source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/ConstantNormalizationSlice.lean)
keeps the constant-slice case visible. A constant row or channel slice is not exotic; it appears in
padding-heavy batches, masked tokens, uniform images, and clipped signals. If the normalization
implementation accidentally assumes positive variance, a constant slice can produce a division by
zero, `NaN`, or a backend-specific branch.

The contract is the same discipline used throughout BugZoo: name the slice, name the variance, and
name the epsilon-protected normalization result. An optimized kernel can then be tested or
proved against the reference object.

## Batch Invariance

[NN.Examples.BugZoo.BatchInvariance source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/BatchInvariance.lean) is about
serving systems that change outputs depending on which other requests share a batch. This can come
from dynamic batching, kernel selection, reduction order, and scheduling details even when user
randomness is off. The catalog points at recent LLM serving discussions and studies:
[Thinking Machines on inference nondeterminism](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
and an [LLM inference engine bug study](https://arxiv.org/abs/2506.09713).

The checked reference semantics is `mapBatch`: apply the same function to each example independently
to each row. The theorem `mapBatch_select_eq_single` says that selecting one row from the batched
result equals evaluating that row alone. That theorem is the semantic target a runtime path should refine;
any explicit float32 tolerance, reduction-order difference, or serving-system batching policy
belongs in a separate runtime assumption.

## KV Cache

[NN.Examples.BugZoo.KVCache source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/KVCache.lean) models cache accounting in
autoregressive inference. LLM engines fail through shifted caches, wrong cache slots, config/shape
mismatches, resource scheduling, and interaction with positions or tokenizers. The broader source
trail is the [LLM inference engine bug study](https://arxiv.org/abs/2506.09713).

The local contract is exact: a cache append operation preserves existing entries and puts the newly
decoded key/value in the final slot. That sounds obvious, but it is the kind of invariant that
becomes fragile when a serving system combines paged attention, batching, and mutable buffers. The
BugZoo example says what the reference operation means before any external cache manager is trusted
to implement it.

## RoPE Position

[NN.Examples.BugZoo.RoPEPosition source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/RoPEPosition.lean) pairs naturally
with the KV cache example. Rotary position embeddings make position accounting part of the model's
meaning. A decode position off by one can be hard to notice because the tensor shapes still line
up and the model still produces tokens.

The file introduces `PositionSchedule` and `appendNextPosition`. The theorem
`appendNextPosition_last` states that the newly appended token gets exactly the next sequence index.
The schedule is explicit rather than derived from ambient mutable state. It follows the same design
move as the normalization example: if the state affects semantics, it should appear in the object we
inspect.

## Tokenizer Boundary

[NN.Examples.BugZoo.TokenizerBoundary source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/TokenizerBoundary.lean) marks
the boundary before tensors even reach the model. Tokenizer/config mismatches can disagree about
vocabulary size, padding, EOS, or special token IDs while the neural network code itself looks
ordinary. The [LLM inference engine bug study](https://arxiv.org/abs/2506.09713) lists
tokenizer/config bugs as a real serving class.

TorchLean's current contract is small but valuable: token IDs inside the typed fragment can be
represented as `Fin vocabSize`. `padId_in_vocab` and `tokenAt_in_vocab` are almost tautological,
which means token IDs outside the vocabulary are no longer a late runtime condition once data has
crossed into this representation. The remaining producer step is the importer or tokenizer bridge
that constructs those `Fin` values from external bytes.

## Geometry 3D Projection

[NN.Examples.BugZoo.Geometry3DProjection source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/Geometry3DProjection.lean)
shows why BugZoo is not limited to language-model incidents. Vision and robotics pipelines often
project 3D points to image coordinates with a camera matrix. The formula is familiar:

$$`(u,v)=\left(\frac{x}{z},\frac{y}{z}\right)`

up to intrinsics and coordinate conventions. The bug family is also familiar: depth is zero or has
the wrong sign, coordinate frames are swapped, or the denominator convention is implicit. A tensor
can still be emitted, but the geometry claim no longer matches the camera model.

The TorchLean response is to expose projection preconditions and the safe division boundary. A
verified perception or control argument should not inherit an unspoken "all points have valid
positive depth" assumption from preprocessing code.

# Use A Contract In A Real Review

Return to the causal-attention optimization from the opening. Start with the reference theorem in
`AttentionMask`: for every query row, a strict-future key has exactly zero weight. If the optimized
path changes layouts, use the shape and broadcast examples to make that rearrangement explicit. If
it crosses into a native compiler or float kernel, the compiler and Float32 examples show the two
additional obligations: preserve the source computation and state where runtime arithmetic is
assumed to match the reference semantics.

Now test both sides of the boundary. A past or current token should be able to contribute; a future
token must not. A fully blocked row should follow the declared zero-row policy instead of producing
an accidental `NaN`. Compile `All.lean` after adding the focused example so the maintained catalog
actually imports it.

The result is narrower and auditable: one recognizable failure becomes an explicit contract, one
negative control, and one visible conformance obligation. The same pattern turns unstable losses
into domain-aware specs, state bugs into explicit arguments, tokenizer and cache mistakes into
import or append contracts, and wrong code into a semantic-preservation question.
