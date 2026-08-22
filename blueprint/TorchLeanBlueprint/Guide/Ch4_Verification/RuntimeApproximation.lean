import VersoManual

open Verso.Genre Manual

#doc (Manual) "Runtime Approximation" =>
%%%
tag := "runtime-approximation"
%%%

A theorem over the reals and a claim about execution are different statements. Runtime
approximation is the layer that connects them.

The basic claim bounds the interpreted runtime value within an explicit tolerance of the ideal
value:

$$`\operatorname{toSpec}(y_{\mathrm{run}})\approx_\varepsilon y_{\mathrm{spec}}`

or, for a normed tensor statement,

$$`\left\|\operatorname{toSpec}(Y_{\mathrm{run}})-Y_{\mathrm{spec}}\right\|_\infty
\le \varepsilon.`

We encourage readers to come here after the autograd proof layer. The autograd theorems explain what
the ideal forward and backward maps mean; here we explain how far an executable or rounded path
may differ from that ideal, and how we keep that difference explicit.

# Why Real Proofs Are Not Float Proofs

It is tempting to prove a property over real numbers and then deploy float32 code under the same
assumption. TorchLean requires a bridge theorem relating the runtime path to the real specification
before applying the real-valued result to float32 execution.

That implication is not free. It needs a bridge theorem. Rounding, cancellation, overflow,
different reduction orders, fused kernels, and domain guards can all change the behavior. A
mathematical softmax is smooth; an implementation may use max-subtraction, exponent approximations,
finite accumulators, and library calls specific to a backend. A real convolution is a sum; a GPU
convolution may choose a different accumulation order or fused algorithm.

TorchLean's answer is to name the bridge:

1. prove the claim over the reals or over the spec;
2. prove an approximation relation between runtime tensors and spec tensors;
3. inflate margins, enclosures, or certificates by the approximation budget;
4. leave native hardware assumptions explicit when they are outside Lean.

This matches standard numerical analysis practice; see Higham's *Accuracy and Stability of
Numerical Algorithms* (https://epubs.siam.org/doi/book/10.1137/1.9780898718027) for the classical
background. It also matches the caution needed in neural network verification, where a tiny
rounding gap can matter if the verified margin is tiny.

# The Core Relation: Spec Tensors And Tolerances

The small vocabulary is defined by
[runtime approximation core](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Core.lean),
[spec approximation](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Core/SpecApprox.lean), and
[tolerances](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Core/Tolerance.lean).

The spec approximation layer defines the basic tensor relation. Given a conversion
`toSpec : α → SpecScalar`, a runtime tensor approximates a spec tensor when every component is close after
conversion to the spec scalar. The key names are:

- `tensorToSpec`: pointwise conversion from runtime tensor values to spec scalars.
- `linfNorm`: max style tensor norm used for error statements.
- `approxWith`: absolute tensor approximation using an explicit error tensor.
- `approxWithTol`: approximation using a tolerance object.
- `approxTensorWithTol`: packaged tensor tolerance relation.
- `Witness`: a small record for carrying a runtime value and its error evidence.

The tolerance API defines `ApproxTol`, with absolute, relative, and slack components.
`ApproxTol.absOnly`, `approxBound`, and `approxR` let later proofs move between simple absolute
error statements and more scale aware claims.

For a single scalar, the scale-aware shape is:

$$`|r-s|
\le
\varepsilon_{\mathrm{abs}}
+\varepsilon_{\mathrm{rel}}\,|s|
+\varepsilon_{\mathrm{slack}}.`

The choice is practical. An absolute tolerance of `1e-6` may be meaningful near zero and
irrelevant near `1e9`. A relative tolerance captures "small compared with the scale of the value."
TorchLean uses both styles, but it makes the choice explicit instead of burying it in test
thresholds.

## Reading A Tolerance Statement As A Contract

The tolerance relation is deliberately stronger than a unit test. A test has observed inputs. A
tolerance theorem has quantified inputs and a stated scalar interpretation:

```
toSpec : RuntimeScalar -> SpecScalar
runtime : Tensor RuntimeScalar s
spec    : Tensor SpecScalar s
eps     : ApproxTol
```

The proposition says every coordinate of `tensorToSpec toSpec runtime` is close to the matching
coordinate of `spec` under `eps`. That makes the trusted boundary easy to locate. If `toSpec`
interprets an executable rounded-real model, the theorem is about that rounded-real model. If the
actual deployment path is CUDA, cuBLAS, PyTorch, or a fused native kernel, a separate agreement
statement is needed before the theorem says anything about that path.

The most common deployment mistake is to skip this last sentence. A real theorem plus a numerical
test is useful evidence, but it is not a theorem about the tested runtime unless the tested runtime
is connected to the theorem's scalar model.

# Forward Graph Approximation

The forward graph theorem is in
[NN.Proofs.RuntimeApprox.Graph.ForwardApprox API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Graph/ForwardApprox.lean).
It mirrors the structure of the autograd tape proof, but the invariant is approximation rather than
derivative soundness.

The main objects are:

- `EList`: list of scalar error budgets indexed by shape.
- `approxCtx`: approximation relation for a whole context.
- `FwdNode`: a node with spec forward, runtime forward, bound, and local soundness.
- `FwdGraph`: a snoc list graph of forward nodes that carry approximation evidence.
- `FwdGraph.eval_approx`: theorem for the whole forward graph.

The local theorem on a `FwdNode` says: if each runtime input approximates the corresponding spec
input, then the runtime output approximates the spec output within this node's bound.

`FwdGraph.eval_approx` composes those local statements. When a graph appends a node, the proof uses
the node's local `sound` theorem, appends the new bound to the error context, and continues. It uses
the same architecture as the autograd soundness theorem: local correctness first, then a
global induction over the graph.

A tiny example is multiplication. The spec value is the real product, while the runtime value is the
rounded product computed by the chosen scalar model.

If $`x_{\mathrm{run}}` is within $`\varepsilon_x` of $`x_{\mathrm{real}}`, and $`y_{\mathrm{run}}` is
within $`\varepsilon_y` of $`y_{\mathrm{real}}`, the local multiplication lemma supplies a bound for
$`z_{\mathrm{run}}` versus $`z_{\mathrm{real}}`. The graph theorem then lets that
new bound feed the next node.

It is worth doing that local estimate once by hand. Writing the interpreted runtime inputs as
$`x+dx` and $`y+dy`,

$$`
(x+dx)(y+dy)-xy=x\,dy+y\,dx+dx\,dy.
`

Hence input uncertainty alone contributes at most

$$`|x|\,\varepsilon_y+|y|\,\varepsilon_x
  +\varepsilon_x\varepsilon_y.`

If $`x=2`, $`y=3`, $`\varepsilon_x=0.01`, and $`\varepsilon_y=0.02`, this part of the budget is
$`0.0702`.
The local rule then adds the rounding error of the multiplication itself. The number is not meant
as a universal tolerance; it shows why the graph carries operand scale and incoming error rather
than attaching one unexplained epsilon to every multiplication.

# Backward Graph Approximation

Backward approximation is developed in the
[backward approximation API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Graph/BackwardApprox.lean).
This file is the runtime approximation analogue of reverse mode AD.

The key objects are:

- `RevNode`: a forward node plus spec/runtime VJP functions and a VJP error transformer.
- `RevGraph`: reverse graph with local approximation evidence.
- `RevGraph.eval_approx`: forward approximation inherited from the forward graph.
- `RevGraph.backprop_approx`: theorem for reverse accumulation across the whole graph.

The theorem `RevGraph.backprop_approx` states that the runtime reverse pass approximates the spec
reverse pass, provided the input context, forward tape, and seed cotangents are appropriately
related. Addition during gradient accumulation is not treated as automatic; the theorem takes an
explicit `addBound` and `addSound` describing how accumulation affects error.

Backward passes are often dominated by sums: cotangents from fanout, reductions, convolution
gradients, and parameter gradient accumulation. Floating point addition is not associative, and
accumulation order matters. By making addition soundness an explicit parameter, the theorem states
the arithmetic model being used.

# Link Back To Autograd Algebra

The bridge file
[NN.Proofs.RuntimeApprox.Graph.LinkAutogradAlgebra API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Graph/LinkAutogradAlgebra.lean)
connects the runtime approximation graph shape back to the autograd algebra graph.

This bridge is structural. The runtime approximation layer uses the same
shape-indexed tensor-context idea that the typed API exposes as `TensorPack`, so the conversion is
not a semantic reinterpretation. The file defines `toNodeData` and `toGraphData`, then proves:

- `evalRuntime_of_toGraphData`;
- `backpropRuntime_of_toGraphData`.

In words, the runtime approximation graph can be viewed as an autograd algebra graph by forgetting
the approximation evidence and keeping the same forward/VJP structure. That means the two layers
compose cleanly:

autograd algebra: the reverse pass is the correct ideal VJP; runtime approximation: the executable
reverse pass stays close to that ideal VJP.

This claim needs careful wording. TorchLean does not collapse "correct gradient" and
"float gradient close to correct gradient" into one claim. The first is an equality theorem over the
ideal semantics. The second is an approximation theorem with a tolerance.

# Numerical Traces For The Canonical IR

The proof-relevant `FwdGraph` and `RevGraph` explain how local approximation theorems compose. Model
export and kernel selection, however, use the canonical op-tagged `NN.IR.Graph`. TorchLean connects
that graph directly to executable binary32 through
[the graph numerical certificate checker](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Graph/NumericalCertificate.lean).
It does not introduce a second deployment graph or a second interval type.

A certificate contains source enclosures, one derived range per IR node, the name of the range
registry, and the existing backend execution audit. A successful `check` stores the graph inside
`CheckedCertificate`; later replay cannot silently substitute a different graph or rule set.
Checking performs three independent executable validations:

1. validate that source and derived endpoints are finite and ordered;
2. reconstruct every supported range transfer from the graph;
3. re-run kernel selection and compare the selected capsules and numerical policies.

`GraphRangeRegistry` dispatches by primitive operation, not model family. The built-in transfers
cover source and shape-only nodes, pooling, arithmetic, inverse, ReLU, absolute value, directed
square root with a checked nonnegative domain, fixed-left reductions, matrix multiplication, MSE,
LayerNorm, softmax, sigmoid, tanh, sine, and cosine. Exponential is currently unsupported.
An unsupported operation fails at its node id; it is not replaced by an uninformative whole
interval.

```
#check Proofs.RuntimeApprox.NumericalCertificate.generateChecked
#check Proofs.RuntimeApprox.NumericalCertificate.GraphRangeRegistry
#check Proofs.RuntimeApprox.NumericalCertificate.numericalCoverage
#check Proofs.RuntimeApprox.NumericalCertificate.executeIEEE32
#check Proofs.RuntimeApprox.NumericalCertificate.CheckedRealExecution
#check Proofs.RuntimeApprox.NumericalCertificate.CheckedExecution.errorTrace
#check Proofs.RuntimeApprox.NumericalCertificate.tensor_error_le_width_of_check
#check Proofs.RuntimeApprox.NumericalCertificate.execution_error_trace_of_check
```

`GraphRangeContract.derive` is an executable range transformer, not a soundness theorem.
`generateChecked` and `check` establish that the stored trace is exactly the trace reconstructed by
the selected registry and kernel plan; they do not establish that every reconstructed interval
encloses the graph's real denotation.

`executeIEEE32` evaluates the same `NN.IR.Graph` with TorchLean's bit-level `IEEE32Exec` context and
checks every intermediate value against the stored ranges, rejecting NaN and infinity. This is a
reference replay, not an agreement theorem for a high-throughput backend.
`CheckedRealExecution` carries the separate semantic evidence: its fields require both the real
denotation equality and a pointwise real enclosure proof. When that evidence is supplied,
`CheckedExecution.errorTrace` combines the real enclosure with the successful IEEE replay to prove
a pointwise error trace whose bound is the interval width. A successful range check or replay alone
cannot stand in for the missing real-semantics proof.

Reduction order is read from the selected capsule. The portable reference capsules advertise the
fixed left fold used by the canonical tensor semantics. Native CUDA and LibTorch accumulations are
marked implementation-defined, so a fixed-left certificate cannot accidentally certify a cuBLAS,
cuDNN, fused-attention, or parallel-reduction schedule. Those paths require the order-independent
reduction bounds described in the floating-point chapter or a stronger backend-specific contract.

The local interval lemmas for arithmetic and selected nonlinear operations follow the inclusion
principle of IEEE 1788-2015. The current registry does not yet compose those lemmas into a theorem
that every accepted graph trace encloses exact graph semantics. The separation between local
rounding facts and a composed global error follows the standard treatment in Higham,
*Accuracy and Stability of Numerical Algorithms*, 2nd edition.

The canonical `NN.IR.Graph` lowering currently proves forward semantic preservation only. Its IR
nodes do not yet carry proved VJPs, so this certificate should not be described as a
canonical-IR backward certificate. Backward numerical theorems use the proof-bearing `RevGraph`
path below, which erases to executable autograd `GraphData` without discarding its VJP rules. An
autograd-capable lowering from canonical IR would need an additional correspondence theorem.

# Run A Complete Model Check

The executable example
[GraphNumericalCertificate.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/DeepDives/Floats/GraphNumericalCertificate.lean)
ends with a two-layer MLP rather than a single isolated operator. Run it from the repository root:

```
lake exe torchlean numerical_certificate
```

The model is a matrix pipeline with shapes that remain visible in the IR:

```
input [1,2]
  -> matmul [2,3]
  -> add bias [1,3]
  -> ReLU
  -> matmul [3,1]
  -> add bias [1,1]
```

Its last two report rows are labels emitted by the example:

```
  ok  two-layer MLP certificate
  ok  two-layer MLP IEEE replay
```

`mlpCertificate` checks that all ten graph nodes have a registered numerical rule. It derives every
range, selects the CPU capsules, and stores the graph, registry identity, source assumptions,
ranges, and backend audit in one artifact. `mlpReplay` then supplies concrete weights, biases, and
input values, executes the stored graph with `IEEE32Exec`, and checks every intermediate tensor.
The same file tests rejection of a tampered range, an unsupported operation, a violated
square-root domain, and an incompatible reduction policy. These Boolean and `Except` checks are
useful regression evidence; the example does not construct a `CheckedRealExecution`, so it is not
by itself a proof that the MLP's exact real execution is enclosed.

There is no MLP-specific branch in this process. The checker sees input, constant, matrix
multiplication, addition, and ReLU nodes. Other architectures can use the same walk when their
primitive operations are covered. New primitives extend the executable registry with a
`GraphRangeContract`; a semantic certificate additionally needs a theorem connecting that
contract's derived interval to the operation's real semantics.

The [complete numerical-runtime walkthrough](https://lean-dojo.github.io/TorchLean/examples/numerical-runtime/)
shows the model definitions, the five replay stages, the backend-capsule audit, and the handoff to
backward and optimizer bounds. It also states the current lowering boundary explicitly: canonical
IR has checked forward replay, while backward and optimizer composition currently begins from a
proof-bearing `RevGraph`.

# NF Operations: Rounded Real Arithmetic

The largest collection of local rules is
[NN.Proofs.RuntimeApprox.NF.Ops API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/NF/Ops.lean). `NF` is
TorchLean's rounded real neural float model. It gives arithmetic for proofs that resembles finite
precision without claiming that every hardware corner case has disappeared.

The file includes scalar and tensor approximation lemmas for common operations:

- arithmetic: `approx_add_nf`, `approx_sub_nf`, `approx_mul_nf`, `approx_div_nf_of_lb`;
- unary functions: `approx_exp_nf`, `approx_tanh_nf`, `approx_abs_nf`, `approx_neg_nf`;
- guarded operations: `safeLog`, `safeDiv`, `safe_log`;
- tensor rules: `approxTensor_add_spec`, `approxTensor_mul_spec`, `approxTensor_exp_spec`, `approxTensor_relu_spec`;
- graph nodes: `addNode`, `mulNode`, `expNode`, `reluNode`, `safeDivNode`, `softmaxNode`, `sumNode`.

Several of these lemmas make the numerical analysis tradeoff visible. Division requires a lower
bound away from zero (`approx_div_nf_of_lb`) or a guarded form (`safeDiv`). Square root and log need
domain protection. Exponential uses a mean value bound. Multiplication propagates both input
errors and a product term. These are the exact places where a proof over real numbers would be too
optimistic if copied directly onto a float implementation.

A safe division example has three pieces: the mathematical value is a guarded division, the runtime
value is computed by `safeDivR eps xR yR`, and the theorem states that the runtime value
approximates the guarded spec value under the declared tolerance.

The theorem covers guarded division, including the declared behavior at $`y=0`, so the proof
matches the stable programming pattern users ought to deploy.

## Reductions And Softmax: Where Error Budgets Grow

Reductions and normalization layers create coupled
error terms because many inputs flow into one output. TorchLean has explicit reduction approximation
lemmas in
[NN.Proofs.RuntimeApprox.NF.ReductionOps](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/NF/ReductionOps.lean):

```
#check Proofs.RuntimeApprox.NFBackend.approxTensor_reduce_sum_by_row_2d
#check Proofs.RuntimeApprox.NFBackend.approxTensor_reduce_mean_by_row_2d
#check Proofs.RuntimeApprox.NFBackend.approxTensor_reduce_sum_by_column_2d
```

The row-sum theorem accounts for the number of accumulated
terms and for the same row/column indexing used by the executable reducer. That is the sort of
detail that matters for LayerNorm, attention logits, pooled features, and minibatch losses.

Softmax needs even more care. Scalar logistic-style bounds are not a proof of axis softmax, because
axis softmax couples every coordinate through the denominator. TorchLean's
[axis softmax approximation API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/NF/SoftmaxAxis.lean)
proves the conditional NF rounded-real forward theorem `approxTensor_softmaxVecSpec`: it accounts for
max subtraction, exponential approximation, a sequential denominator sum, and division, under an
explicit denominator-error margin. `approxTensor_softmaxRowsSpec` lifts the result rowwise.

Hard masking uses exact Boolean mask semantics, including an exact-zero theorem for an all-blocked
row. `HardMaskedRowsEvidence` records the selected maxima, their approximation proofs, positive
real denominator lower bounds, and rounded denominator margins required by
`approxTensor_hardMaskedSoftmaxRowsSpec_of_max`. Backward bounds are provided by
`approxTensor_softmaxBackwardFromWeightsVecSpec` and `approxTensor_softmaxBackwardVecSpec`. The analytic facts
`sum_softmaxVec`, `sum_softmaxJvp`, and `abs_softmaxJvp_le_two_mul` establish normalization,
zero-sum JVP coordinates, and the dimension-independent bound
$`\lvert\operatorname{vjp}_i\rvert\le 2G`.

These are NF rounded-real theorems, not automatic claims about `IEEE32Exec`, a fused attention
kernel, or native binary32. The numerical-certificate registry's softmax rule only derives the
coarse range $`[0,1]`; that range is not a forward-error theorem.

## Normalization And Attention

Normalization and attention compose several domain-sensitive operations, so TorchLean records an
intermediate error trace instead of assigning one unexplained tolerance to the layer. The
[normalization approximation API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/NF/Normalization.lean)
handles arbitrary tensor rank once the selected mean and variance reductions have been certified.
Its centering, variance stabilization, square root, division, and affine stages expose the lower
bounds needed to keep the denominator away from zero. In particular,
`approxTensor_normalizeCore` assumes approximation evidence for the input, mean, variance, scale, bias,
and epsilon, plus a positive exact stabilized-variance lower bound and strict rounded-error
margins. It does not derive the mean and variance reduction bounds itself.

The [attention approximation API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/NF/Attention.lean)
builds scaled dot-product attention from matrix multiplication, scaling, stable axis softmax, and a
second matrix multiplication. A hard attention mask is semantic: blocked entries have zero
softmax numerator. It is not represented by adding a large finite negative constant, which would
change the function for sufficiently large logits. Backend capsules must therefore advertise a
matching mask convention before their output can inherit this theorem. The masked theorem
`approxTensor_scaledDotProductAttention_masked` consumes `HardMaskedRowsEvidence`; the canonical
inverse-square-root scale theorem also requires a positive feature dimension and a square-root
margin. These are conditional NF approximation theorems, not proofs for arbitrary fused attention
implementations.

```
#check Proofs.RuntimeApprox.NFBackend.normalizeCoreErrorTrace
#check Proofs.RuntimeApprox.NFBackend.approxTensor_normalizeCore
#check Proofs.RuntimeApprox.Attention.approxTensor_scaledDotProductAttention_masked
```

# Convolution Forward And Backward

Convolution stresses the approximation layer because it mixes indexing, padding, nested sums, and
gradients with respect to inputs and parameters. TorchLean has dedicated normal form APIs:

- [NN.Proofs.RuntimeApprox.NF.ConvForward API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/NF/ConvForward.lean)
- [NN.Proofs.RuntimeApprox.NF.ConvBackward API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/NF/ConvBackward.lean)
- [NN.Proofs.RuntimeApprox.NF.Conv API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/NF/Conv.lean)

The convolution forward API proves pointwise and full tensor approximation for convolution forward:

- `approx_conv2d_point` for one output coordinate;
- `approxTensor_conv2d_spec` for the full output tensor;
- `conv2dNode` to package the theorem as a graph node.

The proof has a lot of indexing work because the spec definition and the runtime replay need to
read the same padded input entries. Lemmas such as `foldl_flatMap`,
`entry_eq_scalar_get_at_or_zero3`, and facts about padded input alignment are essential; they are
what prevents the proof from silently comparing two different convolutions.

The convolution backward API covers the three reverse operators:

- `approxTensor_conv2d_bias_deriv_spec`;
- `approxTensor_conv2d_kernel_deriv_spec`;
- `approxTensor_conv2d_input_deriv_spec`.

It also packages the result as `conv2dRevNode`, so the backward approximation theorem can compose
convolution with the rest of a reverse graph. Here is the local-to-global pattern in its cleanest
form: first prove the hard local operator approximation, then hand it to `RevGraph`.

# Rounded Optimizer Steps

The backward theorem produces approximate gradients; a training claim must still account for the
optimizer arithmetic. The generic
[optimizer numerical contract](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Optimizer.lean)
records an exact state, a rounded state, their relation, a one-step bound transformer, and the proof
that the relation survives one update. `NumericalStepContract.run_approx` proves the corresponding
finite-run result once for every optimizer satisfying that interface.

The concrete [NF optimizer proofs](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/NF/Optimizers.lean)
use the optimizer equations directly:

- SGD propagates learning-rate, gradient, and parameter error;
- momentum SGD additionally propagates the momentum-buffer error;
- AdamW records errors for both moments, bias correction, square root, adaptive division,
  decoupled weight decay, and the final subtraction.

AdamW needs more than a nominal epsilon. Its theorem requires a positive lower bound on the exact
bias-corrected second moment and explicit margins showing that rounding does not cross either the
square-root or division boundary. This is the numerical counterpart of the recurrence in Kingma and
Ba's Adam paper (https://arxiv.org/abs/1412.6980) and the decoupled decay in Loshchilov and Hutter's
AdamW paper (https://arxiv.org/abs/1711.05101).

```
#check Proofs.RuntimeApprox.Optimizer.NumericalStepContract.run_approx
#check Proofs.RuntimeApprox.NFBackend.Optimizer.approxTensor_sgd_update
#check Proofs.RuntimeApprox.NFBackend.Optimizer.approxTensor_momentumSGD_update
#check Proofs.RuntimeApprox.NFBackend.Optimizer.approxTensor_adamW_update
```

# Scale Aware Tolerances

The scale layer is split across
[scale bounds](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Scale.lean),
[scale approximation](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Scale/ScaleApprox.lean),
[forward scale propagation](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Scale/ForwardScale.lean), and
[backward scale propagation](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Scale/BackwardScale.lean).

The scale approximation API defines `BList`, a list of nonnegative scale bounds indexed by shape,
plus helpers such as `scaleTensor`, `scaleCtx`, and `tolFromEpsScale`. An absolute error budget is
computed from a machine-like epsilon times a local scale bound.

A graph can then carry both "how close" and "at what scale" information. The lemmas
`approxTensorWithTol_from_scale` and `approxCtx_get_tolFromEpsScale` connect scale estimates back to the
tolerance API used by graph theorems.

This remains a separate layer because not every proof needs scale aware reasoning. Small examples
and operator proofs written by hand are often clearer with absolute tolerances. Larger deployment
claims usually need scale, because one global absolute epsilon is rarely meaningful across all
activations and gradients.

# FP32 And Verification Margins

The [FP32 layer approximation API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/FP32/Layers.lean),
[FP32 MLP approximation API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/FP32/MLP.lean), and
[FP32 CROWN bridge API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/FP32/CROWN.lean) connect the approximation
style to layerwise and verifier reasoning.

The CROWN connection is where runtime approximation meets certification. A verifier over the reals
may prove a margin, but a float runtime can differ from ideal real arithmetic. To transfer the claim,
the real margin must dominate the runtime approximation budget: the certified lower margin after
subtracting the runtime error still has to be positive.

When that inequality is proved, the runtime prediction is still certified.

The Float32 soundness layer uses the same separation. `FP32` is a proof model, `IEEE32Exec` is an
executable bit oriented model, and native hardware remains a named assumption unless a bridge
theorem covers the path being used.

# A Worked Mental Example

Suppose we have a two layer classifier whose hidden layer is $`\operatorname{ReLU}(W_1x+b_1)` and
whose output is $`W_2h+b_2`.

The ideal proof might establish that, for all inputs in a box, the margin over the reals for class
$`0` over class $`1` is at least $`0.05`.

A float deployment theorem adds the runtime approximation statement: for every input in the same
box, $`y_{\mathrm{run}}` approximates $`y_{\mathrm{real}}` within $`0.01` per relevant logit.

Then the margin proof must be adjusted:

$$`f_0^{\mathrm{run}}(x)-f_1^{\mathrm{run}}(x)
\ge 0.05-0.01-0.01
=0.03>0.`

The calculation is small, but it is the whole point of the layer. The real theorem, the runtime
approximation theorem, and the final runtime claim each spend a named budget.

Only after that bridge do we get a classification claim about runtime behavior. TorchLean keeps
autograd algebra, verifier bounds, FP32 semantics, and runtime approximation as separate proof
layers because they are different mathematical facts that happen to support one deployment
workflow.

# End To End Rounded Training

The normal-form end-to-end file connects the proof-bearing reverse graph to executable autograd
`GraphData`, extracts typed parameter gradients, and composes those gradients with the optimizer
contracts:

```
#check Proofs.RuntimeApprox.NFBackend.eval_approx_graphData
#check Proofs.RuntimeApprox.NFBackend.backprop_approx_graphData
#check Proofs.RuntimeApprox.NFBackend.backprop_gradient_approx_graphData
#check Proofs.RuntimeApprox.NFBackend.backprop_optimizer_update_approx_graphData
#check Proofs.RuntimeApprox.NFBackend.trainingStepTrace
```

These are graph-level bridge theorems:

- `eval_approx_graphData` says evaluating the executable forward graph is close to evaluating the
  spec forward graph when the local node approximation obligations have been supplied.
- `backprop_approx_graphData` says the same style of statement for reverse accumulation, with the
  accumulation error model still explicit.
- `backprop_optimizer_update_approx_graphData` applies any numerical optimizer contract to one typed
  parameter gradient produced by that executable reverse pass. SGD uses trivial step evidence;
  AdamW supplies its bias-correction and positivity margins through `StepData`. A model with several
  parameter tensors applies the same theorem at each parameter index.
- `trainingStepTrace` computes a proof-free report of forward, backward, gradient, parameter, and
  optimizer-state bounds. Its interpretation comes from the surrounding approximation theorems,
  not from the report record itself.

That is the runtime approximation analogue of the autograd proof architecture. Local operator
lemmas are the leaves; graph theorems compose them; deployment claims then combine the graph theorem
with any scalar/backend assumptions.

## Reading One Bounded Training Step

Take a parameter tensor at index $`i`. The real reverse pass produces $`g_{\mathrm{spec}}`; executable
rounded reverse mode produces $`g_{\mathrm{run}}`; and `backprop_gradient_approx_graphData` proves

$$`\|\operatorname{toSpec}(g_{\mathrm{run}})-g_{\mathrm{spec}}\|_\infty
\le \varepsilon_g.`

Suppose the current parameters and learning rate have errors $`\varepsilon_p` and
$`\varepsilon_{\mathrm{lr}}`. For SGD,
the two executions perform the same equation in their respective scalar systems,

$$`p' = p-\eta g.`

`sgdStepBound` first bounds the rounded product $`\eta g`, including both input errors and the new
multiplication rounding, then bounds the final subtraction. The graph-level theorem returns

$$`\|\operatorname{toSpec}(p'_{\mathrm{run}})-p'_{\mathrm{spec}}\|_\infty
\le \varepsilon_{p'}`

with $`\varepsilon_{p'}` equal to that computed bound, not a user-chosen test tolerance. Momentum SGD uses
the same theorem and additionally returns a bound for the updated momentum buffer.

For AdamW, the route is longer: update both moments, apply bias correction, take the second-moment
square root, form the adaptive learning rate, apply decoupled decay, and subtract the Adam update.
`AdamWStepErrorTrace` keeps the error after each stage. Its `StepData` asks for a positive `eta`
below every exact corrected second moment and checks that the rounded error remains smaller than
the square-root and denominator margins. The optimizer theorem remains the same; only the local
contract's validity evidence is richer than SGD's.

These proofs describe numerical recurrences for explicit states. They do not automatically cover
the native trainer's parameter lookup, mutable moment-buffer allocation, per-parameter clocks, or
checkpoint restore. Those runtime mechanisms must be shown to instantiate the same state and step
relation before `NumericalStepContract.run_approx` can be cited for a concrete training process.

# Runtime Approximation APIs

The definitions are organized in the same order as the proof: define closeness, prove local operator
bounds, compose them over forward and backward graphs, and finally connect the result to autograd.

- The [tolerance API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Core/Tolerance.lean) and
  [spec approximation API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Core/SpecApprox.lean) define the approximation relation.
- The [forward graph approximation API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Graph/ForwardApprox.lean) contains
  `FwdGraph.eval_approx`; the [backward graph approximation API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Graph/BackwardApprox.lean)
  contains `RevGraph.backprop_approx`.
- The [normal form operator API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/NF/Ops.lean) supplies local obligations, including
  domain-sensitive operations such as division and safe log.
- The [convolution forward API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/NF/ConvForward.lean) and
  [convolution backward API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/NF/ConvBackward.lean) handle a larger operator family.
- The [scale approximation API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Scale/ScaleApprox.lean) supports scale-aware error bounds.
- The [autograd algebra link API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Graph/LinkAutogradAlgebra.lean)
  connects approximation to the autograd proof layer.
- The [optimizer contract API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Optimizer.lean)
  and [NF optimizer instances](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/NF/Optimizers.lean)
  continue the backward error budget through parameter updates.

# Runtime Agreement

For supported graph and operator fragments, runtime approximation proves that a runtime or rounded
computation stays within a stated tolerance of a spec computation. CUDA kernels, vendor library
paths, compiler rewrites, and PyTorch-exported graphs need their own agreement statements when a
claim is about those paths.

The intended TorchLean shape is to allow fast code and imported artifacts while keeping the
mathematical promise visible. A deployment claim combines a real proof, an approximation theorem,
and any finite or runtime assumptions that remain outside the theorem.

When the approximation theorem is present, the bridge is proved. When it is not present, the claim
should say which runtime path or external producer supplies the remaining evidence.
