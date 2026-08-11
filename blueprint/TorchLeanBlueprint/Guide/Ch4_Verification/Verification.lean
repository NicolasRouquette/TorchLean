import VersoManual

open Verso.Genre Manual

#doc (Manual) "Neural Network Verification" =>
%%%
tag := "verification"
%%%

Testing asks what a network did on the inputs we tried. Verification asks what it must do on every
input in a set. For a classifier `f`, an input region `X`, the intended class `y`, and a competing
class `j`, a typical robustness target is

$$`\forall x\in X,\qquad f_y(x)-f_j(x)>0.`

That single formula hides most of the engineering. We need to know which model `f` denotes, how
`X` was represented, how the output bounds were obtained, and whether the arithmetic was exact or
rounded. TorchLean therefore refuses to make the word "certificate" carry the whole argument by
itself. There is a graph with an exact meaning, a theorem explaining why a bound procedure encloses
that meaning, a finite artifact or checker run, and, when the checker is meant to feed the theorem, a
bridge from acceptance to the theorem's hypotheses. Only their composition yields the final claim.

We begin with a network small enough to work out on paper. That lets us compare the printed result
with the mathematics before opening the general theorem stack.

# A Complete Robustness Run

The bundled robustness workflow constructs a two-output network, compiles it to TorchLean's
canonical IR, places an $`\ell_\infty` box of radius `0.1` around `[1, 1]`, and computes both IBP and CROWN
bounds:

```
lake exe verify -- torchlean-robustness
```

The relevant part of the output is:

```
compiled IR nodes: 4
x0 = [1.000000, 1.000000], eps = 0.100000
[IBP] logits lo = [1.800000, -2.200000]
[IBP] logits hi = [2.200000, -1.800000]
[IBP] margin(lo0 - hi1) = 3.600000
[IBP] certified? true
[CROWN] margin(lo0 - hi1) = 3.600000
[CROWN] certified? true
[CROWN-backward] margin lo = 3.600000
[CROWN-backward] margin hi = 4.400000
[CROWN-backward] certified? true
```

The interval calculation says that class-zero's logit is at least `1.8`, while class-one's logit is
at most `-1.8`. Consequently,

$$`\inf_{x\in X}(f_0(x)-f_1(x))\ge 1.8-(-1.8)=3.6>0.`

For this example there is no mystery hidden in those numbers. Each input coordinate lies in
$`[0.9,1.1]`; the first layer adds them, so its preactivation lies in $`[1.8,2.2]`. That interval is
strictly positive, hence ReLU is the identity throughout the box. The output weights are $`1` and
$`-1`, giving logits in $`[1.8,2.2]` and $`[-2.2,-1.8]`. The margin is twice the hidden value and lies
in $`[3.6,4.4]`. IBP is exact on this tiny path because the ReLU phase never changes.

The printed `true` is useful, but it is not itself the theorem. To turn the run into a proof, we
must connect the compiled graph to the source model, the propagated boxes to the graph denotation,
and the positive lower margin to the classification property. A claim about native Float32 needs
one more link: that native execution refines the arithmetic used in the proof.

The obligations also localize maintenance. A compiler change affects the source-to-graph argument;
a new CROWN activation affects bound propagation; and a CUDA deployment claim requires the native
arithmetic link in addition to the real-valued theorem.

# Semantic Target And Graph Boundary

The verifier operates on the canonical `NN.IR.Graph`. An interval or affine form is meaningful only
relative to a denotation of that same graph, parameter store, and input box. A compiler theorem is
therefore part of a source-model claim.

TorchLean has two relevant forward correspondences. The typed first-order
[proved forward fragment](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/TorchLean/Proved/Public.lean)
compiles `NN.Verification.TorchLean.Proved.Program` values. Its constructors cover constants,
parameters, arithmetic, ReLU, `exp`, `log`, inverse, matrix products, reshapes and permutations,
last-axis softmax, 2D LayerNorm, linear and convolution layers, and MSE loss.
`compileForward_wellFormed` proves structural well-formedness, while
`runForwardIR_eq_evalForward` proves equality with the typed program evaluator.

The second correspondence starts from canonical IR rather than the typed source language.
`execGraphOfIR_semantics_eq` proves that a successful lowering to Lean's executable autograd
`ExecGraphData` preserves denotation for every input, under `NoMSELoss`, `NoRawLog`, and
`NoConcat`.

Both are Lean semantic equalities over an abstract scalar `Context`. They are not statements that a
PyTorch module, CUDA kernel, or vendor library agrees with the graph. General API compilation also
does not inherit the typed-fragment theorem merely because it returns the same IR type.

```
import NN.Verification.TorchLean.Proved
import NN.Runtime.Autograd.Compiled.IRExec.Correctness.SemanticEquivalence

#check NN.Verification.TorchLean.Proved.compileForward_wellFormed
#check NN.Verification.TorchLean.Proved.runForwardIR_eq_evalForward
#check Runtime.Autograd.Compiled.execGraphOfIR_semantics_eq
```

# IBP

Interval bound propagation assigns each node a box. For an affine layer

$$`y=Wx+b,\qquad x\in[\ell,u],`

the usual sign split gives

$$`\ell_y=W^+\ell+W^-u+b,\qquad
u_y=W^+u+W^-\ell+b,`

where $`W^+=\max(W,0)` and $`W^-=\min(W,0)`. Monotone activations transform endpoints; ReLU maps
$`[\ell,u]` to $`[\max(0,\ell),\max(0,u)]`. Elementwise multiplication needs all endpoint products.

The generic real soundness theorem is `cert_encloses_semantics`. It requires:

- `TopoSorted g`;
- `Supported g`;
- exact local certificate consistency `CertLocalOK`;
- exact local value consistency `SemLocalOK`;
- `InputsEnclosed`.

It concludes that each available certificate box encloses the matching semantic value. The current
`Supported` predicate contains input, constant, detach, addition, subtraction, elementwise
multiplication, ReLU, linear, matrix multiplication, tanh, sigmoid, sine, and cosine nodes.

The proof-side real evaluator has the stronger end-to-end theorem
`runIBP?_encloses_evalGraphRec`. It proves that the particular real `runIBP?` construction encloses
`evalGraphRec`, under topological order, supported operations, and enclosed inputs. This is a
theorem about those proof-side definitions; it is not automatically a theorem about every
executable `Graph.runIBP` path.

```
#check NN.MLTheory.CROWN.CertSoundness.cert_encloses_semantics
#check NN.MLTheory.CROWN.Proofs.runIBP?_encloses_evalGraphRec
```

## Train, Compile, Then Bound

The robustness command above begins with fixed parameters so the arithmetic is easy to inspect.
The MLP workflow exercises a longer path: train a `2 -> 100 -> 1` model with the compiled backend,
lower the trained model, and run the maintained IBP implementation over a small input box.

```
lake exe verify -- torchlean-mlp-workflow
```

One seeded run prints:

```
== TorchLean MLP workflow (2 → 100 → 1) ==
Training with backend=Runtime.Autograd.Torch.Backend.compiled, device=cpu
dataset size = 3
mean_loss(before) = 4.751697
mean_loss(after) = 0.834089
Checking IBP bounds on a small input box
IBP nodes=20 output_dim=1 lo=[1.524955] hi=[1.826789]
```

The loss decrease is a runtime observation. The final interval is a bound produced by the
IBP implementation. A theorem about the trained model additionally needs the exact parameter
store used in compilation and a soundness bridge for this executable bound path. Keeping those
claims separate prevents a successful training log from being mistaken for a robustness proof.

# CROWN, Alpha-CROWN, And Alpha-Beta-CROWN

CROWN propagates affine lower and upper forms rather than only boxes. At an uncertain ReLU with
$`\ell<0<u`, the secant upper relaxation is

$$`\operatorname{ReLU}(z)
\le \frac{u}{u-\ell}(z-\ell),`

while a lower relaxation may use a slope $`\alpha` constrained to a valid range. Alpha-CROWN
optimizes these slopes. Alpha-beta-CROWN additionally records branch phases: an active branch uses
$`z\ge 0`, and an inactive branch uses $`z\le 0`.

TorchLean's generic theorem `crown_checker_encloses_semantics` takes an exact
`CrownCertLocalOK` hypothesis and a separate `CrownTransferSound` proof. The local transfer
theorems

- `alphaCrown_transfer_sound`;
- `alphaBetaCrown_transfer_sound`

show that the proposition-level alpha and alpha-beta step functions satisfy
`CrownTransferSound` under their explicit real-semantic and enclosure hypotheses. The alpha-beta
step rejects phase choices inconsistent with the current IBP interval.

```
#check NN.MLTheory.CROWN.Graph.CrownCertSoundness.crown_checker_encloses_semantics
#check NN.MLTheory.CROWN.Graph.AlphaCrownTransferSoundness.alphaCrown_transfer_sound
#check NN.MLTheory.CROWN.Graph.AlphaCrownTransferSoundness.alphaBetaCrown_transfer_sound
```

These theorems should not be confused with the JSON node-certificate checkers:

- `checkCROWNNodeCertificate`;
- `checkAlphaBetaCROWNNodeCertificate`.

Those `IO Bool` functions parse finite decimal fields into `IEEE32Exec`, recompute the complete IBP
trace from trusted inputs and parameters, and replay affine nodes from previously recomputed affine
data. A serialized interval may be wider than the recomputed interval but may not move inward.
Affine replay data must match exactly at the binary32 level. The alpha-beta checker also validates
branch-vector lengths, entries, and phase consistency.

The shared JSON boundary validates input regions before replay begins. Endpoint boxes require
finite arrays of the declared dimension with `lo[i] <= hi[i]`; center-radius boxes additionally
require a finite nonnegative radius. Incomplete or mixed schemas are rejected. Artifact formats
that prescribe endpoints, including the alpha-beta-CROWN leaf format, request that exact schema
rather than accepting the alternate center-radius notation.

The final decisions are `CROWNNodeCert.certificateAccepts` and
`CROWNNodeCertAlphaBeta.AlphaBetaCROWNNodeCertificate.accepts`. Their soundness theorems prove that
acceptance supplies `CrownCertLocalOK` for the exact `IEEE32Exec` replay function used by the
checker.

```
#check NN.Verification.CROWNNodeCert.certificateAccepts_eq_true
#check NN.Verification.CROWNNodeCertAlphaBeta.AlphaBetaCROWNNodeCertificate.accepts_eq_true
```

This closes the structural replay boundary; it does not identify binary32 execution with the
real-valued semantics of `crown_checker_encloses_semantics`. A final real enclosure still requires
the transfer theorem's hypotheses together with a finite-precision refinement argument for the
operations in the accepted graph.

# Directed Arithmetic In The Executable Pass

The graph engine does not assume that every scalar type can safely evaluate every interval rule.
`BoundOps` supplies executable lower and upper addition, subtraction, and multiplication.
`LawfulBoundOps` is the corresponding proof interface: it interprets each endpoint as a real
number and proves that the lower operation is below exact arithmetic and the upper operation is
above it. Sound arithmetic lemmas require this second interface. `NonlinearBoundOps` supplies
executable interval transfers for operations such as division, square root, exponential, logarithm,
and layer normalization. A successful computation is not itself a theorem. The separate
`LawfulNonlinearBoundOps` class proves that every returned interval encloses the corresponding real
operation; sound entrypoints must require this class. A transfer returns `none` when the scalar
backend has no finite implementation for that operation.

`ℝ` and `FP32` have lawful nonlinear instances. The `FP32` proofs use its exact-real floor and
ceiling rounding theorems. `IEEE32Exec` instead states finite-path soundness in its IEEE semantics modules, where
overflow, infinities, and NaNs can be handled explicitly. Its directed binary32 division and square
root are proved, but executable exponential is not yet proved to enclose real exponentiation. An
IBP node using exponential therefore remains unresolved under that instance. Host `Float` widens
basic binary64 arithmetic by one adjacent representable value, but it has no global
`LawfulBoundOps` instance and makes no enclosure claim for the host transcendental library.
Sigmoid, tanh, sine, and cosine can still use their global codomain bounds when a sharper transfer
is unavailable.

When the IBP pass has a valid nonlinear box but no directed affine relaxation, forward CROWN keeps
the box as a constant affine bound. Objective-dependent backward CROWN carries affine coefficients
through algebraic rewrites. Over the reals those rewrites are exact; over rounded execution they
need a separate runtime-approximation theorem. A printed floating-point backward-CROWN result is
therefore computational evidence until that bridge is supplied.

The first-derivative interval pass is seeded. `runDirectionalDerivative` propagates a point or
interval of input directions, so coordinate vectors give partial-derivative bounds without a
second operator traversal. `runFirstDerivative1D` is the scalar-input specialization and rejects a
non-scalar input box. Both entrypoints use the same local transfer function; their supported
operators cannot drift apart. The current second-derivative pass remains one-dimensional.

# From Bounds To A Robustness Claim

Suppose a sound bound procedure produces, for every `x` in the input box,

$$`f_y(x)\ge L_y,\qquad f_j(x)\le U_j.`

Then $`L_y-U_j>0` proves the pairwise class margin. A multiclass certificate repeats this for every
$`j\ne y`. The arithmetic is elementary; the substantive obligations are that:

- the bounds enclose the exact graph semantics;
- the graph denotes the intended model;
- the input box denotes the intended perturbation set;
- any rounded or native execution is related to the exact graph.

For a rounded implementation with coordinate errors
$`\lvert f_i^{\mathrm{run}}(x)-f_i(x)\rvert\le\varepsilon_i`, the transferred margin is

$$`f_y^{run}(x)-f_j^{run}(x)
\ge L_y-U_j-\varepsilon_y-\varepsilon_j.`

The right-hand side must remain positive. A real CROWN theorem alone does not prove the native
binary32 claim; the FP32 and runtime-approximation sections describe the additional bridge.

# Executable Certificates And Imported Artifacts

TorchLean currently checks several kinds of artifact, each with a deliberately limited meaning.

The graph numerical certificate records source ranges, derived node ranges, a registry identity,
and a backend-plan audit. `generateChecked` reconstructs this data, and `executeIEEE32` performs a
bit-level reference replay while checking each intermediate tensor. A `GraphRangeContract`
contains an executable `derive` function but no semantic soundness field. A proof-level error trace
therefore also requires a separately constructed `CheckedRealExecution`, whose fields supply the
real denotation and enclosure proof. See the runtime-approximation section for the complete
boundary.

The external alpha-beta-CROWN leaf checker `checkAbCrownLeafArtifact` validates the JSON schema,
finite ordered root and leaf boxes, dimensions, containment of each represented leaf in the root,
and the exported lower-bound witness relative to its threshold. It does not prove that the lower
bound came from the network semantics, and it does not prove that the represented leaves cover the
root box. Those are producer obligations.

This yields three distinct uses of "certificate":

- a Lean term containing proof fields;
- an artifact accepted by a structural or numerical checker;
- an externally produced claim whose semantic validity is assumed.

The certificate chapter runs the leaf checker, changes a witness so that it must fail, and explains
which stronger artifact would be needed to obtain root-region soundness.

## Other Maintained Checkers

The verifier registry includes several implemented families that exercise different semantic
objects:

- `lirpa-mlp`, `lirpa-cnn`, `lirpa-attention`, `lirpa-gru`, and `lirpa-encoder` replay their
  corresponding finite LiRPA JSON formats. Acceptance is specific to each format and supported
  operator fragment.
- `camera-box3d-cert` checks a camera/3D-box artifact. Its
  [`Box3D`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/Geometry3D/Box3D.lean)
  implementation includes interval-operation soundness lemmas and `checkCert_sound`, connecting a
  successful pure check to the stated projection, positive-depth, and image/bounding-box guards.
- `vnncomp-mnistfc` parses the supported VNN-COMP-style MNIST-FC suite and runs the in-repo bound
  workflow. This is support for that declared suite, not for every VNN-LIB/ONNX benchmark.
- `digits` and `digits-train-certify` run the prepared sklearn-digits robustness paths; one checks
  supplied weights and the other trains before compiling and reporting bounds.
- the three `twostage-*` commands implement the Lyapunov workflows described in the two-stage
  chapter, with distinct external-producer, hybrid, and all-in-Lean boundaries.

`lake exe verify -- list` is the authoritative command inventory. A successful run establishes the
acceptance predicate or reported computation documented for that command; it does not merge these
heterogeneous formats into one global verification theorem.

# Floating-Point Boundary

The proof float `FP32` is `NF binaryRadix fexp32 rnd32`: a rounded-real model with gradual
underflow. Its exponent description has no upper bound, so it does not model overflow, NaN,
infinity, or signed-zero payload behavior. `IEEE32Exec` is the executable bit-level model.

Finite refinement lemmas such as `toReal_add_eq_fp32Round`, and corresponding multiplication,
division, and square-root bridges, connect individual finite IEEE executions to the proof-level
rounding model. Layer and MLP theorems then propagate explicit error budgets. Neither layer is an
unstated theorem about native hardware or a vendor reduction schedule.

The generic IEEE32 CROWN theorem likewise leaves the node evaluator and
`CrownTransferSound` proof to its caller. Choosing an IEEE scalar type does not discharge the
floating refinement obligations.

# Trust Ledger

A verification report should make the following boundary visible:

:::table +header
*
  * Evidence
  * Established in current source
  * Not established by that evidence
*
  * `runForwardIR_eq_evalForward`
  * typed proved-program and IR evaluator agree
  * arbitrary frontend or native runtime agreement
*
  * `runIBP?_encloses_evalGraphRec`
  * proof-side real IBP encloses proof-side real semantics
  * every executable IBP implementation
*
  * `cert_encloses_semantics`
  * enclosure from exact local IBP and semantic hypotheses
  * that a JSON/Float checker supplied those hypotheses
*
  * `alphaCrown_transfer_sound` / `alphaBetaCrown_transfer_sound`
  * exact real local affine transfer soundness
  * approximate artifact-checker acceptance
*
  * CROWN node checker returns `true`
  * artifact intervals contain the authoritative `IEEE32Exec` IBP trace and affine entries exactly
    match a sequential `IEEE32Exec` replay
  * exact-real CROWN soundness
*
  * alpha-beta leaf checker succeeds
  * represented boxes and witness fields pass structural/numeric checks
  * network-bound provenance or root coverage
*
  * numerical range check plus IEEE replay
  * stored trace and one reference execution pass executable checks
  * real semantic enclosure without `CheckedRealExecution`
*
  * FP32 approximation theorem
  * rounded-real output is within its stated budget
  * native IEEE behavior without finite refinement
:::

The ledger is the practical rule for reading the rest of the chapter: tests catch regressions,
checkers reject malformed evidence, contracts state obligations, and theorems prove propositions.
One does not silently substitute for another.

# References

- G. E. Peterson, "Interval Arithmetic," 1969, and IEEE 1788-2015 for interval arithmetic.
- Eric Wong and J. Zico Kolter,
  ["Provable Defenses against Adversarial Examples via the Convex Outer Adversarial Polytope"](https://arxiv.org/abs/1711.00851),
  ICML 2018.
- Huan Zhang et al.,
  ["Efficient Neural Network Robustness Certification with General Activation Functions"](https://arxiv.org/abs/1811.00866),
  NeurIPS 2018.
- Kaidi Xu et al.,
  ["Automatic Perturbation Analysis for Scalable Certified Robustness and Beyond"](https://arxiv.org/abs/2002.12920),
  NeurIPS 2020.
- Shiqi Wang et al.,
  ["Beta-CROWN: Efficient Bound Propagation with Per-neuron Split Constraints for Neural Network Robustness Verification"](https://arxiv.org/abs/2103.06624),
  NeurIPS 2021.
