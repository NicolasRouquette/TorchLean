import VersoManual

open Verso.Genre Manual

#doc (Manual) "Where The Pieces Meet" =>
%%%
tag := "conclusion"
%%%

We began with a two-input regression model small enough to fit on one line:

$$`
F_\theta(x)
=
W_2\operatorname{ReLU}(W_1x+b_1)+b_2.
`

At first it was tempting to call that one object “the model.” By the end of the guide, the formula
had acquired a shape-checked declaration, seeded parameters, an autograd tape, a lowered operation
graph, several scalar interpretations, a kernel plan, and verification evidence. A training run
gave us values; a theorem spoke about a mathematical object; a certificate connected a particular
checker result to a proposition.

None of those views replaces the others. The useful part of TorchLean is the trail between them:

```
typed model
  -> initialized parameters
  -> executable graph and derivatives
  -> explicit scalar and backend semantics
  -> verifier input
  -> checked proposition
```

When that trail is intact, a surprising result has somewhere concrete to investigate. When a link
is missing, saying that “the model is verified” only hides the missing work.

# Reproduce The Path

If you want to feel that trail rather than merely read about it, run these commands in order:

```
lake exe torchlean quickstart_tensors
lake exe torchlean quickstart_autograd
lake exe torchlean quickstart_mlp \
  --device cpu --steps 200 --seed 2026
lake exe torchlean graphspec --device cpu --execution eager
lake exe torchlean one_semantic_universe
lake exe torchlean float32_semantics
lake exe torchlean numerical_certificate
```

The first three stay close to concrete tensors: print them, differentiate a linear map, and train a
small network. `graphspec` and `one_semantic_universe` then show why lowering matters: the same
operation graph can be evaluated for values or interpreted for bounds. `float32_semantics` separates
host arithmetic from executable binary32 semantics, and `numerical_certificate` finishes with a
graph-level checker that also rejects deliberately malformed evidence.

A successful line of output is therefore a waypoint, not the end of the argument. It tells us which
artifact exists and which question we can ask next.

# Validate The Layer You Depend On

The small demonstrations above explain individual ideas. Before relying on a larger change, use the
validation command that reaches the relevant boundary:

```
# Compile and run the curated suite against the CPU CUDA stub.
lake build nn_tests_suite
lake exe nn_tests_suite

# Compile the native CUDA implementation, then execute it on an actual device.
lake -R -K cuda=true build nn_tests_suite
CUDA_VISIBLE_DEVICES=0 lake env ./.lake/build/bin/nn_tests_suite

# Replay the default checked-artifact suite.
lake exe verify -- all

# Check repository conventions and rebuild the complete documentation site.
lake lint
scripts/docs/build_site.sh
```

A CUDA-enabled build proves that the native sources compile and link. It does not prove that a GPU
kernel ran; that requires the second command and a visible CUDA device. Likewise, a theorem build
checks Lean declarations, while certificate replay exercises parsers, policy gates, and the stored
artifacts that instantiate those declarations. Documentation is part of validation
because examples, import paths, and capability claims can become stale even when source code still
compiles.

# A Claim, Written Carefully

Suppose a colleague sends you a result that says:

> The trained model is robust on a box of inputs.

That sentence is a good beginning. To make it reusable, sit down with the artifact and fill in the
missing nouns:

1. *Which model?* Identify architecture, parameter artifact, and graph payload.
2. *Which input box?* Give lower and upper tensors with checked shapes.
3. *Which robustness property?* State the output inequality or class-margin condition.
4. *Which scalar semantics?* Exact real, finite FP32, executable IEEE32, or a native runtime.
5. *Which method?* IBP, CROWN, α,β-CROWN artifact replay, branch-and-bound, or another checker.
6. *Which theorem?* Name the proposition obtained when the checker accepts.
7. *Which boundary remains trusted?* External search, parser, backend kernel, compiler, or hardware.

The result might then read: “checker `C` accepted parameter artifact `P`, lowered to graph `G`, and
established class margin `Q` for the box $`[\mathrm{lo},\mathrm{hi}]` under the graph's declared scalar semantics;
artifact generation and the named native providers remain outside the checked implication.” The
actual names matter more than this template. With them recorded, replacing the parameters, box, or
backend reveals exactly which part of the argument must be repeated.

# What Is Checked Or Proved In Lean

Much of the argument can live entirely in Lean. Shapes and layer composition, mathematical operator
specifications, graph well-formedness, autograd rules, generic rounding, finite FP32 mathematics,
executable IEEE32 reference algorithms, optimizer laws, and checker soundness are all stated as
named definitions or theorems with explicit hypotheses.

The exact scope still comes from each declaration. A theorem for a supported graph fragment does
not become a theorem about every model because it appears in a broad chapter. In particular, real
training crosses boundaries that the Lean kernel cannot inspect directly:

:::table +header
*
  * Lean-side object
  * Boundary needed to connect it to a run
*
  * typed tensor or model
  * dataset loader, parameter artifact, and preprocessing
*
  * operation graph semantics
  * model lowering, importer, or compiler
*
  * `IEEE32Exec` reference operation
  * CPU/CUDA kernel, compiler, driver, and hardware conformance
*
  * spectral or matrix operation contract
  * cuFFT, cuBLAS, LibTorch, or another selected provider
*
  * checker acceptance theorem
  * artifact producer and any external search that proposed the evidence
*
  * PINN or neural-operator proposition
  * equation encoding, simulator, sampling, and dataset provenance
:::

Kernel capsules record provider, device, shape/layout contracts, numerical policy, derivative
implementation responsibility, and evidence. Artifact parsers and policy gates reject malformed or
inadmissible input. Those mechanisms make the remaining trust visible; wrapping a native library or
Python script in a Lean function would not make it proved.

# Numerical Guarantees

The floating-point stack has three central levels:

```
NeuralFloat / NF
  generic radix, format, and rounding mathematics

FP32
  binary32-precision, gradual-underflow rounded-real semantics with no upper exponent cutoff

IEEE32Exec
  executable 32-bit IEEE representation and operations
```

The runtime then adds CPU, CUDA, or external providers. A real-valued approximation theorem, a
half-ULP rounding theorem, an executable bit-pattern theorem, and a CUDA parity test are different
evidence.

This separation makes useful compositions possible. For example:

$$`
|F_{\mathrm{runtime}}(x)-f(x)|
\leq
|F_{\mathrm{runtime}}(x)-F_{\mathbb R}(x)|
+
|F_{\mathbb R}(x)-f(x)|.
`

The first term is numerical implementation error. The second is model approximation error. A
meaningful end-to-end result needs both, with compatible domains and semantics.

# Configuration Fails Closed

Malformed numerical configurations are rejected before they enter a graph or checker. Checked
format constructors, operation-specific runtime guards, native-buffer validation, and certificate
parsers all apply this principle at their own boundary. These checks do not prove an external
implementation; they keep later theorems and tests from silently operating on a different problem.

# Scaling Through Backend Contracts

TorchLean is not meant to replace every tuned numeric kernel with a slow Lean implementation.
Large models need industrial matrix multiplication, convolution, attention, FFT, and communication
libraries.

The architectural goal is:

```
one semantic operation graph
  -> several admissible kernel providers
  -> explicit contract and evidence per boundary
```

TorchLean records graph structure, shapes, loss, optimizer meaning, and proof statements while a
provider supplies a fast value or local VJP. The assurance level may range from a proved internal
implementation to a checked or explicitly trusted external kernel.

Scaling and verification therefore meet at the backend contract, not by pretending that outsourced
numerics were executed inside the theorem prover.

# Scientific ML

A PINN or neural operator usually participates in a larger chain:

```
equation and domain
  -> discretization or simulator
  -> dataset
  -> model and training
  -> prediction artifact
  -> residual, invariant, or error certificate
  -> Lean checker and theorem
```

The neural network is only one part. Boundary conditions, quadrature, sampling coverage, simulator
accuracy, and interpolation between grid points can dominate the final claim.

TorchLean's role is to give each artifact a typed meaning and to make the accepted implication
precise. External computation can remain large; the checker and proposition should remain small
enough to audit.

# A Productive Development Loop

When adding an operation, begin where its meaning is simplest: write the tensor shape and scalar
semantics before choosing a fast kernel. Once that object is stable, teach the runtime its forward
and derivative behavior, and teach the IR how to represent it if export or verification needs the
operation.

Only then add providers. Each implemented runtime needs the layout and numerical policy it actually
uses, together with an honest evidence level. Prove the reusable semantic facts, but also add a
small executable example with a negative control: a bad shape, nonfinite parameter, unsupported
policy, or malformed artifact should fail for the reason the documentation promises.

Finally, run the same small model through every path you claim. This last step catches the gaps that
are easy to miss when a semantic definition, derivative rule, lowering case, and CUDA dispatch are
reviewed in isolation.

Tests and proofs are complementary. Proofs establish universal propositions about formal objects.
Tests catch wiring, FFI, build, CLI, documentation, and platform regressions that are outside or not
yet covered by those propositions.

# The Seams Still Visible

The guide also leaves several gaps in plain sight. Local interval transfers are not yet one
universal graph-wide exact-real enclosure theorem. Reverse lowering and optimizer-error propagation
cover selected paths rather than every training graph. Native and external kernels usually carry a
recorded contract and tests, not a machine-checked implementation proof. Quantization theory is
farther along than packed-kernel conformance.

Mixed precision remains the concrete graph-design seam described in *TorchLean And PyTorch*; the
current runtime follows the homogeneous-scalar contract from *Tensors And Shapes*.

The same honesty applies to devices. A target name in a parser or registry is not a runtime. A new
device becomes meaningful when it has real provider profiles, capsules, execution, and validation.
The useful measure of progress is not the length of the feature list, but whether a runnable path
and a precise claim are closer without a new boundary being hidden.

# A Final Exercise

Choose one example from the model zoo and write down:

```
input and output shapes
parameter shapes
loss
data source and preprocessing
scalar semantics
execution mode
device and selected providers
forward and backward implementation responsibility
available theorem or checker
remaining trusted assumptions
```

Then run it with `--show-backend` and compare the report with your list. Any missing item is a
concrete documentation, logging, or verification task.

That habit is the central lesson of the guide: do not ask whether “the model” is verified as if it
were one indivisible thing. Ask which object carries the claim, which transformation produced it,
and which theorem or boundary connects it to what ran.

# References

- George et al., [*TorchLean: Formalizing Neural Networks in Lean*](https://arxiv.org/abs/2602.22631),
  2026.
- Boldo and Melquiond,
  [*Flocq: A Unified Library for Proving Floating-Point Algorithms in
  Coq*](https://doi.org/10.1109/ARITH.2011.40), 2011.
- George C. Necula, [*Proof-Carrying Code*](https://doi.org/10.1145/263699.263712), 1997.
- Odena et al., [*TensorFuzz*](https://proceedings.mlr.press/v97/odena19a.html), 2019.
- Liu et al., [*NNSmith*](https://arxiv.org/abs/2207.13066), 2022/2023.
