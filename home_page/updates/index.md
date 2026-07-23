---
title: Updates
usemathjax: true
---

<nav class="timeline-nav" aria-label="TorchLean update timeline">
  <a href="#july-2026-refactor">July 2026 refactor</a>
  <a href="#june-2026-reliability">June 2026 reliability</a>
  <a href="#june-2026-lean-431">Lean 4.31</a>
  <a href="#may-2026-cleanup">Comment cleanup</a>
  <a href="#may-2026-runtime-api">Runtime API</a>
  <a href="#may-2026-cuda-stability">CUDA stability</a>
  <a href="#may-2026-data-note">Data note</a>
  <a href="#may-2026-release">Initial release</a>
</nav>

<div class="updates-timeline">

<article class="update-card" id="july-2026-refactor" markdown="1">
  <div class="update-date">July 2026</div>
  <div class="update-body" markdown="1">

## Lean 4.32 and a Leaner TorchLean

<p class="update-kicker">A smaller API, stronger numerical foundations, and harder runtime tests</p>
<p class="update-summary">
This release began as a cleanup and grew into a fairly thorough pass over the library. We removed
duplicate entry points, made tensor operations less image-specific, brought the floating-point
work into one coherent hierarchy, and tested the training runtime on models large enough to expose
bugs that the small examples never reached.
</p>

### One Public Library

Model code now starts with `import NN`. There is no second convenience namespace sitting in front
of the real library: the old `NN.Library` and `NN.Entrypoint.*` shells are gone, and focused imports
such as `NN.Spec`, `NN.Runtime`, `NN.Floats`, and `NN.Verification` lead directly to the modules that
own those declarations. The model zoo is still part of TorchLean; only its forwarding import modules were
removed.

We also broke up several files that had become difficult to navigate. Training, data handling,
schedulers, CROWN propagation, graph compilation, runtime operations, normalization, Muon, and
floating-point semantics now live in smaller modules with narrower imports. The public API tree is
about 300 lines smaller and the guide is more than 5,000 lines shorter. The proof tree is larger
because the numerical certificates, rounded backpropagation, optimizer contracts, and floating-point
results below are new formal developments rather than forwarding wrappers.

<div class="update-grid">
  <section>
    <h3>General tensors</h3>
    <p>
      A batch is now an axis of a tensor, not a separate kind of object. Generic permutation,
      reduction, reshape, and global-average-pooling operations replace the old public CHW/NCHW
      helpers. We keep layout names only where the operation itself depends on a layout, as
      channel-first batch normalization does.
    </p>
  </section>
  <section>
    <h3>Models</h3>
    <p>
      CNNs, ResNets, ViTs, FNOs, transformers, GPT, Mamba, recurrent models, generative models,
      reinforcement learning, and self-supervised examples all remain available. They now use the
      same tensor and layer API instead of carrying model-specific forwarding stacks.
    </p>
  </section>
  <section>
    <h3>Training</h3>
    <p>
      Optimizers now share one stateful tensor interface. We kept laws that say something useful
      about update rules and stream composition, and removed generated tables and `rfl` theorems
      that only repeated a definition.
    </p>
  </section>
</div>

### Floating-Point Semantics

The floating-point library now has a strict numerical boundary. `import NN.Floats` provides formats,
rounding, finite binary32 semantics, executable IEEE binary32 operations, interval rounders, and
scalar quantization without importing tensors, models, autograd, CUDA, certificate checkers, or
external tools. Tensor and proof integrations live above that boundary, while optional Arb checks
require an explicit import.

The generic development under `NN.Floats.NeuralFloat` is organized by format, rounding, scalar
operations, analysis, error bounds, and execution policy. It covers radix and exponent formats,
directed and nearest rounding, round-to-odd, ULPs and neighboring values, double rounding, Sterbenz
subtraction, and absolute and relative error bounds. Flocq influenced this organization, but this
is a native Lean development rather than a port of the whole Coq library.

Sterbenz subtraction now covers gradual underflow and has a binary32 specialization. Every finite
`IEEE32Exec` bit pattern is proved representable in that specification, so the executable Sterbenz
theorem can identify nearby subtraction with the exact real difference. Finite executable values
also expose a checked ULP exponent, and an absorption theorem connects an unchanged binary32
accumulator to the rounded-real specification.

We use the following distinction throughout TorchLean:

- `NeuralFloat` and `NF` describe configurable rounded-real arithmetic used in proofs;
- `FP32` specializes the rounded-real model to binary32-sized parameters;
- `IEEE32Exec` models executable IEEE-754-style binary32 behavior, including special values;
- runtime bridges state how native values are interpreted by those models.

The effective-rounding example shows the whole argument on one value: choose a format and rounding
mode, perform the rounding, and derive the resulting error bound. The runtime-approximation proofs
then start from the ideal autograd theorems and make every extra hypothesis about rounded execution
explicit.

### Whole-Graph Numerical Certificates

TorchLean can now build a numerical trace over the canonical `NN.IR.Graph` rather than stopping at
isolated scalar lemmas. Source intervals use exact binary32 endpoints. The checker reconstructs
outward-rounded ranges for supported arithmetic, activations, directed square root, reductions,
matrix multiplication, pooling, MSE, and stable softmax; malformed domains and non-finite ranges
fail at the node that produced them.

Range propagation is now an operation registry rather than one large match over model cases. The
same traversal handles any architecture after lowering. Before propagation, a coverage pass lists
the exact nodes whose primitives lack a range contract. Custom registries are named and the name is
stored in the certificate, so an artifact cannot be replayed under a different set of rules.

The same certificate contains the backend planning audit. Rounding mode, subnormal behavior,
FMA/contraction, and reduction order are recorded by each kernel capsule. Portable accumulations
use the fixed left fold from the tensor semantics. CUDA and LibTorch accumulations are marked
implementation-dependent, so their matrix products, convolutions, normalizations, FFT/FNO paths,
scans, and attention kernels cannot accidentally inherit a proof for a different reduction order.

The bit-level replay evaluates every graph intermediate with `IEEE32Exec`, checks its shape and
range, and rejects NaN or infinity. A checked certificate now stores the exact graph it was checked
against, so replay cannot substitute a different graph. A separate proved real execution supplies
the semantic enclosure; combining it with the bit-level replay yields an entrywise error trace for
every node. The deep-dive example includes successful arithmetic, reduction, matmul, LayerNorm,
`abs -> sqrt`, and softmax traces, together with deliberately tampered, invalid-domain, and
wrong-reduction-policy cases. It now ends with a complete two-layer MLP: ten graph nodes pass
coverage, range generation, backend-capsule audit, and bit-level replay. The
[numerical-runtime walkthrough]({{ '/examples/numerical-runtime/' | relative_url }}) follows that
run from source enclosures to its checked output.

### Rounded Backpropagation and Optimizers

The numerical proof now continues past the forward graph. Proof-bearing reverse nodes carry both
their ideal VJP and their rounded VJP error transformer. The global reverse theorem composes those
local bounds through gradient accumulation and connects the result to executable autograd
`GraphData`.

One optimizer contract carries the gradient error through parameter updates. SGD and momentum SGD
have no extra domain condition. AdamW uses the same interface, with step data recording errors from
both moments, bias correction, square root, adaptive division, decoupled weight decay, and the final
subtraction; explicit margins keep the rounded denominator away from zero. The end-to-end theorem
therefore works unchanged for all three optimizers and for every model represented by a `RevGraph`.
A model-wide update applies it at each typed parameter index.

The canonical `NN.IR.Graph` certificate remains a forward certificate. Its current compiler does not
attach proved VJPs, so backward claims use the proof-bearing reverse graph path instead of silently
attributing autograd semantics to a forward-only lowering.

### Tensor Quantization

Uniform affine quantization has one scalar definition under `NN.Floats.Quantization` and one
rank-polymorphic tensor adapter under `NN.Spec.Quantization`. The proofs cover code-range
preservation, monotonicity, exact dequantize/quantize round trips for in-range integer tensors, and
the half-step reconstruction bound when saturation is inactive. Layout and storage width are not
part of the arithmetic: int8, uint8, int4, and custom code sets differ through their integer bounds
rather than separate image-specific APIs.

### Backend Contracts

A user should be able to choose where a model runs without learning a pile of unrelated switches.
The backend API therefore separates three questions. `Device` says where the work runs, `Provider`
says whose implementation performs it, and `BackendOp` names the operation being requested. For
each available implementation, a kernel capsule records its shape and layout requirements, whether
it supplies forward and backward computation, and what evidence supports its numerical contract.

The backend catalog is modular as well. Attention, native CUDA, portable reference code, and
optional LibTorch providers contribute named capsule modules. A downstream provider can extend a
profile with another module; model definitions do not change, because they request operations rather
than provider-specific kernels.

Capability names are now rank-polymorphic operation families. Convolution, pooling, reduction,
permutation, slicing, gathering, and matrix multiplication each have one backend capability; rank,
axes, padding, strides, and index tensors remain in the graph payload. Numerical certificates use
their own transfer keys where two payloads need different interval rules, so provider discovery no
longer doubles as a mathematical semantics table.

Named profiles bundle choices that must agree. The checked CPU and native CUDA profiles keep the
TorchLean tape and backward rules. The LibTorch-forward profile prefers its registered attention
kernel, falls back to native CUDA for other operations, records the same TorchLean tape node, and
uses TorchLean's local VJP. At present, the LibTorch bridge covers scaled dot-product attention; it
is not a general PyTorch dispatcher.

TorchLean runs on macOS CPU and on Linux CPU or native CUDA. For Windows, WSL2 is currently the
documented route. The type system already has room for Metal, ROCm, WebGPU, TPU, Trainium, native
Windows, and custom accelerators, but naming a target is not the same as implementing it. If a
requested provider is unavailable, TorchLean reports that fact instead of silently falling back.

### Mathematical and Verification Corrections

<div class="update-grid">
  <section>
    <h3>Losses and masks</h3>
    <p>
      Huber loss and Smooth L1 now have their intended, distinct scaling. Hard attention masks use
      exact exclusion in the softmax semantics: a blocked entry contributes zero numerator. The old
      finite <code>-1000</code> masking convention was removed from attention paths and examples.
    </p>
  </section>
  <section>
    <h3>Bounds</h3>
    <p>
      Leaky-ReLU interval propagation handles a negative slope across the kink at zero. Logarithm
      interval checks reject nonpositive domains. Unsound or undocumented GELU and ELU candidate
      bounds were removed instead of being exposed under names that suggested certified enclosure.
    </p>
  </section>
  <section>
    <h3>Certificates</h3>
    <p>
      JSON certificate readers reject non-finite claims before array comparisons. IBP certificates
      are checked by recomputing the complete <code>IEEE32Exec</code> trace from the trusted graph,
      parameters, and input box; an artifact may widen that trace but may not shrink it. CROWN and
      α,β-CROWN affine entries are compared exactly with a sequential replay instead of being
      propagated from certificate-supplied parents.
    </p>
  </section>
  <section>
    <h3>Classical models</h3>
    <p>
      HMM normalization records zero probability for an impossible observation, and log-likelihood
      is partial at that boundary. GMM covariance matrices must be symmetric positive definite,
      mixture weights must be positive and normalized, and singular inversion now fails rather
      than returning the identity. The covariance gradients use the transpose-correct formulas.
    </p>
  </section>
  <section>
    <h3>Attention and diffusion</h3>
    <p>
      Multi-head attention now reshapes sequence data to
      <code>(sequence, heads, head-dimension)</code> before exchanging the sequence and head axes.
      The probability-flow ODE uses the required one-half score coefficient, and its Euler sampler
      visits time points in descending order from the noisy endpoint.
    </p>
  </section>
  <section>
    <h3>Layer and model edges</h3>
    <p>
      Dropout is the identity in evaluation mode. Max pooling rejects padding configurations that
      would create windows containing no input values. PCA requires at least two samples for its
      unbiased covariance and exports the centering term as a linear bias. Linear SVM fitting calls
      its regularization coefficient <code>lambda</code>, leaving <code>C</code> for the standard
      inverse-strength convention.
    </p>
  </section>
</div>

The graph evaluator and verifier compiler are now split by operation. Their coverage theorems still
range over the full operation vocabulary, so adding a new file does not weaken the statement being
proved. We removed theorems tied to incidental list lengths and retained small definitional lemmas
only when later correctness proofs actually use them.

### Lean 4.32

TorchLean now builds with Lean, mathlib, DocGen, and Verso 4.32. During the upgrade we replaced the
deprecated `Lean.RBMap` with `Std.TreeMap`, used mathlib's stronger sine remainder estimate, and made
several dependent casts explicit. Proof-valued runtime helpers are theorems when they serve as
opaque evidence; constructors that must compute remain reducible `abbrev`s. We fixed the new
linters rather than suppressing them.

### Runtime Scaling and CUDA Ownership

The small examples had hidden an expensive habit: large parameters were first expanded into nested
Lean values and only then copied into the execution engine. Parameters and gradients are now
materialized directly where they will run. We also stopped generic convolution backward from
rebuilding the same derivative structure, and taught CUDA attention and fused FNO paths to release
temporary buffers as soon as their contribution is consumed.

The most useful failure came from sparse reverse mode. A pure expression allocating a one-element
CUDA seed could be shared by Lean, even though backward consumed and released the native buffer.
The next use then referred to storage that was no longer alive. Seeds that cross an ownership
boundary now come from an effectful constructor, and transfers between gradient maps use explicit
copy-and-release operations. A stress test repeats this path and fails if live CUDA allocation
grows or a supposedly fresh seed is reused.

A second lifetime problem was in the FFI signatures themselves. Buffer and array inputs were being
passed as owned Lean objects to native functions that treated them as borrowed, so neither side
released the wrapper reference. The declarations now mark those inputs as borrowed. Separate
payload and wrapper counters make the distinction visible, and the stress suite checks thousands
of allocations for matching finalization counts.

We exercised 21 CPU workflows and 24 CUDA workflows, including dense, convolutional, attention,
recurrent, operator-learning, generative, and reinforcement-learning models. On the machine used
for this release, a roughly 100-million-parameter MLP completed ten CUDA optimizer steps in about
15.2 seconds. The fused Burgers FNO ran for 100 steps with no growth in live buffers; training MSE
fell from 0.3260 to 0.0172 and test MSE ended at 0.0220. These numbers record what we tested on one
machine. They are not a general performance promise.

### Documentation and Validation

The Guide and API reference now follow the new module layout. Installation has separate notes for
Linux, macOS, WSL2, native Windows, CUDA, and optional LibTorch support, and the floating-point and
backend chapters explain where a theorem ends and a runtime assumption begins. Repository checks
now build `NN` directly rather than passing through the deleted `NN.Library` shell.

<div class="validation-list" markdown="1">
  <h3>Validation</h3>

- `lake lint`
- `lake build`
- `lake build NN NN.CI.All`
- `lake exe nn_tests_suite`
- `lake -R -K cuda=true exe nn_tests_suite`
- `scripts/checks/example_regression.sh` across 42 registered commands and examples
- `scripts/checks/example_regression.sh --cuda --extended-cuda --skip-help --skip-default`
- sustained 20-update CPU runs across 21 model workflows
- sustained 100-update CUDA runs across 24 model workflows
- repeated sparse-backward ownership and allocator-drift regression
- external-wrapper allocation/finalization regression on both the CUDA and CPU-stub builds
- NVIDIA Compute Sanitizer memcheck (`ERROR SUMMARY: 0 errors`)
- DocGen API generation
- Verso Guide generation
- dependency audit and interactive import-graph generation
- Jekyll production build
- `git diff --check`

All of these checks passed on the Linux machine used for the release. That gives us evidence for the
paths we exercised, but it does not turn CUDA machine code or LibTorch into Lean proofs. Their trust
levels remain explicit in the backend contracts and in `TRUST_BOUNDARIES.md`.
</div>

  </div>
</article>

<article class="update-card" id="june-2026-reliability" markdown="1">
  <div class="update-date">June 2026</div>
  <div class="update-body" markdown="1">

## CUDA, CROWN, and PINN Reliability

<p class="update-kicker">Native runtime, certificates, scientific ML</p>
<p class="update-summary">
The June reliability pass tightened CUDA allocation checks, made CROWN certificate statements more
direct, and moved duplicated PINN training code into shared helpers.
</p>

<div class="update-grid">
  <section>
    <h3>Runtime checks</h3>
    <p>
      CUDA and CPU stubs now use shared checked size arithmetic for products,
      byte counts, and additions at the Lean FFI boundary. Broadcast, reduction,
      swap, gather/scatter, attention, convolution/pooling, tensor-copy, and
      spectral-convolution paths reject impossible sizes before allocation or
      kernel launch.
    </p>
  </section>
  <section>
    <h3>Proof API</h3>
    <p>
      The graph CROWN certificate theorem now returns the enclosure for the
      node being checked. The IEEE32 version records the no-self-dependency
      condition on the evaluator trace, and the two-layer MLP CROWN code exposes
      the affine forms used by <code>boundAffineCrown</code>.
    </p>
  </section>
  <section>
    <h3>PINNs</h3>
    <p>
      Python PINN trainers now share dataset loading, MLP construction,
      expression evaluation, gradients, constant parsing, and export helpers
      through <code>scripts/verification/pinn/pinn_common.py</code>.
    </p>
  </section>
</div>

The focused API import now exposes the short `TorchLean.*` namespaces directly.
With `import NN.API`, users get `TorchLean.nn`, `TorchLean.optim`,
`TorchLean.Trainer`, `TorchLean.Data`, `TorchLean.Loss`, and
`TorchLean.Metrics` without importing the broader `NN` umbrella.

<div class="validation-list" markdown="1">
  <h3>Validation</h3>

- `lake test`
- `lake build NN.CI.All`
- `lake lint -R -K cuda=true -K cuda_home=/usr/local/cuda-13.0`
- `scripts/checks/check.sh --cuda --cuda-home /usr/local/cuda-13.0`
- `scripts/checks/cuda_sanitize_tests.sh --cuda-home /usr/local/cuda-13.0 --all-tools`
- focused Lean checks for the CROWN MLP and graph CROWN certificate modules
- PyTorch CUDA regression runs for the PINN trainers on an A100 GPU

CUDA sanitizer reported zero memcheck/initcheck/synccheck errors and no
racecheck hazards on the exercised runtime suite.
</div>

  </div>
</article>

<article class="update-card" id="june-2026-lean-431" markdown="1">
  <div class="update-date">June 2026</div>
  <div class="update-body" markdown="1">

## Lean 4.31 Migration

<p class="update-kicker">Toolchain alignment</p>
<p class="update-summary">
TorchLean now builds with <code>leanprover/lean4:v4.31.0</code>. The root Lake
manifest, Mathlib pin, documentation generator pin, Verso blueprint toolchain,
website metadata, README, and formalization metadata were moved together.
</p>

The migration fixed proof-term breakages in differentiability and autograd
composition files where Lean 4.31 became stricter about composed functions and
eventual equality. The full repository build was rerun on the new toolchain.

  </div>
</article>

<article class="update-card" id="may-2026-cleanup" markdown="1">
  <div class="update-date">May 2026</div>
  <div class="update-body" markdown="1">

## Repository Modularization and Comment Cleanup

<p class="update-kicker">Source organization</p>
<p class="update-summary">
The public source tree is easier to read, easier to review, and easier to extend without changing
TorchLean's intended behavior.
</p>

<div class="update-grid">
  <section>
    <h3>Structure</h3>
    <ul>
      <li>Large proof and runtime files were split along conceptual boundaries.</li>
      <li>Umbrella modules were kept only where they make imports clearer.</li>
      <li>Obsolete import shells and example names were removed.</li>
    </ul>
  </section>
  <section>
    <h3>Documentation</h3>
    <p>
      Comments were rewritten in a more mathlib-style voice: definitions state
      mathematical intent, runtime boundaries name assumptions, and examples
      avoid stale narration.
    </p>
  </section>
  <section>
    <h3>Scope</h3>
    <p>
      Model semantics, verification claims, CUDA behavior, and trusted boundaries are unchanged.
    </p>
  </section>
</div>

The examples and website pages were rebuilt against the new module layout.

  </div>
</article>

<article class="update-card" id="may-2026-runtime-api" markdown="1">
  <div class="update-date">May 2026</div>
  <div class="update-body" markdown="1">

## Runtime API Update

<p class="update-kicker">Training loops, streams, initialization</p>
<p class="update-summary">
Longer examples now use the same public runtime API for initialization, minibatches, optimizer
steps, logging, and checkpoint-style parameter files.
</p>

<div class="update-grid">
  <section>
    <h3>Initialization</h3>
    <p>
      Runtime-side Float initializers and shape-indexed initializer plans give parameter bundles a
      clear construction story: the model declares the parameter shape, the initializer plan selects
      the distribution or constant, and the runtime builds each tensor once.
    </p>
  </section>
  <section>
    <h3>Streams</h3>
    <p>
      Step-indexed training streams make minibatches explicit. A batch may come from a rule, a
      simulator, a replay buffer, or a file-backed window source, but the training loop sees a typed
      stream of inputs, targets, and metadata.
    </p>
  </section>
  <section>
    <h3>Language models</h3>
    <p>
      Integer-token embedding and row-wise cross-entropy helpers let GPT-style examples train on
      token ids directly, instead of expanding every target into a one-hot vector.
    </p>
  </section>
</div>

The runtime documentation now follows the same path as the examples: initialize parameters, produce
batches, run forward/autograd, update parameters, save reports, and state the native or external
boundary when a backend is selected.

  </div>
</article>

<article class="update-card" id="may-2026-cuda-stability" markdown="1">
  <div class="update-date">May 2026</div>
  <div class="update-body" markdown="1">

## CUDA Training Stability

<p class="update-kicker">Memory lifetime, allocator diagnostics</p>
<p class="update-summary">
Longer CUDA training runs exposed allocator pressure from intermediate values
that could stay attached to a run longer than needed.
</p>

The issue was not model size. Some intermediate values created during training
-- tape entries, gradient buffers, and kernel workspace buffers -- could remain
alive across many optimizer steps. The fix made the training loop and CUDA eager
backend explicit about which values are returned to the caller and which buffers
can be released after the step finishes.

<div class="update-grid">
  <section>
    <h3>Step counts</h3>
    <p>Loader-based model commands now treat <code>--steps</code> as optimizer updates.</p>
  </section>
  <section>
    <h3>Buffer lifetime</h3>
    <p>CUDA eager/autograd releases ephemeral tape, gradient, and workspace buffers after each step.</p>
  </section>
  <section>
    <h3>Diagnostics</h3>
    <p>Longer CUDA examples report allocator telemetry by default, with <code>--cuda-mem-watch N</code> for exact sampling.</p>
  </section>
</div>

<div class="validation-list" markdown="1">
  <h3>Validation</h3>

- `lake build`
- `lake -R -K cuda=true build`
- `lake exe torchlean mlp --device cpu --steps 10 --log false`
- `lake exe torchlean mlp --steps 10 --dtype float --backend eager --log false`
- `lake -R -K cuda=true exe torchlean mlp --device cuda --steps 1000 --log false`
- `lake -R -K cuda=true exe torchlean cnn --device cuda --steps 1000 --log false`
- `lake -R -K cuda=true exe torchlean gpt2 --device cuda --steps 1200 --generate 0 --log false`
- `lake -R -K cuda=true exe torchlean fno1d_burgers --device cuda --steps 50 --log false`

The validation checked representative losses or MSE values going down and the
CUDA allocator staying bounded on the exercised runs.
</div>

  </div>
</article>

<article class="update-card" id="may-2026-data-note" markdown="1">
  <div class="update-date">May 2026</div>
  <div class="update-body" markdown="1">

## Introductory Data Note

<p class="update-kicker">Reproducible builds</p>
<p class="update-summary">
Some examples use public datasets and do not download them during
<code>lake build</code>. Keeping data downloads explicit makes ordinary builds
deterministic and avoids silently committing large external data.
</p>

For the model-zoo MLP example:

```bash
python3 scripts/datasets/download_example_data.py --auto-mpg
```

For the text and vision examples:

```bash
python3 scripts/datasets/download_example_data.py --tiny-shakespeare --cifar10
```

  </div>
</article>

<article class="update-card" id="may-2026-release" markdown="1">
  <div class="update-date">May 2026</div>
  <div class="update-body" markdown="1">

## TorchLean Released

<p class="update-kicker">Initial public release</p>
<p class="update-summary">
TorchLean became public as a Lean 4 framework for writing, running, inspecting,
and verifying neural-network programs.
</p>

<div class="update-grid">
  <section>
    <h3>Core system</h3>
    <p>
      Typed tensors, layers, model APIs, loaders, training loops, examples, and
      a shared graph IR for execution, inspection, verification, and
      import/export.
    </p>
  </section>
  <section>
    <h3>Semantics</h3>
    <p>
      Finite-precision semantics, executable IEEE-style Float32 models,
      autograd, optimizer support, and explicit runtime agreement boundaries.
    </p>
  </section>
  <section>
    <h3>Verification</h3>
    <p>
      IBP/CROWN-style bounds, certificate checkers, VNN-COMP-style bundles, Bug
      Zoo contracts, 3D geometry certificates, and optional CUDA/native runtime
      paths with documented trust boundaries.
    </p>
  </section>
</div>

The README example is the shortest entry point; the Guide and Examples pages carry the longer
walkthroughs.

  </div>
</article>

</div>
