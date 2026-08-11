import VersoManual

open Verso.Genre Manual

#doc (Manual) "Inside The Backend Planner" =>
%%%
tag := "backend-selection"
%%%

The previous page selected CPU, CUDA, or an optional provider through the runtime API. A less visible
question remains: when a graph asks for matrix multiplication, attention, or a
reduction, how does TorchLean decide which implementation is allowed to answer?

A device name is not enough. One CUDA build may contain a hand-written kernel, a cuBLAS call, and a
LibTorch bridge for different operations. Their layouts, numerical behavior, backward support, and
supporting evidence differ. The backend planner keeps those differences in data and either returns
an accepted plan or explains why it could not make one.

The path is:

$$`
\text{operation}+\text{profile}+\text{available providers}
\longrightarrow \text{capsule}
\longrightarrow \text{audit}
\longrightarrow \text{accepted kernel}
\longrightarrow \text{typed handler}.
`

The sections below trace that path from an operation request to the handler that executes it.

# Kernel Capsules

Suppose a graph reaches scaled dot-product attention. TorchLean currently knows three maintained
ways to compute it: a composed TorchLean expression, a native fused CUDA implementation, and a
LibTorch forward bridge with a TorchLean-owned backward pass. The operation is the same; the
implementation contract is not.

A `KernelCapsule` records those differences:

```
structure KernelCapsule where
  name : String
  op : BackendOp
  provider : Provider
  device : Device
  trustLevel : TrustLevel
  supportsForward : Bool
  vjpMode : VJPMode
  shapeContract : ContractDescriptor
  layoutContract : ContractDescriptor
  valueContract : ContractDescriptor
  vjpContract : ContractDescriptor
  numericalPolicy : NumericalPolicy
  notes : String
```

Capsules are declared before the run and registered with the backend. The planner may select one
only when its device, provider, gradient mode, and assurance level fit the requested profile. If no
capsule fits, planning stops with an error.

Capsules are collected in named `CapsuleModule`s. Built-in attention, native CUDA, portable
reference, and optional LibTorch code contribute modules to the same registry. A downstream
provider can prepend another module with `BackendProfile.withCapsuleModules`; it does not add a new
model class or a branch to the graph walker. The model still lowers to ordinary `BackendOp`s, and
the planner either finds an admissible capsule for each operation or reports the missing operation.
Adding a module with an existing name replaces that module. Planning rejects repeated module names
and repeated capsule identities, so provider precedence cannot change through accidental duplicate
registration.

`BackendOp` names semantic operation families such as matrix multiplication, reduction, pooling, or
convolution. Rank, axes, padding, strides, and index tensors remain in the graph payload. This keeps
capability discovery general without erasing the information needed to state the operation
correctly.

```
#check NN.Backend.Registry.CapsuleModule
#check NN.Backend.BackendProfile.withCapsuleModules
```

The complete installation and platform guide includes a
[worked capsule example](https://lean-dojo.github.io/TorchLean/installation/#what-a-kernel-capsule-looks-like).

# Looking Inside A Capsule

Each capsule has four obligations: shape, layout, forward value, and VJP. The current registry
records runtime guards, regression suites, fuzz oracles, and explicit trusted boundaries. Source
references identify the implementation being discussed; they are provenance, not evidence of
correctness.

The obligation itself is a structured `ContractClaim`: shape safety, compatibility with a named
tensor layout, refinement of a forward specification, or refinement of a VJP specification. The
free-form note is there for readers; the planner works with the structured claim.

Before planning, TorchLean checks that each descriptor has the right obligation kind and operation,
and that a VJP descriptor names the capsule's declared VJP mode. Thus a value test placed in the
shape field is rejected rather than counted as shape evidence. A forward-only capsule uses a
separate `vjpUnavailable` claim; it cannot describe `.none` as though it were a VJP refinement.

Each evidence kind records a different claim:

- *runtime-guard evidence* records validation performed at an execution boundary;
- *test evidence* records a regression or differential test suite;
- *fuzz evidence* records sampled differential testing, not a universal statement;
- *trusted external evidence* names code whose correctness is assumed for the claim;
- *not applicable* records an obligation that the capsule intentionally does not provide;
- *missing evidence* prevents acceptance under the normal strict policies.

Numerical policy is recorded separately from evidence. It states which rounding mode, subnormal
behavior, multiply-add contraction, and reduction order the implementation uses. For example, the
portable matrix-product capsule records the fixed left fold used by the tensor semantics, whereas
the CUDA capsule records an implementation-dependent reduction. A range proof for one order cannot
therefore be reused for the other merely because both capsules implement `matmul`.

Source paths and native symbols identify the code behind a capsule. The maintained profiles accept
the guards and tests appropriate to their runtime paths. The stricter `verified` policy uses a
different entrypoint: `planVerifiedKernel` selects a `ProofCarryingKernel` whose type contains both
the implementation and its pointwise refinement theorem. Erasing that object to ordinary capsule
metadata also erases the proof, so the metadata planner rejects a capsule merely labelled
`verified`. No maintained production kernel is registered through the proof-carrying entrypoint yet.

```
#check NN.Backend.ProofCarryingKernel
#check NN.Backend.planVerifiedKernel
#check NN.Backend.VerifiedPlannedKernel.run_eq_specification
```

# A Real Attention Theorem

TorchLean's FlashAttention specification illustrates the proof boundary. The following are genuine
Lean declarations:

```
#check Spec.flashAttention_eq_scaledDotProductAttention
#check Spec.flashAttentionBackward_eq_scaledDotProductAttentionBackward
```

They prove that the fused *Lean specification* has the same forward and backward denotation as
TorchLean's standard scaled-dot-product-attention specification. They are useful for semantic graph
rewrites and for stating the contract expected of a fused implementation.

Those theorems compare two Lean specifications. The native CUDA capsule additionally records the
runtime guards, regression tests, source provenance, and `checked` trust level used for the actual
kernel. Connecting PTX or a library call all the way to the specification would require another
refinement argument over Float32, layout, compiler, and hardware behavior.

# Forward And Backward Ownership

Inference asks for a forward value. Training asks for more: the value must remain connected to the
derivative rule used by the optimizer.

TorchLean distinguishes three VJP modes:

- `none`: no gradient is requested;
- `torchLeanTape`: TorchLean owns the tape and backward traversal; each capsule declares whether its
  local VJP is expressed through TorchLean operations or a named backend kernel;
- `backendVJP`: require capsules whose local VJP is computed by a backend kernel.

The preferred external-forward design is therefore precise: a provider may compute a fast forward
value, TorchLean records the same operation on its tape, and TorchLean applies the backward rule.
This requires enough forward information to be retained for that rule. If the bridge cannot provide
it, the implementation must fall back or expose a larger trust boundary.

The maintained LibTorch-forward profile implements this design for scaled-dot-product attention.
It prefers the registered LibTorch forward capsule and selects native CUDA capsules for surrounding
operations. Provider selection is therefore per operation, not a second model API or an
attention-specific boolean switch.

No-grad sessions request `none` automatically. During training, a differentiable operation cannot
select a forward-only capsule. Seeded random sources are the deliberate exception: they create
non-differentiable values, so they do not need a local VJP of their own.

# Boolean Attention Masks

TorchLean gives boolean attention masks one semantics across specifications and runtimes. A blocked
entry contributes exactly zero to the softmax numerator, as if its score were negative infinity.
Native CUDA skips blocked entries, while the LibTorch bridge passes a boolean mask directly to
scaled dot-product attention. Additive score biases remain a separate operation.

# Acceptance Gates

Planning and acceptance are separate steps. Planning finds capsules for graph operations. Auditing
turns their contract fields into obligation reports. An `AssurancePolicy` then decides whether the
run may proceed.

The implementation follows one explicit path:

1. `Target` describes the operating system, architecture, accelerator, and compiled features.
2. `Availability` states the devices and providers declared for planning. Eager execution performs
   the separate linked-library and hardware probes before launching a kernel.
3. `Registry` supplies compatible capsules, and `Planner` chooses one `PlannedKernel` for each graph
   operation. Together these choices form an `ExecutionPlan`.
4. `Audit` and `Recheck` expose the selected evidence and any obligations that must be discharged
   again for this run.
5. `Gate` applies the requested assurance policy. Eager execution receives an `AcceptedKernel` for
   each operation. Graph lowering can similarly produce an `AcceptedGraphPlan` for a later graph
   executor; the current eager runtime does not pretend that this data-level graph plan is executable.
6. The eager session binds the selected capsule to a `KernelHandler` with the same operation,
   provider, and device. If this build has no such handler, execution fails before entering a
   different provider's code.
7. `Report` renders the providers, devices, trust levels, and recheck dispositions for logs and
   benchmark records.

These are Lean data structures rather than an informal convention between command-line flags. The
eager runtime consumes the accepted per-operation value, binds it to the implementation it will
call, and records the capsule it actually used. Inspection tools can retain rejected graph plans
and explain why they failed.

`AcceptedKernel` and `AcceptedGraphPlan` carry the equality proof that their policy gate returned
`accepted`; they are not records that a caller can populate while omitting the gate result.

Runtime policies state whether guards, regression tests, fuzzing, or a named external dependency
are acceptable for a run. The `verified` policy rejects all of those engineering evidence classes.
The gate itself has a small Lean theorem:

```
#check ExecutionAudit.gate_eq_accepted_iff_gateFailures_eq_nil
```

This theorem says exactly what the policy function does: it accepts precisely when the audit has no
failures. It proves the gate's behavior, not the numerical correctness of a selected foreign
kernel.

The maintained CUDA wrappers also perform concrete checks at the FFI boundary. Convolution and
pooling validate rank and dimension conversion, nonzero strides, representable element counts,
buffer lengths, and operation-specific domains such as finite nonzero smooth-max $`\beta` after
conversion to Float32. The C/CUDA boundary repeats critical size and overflow checks. These guards
prevent malformed launches; they complement rather than replace a mathematical value-refinement
argument.

Capsule selection follows the run's scalar semantics, while a native capsule may separately state
that its device storage is Float32. The mixed-precision distinction is described once in
*TorchLean And PyTorch*; choosing another provider does not alter it.

# Runtime Configuration

The model API stays independent of these implementation details:

```
def trainerFor (backend : TorchLean.Runtime.Backend) :=
  Trainer.new mkModel
    { task := .regression
      optimizer := optim.adam { lr := 0.01 }
      dtype := .float32
      backend := backend }

let eagerTrainer := trainerFor .eager
let compiledTrainer := trainerFor .compiled
```

Device and provider selection live in runtime configuration and command-line options. Backend
selection leaves the model's forward function unchanged, much like the separation
between calling a PyTorch model and wrapping it with `torch.compile`: compilation changes
execution, not the mathematical intention of the model.

# Reading The Report

These statements have different strengths:

- "the example ran on CUDA" reports an execution path;
- "CUDA matched the CPU reference on this test suite" reports finite parity evidence;
- "the fused attention spec equals standard attention" cites a Lean semantic theorem;
- "the native attention kernel implements the fused spec" requires a native refinement argument;
- "the LibTorch result is correct" depends on the explicitly named LibTorch boundary unless a
  stronger checker or theorem covers it.

A backend report records the selected provider and the evidence attached to it. Keeping that report
beside a benchmark makes “CUDA” concrete: readers can see which operations were native, external,
checked, or proved.

# Where To Continue

Read [Choosing How A Model Runs](Runtime___-Autograd___-and-Interop/Choosing-How-A-Model-Runs/)
for the runtime API. Read
[From A Tensor Operation To A GPU Kernel](Floating-Point-and-Native-Boundaries/From-A-Tensor-Operation-To-A-GPU-Kernel/)
for the native implementation details. The
[Installation page](https://lean-dojo.github.io/TorchLean/installation/) lists platform commands and
the profiles currently wired into the repository.

# References

- Paszke et al., ["PyTorch: An Imperative Style, High-Performance Deep Learning
  Library"](https://arxiv.org/abs/1912.01703), NeurIPS 2019.
- PyTorch, [`torch.compile` reference](https://docs.pytorch.org/docs/stable/generated/torch.compile.html).
- PyTorch, [C++ and LibTorch API](https://docs.pytorch.org/cppdocs/).
- NVIDIA, [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).
- Dao et al., ["FlashAttention: Fast and Memory-Efficient Exact Attention with
  IO-Awareness"](https://arxiv.org/abs/2205.14135), NeurIPS 2022.
- George C. Necula, ["Proof-Carrying Code"](https://doi.org/10.1145/263699.263712), POPL 1997.
- Lean, [validating proofs](https://lean-lang.org/doc/reference/latest/ValidatingProofs/).
