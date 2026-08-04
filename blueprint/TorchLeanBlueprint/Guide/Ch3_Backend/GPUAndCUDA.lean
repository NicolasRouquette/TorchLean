import VersoManual

open Verso.Genre Manual

#doc (Manual) "From A Tensor Operation To A GPU Kernel" =>
%%%
tag := "gpu-and-cuda"
%%%

When the running MLP evaluates

$$`y=W x+b,`

the mathematical specification sees matrix-vector multiplication and addition. The CUDA runtime
sees something quite different: contiguous buffers, dimensions, an execution stream, a matrix
kernel, a broadcast, and several error checks. TorchLean keeps both views and records the contract
at the boundary between them.

This chapter follows one training step all the way to the GPU. CUDA is the maintained accelerator
today; the capsule and target types are intentionally not CUDA-specific, so a future Metal, ROCm,
TPU, or custom-chip provider can state the same kinds of obligations without pretending that its
implementation already exists.

# Build A Native CUDA Runtime

An ordinary CPU build compiles stub archives for CUDA symbols. The stubs let the package link on a
machine without the NVIDIA toolchain, but they reject CUDA session creation. To compile the native
implementation:

```
lake build -R -K cuda=true
```

Run this command from the repository root. `-R` rebuilds targets affected by the Lake configuration,
and `-K cuda=true` selects the CUDA source and link configuration. The build compiles TorchLean's
CUDA code and links the CUDA runtime, cuBLAS, and cuFFT where those libraries are used.

Run two optimizer steps and print the selected kernel contracts:

```
lake -R -K cuda=true exe torchlean quickstart_mlp \
  --device cuda --steps 2 --seed 2026 --show-backend
```

The model reports the same 25-example dataset as the CPU run. The backend report includes:

```
matmul: native_cuda.matmul
  provider=native-cuda trust=checked vjp=backend-vjp
  numeric=[round=nearest-even,
           subnormal=implementation-defined,
           contract=implementation-defined,
           reduce=implementation-defined]

add: native_cuda.add
  provider=native-cuda trust=checked vjp=backend-vjp

relu: native_cuda.relu
  provider=native-cuda trust=checked vjp=backend-vjp

mse_loss: native_cuda.mse_loss
  provider=native-cuda trust=checked vjp=backend-vjp
```

This output is worth reading closely. The model description contains two *linear layers*, but the runtime
decomposes them into reshape, permutation, matrix multiplication, broadcast, and addition. Capsules
describe the operations that actually crossed a backend boundary, not only the layer names in
source code.

# The Path Of One Linear Layer

For an unbatched input `x : [2]` and weight `W : [8,2]`, the eager CUDA path proceeds roughly as:

```
typed Tensor [2]
    ↓ upload / existing CUDA handle
opaque contiguous float32 buffer
    ↓ reshape and matrix-layout preparation
native_cuda.matmul
    ↓ broadcast bias [8] to output shape
native_cuda.add
    ↓
typed runtime Tensor [8]
```

The Lean wrapper knows the logical shapes and element counts. The device buffer is opaque; Lean
does not inspect its contents by reducing a theorem. Before each FFI call, wrappers check the
conditions they can observe:

- the session really targets CUDA;
- the selected capsule implements the requested operation;
- flat lengths match the logical shapes;
- every shape axis, runtime index, and element count fits the `UInt32` CUDA ABI;
- ranks, axes, strides, and operation-specific dimensions are supported.

Shape-erased values, output seeds, and supplied gradients pass through `AnyBuffer.validate` before
shape-derived indexing. Convolution and pooling also check spatial products independently, so a
zero channel dimension cannot hide an overflowing spatial shape. Their forward wrappers validate
the native output length immediately after the FFI returns, before storing the result on the tape.

Those checks prevent many ABI and memory errors. They do not prove that a CUDA thread computes the
correct arithmetic expression.

# What A Kernel Capsule Contains

A `KernelCapsule` is the audit record for one operation-provider pair:

```
structure KernelCapsule where
  name             : String
  op               : BackendOp
  provider         : Provider
  device           : Device
  trustLevel       : TrustLevel
  supportsForward  : Bool
  vjpMode          : VJPMode
  shapeContract    : ContractDescriptor
  layoutContract   : ContractDescriptor
  valueContract    : ContractDescriptor
  vjpContract      : ContractDescriptor
  numericalPolicy  : NumericalPolicy
```

The four descriptors answer different questions.

## Shape contract

For matrix multiplication, are the dimensions compatible and is the declared result shape correct?
This field is often supported by Lean-side runtime guards and shape-indexed source objects.

## Layout contract

Does `[m,n]` mean contiguous row-major storage with the last axis varying fastest? A correct matrix
formula paired with a transposed native layout still computes the wrong model.

## Value contract

Which mathematical function should the forward result refine? Evidence may be a Lean theorem, a
sound checker, a test suite, a fuzz oracle, or an explicit trusted boundary. The evidence variant
is visible in reports.

## VJP contract

Who owns backward, and which derivative specification should it implement? `backend-vjp` means the
runtime calls a backend derivative kernel while retaining TorchLean's tape structure.
`torchLeanTape` means TorchLean owns the graph and reverse traversal even if a capsule uses a named
backend kernel for its local VJP. A capsule marked `backend-vjp` makes that local numerical boundary
explicit in the audit.

Capsules record contracts. Runtime code supplies a typed `KernelHandler` for the result type of the
operation. Binding produces an `ExecutableKernel` only when the handler and capsule have equal
operation, provider, and device fields:

```
structure KernelHandler (β : Type) where
  name     : String
  op       : BackendOp
  provider : Provider
  device   : Device
  execute  : KernelCapsule → IO β

structure ExecutableKernel (β : Type) where
  capsule             : KernelCapsule
  handler             : KernelHandler β
  operation_matches   : handler.op = capsule.op
  provider_matches    : handler.provider = capsule.provider
  device_matches      : handler.device = capsule.device
```

The identity check prevents `native_cuda.matmul` from running the reference CPU closure under that
name. Numerical correctness remains exactly as strong as the capsule's theorem, checker, test, or
trusted-boundary evidence.

# Planning, Selection, And Execution

Three stages are intentionally separate:

```
registry
  all known capsule descriptions
       ↓ profile + availability + assurance gate
accepted plan
  capsules allowed for this target
       ↓ bind matching typed handler
executable kernel
  operation/provider/device identities agree
       ↓ provider-aware runtime dispatch
executed operation
  the native symbol actually called
```

`BackendProfile.acceptGraph` exposes an `AcceptedGraphPlan` only after planning, lowering, and the
assurance gate accept every obligation. This is still not a hardware probe: a profile target
declares build capabilities, while CUDA session creation separately calls
`Cuda.Buffer.requireNativeRuntime` to distinguish native CUDA, a native build with no visible GPU,
and the host-memory parity stubs.

This separation catches two bad failure modes:

1. a registry entry cannot silently claim that an unavailable library is executable;
2. a runtime wrapper cannot silently call native CUDA after the profile selected another provider.

Most fixed CUDA wrappers currently require `provider = nativeCuda`. If a profile selects an
unwired provider, execution fails with an error. There is no hidden CPU fallback for an unsupported
CUDA operation, because moving a tensor between devices behind the user's back would change both
performance and the execution claim.

# Try A Deliberate Failure

Build without native CUDA and request it:

```
lake build
lake exe torchlean quickstart_mlp --device cuda --steps 1
```

On a stub build, session initialization rejects the request. This is not an accidental limitation:
the CLI's `Device.cuda` value describes the requested target, while
`Cuda.Buffer.requireNativeRuntime` probes whether this build can execute it.

Likewise, names such as `metal`, `rocm`, `tpu`, and `trainium` parse as devices so profiles and
future integrations can describe them. The maintained profile lookup currently returns no runtime
for those devices. Selecting one produces a clear “no maintained runtime profile” error rather than
running on CPU.

# Native CUDA Versus cuBLAS

“CUDA” names the NVIDIA programming platform, not one kernel implementation. TorchLean's CUDA path
can use:

- custom `.cu` kernels for elementwise, reduction, indexing, convolution, attention, and other
  operations;
- cuBLAS for tuned dense linear algebra;
- cuFFT for Fourier transforms;
- CUDA runtime calls for allocation, copies, streams, and launch management.

cuBLAS is a vendor library inside the CUDA ecosystem. Calling it can be much faster than a simple
handwritten matrix kernel because it chooses algorithms specialized for dimensions, datatype, and
GPU generation. It is also a larger external trust boundary. The capsule should identify the
provider and numerical policy precisely enough that “native CUDA” does not hide whether the
arithmetic came from custom code or a vendor library.

# Maintained CUDA Operations

The eager CUDA tape currently covers elementwise arithmetic and activations, reductions and
broadcasting, shape transforms, gather/scatter, dense and batched matrix multiplication,
normalization and softmax, two-dimensional and N-dimensional convolution/transposed convolution,
max/average/smooth-max pooling, composed attention, a direct attention reference kernel, and fused
spectral convolution. Backward rules are recorded as tape nodes rather than delegated to an
invisible global autograd engine. Lower-level kernels additionally expose packed real FFT and
selective-scan entrypoints used by focused runtime paths and tests.

Layer normalization and tanh-approximate GELU are examples of the distinction between semantics
and scheduling. Each is one operation on the TorchLean tape, with a local forward rule and VJP.
The CUDA implementation evaluates that rule with one forward kernel and one backward kernel rather
than constructing the formula from a chain of temporary tensors. The fusion changes launch and
allocation behavior; it does not replace the TorchLean operation or hand backward ownership to an
external autograd engine.

Matrix derivatives use the same principle without introducing a fused model operation. Products
such as $`A^\mathsf{T}B` and $`AB^\mathsf{T}` are passed to cuBLAS with logical transpose flags.
The old path first copied an operand into a transposed buffer and then multiplied it. The new path
reads the original row-major storage directly. This applies to ordinary and batched matrix
multiplication, linear layers, projection weights, and any model built from them; attention is only
one caller. TorchLean still records the original matrix operation and its local VJP. Only the
schedule used to evaluate that VJP has changed.

This list does not mean every TorchLean operation has a CUDA implementation. Provider-aware
wrappers reject unsupported capsules and shapes; they do not copy a tensor to CPU and continue
silently. The native source map in `NN.Runtime.Autograd.Engine.Cuda.NativeSources` identifies the
Lean declarations and corresponding files under `csrc/cuda`.

# Batched Attention

A transformer block receives a tensor of shape `(batch, tokens, modelDim)`. The mathematical
operation applies the same attention layer to every batch entry, with shared projection matrices.
TorchLean keeps that description in the specification and compiled graph. Each sample is expressed
through the existing attention node, so its forward map, JVP, and VJP are the same definitions used
for unbatched attention.

The eager CUDA runtime schedules the work differently. It flattens `(batch, tokens)` for the four
shared projections and folds `(batch, head)` into the batch axis of the matrix multiplications.
Hard-masked softmax and the local backward rule still belong to TorchLean. The backward pass sums
the four projection-matrix gradients across the full batch.

This removes the old host loop over samples without introducing a second mathematical operation.
The verifier lowers batched attention to the per-sample graph, while CUDA executes one
batch-aware tape node. Regression tests compare the batched forward value, input gradient, and
shared weight gradients with repeated single-sample attention. The comparison is runtime evidence;
the cuBLAS calls and float32 behavior remain covered by the capsule's stated boundary.

# Forward Values And Backward Ownership

There are three useful configurations for an operation:

:::table +header
*
  * Forward
  * Backward
  * TorchLean owns
*
  * TorchLean native
  * TorchLean native VJP
  * graph, tape, value/VJP rules, native boundary
*
  * external fast kernel
  * TorchLean VJP
  * graph, tape, backward semantics; external forward boundary
*
  * external autograd
  * external autograd
  * only the surrounding contract and imported gradients
:::

The middle row is the preferred scaling direction when practical. An external provider computes a
fast forward value, but the wrapper still records a TorchLean tape node and applies TorchLean's
selected VJP. This keeps parameter ownership and optimizer flow local.

It does not magically prove the external forward. The forward capsule still needs checked,
tested, fuzzed, or trusted evidence connecting its value to the spec.

The checked CUDA profile does not use this hybrid LibTorch row by default. Its attention path is
the composed TorchLean operation described above; LibTorch forward is available only through the
explicit `libTorchForwardCuda` profile.

# The LibTorch Attention Example

LibTorch is PyTorch's C++ distribution. ATen is the lower tensor/operator library used inside it.
TorchLean's optional adapter calls LibTorch/ATen scaled-dot-product attention; it does not embed a
Python interpreter.

Build the optional provider with:

```
lake -R -K cuda=true -K libtorch=true build
lake -K cuda=true -K libtorch=true exe libtorch_sdpa_test
```

The maintained `libTorchForwardCuda` profile delegates scaled-dot-product-attention *forward* to
LibTorch while keeping a TorchLean tape VJP. A raw LibTorch backward test exists to compare
gradients, but the maintained profile does not hand default backward ownership to LibTorch
autograd.

Programmatic selection is explicit:

```
let run : Trainer.RunConfig :=
  ({} : Trainer.RunConfig)
    |>.withBackendProfile NN.Backend.BackendProfile.libTorchForwardCuda
    |>.withBackendReport true
```

This cannot be obtained by merely spelling `--device cuda`; ordinary CUDA selects
`BackendProfile.checkedCuda`.

# Hard Attention Masks

TorchLean's boolean attention meaning is:

```
true  = this key participates
false = this key has exactly zero softmax numerator
```

A fully blocked row returns zero. Native fused attention and the LibTorch adapter must preserve this
convention. The adapter constructs a boolean CUDA mask rather than replacing `false` by $`-1000`.

Why not use a finite sentinel? If an allowed logit is $`-5000` and a blocked logit receives
$`-1000`,
the blocked entry becomes *larger* and can dominate softmax. Negative infinity expresses a hard
support restriction; a finite additive bias expresses a different operation.

The attention regression suite includes masked forward values, $`\mathrm dQ`, $`\mathrm dK`,
$`\mathrm dV`, and fully blocked
rows. Those are tests of concrete cases. The pure FlashAttention theorem separately proves that the
spec-level tiled online-softmax definition equals the ordinary attention specification.

# Numerical Policy Is Not Decoration

The backend report for `matmul` says:

```
round=nearest-even
subnormal=implementation-defined
contract=implementation-defined
reduce=implementation-defined
```

This is more honest than claiming bitwise equality with a left-fold real specification.

- Hardware operations usually round to nearest-even, subject to instruction and library choices.
- Subnormal inputs may be preserved or flushed depending on hardware mode and kernel.
- multiplication and addition may be contracted into FMA;
- parallel reductions can choose a tree that differs from the pure interpreter's order.

For floating-point addition,

$$`(a+b)+c`

need not equal

$$`a+(b+c).`

A mathematically equivalent reduction tree can therefore produce different final bits. Numerical
certificates must use an error model or an exact policy strong enough for the claim; shape safety
alone is insufficient.

# Deterministic Reduction Mode

For reproducibility experiments, the CUDA buffer runtime can replace supported atomic reductions
with fixed-order implementations:

```
let enabled :=
  Runtime.Autograd.Cuda.Buffer.setDeterministicReductionsChecked true
```

The environment variable `TORCHLEAN_CUDA_DETERMINISTIC_REDUCTIONS=1` selects the same mode before
startup. Coverage includes scalar/axis/broadcast reductions, gather/scatter accumulation, and the
pooling backward kernels. It does not make seeded RNG unnecessary, force cuBLAS to a universal
bitwise contract, or prove equality across GPU models. `getDeterministicReductions` reports the
active setting so a training log can record it.

# Bound The Device Reuse Cache

Dropped CUDA buffers are not always returned immediately to the driver. TorchLean keeps exact-size
blocks in a process-wide reuse cache so a later allocation can avoid another `cudaMalloc`. This is
useful in a steady training loop, but a workload that visits many distinct tensor sizes can retain
more device memory than it will reuse.

Set a byte limit before starting the process:

```
TORCHLEAN_CUDA_CACHE_CAP_BYTES=$((512 * 1024 * 1024)) \
  lake -K cuda=true exe torchlean gpt2 --device cuda --steps 100
```

The limit is read once by the native allocator. A returned block that would exceed it waits for its
recorded CUDA event and is then freed instead of cached. `0`, or an unset variable, keeps the cache
unbounded. The value must be a decimal byte count; malformed and overflowing values produce a
warning and are ignored.

This limit applies only to released blocks in the reuse cache. It does not cap live model
parameters, activations, gradients, or workspaces.

Allocator telemetry distinguishes live tensors from reusable memory:

```
open Runtime.Autograd.Cuda

def printCudaMemory : IO Unit := do
  let stats ← Buffer.allocatorStats
  IO.println stats.format
```

`liveBytes` counts payloads owned by live TorchLean buffers. `cacheBytes` counts released device
blocks waiting for reuse and is therefore not included in `liveBytes`. `cacheCapBytes` reports the
limit installed by the native parser, with `0` meaning unbounded. The formatted report includes all
three values together with wrapper counts and `cudaMemGetInfo` totals.

The CUDA stress test starts fresh subprocesses because the limit cannot change after the allocator
has initialized. It checks a finite cap, an explicit unbounded control, malformed input, and integer
overflow against the real allocator. The CPU parity stub reports zero cached bytes and zero cap
because it frees host buffers directly.

# What The Tests Establish

The maintained CUDA checks are:

```
lake build -R -K cuda=true
scripts/checks/check.sh --cuda
```

They exercise allocation, uploads and downloads, shapes, operation values, gradients, error paths,
and selected numerical behavior. Native sanitizer runs add memory diagnostics. The LibTorch SDPA
test compares native and external forward/backward results within declared tolerances.

Passing these checks establishes that those cases worked on the tested build, driver, and GPU. It
does not universally prove:

- the C/CUDA source refines every spec operation;
- every GPU architecture behaves identically;
- every reduction is deterministic;
- the compiler and driver preserve source semantics.

Those components remain named in `TRUST_BOUNDARIES.md` and in capsule evidence.
