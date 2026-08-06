---
title: Installation
layout: default
redirect_from:
  - /backends/
---

# Installation

If you want to try TorchLean on a laptop, start with the CPU build. It does not require PyTorch,
CUDA, or a GPU. The repository pins its Lean version in `lean-toolchain`, so Elan will select the
right compiler for you.

## A Five-Minute CPU Install

First install [Elan](https://github.com/leanprover/elan), the Lean toolchain manager. On Linux or
macOS:

```bash
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
```

Open a new terminal so that `elan`, `lean`, and `lake` are on your `PATH`. Then clone and build
TorchLean:

```bash
git clone https://github.com/lean-dojo/TorchLean.git
cd TorchLean
lake exe cache get
lake build
```

The cache command downloads compatible prebuilt Lean dependencies when they are available. It is
safe to omit; `lake build` will compile anything that is missing.

Run a small model to check the executable path:

```bash
lake exe torchlean quickstart_mlp --device cpu --steps 10
```

If those commands succeed, TorchLean is installed. You can inspect the available examples and
verification commands with:

```bash
lake exe torchlean --help
lake exe verify --help
```

That CPU build is the common starting point on every platform. From there, TorchLean can compile
its native CUDA runtime or link an external provider without changing the Lean model being run.
The table below separates paths that work today from targets that are represented in the backend
architecture but still need platform-specific runtime work.

| Platform | CPU | NVIDIA GPU | LibTorch provider | Current status |
| --- | --- | --- | --- | --- |
| Linux | &#10003; | &#10003; Native CUDA | SDPA forward with TorchLean backward | Supported |
| macOS, Intel or Apple silicon | &#10003; | Not applicable | Not yet | CPU supported; Metal is planned |
| Windows with WSL2 | &#10003; Linux path | &#10003; CUDA on WSL2 | Linux path | Recommended Windows setup |
| Native Windows | Bring-up target | Not validated | Not wired | Backend target exists; native toolchain work remains |

Here, "LibTorch provider" means the current scaled-dot-product-attention bridge, not a requirement
for ordinary TorchLean models and not a claim that every operation is delegated to PyTorch. The
CPU, native CUDA, and LibTorch sections below give the corresponding build commands.

## Linux

### CPU

You need Git, `curl`, and a C/C++ compiler. On Ubuntu or Debian:

```bash
sudo apt update
sudo apt install -y git curl build-essential
```

Then follow the five-minute install above. The default build uses the portable CPU runtime. It also
builds harmless CUDA stub archives so that CPU-only machines can compile the complete Lean project;
the stubs do not pretend that a GPU is present.

### NVIDIA CUDA

Install a supported NVIDIA driver and CUDA toolkit using NVIDIA's
[CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
TorchLean needs `nvcc`, cuBLAS, and cuFFT. Check the machine before rebuilding:

```bash
nvidia-smi
nvcc --version
```

Build and run the CUDA configuration:

```bash
lake -R -K cuda=true build
lake -R -K cuda=true exe torchlean quickstart_mlp \
  --device cuda --steps 10 --show-backend
```

The two CUDA choices happen at different times. `-K cuda=true` tells Lake to compile and link the
native CUDA implementation. `--device cuda` asks the executable to use it. A CPU-linked executable
rejects `--device cuda` instead of silently moving the run back to the CPU.

`nvcc` compiles the kernels for its own default architecture unless told otherwise. Add
`-K cuda_arch=sm_86` (or `native`, or a verbatim `-gencode` list) to compile for the GPU you deploy
on; the [CUDA guide]({{ '/cuda/' | relative_url }}) explains what the default costs you.

Use `-R` whenever you switch between CPU and CUDA configurations; it forces Lake to recompute the
build description. The CUDA regression suite is:

```bash
lake -R -K cuda=true exe nn_tests_suite
```

The [CUDA guide]({{ '/cuda/' | relative_url }}) covers deterministic reductions, parity checks,
sanitizers, and the remaining native-code trust boundary.

### Optional LibTorch Attention

The normal CPU and CUDA builds do not need LibTorch. TorchLean currently uses LibTorch only through
an optional scaled-dot-product-attention bridge. Download a matching GPU-enabled distribution from
the [official LibTorch installation page](https://docs.pytorch.org/cppdocs/installing.html) and
extract it somewhere outside the repository.

The extracted directory must contain `include/` and `lib/`. Pass its absolute path to Lake:

```bash
lake -R -K cuda=true -K libtorch=true \
  -K libtorch_home=/absolute/path/to/libtorch build
lake -R -K cuda=true -K libtorch=true \
  -K libtorch_home=/absolute/path/to/libtorch exe libtorch_sdpa_test
```

This enables one registered external provider. With the `libtorch_forward_cuda` profile,
scaled-dot-product attention uses LibTorch for its forward value while the surrounding operations
continue through TorchLean's native CUDA providers. TorchLean records the attention node and applies
its own local VJP during backward. It does not route every operation through PyTorch, and it does
not make LibTorch part of Lean's trusted kernel.

## macOS

Install Apple's command-line developer tools, then Elan and TorchLean:

```bash
xcode-select --install
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
git clone https://github.com/lean-dojo/TorchLean.git
cd TorchLean
lake exe cache get
lake build
lake exe torchlean quickstart_mlp --device cpu --steps 10
```

The CPU path works on Intel and Apple silicon. Modern macOS has no NVIDIA CUDA execution path.
TorchLean already reserves `--device metal` in its device vocabulary, but Metal/MPS kernels are not
implemented yet. Selecting Metal therefore returns an unsupported-device error; it does not quietly
run the CPU implementation.

## Windows

### Recommended: WSL2

The most reliable Windows setup is Ubuntu under WSL2. Open an administrator PowerShell prompt:

```powershell
wsl --install -d Ubuntu
```

After Windows restarts, open Ubuntu and follow the Linux instructions. For an NVIDIA GPU, follow
NVIDIA's [CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html). Install the
Windows NVIDIA driver and the CUDA toolkit inside WSL; do not install a second Linux display driver
inside WSL.

### Native Windows

Native Windows is represented in the backend target vocabulary, but it is still a bring-up target
rather than a regularly tested release path. Install Git, the Visual Studio C++ build tools, and the
Windows SDK. Then install Elan from PowerShell:

```powershell
curl -O --location https://elan.lean-lang.org/elan-init.ps1
powershell -ExecutionPolicy Bypass -f elan-init.ps1
del elan-init.ps1
```

The intended native CPU commands are:

```powershell
lake exe cache get
lake build
```

The remaining work is platform engineering: the native libraries must be compiled with a compatible
Windows C/C++ toolchain; CUDA and LibTorch must be discovered as `.lib` and DLL artifacts; Linux
linker options such as `-Wl,-rpath` must be replaced; and the CPU stubs and GPU runtime must be tested
under the Windows loader and ABI. Once those pieces are wired, the existing device, provider, and
capsule abstractions do not need to be redesigned. Until then, WSL2 is the supported route for both
CPU and NVIDIA GPU execution on Windows.

## Use TorchLean From Another Lean Project

Add TorchLean to the downstream project's `lakefile.lean`:

```lean
require TorchLean from git "https://github.com/lean-dojo/TorchLean.git" @ "main"
```

Then update and build:

```bash
lake update
lake exe cache get
lake build
```

Most model files need only:

```lean
import NN.API
open TorchLean
```

The numerical library has a smaller independent import:

```lean
import NN.Floats
open TorchLean.Floats
```

It includes formats, rounding, finite binary32 semantics, executable IEEE binary32 operations,
interval rounders, and scalar quantization. It does not load the tensor, model, autograd, CUDA,
certificate, or external-tool layers. Use `NN.Spec.Quantization` when tensor quantization is needed,
and `NN.Proofs.RuntimeApprox.FP32` when connecting binary32 arithmetic to runtime-approximation
proofs.

For development against a neighboring checkout, use a path dependency:

```lean
require TorchLean from "../TorchLean"
```

## From A Model To A Kernel

Installation ends once the project builds, but accelerated execution introduces one more question:
when a graph contains `matmul`, attention, or convolution, which implementation is allowed to run
it?

TorchLean separates three names:

- the **device** is where the work runs, such as CPU or CUDA;
- the **provider** is the implementation family, such as TorchLean, native CUDA, cuBLAS, cuFFT, or
  LibTorch;
- the **operation** is the mathematical work requested by the graph.

A device alone is not enough. Two CUDA providers may use different layouts, mask conventions,
gradient implementations, or trust assumptions. TorchLean records those facts in a **kernel
capsule**.

## What A Kernel Capsule Looks Like

A capsule is a Lean record for one implementation of one operation. The checked CUDA attention
capsule is defined in `NN.Backend.Attention`; the essential fields look like this:

```lean
def torchLeanComposed : KernelCapsule :=
  { name := "torchlean.composed_attention"
    op := .scaledDotProductAttention
    provider := .torchLean
    device := .cuda
    trustLevel := .checked
    supportsForward := true
    vjpMode := .torchLeanTape
    shapeContract := ...
    layoutContract := ...
    valueContract := ...
    vjpContract := ...
    numericalPolicy :=
      { rounding := .nearestEven
        subnormals := .implementationDefined
        contraction := .implementationDefined
        reduction := .implementationDefined } }
```

Each contract field contains a structured claim rather than a free-form success label. For example,
the layout field can state

```lean
{ claim := .layoutCompatibility .scaledDotProductAttention .flatRowMajor
  summary := "Q, K, V, mask, and output use contiguous row-major buffers"
  evidence := .runtimeGuard "Runtime.Autograd.Cuda.Tape.flashAttention" }
```

The claim says what must hold; the evidence says why the current profile is willing to use that
implementation. A runtime guard or regression suite remains visible as a guard or test. It is not
promoted to a theorem by placing it in the record.

When an implementation has a Lean refinement theorem, TorchLean keeps more than this metadata.
`ProofCarryingKernel` is indexed by the exact specification and stores the implementation together
with a proof that both functions agree on every input. `planVerifiedKernel` returns that typed
object without erasing the theorem. The ordinary metadata planner deliberately rejects capsules
that are only labelled `verified`; a string or trust tag cannot stand in for a proof.

The planner also checks that the shape, layout, value, and VJP fields contain the corresponding
claim for the capsule's operation. Evidence attached to the wrong field is rejected. A forward-only
capsule records that its VJP is unavailable instead of presenting `.none` as a refinement claim. A
summary or source name is documentation, not a theorem by itself.

The record answers questions that are otherwise easy to lose in runtime plumbing:

| Field | Question it answers |
| --- | --- |
| `op` | Which graph operation does this implementation claim to execute? |
| `provider`, `device` | Which runtime family and hardware target may select it? |
| shape and layout contracts | Which dimensions, memory order, mask convention, and payload assumptions are required? |
| value contract | What supports the forward-value claim? |
| VJP mode and contract | Who computes the local gradient, and what supports that claim? |
| trust level | Is the implementation checked, fuzzed, or an explicit external assumption? |
| numerical policy | Which rounding, subnormal, contraction, and reduction conventions does execution use? |

The operation tag records a semantic family such as convolution, pooling, reduction, or matrix
multiplication. Rank, axes, strides, padding, and index tensors stay in the graph payload and its
shape proofs. A provider therefore registers one capability for a family instead of separate
capsules for every rank-specific helper name.

The numerical policy records the conventions claimed by the capsule and retained in the execution
audit. A theorem or checker still has to
justify any numerical conclusion drawn from it. In particular, the graph interval checker accepts
its fixed-left reduction transfer only for capsules that advertise `.fixedLeft`; CUDA, cuBLAS,
cuDNN, and LibTorch reductions need an order-independent or backend-specific bound.

Capsules are declared in named source modules. They are not invented after a run to make its backend
look acceptable. Built-in operation and provider modules are flattened into the planner registry;
downstream code can extend a profile with `BackendProfile.withCapsuleModules`. Model architectures
do not register backends themselves: they lower to operations, and the planner selects an existing
capsule whose device, provider, gradient mode, and trust level satisfy the requested profile.
Adding a module with an existing name replaces the earlier contribution. Duplicate module names,
duplicate capsule identities, and missing or inadmissible capsules cause planning to fail.

## From A Capsule To Running Code

The capsule and the implementation are separate objects. Once the planner selects a capsule, the
eager runtime looks for a typed `KernelHandler` with the same operation, provider, and device. The
pair becomes executable only after those three identities agree:

```lean
structure KernelHandler (β : Type) where
  name : String
  op : BackendOp
  provider : Provider
  device : Device
  execute : KernelCapsule → IO β

structure ExecutableKernel (β : Type) where
  capsule : KernelCapsule
  handler : KernelHandler β
  operation_matches : handler.op = capsule.op
  provider_matches : handler.provider = capsule.provider
  device_matches : handler.device = capsule.device
```

This prevents a profile from selecting, for example, a native CUDA capsule while the runtime
quietly executes the reference CPU closure. If a provider is present in the registry but its
handler is not linked for that operation, execution fails with both the selected capsule and the
available handlers in the error.

The equalities above concern dispatch identity. They do not prove the body of `execute`. A CUDA or
LibTorch kernel still has exactly the evidence recorded in its capsule: runtime guards and tests,
fuzzing, or an explicit external assumption.

## Where The Theorems Enter

A capsule does not prove foreign code. It records the operation contract, implementation identity,
and evidence class used to admit that implementation. Theorems about TorchLean specifications live
in the proof modules, where Lean checks their statements and proof terms.

FlashAttention gives a concrete example. TorchLean already proves a semantic equality between its
fused attention specification and standard scaled dot-product attention:

```lean
#check Spec.flashAttention_eq_scaledDotProductAttention
#check Spec.flashAttentionBackward_eq_scaledDotProductAttentionBackward
```

Those are genuine Lean theorems about two Lean definitions. They justify replacing the standard
attention **specification** with the fused specification. They do not prove that a particular CUDA
binary implements either definition. The current native CUDA capsule therefore records runtime
guards, regression tests, source provenance, and trust level `checked`, not `verified`.

To introduce a genuinely verified native capsule, the work has to happen in this order:

1. define the mathematical operation and its domain assumptions in Lean;
2. state the runtime refinement claim, including shape, layout, Float32, and error assumptions;
3. define semantics for the implementation or its certificate format;
4. prove a typed refinement theorem connecting those semantics to the operation contract;
5. keep any remaining FFI, compiler, driver, or hardware assumption explicit.

The current capsule record deliberately has no constructor for an arbitrary proposition. That
prevents an unrelated theorem from being presented as a kernel refinement. Source files, native
symbols, runtime guards, regression suites, and fuzz runs remain useful audit information, but the
strict policy does not treat them as theorems.

## Planning And Assurance Policies

For each backend-visible graph operation, TorchLean follows the same sequence:

<ol class="install-flow">
  <li><strong>Read the graph.</strong> Keep the operation, shape, payload, and proof-facing specification.</li>
  <li><strong>Apply the profile.</strong> Restrict the device, providers, gradient ownership, and accepted trust levels.</li>
  <li><strong>Select a capsule.</strong> Choose a compatible implementation already present in the registry.</li>
  <li><strong>Audit and gate.</strong> Reject missing evidence or forbidden external boundaries.</li>
  <li><strong>Bind, execute, and report.</strong> The eager runtime pairs the selected capsule with a matching typed handler, rejects missing handlers, and prints the capsule on first use when `--show-backend` is set.</li>
</ol>

An accepted kernel or graph plan carries a Lean proof that its selected policy gate returned
`accepted`. No-grad execution requests only a forward contract; training rejects a forward-only
capsule for differentiable operations. Random sources are non-differentiable and therefore do not
need a local VJP.

The maintained profiles currently have these meanings:

| Profile | Forward provider | Backward ownership | Current status |
| --- | --- | --- | --- |
| `checked_cpu` | Reference CPU runtime | TorchLean tape | Implemented |
| `checked_cuda` | Native CUDA, cuBLAS, cuFFT | TorchLean tape or named backend VJP | Implemented |
| `libtorch_forward_cuda` | Prefer registered LibTorch forward capsules, otherwise native CUDA | TorchLean tape | Implemented for scaled-dot-product attention |

The default CUDA profile does not silently choose LibTorch. The LibTorch-forward profile is the
hybrid training path: it prefers a registered LibTorch forward capsule and uses native CUDA for
other operations. External providers must be enabled explicitly and remain visible in the backend
report.

Programmatic training code selects the complete profile as one value:

```lean
let run :=
  ({} : TorchLean.Trainer.RunConfig)
    |>.withBackendProfile NN.Backend.BackendProfile.libTorchForwardCuda
    |>.withBackendReport
```

This sets the device, provider preference, assurance policy, VJP ownership, and registry together.
Calling `withDevice` later selects the maintained profile for the new device, so a stale CUDA policy
cannot survive a device change.

The corresponding Lean definitions are in
[`NN.Backend.Capsule`]({{ '/docs/NN/Backend/Capsule.html' | relative_url }}),
[`NN.Backend.Profile`]({{ '/docs/NN/Backend/Profile.html' | relative_url }}), and
[`NN.Backend.Gate`]({{ '/docs/NN/Backend/Gate.html' | relative_url }}).

## Adding Another Platform

Metal, ROCm, WebGPU/WASM, TPU/XLA, Trainium/Neuron, custom chips, and caller-supplied accelerators
all use the same extension points:

1. add the device and provider vocabulary;
2. report build and runtime availability accurately;
3. register capsules only for the operations the provider implements;
4. connect the selected capsules to executable runtime dispatch;
5. supply shape, layout, value, and VJP evidence at the right trust level;
6. add platform CI and parity tests before calling the target supported.

Until those pieces exist, target selection fails. Asking for Metal or TPU must never become an
unreported CPU run.

## Check An Installation

These commands cover the normal CPU installation:

```bash
lake build
lake lint
lake exe nn_tests_suite
lake exe torchlean --help
lake exe verify --help
```

For CUDA, rebuild and run the suite with `-R -K cuda=true`.

For a complete account of Lean axioms, executable checkers, CUDA and FFI code, external artifact
producers, and floating-point assumptions, read
[`TRUST_BOUNDARIES.md`](https://github.com/lean-dojo/TorchLean/blob/main/TRUST_BOUNDARIES.md).

## References

- [Elan: Lean toolchain manager](https://github.com/leanprover/elan).
- [Lean reference: validating proofs](https://lean-lang.org/doc/reference/latest/ValidatingProofs/).
- [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
- [NVIDIA CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
- [Installing LibTorch](https://docs.pytorch.org/cppdocs/installing.html).
- George C. Necula, ["Proof-Carrying Code"](https://doi.org/10.1145/263699.263712), POPL 1997.
  Ordinary kernel capsules are contract and provenance records, not proof-carrying binaries.
  TorchLean's separate typed `ProofCarryingKernel` interface retains a Lean refinement theorem with
  an implementation when such a proof is available.
