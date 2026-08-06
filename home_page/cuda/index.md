---
title: CUDA
layout: default
---

# CUDA

CUDA is a runtime path, not a separate mathematical model. When CUDA is enabled, TorchLean can use
native GPU kernels for selected tensor operations while the graph IR, verification artifacts, and
proof statements keep their own stated semantics and assumptions.

The responsibilities are separate:

- the spec layer owns the mathematical meaning;
- the TorchLean runtime owns the graph or tape node used for training;
- a backend capsule records whether the local VJP comes from TorchLean's tape or from the native
  provider;
- CUDA owns selected Float32 kernels, device buffers, launches, and library calls;
- tests, sanitizer runs, and trust-boundary docs say what evidence supports the native path.

The distinction matters because "ran on GPU" is not the same statement as "proved correct." CUDA can
make training and inference realistic without making the CUDA machine code part of Lean's kernel.

## Build and Run

Build with CUDA support:

```bash
lake -R -K cuda=true build
```

Run a small CUDA example:

```bash
lake -R -K cuda=true exe torchlean mlp --device cuda --steps 100
```

Run a broader CUDA regression pass:

```bash
scripts/checks/example_regression.sh --cuda
```

Run the CUDA sanitizer suite when changing native kernels:

```bash
scripts/checks/cuda_sanitize_tests.sh --all-tools
```

## Compilation Target

`-K cuda=true` says to compile the native kernels; it does not say which GPU to compile them
*for*. That second choice belongs to `nvcc`, which without instruction applies a built-in default
that changes with the toolkit version — CUDA 13.0, for instance, emits `sm_75` machine code plus
`compute_75` PTX. Such a binary runs at full speed on that architecture, reaches a newer GPU only
through forward PTX just-in-time compilation, and does not load on an older one at all. Name the
target when the deployment GPU is known:

```bash
lake -R -K cuda=true -K cuda_arch=sm_86 build     # an A10G or an RTX A4500
lake -R -K cuda=true -K cuda_arch=native build    # whatever GPU this machine has
```

A value starting with `-` is passed to `nvcc` verbatim, which is how one binary is compiled for
several architectures at once:

```bash
lake -R -K cuda=true \
  -K cuda_arch="-gencode arch=compute_75,code=sm_75 \
                -gencode arch=compute_90,code=[sm_90,compute_90]" build
```

`TORCHLEAN_CUDA_ARCH` carries the same value when the Lake option is absent, so a container image
or CI job can select a target without rewriting its build command. Both spellings participate in
Lake's build trace: changing the target recompiles the kernels. `nvcc`'s own `NVCC_APPEND_FLAGS`
does not — Lake cannot see it, judges the existing objects current, and links kernels compiled for
the previous target, which is visible only as unexplained throughput. Prefer the option, and delete
`.lake/build/torchlean_*.o` if a target was ever set that way.

`scripts/checks/cuda_arch_target.sh` asserts this behavior, and needs neither a toolkit nor a GPU.

## What CUDA Covers

The CUDA path is used for supported Float32 tensor operations: elementwise arithmetic, reductions,
matmul/cuBLAS paths, convolution/pooling kernels, shape/view operations, attention kernels, FFT/FNO
support where enabled, and model examples that choose `--device cuda`.

The public API should still look like one model with a backend choice. A user should not need a
separate "CUDA forward" function in ordinary code. The backend changes where supported kernels run;
it should not silently change tensor shapes, graph identities, mask semantics, parameter layout, or
the theorem statement attached to a checker.

Before a supported operation runs, the eager session binds its selected capsule to a handler with
the same operation, provider, and device. A missing handler is an execution error; TorchLean does
not print one provider in the audit report and quietly call another.

## Training Boundary

For training, TorchLean keeps the derivative boundary visible. A backend capsule states both the
forward provider and the VJP mode. The native fused-attention capsule uses CUDA forward and VJP
kernels. The optional LibTorch SDPA capsule uses LibTorch only for the forward value and records a
TorchLean tape node for its local VJP. Other operations follow the mode declared by their selected
capsule.

What TorchLean avoids is an unrecorded switch to a foreign autograd tape. Such a switch changes
parameter ownership, graph identity, and the assumptions behind backward execution. If no capsule
satisfies the requested forward, VJP, device, and trust policy, planning fails instead of silently
claiming a different boundary.

## Determinism and Evidence

Some CUDA reduction and backward paths use floating-point accumulation. Because Float32 addition is
not associative, atomic accumulation can be schedule-dependent. TorchLean also provides an opt-in
deterministic reductions mode for the covered reduction, gather/scatter, and pooling-backward paths:

```lean
let _ := Runtime.Autograd.Cuda.Buffer.setDeterministicReductionsChecked true
```

or:

```bash
TORCHLEAN_CUDA_DETERMINISTIC_REDUCTIONS=1 lake -R -K cuda=true exe torchlean mlp --device cuda
```

Evidence levels should be stated carefully:

| Statement | Meaning |
| --- | --- |
| CUDA example ran | The native runtime path executed for that command. |
| CUDA parity/regression test passed | Tested kernels matched reference cases on the exercised inputs. |
| cuda-memcheck passed | The sanitizer did not find the checked memory/synchronization issue class on that suite. |
| Lean theorem applies | A Lean theorem connects Lean side specifications, graph semantics, or certificate checks. |
| Native kernel verified | A theorem about the native CUDA implementation itself, rather than only the Lean side spec or boundary contract. |

For the full explanation, read
[GPU and CUDA Boundaries]({{ '/blueprint/Floating-Point-and-Native-Boundaries/From-A-Tensor-Operation-To-A-GPU-Kernel/' | relative_url }}).
For the public runtime choice, read
[Backend Selection and Trust]({{ '/blueprint/Runtime___-Autograd___-and-Interop/Choosing-How-A-Model-Runs/' | relative_url }}).
