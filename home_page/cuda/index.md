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

Running on a GPU and proving an implementation correct are separate claims. CUDA makes realistic
training and inference possible, while its machine code remains outside Lean's kernel.

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
