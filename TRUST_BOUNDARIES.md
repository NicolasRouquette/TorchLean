# TorchLean Trust Boundaries

TorchLean uses Lean to state and check mathematical claims about neural-network artifacts. Some
parts of the system are inside Lean's proof kernel; others are executable tools, native runtimes, or
external producers whose outputs may be checked by Lean.

This document records the assumptions that matter for correctness claims: Lean axioms, Prop-valued
contracts, CUDA and FFI code, external numeric oracles, PyTorch import/export scripts, Julia/Python
producers, and artifact-checking conventions.

## Reading This File

| Layer | Example | How to read it |
| --- | --- | --- |
| Lean theorem | graph semantics, selected autograd correctness theorems | checked by Lean |
| Executable checker | certificate parser, shape checker | checked by code/tests |
| Prop-valued contract | runtime Float32 agreement | assumption supplied by caller |
| FFI/native runtime | CUDA kernels, cuBLAS, cuFFT | external implementation path |
| External producer | Python, Julia, Arb, alpha-beta-CROWN | produces artifacts Lean may check |

When writing a correctness claim, name the layer explicitly:

- theorem claim: cite the Lean theorem and its hypotheses;
- executable-checker claim: cite the checker command, artifact schema, and accepted predicate;
- runtime claim: cite the backend, tests, sanitizer/parity evidence, and remaining native boundary;
- producer claim: cite the external tool or script and the artifact that Lean later checks.

This avoids collapsing "the command ran", "the checker accepted this artifact", and "Lean proved a
mathematical statement" into one sentence.

The backend planner and capsule vocabulary are documented in the
[Installation guide](https://lean-dojo.github.io/TorchLean/installation/#devices-providers-and-kernel-capsules).
It describes how TorchLean names native CUDA, LibTorch, and future platform providers before
a runtime path uses them. Capsule modules may extend a backend profile, but the ordinary alignment,
availability, trust-policy, and VJP gates apply to every contributed capsule.

## Lean Axioms

- `NN/MLTheory/CROWN/Lyapunov/Oracle.lean`: `crown_oracle` assumes an external CROWN
  checker has produced a `CrownOracleWitness lyap cert`; given that witness, the certificate
  soundly bounds `V` and `Vdot` over the stated region.
- `NN/Runtime/Autograd/Engine/Cuda/Trusted.lean`: `instNonemptyBuffer` is the nonemptiness witness
  Lean needs for opaque extern declarations returning `Cuda.Buffer`. It does not allocate or
  validate a CUDA buffer; real buffers still come from explicit FFI constructors/copy operations.
- `scripts/checks/repo_lint.py` allowlists these exact axiom names. New axioms must be added here
  deliberately and documented in this file.

You can inspect theorem dependencies inside Lean with:

```lean
#print axioms Runtime.Autograd.Cuda.instNonemptyBuffer
#print axioms NN.MLTheory.CROWN.Lyapunov.crown_oracle
```

For an audit from the shell:

```bash
rg -n "^(noncomputable\\s+)?opaque |^axiom " NN -g'*.lean'
```

## Prop-Valued Contracts

Some declarations are `class ... : Prop` or `structure ... : Prop` rather than axioms. These are
not kernel assumptions by themselves: a theorem using one is conditional on the caller supplying the
fields. We still treat them as part of the trust model, because public theorem names and docs should
make those assumptions visible.

Important examples include:

- `TorchLean.Floats.IEEE754.Float32Bridge.RuntimeFloat32FiniteMatchesIEEE32Exec`, the runtime
  contract for bit-level agreement on finite inputs and finite results, plus classification
  agreement for special values. NaN payload propagation is deliberately not assumed.
- `NN.MLTheory.CROWN.Graph.CrownCertSoundness.CrownTransferSound`, the transfer-rule soundness
  assumption used by graph-CROWN certificate theorems for backend/oracle-dependent relaxations.
- `NN.MLTheory.Proofs.UniversalApproximation.FloatIntervalApprox.OpsExact.Sound`, the local
  operation level exact interval soundness contract for finite IEEE32 interval arithmetic.

## Opaque Non-FFI Declarations

- `NN.MLTheory.CROWN.Lyapunov.CrownOracleWitness` is an abstract witness type for the external
  Lyapunov oracle.
- `NN.MLTheory.CROWN.betaAt` is an executable wrapper around a length-checked beta-phase array
  lookup. It keeps the checker executable without exposing brittle `Array.get!` internals to every
  proof.

## CUDA Runtime

- Files under `csrc/cuda/` are trusted FFI code. Lean checks shape metadata around calls, but kernel
  memory safety, launch behavior, and numerical behavior are outside Lean's proof kernel.
- Shape-erased tape inputs must match their native buffer length, and dimensions, indices, and
  output element counts must fit the CUDA `UInt32` ABI before FFI calls. Native geometry checks
  repeat critical guards; these are executable checks, not proofs of the kernels.
- `csrc/cuda/tensor/torchlean_cuda_tensor.cu` stores CUDA buffers as float32 and converts Lean `Float`
  values to/from float32 at the buffer boundary.
- CUDA externs borrow Lean buffers, arrays, and float arrays passed as inputs. Their Lean
  declarations use `@&` for that calling convention; the native functions must neither retain nor
  decrement those borrowed objects. The CUDA stress suite creates thousands of short-lived
  wrappers and checks that every wrapper created in the loop is finalized.
- CUDA buffer finalizers free device memory through `cudaFree`. This is safe for TorchLean's current
  default-stream runtime, where launches and host copies are ordered through the default stream. If
  future backends introduce user streams or asynchronous graph replay, finalizer/free ordering must
  be revisited explicitly.
- Sparse backward consumes some native gradient buffers. Seeds entering that path therefore come
  from effectful constructors, which guarantee a fresh allocation, and ownership transfers use
  copy-and-release operations. Repeated-backward tests check that seeds remain usable and that live
  allocation stays flat; NVIDIA Compute Sanitizer checks the exercised path for native memory
  errors. These checks can catch bad lifetime handling, but Lean does not prove `cudaMalloc`,
  `cudaMemcpy`, or `cudaFree`.
- GPU matmul supports two explicit precision paths:
  - FP32: `NN/Runtime/Autograd/Engine/Cuda/Kernels.lean` uses `Cuda.Buffer.bmm`, backed by
    `cublasSgemmStridedBatched` in `csrc/cuda/kernels/torchlean_cuda_kernels.cu`.
  - FP64: `NN/Runtime/Autograd/Engine/Cuda/DGemm.lean` uses `torchleanDgemmCuda`, backed by
    `cublasDgemm` in `csrc/cuda/blas/torchlean_dgemm_cuda.cu`.
- The fast-kernel Float dispatcher makes this choice explicit via `GpuMatmulPrecision`.
- Several CUDA backward/reduction paths use `atomicAdd`. These are mathematically standard for
  accumulation but are not bit-deterministic across schedules because float32 addition is not
  associative.
- TorchLean provides an opt-in deterministic reductions mode that replaces the `atomicAdd`-based
  accumulation paths with fixed-order algorithms (slower, but bit-stable across runs on the same
  GPU). You can enable it either:
  - from Lean (recommended): `let _ := Runtime.Autograd.Cuda.Buffer.setDeterministicReductionsChecked true`
  - via env var: `TORCHLEAN_CUDA_DETERMINISTIC_REDUCTIONS=1`
  Coverage includes:
  - reductions: `Buffer.reduceSum`, `Buffer.reduceMean`, `reduceFromBroadcastTo`, `reduceSumAxis`
  - gather/scatter backprop: `scatterAdd`, `scatterAddRows`
  - pooling backward: `max_pool*`, `avg_pool*`, `smooth_max_pool*` (2D and N-D entrypoints)
  Does not cover:
  - nondeterminism from RNG (use seeded RNG ops, or manage seeds/counters explicitly)
  - numerically different results across GPU architectures, CUDA toolkit versions, or driver versions
  - kernels that are not on the deterministic-reductions allowlist (only the atomic-accumulation paths above)
- CUDA max-pooling follows the TorchLean spec, which models PyTorch-style negative-infinity padding
  by ignoring padded cells outside the domain when selecting the max. Backward tie-breaking is
  TorchLean-spec row-major deterministic when deterministic reductions are enabled, while external
  runtimes may choose different tie-breaking policies.
- FlashAttention has a fused-operator denotation for proofs in
  `NN/Spec/Layers/FlashAttention.lean`: over the spec semantics it denotes the same masked scaled
  dot-product attention as the standard `QKᵀ -> mask -> softmax -> PV` graph. The CUDA eager
  multi-head attention path can use native fused runtime kernels exposed through
  `NN/Runtime/Autograd/Engine/Cuda/Kernels.lean` and implemented in
  `csrc/cuda/kernels/torchlean_cuda_kernels.cu`. Those kernels favor clarity and correctness: fused
  forward/VJP kernels over already-split heads, not a production clone of Dao-AILab's tiled
  implementation. The Lean equalities cover the denotational target; online-softmax tiling, CUDA
  memory behavior, and float32 arithmetic remain part of the native runtime boundary. TorchLean
  regression-tests the fused kernels against the composed attention path, and theorem claims should
  cite the spec denotation rather than the CUDA machine code. References: FlashAttention
  (arXiv:2205.14135), FlashAttention-2 (arXiv:2307.08691), FlashAttention-3 (arXiv:2407.08608),
  and the Dao-AILab `flash-attention` implementation.
- Boolean attention masks use hard masking throughout the spec semantics: blocked entries
  contribute zero softmax numerator, matching true `-inf` masking at the denotational level. The
  CUDA attention kernels implement that same hard-mask convention. Separate finite additive-bias
  attention lemmas still exist for models that intentionally add a fixed score bias, but those
  lemmas are not the semantics of boolean causal masks.
- Kernel launch synchronization is an implementation detail of the native runtime. Tensor/view
  kernels usually rely on default-stream ordering and later host copies to synchronize; conv/pool
  kernels explicitly synchronize after exported operations for clearer error attribution around
  heavier kernels. Both policies are outside Lean's kernel and should not be used as proof evidence.

## Executable Floating Point

- `NN/Floats/IEEEExec/` proves and implements a deterministic IEEE-style executable model for many
  core operations.
- `NN/Proofs/RuntimeApprox/Graph/NumericalCertificate.lean` checks graph-wide binary32 interval
  traces against the canonical `NN.IR.Graph`. It rebuilds ranges rather than trusting claimed
  endpoints, rejects non-finite replay values, and re-runs backend planning before accepting the
  embedded execution audit. A `CheckedCertificate` stores the exact graph checked, and
  `executeIEEE32` can replay only that stored graph.
- Transcendental functions such as `exp`, `log`, and `tanh` are deterministic approximations unless
  a file states a stronger theorem for a specific operation.

Kernel capsules now record four numerical choices: rounding, subnormal handling, contraction/FMA,
and reduction order. These fields are audited contract data, not proof evidence. Portable reference
accumulations advertise their fixed left fold. Native CUDA and LibTorch matrix products, convolutions,
normalizations, pooling operations, FFT/FNO paths, scans, and attention advertise
implementation-dependent reductions. Consequently, the fixed-left graph certificate refuses to
reuse its transfer for those accelerated paths. A theorem about such a path needs either a
backend-specific schedule or the order-independent enclosure from
`NN/Floats/IEEEExec/Reductions.lean`.

For a checked replay, interval validity proves that each endpoint is finite and ordered. The replay
also checks every computed entry for finiteness. `CheckedRealExecution` separately proves that the
exact-real denotation of the stored graph lies in the same trace. Pairing it with a checked bit-level
execution through `CheckedExecution.errorTrace` gives the pointwise interval-width bound for every
intermediate. This is a theorem about the `IEEE32Exec` replay. Transporting it to Lean runtime
`Float32`, CUDA, LibTorch, cuBLAS, or cuDNN still requires the agreement recorded by that backend's
capsule.

The proof-bearing `RevGraph` path has rounded forward and VJP theorems and erases to executable
autograd `GraphData`. One optimizer contract carries those gradient bounds through SGD,
momentum-SGD, and AdamW; AdamW supplies additional positivity and denominator-margin evidence at
each step. These are Lean theorems about the `NF` rounded-real scalar model. The canonical
`NN.IR.Graph` compiler currently proves forward semantic preservation only. It must not be cited as
an autograd or backward-certificate theorem until an autograd-capable lowering and correspondence
proof are added.

Use the float layers as follows:

| Claim | Layer to cite |
| --- | --- |
| executable binary32 behavior inside Lean | `NN/Floats/IEEEExec` |
| finite rounded-real float32 error bound | `NN/Floats/FP32` |
| precision-parametric rounding theorem | `NN/Floats/NeuralFloat` |
| endpoint interval enclosure | `NN/Floats/Interval` |
| external high-precision enclosure evidence | `NN/Floats/Arb` plus the oracle boundary |
| runtime CUDA/LibTorch/Lean `Float` behavior | runtime bridge or trust-boundary statement |

## External Numeric Oracles

- LibTorch may be used as an external forward-kernel provider for selected runtime paths. The
  maintained LibTorch-forward attention capsule returns the forward value, records the ordinary
  TorchLean tape node, and uses TorchLean's local VJP. The forward value is still trusted under the
  capsule's runtime agreement assumption. TorchLean does not maintain a LibTorch-autograd profile;
  tape ownership, gradient extraction, and optimizer handoff remain in the TorchLean runtime.
- CROWN/Lyapunov certificate generation is an external evidence producer when used through the
  oracle-backed workflow. Lean isolates that assumption behind `crown_oracle`; theorem claims should
  state exactly which certificate predicate was checked and which external completeness assumption is
  being used.
- The Arb / `python-flint` integration under `NN/Floats/Arb/` is an external subprocess backend. It
  can produce high-quality interval evidence, but an Arb response is still an oracle result unless
  the relevant certificate is independently checked in Lean.
- PyTorch import/export scripts and training helpers are external producers of weights, examples,
  or JSON artifacts. TorchLean can parse and replay those artifacts, but PyTorch training itself is
  not part of Lean's trusted kernel.
- The optional Julia wrapper `NN/Runtime/External/Julia.lean` follows the same pattern. It resolves
  `TORCHLEAN_JULIA` when set, otherwise falls back to `julia` on `PATH`, and does not require Julia
  at compile time. It supports “untrusted producer, Lean checker” workflows such as the
  piecewise-polynomial spline certificate workflow (producer scripts under `scripts/verification/splines/`,
  bundled fixtures under `NN/Examples/Verification/Splines/`).
- A Julia-produced spline or PINN artifact is trusted only after a Lean checker validates the small
  certificate data it needs: for example cell domains, polynomial coefficients, interval bounds, and
  claimed residual inequalities. Lean does not trust Julia's fitting process, optimizer, GPU use, or
  floating-point arithmetic merely because the subprocess returned successfully.
