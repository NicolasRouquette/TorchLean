/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

CUDA buffer primitives (float32).

Implementation:
- CUDA: `csrc/cuda/tensor/torchlean_cuda_tensor.cu`
- CPU stub (default `lake build`): `csrc/cuda/tensor/torchlean_cuda_tensor_stub.c`

These are low-level, runtime-only kernels for the native GPU tape/buffer path.
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Trusted

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

namespace Buffer

/-!
### Deterministic Reductions Mode

TorchLean's CUDA runtime uses `atomicAdd` in a few kernels to accumulate float32 results. This is
fast, but floating-point addition is non-associative, and CUDA does not fix a global order for the
interleaving of atomic updates. As a result, some kernels can be bit-nondeterministic across runs.

TorchLean therefore exposes an opt-in deterministic mode that replaces those atomic accumulation
paths with fixed-order reductions. This trades performance for reproducibility.

This flag is a *runtime* setting affecting only the CUDA/stub backends; it has no effect on the
pure Lean Spec.
-/

@[extern "torchlean_cuda_set_deterministic_reductions"]
opaque setDeterministicReductionsRaw (on : UInt32) : Unit

@[extern "torchlean_cuda_get_deterministic_reductions_u"]
opaque getDeterministicReductionsRaw (u : UInt32) : UInt32

@[extern "torchlean_cuda_set_deterministic_reductions_checked"]
opaque setDeterministicReductionsCheckedRaw (on : UInt32) : UInt32

/--
Enable/disable deterministic reductions mode and return the observed flag value.

Why this helper exists: the raw setter returns `Unit`, so if you write `let _ := set...` in
Lean, the compiler is free (under pure semantics) to reorder or eliminate that call. The runtime
therefore provides a `*_checked` wrapper that both sets the flag and returns the observed value,
giving us a single call with an explicit return value dependency.
-/
def setDeterministicReductionsChecked (on : Bool) : Bool :=
  setDeterministicReductionsCheckedRaw (if on then 1 else 0) != 0

/-- Enable/disable deterministic reductions mode (see module docstring). -/
def setDeterministicReductions (on : Bool) : Unit :=
  let _ := setDeterministicReductionsChecked on
  ()

/-- Query whether deterministic reductions mode is enabled. -/
def getDeterministicReductions : Bool :=
  getDeterministicReductionsRaw 0 != 0

/-! ### Allocator Telemetry -/

@[extern "torchlean_cuda_allocator_live_bytes"]
opaque allocatorLiveBytesRaw (u : UInt32) : UInt64

@[extern "torchlean_cuda_allocator_peak_bytes"]
opaque allocatorPeakBytesRaw (u : UInt32) : UInt64

@[extern "torchlean_cuda_allocator_alloc_count"]
opaque allocatorAllocCountRaw (u : UInt32) : UInt64

@[extern "torchlean_cuda_allocator_free_count"]
opaque allocatorFreeCountRaw (u : UInt32) : UInt64

@[extern "torchlean_cuda_allocator_device_free_bytes"]
opaque allocatorDeviceFreeBytesRaw (u : UInt32) : UInt64

@[extern "torchlean_cuda_allocator_device_total_bytes"]
opaque allocatorDeviceTotalBytesRaw (u : UInt32) : UInt64

@[extern "torchlean_cuda_allocator_cache_bytes"]
opaque allocatorCacheBytesRaw (u : UInt32) : UInt64

/--
Snapshot of the CUDA buffer allocator.

`liveBytes`/`peakBytes` count TorchLean buffers created by this runtime layer. `deviceFreeBytes`
and `deviceTotalBytes` come from `cudaMemGetInfo` in the CUDA build and are `0` in the CPU stub.
Together they let long-running examples distinguish a TorchLean lifetime leak from broader CUDA
memory pressure or fragmentation.

`cacheBytes` is the device memory held in the buffer reuse cache — dropped buffers awaiting reuse,
which are *not* counted in `liveBytes`. `TORCHLEAN_CUDA_CACHE_CAP_BYTES` bounds it; it is always `0`
in the CPU stub, which keeps no cache.
-/
structure AllocatorStats where
  liveBytes : UInt64
  peakBytes : UInt64
  allocCount : UInt64
  freeCount : UInt64
  deviceFreeBytes : UInt64
  deviceTotalBytes : UInt64
  cacheBytes : UInt64
deriving Repr

/--
Read the current CUDA allocator counters.

`token` is ignored by the native implementation.  It exists so call sites that sample repeatedly
can pass a changing value (for example, the training step), preventing Lean from treating repeated
FFI reads as identical pure expressions.
-/
def allocatorStatsWithToken (token : UInt32) : IO AllocatorStats := do
  pure
    { liveBytes := allocatorLiveBytesRaw token
      peakBytes := allocatorPeakBytesRaw token
      allocCount := allocatorAllocCountRaw token
      freeCount := allocatorFreeCountRaw token
      deviceFreeBytes := allocatorDeviceFreeBytesRaw token
      deviceTotalBytes := allocatorDeviceTotalBytesRaw token
      cacheBytes := allocatorCacheBytesRaw token }

/-- Read the current CUDA allocator counters. Prefer `allocatorStatsWithToken` in repeated loops. -/
def allocatorStats : IO AllocatorStats :=
  allocatorStatsWithToken 0

/-- Format a byte count as MiB for allocator progress messages. -/
def mibString (bytes : UInt64) : String :=
  let mib := (Float.ofNat bytes.toNat) / (1024.0 * 1024.0)
  toString mib ++ " MiB"

/-- One-line allocator report for progress logs. -/
def AllocatorStats.format (s : AllocatorStats) : String :=
  "live=" ++ mibString s.liveBytes ++
  " peak=" ++ mibString s.peakBytes ++
  " allocs=" ++ toString s.allocCount ++
  " frees=" ++ toString s.freeCount ++
  " cuda_free=" ++ mibString s.deviceFreeBytes ++
  " cuda_total=" ++ mibString s.deviceTotalBytes ++
  " cache=" ++ mibString s.cacheBytes

/--
Create a device buffer by copying from a host `FloatArray` (casts each element to float32).

This primitive has a pure Lean type, but the native implementation allocates a fresh device buffer.
Runtime code that repeatedly uploads the same host value should prefer `ofFloatArrayIO`, which adds
an IO token so two uploads cannot be collapsed into the same external object after one is released.
-/
@[extern "torchlean_cuda_buffer_of_float_array"]
opaque ofFloatArray (a : FloatArray) : Buffer

@[extern "torchlean_cuda_buffer_of_float_array_with_token"]
opaque ofFloatArrayWithToken (a : FloatArray) (token : UInt32) : Buffer

/--
Effectful host-to-device upload.

The token is ignored by C/CUDA. Its purpose is semantic: repeated uploads of the same `FloatArray`
must still allocate distinct device buffers. Without a changing token, Lean can treat the extern as
a pure expression, which is not the ownership model we want for long eager CUDA training loops.
-/
def ofFloatArrayIO (a : FloatArray) : IO Buffer := do
  let t ← IO.monoNanosNow
  pure <| ofFloatArrayWithToken a (UInt32.ofNat t)

/-- Copy a buffer back to a host `FloatArray` (casts float32 elements to `Float`). -/
@[extern "torchlean_cuda_buffer_to_float_array"]
opaque toFloatArray (b : Buffer) : FloatArray

/-- Number of float32 elements in the buffer. -/
@[extern "torchlean_cuda_buffer_size"]
opaque size (b : Buffer) : UInt32

/--
Release the device allocation held by a buffer, returning `1` when a live allocation was released.

This is a runtime pressure valve for eager training loops that create many short-lived CUDA buffers.
The C finalizer is still safe after an explicit release because the pointer is nulled out.
-/
@[extern "torchlean_cuda_buffer_release"]
opaque release (b : Buffer) : UInt32

/--
Release `workspace` and return `keep`.

This exists for pure CUDA tape code: because the returned buffer is used downstream, Lean cannot
erase the native release call as dead code.
-/
@[extern "torchlean_cuda_buffer_release_then"]
opaque releaseThen (workspace keep : Buffer) : Buffer

/--
Release a collection of workspace buffers and return `keep`.

Many CUDA tape formulas create a group of intermediate buffers, then continue with one final result
buffer. Threading cleanup through the result keeps ownership local to the formula and avoids waiting
for external-object finalizers in long training loops.
-/
def releaseManyThen (workspace : List Buffer) (keep : Buffer) : Buffer :=
  workspace.foldr (fun b acc => releaseThen b acc) keep

/--
A CUDA result together with workspace buffers that were needed to compute it.

This is the common ownership shape for eager CUDA formulas.  Some forward computations need
intermediate buffers again during the backward pass, so the tape keeps those buffers on the node
and releases them when the node is retired.  Backward formulas use the same shape when they
recompute a value only to differentiate through it.
-/
structure WithWorkspace where
  value : Buffer
  workspace : List Buffer := []

namespace WithWorkspace

/-- Return `keep` after releasing all workspace buffers owned by this result. -/
def releaseWorkspaceThen (r : WithWorkspace) (keep : Buffer) : Buffer :=
  releaseManyThen r.workspace keep

/-- Return `keep` after releasing both the result buffer and its workspace buffers. -/
def releaseAllThen (r : WithWorkspace) (keep : Buffer) : Buffer :=
  releaseThen r.value <| releaseManyThen r.workspace keep

end WithWorkspace

/--
Ask the Lean runtime allocator (mimalloc) to collect abandoned/free pages.

This does not change any TorchLean value. It is a pressure valve for long native eager loops where
many short-lived tape closures and external-buffer wrappers are created every step.

In the CUDA build, `force = true` also releases cached device blocks held by the native buffer pool.
That gives training code a way to trade reuse for returning memory to the CUDA driver at clear
phase boundaries.
-/
@[extern "torchlean_runtime_collect_allocator"]
opaque collectAllocatorRaw (force : UInt32) : UInt32

/-- Collect the native allocator's free pages. -/
def collectAllocator (force : Bool := true) : UInt32 :=
  collectAllocatorRaw (if force then 1 else 0)

/-!
### Scoped device-memory arena (`withCudaArena`)

A long *pure* eager loop — a `foldl` of `Buffer → Buffer` ops with no IO sequencing — keeps every
intermediate `Buffer` GC-*reachable* until the final readback, so the finalizers that would free the
device memory never run and the working set grows with the loop length. Explicit `release` cannot
reach those buffers: a pure carrier has no IO point at which to call it.

A scoped arena fixes this. `arenaEnter` opens an allocation epoch; every device buffer allocated while
it is open is tracked. `arenaExit keep` frees the device data of *all* of them — reachable or not —
except the `keep` results, which survive: promoted to an enclosing arena if there is one, or left to
ordinary reference-counted finalization at the outermost level. This matches a training loop's natural
phase boundary (one LM step / one fold).

**Contract.** Every buffer that must outlive the scope MUST appear in `keep` (or be reachable only
through a kept buffer). Touching any other in-scope buffer after `arenaExit` is a use-after-free, the
same hazard as reading a buffer after `release`.
-/

@[extern "torchlean_cuda_arena_enter"]
opaque arenaEnter : IO Unit

@[extern "torchlean_cuda_arena_exit"]
opaque arenaExit (keep : @& Array Buffer) : IO Unit

/--
Run `body` inside a fresh device-memory arena, then reclaim every buffer it allocated except those
named by `keep result`.

`keep` extracts the buffers that must survive the scope (typically the step's result parameters). If
`body` raises, the arena is still closed and every in-scope buffer is reclaimed before the exception
propagates.
-/
def withCudaArena {α : Type} (keep : α → Array Buffer) (body : IO α) : IO α := do
  arenaEnter
  try
    let r ← body
    arenaExit (keep r)
    return r
  catch e =>
    arenaExit #[]
    throw e

/-- Allocate a length-`n` buffer filled with zeros. -/
@[extern "torchlean_cuda_buffer_zeros"]
opaque zeros (n : UInt32) : Buffer

/-- Allocate a length-`n` buffer filled with `v` (host `Float`, cast to float32). -/
@[extern "torchlean_cuda_buffer_full"]
opaque full (n : UInt32) (v : Float) : Buffer

/-!
### Deterministic RNG (device-side)

These are low-level building blocks used by TorchLean's seeded RNG ops (`rand_uniform`,
`bernoulli_mask`) when running on the eager CUDA backend.

They use the same SplitMix64-style mixing as `TorchLean.Random` so results are
deterministic given `(seed, counter)` and a row-major linear index.
-/

/-- Deterministic `U[0,1)` generator: returns a length-`n` buffer (float32) keyed by `key`. -/
@[extern "torchlean_cuda_buffer_rand_uniform"]
opaque randUniform (n : UInt32) (key : UInt64) : Buffer

/-- Deterministic `{0,1}` mask generator: returns a length-`n` buffer keyed by `key`. -/
@[extern "torchlean_cuda_buffer_bernoulli_mask"]
opaque bernoulliMask (n : UInt32) (keepProb : Float) (key : UInt64) : Buffer

/-- Absolute value applied pointwise to a CUDA buffer. -/
@[extern "torchlean_cuda_buffer_abs"]
opaque abs (b : Buffer) : Buffer

/-- Backward for `abs`: `dx = sign(x) * dLdy` (with `sign(0)=0`). -/
@[extern "torchlean_cuda_buffer_abs_bwd"]
opaque absBwd (x dLdy : Buffer) : Buffer

@[extern "torchlean_cuda_buffer_sqrt"]
opaque sqrt (b : Buffer) : Buffer

/--
Backward for `sqrt`.

Uses the TorchLean convention: `dx = dLdy * (1 / (2*sqrt(x)))` for `x > 0`, else `0`.
-/
@[extern "torchlean_cuda_buffer_sqrt_bwd"]
opaque sqrtBwd (x dLdy : Buffer) : Buffer

@[extern "torchlean_cuda_buffer_exp"]
opaque exp (b : Buffer) : Buffer

@[extern "torchlean_cuda_buffer_log"]
opaque log (b : Buffer) : Buffer

/-- Reciprocal: `1/x`. -/
@[extern "torchlean_cuda_buffer_inv"]
opaque inv (b : Buffer) : Buffer

/-- Clamp each element to `[lo, hi]` (bounds are host `Float`s). -/
@[extern "torchlean_cuda_buffer_clamp"]
opaque clamp (b : Buffer) (lo hi : Float) : Buffer

/--
Backward for `clamp`.

Uses the TorchLean convention: derivative is `1` strictly inside `(lo, hi)`, else `0`.
-/
@[extern "torchlean_cuda_buffer_clamp_bwd"]
opaque clampBwd (x dLdy : Buffer) (lo hi : Float) : Buffer

/-- Pointwise maximum of two equal-length CUDA buffers. -/
@[extern "torchlean_cuda_buffer_max"]
opaque max (a b : Buffer) : Buffer

/--
Backward for `max`, returning `(dA, dB)`.

Tie-breaking follows the spec: when `a = b`, split upstream gradient evenly (`0.5`) between both.
-/
@[extern "torchlean_cuda_buffer_max_bwd"]
opaque maxBwd (a b dLdy : Buffer) : Buffer × Buffer

@[extern "torchlean_cuda_buffer_min"]
opaque min (a b : Buffer) : Buffer

/--
Backward for `min`, returning `(dA, dB)`.

Tie-breaking follows the spec: when `a = b`, split upstream gradient evenly (`0.5`) between both.
-/
@[extern "torchlean_cuda_buffer_min_bwd"]
opaque minBwd (a b dLdy : Buffer) : Buffer × Buffer

/-- Pointwise division of two equal-length CUDA buffers. -/
@[extern "torchlean_cuda_buffer_div"]
opaque div (a b : Buffer) : Buffer

/-- Pointwise ReLU activation on a CUDA buffer. -/
@[extern "torchlean_cuda_buffer_relu"]
opaque relu (b : Buffer) : Buffer

/-- Backward for `relu`: `dx = dLdy` where `x > 0`, else `0`. -/
@[extern "torchlean_cuda_buffer_relu_bwd"]
opaque reluBwd (x dLdy : Buffer) : Buffer

/-- Elementwise addition (sizes must match). -/
@[extern "torchlean_cuda_buffer_add"]
opaque add (a b : Buffer) : Buffer

/-- Elementwise subtraction (sizes must match). -/
@[extern "torchlean_cuda_buffer_sub"]
opaque sub (a b : Buffer) : Buffer

/-- Elementwise multiplication (sizes must match). -/
@[extern "torchlean_cuda_buffer_mul"]
opaque mul (a b : Buffer) : Buffer

/--
Multiply each element by a scalar `c` (host `Float`, cast to float32).

This is a primitive building block for many ops (e.g. scaling gradients).
-/
@[extern "torchlean_cuda_buffer_scale"]
opaque scale (b : Buffer) (c : Float) : Buffer

/-- Device-to-device copy, implemented as a scale-by-one kernel. -/
def copy (b : Buffer) : Buffer :=
  scale b 1.0

/--
Fused multiply-add: `a + c * b` (sizes must match; `c` is a host `Float`, cast to float32).

This is the classic BLAS-style `axpy` primitive and is useful for optimizers and bias-like updates.
-/
@[extern "torchlean_cuda_buffer_axpy"]
opaque axpy (a b : Buffer) (c : Float) : Buffer

/-- Reductions (return a length-1 buffer). -/
@[extern "torchlean_cuda_buffer_reduce_sum"]
opaque reduceSum (b : Buffer) : Buffer

@[extern "torchlean_cuda_buffer_reduce_mean"]
opaque reduceMean (b : Buffer) : Buffer

end Buffer

end Cuda
end Autograd
end Runtime
