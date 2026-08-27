/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# Backend Types

Small vocabulary for backend selection and trust boundaries.

TorchLean owns the spec, graph, and proof-facing contracts. Backends are execution providers for
parts of that graph: a Lean reference path, the TorchLean runtime, native CUDA kernels, LibTorch,
ATen, cuBLAS/cuDNN/cuFFT, TPU/XLA, AWS Neuron/Trainium, or future platform-specific providers. This
file deliberately contains only data. It should stay cheap to import from specs, runtime wrappers,
docs generators, and tests.
-/

@[expose] public section

namespace NN
namespace Backend

/-- Hardware or execution target visible to the planner. -/
inductive Device where
  | cpu
  | cuda
  | rocm
  | metal
  | wasm
  | tpu
  | trainium
  | custom
  | external
  deriving DecidableEq, Repr

namespace Device

/-- Stable spelling used in profile names, reports, and CLI bridges. -/
def cliName : Device → String
  | .cpu => "cpu"
  | .cuda => "cuda"
  | .rocm => "rocm"
  | .metal => "metal"
  | .wasm => "wasm"
  | .tpu => "tpu"
  | .trainium => "trainium"
  | .custom => "custom"
  | .external => "external"

/-- Parse an explicit backend device name. CLI layers may resolve policy names such as `auto`
before calling this function. -/
def parse? : String → Option Device
  | "cpu" => some .cpu
  | "cuda" => some .cuda
  | "rocm" => some .rocm
  | "metal" => some .metal
  | "wasm" => some .wasm
  | "tpu" => some .tpu
  | "trainium" => some .trainium
  | "custom" => some .custom
  | "external" => some .external
  | _ => none

/-- Parse an explicit backend device name or return a diagnostic suitable for command-line use. -/
def parse (value : String) : Except String Device :=
  match parse? value with
  | some device => pure device
  | none =>
      throw s!"unknown device {value} (known targets: cpu | cuda | rocm | metal | wasm | tpu | trainium | custom | external)"

end Device

/-- Concrete provider family used to execute a kernel capsule. -/
inductive Provider where
  | reference
  | torchLean
  | nativeCuda
  | libTorch
  | aten
  | mps
  | webGpu
  | cuBLAS
  | cuDNN
  | cuFFT
  | xla
  | neuron
  | customChip
  | external
  deriving DecidableEq, Repr

/-- Stable operation vocabulary used by backend capsules, graph planning, and runtime guards.

This is deliberately a closed vocabulary. New backend-visible operations should be added here and
then wired through the IR adapter and capsule registry. Runtime tape/debug labels may still be
strings, but the backend planner should not accept arbitrary stringly-typed operation names.
-/
inductive BackendOp where
  | randUniform
  | bernoulliMask
  | add
  | sub
  | mul
  | scale
  | abs
  | sqrt
  | clamp
  | max
  | min
  | relu
  | gelu
  | sigmoid
  | tanh
  | softplus
  | exp
  | log
  | sin
  | cos
  | inv
  | safeLog
  | logSoftmax
  | softmax
  | hardMaskedSoftmax
  | reduceSum
  | reduceMean
  | reshape
  | permute
  | broadcast
  | concat
  | slice
  | gather
  | scatterAdd
  | matmul
  | linear
  | mseLoss
  | layerNorm
  | batchNorm
  | conv
  | convTranspose
  | maxPool
  | smoothMaxPool
  | avgPool
  | fftFno
  | selectiveScan
  | scaledDotProductAttention
  deriving DecidableEq, BEq, Repr

/-- Broad semantic category used to summarize a backend-visible operation.

This classification is descriptive metadata. Backend support, numerical policy, trust, and
executable behavior remain properties of individual kernel capsules.
-/
inductive OpClass where
  | source
  | pointwise
  | reduction
  | accumulation
  | view
  | selection
  | composite
  deriving DecidableEq, Repr

/-- Backend-invariant metadata for one operation. -/
structure OpSchema where
  /-- Stable spelling used in reports, capsule names, and CLI diagnostics. -/
  name : String
  /-- Broad semantic category of the operation. -/
  opClass : OpClass
  /-- Whether differentiating through the operation requires a registered local VJP. -/
  requiresVJP : Bool
  deriving DecidableEq, Repr

namespace BackendOp

/-- Canonical backend-invariant metadata for an operation. -/
def schema : BackendOp → OpSchema
  | .randUniform => ⟨"rand_uniform", .source, false⟩
  | .bernoulliMask => ⟨"bernoulli_mask", .source, false⟩
  | .add => ⟨"add", .pointwise, true⟩
  | .sub => ⟨"sub", .pointwise, true⟩
  | .mul => ⟨"mul", .pointwise, true⟩
  | .scale => ⟨"scale", .pointwise, true⟩
  | .abs => ⟨"abs", .pointwise, true⟩
  | .sqrt => ⟨"sqrt", .pointwise, true⟩
  | .clamp => ⟨"clamp", .pointwise, true⟩
  | .max => ⟨"max", .pointwise, true⟩
  | .min => ⟨"min", .pointwise, true⟩
  | .relu => ⟨"relu", .pointwise, true⟩
  | .gelu => ⟨"gelu", .pointwise, true⟩
  | .sigmoid => ⟨"sigmoid", .pointwise, true⟩
  | .tanh => ⟨"tanh", .pointwise, true⟩
  | .softplus => ⟨"softplus", .pointwise, true⟩
  | .exp => ⟨"exp", .pointwise, true⟩
  | .log => ⟨"log", .pointwise, true⟩
  | .sin => ⟨"sin", .pointwise, true⟩
  | .cos => ⟨"cos", .pointwise, true⟩
  | .inv => ⟨"inv", .pointwise, true⟩
  | .safeLog => ⟨"safe_log", .pointwise, true⟩
  | .logSoftmax => ⟨"log_softmax", .accumulation, true⟩
  | .softmax => ⟨"softmax", .accumulation, true⟩
  | .hardMaskedSoftmax => ⟨"hard_masked_softmax", .accumulation, true⟩
  | .reduceSum => ⟨"reduce_sum", .reduction, true⟩
  | .reduceMean => ⟨"reduce_mean", .reduction, true⟩
  | .reshape => ⟨"reshape", .view, true⟩
  | .permute => ⟨"permute", .view, true⟩
  | .broadcast => ⟨"broadcast", .view, true⟩
  | .concat => ⟨"concat", .view, true⟩
  | .slice => ⟨"slice", .selection, true⟩
  | .gather => ⟨"gather", .selection, true⟩
  | .scatterAdd => ⟨"scatter_add", .accumulation, true⟩
  | .matmul => ⟨"matmul", .accumulation, true⟩
  | .linear => ⟨"linear", .accumulation, true⟩
  | .mseLoss => ⟨"mse_loss", .reduction, true⟩
  | .layerNorm => ⟨"layer_norm", .composite, true⟩
  | .batchNorm => ⟨"batch_norm", .composite, true⟩
  | .conv => ⟨"conv", .accumulation, true⟩
  | .convTranspose => ⟨"conv_transpose", .accumulation, true⟩
  | .maxPool => ⟨"max_pool", .selection, true⟩
  | .smoothMaxPool => ⟨"smooth_max_pool", .accumulation, true⟩
  | .avgPool => ⟨"avg_pool", .accumulation, true⟩
  | .fftFno => ⟨"fft_fno", .composite, true⟩
  | .selectiveScan => ⟨"selective_scan", .composite, true⟩
  | .scaledDotProductAttention => ⟨"scaled_dot_product_attention", .composite, true⟩

/-- Stable spelling used in reports, capsule names, and CLI diagnostics. -/
def name (op : BackendOp) : String := op.schema.name

instance : ToString BackendOp where
  toString op := op.name

/-- Whether training through this operation requires a registered local VJP.

Random sources create values but are not themselves differentiated. Every other backend-visible
operation must provide a compatible VJP whenever gradient tracking is requested.
-/
def requiresVJP (op : BackendOp) : Bool := op.schema.requiresVJP

end BackendOp

/--
How much TorchLean knows about an implementation.

`trustedExternal` is allowed, but it is intentionally loud: the contract names the boundary instead
of silently treating an industrial kernel as though Lean had verified its source.
-/
inductive TrustLevel where
  | verified
  | checked
  | fuzzed
  | trustedExternal
  deriving DecidableEq, Repr

/--
One policy for the complete assurance boundary of a kernel plan.

The first three fields control which implementation trust levels the planner may select. The
remaining fields control which kinds of evidence may discharge the selected capsule's shape,
layout, value, and VJP obligations. Keeping these decisions in one record prevents a profile from
selecting a capsule under one policy and auditing it under a contradictory second policy.
-/
structure AssurancePolicy where
  allowChecked : Bool := false
  allowFuzzed : Bool := false
  allowTrustedExternal : Bool := false
  requireEvidence : Bool := true
  allowRuntimeGuards : Bool := false
  allowTestEvidence : Bool := false
  deriving DecidableEq, Repr

namespace AssurancePolicy

/--
Proof-oriented policy reserved for typed, proof-carrying implementations.

No maintained runtime capsule currently satisfies this policy: guards, tests, fuzzing, and trusted
boundaries are all rejected. A future verified capsule must connect its implementation semantics to
the operation contract directly instead of attaching an arbitrary proposition as metadata.
-/
def verified : AssurancePolicy := {}

/--
Maintained TorchLean runtime policy.

Checked implementations, runtime guards, regression evidence, and fuzz evidence are accepted, but
trusted external implementations are not.
-/
def checked : AssurancePolicy :=
  { allowChecked := true
    allowFuzzed := true
    allowRuntimeGuards := true
    allowTestEvidence := true }

/--
Explicit external-provider policy.

This is the policy used when a caller deliberately delegates a numerical kernel to LibTorch or
another external implementation. The selected boundary remains visible in the execution audit.
-/
def external : AssurancePolicy :=
  { checked with allowTrustedExternal := true }

/-- Whether the policy admits a capsule with the given implementation trust level. -/
def acceptsTrust (policy : AssurancePolicy) : TrustLevel → Bool
  | .verified => true
  | .checked => policy.allowChecked
  | .fuzzed => policy.allowFuzzed
  | .trustedExternal => policy.allowTrustedExternal

end AssurancePolicy

/-- How a backend capsule treats gradients. -/
inductive VJPMode where
  | none
  | torchLeanTape
  | backendVJP
  deriving DecidableEq, Repr

/-- Provider preference used when selecting an implementation for an operation. -/
inductive ProviderPreference where
  | auto
  | prefer (provider : Provider)
  | only (provider : Provider)
  deriving DecidableEq, Repr

/-- Policy used to select kernel capsules for a device and assurance boundary. -/
structure KernelPolicy where
  device : Device := .cpu
  provider : ProviderPreference := .auto
  assurance : AssurancePolicy := .checked
  vjpMode : VJPMode := .torchLeanTape
  deriving Repr

end Backend
end NN
