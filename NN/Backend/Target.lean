/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Availability

/-!
# Backend Targets

Cross-platform target descriptions for backend planning.

`KernelPolicy` states the requested device, provider preference, VJP mode, and assurance level.
`Availability` says which devices and providers a planner may consider. `Target` describes declared
platform/build capabilities: CPU-only Linux, CUDA Linux,
Apple Metal, TPU/XLA, AWS Trainium/Neuron, WASM, or a caller-supplied accelerator all map into the
same capsule planner. A target declaration is not runtime discovery; executable paths must still
probe the linked runtime before launching work.
-/

@[expose] public section

namespace NN
namespace Backend

/-- Operating-system family relevant to backend packaging and dynamic-library loading. -/
inductive OperatingSystem where
  | linux
  | macOS
  | windows
  | wasi
  | unknown
  deriving DecidableEq, Repr

/-- Machine architecture relevant to native extension availability. -/
inductive Architecture where
  | x86_64
  | aarch64
  | wasm32
  | unknown
  deriving DecidableEq, Repr

/-- Primary accelerator family for a build or machine. -/
inductive Accelerator where
  | none
  | cuda
  | rocm
  | metal
  | wasm
  | tpu
  | trainium
  | custom
  | external
  deriving DecidableEq, Repr

/-- Optional backend feature compiled into or discoverable from a build. -/
inductive BuildFeature where
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
  | externalProvider
  deriving DecidableEq, Repr

/-- Platform/build description used to produce backend availability. -/
structure Target where
  os : OperatingSystem := .unknown
  arch : Architecture := .unknown
  accelerator : Accelerator := .none
  features : Array BuildFeature := #[]
  deriving Repr

namespace BuildFeature

/-- Provider exposed by a build feature, if it corresponds to one. -/
def provider? : BuildFeature → Option Provider
  | .nativeCuda => some .nativeCuda
  | .libTorch => some .libTorch
  | .aten => some .aten
  | .mps => some .mps
  | .webGpu => some .webGpu
  | .cuBLAS => some .cuBLAS
  | .cuDNN => some .cuDNN
  | .cuFFT => some .cuFFT
  | .xla => some .xla
  | .neuron => some .neuron
  | .customChip => some .customChip
  | .externalProvider => some .external

end BuildFeature

namespace Accelerator

/-- Device exposed by an accelerator target. CPU is added separately by `Target.devices`. -/
def device? : Accelerator → Option Device
  | .none => Option.none
  | .cuda => some .cuda
  | .rocm => some .rocm
  | .metal => some .metal
  | .wasm => some .wasm
  | .tpu => some .tpu
  | .trainium => some .trainium
  | .custom => some .custom
  | .external => some .external

end Accelerator

namespace Target

/-- Device set exposed by the target. CPU/reference remains available unless a caller filters it. -/
def devices (t : Target) : Array Device :=
  match t.accelerator.device? with
  | none => #[.cpu]
  | some d => #[.cpu, d]

/-- Providers exposed by the target and its compiled/discovered features. -/
def providers (t : Target) : Array Provider :=
  #[.reference, .torchLean] ++ t.features.filterMap BuildFeature.provider?

/-- Convert a target declaration to planner capabilities; this does not probe the current machine. -/
def declaredAvailability (t : Target) : Availability :=
  { devices := t.devices
    providers := t.providers }

/-- Portable CPU/reference target. -/
def portableCpu (os : OperatingSystem := .unknown)
    (arch : Architecture := .unknown) : Target :=
  { os, arch, accelerator := .none, features := #[] }

/-- Linux CPU/reference target. -/
def linuxCpu (arch : Architecture := .x86_64) : Target :=
  portableCpu .linux arch

/-- macOS CPU/reference target. -/
def macOSCpu (arch : Architecture := .aarch64) : Target :=
  portableCpu .macOS arch

/-- Windows CPU/reference target. -/
def windowsCpu (arch : Architecture := .x86_64) : Target :=
  portableCpu .windows arch

/-- Linux CUDA target with native CUDA provider features. -/
def linuxCuda (enableLibTorch : Bool := false) : Target :=
  { os := .linux
    arch := .x86_64
    accelerator := .cuda
    features :=
      #[.nativeCuda, .cuBLAS, .cuFFT] ++
        (if enableLibTorch then #[.libTorch] else #[]) }

/-- Windows CUDA target shape. Native support depends on the future Windows build/test pass. -/
def windowsCuda (enableLibTorch : Bool := false) : Target :=
  { os := .windows
    arch := .x86_64
    accelerator := .cuda
    features :=
      #[.nativeCuda, .cuBLAS, .cuFFT] ++
        (if enableLibTorch then #[.libTorch] else #[]) }

/-- Linux ROCm target shape for future AMD/HIP capsules. -/
def linuxRocm : Target :=
  { os := .linux
    arch := .x86_64
    accelerator := .rocm
    features := #[] }

/-- macOS target for future Metal/MPS capsules. -/
def macOSMetal : Target :=
  { os := .macOS
    arch := .aarch64
    accelerator := .metal
    features := #[.mps] }

/-- WASM target for browser or WASI-style execution. -/
def wasm : Target :=
  { os := .wasi
    arch := .wasm32
    accelerator := .wasm
    features := #[.webGpu] }

/-- TPU/XLA target shape for future accelerator capsules. -/
def linuxTpu : Target :=
  { os := .linux
    arch := .x86_64
    accelerator := .tpu
    features := #[.xla] }

/-- AWS Trainium/Neuron target shape for future accelerator capsules. -/
def linuxTrainium : Target :=
  { os := .linux
    arch := .x86_64
    accelerator := .trainium
    features := #[.neuron] }

/-- Target shape for a first-party or lab-specific accelerator with its own capsule provider. -/
def customChip (os : OperatingSystem := .linux)
    (arch : Architecture := .unknown) : Target :=
  { os, arch, accelerator := .custom, features := #[.customChip] }

/-- Target shape for a caller-supplied accelerator provider. -/
def external (os : OperatingSystem := .unknown)
    (arch : Architecture := .unknown) : Target :=
  { os, arch, accelerator := .external, features := #[.externalProvider] }

end Target

end Backend
end NN
