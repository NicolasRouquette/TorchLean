/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Capsule

/-!
# Backend Availability

Machine- and build-dependent availability for backend capsules.

`KernelPolicy` says which capsules selection may use. `Availability` says what this checkout or
machine can provide. CPU-only, CUDA, and optional LibTorch builds therefore share one semantic
registry while exposing different capsule subsets to the planner.
-/

@[expose] public section

namespace NN
namespace Backend

/-- Runtime/build capabilities visible to backend planning. -/
structure Availability where
  devices : Array Device := #[.cpu]
  providers : Array Provider := #[.reference, .torchLean]
  deriving Repr

namespace Availability

/-- Whether a device is available on this machine/build. -/
def hasDevice (a : Availability) (d : Device) : Bool :=
  a.devices.contains d

/-- Whether a provider is available. -/
def hasProvider (a : Availability) (p : Provider) : Bool :=
  a.providers.contains p

/-- Whether a capsule can even be considered on this machine/build. -/
def admitsCapsule (a : Availability) (c : KernelCapsule) : Bool :=
  a.hasDevice c.device && a.hasProvider c.provider

/-- Keep only capsules available on this machine/build. -/
def filterCapsules (a : Availability) (capsules : Array KernelCapsule) : Array KernelCapsule :=
  capsules.filter fun c => a.admitsCapsule c

/-- CPU/reference-only availability. -/
def cpu : Availability :=
  { devices := #[.cpu]
    providers := #[.reference, .torchLean] }

/-- CUDA availability without optional external providers. -/
def cudaNative : Availability :=
  { devices := #[.cpu, .cuda]
    providers := #[.reference, .torchLean, .nativeCuda, .cuBLAS, .cuFFT] }

/-- CUDA availability with optional LibTorch enabled and installed. -/
def cudaWithLibTorch : Availability :=
  { devices := #[.cpu, .cuda]
    providers := #[.reference, .torchLean, .nativeCuda, .cuBLAS, .cuFFT, .libTorch] }

end Availability

end Backend
end NN
