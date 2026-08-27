/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Attention

/-!
# LibTorch Backend Capsules

Optional LibTorch-backed capsules.

These capsules are intentionally separated from the maintained registry. Adding the module means
accepting an external implementation boundary. The SDPA bridge uses TorchLean's hard boolean-mask
convention and supplies forward values only; TorchLean retains the tape and local backward rule.
-/

@[expose] public section

namespace NN
namespace Backend
namespace LibTorch

/-- Optional LibTorch capsules currently known to the planner. -/
def capsules : Array KernelCapsule :=
  #[Attention.libTorchSDPAForward]

end LibTorch
end Backend
end NN
