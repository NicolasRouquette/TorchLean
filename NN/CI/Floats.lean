/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import NN.Floats
import NN.Floats.Arb
import NN.Floats.Interval.IEEEExec32ArbTrans
import NN.Proofs.RuntimeApprox.FP32

/-!
# Additional Floating-Point Modules

The public floating-point umbrella omits the optional Arb oracle and a few proof-heavy integration
modules. Ordinary CI imports them here so they cannot first fail during documentation generation.
-/

@[expose] public section
