/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.JAX.Import

/-!
# JAX interop

Import static Jaxpr computations into the existing TorchLean graph and payload APIs.
The Python adapter is `NN/Runtime/JAX/jaxpr.py`; JAX is needed only for capture.
-/

@[expose] public section
