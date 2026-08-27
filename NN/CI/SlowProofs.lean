/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import NN.Runtime.Autograd.IRExec.Correctness.SemanticEquivalence

/-!
# Slow Proof CI Target

This target elaborates the end-to-end typed graph IR semantic-equivalence theorem. The documentation
workflow typechecks it before DocGen; locally, use `lake build NNSlowProofs`.
-/

@[expose] public section
