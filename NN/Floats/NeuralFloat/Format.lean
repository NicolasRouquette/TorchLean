/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Format.Digits
public import NN.Floats.NeuralFloat.Format.Theorems

/-!
# Generic Floating-Point Formats

This module collects the radix, magnitude, exponent-function, and representability theory used by
TorchLean's real-valued floating-point model.  The organization follows the generic-format layer of
Flocq, while the declarations and proofs here are native Lean.

The umbrella deliberately excludes rounding modes and error bounds.  A client that only needs to
state that a value belongs to a format should not need to import the rest of numerical analysis.

## References

- S. Boldo and G. Melquiond, "Flocq: A Unified Library for Proving Floating-Point Algorithms in
  Coq," *IEEE ARITH*, 2011, doi:10.1109/ARITH.2011.40.
- IEEE, *IEEE Standard for Floating-Point Arithmetic*, IEEE 754-2019.
-/

@[expose] public section
