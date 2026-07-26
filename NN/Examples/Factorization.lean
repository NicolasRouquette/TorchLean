/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Factorization.Common
public import NN.Examples.Factorization.Cholesky
public import NN.Examples.Factorization.QR

/-!
# Matrix-factorization examples (Cholesky and QR)

Executable `#eval` witnesses for the exact finite factorizations: Cholesky $A=LL^\mathsf{T}$ and QR
$A=QR$ (with $Q^\mathsf{T}Q=I$). Each pairs a positive reconstruction/orthonormality check with a
negative control,
over `Float`, sorry/admit-free.
-/
