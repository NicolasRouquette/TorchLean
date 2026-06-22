/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Factorization.Common
public import NN.Examples.Factorization.Cholesky
public import NN.Examples.Factorization.QR
public import NN.Examples.Factorization.SymEig
public import NN.Examples.Factorization.SVD
public import NN.Examples.Factorization.GenSymEig
public import NN.Examples.Factorization.JacobiDecrease
public import NN.Examples.Factorization.JacobiRate

/-!
# Matrix-factorization examples

Executable `#eval` witnesses for the verified factorization foundation: Cholesky, QR, the cyclic-Jacobi
symmetric eigendecomposition (with its per-rotation off-diagonal decrease and linear rate), SVD, and the
generalized symmetric eigenproblem solved by Cholesky whitening (`GenSymEig`). Each example pairs a
positive reconstruction/orthonormality check with a negative control, over `Float`,
sorry/admit-free.
-/
