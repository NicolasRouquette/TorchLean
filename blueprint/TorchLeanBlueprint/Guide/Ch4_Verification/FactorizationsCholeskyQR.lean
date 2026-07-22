import VersoManual

open Verso.Genre Manual

#doc (Manual) "Matrix Factorizations: Cholesky and QR" =>
%%%
tag := "factorizations-cholesky-qr"
%%%

Matrix factorization is a good test of what “the algorithm is proved” should mean. A numerical
routine may return two arrays of the expected sizes and still be wrong: a Cholesky factor could
contain entries above the diagonal, a QR routine could reconstruct the input with a non-orthogonal
`Q`, or a zero pivot could be hidden behind a NaN. Shape safety catches none of these errors.

TorchLean therefore gives each factorization two parts:

- a program that constructs finite tensor values;
- a proposition describing the algebraic object that the program must return.

For Cholesky, the proposition is

$$`\operatorname{IsCholesky}(A,L)
\;:\!\iff\;
\bigl(\forall i<j,\;L_{ij}=0\bigr)
\land A=LL^\top.`

For a rectangular `m × k` matrix, the QR proposition is

$$`\operatorname{IsQR}(A,Q,R)
\;:\!\iff\;
Q^\top Q=I
\land\bigl(\forall j<i,\;R_{ij}=0\bigr)
\land A=QR.`

These are the literal definitions in
[`NN.Proofs.Tensor.Basic.Factorizations`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Tensor/Basic/Factorizations.lean):

```
def IsCholesky (A L : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  (∀ i j, i < j → L i j = 0) ∧ A = L * Lᵀ

def IsQR (A Q : Matrix (Fin m) (Fin k) ℝ)
    (R : Matrix (Fin k) (Fin k) ℝ) : Prop :=
  Qᵀ * Q = 1 ∧
  (∀ i j, j < i → R i j = 0) ∧
  A = Q * R
```

Notice what is not present: there is no residual tolerance and no phrase such as “approximately
orthogonal.” These predicates live over `ℝ`, and their equalities are exact.

# Cholesky, One Column At A Time

For a symmetric matrix, Cholesky computes a lower-triangular `L` using

$$`L_{jj}
=\sqrt{A_{jj}-\sum_{k<j}L_{jk}^{\,2}},`

and, below the diagonal,

$$`L_{ij}
=\frac{A_{ij}-\sum_{k<j}L_{ik}L_{jk}}{L_{jj}}
\qquad(i>j).`

Entries with `i < j` are set to zero. In the executable specification,
[`choleskyColsFn`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/Core/Tensor/Factorizations.lean)
is a left fold that appends one column at a time; `choleskyFn` reads the resulting columns as a
matrix; and `choleskySpec` wraps that function as a shaped tensor.

The fold representation matters in the proof. Once column `j` is appended, later iterations never
change it. Generic “fold that appends” lemmas make that invariant reusable instead of reproving list
indexing at every matrix entry.

The first structural theorem is unconditional:

```
theorem choleskyFn_lower_triangular
    (A : Fin n → Fin n → ℝ)
    {i j : Fin n} (hij : i.val < j.val) :
  choleskyFn A i j = 0
```

Reconstruction needs two hypotheses:

```
theorem isCholesky_of_pos
    (A : Fin n → Fin n → ℝ)
    (hsymm : ∀ i j, A i j = A j i)
    (hpos : ∀ j, 0 < choleskyFn A j j) :
  IsCholesky (Matrix.of A) (Matrix.of (choleskyFn A))
```

The positive-pivot condition is the algorithm’s success condition. It permits the division by
`L[j,j]` and identifies the positive square root. A symmetric positive-definite matrix is the
standard sufficient condition, but the current theorem does not prove

$$`\operatorname{PosDef}(A)
\Longrightarrow
\forall j,\;0<L_{jj}.`

It assumes the executable pivots are positive directly. This is an important unfinished bridge,
not a reason to describe the theorem as “Cholesky correctness for SPD matrices” without
qualification.

At tensor level, `choleskySpec_reconstruction` states the entrywise identity

$$`A_{ij}=\sum_k L_{ik}L_{jk}`

for `L = choleskySpec A`, under symmetry and positive tensor pivots.

# Run The Cholesky Witness

The checked example factors

$$`A=
\begin{pmatrix}
4&2&2\\
2&5&3\\
2&3&6
\end{pmatrix}.`

From the repository root, run:

```
lake env lean NN/Examples/Factorization/Cholesky.lean
```

The command is silent on success because the file uses guarded compiled assertions. To inspect the
two residuals explicitly, use this scratch file:

```
import NN.Examples.Factorization

#eval NN.Examples.Factorization.Cholesky.reconErr
#eval NN.Examples.Factorization.Cholesky.reconErrBad
```

The current output is:

```
0.000000
NaN
```

The second value comes from the symmetric but indefinite matrix

$$`\begin{pmatrix}1&2\\2&1\end{pmatrix},`

whose eigenvalues are `3` and `-1`. A diagonal step asks for the square root of a negative number.
The example intentionally uses a summed Frobenius error for this negative control because IEEE
`max` can ignore a NaN operand. That detail is part of the test’s meaning: even a diagnostic norm
must choose NaN behavior deliberately.

# Classical Gram–Schmidt As QR

For columns `a₀, …, aₖ₋₁`, classical Gram–Schmidt computes

$$`\begin{aligned}
v_j &= a_j-\sum_{i<j}\langle q_i,a_j\rangle q_i,\\
r_{jj} &= \|v_j\|,\\
q_j &= v_j/r_{jj},\\
r_{ij} &= \langle q_i,a_j\rangle\quad(i<j).
\end{aligned}`

The TorchLean specification uses the same column-building pattern as Cholesky. `gramSchmidtFn`
threads lists of `Q` and `R` columns, while `qrQSpec`, `qrRSpec`, and `qrSpec` expose tensor-shaped
results.

Three separate theorems correspond to the three parts of `IsQR`:

- `Rmat_upper_triangular` proves entries below the diagonal vanish;
- `qr_mul_eq` proves `A = Q * R`;
- `QT_mul_Q_eq_one` proves `Qᵀ * Q = 1`.

The packaged theorem
[`isQR_of_pos`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Tensor/Basic/FactorizationsOrthonormal.lean)
requires

$$`\forall j,\;0<R_{jj}.`

For classical Gram–Schmidt this is the executable form of full column rank: every new column has a
nonzero component orthogonal to its predecessors. The current API again states the pivot condition
directly rather than deriving it from a separately formalized rank predicate.

At the tensor boundary:

```
theorem qrSpec_reconstruction
    (A : Spec.Tensor ℝ (.dim m (.dim n .scalar)))
    (hrank : ∀ j, 0 < Spec.get2 (Spec.qrRSpec A) j j)
    (i : Fin m) (j : Fin n) :
  Spec.get2 A i j =
    ∑ k,
      Spec.get2 (Spec.qrQSpec A) i k *
      Spec.get2 (Spec.qrRSpec A) k j

theorem qrSpec_orthonormal
    (A : Spec.Tensor ℝ (.dim m (.dim n .scalar)))
    (hrank : ∀ j, 0 < Spec.get2 (Spec.qrRSpec A) j j)
    (a b : Fin n) :
  (∑ i,
    Spec.get2 (Spec.qrQSpec A) i a *
    Spec.get2 (Spec.qrQSpec A) i b) =
      if a = b then 1 else 0
```

# Run QR And Break Its Rank Assumption

The QR example uses the classical matrix

$$`A=
\begin{pmatrix}
12&-51&4\\
6&167&-68\\
-4&24&-41
\end{pmatrix}.`

Run the guarded file:

```
lake env lean NN/Examples/Factorization/QR.lean
```

Or inspect all four diagnostics:

```
import NN.Examples.Factorization

#eval NN.Examples.Factorization.QR.reconErr
#eval NN.Examples.Factorization.QR.orthoErr
#eval NN.Examples.Factorization.QR.reconErrDef
#eval NN.Examples.Factorization.QR.orthoErrDef
```

The present implementation prints:

```
0.000000
0.000000
0.000000
1.000000
```

The last two lines use a rank-deficient matrix whose second column is twice its first. The
executable algorithm still reconstructs this particular input, but one `Q` column is zero, so
`QᵀQ` differs from the identity by one on the diagonal. This is a useful distinction:
reconstruction observed for one rank-deficient example does not discharge the positive-pivot
hypothesis of the general theorem.

As another variation, duplicate any column of the good matrix. The reconstruction diagnostic may
remain small, while the orthonormality diagnostic must fail. If a proof attempt uses
`qrSpec_orthonormal`, Lean asks for the missing positive pivot rather than accepting the numerical
residual.

# Exact Proofs And Floating Execution

There are three objects in play:

:::table +header
*
  * Object
  * Scalar
  * Guarantee
*
  * `IsCholesky`, `IsQR`
  * `ℝ`
  * exact algebraic specification
*
  * `choleskySpec`, `qrQSpec`, `qrRSpec` in the proofs
  * `ℝ`
  * exact reconstruction under pivot hypotheses
*
  * factorization examples
  * `Float`
  * executable residual checks on concrete matrices
:::

The `Float` output is evidence that the executable definitions behave as expected on those inputs.
It is not the proof of `A = LLᵀ` or `QᵀQ = I`; machine arithmetic cannot generally satisfy those
identities exactly. Conversely, the real theorem does not prove a forward-error or backward-error
bound for the Float execution.

The strict-array `@[implemented_by]` paths used for faster evaluation are another boundary. The
proof definitions are clean finite functions and folds. A replacement implementation can make
evaluation practical, but its equality to the proof definition must be established separately or
listed as trusted runtime code.

# What Remains

The exact reconstruction developments are substantial, but they are not a complete verified
numerical linear algebra package. The most useful next theorems are:

1. positive definiteness implies positive executable Cholesky pivots;
2. full column rank implies positive executable Gram–Schmidt pivots;
3. correctness of the triangular and ridge-solve helpers;
4. finite-precision stability bounds, especially because classical Gram–Schmidt is less stable
   than modified Gram–Schmidt or Householder QR.

The exact identities follow the standard mathematics in Golub and Van Loan’s *Matrix
Computations*. The distinction between exact factorization and floating-point stability follows
Higham’s *Accuracy and Stability of Numerical Algorithms*. TorchLean currently proves the former
for its real specifications and tests concrete instances of the latter; it does not conflate the
two.
