import VersoManual

open Verso.Genre Manual

#doc (Manual) "Matrix Factorizations: Cholesky and QR" =>
%%%
tag := "factorizations-cholesky-qr"
%%%

Classical and scientific-ML models — Gaussian processes, kernel-ridge regression, PCA, least squares —
rest on a handful of matrix factorizations that were missing from the spec layer. This chapter adds the
two *finite, exact* ones: the Cholesky factorization `A = L·Lᵀ` of an SPD matrix and the QR
factorization `A = Q·R` by classical Gram–Schmidt. Both are *direct* algorithms — no iteration, no
convergence caveat — so their correctness is an *exact identity*, not an a-posteriori bound.

# Specifications

Each factorization is given a `Prop`-level meaning over real matrices, independent of any particular
algorithm:

- `IsCholesky A L` — `L` is lower-triangular and `A = L·Lᵀ`;
- `IsQR A Q R` — `Q` has orthonormal columns (`Qᵀ·Q = 1`), `R` is upper-triangular, and `A = Q·R`.

The executable specs `choleskySpec` / `qrSpec` (over the readable `Fin n → Fin n → α` function
representation, wrapped back into `Spec.Tensor` at the boundary) are then proved to *produce* objects
satisfying these predicates.

# Exact Cholesky reconstruction

`choleskyFn` builds `L` one column at a time by a left fold. Two structural facts are proved directly
from that fold. First, *lower-triangularity*: the entry strictly above the diagonal is forced to `0` by
construction (`choleskyFn_lower_triangular`, lifted to the tensor level as
`choleskySpec_lower_triangular`). Second, *reconstruction*: under the success condition that every pivot
is positive — which holds for an SPD `A`, discharged from `Matrix.PosDef` — the fold satisfies

$$`A = L\,L^{\top},`

i.e. `IsCholesky A (choleskyFn A)`. This is exact over `ℝ`; the only hypothesis is positivity of the
pivots, which is exactly the SPD success condition.

# Exact QR reconstruction and orthonormality

`qrSpec` runs classical Gram–Schmidt. Bridging the executable column fold to Mathlib's `gramSchmidt`
gives the two QR guarantees, both exact under a full-column-rank hypothesis (each `R` diagonal entry
positive):

$$`Q^{\top} Q = 1 \qquad\text{and}\qquad A = Q\,R, \quad R \text{ upper-triangular}.`

Orthonormality (`qrSpec_orthonormal` / `QT_mul_Q_eq_one`) follows because each Gram–Schmidt column is the
normalization of a vector orthogonal to the span of its predecessors; reconstruction follows by
re-expanding that span. Together they give `IsQR A Q R`.

# Honest scope

Everything in this chapter is an *exact finite identity*: there is no sweep count, no residual, no
asymptotic limit. Three layers are kept distinct, and only the first is a verified claim:

- *Proved specs* (over `ℝ`): the predicates `IsCholesky` / `IsQR`, together with reconstruction,
  triangularity, and orthonormality, each derived from the executable column fold.
- *Executable examples* (over `Float`): concrete witnesses with residual checks — evidence that the
  definitions run and reconstruct, not proofs about floating-point arithmetic.
- *Trusted runtime hooks*: the strict-array `@[implemented_by]` replacements are runtime substitutions
  used for fast evaluation; they are *not* proved equal to the clean proof definitions.

The formal Cholesky hypothesis is *positivity of the executable pivots*, `∀ j, 0 < choleskyFn A j j` —
*not* SPD. `Matrix.PosDef A` is the expected sufficient condition for those pivots to be positive, but
the implication `PosDef A → ∀ j, 0 < choleskyFn A j j` is *not* formalized in this PR. QR likewise
assumes positive executable `R`-pivots (full column rank), not a separately proved rank hypothesis. Under
those pivot hypotheses no goal in the chapter is left unproved. The triangular- and ridge-solve helpers
that ride on the Cholesky factor are shipped as executable APIs only; their correctness theorems are not
part of this PR.

# Executable witnesses

`NN.Examples.Factorization.Cholesky` and `…QR` exhibit each factorization on a concrete matrix: a
positive reconstruction check (`‖A − L·Lᵀ‖`, `‖A − Q·R‖`, `‖Qᵀ·Q − I‖` all at machine zero) paired with a
negative control, every check a `#eval` over `Float`, with no unproved goals, green on
`lake build NN.Examples.Factorization`.
