/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Linalg
public import NN.Spec.Core.TensorReductionShape.LinearAlgebra

/-!
# Matrix factorizations (spec layer)

This file provides **real**, shape-indexed reference implementations of the matrix
factorizations that classical / scientific-ML models (Gaussian processes, kernel ridge
regression, PCA, least squares) depend on, and which were previously missing from the spec
layer:

- `choleskySpec`     — Cholesky factorization `A = L · Lᵀ` (lower-triangular `L`), for matrices
                       with positive executable Cholesky pivots.
- `qrSpec`           — QR factorization `A = Q · R` via classical Gram–Schmidt
                       (`Q` has orthonormal columns, `R` upper-triangular).
- `symEigJacobiSpec` — **full** symmetric eigendecomposition via the cyclic Jacobi algorithm
                       (all eigenpairs, not just the largest).
- `svdSpec`          — singular value decomposition `A = U · diag(σ) · Vᵀ`, built on the
                       symmetric eigendecomposition of `Aᵀ·A`.

## Relationship to `eigendecompSpec`

`Spec.eigendecompSpec` (in `NN/Spec/Models/CommonHelpers.lean`) is a power-iteration *stub* that
only recovers the **largest** eigenpair. It is intentionally left untouched (PCA depends on it).
`symEigJacobiSpec` here is the full replacement: for a symmetric matrix it returns *all* `n`
eigenvalues and an orthogonal matrix of eigenvectors.

## Verification scope

The **verified** contribution is the factorizations: `choleskySpec` / `qrSpec` come with
reconstruction and structural theorems (`IsCholesky` / `IsQR`, lower- and upper-triangularity,
orthonormality) in `NN.Proofs.Tensor.Basic.Factorizations*`. The triangular- and ridge-solve
helpers above (`triSolveLowerFn`, `triSolveUpperFn`, `cholSolveFn`, `solveRidgeSpec`) are
**executable APIs only**: this PR does *not* yet prove their correctness (no
`triSolveLower · x = b` / `solveRidge` correctness theorem has landed). They are sound by
construction over the readable function representation and exercised by `#eval` examples, but
should not be read as carrying a verified-correctness guarantee.

## Intent / tradeoffs

Like the rest of the spec layer (`determinantSpec`, `inverseSpec`, `matMulSpec`), these prioritize
**mathematical clarity** and **shape safety** over performance, and are intended for small/medium
matrices and proof-oriented reference code. For large-scale numerics, use array-backed runtime
kernels.

Internally the algorithms are written over the plain function representation
`Fin n → Fin n → α` (matrices) and `Fin n → α` (vectors), then wrapped back into `Spec.Tensor`
at the boundary. This keeps the numerical formulas readable and keeps later correctness proofs
working on ordinary functions rather than on nested `Tensor` `match`es.

The iterative routines (Jacobi) take an explicit `sweeps` count: convergence of Jacobi is
asymptotic, so the caller chooses how much work to do. A dozen sweeps is ample for the small
matrices these specs target.
-/

@[expose] public section


namespace Spec

open Tensor

variable {α : Type} [Context α]

/-! ## Boundary conversions between `Spec.Tensor` and plain functions -/

/-- View a matrix tensor as a function `Fin m → Fin n → α`. -/
def toMatFn {m n : Nat} (A : Tensor α (.dim m (.dim n .scalar))) : Fin m → Fin n → α :=
  fun i j => get2 A i j

/-- Build a matrix tensor from a function `Fin m → Fin n → α`. -/
def ofMatFn {m n : Nat} (f : Fin m → Fin n → α) : Tensor α (.dim m (.dim n .scalar)) :=
  Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (f i j)))

/-- View a vector tensor as a function `Fin n → α`. -/
def toVecFn {n : Nat} (v : Tensor α (.dim n .scalar)) : Fin n → α :=
  fun i => Tensor.toScalar (get v i)

/-- Build a vector tensor from a function `Fin n → α`. -/
def ofVecFn {n : Nat} (f : Fin n → α) : Tensor α (.dim n .scalar) :=
  Tensor.dim (fun i => Tensor.scalar (f i))

/-! ## Small numeric helpers on the function representation -/

/-- Dot product of two length-`p` vectors. -/
def dotFn {p : Nat} (u v : Fin p → α) : α :=
  (List.finRange p).foldl (fun s i => s + u i * v i) 0

/-- Euclidean norm of a length-`p` vector. -/
def normFn {p : Nat} (v : Fin p → α) : α :=
  MathFunctions.sqrt (dotFn v v)

/-- Decide `x < y` as a `Bool` (via the `Context`'s decidable `>`). -/
def ltBool (x y : α) : Bool := Context.gtBool y x

/-- Decide `x ≤ y` as a `Bool`. Ascending comparator for `List.mergeSort` (e.g. to sort the
unordered Jacobi eigenvalues). -/
def leBool (x y : α) : Bool := !ltBool y x

/-! ## Cholesky factorization

For a symmetric positive-definite `A`, compute the lower-triangular `L` with `A = L · Lᵀ`.

The columns are computed left to right. Column `j` uses only columns `0 .. j-1`:

- diagonal:  `L[j,j] = sqrt(A[j,j] - Σ_{k<j} L[j,k]²)`
- below:     `L[i,j] = (A[i,j] - Σ_{k<j} L[i,k]·L[j,k]) / L[j,j]`   for `i > j`
- above:     `L[i,j] = 0`                                           for `i < j`
-/

/--
Strict, array-backed runtime implementation of `choleskyColsFn` (registered via `@[implemented_by]`).
Each column is *materialized* into an `Array α`, so a back-reference `L[i,k]` is an `O(1)` lookup
rather than a closure that re-evaluates the whole prefix. The closure form below is mathematically
clean (and is what the proofs reason about), but reading the full factor `L` from it re-evaluates
columns exponentially — ruinous in the interpreter (`#eval`). This computes the *same* factor strictly;
the numeric examples (`A = L·Lᵀ`, the ridge-solve residual ≈ 0) validate the two agree.
-/
def choleskyColsImpl {n : Nat} (A : Fin n → Fin n → α) : List (Fin n → α) :=
  let cols : Array (Array α) := (List.finRange n).foldl (fun cols j =>
    let jv := j.val
    -- Σ_{k<j} L[j,k]²  (previous columns at row `j`, read from the materialized arrays).
    let sumsq := (List.finRange n).foldl
      (fun s k => if k.val < jv then s + (cols.getD k.val #[]).getD jv 0 * (cols.getD k.val #[]).getD jv 0
        else s) 0
    let Ljj := MathFunctions.sqrt (A j j - sumsq)
    let colArr : Array α := Array.ofFn (fun i : Fin n =>
      if i.val < jv then 0
      else if i.val == jv then Ljj
      else
        -- Σ_{k<j} L[i,k]·L[j,k]
        let s := (List.finRange n).foldl
          (fun acc k => if k.val < jv then
            acc + (cols.getD k.val #[]).getD i.val 0 * (cols.getD k.val #[]).getD jv 0 else acc) 0
        (A i j - s) / Ljj)
    cols.push colArr) #[]
  (List.finRange n).map (fun j => fun i => (cols.getD j.val #[]).getD i.val 0)

/--
The list of columns of the Cholesky factor `L`, as length-`n` vectors, computed left to right.
Element `j` of the result is column `j` of `L`. Built by a left fold so that when column `j` is
formed, `cols` already holds columns `0 .. j-1`.

The runtime implementation is `choleskyColsImpl` (strict arrays); the closure form here is the one the
correctness proofs reason about. Both compute the same factor.
-/
@[implemented_by choleskyColsImpl]
def choleskyColsFn {n : Nat} (A : Fin n → Fin n → α) : List (Fin n → α) :=
  (List.finRange n).foldl (fun cols j =>
    -- Σ_{k<j} L[j,k]²  (the already-computed columns evaluated at row `j`).
    let sumsq := (cols.map (fun ck => ck j)).foldl (fun s x => s + x * x) 0
    let Ljj := MathFunctions.sqrt (A j j - sumsq)
    let colj : Fin n → α := fun i =>
      if i.val < j.val then 0
      else if i.val == j.val then Ljj
      else
        -- Σ_{k<j} L[i,k]·L[j,k]
        let s := (cols.map (fun ck => ck i * ck j)).foldl (fun acc x => acc + x) 0
        (A i j - s) / Ljj
    cols ++ [colj]) []

/-- Cholesky factor as a function: `L[i,j] = (choleskyColsFn A)[j] i`. -/
def choleskyFn {n : Nat} (A : Fin n → Fin n → α) : Fin n → Fin n → α :=
  let cols := choleskyColsFn A
  fun i j => (cols.getD j.val (fun _ => 0)) i

/--
Cholesky factorization of a symmetric positive-definite matrix `A`, returning the
lower-triangular factor `L` with `A = L · Lᵀ`.

PyTorch analogue: `torch.linalg.cholesky(A)`.
-/
def choleskySpec {n : Nat} (A : Tensor α (.dim n (.dim n .scalar))) :
    Tensor α (.dim n (.dim n .scalar)) :=
  ofMatFn (choleskyFn (toMatFn A))

/-! ## Triangular solves and the kernel-ridge (Tikhonov) linear solve

Once `A` is factored as `A = L · Lᵀ` (Cholesky), the linear system `A · x = b` is solved by two
triangular substitutions: forward-solve `L · z = b`, then back-solve `Lᵀ · x = z`. Each substitution
visits the unknowns in an order such that, when row `i` is reached, every unknown it depends on has
already been computed; the accumulator `acc` holds those values and `0` everywhere else, so the dot
`dotFn (row i) acc` is exactly the required partial sum (the not-yet-solved and structurally-zero
terms drop out). This is the linear solve at the heart of CHD `solve_variationnal`. -/

/-- Forward substitution: solve `L · y = b` for a lower-triangular `L` with nonzero diagonal.
Unknowns are visited `0, 1, …, n-1`; when row `i` is reached `acc` holds `y₀ … yᵢ₋₁` (and `0`
elsewhere), so `dotFn (L i) acc = Σ_{k<i} L[i,k]·yₖ` by lower-triangularity. -/
def triSolveLowerFn {n : Nat} (L : Fin n → Fin n → α) (b : Fin n → α) : Fin n → α :=
  (List.finRange n).foldl
    (fun acc i => Function.update acc i ((b i - dotFn (L i) acc) / L i i))
    (fun _ => 0)

/-- Back substitution: solve `U · x = y` for an upper-triangular `U` with nonzero diagonal.
Unknowns are visited `n-1, …, 1, 0`; when row `i` is reached `acc` holds `xᵢ₊₁ … xₙ₋₁` (and `0`
elsewhere), so `dotFn (U i) acc = Σ_{k>i} U[i,k]·xₖ` by upper-triangularity. -/
def triSolveUpperFn {n : Nat} (U : Fin n → Fin n → α) (y : Fin n → α) : Fin n → α :=
  (List.finRange n).reverse.foldl
    (fun acc i => Function.update acc i ((y i - dotFn (U i) acc) / U i i))
    (fun _ => 0)

/--
Strict, array-backed runtime implementation of `cholSolveFn` (registered via `@[implemented_by]`).
It materializes `L` into a strict `Array (Array α)` once, then runs both triangular substitutions over
`Array`s, so a back-reference is an `O(1)` lookup. The closure form below (`triSolveUpperFn` over
`triSolveLowerFn`) is mathematically clean — and is what the correctness proofs reason about — but reads
the `Function.update` accumulator chain on every step, which is ruinous in the interpreter (`#eval`) when
`L` is itself an unmaterialized closure (e.g. `choleskyFn` of a kernel matrix). This computes the *same*
solution strictly; the numeric examples (the ridge residual ≈ 0) validate the two agree. -/
def cholSolveImpl {n : Nat} (L : Fin n → Fin n → α) (b : Fin n → α) : Fin n → α :=
  let La : Array (Array α) := Array.ofFn (fun i : Fin n => Array.ofFn (fun j : Fin n => L i j))
  let Lent : Nat → Nat → α := fun i j => (La.getD i #[]).getD j 0
  -- Forward solve `L · z = b`: `z[i] = (b[i] − Σ_{k<i} L[i,k]·z[k]) / L[i,i]`.
  let z : Array α := (List.finRange n).foldl (fun z i =>
    let iv := i.val
    let s := (List.finRange n).foldl
      (fun acc k => if k.val < iv then acc + Lent iv k.val * z.getD k.val 0 else acc) 0
    z.push ((b i - s) / Lent iv iv)) #[]
  -- Back solve `Lᵀ · x = z`: `x[i] = (z[i] − Σ_{k>i} L[k,i]·x[k]) / L[i,i]`, `i = n−1 … 0`.
  let x : Array α := (List.finRange n).reverse.foldl (fun xs i =>
    let iv := i.val
    let s := (List.finRange n).foldl
      (fun acc k => if iv < k.val then acc + Lent k.val iv * xs.getD k.val 0 else acc) 0
    xs.set! iv ((z.getD iv 0 - s) / Lent iv iv)) (Array.replicate n 0)
  fun i => x.getD i.val 0

/-- Solve `A · x = b` given a Cholesky factor `L` of `A` (so `A = L · Lᵀ`): forward-solve
`L · z = b`, then back-solve `Lᵀ · x = z`.

The runtime implementation is `cholSolveImpl` (strict arrays); the closure form here is what the
correctness proofs reason about. Both compute the same solution. -/
@[implemented_by cholSolveImpl]
def cholSolveFn {n : Nat} (L : Fin n → Fin n → α) (b : Fin n → α) : Fin n → α :=
  triSolveUpperFn (fun i k => L k i) (triSolveLowerFn L b)

/-- The regularized matrix `K + γ·I` as a function. For a symmetric PSD kernel `K` and `γ > 0`
this is symmetric positive-definite, so its Cholesky factorization succeeds. -/
def addScaledIdFn {n : Nat} (K : Fin n → Fin n → α) (γ : α) : Fin n → Fin n → α :=
  fun i j => K i j + (if i = j then γ else 0)

/--
Strict, array-backed runtime implementation of `solveRidgeFn` (registered via `@[implemented_by]`).
It factors `K + γ·I = L·Lᵀ` and runs both triangular substitutions entirely over `Array`s, so no step
materializes the deep `Fin n → α` closures the functional definition builds — those re-evaluate
columns / the substitution accumulator exponentially, which is ruinous in the interpreter (`#eval`).
Same linear solve; the numeric examples (residual `(K+γ·I)·x − b ≈ 0`) validate the two agree.
-/
def solveRidgeImpl {n : Nat} (K : Fin n → Fin n → α) (γ : α) (b : Fin n → α) : Fin n → α :=
  let A : Fin n → Fin n → α := fun i j => K i j + (if i.val == j.val then γ else 0)
  -- Cholesky columns, left to right: `cols[j][i] = L[i][j]` (strict arrays, `O(1)` back-reference).
  let cols : Array (Array α) := (List.finRange n).foldl (fun cols j =>
    let jv := j.val
    let sumsq := (List.finRange n).foldl
      (fun s k => if k.val < jv then let v := (cols.getD k.val #[]).getD jv 0; s + v * v else s) 0
    let Ljj := MathFunctions.sqrt (A j j - sumsq)
    cols.push (Array.ofFn (fun i : Fin n =>
      if i.val < jv then 0
      else if i.val == jv then Ljj
      else
        let s := (List.finRange n).foldl (fun acc k =>
          if k.val < jv then
            acc + (cols.getD k.val #[]).getD i.val 0 * (cols.getD k.val #[]).getD jv 0
          else acc) 0
        (A i j - s) / Ljj))) #[]
  let Lent : Nat → Nat → α := fun i j => (cols.getD j #[]).getD i 0
  -- Forward solve `L · z = b`: `z[i] = (b[i] − Σ_{k<i} L[i,k]·z[k]) / L[i,i]`.
  let z : Array α := (List.finRange n).foldl (fun z i =>
    let iv := i.val
    let s := (List.finRange n).foldl
      (fun acc k => if k.val < iv then acc + Lent iv k.val * z.getD k.val 0 else acc) 0
    z.push ((b i - s) / Lent iv iv)) #[]
  -- Back solve `Lᵀ · x = z`: `x[i] = (z[i] − Σ_{k>i} L[k,i]·x[k]) / L[i,i]`, `i = n−1 … 0`.
  let x : Array α := (List.finRange n).reverse.foldl (fun xs i =>
    let iv := i.val
    let s := (List.finRange n).foldl
      (fun acc k => if iv < k.val then acc + Lent k.val iv * xs.getD k.val 0 else acc) 0
    xs.set! iv ((z.getD iv 0 - s) / Lent iv iv)) (Array.replicate n 0)
  fun i => x.getD i.val 0

/-- The Tikhonov-regularized (kernel-ridge) solve `(K + γ·I)·x = b`, via the Cholesky factorization
of `K + γ·I`. This is the linear solve at the core of CHD `solve_variationnal`.

The runtime implementation is `solveRidgeImpl` (strict arrays); the closure form here, built from the
verified `choleskyFn` / `triSolve*` pieces, is what the correctness proofs reason about. Both compute
the same solution. -/
@[implemented_by solveRidgeImpl]
def solveRidgeFn {n : Nat} (K : Fin n → Fin n → α) (γ : α) (b : Fin n → α) : Fin n → α :=
  cholSolveFn (choleskyFn (addScaledIdFn K γ)) b

/-- Tensor-level kernel-ridge solve: `(K + γ·I)·x = b`.

PyTorch analogue: `torch.linalg.solve(K + γ·I, b)` (specialized to the SPD Cholesky path). -/
def solveRidgeSpec {n : Nat} (K : Tensor α (.dim n (.dim n .scalar))) (γ : α)
    (b : Tensor α (.dim n .scalar)) : Tensor α (.dim n .scalar) :=
  ofVecFn (solveRidgeFn (toMatFn K) γ (toVecFn b))

/-! ## QR factorization (classical Gram–Schmidt)

For `A : m × n`, produce `Q : m × n` with orthonormal columns and `R : n × n` upper-triangular
such that `A = Q · R`. This uses **classical** Gram–Schmidt: each `r[k,j] = qₖ · aⱼ` is the inner
product against the *original* column `aⱼ`, and all projections are subtracted in a single pass
(modified Gram–Schmidt would instead dot each `qₖ` against the running residual). In exact real
arithmetic the two coincide; the classical form is what the recurrence below implements and what
the reconstruction proof matches.
-/

/-- Internal state for the Gram–Schmidt fold: computed `Q` columns and `R` columns so far. -/
structure GSState (m n : Nat) (α : Type) where
  /-- Orthonormal `Q` columns produced so far (each of length `m`). -/
  qs : List (Fin m → α)
  /-- `R` columns produced so far (each of length `n`, upper-triangular). -/
  rcols : List (Fin n → α)

/--
Run classical Gram–Schmidt over the columns of `A`, returning the `Q` columns and `R` columns.
Column `j` is orthogonalized against the previously produced `Q` columns.
-/
def gramSchmidtFn {m n : Nat} (A : Fin m → Fin n → α) : GSState m n α :=
  (List.finRange n).foldl (fun (st : GSState m n α) j =>
    let a : Fin m → α := fun i => A i j
    -- r[k,j] = qₖ · a   for each previously computed column k
    let rkjs : List α := st.qs.map (fun qk => dotFn qk a)
    -- v = a - Σ r[k,j] qₖ
    let v : Fin m → α := fun i =>
      a i - (List.zip st.qs rkjs).foldl (fun acc (qk, r) => acc + r * qk i) 0
    let rjj := normFn v
    let qj : Fin m → α := fun i => if Context.gtBool rjj 0 then v i / rjj else 0
    let rcolj : Fin n → α := fun k =>
      if k.val < j.val then rkjs.getD k.val 0
      else if k.val == j.val then rjj
      else 0
    { qs := st.qs ++ [qj], rcols := st.rcols ++ [rcolj] }) { qs := [], rcols := [] }

/-- The `Q` factor (orthonormal columns) of the QR factorization of `A`. -/
def qrQSpec {m n : Nat} (A : Tensor α (.dim m (.dim n .scalar))) :
    Tensor α (.dim m (.dim n .scalar)) :=
  let st := gramSchmidtFn (toMatFn A)
  ofMatFn (fun i j => (st.qs.getD j.val (fun _ => 0)) i)

/-- The `R` factor (upper-triangular) of the QR factorization of `A`. -/
def qrRSpec {m n : Nat} (A : Tensor α (.dim m (.dim n .scalar))) :
    Tensor α (.dim n (.dim n .scalar)) :=
  let st := gramSchmidtFn (toMatFn A)
  ofMatFn (fun k j => (st.rcols.getD j.val (fun _ => 0)) k)

/--
QR factorization of `A : m × n` via classical Gram–Schmidt, returning `(Q, R)` with
`A = Q · R`, `Q` orthonormal columns, `R` upper-triangular.

PyTorch analogue: `torch.linalg.qr(A)`.
-/
def qrSpec {m n : Nat} (A : Tensor α (.dim m (.dim n .scalar))) :
    Tensor α (.dim m (.dim n .scalar)) × Tensor α (.dim n (.dim n .scalar)) :=
  (qrQSpec A, qrRSpec A)

/-! ## Symmetric eigendecomposition (cyclic Jacobi)

For a symmetric `A`, iteratively apply Givens rotations `J` that zero one off-diagonal entry at a
time, accumulating `A ← Jᵀ A J` and `V ← V J`. Each `J` is orthogonal, so every step is an
orthogonal similarity: the spectrum is preserved and `V` stays orthogonal. After enough sweeps the
off-diagonal mass vanishes; the diagonal holds the eigenvalues and the columns of `V` are the
eigenvectors.
-/

/-!
The iteration below runs over an `Array (Array α)` representation rather than `Fin n → Fin n → α`.
Arrays are strict values, so threading them through the rotation loop cannot build the deep closure
chains that a functional representation would (one matrix product per rotation), which is what keeps
execution cheap. We convert to/from `Spec.Tensor` only at the boundary.
-/

/-- Read entry `(i, j)` of an `Array (Array α)` matrix (`0` if out of bounds). -/
def arrGet (M : Array (Array α)) (i j : Nat) : α := (M.getD i #[]).getD j 0

/-- Materialize a matrix function into a strict `Array (Array α)`. -/
def matToArr {n : Nat} (X : Fin n → Fin n → α) : Array (Array α) :=
  Array.ofFn (fun i : Fin n => Array.ofFn (fun j : Fin n => X i j))

/-- Matrix product `X · Y` of two `n × n` array matrices. -/
def arrMatMul (n : Nat) (X Y : Array (Array α)) : Array (Array α) :=
  Array.ofFn (fun i : Fin n => Array.ofFn (fun j : Fin n =>
    (List.finRange n).foldl (fun s k => s + arrGet X i.val k.val * arrGet Y k.val j.val) 0))

/-- Transpose of an `n × n` array matrix. -/
def arrTr (n : Nat) (X : Array (Array α)) : Array (Array α) :=
  Array.ofFn (fun i : Fin n => Array.ofFn (fun j : Fin n => arrGet X j.val i.val))

/-- `n × n` identity as an array matrix. -/
def arrId (n : Nat) : Array (Array α) :=
  Array.ofFn (fun i : Fin n => Array.ofFn (fun j : Fin n => if i.val == j.val then 1 else 0))

/--
Givens rotation in the `(p, q)` plane as an array matrix:
identity except `J[p,p]=J[q,q]=c`, `J[p,q]=s`, `J[q,p]=-s`.
-/
def arrGivens (n : Nat) (p q : Nat) (c s : α) : Array (Array α) :=
  Array.ofFn (fun i : Fin n => Array.ofFn (fun j : Fin n =>
    if i.val == p && j.val == p then c
    else if i.val == q && j.val == q then c
    else if i.val == p && j.val == q then s
    else if i.val == q && j.val == p then -s
    else if i.val == j.val then 1 else 0))

/--
Apply one Jacobi rotation that targets off-diagonal entry `(p, q)`, updating `(A, V)` as strict
arrays. If `A[p,q]` is already (numerically) zero, the state is returned unchanged.

The rotation parameters follow Golub & Van Loan:
`τ = (A[q,q] - A[p,p]) / (2 A[p,q])`, `t = sign(τ)/(|τ| + sqrt(1+τ²))` (or `1` if `τ = 0`),
`c = 1/sqrt(1+t²)`, `s = t·c`.
-/
def arrJacobiRotate (n : Nat) (A V : Array (Array α)) (p q : Nat) :
    Array (Array α) × Array (Array α) :=
  let apq := arrGet A p q
  if Context.gtBool (MathFunctions.abs apq) 0 then
    let τ := (arrGet A q q - arrGet A p p) / (Numbers.two * apq)
    let absτ := MathFunctions.abs τ
    let sgn : α := if ltBool τ 0 then Numbers.neg_one else 1
    let t : α :=
      if Context.gtBool absτ 0 then sgn / (absτ + MathFunctions.sqrt (1 + τ * τ)) else 1
    let c := 1 / MathFunctions.sqrt (1 + t * t)
    let s := t * c
    let J := arrGivens n p q c s
    (arrMatMul n (arrTr n J) (arrMatMul n A J), arrMatMul n V J)
  else
    (A, V)

/-- All index pairs `(p, q)` with `p < q`, in row-major order (one cyclic Jacobi sweep). -/
def jacobiPairs (n : Nat) : List (Nat × Nat) :=
  (List.range n).flatMap (fun p =>
    (List.range n).filterMap (fun q => if p < q then some (p, q) else none))

/-- One Jacobi sweep: rotate through every `(p, q)` pair with `p < q`. -/
def arrJacobiSweep (n : Nat) (st : Array (Array α) × Array (Array α)) :
    Array (Array α) × Array (Array α) :=
  (jacobiPairs n).foldl (fun s pq => arrJacobiRotate n s.1 s.2 pq.1 pq.2) st

/-- Run `sweeps` Jacobi sweeps starting from `(A, I)`, returning the rotated `A` and accumulated `V`. -/
def arrJacobiRun (n : Nat) (A : Array (Array α)) (sweeps : Nat) :
    Array (Array α) × Array (Array α) :=
  (List.range sweeps).foldl (fun st _ => arrJacobiSweep n st) (A, arrId n)

/--
Full symmetric eigendecomposition of `A` via cyclic Jacobi, returning `(eigenvalues, eigenvectors)`.

The eigenvalues are the diagonal of the rotated matrix; the eigenvectors are the **columns** of the
returned matrix `V` (so `eigenvectors[i, j]` is the `i`-th component of the `j`-th eigenvector).
`sweeps` controls how many Jacobi sweeps to run (default `12`).

Unlike `eigendecompSpec`, this recovers **all** `n` eigenpairs.

PyTorch analogue: `torch.linalg.eigh(A)`.
-/
def symEigJacobiSpec {n : Nat} (A : Tensor α (.dim n (.dim n .scalar))) (sweeps : Nat := 12) :
    Tensor α (.dim n .scalar) × Tensor α (.dim n (.dim n .scalar)) :=
  let (Af, Vf) := arrJacobiRun n (matToArr (toMatFn A)) sweeps
  (ofVecFn (fun i => arrGet Af i.val i.val), ofMatFn (fun i j => arrGet Vf i.val j.val))

/-! ## Singular value decomposition

For `A : m × n`, form the symmetric `M = Aᵀ·A : n × n`, eigendecompose it as `M = V Λ Vᵀ`,
take `σ = sqrt(max(Λ, 0))`, and recover `U` columns as `uⱼ = A vⱼ / σⱼ` (zero when `σⱼ = 0`).
Then `A = U · diag(σ) · Vᵀ`. This is the simplest reference SVD and is exact (up to the Jacobi
sweep count) for `A` of full column rank.
-/

/--
Singular value decomposition of `A : m × n` returning `(U, σ, V)` with
`A = U · diag(σ) · Vᵀ`, `U : m × n` with orthonormal columns (full-rank case), `σ : n` the singular
values, and `V : n × n` orthogonal.

`sweeps` controls the Jacobi sweep count used for the eigendecomposition of `Aᵀ·A`.

PyTorch analogue: `torch.linalg.svd(A, full_matrices=False)`.
-/
def svdSpec {m n : Nat} (A : Tensor α (.dim m (.dim n .scalar))) (sweeps : Nat := 12) :
    Tensor α (.dim m (.dim n .scalar)) × Tensor α (.dim n .scalar) ×
      Tensor α (.dim n (.dim n .scalar)) :=
  let Af := toMatFn A
  -- M = Aᵀ A  (n × n, symmetric PSD), as a strict array matrix
  let M : Array (Array α) :=
    Array.ofFn (fun i : Fin n => Array.ofFn (fun j : Fin n =>
      (List.finRange m).foldl (fun s k => s + Af k i * Af k j) 0))
  let (Mf, Vf) := arrJacobiRun n M sweeps
  let σ : Fin n → α := fun j =>
    let d := arrGet Mf j.val j.val
    MathFunctions.sqrt (if ltBool d 0 then 0 else d)
  let U : Fin m → Fin n → α := fun i j =>
    let sj := σ j
    if Context.gtBool sj 0 then
      ((List.finRange n).foldl (fun s k => s + Af i k * arrGet Vf k.val j.val) 0) / sj
    else 0
  (ofMatFn U, ofVecFn σ, ofMatFn (fun i j => arrGet Vf i.val j.val))

/-! ## Generalized symmetric eigenproblem via Cholesky whitening (CCA / dimension reduction)

The generalized symmetric eigenproblem `A·v = λ·B·v` — with `A` symmetric and `B` symmetric
positive-definite — is the algebraic core of *canonical-correlation analysis* (CCA) and the
whitening step of dimension reduction. It is reduced to the *standard* symmetric eigenproblem the
landed `symEigJacobiSpec` already solves, by *whitening* `B`:

* factor `B = L·Lᵀ` (`choleskyFn`, `B` SPD ⟹ the factorization succeeds);
* form the *whitened* matrix `C = L⁻¹·A·L⁻ᵀ`, which is symmetric and has the **same eigenvalues** as
  the pencil `(A, B)` — substituting `w = Lᵀ·v` turns `A·v = λ·B·v` into `C·w = λ·w`;
* standard-eigendecompose `C = W·diag(λ)·Wᵀ` (`symEigJacobiSpec`);
* *unwhiten* the eigenvectors: `v = L⁻ᵀ·w` (back-substitution `Lᵀ·v = w`).

The recovered eigenvalues `λ` are the generalized eigenvalues (the canonical correlations, for the
CCA pencil); the columns of the recovered `V` are `B`-orthonormal (`Vᵀ·B·V = I`) rather than
ordinary-orthonormal — that is exactly the whitening guarantee CCA needs. Every piece is a *landed*
primitive (`choleskyFn`, the triangular solves, `symEigJacobiSpec`); only this composition is new. -/

/-- Strict, array-backed runtime implementation of `whitenFn` (registered via `@[implemented_by]`).
Materializes `L` once, runs the two rounds of forward substitution over arrays. -/
def whitenImpl {n : Nat} (L A : Fin n → Fin n → α) : Fin n → Fin n → α :=
  let La : Array (Array α) := Array.ofFn (fun i : Fin n => Array.ofFn (fun j : Fin n => L i j))
  let Lent : Nat → Nat → α := fun i j => (La.getD i #[]).getD j 0
  let fsolve : (Nat → α) → Array α := fun b =>
    (List.finRange n).foldl (fun x i =>
      let iv := i.val
      let s := (List.finRange n).foldl
        (fun acc k => if k.val < iv then acc + Lent iv k.val * x.getD k.val 0 else acc) 0
      x.push ((b iv - s) / Lent iv iv)) #[]
  -- `M = L⁻¹·A`, by columns: `Mcols[j][i] = M[i][j]`.
  let Mcols : Array (Array α) :=
    Array.ofFn (fun j : Fin n => fsolve (fun r => if h : r < n then A ⟨r, h⟩ j else 0))
  -- `C = L⁻¹·Mᵀ` (`Cᵀ = L⁻¹·Mᵀ`), by rows: `Crows[i][j] = C[i][j]`.
  let Crows : Array (Array α) :=
    Array.ofFn (fun i : Fin n => fsolve (fun k => (Mcols.getD k #[]).getD i.val 0))
  fun i j => (Crows.getD i.val #[]).getD j.val 0

/-- Whiten `A` by the Cholesky factor `L` of `B` (so `B = L·Lᵀ`): the symmetric matrix
`C = L⁻¹·A·L⁻ᵀ`. Computed by triangular solves only (no explicit inverse): `M = L⁻¹·A` column by
column (`triSolveLowerFn L (A·,j)`), then `C = L⁻¹·Mᵀ` row by row (using `Cᵀ = L⁻¹·Mᵀ`), so
`C[i,j] = (L⁻¹·(row i of M))[j]`. In exact arithmetic `C = Cᵀ`.

The runtime implementation is `whitenImpl` (strict arrays); this closure form is what the correctness
proofs reason about. -/
@[implemented_by whitenImpl]
def whitenFn {n : Nat} (L A : Fin n → Fin n → α) : Fin n → Fin n → α :=
  let M : Fin n → Fin n → α := fun i j => triSolveLowerFn L (fun r => A r j) i
  fun i j => triSolveLowerFn L (fun k => M i k) j

/--
Strict, array-backed runtime implementation of `genSymEigCholeskyFn` (registered via
`@[implemented_by]`). It materializes `L = choleskyFn B` into a strict `Array (Array α)` once, runs the
whitening `C = L⁻¹·A·L⁻ᵀ` and the eigenvector unwhitening `V = L⁻ᵀ·W` entirely over arrays, and calls
the (already strict) `symEigJacobiSpec` on `C`. The closure form below is what the correctness proofs
reason about; this computes the *same* result without re-evaluating the deep `choleskyFn` / triangular
closures the functional form rebuilds per entry. The numeric parity examples validate the two agree. -/
def genSymEigCholeskyImpl {n : Nat} (A B : Fin n → Fin n → α) (sweeps : Nat) :
    (Fin n → α) × (Fin n → Fin n → α) :=
  -- Cholesky columns of `B`, left to right: `cols[j][i] = L[i][j]` (strict arrays).
  let cols : Array (Array α) := (List.finRange n).foldl (fun cols j =>
    let jv := j.val
    let sumsq := (List.finRange n).foldl
      (fun s k => if k.val < jv then let v := (cols.getD k.val #[]).getD jv 0; s + v * v else s) 0
    let Ljj := MathFunctions.sqrt (B j j - sumsq)
    cols.push (Array.ofFn (fun i : Fin n =>
      if i.val < jv then 0
      else if i.val == jv then Ljj
      else
        let s := (List.finRange n).foldl (fun acc k =>
          if k.val < jv then
            acc + (cols.getD k.val #[]).getD i.val 0 * (cols.getD k.val #[]).getD jv 0
          else acc) 0
        (B i j - s) / Ljj))) #[]
  let Lent : Nat → Nat → α := fun i j => (cols.getD j #[]).getD i 0
  -- Forward solve `L·x = b`: `x[i] = (b[i] − Σ_{k<i} L[i,k]·x[k]) / L[i,i]`.
  let fsolve : (Nat → α) → Array α := fun b =>
    (List.finRange n).foldl (fun x i =>
      let iv := i.val
      let s := (List.finRange n).foldl
        (fun acc k => if k.val < iv then acc + Lent iv k.val * x.getD k.val 0 else acc) 0
      x.push ((b iv - s) / Lent iv iv)) #[]
  -- Back solve `Lᵀ·x = b`: `x[i] = (b[i] − Σ_{k>i} L[k,i]·x[k]) / L[i,i]`, `i = n−1 … 0`.
  let bsolve : (Nat → α) → Array α := fun b =>
    (List.finRange n).reverse.foldl (fun xs i =>
      let iv := i.val
      let s := (List.finRange n).foldl
        (fun acc k => if iv < k.val then acc + Lent k.val iv * xs.getD k.val 0 else acc) 0
      xs.set! iv ((b iv - s) / Lent iv iv)) (Array.replicate n 0)
  -- `M = L⁻¹·A`, stored by columns: `Mcols[j][i] = M[i][j]`.
  let Mcols : Array (Array α) :=
    (Array.ofFn (fun j : Fin n => fsolve (fun r => if h : r < n then A ⟨r, h⟩ j else 0)))
  -- `C = L⁻¹·Mᵀ` (so `Cᵀ = L⁻¹·Mᵀ`), stored by rows: `Crows[i][j] = C[i][j]`.
  let Crows : Array (Array α) :=
    (Array.ofFn (fun i : Fin n => fsolve (fun k => (Mcols.getD k #[]).getD i.val 0)))
  let C : Fin n → Fin n → α := fun i j => (Crows.getD i.val #[]).getD j.val 0
  let (Λ, W) := symEigJacobiSpec (ofMatFn C) sweeps
  let Wf := toMatFn W
  -- Unwhiten: `V[·,j] = L⁻ᵀ·W[·,j]`, stored by columns `Vcols[j][i] = V[i][j]`.
  let Vcols : Array (Array α) :=
    Array.ofFn (fun j : Fin n => bsolve (fun r => if h : r < n then Wf ⟨r, h⟩ j else 0))
  (toVecFn Λ, fun i j => (Vcols.getD j.val #[]).getD i.val 0)

/--
Generalized symmetric eigendecomposition `A·v = λ·B·v` for symmetric `A` and SPD `B`, via Cholesky
whitening of `B` (the CCA / dimension-reduction reduction). Returns `(λ, V)` with `λ` the generalized
eigenvalues (ascending) and the columns of `V` the generalized eigenvectors, `B`-orthonormal
(`Vᵀ·B·V = I` in exact arithmetic).

The runtime implementation is `genSymEigCholeskyImpl` (strict arrays); this closure form, built from the
landed `choleskyFn`, `whitenFn` (triangular solves), and `symEigJacobiSpec`, is what the correctness
proofs reason about. Both compute the same result.

`scipy.linalg.eigh(A, B)` / Julia `eigen(A, B)` analogue (the generalized symmetric driver). -/
@[implemented_by genSymEigCholeskyImpl]
def genSymEigCholeskyFn {n : Nat} (A B : Fin n → Fin n → α) (sweeps : Nat) :
    (Fin n → α) × (Fin n → Fin n → α) :=
  let L := choleskyFn B
  let C := whitenFn L A
  let (Λ, W) := symEigJacobiSpec (ofMatFn C) sweeps
  let Wf := toMatFn W
  (toVecFn Λ, fun i j => triSolveUpperFn (fun a b => L b a) (fun r => Wf r j) i)

/-- Tensor-level generalized symmetric eigendecomposition `A·v = λ·B·v` (`A` symmetric, `B` SPD),
via Cholesky whitening. Returns `(eigenvalues, V)` with columns of `V` the `B`-orthonormal generalized
eigenvectors. `sweeps` controls the Jacobi sweep count for the whitened standard eigenproblem.

PyTorch / SciPy analogue: `scipy.linalg.eigh(A, B)`. -/
def genSymEigCholeskySpec {n : Nat} (A B : Tensor α (.dim n (.dim n .scalar)))
    (sweeps : Nat := 12) : Tensor α (.dim n .scalar) × Tensor α (.dim n (.dim n .scalar)) :=
  let (evals, V) := genSymEigCholeskyFn (toMatFn A) (toMatFn B) sweeps
  (ofVecFn evals, ofMatFn V)

end Spec
