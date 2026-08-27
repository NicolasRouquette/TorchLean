/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Basic.Folds

/-!
Linear-algebra facts for dependent tensors.

The results here cover dot products, matrix-vector structure, and linearity facts used by
autograd, runtime approximation, and model proofs.
-/

@[expose] public section

namespace Spec

open Tensor
open scoped BigOperators

/-- `sum_spec` on a 1D tensor equals the `Finset` sum of its coordinates (`getScalar`). -/
lemma sum_spec_vec {n : Nat} (v : Tensor ℝ [n]) :
  sumSpec v = ∑ i : Fin n, getScalar v i := by
  classical
  cases v with
  | dim values =>
      -- `sum_spec_dim` reduces a vector sum to a sum of scalar `sum_spec`.
      have h :=
        (sum_spec_dim (t := (Tensor.dim values : Tensor ℝ [n])) (s := .scalar))
      -- Turn each scalar `sum_spec` into the corresponding coordinate.
      refine h.trans ?_
      refine Finset.sum_congr rfl ?_
      intro i _
      cases hval : values i with
      | scalar x =>
          simp [get_eq, getScalar, sumSpec, tensorFoldlSpec, hval]

-- Pointwise product of vectors under `getScalar`.
/-- `getScalar` of `mul_spec` is pointwise multiplication of coordinate functions. -/
lemma getScalar_mul_spec {n : Nat} (a b : Tensor ℝ [n]) (i : Fin n) :
  getScalar (mulSpec a b) i = getScalar a i * getScalar b i := by
  cases a with
  | dim fa =>
    cases b with
    | dim fb =>
      cases ha : fa i with
      | scalar x =>
        cases hb : fb i with
        | scalar y =>
          simp [mulSpec, map2Spec, getScalar, ha, hb]

-- Dot product of vectors as a `Finset` sum over coordinates.
/-- Dot product of vectors is the coordinate-wise sum `∑ i, a[i] * b[i]`. -/
lemma dot_vec_eq_sum {n : Nat} (a b : Tensor ℝ [n]) :
  dot a b = ∑ i : Fin n, getScalar a i * getScalar b i := by
  calc
    dot a b = Proofs.TensorAlgebra.dot (α := ℝ) a b := by
      exact dot_eq_tensorAlgebra_dot (a := a) (b := b)
    _ = ∑ i : Fin n, getScalar a i * getScalar b i := by
      simpa using Proofs.TensorAlgebra.dot_vec_eq_sum (α := ℝ) (a := a) (b := b)

-- Converting the spec-level `List.finRange` fold for `vec_mat_mul_spec` into a `Finset.univ` sum.
/-- Coordinate formula for `vec_mat_mul_spec` as a `Finset` sum: `(v @ A)[j] = ∑ i, v[i] * A[i,j]`.
  -/
lemma getScalar_vec_mat_mul_spec {m n : Nat}
  (v : Tensor ℝ [m])
  (A : Tensor ℝ [m, n]) (j : Fin n) :
  getScalar (vecMatMulSpec v A) j = ∑ i : Fin m, (getScalar v i) * (get2 A i j) := by
  simpa using
    (Proofs.TensorAlgebra.getScalar_vec_mat_mul_spec (α := ℝ) (v := v) (A := A) (j := j))

/--
Adjointness of matrix-vector and vector-matrix multiplication under the `dot` product:
`⟪y, W x⟫ = ⟪y W, x⟫` (a.k.a. `⟪y, W x⟫ = ⟪Wᵀ y, x⟫` depending on conventions).

This is the algebraic heart of the linear-layer gradient rule.
-/
theorem dot_mat_linear_adjoint
  {inDim outDim : Nat}
  (W : Tensor ℝ [outDim, inDim])
  (dLdy : Tensor ℝ [outDim])
  (dx : Tensor ℝ [inDim]) :
  dot dLdy (matVecMulSpec W dx)
  = dot (vecMatMulSpec dLdy W) dx := by
  calc
    dot dLdy (matVecMulSpec W dx)
        = Proofs.TensorAlgebra.dot (α := ℝ) dLdy (matVecMulSpec W dx) := by
          exact dot_eq_tensorAlgebra_dot (a := dLdy) (b := matVecMulSpec W dx)
    _ = Proofs.TensorAlgebra.dot (α := ℝ) (vecMatMulSpec dLdy W) dx := by
          exact Proofs.TensorAlgebra.dot_mat_linear_adjoint (α := ℝ) (W := W) (dLdy := dLdy)
            (dx := dx)
    _ = dot (vecMatMulSpec dLdy W) dx := by
          exact (dot_eq_tensorAlgebra_dot (a := vecMatMulSpec dLdy W) (b := dx)).symm

/--
`shapeOf` recovers the shape already tracked in the tensor type.

This is a small bridge for proofs that move between value-level shape computations and type-indexed
tensor operations.
-/
theorem shapeOf_eq_shape {α : Type} {s : Shape} (t : Tensor α s) :
  shapeOf t = s := by
  induction s with
  | scalar =>
    match t with
    | Tensor.scalar _ => rfl
  | dim n s ih =>
    match t with
    | Tensor.dim f =>
      match n with
      | 0 => rfl
      | Nat.succ n' =>
        -- apply induction hypothesis to first element
        have h := ih (f ⟨0, Nat.zero_lt_succ n'⟩)
        simpa [shapeOf, h]


/-- Indexing the outer dimension of a tensor exposes a subtensor with the declared inner shape. -/
theorem get_preserves_inner_shape {n : Nat} {s : Shape}
  (t : Tensor ℝ (.dim n s)) (i : Fin n) :
  shapeOf (get t i) = s := by
  cases t with
  | dim f =>
    simp only [get]
    exact shapeOf_eq_shape (f i)

/-! ## Map and elementwise operation laws -/

/-- Functor identity law for `map_spec`: mapping `id` is a no-op. -/
theorem map_spec_id {s : Shape} (t : Tensor ℝ s) :
  mapSpec id t = t := by
  induction s with
  | scalar => cases t; rfl
  | dim n s ih =>
    cases t with | dim f =>
    simp [mapSpec]
    funext i
    exact ih (f i)

/-- Functor law for `map_spec`: mapping `g` then `f` equals mapping `f ∘ g`. -/
theorem map_spec_comp {s : Shape} (f g : ℝ → ℝ) (t : Tensor ℝ s) :
  mapSpec f (mapSpec g t) = mapSpec (f ∘ g) t := by
  induction s with
  | scalar => cases t; rfl
  | dim n s ih =>
    cases t with | dim h =>
    simp [mapSpec]
    funext i
    exact ih (h i)

/-- A scalar additivity law lifts pointwise through `map_spec` and `add_spec`. -/
theorem map_spec_add_distrib {s : Shape} (f : ℝ → ℝ) (a b : Tensor ℝ s)
  (h : ∀ x y, f (x + y) = f x + f y) :
  mapSpec f (addSpec a b) = addSpec (mapSpec f a) (mapSpec f b) := by
  induction s with
  | scalar =>
    cases a; cases b
    simp [mapSpec, addSpec, map2Spec]
    exact h _ _
  | dim n s ih =>
    cases a; cases b; rename_i fa fb
    simp [mapSpec, addSpec, map2Spec]
    funext i
    exact ih (fa i) (fb i)

/-- Commutativity transfer: if `f` is commutative, then `map2_spec f` is commutative on tensors. -/
theorem map2_spec_comm {s : Shape} (f : ℝ → ℝ → ℝ) (a b : Tensor ℝ s)
  (h : ∀ x y, f x y = f y x) :
  map2Spec f a b = map2Spec f b a := by
  induction s with
  | scalar => cases a; cases b; simp [map2Spec]; exact h _ _
  | dim n s ih =>
    cases a; cases b; rename_i fa fb
    simp [map2Spec]
    funext i
    exact ih (fa i) (fb i)

/-! ## Matrix and vector algebra -/

/-- Associativity of matrix-vector multiplication: `A (B x) = (A B) x`. -/
theorem mat_vec_assoc {m n p : Nat}
  (A : Tensor ℝ [m, n])
  (B : Tensor ℝ [n, p])
  (x : Tensor ℝ [p]) :
  matVecMulSpec A (matVecMulSpec B x) =
  matVecMulSpec (matMulSpec A B) x := by
  classical
  have hto :
      getScalar (matVecMulSpec A (matVecMulSpec B x)) =
        getScalar (matVecMulSpec (matMulSpec A B) x) := by
    funext i
    have hBx : ∀ k : Fin n,
        getScalar (matVecMulSpec B x) k = ∑ j : Fin p, (get2 B k j) * (getScalar x j) := by
      intro k
      simpa using (getScalar_mat_vec_mul_spec (A := B) (v := x) (i := k))

    -- Expand both sides into finite sums and use a Fubini-style swap.
    have h_expand :
        (∑ k : Fin n, (get2 A i k) * (∑ j : Fin p, (get2 B k j) * (getScalar x j))) =
          (∑ j : Fin p, (∑ k : Fin n, (get2 A i k) * (get2 B k j)) * (getScalar x j)) := by
      -- This is a finite-dimensional distributivity/commutation identity.
      -- We follow the standard pattern: expand, swap sums, factor.
      classical
      -- Expand `get2 A i k * (∑ j, ...)` into a double sum.
      have h1 :
          (∑ k : Fin n, (get2 A i k) * (∑ j : Fin p, (get2 B k j) * (getScalar x j))) =
            (∑ k : Fin n, ∑ j : Fin p, (get2 A i k) * ((get2 B k j) * (getScalar x j))) := by
        simp [Finset.mul_sum]
      -- Swap the order of summation.
      have h2 :
          (∑ k : Fin n, ∑ j : Fin p, (get2 A i k) * ((get2 B k j) * (getScalar x j))) =
            (∑ j : Fin p, ∑ k : Fin n, (get2 A i k) * ((get2 B k j) * (getScalar x j))) := by
        simpa using
          (Finset.sum_comm (s := (Finset.univ : Finset (Fin n))) (t := (Finset.univ : Finset (Fin
            p)))
            (f := fun k j => (get2 A i k) * ((get2 B k j) * (getScalar x j))))
      -- Factor `(getScalar x j)` out of the inner sum.
      have h3 :
          (∑ j : Fin p, ∑ k : Fin n, (get2 A i k) * ((get2 B k j) * (getScalar x j))) =
            (∑ j : Fin p, (∑ k : Fin n, (get2 A i k) * (get2 B k j)) * (getScalar x j)) := by
        refine Finset.sum_congr rfl ?_
        intro j _
        have h_reassoc :
            (∑ k : Fin n, (get2 A i k) * ((get2 B k j) * (getScalar x j))) =
              (∑ k : Fin n, ((get2 A i k) * (get2 B k j)) * (getScalar x j)) := by
          refine Finset.sum_congr rfl ?_
          intro k _
          simpa using (mul_assoc (get2 A i k) (get2 B k j) (getScalar x j)).symm
        have h_pull :
            (∑ k : Fin n, ((get2 A i k) * (get2 B k j)) * (getScalar x j)) =
              (∑ k : Fin n, (get2 A i k) * (get2 B k j)) * (getScalar x j) := by
          simp [Finset.sum_mul]
        exact h_reassoc.trans h_pull

      exact h1.trans (h2.trans h3)

    -- Turn the vector components into the needed sum forms, then apply `h_expand`.
    have lhs :
        getScalar (matVecMulSpec A (matVecMulSpec B x)) i =
          ∑ k : Fin n, (get2 A i k) * (∑ j : Fin p, (get2 B k j) * (getScalar x j)) := by
      -- start from `getScalar_mat_vec_mul_spec` and rewrite each inner component via `hBx`
      have hA :
          getScalar (matVecMulSpec A (matVecMulSpec B x)) i =
            ∑ k : Fin n, (get2 A i k) * (getScalar (matVecMulSpec B x) k) := by
        simpa using (getScalar_mat_vec_mul_spec (A := A) (v := matVecMulSpec B x) (i := i))
      -- rewrite `getScalar (mat_vec_mul_spec B x) k`
      classical
      refine hA.trans ?_
      refine Finset.sum_congr rfl ?_
      intro k _
      simp [hBx k]

    have rhs :
        getScalar (matVecMulSpec (matMulSpec A B) x) i =
          ∑ j : Fin p, (∑ k : Fin n, (get2 A i k) * (get2 B k j)) * (getScalar x j) := by
      -- rewrite the matrix multiplication entry via `get2_mat_mul_spec`
      have hR :
          getScalar (matVecMulSpec (matMulSpec A B) x) i =
            ∑ j : Fin p, (get2 (matMulSpec A B) i j) * (getScalar x j) := by
        simpa using (getScalar_mat_vec_mul_spec (A := matMulSpec A B) (v := x) (i := i))
      -- now rewrite `get2 (mat_mul_spec A B) i j`
      classical
      refine hR.trans ?_
      refine Finset.sum_congr rfl ?_
      intro j _
      simp [get2_mat_mul_spec]

    -- Combine.
    simpa [lhs, rhs] using h_expand

  -- Lift pointwise equality back to tensors via `ofFn`.
  have h := congrArg ofFn hto
  -- `ofFn (getScalar t) = t` for vectors.
  simpa [ofFn_getScalar] using h

/-- Coordinate rule for `matrix_transpose_spec`: `(Aᵀ)[i,j] = A[j,i]`. -/
lemma get2_matrix_transpose_spec {m n : Nat}
  (A : Tensor ℝ [m, n]) (i : Fin n) (j : Fin m) :
  get2 (swapAdjacentAxes A 0) i j = get2 A j i := by
  rw [swapAdjacentAxes_zero]
  cases A with
  | dim rows =>
    cases hrow : rows j with
    | dim cols =>
      cases hcol : cols i with
      | scalar value =>
        simp [get2_eq, hrow, hcol]

/-- Matrix extensionality: matrices are equal when all their entries are equal. -/
lemma matrix_ext {m n : Nat} {A B : Tensor ℝ [m, n]} :
  (∀ i : Fin m, ∀ j : Fin n, get2 A i j = get2 B i j) → A = B := by
  intro h
  cases A with
  | dim rowsA =>
    cases B with
    | dim rowsB =>
      apply congrArg Tensor.dim
      funext i

      -- Prove row equality via `getScalar` and then lift back with `ofFn`.
      have hto : getScalar (rowsA i) = getScalar (rowsB i) := by
        funext j
        cases hrowA : rowsA i with
        | dim colsA =>
          cases hrowB : rowsB i with
          | dim colsB =>
            cases hcolA : colsA j with
            | scalar a =>
              cases hcolB : colsB j with
              | scalar b =>
                have hij : get2 (Tensor.dim rowsA) i j = get2 (Tensor.dim rowsB) i j := h i j
                have hab : a = b := by
                  simpa [get2_eq, get_eq, hrowA, hrowB, hcolA, hcolB] using hij
                simp [getScalar, hcolA, hcolB, hab]

      have hrow := congrArg ofFn hto
      simpa [ofFn_getScalar] using hrow

/-- Matrix transpose is an involution. -/
theorem matrix_transpose_involution {m n : Nat}
    (A : Tensor ℝ [m, n]) :
    swapAdjacentAxes (swapAdjacentAxes A 0) 0 = A := by
  apply matrix_ext
  intro i j
  rw [get2_matrix_transpose_spec, get2_matrix_transpose_spec]

/-- Transpose of a product: `(A ⬝ B)ᵀ = Bᵀ ⬝ Aᵀ`. -/
theorem matrix_transpose_mul {m n p : Nat}
  (A : Tensor ℝ [m, n])
  (B : Tensor ℝ [n, p]) :
  swapAdjacentAxes (matMulSpec A B) 0 =
  matMulSpec (swapAdjacentAxes B 0) (swapAdjacentAxes A 0) := by
  classical
  -- Prove equality by `get2`-extensionality on matrix entries.
  apply matrix_ext
  intro j i
  -- Compare the `(j,i)` entry of both sides.
  calc
    get2 (swapAdjacentAxes (matMulSpec A B) 0) j i
        = get2 (matMulSpec A B) i j := by
            simpa using (get2_matrix_transpose_spec (A := matMulSpec A B) (i := j) (j := i))
    _ = ∑ k : Fin n, (get2 A i k) * (get2 B k j) := by
          simpa using (get2_mat_mul_spec (A := A) (B := B) (i := i) (j := j))
    _ = ∑ k : Fin n, (get2 B k j) * (get2 A i k) := by
          refine Finset.sum_congr rfl ?_
          intro k _
          simp [mul_comm]
    _ = ∑ k : Fin n, (get2 (swapAdjacentAxes B 0) j k) * (get2 (swapAdjacentAxes A 0) k i) :=
      by
          refine Finset.sum_congr rfl ?_
          intro k _
          simp [get2_matrix_transpose_spec]
    _ = get2 (matMulSpec (swapAdjacentAxes B 0) (swapAdjacentAxes A 0)) j i := by
          symm
          simpa using
            (get2_mat_mul_spec (A := swapAdjacentAxes B 0) (B := swapAdjacentAxes A 0) (i :=
              j) (j := i))

-- ---------------------------------------------------------------------------
-- Frobenius dot / matmul adjointness
-- ---------------------------------------------------------------------------

/-- Expand the matrix dot-product as a double sum over entries (Frobenius inner product). -/
lemma dot_mat_eq_sum {m n : Nat}
  (A B : Tensor ℝ [m, n]) :
  dot A B = ∑ i : Fin m, ∑ j : Fin n, (get2 A i j) * (get2 B i j) := by
  classical
  cases A with
  | dim rowsA =>
    cases B with
    | dim rowsB =>
      -- Unfold `dot` to `sum_spec (mul_spec ...)` and sum over the outer dimension.
      have hout :
          dot (Tensor.dim rowsA) (Tensor.dim rowsB)
            =
          ∑ i : Fin m, sumSpec (mulSpec (rowsA i) (rowsB i)) := by
        -- `mul_spec` is rowwise, and `sum_spec_dim` unfolds the outer fold.
        simpa [dot, mulSpec, map2Spec, get_eq] using
          (sum_spec_dim (t := mulSpec (Tensor.dim rowsA) (Tensor.dim rowsB)))
      -- Unfold each row sum as a coordinate sum.
      calc
        dot (Tensor.dim rowsA) (Tensor.dim rowsB)
            = ∑ i : Fin m, sumSpec (mulSpec (rowsA i) (rowsB i)) := hout
        _ = ∑ i : Fin m, ∑ j : Fin n,
              (get2 (Tensor.dim rowsA) i j) * (get2 (Tensor.dim rowsB) i j) := by
              refine Finset.sum_congr rfl ?_
              intro i _
              cases hA : rowsA i with
              | dim colsA =>
                cases hB : rowsB i with
                | dim colsB =>
                  -- Rowwise: reduce to the vector lemma `sum_spec_vec`.
                  have hsum :
                      sumSpec (mulSpec (Tensor.dim colsA) (Tensor.dim colsB))
                        =
                      ∑ j : Fin n, getScalar (mulSpec (Tensor.dim colsA) (Tensor.dim colsB)) j := by
                      simpa using (sum_spec_vec (v := mulSpec (Tensor.dim colsA) (Tensor.dim
                        colsB)))
                  -- Rewrite via `sum_spec_vec`, then compare summands coordinatewise.
                  rw [hsum]
                  refine Finset.sum_congr rfl ?_
                  intro j _
                  -- Everything is definitional on scalar entries.
                  cases hcolA : colsA j with
                  | scalar a =>
                    cases hcolB : colsB j with
                    | scalar b =>
                      simp [getScalar, mulSpec, map2Spec, get2_eq, get_eq, hA, hB, hcolA, hcolB]

/-- Right-adjointness of matrix multiplication under the Frobenius dot-product.

Informally: `⟪A ⬝ B, C⟫ = ⟪A, C ⬝ Bᵀ⟫`.
-/
theorem dot_mat_mul_right_adjoint
  {m n p : Nat}
  (A : Tensor ℝ [m, n])
  (B : Tensor ℝ [n, p])
  (C : Tensor ℝ [m, p]) :
  dot (matMulSpec A B) C = dot A (matMulSpec C (swapAdjacentAxes B 0)) := by
  classical
  -- Expand both sides into entry sums; then it's just rearranging a finite triple sum.
  -- LHS: ∑ i ∑ j (∑ k Aik*Bkj) * Cij
  -- RHS: ∑ i ∑ k Aik * (∑ j Cij*Bkj)
  rw [dot_mat_eq_sum (A := matMulSpec A B) (B := C)]
  rw [dot_mat_eq_sum (A := A) (B := matMulSpec C (swapAdjacentAxes B 0))]
  -- Rewrite matrix products and transpose entries.
  simp [get2_mat_mul_spec, get2_matrix_transpose_spec, Finset.mul_sum, Finset.sum_mul]
  -- The goal is now exactly `Finset.sum_comm` on the two inner indices.
  refine Finset.sum_congr rfl ?_
  intro i _
  simpa [mul_assoc, mul_left_comm, mul_comm] using
    (Finset.sum_comm (s := (Finset.univ : Finset (Fin p))) (t := (Finset.univ : Finset (Fin n)))
      (f := fun j k => get2 A i k * (get2 C i j * get2 B k j)))

/-- Transpose invariance of the Frobenius dot-product: `⟪Aᵀ, Bᵀ⟫ = ⟪A, B⟫`. -/
lemma dot_mat_transpose {m n : Nat}
  (A B : Tensor ℝ [m, n]) :
  dot (swapAdjacentAxes A 0) (swapAdjacentAxes B 0) = dot A B := by
  classical
  calc
    dot (swapAdjacentAxes A 0) (swapAdjacentAxes B 0) =
        ∑ i : Fin n, ∑ j : Fin m,
          get2 (swapAdjacentAxes A 0) i j * get2 (swapAdjacentAxes B 0) i j :=
      dot_mat_eq_sum _ _
    _ = ∑ i : Fin n, ∑ j : Fin m, get2 A j i * get2 B j i := by
      simp [get2_matrix_transpose_spec]
    _ = ∑ j : Fin m, ∑ i : Fin n, get2 A j i * get2 B j i := Finset.sum_comm
    _ = dot A B := (dot_mat_eq_sum A B).symm

/-- Left-adjointness of matrix multiplication under the Frobenius dot-product.

Informally: `⟪A ⬝ B, C⟫ = ⟪B, Aᵀ ⬝ C⟫`.
-/
theorem dot_mat_mul_left_adjoint
  {m n p : Nat}
  (A : Tensor ℝ [m, n])
  (B : Tensor ℝ [n, p])
  (C : Tensor ℝ [m, p]) :
  dot (matMulSpec A B) C = dot B (matMulSpec (swapAdjacentAxes A 0) C) := by
  classical
  -- Reduce to the right-adjoint lemma via transpose.
  -- ⟪A·B, C⟫ = ⟪(A·B)ᵀ, Cᵀ⟫ = ⟪Bᵀ·Aᵀ, Cᵀ⟫ = ⟪B, (Cᵀ·A)ᵀ⟫ = ⟪B, Aᵀ·C⟫.
  have htrans :=
    (dot_mat_transpose (m := m) (n := p) (A := matMulSpec A B) (B := C)).symm
  -- rewrite `(A·B)ᵀ`
  have hmulT :
      swapAdjacentAxes (matMulSpec A B) 0 =
        matMulSpec (swapAdjacentAxes B 0) (swapAdjacentAxes A 0) :=
    matrix_transpose_mul (A := A) (B := B)
  -- apply the right-adjoint lemma to `Bᵀ·Aᵀ` against `Cᵀ`
  have hadj :
      dot (matMulSpec (swapAdjacentAxes B 0) (swapAdjacentAxes A 0)) (swapAdjacentAxes
        C 0)
        =
      dot (swapAdjacentAxes B 0)
        (matMulSpec (swapAdjacentAxes C 0) (swapAdjacentAxes (swapAdjacentAxes A 0) 0))
          := by
    simpa using
      (dot_mat_mul_right_adjoint (A := swapAdjacentAxes B 0) (B := swapAdjacentAxes A 0)
        (C := swapAdjacentAxes C 0))
  -- simplify involutions and transpose the last dot back
  have hAinv : swapAdjacentAxes (swapAdjacentAxes A 0) 0 = A :=
    matrix_transpose_involution (A := A)
  have hCinv : swapAdjacentAxes (swapAdjacentAxes C 0) 0 = C :=
    matrix_transpose_involution (A := C)
  -- `dot (Bᵀ) D = dot B (Dᵀ)` for matching shapes.
  have hdot_swap :
      dot (swapAdjacentAxes B 0) (matMulSpec (swapAdjacentAxes C 0) A)
        =
      dot B (swapAdjacentAxes (matMulSpec (swapAdjacentAxes C 0) A) 0) := by
    -- Apply `dot_mat_transpose` to `B` and `((Cᵀ·A)ᵀ)`.
    have := dot_mat_transpose (m := n) (n := p)
      (A := B) (B := swapAdjacentAxes (matMulSpec (swapAdjacentAxes C 0) A) 0)
    -- Rewrite involutions.
    simpa [matrix_transpose_involution, hCinv] using this
  -- Finish by rewriting `transpose (Cᵀ·A) = Aᵀ·C`.
  calc
    dot (matMulSpec A B) C
        = dot (swapAdjacentAxes (matMulSpec A B) 0) (swapAdjacentAxes C 0) := htrans
    _ = dot (matMulSpec (swapAdjacentAxes B 0) (swapAdjacentAxes A 0))
      (swapAdjacentAxes C 0) := by
          simp [hmulT]
    _ = dot (swapAdjacentAxes B 0) (matMulSpec (swapAdjacentAxes C 0) A) := by
          simpa [hAinv] using hadj
    _ = dot B (swapAdjacentAxes (matMulSpec (swapAdjacentAxes C 0) A) 0) := hdot_swap
    _ = dot B (matMulSpec (swapAdjacentAxes A 0) C) := by
          -- `transpose (Cᵀ·A) = Aᵀ·C`
          simp [matrix_transpose_mul, hCinv]

/--
Outer product properties.
Essential for proving weight gradient correctness.
-/
theorem outer_product_transpose {m n : Nat}
  (a : Tensor ℝ [m])
  (b : Tensor ℝ [n]) :
  swapAdjacentAxes (outerProductSpec a b) 0 = outerProductSpec b a := by
  rw [swapAdjacentAxes_zero]
  cases a with | dim fa =>
  cases b with | dim fb =>
  simp only [outerProductSpec]
  -- extensionality on the outer/inner indices
  apply congrArg Tensor.dim
  funext i
  apply congrArg Tensor.dim
  funext j
  cases hfa : fa j with
  | scalar x =>
    cases hfb : fb i with
    | scalar y =>
      simp [hfa, hfb, mul_comm]

/-! ## Reductions and aggregation -/


end Spec
