/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Basic.Core

/-!
Fold and reduction lemmas for dependent tensors.

This module packages the algebra needed to reason about tensor reductions, finite sums, and
shape-indexed traversals.
-/

@[expose] public section

namespace Spec

open Tensor
open scoped BigOperators

/-- Elementwise multiplication is associative (`mul_spec` is pointwise `(*)`). -/
theorem mul_spec_assoc {s : Shape} (a b c : Tensor ℝ s) :
  mulSpec a (mulSpec b c) = mulSpec (mulSpec a b) c := by
  induction s with
  | scalar =>
      cases a
      cases b
      cases c
      simp [mulSpec, map2Spec, mul_assoc]
  | dim n s ih =>
      cases a with
      | dim fa =>
        cases b with
        | dim fb =>
          cases c with
          | dim fc =>
            simp [mulSpec, map2Spec]
            funext i
            simpa [mulSpec, map2Spec] using ih (a := fa i) (b := fb i) (c := fc i)

/-- Elementwise multiplication is commutative (`mul_spec` is pointwise `(*)`). -/
theorem mul_spec_comm {s : Shape}
  (a b : Tensor ℝ s) : mulSpec a b = mulSpec b a := by
  induction s with
  | scalar =>
      cases a
      cases b
      simp [mulSpec, map2Spec, mul_comm]
  | dim n s ih =>
      cases a with
      | dim fa =>
        cases b with
        | dim fb =>
          simp [mulSpec, map2Spec]
          funext i
          simpa [mulSpec, map2Spec] using ih (a := fa i) (b := fb i)

/-- Elementwise addition is commutative (`add_spec` is pointwise `(+)`). -/
theorem add_spec_comm {s : Shape}
  (a b : Tensor ℝ s) : addSpec a b = addSpec b a := by
  induction s with
  | scalar =>
      cases a
      cases b
      simp [addSpec, map2Spec, add_comm]
  | dim n s ih =>
      cases a with
      | dim fa =>
        cases b with
        | dim fb =>
          simp [addSpec, map2Spec]
          funext i
          simpa [addSpec, map2Spec] using ih (a := fa i) (b := fb i)

/-- Elementwise multiplication of two all-zero tensors is the all-zero tensor. -/
theorem mul_spec_fill_zero {s : Shape} :
    mulSpec (fill (0 : ℝ) s) (fill (0 : ℝ) s) = fill (0 : ℝ) s := by
  induction s with
  | scalar =>
    simp [mulSpec, map2Spec, fill]
  | dim n s ih =>
    simp [mulSpec, map2Spec, fill]
    funext i
    simpa [mulSpec, map2Spec] using ih

/-! ## Real dot product and fold bridges -/

/--
Real dot product for same-shape tensors, defined as the sum of elementwise products.

This matches the common PyTorch idiom `(a * b).sum()` for same-shape tensors.
Citations:
https://pytorch.org/docs/stable/generated/torch.sum.html

This is the `Spec`-namespace dot product used by real-analysis proofs. The backend-generic
recursive dot product is `Proofs.TensorAlgebra.dot`.
-/
noncomputable def dot {s : Shape} (a b : Tensor ℝ s) : ℝ :=
  sumSpec (mulSpec a b)

/--
One-step unfolding of the internal tail-recursive helper `tensor_foldl_spec.go` when the loop
condition holds (`k < n`).

This lemma is a proof tool: it lets proofs *peel one loop step* without using `unfold` directly.
-/
lemma tensor_foldl_spec_go_of_lt {α β : Type} (f : β → α → β)
    {n : Nat} {s : Shape} (values : Fin n → Tensor α s) {k : Nat} (acc : β) (hk : k < n) :
    tensorFoldlSpec.go f n s values k acc =
      tensorFoldlSpec.go f n s values (k + 1) (tensorFoldlSpec f acc (values ⟨k, hk⟩)) := by
  -- Use the definitional equation once, but prevent `simp` from unfolding `go` recursively.
  rw [tensorFoldlSpec.go.eq_1]
  simp [hk]

/--
One-step unfolding of the internal tail-recursive helper `tensor_foldl_spec.go` when the loop
condition fails (`¬ k < n`), i.e. the loop terminates and returns the accumulator.
-/
lemma tensor_foldl_spec_go_of_not_lt {α β : Type} (f : β → α → β)
    {n : Nat} {s : Shape} (values : Fin n → Tensor α s) {k : Nat} (acc : β) (hk : ¬ k < n) :
    tensorFoldlSpec.go f n s values k acc = acc := by
  rw [tensorFoldlSpec.go.eq_1]
  simp [hk]

/--
Accumulator lemma for `tensor_foldl_spec` specialized to addition.

Informally: folding with `(+)` over a tensor adds `sum_spec t` to the initial accumulator.
This is frequently used to move between “fold-style” specs and “sum-style” algebra.
-/
lemma tensor_foldl_spec_add_init {s : Shape} (acc : ℝ) (t : Tensor ℝ s) :
    tensorFoldlSpec (· + ·) acc t = acc + sumSpec t := by
  induction s generalizing acc with
  | scalar =>
    cases t with
    | scalar x =>
      simp [tensorFoldlSpec, sumSpec]
  | dim n s ih =>
    cases t with
    | dim values =>
      -- `tensor_foldl_spec` uses an internal tail-recursive `go`; prove the accumulator lemma for
      -- it.
      -- We show: `go k acc = acc + go k 0` by induction on `n - k`.
      have go_add : ∀ k acc, k ≤ n →
          tensorFoldlSpec.go (· + ·) n s values k acc =
            acc + tensorFoldlSpec.go (· + ·) n s values k 0 := by
        intro k acc hk
        induction hn : n - k generalizing k acc with
        | zero =>
          have hk' : k = n := by
            exact Nat.le_antisymm hk (Nat.sub_eq_zero_iff_le.mp hn)
          subst k
          -- Since `k = n`, the loop condition is false and `go` returns the accumulator.
          have hgo_acc :
              tensorFoldlSpec.go (· + ·) n s values n acc = acc := by
            simpa using
              (tensor_foldl_spec_go_of_not_lt (f := (· + ·)) (values := values) (k := n) (acc := acc)
                (by simp))
          have hgo_0 :
              tensorFoldlSpec.go (· + ·) n s values n (0 : ℝ) = (0 : ℝ) := by
            simpa using
              (tensor_foldl_spec_go_of_not_lt (f := (· + ·)) (values := values) (k := n)
                (acc := (0 : ℝ)) (by simp))
          simp [hgo_acc, hgo_0]
        | succ m ih_go =>
          have hlt : k < n := by
            exact Nat.sub_pos_iff_lt.mp (by simp [hn])
          have hk1 : k + 1 ≤ n := Nat.succ_le_of_lt hlt
          -- Peel one `go` step at index `k` on both sides.
          rw [tensor_foldl_spec_go_of_lt (f := (· + ·)) (values := values) (k := k) (acc := acc) hlt]
          rw [tensor_foldl_spec_go_of_lt (f := (· + ·)) (values := values) (k := k)
            (acc := (0 : ℝ)) hlt]
          -- Apply the induction hypothesis for `k+1` at the updated accumulators.
          have h_next : n - (k + 1) = m := by
            simp [Nat.sub_succ, hn]
          -- IH on the sub-tensor: `fold acc = acc + sum`
          have h_step :
              tensorFoldlSpec (· + ·) acc (values ⟨k, hlt⟩) =
                acc + sumSpec (values ⟨k, hlt⟩) := ih acc (values ⟨k, hlt⟩)
          have h_step0 :
              tensorFoldlSpec (· + ·) 0 (values ⟨k, hlt⟩) =
                0 + sumSpec (values ⟨k, hlt⟩) := ih 0 (values ⟨k, hlt⟩)
          -- Use IH on `go` for the (k+1)-suffix.
          have hgo_acc :
              tensorFoldlSpec.go (· + ·) n s values (k + 1)
                  (tensorFoldlSpec (· + ·) acc (values ⟨k, hlt⟩))
                =
              tensorFoldlSpec (· + ·) acc (values ⟨k, hlt⟩)
                + tensorFoldlSpec.go (· + ·) n s values (k + 1) 0 := by
            -- `ih_go` is stated for the successor case `m = n - k - 1`
            -- so rewrite `n - (k+1)` to `m`.
            have := ih_go (k := k + 1) (acc := tensorFoldlSpec (· + ·) acc (values ⟨k, hlt⟩)) hk1
            simpa [h_next] using this
          have hgo_0 :
              tensorFoldlSpec.go (· + ·) n s values (k + 1)
                  (tensorFoldlSpec (· + ·) 0 (values ⟨k, hlt⟩))
                =
              tensorFoldlSpec (· + ·) 0 (values ⟨k, hlt⟩)
                + tensorFoldlSpec.go (· + ·) n s values (k + 1) 0 := by
            have := ih_go (k := k + 1) (acc := tensorFoldlSpec (· + ·) 0 (values ⟨k, hlt⟩)) hk1
            simpa [h_next] using this
          -- Put everything together.
          -- Goal is:
          --   go (k+1) (fold acc t_k) = acc + go (k+1) (fold 0 t_k)
          -- Rewrite both sides via the two `hgo_*` lemmas, then use `h_step`/`h_step0`.
          -- Note: `0 + x = x`, and rearrange via associativity/commutativity.
          calc
            tensorFoldlSpec.go (· + ·) n s values (k + 1)
                (tensorFoldlSpec (· + ·) acc (values ⟨k, hlt⟩))
                =
              tensorFoldlSpec (· + ·) acc (values ⟨k, hlt⟩)
                + tensorFoldlSpec.go (· + ·) n s values (k + 1) 0 := hgo_acc
            _ =
              (acc + sumSpec (values ⟨k, hlt⟩))
                + tensorFoldlSpec.go (· + ·) n s values (k + 1) 0 := by
                  simp [h_step]
            _ =
              acc +
                ((0 + sumSpec (values ⟨k, hlt⟩))
                  + tensorFoldlSpec.go (· + ·) n s values (k + 1) 0) := by
                  ring
            _ =
              acc +
                (tensorFoldlSpec (· + ·) 0 (values ⟨k, hlt⟩)
                  + tensorFoldlSpec.go (· + ·) n s values (k + 1) 0) := by
                  simp [h_step0]
            _ =
              acc +
                tensorFoldlSpec.go (· + ·) n s values (k + 1)
                  (tensorFoldlSpec (· + ·) 0 (values ⟨k, hlt⟩)) := by
                  -- use `hgo_0` backwards and rearrange
                  simp [hgo_0]
      -- Use `go_add` at k=0
      have h0 := go_add (k := 0) (acc := acc) (by exact Nat.zero_le n)
      -- finish: unfold `tensor_foldl_spec`/`sum_spec` for the outer dimension
      simpa [tensorFoldlSpec, sumSpec] using h0


-- Rewriting lemma under dot using associativity/commutativity
/-- Reassociate a `dot` over a pointwise product, using commutativity/associativity of `mul_spec`.
  -/
theorem dot_mul_reassoc {s : Shape}
  (dLdy m dx : Tensor ℝ s) :
  dot dLdy (mulSpec m dx) = dot (mulSpec m dLdy) dx := by
  have hAssoc := mul_spec_assoc (a := dLdy) (b := m) (c := dx)
  have hComm := mul_spec_comm (a := dLdy) (b := m)
  -- `mul_spec dLdy (mul_spec m dx) = mul_spec (mul_spec dLdy m) dx`
  -- and `mul_spec (mul_spec dLdy m) dx = mul_spec (mul_spec m dLdy) dx`.
  simp [dot, hAssoc, hComm]

/--
Coordinate formula for `mat_vec_mul_spec`, converted from the spec's `List.finRange` fold to a
`Finset.univ.sum`.

This is the “PyTorch-looking” statement of matvec: each output entry is a dot product of the
corresponding row with the input vector.
-/
lemma getScalar_mat_vec_mul_spec {m n : Nat}
  (A : Tensor ℝ [m, n])
  (v : Tensor ℝ [n]) (i : Fin m) :
  getScalar (matVecMulSpec A v) i = ∑ k : Fin n, (get2 A i k) * (getScalar v k) := by
  -- Reuse the backend-generic lemma from `NN/Proofs/Tensor/Algebra.lean` (instantiated at `ℝ`).
  simpa using
    (Proofs.TensorAlgebra.getScalar_mat_vec_mul_spec (α := ℝ) (A := A) (v := v) (i := i))

set_option linter.auxLemma false in
/--
Coordinate formula for `mat_mul_spec` (matrix-matrix multiplication).

This is the standard triple-sum identity: `(A @ B)[i,j] = ∑ k, A[i,k] * B[k,j]`, matching the
textbook/PyTorch view of matrix multiplication.

Citations:
https://pytorch.org/docs/stable/generated/torch.matmul.html
-/
lemma get2_mat_mul_spec {m n p : Nat}
  (A : Tensor ℝ [m, n])
  (B : Tensor ℝ [n, p]) (i : Fin m) (j : Fin p) :
  get2 (matMulSpec A B) i j = ∑ k : Fin n, (get2 A i k) * (get2 B k j) := by
  classical
  cases A with
  | dim rowsA =>
    cases B with
    | dim rowsB =>
      -- Unfold the matrix multiplication at `(i,j)` and convert the `finRange` fold to a
      -- `Finset.sum`.
      simp [matMulSpec, get2_eq, get_eq]
      -- Put the fold into the canonical `s + f k` form to apply `List.finRange_foldl_add_eq_finset_sum`.
      let f : Fin n → ℝ := fun k =>
        matMulSpec.match_3 (motive := fun _ _ => ℝ) (rowsA i) (rowsB k) (fun colsA colsB =>
          matMulSpec.match_1 (motive := fun _ _ => ℝ) (colsA k) (colsB j) (fun a b => a * b))
      have hfun :
          (fun (s : ℝ) (k : Fin n) =>
              matMulSpec.match_3 (motive := fun _ _ => ℝ) (rowsA i) (rowsB k) (fun colsA colsB =>
                matMulSpec.match_1 (motive := fun _ _ => ℝ) (colsA k) (colsB j) (fun a b => s + a
                  * b)))
            =
            (fun s k => s + f k) := by
        funext s k
        cases hrowA : rowsA i with
        | dim colsA =>
          cases hrowB : rowsB k with
          | dim colsB =>
            cases hA : colsA k with
            | scalar a =>
              cases hB : colsB j with
              | scalar b =>
                simp [f, hrowA, hrowB, hA, hB]
      have hsum : (List.finRange n).foldl (fun s k => s + f k) 0 = ∑ k : Fin n, f k :=
        List.finRange_foldl_add_eq_finset_sum (f := f)
      rw [hfun, hsum]
      refine Finset.sum_congr rfl ?_
      intro k _
      cases hrowA : rowsA i with
      | dim colsA =>
        cases hrowB : rowsB k with
        | dim colsB =>
          cases hA : colsA k with
          | scalar a =>
            cases hB : colsB j with
            | scalar b =>
              simp [f, hrowA, hrowB, hA, hB]

/--
Sum over the outer dimension unfolds into a `Finset.univ` sum of inner `sum_spec`.

This is the tensor analogue of `torch.sum` reducing over a leading dimension.
-/
lemma sum_spec_dim {n : Nat} {s : Shape} (t : Tensor ℝ (.dim n s)) :
  sumSpec t = ∑ i : Fin n, sumSpec (get t i) := by
  classical
  cases t with
  | dim values =>
      let f : Fin n → ℝ := fun i => sumSpec (values i)
      have go_eq :
          ∀ k acc, k ≤ n →
            tensorFoldlSpec.go (· + ·) n s values k acc =
              acc + (Finset.univ.filter (fun i : Fin n => k ≤ i.val)).sum f := by
        intro k acc hk
        induction hn : n - k generalizing k acc with
        | zero =>
            have hk' : k = n := by
              exact Nat.le_antisymm hk (Nat.sub_eq_zero_iff_le.mp hn)
            subst k
            have hgo :
                tensorFoldlSpec.go (· + ·) n s values n acc = acc := by
              simpa using
                (tensor_foldl_spec_go_of_not_lt (f := (· + ·)) (values := values) (k := n) (acc := acc)
                  (by simp))
            simp [hgo]
            have hfilter : (Finset.univ.filter (fun i : Fin n => n ≤ i.val)) = (∅ : Finset (Fin n))
              := by
              ext i
              simp [Nat.not_le_of_lt i.isLt]
            simp [hfilter]
        | succ m ih =>
            have hlt : k < n := by
              exact Nat.sub_pos_iff_lt.mp (by simp [hn])
            have hk1 : k + 1 ≤ n := Nat.succ_le_of_lt hlt
            rw [tensor_foldl_spec_go_of_lt (f := (· + ·)) (values := values) (k := k) (acc := acc) hlt]
            have hstep :
                tensorFoldlSpec (· + ·) acc (values ⟨k, hlt⟩) = acc + f ⟨k, hlt⟩ := by
              simpa [f] using
                (tensor_foldl_spec_add_init (s := s) (acc := acc) (t := values ⟨k, hlt⟩))
            have h_next : n - (k + 1) = m := by
              simp [Nat.sub_succ, hn]
            have ih' :=
              (ih (k := k + 1) (acc := acc + f ⟨k, hlt⟩) hk1)
            have ih'' :
                tensorFoldlSpec.go (· + ·) n s values (k + 1) (acc + f ⟨k, hlt⟩) =
                  (acc + f ⟨k, hlt⟩) +
                    (Finset.univ.filter (fun i : Fin n => k + 1 ≤ i.val)).sum f := by
              simpa [h_next] using ih'
            let Sk : Finset (Fin n) := Finset.univ.filter (fun i : Fin n => k ≤ i.val)
            let Sk1 : Finset (Fin n) := Finset.univ.filter (fun i : Fin n => k + 1 ≤ i.val)
            have hSk : Sk = insert (⟨k, hlt⟩ : Fin n) Sk1 := by
              ext i
              constructor
              · intro hiSk
                have hle : k ≤ i.val := by
                  simpa [Sk] using hiSk
                have hcase : k = i.val ∨ k < i.val := Nat.eq_or_lt_of_le hle
                refine (Finset.mem_insert).2 ?_
                cases hcase with
                | inl hEq =>
                    left
                    apply Fin.ext
                    exact hEq.symm
                | inr hLt =>
                    right
                    have hk1' : k + 1 ≤ i.val := Nat.succ_le_of_lt hLt
                    simpa [Sk1] using hk1'
              · intro hiIns
                have hi' : i = (⟨k, hlt⟩ : Fin n) ∨ i ∈ Sk1 := (Finset.mem_insert).1 hiIns
                cases hi' with
                | inl hEq =>
                    subst hEq
                    simp [Sk]
                  | inr hiSk1 =>
                      have hk1' : k + 1 ≤ i.val := by
                        simpa [Sk1] using hiSk1
                      have hle : k ≤ i.val := Nat.le_trans (Nat.le_succ k) hk1'
                      simpa [Sk] using hle
            have hk_not_mem1 : (⟨k, hlt⟩ : Fin n) ∉ Sk1 := by
              simp [Sk1]
            have hSk_sum : Sk.sum f = f ⟨k, hlt⟩ + Sk1.sum f := by
              have :
                  (insert (⟨k, hlt⟩ : Fin n) Sk1).sum f = f ⟨k, hlt⟩ + Sk1.sum f := by
                simpa using
                  (Finset.sum_insert (s := Sk1) (a := (⟨k, hlt⟩ : Fin n)) (f := f) hk_not_mem1)
              simpa [hSk] using this
            calc
              tensorFoldlSpec.go (· + ·) n s values (k + 1)
                  (tensorFoldlSpec (· + ·) acc (values ⟨k, hlt⟩))
                  =
                tensorFoldlSpec.go (· + ·) n s values (k + 1) (acc + f ⟨k, hlt⟩) := by
                  simp [hstep]
              _ = (acc + f ⟨k, hlt⟩) + Sk1.sum f := ih''
              _ = acc + (f ⟨k, hlt⟩ + Sk1.sum f) := by ring
              _ = acc + Sk.sum f := by
                    simp [hSk_sum, Sk]
      have hfilter0 :
          (Finset.univ.filter (fun i : Fin n => (0 : Nat) ≤ i.val)) = (Finset.univ : Finset (Fin n))
            := by
        ext i
        simp
      have h0 := go_eq (k := 0) (acc := (0 : ℝ)) (Nat.zero_le n)
      -- turn `Finset.univ.sum f` into the claimed RHS using `get_eq`
      have hget : ∀ i : Fin n, get (Tensor.dim values) i = values i := by
        intro i
        simp [get_eq]
      calc
        sumSpec (Tensor.dim values)
            = tensorFoldlSpec.go (· + ·) n s values 0 0 := by
                simp [sumSpec, tensorFoldlSpec]
        _ = (0 : ℝ) + (Finset.univ.filter (fun i : Fin n => (0 : Nat) ≤ i.val)).sum f := by
              simpa using h0
        _ = ∑ i : Fin n, sumSpec (get (Tensor.dim values) i) := by
              simp [f, hget]

/--
The real-analysis `Spec.dot` agrees with the backend-generic recursive dot.

`Spec.dot` is defined as `sumSpec (mulSpec a b)`, which is the proof layer version of the
PyTorch idiom `(a * b).sum()`.  `Proofs.TensorAlgebra.dot` is recursive over the tensor shape so it
works over arbitrary semiring-like scalar models.  This bridge lets real proofs reuse generic
algebra instead of repeating finite-sum rearrangements.
-/
theorem dot_eq_tensorAlgebra_dot {s : Shape} (a b : Tensor ℝ s) :
    dot a b = Proofs.TensorAlgebra.dot (α := ℝ) a b := by
  induction s with
  | scalar =>
      cases a
      cases b
      simp [dot, Proofs.TensorAlgebra.dot, sumSpec, tensorFoldlSpec, mulSpec, map2Spec]
  | dim n s ih =>
      cases a with
      | dim fa =>
        cases b with
        | dim fb =>
          have hsum :
              sumSpec (mulSpec (Tensor.dim fa) (Tensor.dim fb)) =
                ∑ i : Fin n, sumSpec (mulSpec (fa i) (fb i)) := by
            have h := sum_spec_dim (t := mulSpec (Tensor.dim fa) (Tensor.dim fb))
            simpa [mulSpec, map2Spec, get_eq] using h
          have hrec :
              (∑ i : Fin n, sumSpec (mulSpec (fa i) (fb i))) =
                ∑ i : Fin n, Proofs.TensorAlgebra.dot (α := ℝ) (fa i) (fb i) := by
            refine Finset.sum_congr rfl ?_
            intro i _
            simpa [dot] using ih (a := fa i) (b := fb i)
          have hfold :
              (List.finRange n).foldl
                  (fun acc i => acc + Proofs.TensorAlgebra.dot (α := ℝ) (fa i) (fb i)) 0 =
                ∑ i : Fin n, Proofs.TensorAlgebra.dot (α := ℝ) (fa i) (fb i) := by
            simpa using
              (List.finRange_foldl_add_eq_finset_sum
                (f := fun i : Fin n => Proofs.TensorAlgebra.dot (α := ℝ) (fa i) (fb i)))
          calc
            dot (Tensor.dim fa) (Tensor.dim fb)
                = sumSpec (mulSpec (Tensor.dim fa) (Tensor.dim fb)) := rfl
            _ = ∑ i : Fin n, sumSpec (mulSpec (fa i) (fb i)) := hsum
            _ = ∑ i : Fin n, Proofs.TensorAlgebra.dot (α := ℝ) (fa i) (fb i) := hrec
            _ = (List.finRange n).foldl
                  (fun acc i => acc + Proofs.TensorAlgebra.dot (α := ℝ) (fa i) (fb i)) 0 :=
                hfold.symm
            _ = Proofs.TensorAlgebra.dot (α := ℝ) (Tensor.dim fa) (Tensor.dim fb) := by
              simp [Proofs.TensorAlgebra.dot]

end Spec
