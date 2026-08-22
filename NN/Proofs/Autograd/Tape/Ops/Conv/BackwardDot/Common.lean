/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Basic
public import NN.Proofs.Utils.List
public import NN.Spec.Layers.Conv
public import NN.Spec.Layers.Conv.TwoD.Padding

public import Mathlib.Algebra.BigOperators.Ring.Finset
public import Mathlib.Data.Fintype.BigOperators

/-!
# BackwardDot

`Conv2d` backward dot-level correctness (bridge lemma).

The runtime autograd engine (`NN.Runtime.Autograd.Engine`) computes Conv2d gradients via the
handwritten spec:

* `Spec.conv2d_backward_spec` (kernel/bias/input)

For the **analytic** spec-level theorem over `ℝ`, Conv2d is already covered via `fderiv`/adjoints in:

* `NN.Proofs.Autograd.Tape.Ops.Conv.FDeriv`

That file provides a proof-only Conv2d node whose VJP is `(fderiv forward)†`, so any DAG using it
is covered by the global theorem `Graph.backpropVec_eq_adjoint_fderiv`.

What remains here is the dot/adjointness bridge:

$$
\operatorname{dot}(\operatorname{JVP}_{\mathrm{conv2d}}(\ldots),\delta)
=\operatorname{dot}(dK,g_K)+\operatorname{dot}(dB,g_B)+\operatorname{dot}(dX,g_X).
$$

Here `(gK, gB, gX)` is the tuple returned by `Spec.conv2d_backward_spec` for $\delta$.

The padding-related rewrites needed for the input-gradient proof are factored into:

* `NN/Spec/Layers/Conv/TwoD/Padding.lean` (`getAtOrZero_padChannelsFirst2d*` lemmas)
-/

@[expose] public section


namespace Proofs
namespace Autograd
namespace Conv2d

open Spec
open Tensor

open scoped BigOperators

noncomputable section

-- Shared dot-product and index lemmas used by the convolution backward-dot proofs.

lemma dot_add_right {s : Shape} (a b c : Tensor ℝ s) :
    dot a (addSpec b c) = dot a b + dot a c := by
  -- `dot` is symmetric; reduce to `dot_add_left`.
  calc
    dot a (addSpec b c) = dot (addSpec b c) a := by
      simpa using (dot_comm (a := a) (b := addSpec b c))
    _ = dot b a + dot c a := by
      simpa using (dot_add_left (a := b) (b := c) (c := a))
    _ = dot a b + dot a c := by
      simp [dot_comm]

lemma dot_dim {n : Nat} {s : Shape} (a b : Tensor ℝ (.dim n s)) :
    dot a b = ∑ i : Fin n, dot (get a i) (get b i) := by
  classical
  cases a with
  | dim fa =>
    cases b with
    | dim fb =>
      unfold dot
      have hsum := (Spec.sum_spec_dim (t := mulSpec (Tensor.dim fa) (Tensor.dim fb)))
      -- `mul_spec` is pointwise on `.dim`, so each slice is `mul_spec (fa i) (fb i)`.
      simpa [dot, mulSpec, map2Spec, get_eq] using hsum

lemma dot_scalar (x y : ℝ) :
    dot (Tensor.scalar x) (Tensor.scalar y) = x * y := by
  simp [dot, sumSpec, tensorFoldlSpec, mulSpec, map2Spec]

lemma dot_vec_eq_sum_get
    {n : Nat} (a b : Tensor ℝ (.dim n .scalar)) :
    dot a b = ∑ i : Fin n, (getAtOrZero a [i.val]) * (getAtOrZero b [i.val]) := by
  classical
  cases a with
  | dim fa =>
    cases b with
    | dim fb =>
      -- Use `dot_dim` to reduce to a sum of scalar dots.
      rw [dot_dim (a := Tensor.dim fa) (b := Tensor.dim fb)]
      refine Finset.sum_congr rfl ?_
      intro i _
      cases ha : fa i with
      | scalar x =>
        cases hb : fb i with
        | scalar y =>
          -- Each summand is scalar multiplication.
          have hdot : dot (Tensor.scalar x) (Tensor.scalar y) = x * y := dot_scalar x y
          -- And `get_at_or_zero` reads that scalar at index `i`.
          have hreadA : getAtOrZero (Tensor.dim fa) [i.val] = x := by
            simp [i.isLt, ha]
          have hreadB : getAtOrZero (Tensor.dim fb) [i.val] = y := by
            simp [i.isLt, hb]
          simp [get_eq, ha, hb, hdot, hreadA, hreadB]

@[simp] lemma get_at_or_zero_tensor_cast {s t : Shape} (h : s = t) (x : Tensor ℝ s)
    (idx : List Nat) :
    getAtOrZero (Tensor.castShape (t := x) h) idx = getAtOrZero x idx := by
  cases h
  rfl

lemma get2_eq_get_at_or_zero
    {m n : Nat} (A : Tensor ℝ (.dim m (.dim n .scalar))) (i : Fin m) (j : Fin n) :
    get2 A i j = getAtOrZero A [i.val, j.val] := by
  cases A with
  | dim rows =>
    cases hrow : rows i with
    | dim cols =>
      cases hcell : cols j with
      | scalar v =>
        simp [Spec.get2, get_eq, i.isLt, j.isLt, hrow, hcell]

lemma get_at_or_zero_get_channel
    {C H W : Nat} (t : Tensor ℝ (.dim C (.dim H (.dim W .scalar))))
    (c : Fin C) (i : Fin H) (j : Fin W) :
    getAtOrZero (get t c) [i.val, j.val] = getAtOrZero t [c.val, i.val, j.val] := by
  cases t with
  | dim fC =>
    simp [get_eq, c.isLt]

-- Bias-broadcast dot-product lemma used by convolution-gradient proofs.

/-- Broadcast a bias gradient `outC` across spatial axes into an `outC×outH×outW` tensor. -/
def biasBroadcast {outC outH outW : Nat} (db : Tensor ℝ (.dim outC .scalar)) :
    Tensor ℝ (.dim outC (.dim outH (.dim outW .scalar))) :=
  Tensor.dim (fun oc =>
    Tensor.dim (fun _i =>
      Tensor.dim (fun _j =>
        Tensor.scalar (getAtOrZero db [oc.val]))))

lemma conv2d_bias_deriv_get
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (layer : Spec.Conv2dSpec inC outC kH kW stride padding ℝ h1 h2 h3)
    (input : Spec.Tensor ℝ (.dim inC (.dim inH (.dim inW .scalar))))
    (δ : Spec.Tensor ℝ (.dim outC (.dim (Shape.slidingWindowOutDim inH kH stride padding) (.dim (Shape.slidingWindowOutDim inW kW stride padding) .scalar))))
    (oc : Fin outC) :
    getAtOrZero (Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layer) (input := input)
      (grad_output := δ)) [oc.val]
      =
    ∑ i : Fin (Shape.slidingWindowOutDim inH kH stride padding),
      ∑ j : Fin (Shape.slidingWindowOutDim inW kW stride padding),
        getAtOrZero δ [oc.val, i.val, j.val] := by
  classical
  unfold Spec.conv2dBiasDerivSpec
  -- Unfold the two nested `finRange` folds into `Finset.univ` sums.
  -- First, rewrite the inner fold (over `out_w`) as `acc + ∑ j`.
  simp only []
  -- After unfolding, the value is a scalar tensor, so `get_at_or_zero .. []` extracts the scalar.
  -- Convert the inner fold and then the outer fold.
  have houtW :
      ∀ (i : Fin (Shape.slidingWindowOutDim inH kH stride padding)) (acc : ℝ),
        (List.finRange (Shape.slidingWindowOutDim inW kW stride padding)).foldl
            (fun acc j => acc + getAtOrZero δ [oc.val, i.val, j.val]) acc
          =
        acc + ∑ j : Fin (Shape.slidingWindowOutDim inW kW stride padding), getAtOrZero δ [oc.val, i.val,
          j.val] := by
    intro i acc
    -- Use `foldl_add_init` then `List.finRange_foldl_add_eq_finset_sum`.
    have h1 :=
      List.foldl_add_init (l := List.finRange (Shape.slidingWindowOutDim inW kW stride padding))
        (f := fun j : Fin (Shape.slidingWindowOutDim inW kW stride padding) => getAtOrZero δ [oc.val,
          i.val, j.val])
        (acc := acc)
    -- Convert the `0`-initialized fold into a `Finset` sum.
    have h2 :
        (List.finRange (Shape.slidingWindowOutDim inW kW stride padding)).foldl
            (fun s j => s + getAtOrZero δ [oc.val, i.val, j.val]) 0
          =
        ∑ j : Fin (Shape.slidingWindowOutDim inW kW stride padding), getAtOrZero δ [oc.val, i.val, j.val]
          := by
      simpa using
        (List.finRange_foldl_add_eq_finset_sum
          (f := fun j : Fin (Shape.slidingWindowOutDim inW kW stride padding) =>
            getAtOrZero δ [oc.val, i.val, j.val]))
    simpa [h2] using h1

  -- Now handle the outer fold over `out_h`.
  have houtH :
      (List.finRange (Shape.slidingWindowOutDim inH kH stride padding)).foldl
          (fun acc (i : Fin (Shape.slidingWindowOutDim inH kH stride padding)) =>
            (List.finRange (Shape.slidingWindowOutDim inW kW stride padding)).foldl
                (fun acc j => acc + getAtOrZero δ [oc.val, i.val, j.val]) acc)
          0
        =
      ∑ i : Fin (Shape.slidingWindowOutDim inH kH stride padding),
        ∑ j : Fin (Shape.slidingWindowOutDim inW kW stride padding), getAtOrZero δ [oc.val, i.val, j.val]
          := by
    -- Rewrite the step function using `houtW`, then convert the resulting fold to a sum.
    -- First show the fold is equivalent to `foldl (fun acc i => acc + innerSum i) 0`.
    have hstep :
        (fun acc (i : Fin (Shape.slidingWindowOutDim inH kH stride padding)) =>
          (List.finRange (Shape.slidingWindowOutDim inW kW stride padding)).foldl
              (fun acc j => acc + getAtOrZero δ [oc.val, i.val, j.val]) acc)
        =
        (fun acc (i : Fin (Shape.slidingWindowOutDim inH kH stride padding)) =>
          acc + ∑ j : Fin (Shape.slidingWindowOutDim inW kW stride padding), getAtOrZero δ [oc.val, i.val,
            j.val]) := by
      funext acc i
      simpa using (houtW i acc)
    -- Convert.
    simp [hstep, List.finRange_foldl_add_eq_finset_sum]

  -- Finish by `simp`ing the definition back to the fold we rewrote (and normalizing away the
  -- `tensor_cast` inserted by `conv2d_bias_deriv_spec`).
  simpa [Spec.convBiasDerivSpec, Spec.Conv.Internal.foldlIndices, Spec.convOutSpatial,
    Spec.Shape.slidingWindowOutDim,
    Vector.get, Vector.toList, oc.isLt] using houtH

lemma dot_biasBroadcast_eq_dot_bias_deriv
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (layer : Spec.Conv2dSpec inC outC kH kW stride padding ℝ h1 h2 h3)
    (input : Spec.Tensor ℝ (.dim inC (.dim inH (.dim inW .scalar))))
    (db : Tensor ℝ (.dim outC .scalar))
    (δ : Spec.Tensor ℝ (.dim outC (.dim (Shape.slidingWindowOutDim inH kH stride padding) (.dim (Shape.slidingWindowOutDim inW kW stride padding) .scalar)))) :
    dot (biasBroadcast
          (outC := outC)
          (outH := (Shape.slidingWindowOutDim inH kH stride padding))
          (outW := (Shape.slidingWindowOutDim inW kW stride padding))
          db) δ
      =
    dot db (Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layer) (input := input) (grad_output :=
      δ)) := by
  classical
  -- Expand the LHS dot on the 3D output as `∑ oc, dot (sliceA oc) (sliceδ oc)`,
  -- then expand each matrix dot with `dot_mat_eq_sum`.
  have hdot3 :
      dot (biasBroadcast
            (outC := outC)
            (outH := (Shape.slidingWindowOutDim inH kH stride padding))
            (outW := (Shape.slidingWindowOutDim inW kW stride padding))
            db) δ
        =
      ∑ oc : Fin outC,
        ∑ i : Fin (Shape.slidingWindowOutDim inH kH stride padding),
          ∑ j : Fin (Shape.slidingWindowOutDim inW kW stride padding),
            (getAtOrZero db [oc.val]) * (getAtOrZero δ [oc.val, i.val, j.val]) := by
    -- Outer dimension over channels.
    rw [dot_dim (a := biasBroadcast (outC := outC)
            (outH := (Shape.slidingWindowOutDim inH kH stride padding))
            (outW := (Shape.slidingWindowOutDim inW kW stride padding)) db) (b := δ)]
    refine Finset.sum_congr rfl ?_
    intro oc _
    -- Matrix dot on the spatial slice.
    have hmat :=
      Spec.dot_mat_eq_sum (m := (Shape.slidingWindowOutDim inH kH stride padding))
        (n := (Shape.slidingWindowOutDim inW kW stride padding))
        (A := get (biasBroadcast (outC := outC)
          (outH := (Shape.slidingWindowOutDim inH kH stride padding))
          (outW := (Shape.slidingWindowOutDim inW kW stride padding)) db) oc)
        (B := get δ oc)
    -- Rewrite `get2` to `get_at_or_zero`, and simplify the broadcast slice entry.
    -- `get2 (get (biasBroadcast db) oc) i j = db[oc]`.
    -- `get2 (get δ oc) i j = δ[oc,i,j]`.
    -- Rewrite the slice `get2` in terms of `get_at_or_zero`, then lift it back to the 3D index.
    -- `get δ oc` is the `oc`-th channel slice.
    have hδ :
        ∀ (i : Fin (Shape.slidingWindowOutDim inH kH stride padding))
          (j : Fin (Shape.slidingWindowOutDim inW kW stride padding)),
          getAtOrZero (get δ oc) [i.val, j.val] = getAtOrZero δ [oc.val, i.val, j.val] := by
      intro i j
      simpa using (get_at_or_zero_get_channel (t := δ) (c := oc) (i := i) (j := j))
    -- Now rewrite each summand.
    refine (hmat.trans ?_ )
    -- Rewrite the double sum term-by-term.
    refine Finset.sum_congr rfl ?_
    intro i _
    refine Finset.sum_congr rfl ?_
    intro j _
    -- Convert `get2` to `get_at_or_zero` on both slices.
    have hA2 :
        get2 (get (biasBroadcast (outC := outC)
          (outH := (Shape.slidingWindowOutDim inH kH stride padding))
          (outW := (Shape.slidingWindowOutDim inW kW stride padding)) db) oc) i j
          =
        getAtOrZero db [oc.val] := by
      -- `biasBroadcast` is constant across the spatial indices, and all three reads are in bounds.
      simp [biasBroadcast, get_eq]
    have hB2 :
        get2 (get δ oc) i j = getAtOrZero δ [oc.val, i.val, j.val] := by
      have h := get2_eq_get_at_or_zero (A := get δ oc) i j
      -- Lift slice read back to the 3D index using `hδ`.
      simpa [hδ i j] using h
    -- Finish.
    simp [hA2, hB2]

  -- Expand the RHS dot on the bias vector using `dot_vec_eq_sum`.
  have hdot1 :
      dot db (Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layer) (input := input) (grad_output
        := δ))
        =
      ∑ oc : Fin outC,
        (getAtOrZero db [oc.val]) *
          (getAtOrZero (Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layer) (input := input)
            (grad_output := δ)) [oc.val]) := by
    simpa using
      (dot_vec_eq_sum_get (a := db)
        (b := Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layer) (input := input) (grad_output
          := δ)))

  -- Rewrite each bias-deriv coordinate as the spatial sum, then compare with `hdot3`.
  calc
    dot (biasBroadcast (outC := outC)
          (outH := (Shape.slidingWindowOutDim inH kH stride padding))
          (outW := (Shape.slidingWindowOutDim inW kW stride padding)) db) δ
        =
      ∑ oc : Fin outC,
        ∑ i : Fin (Shape.slidingWindowOutDim inH kH stride padding),
          ∑ j : Fin (Shape.slidingWindowOutDim inW kW stride padding),
            (getAtOrZero db [oc.val]) * (getAtOrZero δ [oc.val, i.val, j.val]) := hdot3
    _ =
      ∑ oc : Fin outC,
        (getAtOrZero db [oc.val]) *
          (∑ i : Fin (Shape.slidingWindowOutDim inH kH stride padding),
            ∑ j : Fin (Shape.slidingWindowOutDim inW kW stride padding),
              getAtOrZero δ [oc.val, i.val, j.val]) := by
        simp [Finset.mul_sum]
    _ =
      ∑ oc : Fin outC,
        (getAtOrZero db [oc.val]) *
          (getAtOrZero (Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layer) (input := input)
            (grad_output := δ)) [oc.val]) := by
        refine Finset.sum_congr rfl ?_
        intro oc _
        simp [conv2d_bias_deriv_get (layer := layer) (input := input) (δ := δ) (oc := oc)]
    _ = dot db (Spec.conv2dBiasDerivSpec (α := ℝ) (layer := layer) (input := input)
      (grad_output := δ)) := by
        simp [hdot1]

-- ---------------------------------------------------------------------------
-- Dot expansions for 3D/4D tensors in terms of `get_at_or_zero`
-- ---------------------------------------------------------------------------

lemma dot3_eq_sum
    {C H W : Nat}
    (a b : Tensor ℝ (.dim C (.dim H (.dim W .scalar)))) :
    dot a b =
      ∑ c : Fin C, ∑ i : Fin H, ∑ j : Fin W,
        (getAtOrZero a [c.val, i.val, j.val]) * (getAtOrZero b [c.val, i.val, j.val]) := by
  classical
  -- Outer dimension.
  rw [dot_dim (a := a) (b := b)]
  refine Finset.sum_congr rfl ?_
  intro c _
  -- Matrix dot on the slice.
  have hmat := Spec.dot_mat_eq_sum (m := H) (n := W) (A := get a c) (B := get b c)
  -- Rewrite each `get2` as `get_at_or_zero`, and lift slice reads back to 3D.
  refine hmat.trans ?_
  refine Finset.sum_congr rfl ?_
  intro i _
  refine Finset.sum_congr rfl ?_
  intro j _
  have hA : get2 (get a c) i j = getAtOrZero a [c.val, i.val, j.val] := by
    -- `get2` -> `get_at_or_zero` on the slice, then lift to 3D.
    have h1 := get2_eq_get_at_or_zero (A := get a c) i j
    have h2 : getAtOrZero (get a c) [i.val, j.val] = getAtOrZero a [c.val, i.val, j.val] :=
      (get_at_or_zero_get_channel (t := a) (c := c) (i := i) (j := j))
    simpa [h2] using h1
  have hB : get2 (get b c) i j = getAtOrZero b [c.val, i.val, j.val] := by
    have h1 := get2_eq_get_at_or_zero (A := get b c) i j
    have h2 : getAtOrZero (get b c) [i.val, j.val] = getAtOrZero b [c.val, i.val, j.val] :=
      (get_at_or_zero_get_channel (t := b) (c := c) (i := i) (j := j))
    simpa [h2] using h1
  simp [hA, hB]

lemma get_at_or_zero_get_outer3
    {OC IC KH KW : Nat}
    (k : Tensor ℝ (.dim OC (.dim IC (.dim KH (.dim KW .scalar)))))
    (oc : Fin OC) (ic : Fin IC) (di : Fin KH) (dj : Fin KW) :
    getAtOrZero (get k oc) [ic.val, di.val, dj.val] =
      getAtOrZero k [oc.val, ic.val, di.val, dj.val] := by
  cases k with
  | dim fOC =>
    simp [get_eq, oc.isLt]

lemma dot4_eq_sum
    {OC IC KH KW : Nat}
    (a b : Tensor ℝ (.dim OC (.dim IC (.dim KH (.dim KW .scalar))))) :
    dot a b =
      ∑ oc : Fin OC, ∑ ic : Fin IC, ∑ di : Fin KH, ∑ dj : Fin KW,
        (getAtOrZero a [oc.val, ic.val, di.val, dj.val]) *
          (getAtOrZero b [oc.val, ic.val, di.val, dj.val]) := by
  classical
  -- Outer OC dimension.
  rw [dot_dim (a := a) (b := b)]
  refine Finset.sum_congr rfl ?_
  intro oc _
  -- Apply the 3D lemma to the slice.
  have hs := dot3_eq_sum (a := get a oc) (b := get b oc)
  -- Lift slice reads back to 4D.
  -- The RHS of `hs` is `∑ ic, ∑ di, ∑ dj, get_at_or_zero (get a oc) [...] * ...`.
  -- Rewrite those reads to 4D indices using `get_at_or_zero_get_outer3`.
  simpa [get_at_or_zero_get_outer3] using hs

lemma sum_mul {ι : Type} [Fintype ι] (f : ι → ℝ) (a : ℝ) :
    (∑ i : ι, f i) * a = ∑ i : ι, f i * a := by
  classical
  simpa using (Finset.sum_mul (s := (Finset.univ : Finset ι)) (f := f) a)

lemma mul_sum {ι : Type} [Fintype ι] (a : ℝ) (f : ι → ℝ) :
    a * (∑ i : ι, f i) = ∑ i : ι, a * f i := by
  classical
  simpa using (Finset.mul_sum (s := (Finset.univ : Finset ι)) (f := f) a)

lemma sum_comm {α β : Type} [Fintype α] [Fintype β] (f : α → β → ℝ) :
    (∑ a : α, ∑ b : β, f a b) = ∑ b : β, ∑ a : α, f a b := by
  classical
  simpa using (Finset.sum_comm (s := (Finset.univ : Finset α)) (t := (Finset.univ : Finset β)) (f :=
    f))

/-- Move the third index of a finite triple sum to the outside. -/
lemma sum_third_first
    {α β γ : Type} [Fintype α] [Fintype β] [Fintype γ]
    (f : α → β → γ → ℝ) :
    (∑ a : α, ∑ b : β, ∑ c : γ, f a b c) =
      ∑ c : γ, ∑ a : α, ∑ b : β, f a b c := by
  classical
  calc
    (∑ a : α, ∑ b : β, ∑ c : γ, f a b c) =
        ∑ a : α, ∑ c : γ, ∑ b : β, f a b c := by
      refine Fintype.sum_congr _ _ ?_
      intro a
      exact sum_comm (fun b c => f a b c)
    _ = ∑ c : γ, ∑ a : α, ∑ b : β, f a b c :=
      sum_comm (fun a c => ∑ b : β, f a b c)

/-- Move the fourth index of a finite quadruple sum to the outside. -/
lemma sum_fourth_first
    {α β γ δ : Type} [Fintype α] [Fintype β] [Fintype γ] [Fintype δ]
    (f : α → β → γ → δ → ℝ) :
    (∑ a : α, ∑ b : β, ∑ c : γ, ∑ d : δ, f a b c d) =
      ∑ d : δ, ∑ a : α, ∑ b : β, ∑ c : γ, f a b c d := by
  classical
  calc
    (∑ a : α, ∑ b : β, ∑ c : γ, ∑ d : δ, f a b c d) =
        ∑ a : α, ∑ b : β, ∑ d : δ, ∑ c : γ, f a b c d := by
      refine Fintype.sum_congr _ _ ?_
      intro a
      refine Fintype.sum_congr _ _ ?_
      intro b
      exact sum_comm (fun c d => f a b c d)
    _ = ∑ a : α, ∑ d : δ, ∑ b : β, ∑ c : γ, f a b c d := by
      refine Fintype.sum_congr _ _ ?_
      intro a
      exact sum_comm (fun b d => ∑ c : γ, f a b c d)
    _ = ∑ d : δ, ∑ a : α, ∑ b : β, ∑ c : γ, f a b c d :=
      sum_comm (fun a d => ∑ b : β, ∑ c : γ, f a b c d)

/-- Reorder the eight indices used when expanding the Conv2d input adjoint. -/
lemma sum_eight_conv_input_order
    {α β γ δ ε ζ η θ : Type}
    [Fintype α] [Fintype β] [Fintype γ] [Fintype δ]
    [Fintype ε] [Fintype ζ] [Fintype η] [Fintype θ]
    (f : α → β → γ → δ → ε → ζ → η → θ → ℝ) :
    (∑ a : α, ∑ b : β, ∑ c : γ, ∑ d : δ,
      ∑ e : ε, ∑ z : ζ, ∑ g : η, ∑ h : θ, f a b c d e z g h) =
    ∑ d : δ, ∑ e : ε, ∑ z : ζ, ∑ a : α,
      ∑ g : η, ∑ h : θ, ∑ b : β, ∑ c : γ, f a b c d e z g h := by
  classical
  calc
    (∑ a : α, ∑ b : β, ∑ c : γ, ∑ d : δ,
        ∑ e : ε, ∑ z : ζ, ∑ g : η, ∑ h : θ, f a b c d e z g h) =
      ∑ d : δ, ∑ a : α, ∑ b : β, ∑ c : γ,
        ∑ e : ε, ∑ z : ζ, ∑ g : η, ∑ h : θ, f a b c d e z g h :=
      sum_fourth_first (fun a b c d =>
        ∑ e : ε, ∑ z : ζ, ∑ g : η, ∑ h : θ, f a b c d e z g h)
    _ = ∑ d : δ, ∑ e : ε, ∑ a : α, ∑ b : β, ∑ c : γ,
          ∑ z : ζ, ∑ g : η, ∑ h : θ, f a b c d e z g h := by
      refine Fintype.sum_congr _ _ ?_
      intro d
      exact sum_fourth_first (fun a b c e =>
        ∑ z : ζ, ∑ g : η, ∑ h : θ, f a b c d e z g h)
    _ = ∑ d : δ, ∑ e : ε, ∑ z : ζ, ∑ a : α, ∑ b : β, ∑ c : γ,
          ∑ g : η, ∑ h : θ, f a b c d e z g h := by
      refine Fintype.sum_congr _ _ ?_
      intro d
      refine Fintype.sum_congr _ _ ?_
      intro e
      exact sum_fourth_first (fun a b c z =>
        ∑ g : η, ∑ h : θ, f a b c d e z g h)
    _ = ∑ d : δ, ∑ e : ε, ∑ z : ζ, ∑ a : α, ∑ g : η, ∑ b : β, ∑ c : γ,
          ∑ h : θ, f a b c d e z g h := by
      refine Fintype.sum_congr _ _ ?_
      intro d
      refine Fintype.sum_congr _ _ ?_
      intro e
      refine Fintype.sum_congr _ _ ?_
      intro z
      refine Fintype.sum_congr _ _ ?_
      intro a
      exact sum_third_first (fun b c g => ∑ h : θ, f a b c d e z g h)
    _ = ∑ d : δ, ∑ e : ε, ∑ z : ζ, ∑ a : α, ∑ g : η, ∑ h : θ,
          ∑ b : β, ∑ c : γ, f a b c d e z g h := by
      refine Fintype.sum_congr _ _ ?_
      intro d
      refine Fintype.sum_congr _ _ ?_
      intro e
      refine Fintype.sum_congr _ _ ?_
      intro z
      refine Fintype.sum_congr _ _ ?_
      intro a
      refine Fintype.sum_congr _ _ ?_
      intro g
      exact sum_third_first (fun b c h => f a b c d e z g h)

-- ---------------------------------------------------------------------------
-- Conv2d padding helper (matches the spec's `if padding = 0 then cast else pad`)
-- ---------------------------------------------------------------------------

/-!
`paddedInput` matches the forward spec’s padding convention:
- if `padding = 0`, it is just a shape cast; and
- otherwise it is `pad_multi_channel`.

PyTorch analogue: `torch.nn.functional.pad` (or implicit padding inside `conv2d`).
https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
-/

def paddedInput {inC inH inW padding : Nat}
    (input : Spec.Tensor ℝ (.dim inC (.dim inH (.dim inW .scalar)))) :
    Spec.Tensor ℝ (.dim inC (.dim (inH + 2 * padding) (.dim (inW + 2 * padding) .scalar))) :=
  if h4 : padding = 0 then
    tensorCast
      (Shape.dim inC (Shape.dim (inH + 2 * padding) (Shape.dim (inW + 2 * padding) .scalar)))
      (by simp; rw [h4])
      input
  else
    padChannelsFirst2d input padding

lemma get_at_or_zero_paddedInput
    {inC inH inW padding : Nat}
    (img : Spec.Tensor ℝ (.dim inC (.dim inH (.dim inW .scalar)))) (c : Fin inC) (p q : Nat) :
    getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) img)
      [c.val, p, q]
      =
    (if _h : p < padding ∨ q < padding then (0 : ℝ) else getAtOrZero img [c.val, p - padding, q -
      padding]) := by
  classical
  by_cases h0 : padding = 0
  · subst h0
    -- No padding: the cast is definitional and the `if` condition is always false.
    simp [paddedInput]
  · -- Positive padding: reduce to `getAtOrZero_padChannelsFirst2d`.
    simpa [paddedInput, h0] using
      (Spec.getAtOrZero_padChannelsFirst2d (α := ℝ) (input := img) (channel := c) (row := p)
        (col := q) (padding := padding))

lemma mkInputIdx_match_eq_paddedInput
    {inC inH inW stride padding : Nat}
    (img : Spec.Tensor ℝ (.dim inC (.dim inH (.dim inW .scalar)))) (c : Fin inC)
    (oi di oj dj : Nat) :
    (match Spec.Conv.Internal.mkInputIdx? [oi, oj] [di, dj] [stride, stride] [padding, padding] with
      | none => (0 : ℝ)
      | some inIdx => getAtOrZero img (c.val :: inIdx))
      =
    getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) img)
      [c.val, oi * stride + di, oj * stride + dj] := by
  classical
  -- Expand the RHS using the explicit padded-input read formula.
  rw [get_at_or_zero_paddedInput (img := img) (c := c) (p := oi * stride + di) (q := oj * stride + dj)]
  by_cases h0 : oi * stride + di < padding
  · simp [Spec.Conv.Internal.mkInputIdx?, h0]
  · by_cases h1 : oj * stride + dj < padding
    · simp [Spec.Conv.Internal.mkInputIdx?, h0, h1]
    · simp [Spec.Conv.Internal.mkInputIdx?, h0, h1]

lemma sum_shift_eq_paddedInput
    {inC inH inW padding : Nat}
    (x : Spec.Tensor ℝ (.dim inC (.dim inH (.dim inW .scalar)))) (ic : Fin inC) (p q : Nat) :
    (∑ i : Fin inH, ∑ j : Fin inW,
        if (p = i.val + padding ∧ q = j.val + padding) then getAtOrZero x [ic.val, i.val, j.val]
          else 0)
      =
    getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) x)
      [ic.val, p, q] := by
  classical
  -- Compare both sides to the explicit `paddedInput` read formula.
  rw [get_at_or_zero_paddedInput (img := x) (c := ic) (p := p) (q := q)]
  by_cases hp : p < padding ∨ q < padding
  · -- In the padded border: the conjunction can never hold, so the sum is `0`.
    cases hp with
    | inl hp' =>
        have hpOr : p < padding ∨ q < padding := Or.inl hp'
        have hne : ∀ i : Fin inH, p ≠ i.val + padding := by
          intro i
          have hle : padding ≤ i.val + padding := by
            simp [Nat.add_comm]
          exact Nat.ne_of_lt (lt_of_lt_of_le hp' hle)
        have hfalse : ∀ i : Fin inH, ∀ j : Fin inW, ¬(p = i.val + padding ∧ q = j.val + padding) :=
          by
          intro i j h
          exact hne i h.1
        simp [hpOr, hfalse]
    | inr hq' =>
        have hpOr : p < padding ∨ q < padding := Or.inr hq'
        have hne : ∀ j : Fin inW, q ≠ j.val + padding := by
          intro j
          have hle : padding ≤ j.val + padding := by
            simp [Nat.add_comm]
          exact Nat.ne_of_lt (lt_of_lt_of_le hq' hle)
        have hfalse : ∀ i : Fin inH, ∀ j : Fin inW, ¬(p = i.val + padding ∧ q = j.val + padding) :=
          by
          intro i j h
          exact hne j h.2
        simp [hpOr, hfalse]
  · -- Interior: reduce to a shifted read of `x` and collapse the sums using `Fintype.sum_ite_eq'`.
    have hpP : padding ≤ p := by
      have : ¬ p < padding := by
        intro hlt
        exact hp (Or.inl hlt)
      exact Nat.le_of_not_gt this
    have hqP : padding ≤ q := by
      have : ¬ q < padding := by
        intro hlt
        exact hp (Or.inr hlt)
      exact Nat.le_of_not_gt this
    let vp : Nat := p - padding
    let vq : Nat := q - padding
    by_cases hvp : vp < inH
    · by_cases hvq : vq < inW
      · let i0 : Fin inH := ⟨vp, hvp⟩
        let j0 : Fin inW := ⟨vq, hvq⟩
        have hip : p = i0.val + padding := by
          simp [i0, vp, Nat.sub_add_cancel hpP]
        have hjq : q = j0.val + padding := by
          simp [j0, vq, Nat.sub_add_cancel hqP]
        have hiff :
            ∀ (i : Fin inH) (j : Fin inW),
              (p = i.val + padding ∧ q = j.val + padding) ↔ (i = i0 ∧ j = j0) := by
          intro i j
          constructor
          · intro h
            have hiVal : vp = i.val := by
              have := congrArg (fun t => t - padding) h.1
              simpa [vp, Nat.add_sub_cancel] using this
            have hjVal : vq = j.val := by
              have := congrArg (fun t => t - padding) h.2
              simpa [vq, Nat.add_sub_cancel] using this
            have hi : i = i0 := by
              apply Fin.ext
              simpa [i0] using hiVal.symm
            have hj : j = j0 := by
              apply Fin.ext
              simpa [j0] using hjVal.symm
            exact ⟨hi, hj⟩
          · rintro ⟨rfl, rfl⟩
            exact ⟨hip, hjq⟩
        have hrewrite :
            (∑ i : Fin inH, ∑ j : Fin inW,
                if (p = i.val + padding ∧ q = j.val + padding) then getAtOrZero x [ic.val, i.val,
                  j.val] else 0)
              =
            (∑ i : Fin inH, ∑ j : Fin inW,
                if (i = i0 ∧ j = j0) then getAtOrZero x [ic.val, i0.val, j0.val] else 0) := by
          refine Fintype.sum_congr _ _ ?_
          intro i
          refine Fintype.sum_congr _ _ ?_
          intro j
          by_cases hij : i = i0 ∧ j = j0
          · have : p = i.val + padding ∧ q = j.val + padding := (hiff i j).2 hij
            rcases hij with ⟨rfl, rfl⟩
            simp [this]
          · have : ¬(p = i.val + padding ∧ q = j.val + padding) := by
              intro h
              exact hij ((hiff i j).1 h)
            simp [hij, this]
        have hsum :
            (∑ i : Fin inH, ∑ j : Fin inW,
                if (i = i0 ∧ j = j0) then getAtOrZero x [ic.val, i0.val, j0.val] else 0)
              =
            getAtOrZero x [ic.val, i0.val, j0.val] := by
          -- Split the conjunction and use `Fintype.sum_ite_eq'` twice.
          calc
            (∑ i : Fin inH, ∑ j : Fin inW,
                  if (i = i0 ∧ j = j0) then getAtOrZero x [ic.val, i0.val, j0.val] else 0)
                =
              ∑ i : Fin inH,
                if i = i0 then (∑ j : Fin inW, if j = j0 then getAtOrZero x [ic.val, i0.val,
                  j0.val] else 0) else 0 := by
                  refine Fintype.sum_congr _ _ ?_
                  intro i
                  by_cases hi : i = i0 <;> simp [hi]
            _ =
              ∑ i : Fin inH, if i = i0 then getAtOrZero x [ic.val, i0.val, j0.val] else 0 := by
                  refine Fintype.sum_congr _ _ ?_
                  intro i
                  by_cases hi : i = i0 <;> simp [hi]
            _ = getAtOrZero x [ic.val, i0.val, j0.val] := by
                  simp
        -- Finish this case.
        simp [hp, vp, vq, i0, j0, hrewrite, hsum]
      · -- `vq` out of bounds: both sides are `0`.
        have rhs0 : getAtOrZero x [ic.val, vp, vq] = 0 := by
          cases x with
          | dim fC =>
            cases hrow : fC ic with
            | dim fH =>
              cases hcell : fH ⟨vp, hvp⟩ with
              | dim fW =>
                simp [ic.isLt, hvp, hvq, vp, vq, hrow, hcell]
        have hvq' : inW ≤ vq := Nat.le_of_not_gt hvq
        have hfalse : ∀ i : Fin inH, ∀ j : Fin inW, ¬(p = i.val + padding ∧ q = j.val + padding) :=
          by
          intro i j h
          have : vq = j.val := by
            have := congrArg (fun t => t - padding) h.2
            simpa [vq, Nat.add_sub_cancel] using this
          exact (Nat.not_lt_of_ge hvq') (this ▸ j.isLt)
        simp [hp, rhs0, hfalse, vp, vq]
    · -- `vp` out of bounds: both sides are `0`.
      have rhs0 : getAtOrZero x [ic.val, vp, vq] = 0 := by
        cases x with
        | dim fC =>
          cases hrow : fC ic with
          | dim fH =>
            simp [ic.isLt, hvp, vp, vq, hrow]
      have hvp' : inH ≤ vp := Nat.le_of_not_gt hvp
      have hfalse : ∀ i : Fin inH, ∀ j : Fin inW, ¬(p = i.val + padding ∧ q = j.val + padding) := by
        intro i j h
        have : vp = i.val := by
          have := congrArg (fun t => t - padding) h.1
          simpa [vp, Nat.add_sub_cancel] using this
        exact (Nat.not_lt_of_ge hvp') (this ▸ i.isLt)
      simp [hp, rhs0, hfalse, vp, vq]

/-- Collapse the shifted input indicator while carrying a constant scalar factor. -/
lemma sum_shift_mul_eq_paddedInput_mul
    {inC inH inW padding : Nat}
    (x : Spec.Tensor ℝ (.dim inC (.dim inH (.dim inW .scalar))))
    (ic : Fin inC) (p q : Nat) (c : ℝ) :
    (∑ i : Fin inH, ∑ j : Fin inW,
        getAtOrZero x [ic.val, i.val, j.val] *
          (if p = i.val + padding ∧ q = j.val + padding then c else 0)) =
      getAtOrZero
          (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) x)
          [ic.val, p, q] * c := by
  classical
  calc
    (∑ i : Fin inH, ∑ j : Fin inW,
        getAtOrZero x [ic.val, i.val, j.val] *
          (if p = i.val + padding ∧ q = j.val + padding then c else 0)) =
      (∑ i : Fin inH, ∑ j : Fin inW,
        if p = i.val + padding ∧ q = j.val + padding then
          getAtOrZero x [ic.val, i.val, j.val]
        else 0) * c := by
      simp only [mul_ite, mul_zero, Finset.sum_mul, ite_mul, zero_mul]
    _ = getAtOrZero
          (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) x)
          [ic.val, p, q] * c := by
      rw [sum_shift_eq_paddedInput]

/-!
Output shape helpers (no dilation): these are the standard “convolution arithmetic” formulas.
They are kept as definitions so later statements can share the same expression.
-/

def outH (inH kH stride padding : Nat) : Nat :=
  Shape.slidingWindowOutDim inH kH stride padding

/-- Output width helper (no dilation). -/
def outW (inW kW stride padding : Nat) : Nat :=
  Shape.slidingWindowOutDim inW kW stride padding

-- ---------------------------------------------------------------------------
-- Kernel bridge: dot(conv(dKernel, x), δ) = dot(dKernel, ∂L/∂kernel)
-- ---------------------------------------------------------------------------

lemma conv2d_spec_noBias_get
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (dKernel : Tensor ℝ (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))))
    (input : Spec.Tensor ℝ (.dim inC (.dim inH (.dim inW .scalar))))
    (oc : Fin outC) (i : Fin (outH inH kH stride padding)) (j : Fin (outW inW kW stride padding)) :
    let layerK : Spec.Conv2dSpec inC outC kH kW stride padding ℝ h1 h2 h3 :=
      { kernel := dKernel, bias := fill (0 : ℝ) (.dim outC .scalar) }
    getAtOrZero (Spec.conv2dSpec (α := ℝ) (layer := layerK) input) [oc.val, i.val, j.val]
      =
    ∑ ic : Fin inC,
      ∑ di : Fin kH,
        ∑ dj : Fin kW,
          getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
            getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding)
              input)
              [ic.val, i.val * stride + di.val, j.val * stride + dj.val] := by
  intro layerK
  classical
  unfold Spec.conv2dSpec
  have hi : i.val < Shape.slidingWindowOutDim inH kH stride padding := by
    simpa [outH] using i.isLt
  have hj : j.val < Shape.slidingWindowOutDim inW kW stride padding := by
    simpa [outW] using j.isLt
  -- Peel the requested output entry and convert the nested `finRange` folds into `Finset` sums.
  simp [outH, outW, layerK, fill, getAtOrZero, oc.isLt, hi, hj,
    List.finRange_foldl_add_acc]
  -- Rewrite the `mkInputIdx?`-based read into the `paddedInput` helper, then commute the product
  -- so the summand matches the statement (`kernel * paddedInput`).
  refine Finset.sum_congr (M := ℝ)
    (rfl : (Finset.univ : Finset (Fin inC)) = Finset.univ) ?_
  intro ic _
  refine Finset.sum_congr (M := ℝ)
    (rfl : (Finset.univ : Finset (Fin kH)) = Finset.univ) ?_
  intro di _
  refine Finset.sum_congr (M := ℝ)
    (rfl : (Finset.univ : Finset (Fin kW)) = Finset.univ) ?_
  intro dj _
  have hread :=
    mkInputIdx_match_eq_paddedInput (stride := stride) (padding := padding) (img := input) (c := ic)
      (oi := i.val) (di := di.val) (oj := j.val) (dj := dj.val)
  -- Expand the match using `hread`, then commute the scalar product so the summand matches the
  -- statement’s `kernel * paddedInput` order.
  calc
    (match Spec.Conv.Internal.mkInputIdx? [i.val, j.val] [di.val, dj.val] [stride, stride] [padding, padding] with
        | none => 0
        | some inIdx => getAtOrZero input (ic.val :: inIdx)) *
        getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val]
        =
      getAtOrZero
          (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) input)
          [ic.val, i.val * stride + di.val, j.val * stride + dj.val] *
        getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] := by
      simp [hread]
    _ =
      getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
        getAtOrZero
          (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) input)
          [ic.val, i.val * stride + di.val, j.val * stride + dj.val] := by
      simpa using
        (mul_comm
          (getAtOrZero
            (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding) input)
            [ic.val, i.val * stride + di.val, j.val * stride + dj.val])
          (getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val]))

lemma conv2d_kernel_deriv_get
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (layer : Spec.Conv2dSpec inC outC kH kW stride padding ℝ h1 h2 h3)
    (input : Spec.Tensor ℝ (.dim inC (.dim inH (.dim inW .scalar))))
    (δ : Spec.Tensor ℝ (.dim outC (.dim (outH inH kH stride padding) (.dim (outW inW kW stride padding) .scalar))))
    (oc : Fin outC) (ic : Fin inC) (di : Fin kH) (dj : Fin kW) :
    getAtOrZero (Spec.conv2dKernelDerivSpec (α := ℝ) (layer := layer) (input := input)
      (grad_output := δ))
        [oc.val, ic.val, di.val, dj.val]
      =
    ∑ i : Fin (outH inH kH stride padding),
      ∑ j : Fin (outW inW kW stride padding),
        getAtOrZero (paddedInput (inC := inC) (inH := inH) (inW := inW) (padding := padding)
          input)
          [ic.val, i.val * stride + di.val, j.val * stride + dj.val] *
          getAtOrZero δ [oc.val, i.val, j.val] := by
  classical
  unfold Spec.conv2dKernelDerivSpec
  simp (config := { maxSteps := 2000000 })
    [List.finRange_foldl_add_acc, outH, outW]
  apply Finset.sum_congr
  · rfl
  intro i _
  apply Finset.sum_congr
  · rfl
  intro j _
  have hread :=
    mkInputIdx_match_eq_paddedInput (stride := stride) (padding := padding) (img := input)
    (c := ic) (oi := i.val) (di := di.val) (oj := j.val) (dj := dj.val)
  exact congrArg (fun x => x * getAtOrZero δ [oc.val, i.val, j.val]) hread

lemma conv2d_input_deriv_get
    {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (layer : Spec.Conv2dSpec inC outC kH kW stride padding ℝ h1 h2 h3)
    (input : Spec.Tensor ℝ (.dim inC (.dim inH (.dim inW .scalar))))
    (δ : Spec.Tensor ℝ (.dim outC (.dim (outH inH kH stride padding) (.dim (outW inW kW stride padding) .scalar))))
    (ic : Fin inC) (i : Fin inH) (j : Fin inW) :
    getAtOrZero (Spec.conv2dInputDerivSpec (α := ℝ) (layer := layer) (input := input)
      (grad_output := δ))
        [ic.val, i.val, j.val]
      =
    ∑ oc : Fin outC,
      ∑ oi : Fin (outH inH kH stride padding),
        ∑ oj : Fin (outW inW kW stride padding),
          ∑ di : Fin kH,
            ∑ dj : Fin kW,
              (if _h :
                  (oi.val * stride + di.val = i.val + padding) ∧
                  (oj.val * stride + dj.val = j.val + padding) then
                getAtOrZero δ [oc.val, oi.val, oj.val] *
                  getAtOrZero layer.kernel [oc.val, ic.val, di.val, dj.val]
              else
                0) := by
  classical
  unfold Spec.conv2dInputDerivSpec
  simp only [getAtOrZero, ic.isLt, i.isLt, j.isLt, ↓reduceDIte]
  simp only [List.finRange_foldl_add_acc, zero_add]
  rfl


end

end Conv2d
end Autograd
end Proofs
