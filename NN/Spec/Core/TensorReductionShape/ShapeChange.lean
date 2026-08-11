/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Constructors
public import NN.Spec.Core.Tensor.Linalg
public import NN.Spec.Core.TensorOps

@[expose] public section


namespace Spec
namespace Tensor

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-!
# Shape Changing

Flatten, unflatten, reshape, and small construction helpers for shape-indexed tensors.
-/


/-- Flatten a tensor into a 1‑D vector (length = `Spec.Shape.size s`).

The order is outermost‑dimension major (row‑major w.r.t. the shape tree).
For proofs, the key invariant is that the output length matches `Spec.Shape.size`.

Why this exists: a lot of shape-changing ops are easiest to specify as "flatten, then rebuild",
and this is also the bridge we use for some runtime interop where we want a plain sequence of
scalars (e.g. importing weights or serializing test vectors).
-/
def flattenSpec {α : Type} [Inhabited α] : ∀ {s : Shape}, Tensor α s → Tensor α (.dim (Spec.Shape.size
  s) .scalar)
| Shape.scalar, Tensor.scalar x =>
  Tensor.dim (fun i =>
    have _ : i.val < 1 := i.isLt
    if i.val = 0 then Tensor.scalar x else Tensor.scalar (Inhabited.default))
| Shape.dim n s', Tensor.dim f =>
  let _ := n * Spec.Shape.size s'
  Tensor.dim (fun i =>
    let outerIdx := i.val / (Spec.Shape.size s')
    let innerIdx := i.val % (Spec.Shape.size s')
    if h1 : outerIdx < n then
      if _ : innerIdx < (Spec.Shape.size s') then
        let innerTensor := flattenSpec (f ⟨outerIdx, h1⟩)
        match innerTensor with
        | Tensor.dim g =>
          if h3 : innerIdx < (Spec.Shape.size s') then
            g ⟨innerIdx, h3⟩
          else
            Tensor.scalar (Inhabited.default)
      else
        Tensor.scalar (Inhabited.default)
    else
      Tensor.scalar (Inhabited.default))

/-- Unflatten a 1‑D vector back into a tensor of a given shape.

PyTorch analogy: `flat.view(shape)` (assuming the element count matches).
This is the inverse of `flattenSpec` up to the ordering convention. -/
def unflattenSpec {α : Type} [Inhabited α] : ∀ (s : Shape), Tensor α (.dim (Spec.Shape.size s) .scalar)
  → Tensor α s
| Shape.scalar, Tensor.dim f =>
  -- `Spec.Shape.size Shape.scalar = 1`, so the input always has an element at index `0`.
  -- Matching directly avoids extra proof obligations downstream.
  match f ⟨0, by simp [Spec.Shape.size]⟩ with
  | Tensor.scalar x => Tensor.scalar x
| Shape.dim n s', Tensor.dim f =>
  Tensor.dim (fun i =>
    -- For each position i in the outer dimension, extract a sub-tensor
    let startIdx := i.val * (Spec.Shape.size s')
    let subTensor : Tensor α (.dim (Spec.Shape.size s') .scalar) :=
      Tensor.dim (fun j =>
        let globalIdx := startIdx + j.val
        if h : globalIdx < n * (Spec.Shape.size s') then
          f ⟨globalIdx, h⟩
        else
          Tensor.scalar (Inhabited.default))
    unflattenSpec s' subTensor)

/-!
## `flattenSpec` / `unflattenSpec` round-trip lemmas

These are shape-transport facts: they justify treating `flattenSpec`/`unflattenSpec` like
`reshape`/`view` in PyTorch, provided you keep the element count consistent.

PyTorch references:
- `torch.flatten`: https://pytorch.org/docs/stable/generated/torch.flatten.html
- `Tensor.view` / `torch.reshape`: https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
- `torch.reshape`: https://pytorch.org/docs/stable/generated/torch.reshape.html

Important nuance:
- PyTorch allows zero-sized dimensions, and its reshape/flatten semantics remain total.
- Our spec definitions are also total (they use `Inhabited.default` for unreachable branches),
  which keeps everything executable, but can make “inverse” proofs a bit index-heavy.
  The theorems below show that the round-trips do work for the spec definitions as written.
-/

namespace Private

/--
Helper lemma: `flattenSpec` on an outer `Tensor.dim` agrees with flattening a chosen slice.

This is used to prove `unflattenSpec s (flattenSpec t) = t` by reducing the statement to the
induction hypothesis on each slice.
-/
private lemma flattenSpec_dim_apply {α : Type} [Inhabited α] {n : Nat} {s : Shape}
    (f : Fin n → Tensor α s) (i : Fin n) (j : Fin (Spec.Shape.size s))
    (hmpos : 0 < Spec.Shape.size s)
    (hidx : i.val * Spec.Shape.size s + j.val < n * Spec.Shape.size s) :
    (match flattenSpec (Tensor.dim f) with
      | Tensor.dim g => g ⟨i.val * Spec.Shape.size s + j.val, hidx⟩) =
    (match flattenSpec (f i) with
      | Tensor.dim g => g j) := by
  have hdiv : (i.val * Spec.Shape.size s + j.val) / Spec.Shape.size s = i.val := by
    calc
      (i.val * Spec.Shape.size s + j.val) / Spec.Shape.size s
          = (Spec.Shape.size s * i.val + j.val) / Spec.Shape.size s := by
              simp [Nat.mul_comm]
      _ = i.val + j.val / Spec.Shape.size s := by
            simpa using (Nat.mul_add_div (m := Spec.Shape.size s) hmpos i.val j.val)
      _ = i.val := by
            simp [Nat.div_eq_of_lt j.isLt]
  have hmod : (i.val * Spec.Shape.size s + j.val) % Spec.Shape.size s = j.val :=
    Nat.mul_add_mod_of_lt (a := i.val) (b := Spec.Shape.size s) (c := j.val) j.isLt

  have houter : (i.val * Spec.Shape.size s + j.val) / Spec.Shape.size s < n := by
    simp [hdiv]
  have hinner : (i.val * Spec.Shape.size s + j.val) % Spec.Shape.size s < Spec.Shape.size s := by
    simp [hmod]

  have hfin_outer : (⟨(i.val * Spec.Shape.size s + j.val) / Spec.Shape.size s, houter⟩ : Fin n) = i := by
    apply Fin.ext
    simp [hdiv]
  have hfin_inner :
      (⟨(i.val * Spec.Shape.size s + j.val) % Spec.Shape.size s, hinner⟩ : Fin (Spec.Shape.size s)) = j := by
    apply Fin.ext
    simp [hmod]

  simp [flattenSpec, hdiv, hmod]

/-!
If a shape has `Spec.Shape.size s = 0`, then it contains **no scalar leaves** (it has a `0`-length
dimension somewhere). In that case, there is essentially only one possible tensor value of shape
`s` (up to definitional equality), because at the `0`-length dimension the indexing function has
domain `Fin 0`.

We use this as a “vacuity” lemma to avoid needing division/modulo arithmetic when `Spec.Shape.size s =
  0`.
-/
/-- If `Spec.Shape.size s = 0`, then any two tensors of shape `s` are equal (vacuity via `Fin 0`). -/
private theorem tensor_eq_of_size_zero {α : Type} :
    ∀ {s : Shape}, Spec.Shape.size s = 0 → (x y : Tensor α s) → x = y
  | .scalar, h, _x, _y => by
      simp [Spec.Shape.size] at h
  | .dim n s, h, x, y => by
      cases x with
      | dim fx =>
          cases y with
          | dim fy =>
              cases n with
              | zero =>
                  apply congrArg Tensor.dim
                  funext i
                  exact Fin.elim0 i
              | succ n =>
                  have hs0 : Spec.Shape.size s = 0 := by
                    have : (Nat.succ n = 0) ∨ (Spec.Shape.size s = 0) := Nat.mul_eq_zero.mp h
                    exact this.resolve_left (Nat.succ_ne_zero n)
                  apply congrArg Tensor.dim
                  funext i
                  exact tensor_eq_of_size_zero (α := α) (s := s) hs0 (fx i) (fy i)

end Private

/--
Round-trip `unflatten ∘ flatten = id`.

This is the spec-layer analogue of `reshape`/`view` round-tripping in PyTorch when the element
count matches.
-/
theorem flatten_unflatten_inverse {α : Type} [Inhabited α] :
    ∀ {s : Shape}, (t : Tensor α s) → unflattenSpec s (flattenSpec t) = t
  | .scalar, t => by
      cases t with
      | scalar x =>
          change unflattenSpec Shape.scalar (Tensor.dim fun _ => Tensor.scalar x) = Tensor.scalar x
          rfl
  | .dim n s, t => by
      cases t with
      | dim f =>
          cases hflat : flattenSpec (Tensor.dim f) with
          | dim flat =>
              simp [unflattenSpec]
              funext i
              by_cases hm : Spec.Shape.size s = 0
              ·
                exact
                  Private.tensor_eq_of_size_zero (α := α) (s := s) hm
                    (unflattenSpec s
                      (Tensor.dim (fun j : Fin (Spec.Shape.size s) =>
                        if hIdx : i.val * Spec.Shape.size s + j.val < n * Spec.Shape.size s then
                          flat ⟨i.val * Spec.Shape.size s + j.val, hIdx⟩
                        else
                          Tensor.scalar Inhabited.default)))
                    (f i)
              ·
                have hmpos : 0 < Spec.Shape.size s := Nat.pos_of_ne_zero hm
                have sub_eq :
                    (Tensor.dim (fun j : Fin (Spec.Shape.size s) =>
                      if hIdx : i.val * Spec.Shape.size s + j.val < n * Spec.Shape.size s then
                        flat ⟨i.val * Spec.Shape.size s + j.val, hIdx⟩
                      else
                        Tensor.scalar Inhabited.default))
                      = flattenSpec (f i) := by
                  cases hfi : flattenSpec (f i) with
                  | dim gfi =>
                      apply congrArg Tensor.dim
                      funext j
                      have hidx : i.val * Spec.Shape.size s + j.val < n * Spec.Shape.size s := by
                        have hisucc : i.val + 1 ≤ n := Nat.succ_le_of_lt i.isLt
                        have hlt :
                            i.val * Spec.Shape.size s + j.val < (i.val + 1) * Spec.Shape.size s := by
                          have := Nat.add_lt_add_left j.isLt (i.val * Spec.Shape.size s)
                          simp [Nat.succ_mul]
                        have hle : (i.val + 1) * Spec.Shape.size s ≤ n * Spec.Shape.size s :=
                          Nat.mul_le_mul_right (Spec.Shape.size s) hisucc
                        exact Nat.lt_of_lt_of_le hlt hle
                      simp [hidx]
                      have :=
                        Private.flattenSpec_dim_apply (α := α) (f := f) (i := i) (j := j)
                          (hmpos := hmpos) (hidx := hidx)
                      simpa [hflat, hfi] using this
                simpa [sub_eq] using
                  (flatten_unflatten_inverse (α := α) (s := s) (t := f i))

/--
Round-trip `flatten ∘ unflatten = id`.

This is the spec-layer analogue of flattening a reshaped/viewed tensor in PyTorch.
-/
theorem unflatten_flatten_inverse {α : Type} [Inhabited α] :
    ∀ {s : Shape}, (v : Tensor α (.dim (Spec.Shape.size s) .scalar)) → flattenSpec (unflattenSpec s v)
      = v
  | .scalar, v => by
      cases v with
      | dim f =>
          let idx0 : Fin Shape.scalar.size := ⟨0, by simp [Spec.Shape.size]⟩
          cases h0 : f idx0 with
          | scalar x =>
              have hunflat : unflattenSpec Shape.scalar (Tensor.dim f) = Tensor.scalar x := by
                simp [unflattenSpec, idx0, h0]
              rw [hunflat]
              simp [flattenSpec, Spec.Shape.size]
              apply congrArg Tensor.dim
              funext i
              have hival : i.val = 0 := by
                have : i.val < 1 := by simp
                have : i.val ≤ 0 := by simp
                exact Nat.eq_zero_of_le_zero this
              have hi : i = idx0 := by
                apply Fin.ext
                simp [idx0]
              simpa [hi] using h0.symm
  | .dim n s, v => by
      cases v with
      | dim g =>
          by_cases hm : Spec.Shape.size s = 0
          ·
            cases hflat : flattenSpec (unflattenSpec (Shape.dim n s) (Tensor.dim g)) with
            | dim gf =>
                apply congrArg Tensor.dim
                funext idx
                have : False := by
                  have : idx.val < 0 := by simpa [Spec.Shape.size, hm] using idx.isLt
                  exact Nat.not_lt_zero _ this
                exact False.elim this
          ·
            let m : Nat := Spec.Shape.size s
            have hmpos : 0 < m := by
              have : m ≠ 0 := by simpa [m] using hm
              exact Nat.pos_of_ne_zero this
            have hunflat :
                unflattenSpec (Shape.dim n s) (Tensor.dim g) =
                  Tensor.dim (fun i : Fin n =>
                    let startIdx := i.val * m
                    let subTensor : Tensor α (.dim m .scalar) :=
                      Tensor.dim (fun j : Fin m =>
                        let globalIdx := startIdx + j.val
                        if h : globalIdx < n * m then
                          g ⟨globalIdx, h⟩
                        else
                          Tensor.scalar (Inhabited.default))
                    unflattenSpec s subTensor) := by
              rfl
            cases hflat : flattenSpec (unflattenSpec (Shape.dim n s) (Tensor.dim g)) with
            | dim gf =>
                have hflat' :
                    flattenSpec
                        (Tensor.dim (fun i : Fin n =>
                          let startIdx := i.val * m
                          let subTensor : Tensor α (.dim m .scalar) :=
                            Tensor.dim (fun j : Fin m =>
                              let globalIdx := startIdx + j.val
                              if h : globalIdx < n * m then
                                g ⟨globalIdx, h⟩
                              else
                                Tensor.scalar (Inhabited.default))
                          unflattenSpec s subTensor))
                      = Tensor.dim gf := by
                  simpa [hunflat] using hflat
                apply congrArg Tensor.dim
                funext idx
                let oi : Nat := idx.val / m
                let ij : Nat := idx.val % m
                have hoi : oi < n := by
                  have : idx.val < n * m := idx.isLt
                  exact (Nat.div_lt_iff_lt_mul hmpos).2 (by simpa [oi] using this)
                have hij : ij < m := by
                  simpa [ij] using Nat.mod_lt idx.val hmpos
                let i : Fin n := ⟨oi, hoi⟩
                let j : Fin m := ⟨ij, hij⟩
                have hrecomp : i.val * m + j.val = idx.val := by
                  simp [i, j, oi, ij, Nat.div_add_mod']
                have hidx : i.val * m + j.val < n * m :=
                  lt_of_eq_of_lt hrecomp idx.isLt
                have hfin : (⟨i.val * m + j.val, hidx⟩ : Fin (n * m)) = idx := by
                  apply Fin.ext
                  simp [hrecomp]

                have hcoord :=
                  Private.flattenSpec_dim_apply
                    (α := α)
                    (f := fun i : Fin n =>
                      let startIdx := i.val * m
                      let subTensor : Tensor α (.dim m .scalar) :=
                        Tensor.dim (fun j : Fin m =>
                          let globalIdx := startIdx + j.val
                          if h : globalIdx < n * m then
                            g ⟨globalIdx, h⟩
                          else
                            Tensor.scalar (Inhabited.default))
                      unflattenSpec s subTensor)
                    (i := i) (j := j) (hmpos := hmpos) (hidx := hidx)

                have hsub :
                    flattenSpec
                        (unflattenSpec s
                          (Tensor.dim (fun j : Fin m =>
                            let globalIdx := i.val * m + j.val
                            if h : globalIdx < n * m then
                              g ⟨globalIdx, h⟩
                            else
                              Tensor.scalar (Inhabited.default))))
                      =
                    Tensor.dim (fun j : Fin m =>
                      let globalIdx := i.val * m + j.val
                      if h : globalIdx < n * m then
                        g ⟨globalIdx, h⟩
                      else
                        Tensor.scalar (Inhabited.default)) := by
                  simpa [i, unflattenSpec] using
                    (unflatten_flatten_inverse (α := α) (s := s)
                      (v := Tensor.dim (fun j : Fin m =>
                        let globalIdx := i.val * m + j.val
                        if h : globalIdx < n * m then
                          g ⟨globalIdx, h⟩
                        else
                          Tensor.scalar (Inhabited.default))))

                have hcoord' :
                    gf ⟨i.val * m + j.val, hidx⟩ =
                      (match flattenSpec
                          (unflattenSpec s
                            (Tensor.dim (fun j : Fin m =>
                              let globalIdx := i.val * m + j.val
                              if h : globalIdx < n * m then
                                g ⟨globalIdx, h⟩
                              else
                                Tensor.scalar (Inhabited.default))))
                        with
                        | Tensor.dim g' => g' j) := by
                  simpa [hflat'] using hcoord

                have hgf0 : gf ⟨i.val * m + j.val, hidx⟩ = g ⟨i.val * m + j.val, hidx⟩ := by
                  simpa [hsub, hidx] using hcoord'
                have hgf' : gf ⟨i.val * m + j.val, hidx⟩ = g idx := by
                  exact hgf0.trans (congrArg g hfin)

                calc
                  gf idx = gf ⟨i.val * m + j.val, hidx⟩ := by
                    exact (congrArg gf hfin).symm
                  _ = g idx := hgf'

/--
Convenience corollary: the `unflatten ∘ flatten` round-trip in the common well-formed regime.
-/
theorem flatten_unflatten_inverse_wf {α : Type} [Inhabited α] {s : Shape}
    [Shape.WellFormed s] (t : Tensor α s) :
    unflattenSpec s (flattenSpec t) = t := by
  simpa using (flatten_unflatten_inverse (α := α) (s := s) (t := t))

/-- Reshape a tensor, given a proof that the number of elements matches. -/
def reshapeSpec {α : Type} [Inhabited α]
  {s₁ s₂ : Shape} (t : Tensor α s₁) (h : s₁.size = s₂.size) : Tensor α s₂ :=
  let flattened := flattenSpec t
  let retyped : Tensor α (.dim (Spec.Shape.size s₂) .scalar) :=
    Eq.recOn h flattened
  unflattenSpec s₂ retyped

/-- Reshape with an explicit equality rewrite (sometimes easier for the elaborator). -/
def reshapeExplicitSpec {α : Type} [Inhabited α] {s₁ s₂ : Shape} (t : Tensor α s₁)
  (h : s₁.size = s₂.size) : Tensor α s₂ :=
  let flattened := flattenSpec t
  let retyped : Tensor α (.dim (Spec.Shape.size s₂) .scalar) :=
    by rw [h.symm]; exact flattened
  unflattenSpec s₂ retyped

/-- Given a partial function `Fin n → Option (Tensor α s)`, build a tensor if all succeed. -/
def sequenceFin {s : Shape} {n : Nat}
  (f : Fin n → Option (Tensor α s)) : Option (Tensor α (.dim n s)) :=
  -- This is basically `Option`-sequencing for `Fin n → _`.
  -- We use it when a shape-level construction can fail (e.g. dynamic runtime checks),
  -- but we still want a *total* spec API (failure is explicit in the `Option`).
  --
  -- Implementation note: we avoid `Array.get!` / `arr[i]!` by building the result function
  -- directly via recursion on `n` (using `Fin.cases`).
  match n with
  | 0 =>
      -- A 0-length `Tensor.dim` is inhabited by the empty function.
      some (Tensor.dim (fun i => nomatch i))
  | n' + 1 =>
      match f ⟨0, Nat.succ_pos n'⟩ with
      | none => none
      | some t0 =>
          match sequenceFin (n := n') (fun j => f j.succ) with
          | none => none
          | some (Tensor.dim g) =>
              some (Tensor.dim (fun i => Fin.cases t0 g i))
              -- Note: the `Tensor.scalar` case is impossible since the shape is `.dim _ _`.

/-- Build a tensor filled with a constant, without using `fill` (used in broadcasts). -/
def broadcastFill {α : Type} [Inhabited α] : ∀ (s : Shape), α → Tensor α s
| .scalar, v => scalar v
| Shape.dim _ s', v => dim (fun _ => broadcastFill s' v)
end Tensor
end Spec
