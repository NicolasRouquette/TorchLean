/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Chain.Syntax
public import NN.GraphSpec.DAG.Core
import Mathlib.Algebra.Order.Algebra

/-!
# Structural conversion of sequential GraphSpec chains to DAG terms

This module embeds each sequential primitive as a DAG primitive and lowers composition to
explicit SSA-style let bindings.
-/

@[expose] public section

namespace NN
namespace GraphSpec

open _root_.Spec
open Spec.Tensor
open _root_.TorchLean.Tensor

/-! ## Lowering: sequential chain → DAG term -/

/-!
GraphSpec has two surface syntaxes:

- `NN.GraphSpec.Core`: a *sequential* DSL (`Chain` + `>>>`), ideal for pure pipelines.
- `NN.GraphSpec.DAG.Core`: a *general* SSA/A-normal-form term language, ideal for sharing/skip
  connections.

The DAG term language is GraphSpec’s “general graph” core: it is the representation that can
express sharing and skip connections.

`Chain` exists because it is the clearest way to write pipelines, and it has its own
direct Spec semantics (`Interp.spec`) and program translation (`Chain.toProgram`).

This lowering is still useful whenever you want to *embed* a sequential pipeline into the DAG world
(e.g. to reuse DAG-only tooling, or to keep a single GraphSpec example surface that can export DAG
models).

The declarations below provide a structural lowering:

- `Chain.toDAGTerm` produces a `DAG.Term (ps ++ [σ]) τ`, i.e. a DAG term whose environment starts
  with the parameter list `ps` and ends with the (single) data input `σ`.
Notes:
- The lowering is *purely structural*: it introduces `let1` binders between stages to make the
  sequential flow explicit in SSA form.
- Each sequential `Primitive ps σ τ` is embedded as a DAG primitive op with inputs `ps ++ [σ]`.
  This embedding is generic: any custom GraphSpec primitive automatically becomes usable in the
  DAG world.
-/

namespace LowerToDAG

/-!
### Lowering internals

The definitions below (`castTerm`, `toTerm`, …) are adapters for the structural lowering. The
principal entry point is `Chain.toDAGTerm`.
-/

/-- Cast a `DAG.Term` across a proven equality of output shapes. -/
def castTerm {Γ : List Shape} {s t : Shape} (h : s = t) :
    DAG.Term Γ s → DAG.Term Γ t :=
  fun x => DAG.Term.cast x h

/-- Cast the environment of a `DAG.Term` across a proven equality of environments. -/
def castEnvTerm {Γ Γ' : List Shape} {τ : Shape} (h : Γ = Γ') :
    DAG.Term Γ τ → DAG.Term Γ' τ :=
  fun x => DAG.Term.castEnv x h

/-- Cast the environment of `DAG.Args` across a proven equality of environments. -/
def castEnvArgs {Γ Γ' : List Shape} {ins : List Shape} (h : Γ = Γ') :
    DAG.Args Γ ins → DAG.Args Γ' ins := by
  cases h
  intro xs
  exact xs

/-! ### `List.get` lemmas (small, self-contained) -/

/-- `List.get` into `as` is unchanged by appending a right list (Nat-index form). -/
lemma get_append_left_nat {α : Type} :
    ∀ (as bs : List α) (i : Nat) (hi : i < as.length),
      (as ++ bs).get ⟨i, by
        simpa [List.length_append] using Nat.lt_of_lt_of_le hi (Nat.le_add_right _ _)⟩
      =
      as.get ⟨i, hi⟩
  | [], _bs, _i, hi => by simp at hi
  | _a :: as, bs, 0, _hi => rfl
  | _a :: as, bs, (i + 1), hi => by
      have hi' : i < as.length := Nat.lt_of_succ_lt_succ hi
      -- Reduce to the tail case.
      -- (The definitional reduction of `List.get` on a successor index handles the index-shift.)
      exact get_append_left_nat as bs i hi'

/--
`List.get` into the right list after appending, using an explicit offset `as.length + j`
(Nat-index form).
-/
lemma get_append_right_offset_nat {α : Type} :
    ∀ (as bs : List α) (j : Nat) (hj : as.length + j < (as ++ bs).length),
      (as ++ bs).get ⟨as.length + j, hj⟩
      =
      bs.get ⟨j, by
        have : as.length + j < as.length + bs.length := by
          simpa [List.length_append] using hj
        exact Nat.lt_of_add_lt_add_left this⟩
  | [], bs, j, _hj => by
      -- `[] ++ bs = bs`, and the two `Fin` proofs are propositionally equal.
      let idxL : Fin bs.length := ⟨j, by simpa using _hj⟩
      let idxR : Fin bs.length := ⟨j, by
        have : ([] : List α).length + j < ([] : List α).length + bs.length := by
          simpa using (by simpa [List.nil_append] using _hj)
        exact Nat.lt_of_add_lt_add_left this⟩
      have hIdx : idxL = idxR := by
        apply Fin.ext
        rfl
      -- Both sides are `bs.get` at the same index.
      simp [idxL, idxR, hIdx]
  | a :: as, bs, j, hj => by
      -- Re-express the index as a successor so `List.get` reduces to the tail.
      have hIdx2 : Nat.succ (as.length + j) < ((a :: as) ++ bs).length := by
        simpa [List.length_append, Nat.succ_add, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]
          using hj
      let idx1 : Fin (((a :: as) ++ bs).length) := ⟨(a :: as).length + j, hj⟩
      let idx2 : Fin (((a :: as) ++ bs).length) := ⟨Nat.succ (as.length + j), hIdx2⟩
      have hidx : idx1 = idx2 := by
        apply Fin.ext
        simp [idx1, idx2, List.length, Nat.succ_add, Nat.add_assoc, Nat.add_comm]
      have hj' : as.length + j < (as ++ bs).length :=
        Nat.lt_of_succ_lt_succ (by
          -- `((a :: as) ++ bs).length = Nat.succ ((as ++ bs).length)`
          simpa [List.length_append] using hIdx2)
      calc
        ((a :: as) ++ bs).get idx1
            = ((a :: as) ++ bs).get idx2 := by simp [hidx]
        _ = (as ++ bs).get ⟨as.length + j, hj'⟩ := by
              -- definitional reduction of `List.get` on a successor index
              rfl
        _ = bs.get ⟨j, by
              have : as.length + j < as.length + bs.length := by
                simpa [List.length_append] using hj'
              exact Nat.lt_of_add_lt_add_left this⟩ := by
              exact get_append_right_offset_nat as bs j hj'

/-- `List.get` of the last element after appending a singleton list. -/
lemma get_append_last {α : Type} :
    ∀ (xs : List α) (x : α),
      (xs ++ [x]).get ⟨xs.length, by simp [List.length_append]⟩ = x
  | [], x => rfl
  | _a :: xs, x => by
      -- Reduce to tail.
      simp

/-! ### Primitive embedding: `Primitive` → `DAG.PrimOp` -/

/--
Embed a sequential GraphSpec primitive as a DAG primitive op.

The resulting op has input shapes `ps ++ [σ]` (parameters followed by the data input).
 -/
def Primitive.toDAGPrimOp {ps : List Shape} {σ τ : Shape} (p : Primitive ps σ τ) :
    DAG.PrimOp (ps ++ [σ]) τ :=
  { name := p.name
    specFwd := fun {α} _ctx xs =>
      let (params, xs') :=
        TorchLean.TensorPack.split
          (α := α) (ss₁ := ps) (ss₂ := [σ]) xs
      match xs' with
      | .cons x .nil => p.specFwd (α := α) params x
    program := fun {α} _ctx _deq => p.program (α := α)
  }

/-! ### Building well-typed DAG arguments for a primitive call -/

lemma get_succ
    {α : Type} (a : α) (as : List α) (i : Fin as.length) :
    (a :: as).get ⟨i.1 + 1, Nat.succ_lt_succ i.2⟩ = as.get i := by
  cases i with
  | mk i hi =>
    -- `List.get` on a successor index reduces definitionally to the tail.
    rfl

/--
Build a typed `DAG.Args` list from an index-based family of argument terms.

This is the bridge from “arguments as a function of `Fin ins.length`” to the inductive `DAG.Args`
encoding used by `DAG.Term.op`.
-/
def argsOfFn {Γ : List Shape} :
    {ins : List Shape} →
    (∀ i : Fin ins.length, DAG.Term Γ (ins.get i)) →
    DAG.Args Γ ins
  | [], _f => .nil
  | s :: ss, f =>
      -- Head: index 0.
      let head : DAG.Term Γ ((s :: ss).get ⟨0, by simp⟩) := f ⟨0, by simp⟩
      -- Tail: shift indices by 1, and cast the `List.get` result to match `ss.get i`.
      let tail : DAG.Args Γ ss :=
        argsOfFn (ins := ss) (fun i =>
          castTerm (Γ := Γ) (s := (s :: ss).get ⟨i.1 + 1, Nat.succ_lt_succ i.2⟩) (t := ss.get i)
            (get_succ (a := s) (as := ss) i) (f ⟨i.1 + 1, Nat.succ_lt_succ i.2⟩))
      .cons (by simpa using head) tail

/-- Append one final term to a typed DAG argument list. -/
def Args.append1 {Γ : List Shape} {ps : List Shape} {σ : Shape} :
    DAG.Args Γ ps → DAG.Term Γ σ → DAG.Args Γ (ps ++ [σ])
  | .nil, x => .cons x .nil
  | .cons t ts, x => .cons t (Args.append1 ts x)

/--
Reference the `i`th parameter block inside a larger environment layout.

The surrounding environment is split as `pre ++ ps ++ post ++ extra`; this helper returns the term
that points at parameter `i : Fin ps.length` while keeping the full ambient environment explicit.
-/
def mkParamTerm
    {pre ps post extra : List Shape}
    (i : Fin ps.length) :
    DAG.Term ((pre ++ ps ++ post) ++ extra) (ps.get i) := by
  let Γ : List Shape := (pre ++ ps ++ post) ++ extra
  let n : Nat := pre.length + i.1
  have hPrePsPost : n < (pre ++ ps ++ post).length := by
    have hi' : i.1 < (ps ++ post).length := by
      have : i.1 < ps.length + post.length := Nat.lt_of_lt_of_le i.2 (Nat.le_add_right _ _)
      simpa [List.length_append, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using this
    have : pre.length + i.1 < pre.length + (ps ++ post).length :=
      Nat.add_lt_add_left hi' pre.length
    simpa [n, List.length_append] using this
  have hΓ : n < Γ.length := by
    have : n < (pre ++ ps ++ post).length + extra.length :=
      Nat.lt_of_lt_of_le hPrePsPost (Nat.le_add_right _ _)
    simpa [Γ, List.length_append, Nat.add_assoc] using this
  let idx : Fin Γ.length := ⟨n, hΓ⟩
  have hGet : Γ.get idx = ps.get i := by
    -- Drop the trailing `extra`.
    have hExtra :
        Γ.get idx
          =
        (pre ++ ps ++ post).get ⟨n, hPrePsPost⟩ := by
      have hIdx :
          idx
            =
          ⟨n, by
            have : n < (pre ++ ps ++ post).length + extra.length :=
              Nat.lt_of_lt_of_le hPrePsPost (Nat.le_add_right _ _)
            simpa [Γ, List.length_append, Nat.add_assoc] using this⟩ := by
        apply Fin.ext
        rfl
      simpa [Γ, hIdx] using
        (get_append_left_nat (as := (pre ++ ps ++ post)) (bs := extra) (i := n) (hi := hPrePsPost))
    -- Strip the leading `pre` inside `pre ++ (ps ++ post)`.
    have hPre :
        (pre ++ ps ++ post).get ⟨n, hPrePsPost⟩
          =
        (ps ++ post).get ⟨i.1, by
          have : i.1 < (ps ++ post).length := by
            have : i.1 < ps.length + post.length := Nat.lt_of_lt_of_le i.2 (Nat.le_add_right _ _)
            simpa [List.length_append, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using this
          simpa [List.length_append] using this⟩ := by
      have hj : pre.length + i.1 < (pre ++ (ps ++ post)).length := by
        simpa [n, List.length_append, Nat.add_assoc] using hPrePsPost
      -- Strip `pre` via `get_append_right_offset_nat` (with `j = i.1`).
      simp [n]
    -- Drop the trailing `post`, focusing to `ps`.
    have hPost :
        (ps ++ post).get ⟨i.1, by
          have : i.1 < (ps ++ post).length := by
            have : i.1 < ps.length + post.length := Nat.lt_of_lt_of_le i.2 (Nat.le_add_right _ _)
            simpa [List.length_append, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using this
          simpa [List.length_append] using this⟩
          =
        ps.get i := by
      -- `get_append_left_nat` returns `ps.get ⟨i.1, i.2⟩`; align that index with `i`.
      have hFin : (⟨i.1, i.2⟩ : Fin ps.length) = i := by
        apply Fin.ext
        rfl
      -- Match the `Fin` proof used by `get_append_left_nat` on the left.
      have hIdx :
          (⟨i.1, by
            simpa [List.length_append] using Nat.lt_of_lt_of_le i.2 (Nat.le_add_right _ _)⟩ : Fin
              (ps ++ post).length)
            =
          ⟨i.1, by
            have : i.1 < (ps ++ post).length := by
              have : i.1 < ps.length + post.length := Nat.lt_of_lt_of_le i.2 (Nat.le_add_right _ _)
              simpa [List.length_append, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using this
            simpa [List.length_append] using this⟩ := by
        apply Fin.ext
        rfl
      have h0 :
          (ps ++ post).get ⟨i.1, by
            simpa [List.length_append] using Nat.lt_of_lt_of_le i.2 (Nat.le_add_right _ _)⟩
            =
          ps.get ⟨i.1, i.2⟩ :=
        get_append_left_nat (as := ps) (bs := post) (i := i.1) (hi := i.2)
      simp
    exact Eq.trans hExtra (Eq.trans hPre (Eq.trans hPost rfl))
  simpa [Γ] using castTerm hGet (DAG.Term.var (Γ := Γ) (DAG.Var.ofFin idx))

/--
Lower a unary `Primitive` application into the DAG term language.

Parameters are read from the middle `ps` segment of the ambient environment, in the same order as
the primitive's parameter ABI, and the final data input is supplied by `x`.
-/
def primCall
    {pre ps post extra : List Shape} {σ τ : Shape}
    (p : Primitive ps σ τ)
    (x : DAG.Term ((pre ++ ps ++ post) ++ extra) σ) :
    DAG.Term ((pre ++ ps ++ post) ++ extra) τ := by
  let Γ : List Shape := (pre ++ ps ++ post) ++ extra
  let op : DAG.PrimOp (ps ++ [σ]) τ := Primitive.toDAGPrimOp (ps := ps) (σ := σ) (τ := τ) p
  let paramsArgs : DAG.Args Γ ps :=
    argsOfFn (Γ := Γ) (ins := ps) (fun i => mkParamTerm (pre := pre) (ps := ps) (post := post)
      (extra := extra) i)
  let args : DAG.Args Γ (ps ++ [σ]) :=
    Args.append1 (Γ := Γ) (ps := ps) (σ := σ) paramsArgs x
  exact (by
    -- Discharge the local `Γ` abbreviation.
    simpa [Γ] using (DAG.Term.op (Γ := Γ) op args))

/-! ### Chain lowering -/

/-- Lower a sequential `Chain` to an SSA-style `DAG.Term`, with parameters read from the
  environment. -/
def toTerm
    {pre ps post extra : List Shape} {σ τ : Shape}
    (g : Chain ps σ τ)
    (x : DAG.Term ((pre ++ ps ++ post) ++ extra) σ) :
    DAG.Term ((pre ++ ps ++ post) ++ extra) τ := by
  let Γ : List Shape := (pre ++ ps ++ post) ++ extra
  match g with
  | .id _ =>
      simpa [Γ] using x
  | .prim p =>
      simpa [Γ] using primCall (pre := pre) (ps := ps) (post := post) (extra := extra) (p := p) x
  | .seq (ps₁ := ps₁) (ps₂ := ps₂) (σ := σ) (τ := τm) (υ := τ) g₁ g₂ =>
      -- Outer env: `(pre ++ (ps₁ ++ ps₂) ++ post) ++ extra`
      let Γ0 : List Shape := (pre ++ (ps₁ ++ ps₂) ++ post) ++ extra
      -- Left subgraph sees post = ps₂ ++ post.
      have hΓ1 :
          (pre ++ ps₁ ++ (ps₂ ++ post)) ++ extra
          =
          Γ0 := by
        simp [Γ0, List.append_assoc]
      let t₁ : DAG.Term Γ0 τm :=
        castEnvTerm (Γ := (pre ++ ps₁ ++ (ps₂ ++ post)) ++ extra) (Γ' := Γ0) hΓ1 <|
          toTerm (pre := pre) (ps := ps₁) (post := ps₂ ++ post) (extra := extra) g₁
            (by
              have x0 : DAG.Term Γ0 σ := by
                simpa [Γ, Γ0] using x
              exact
                castEnvTerm (Γ := Γ0) (Γ' := (pre ++ ps₁ ++ (ps₂ ++ post)) ++ extra) (by
                  simp [Γ0, List.append_assoc]) x0)
      -- `let1`-bind and translate the right subgraph.
      let bodyEnv : List Shape := Γ0 ++ [τm]
      -- Bound var in the body env (the last element, at index `Γ0.length`).
      let boundIdx : Fin bodyEnv.length := ⟨Γ0.length, by simp [bodyEnv, List.length_append]⟩
      let boundVar : DAG.Term bodyEnv τm :=
        -- Align the index used in `get_append_last` with `boundIdx`.
        let idx0 : Fin bodyEnv.length := ⟨Γ0.length, by simp [bodyEnv, List.length_append]⟩
        have hGet0 : bodyEnv.get idx0 = τm := by
          -- Avoid `simp` rewriting `Eq` goals into `True`.
          dsimp [bodyEnv]
          let idxStd : Fin (Γ0 ++ [τm]).length := ⟨Γ0.length, by simp [List.length_append]⟩
          have hidx : idx0 = idxStd := by
            apply Fin.ext
            rfl
          cases hidx
          exact get_append_last (xs := Γ0) (x := τm)
        have hIdx : boundIdx = idx0 := by
          apply Fin.ext
          rfl
        have hGet : bodyEnv.get boundIdx = τm := by
          simpa [hIdx] using hGet0
        castTerm hGet (DAG.Term.var (Γ := bodyEnv) (DAG.Var.ofFin boundIdx))
      -- Translate `g₂` under its own parenthesization, then cast back to `bodyEnv`.
      let rhsEnv : List Shape := ((pre ++ ps₁) ++ ps₂ ++ post) ++ (extra ++ [τm])
      have hRhs : rhsEnv = bodyEnv := by
        simp [rhsEnv, bodyEnv, Γ0, List.append_assoc]
      let boundVar' : DAG.Term rhsEnv τm :=
        castEnvTerm (Γ := bodyEnv) (Γ' := rhsEnv) hRhs.symm boundVar
      let t₂' : DAG.Term rhsEnv τ :=
        toTerm (pre := pre ++ ps₁) (ps := ps₂) (post := post) (extra := extra ++ [τm]) g₂ boundVar'
      let t₂ : DAG.Term bodyEnv τ :=
        castEnvTerm (Γ := rhsEnv) (Γ' := bodyEnv) hRhs t₂'
      let out : DAG.Term Γ0 τ := DAG.Term.let1 t₁ t₂
      -- Discharge the local `Γ` abbreviation.
      simpa [Γ, Γ0] using out

/-! ### Public API -/

/--
Lower a sequential `Chain` to a DAG term with environment `ps ++ [σ]`.
 -/
def Chain.toDAGTerm {ps : List Shape} {σ τ : Shape} (g : Chain ps σ τ) :
    DAG.Term (ps ++ [σ]) τ :=
  let x :
      let Γ : List Shape := ([] ++ ps ++ []) ++ [σ]
      DAG.Term Γ σ := by
    intro Γ
    have hLt : ps.length < Γ.length := by
      simp [Γ, List.length_append]
    let xIdx : Fin Γ.length := ⟨ps.length, hLt⟩
    have hGet0 :
        Γ.get ⟨ps.length, by simp [Γ, List.length_append]⟩ = σ := by
      -- `Γ` is definitional `(([] ++ ps ++ []) ++ [σ])`, so this is the last element.
      simp [Γ]
    have hxIdx : xIdx = ⟨ps.length, by simp [Γ, List.length_append]⟩ := by
      apply Fin.ext
      rfl
    have hGet : Γ.get xIdx = σ := by
      simpa [hxIdx] using hGet0
    exact castTerm hGet (DAG.Term.var (Γ := Γ) (DAG.Var.ofFin xIdx))
  -- `toTerm`’s environment is definitional `([] ++ ps ++ [] ++ [σ])`; normalize to `ps ++ [σ]`.
  by
    simpa [List.nil_append, List.append_nil, List.append_assoc] using
      (toTerm (pre := []) (ps := ps) (post := []) (extra := [σ]) g x)

end LowerToDAG

end GraphSpec
end NN
