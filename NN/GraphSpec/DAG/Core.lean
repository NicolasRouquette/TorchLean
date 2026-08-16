/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.Program
import Mathlib.Algebra.Order.Algebra

/-!
# Canonical GraphSpec DAG

This file defines the canonical general GraphSpec representation: typed SSA/DAG terms.

`NN.GraphSpec.Core` gives a sequential authoring language (`Chain` with `>>>`) for chain-like
architectures. That syntax lowers into this DAG language. Many modern architectures are not purely
sequential:

- **skip connections** (ResNets),
- **shared subcomputations** (reusing the same feature tensor multiple times),
- **multi-input ops** (e.g. `add`, concatenation, attention blocks, …).

`GraphSpec.DAG` is the general version: a small SSA/A-normal-form term language whose terms denote
DAG-shaped computation graphs.

## Core idea: typed environments

We track an environment `Γ : List Shape` at the type level. A term `Term Γ τ` means:

- “if you provide values for every shape in `Γ` (in order),”
- “this term computes a tensor of shape `τ`.”

This is similar in spirit to a simply-typed lambda calculus with de Bruijn variables, except:

- there are no lambdas (only `let`-bindings), so the graph is acyclic by construction,
- primitive ops have arbitrary arity `ins : List Shape`,
- and `Args Γ ins` is a typed list of input subterms for an op.

The constructors are:

- `var`   : read a variable from the environment through a shape-indexed de Bruijn variable,
- `cast` / `castEnv` : explicit casts to handle non-definitional equalities in `Shape` / `Γ`,
- `op`    : apply an n-ary primitive to n arguments,
- `let1`  : bind an intermediate result and extend the environment (`Γ ++ [σ]`).

## Mathematical semantics (intended)

For a fixed scalar type `α` with `[Context α]`, we interpret an environment as a typed list of
tensors `TList α Γ`. Then:

- `Term.eval : TList α Γ → Term Γ τ → Spec.Tensor α τ` is the *pure* reference semantics.
- `Term.lower` produces an execution-polymorphic TorchLean program that computes the same graph, but in
  the executable (monadic, reference-based) runtime world.

## Small example (residual add)

The residual pattern “main path + skip path” is the canonical example that needs DAG syntax:

$$
\begin{aligned}
y &= \operatorname{linear}(W,b,x),\\
\mathrm{out} &= \operatorname{ReLU}(y+x).
\end{aligned}
$$

Here `x` is used twice, so a pure chain representation would need duplication. With `let1`,
sharing is explicit: compute `y` once, then reuse it.

See `NN/GraphSpec/Models/ResidualLinear.lean` for a complete, well-typed instance.

## References / citations

Conceptual background (stable, standard references):

- SSA form: Cytron et al. (1991), “Efficiently Computing Static Single Assignment Form…”.
- A-normal form / let-normal form (used to enforce DAG structure).
- Automatic differentiation survey: Baydin et al. (2018), “Automatic Differentiation in Machine
  Learning: a Survey”.
- Residual networks: He et al. (2016), “Deep Residual Learning for Image Recognition”.
-/

@[expose] public section


namespace NN
namespace GraphSpec
namespace DAG

open _root_.NN.Spec
open Spec.Tensor
open NN.Tensor

/-! ## Primitives (arbitrary arity) -/

/--
An n-ary primitive operation.

Compared to the sequential `GraphSpec.Primitive`, a `PrimOp` here is parameter-free: parameters are
just ordinary inputs in the environment. This is what makes the DAG language flexible: a “layer”
is expressed by `let1`-binding its parameters and then using them as inputs to ops.

Type indices:
- `ins : List Shape` is the ordered list of input tensor shapes the op expects.
- `τ : Shape` is the output tensor shape.
 -/
structure PrimOp (ins : List Shape) (τ : Shape) where
  /-- Debug name for error messages / inspection. -/
  name : String
  /-- Pure reference semantics (`ins` arguments packed as a typed list). -/
  specFwd : ∀ {α : Type 0}, [Context α] → Runtime.Autograd.Torch.TList α ins → Spec.Tensor α τ
  /-- Executable TorchLean program with arguments of shapes `ins`. -/
  program :
    ∀ {α : Type 0}, [Context α] → [DecidableEq Shape] →
      Runtime.Autograd.TorchLean.Program α ins τ

/-! ## Typed variables -/

/-- A de Bruijn variable whose result shape is part of its type.

Unlike a bare `Fin Γ.length`, `Var Γ s` records that the selected entry of `Γ` has shape `s`.
Environment lookup therefore reduces structurally, without dependent casts through `List.get`.
`ofFin` remains available to programmatic lowerings that discover positions dynamically.
-/
inductive Var : (Γ : List Shape) → Shape → Type where
  /-- The first value in a nonempty environment. -/
  | head {s : Shape} {Γ : List Shape} : Var (s :: Γ) s
  /-- A variable inherited from the tail of an environment. -/
  | tail {Γ : List Shape} {s t : Shape} : Var Γ t → Var (s :: Γ) t

namespace Var

/-- Convert a numeric environment position to a shape-indexed variable. -/
def ofFin {Γ : List Shape} (i : Fin Γ.length) : Var Γ (Γ.get i) :=
  match Γ with
  | [] => nomatch i
  | _ :: _ => Fin.cases .head (fun j => .tail (ofFin j)) i

/-- Preserve a variable when one value is appended to its environment. -/
def weakenRight {Γ : List Shape} {s t : Shape} : Var Γ s → Var (Γ ++ [t]) s
  | .head => .head
  | .tail v => .tail (weakenRight v)

/-- Embed a variable from the left side of an appended environment. -/
def inLeft {Γ : List Shape} (right : List Shape) {s : Shape} : Var Γ s → Var (Γ ++ right) s
  | .head => .head
  | .tail v => .tail (inLeft right v)

/-- Embed a variable from the right side of an appended environment. -/
def inRight (left : List Shape) {Γ : List Shape} {s : Shape} : Var Γ s → Var (left ++ Γ) s :=
  match left with
  | [] => fun v => v
  | _ :: rest => fun v => .tail (inRight rest v)

/-- The final variable in an environment extended by one value. -/
def last (Γ : List Shape) {s : Shape} : Var (Γ ++ [s]) s :=
  match Γ with
  | [] => .head
  | _ :: rest => .tail (last rest)

/-- Extend a shape-preserving variable renaming across one value appended to both environments. -/
def liftRight {Γ Δ : List Shape} {t : Shape}
    (ρ : {s : Shape} → Var Γ s → Var Δ s) {s : Shape}
    (v : Var (Γ ++ [t]) s) : Var (Δ ++ [t]) s :=
  match Γ with
  | [] =>
      match v with
      | .head => last Δ
      | .tail impossible => nomatch impossible
  | _ :: _ =>
      match v with
      | .head => weakenRight (ρ .head)
      | .tail inherited => liftRight (fun v => ρ (.tail v)) inherited

end Var

/-! ## DAG terms + arguments (mutual) -/

mutual
  /--
  A well-typed DAG term.

  Read this as: “under environment `Γ`, this term produces a tensor of shape `τ`”.
  -/
  inductive Term : (Γ : List Shape) → Shape → Type 1 where
    /-- Variable read (shape-indexed de Bruijn position in the environment). -/
    | var {Γ : List Shape} {s : Shape} (i : Var Γ s) : Term Γ s
    /--
    Cast a term’s *output shape* along a propositional equality.

    This is an internal hygiene tool: when we build terms programmatically (e.g. by lowering a
    higher-level syntax into DAG form), we often end up with goals like “`Γ.get i = τ`” that are
    true but not definitional.

    Using a `cast` node keeps the term in constructor form (so evaluators/compilers can still
    pattern match), and pushes the non-definitional equality into the semantics where it can be
    handled by `cases h`.
     -/
    | cast {Γ : List Shape} {σ τ : Shape} : Term Γ σ → σ = τ → Term Γ τ
    /--
    Cast a term’s *environment* along a propositional equality.

    This is useful when normalizing list-association/parenthesization choices in `Γ` without
    changing meaning.
     -/
    | castEnv {Γ Γ' : List Shape} {τ : Shape} : Term Γ τ → Γ = Γ' → Term Γ' τ
    /-- Apply an n-ary primitive op to n arguments. -/
    | op {Γ : List Shape} {ins : List Shape} {τ : Shape} :
        PrimOp ins τ → Args Γ ins → Term Γ τ
    /-- Let-bind a single intermediate value, extending the environment. -/
    | let1 {Γ : List Shape} {σ τ : Shape} : Term Γ σ → Term (Γ ++ [σ]) τ → Term Γ τ

  /--
  A typed list of argument terms.

  `Args Γ [s₁, …, sₙ]` is a tuple of `n` terms, each well-typed under the same environment `Γ`,
  with corresponding shapes `s₁, …, sₙ`.
  -/
  inductive Args : (Γ : List Shape) → List Shape → Type 1 where
    | nil {Γ} : Args Γ []
    | cons {Γ} {s : Shape} {ss : List Shape} : Term Γ s → Args Γ ss → Args Γ (s :: ss)
end

mutual
  /-- Rename every free variable in a typed operation-argument list. -/
  def Args.rename {Γ Δ : List Shape} {ss : List Shape}
      (ρ : {t : Shape} → Var Γ t → Var Δ t) : Args Γ ss → Args Δ ss
    | .nil => .nil
    | .cons term rest => .cons (Term.rename ρ term) (Args.rename ρ rest)

  /-- Rename every free variable of a term while preserving its tensor shape. -/
  def Term.rename {Γ Δ : List Shape} {s : Shape}
      (ρ : {t : Shape} → Var Γ t → Var Δ t) : Term Γ s → Term Δ s
    | .var i => .var (ρ i)
    | .cast term h => .cast (Term.rename ρ term) h
    | .castEnv term h =>
        match h.symm with
        | rfl => Term.rename ρ term
    | .op primitive args => .op primitive (Args.rename ρ args)
    | .let1 value body =>
        .let1 (Term.rename ρ value) (Term.rename (Var.liftRight ρ) body)

  /-- Preserve a term when an unrelated value is prepended to its environment. -/
  def Term.weakenLeft {Γ : List Shape} {s t : Shape} : Term Γ s → Term (t :: Γ) s
    | term => Term.rename (fun v => .tail v) term

  /-- Preserve typed operation arguments when a value is prepended to their environment. -/
  def Args.weakenLeft {Γ : List Shape} {ss : List Shape} {t : Shape} :
      Args Γ ss → Args (t :: Γ) ss
    | .nil => .nil
    | .cons term rest => .cons (Term.weakenLeft term) (Args.weakenLeft rest)
end

namespace Term

/-- Preserve a term when an unrelated value is appended to its environment. -/
def weakenRight {Γ : List Shape} {s t : Shape} (term : Term Γ s) : Term (Γ ++ [t]) s :=
  Term.rename Var.weakenRight term

/-- Preserve a term when an arbitrary typed environment is appended. -/
def weakenAppend {Γ : List Shape} (right : List Shape) {s : Shape}
    (term : Term Γ s) : Term (Γ ++ right) s :=
  Term.rename (Var.inLeft right) term

end Term

namespace Args

/-- Select the term assigned to a shape-indexed variable. -/
def get {Γ Δ : List Shape} {s : Shape} : Args Δ Γ → Var Γ s → Term Δ s
  | .cons term _, .head => term
  | .cons _ rest, .tail v => get rest v

/-- Concatenate two typed argument lists living in the same graph environment. -/
def append {Γ left right : List Shape} :
    Args Γ left → Args Γ right → Args Γ (left ++ right)
  | .nil, rightArgs => rightArgs
  | .cons term rest, rightArgs => .cons term (append rest rightArgs)

/-- Split arguments at a type-level list boundary.

This is the argument-list counterpart of `TList.splitAppend`.  It is useful when a model owns a
concatenated parameter ABI but its implementation is assembled recursively from smaller models:
each component receives exactly the terms belonging to its part of the ABI, with every tensor
shape retained by the type checker.
-/
def splitAppend {Γ : List Shape} : {left right : List Shape} →
    Args Γ (left ++ right) → Args Γ left × Args Γ right
  | [], _right, args => (.nil, args)
  | _shape :: left, right, .cons term rest =>
      let parts := splitAppend (left := left) (right := right) rest
      (.cons term parts.1, parts.2)

/-- Splitting arguments immediately after concatenating them recovers both original lists. -/
@[simp] theorem splitAppend_append {Γ left right : List Shape}
    (leftArgs : Args Γ left) (rightArgs : Args Γ right) :
    splitAppend (append leftArgs rightArgs) = (leftArgs, rightArgs) := by
  induction left with
  | nil => cases leftArgs; rfl
  | cons shape left ih =>
      cases leftArgs with
      | cons term rest =>
          simp only [append, splitAppend]
          rw [ih rest]

/-- Concatenating both parts of a split recovers the original typed argument list. -/
theorem append_splitAppend {Γ left right : List Shape}
    (args : Args Γ (left ++ right)) :
    append (splitAppend args).1 (splitAppend args).2 = args := by
  induction left with
  | nil => rfl
  | cons shape left ih =>
      cases args with
      | cons term rest =>
          simp only [splitAppend, append]
          rw [ih rest]

/-- View every entry of a typed environment as a term in that same environment.

The result preserves the order and shape indices of `Γ`.  Large graph definitions can therefore
pattern-match once on `vars Γ` instead of selecting every input by a numeric index and separately
proving that the selected position has the expected shape. -/
def vars : (Γ : List Shape) → Args Γ Γ
  | [] => .nil
  | _ :: rest => .cons (.var .head) (Args.weakenLeft (vars rest))

/-- Selecting from renamed arguments renames the selected term. -/
@[simp] theorem get_rename {Γ Δ ss : List Shape} {s : Shape}
    (ρ : {t : Shape} → Var Γ t → Var Δ t) (args : Args Γ ss) (position : Var ss s) :
    Args.get (Args.rename ρ args) position = Term.rename ρ (Args.get args position) := by
  induction position with
  | head => cases args; rfl
  | tail position ih =>
      cases args with
      | cons _ rest => exact ih rest

/-- Lookup commutes with embedding an argument list below one new graph variable. -/
@[simp] theorem get_weakenLeft {Γ ss : List Shape} {s t : Shape}
    (args : Args Γ ss) (position : Var ss s) :
    Args.get (Args.weakenLeft (t := t) args) position =
      Term.weakenLeft (t := t) (Args.get args position) := by
  induction position with
  | head => cases args; rfl
  | tail position ih =>
      cases args with
      | cons _ rest => exact ih rest

/-- Selecting a variable from the complete environment returns that variable as a term. -/
@[simp] theorem get_vars {Γ : List Shape} {s : Shape} (position : Var Γ s) :
    Args.get (vars Γ) position = Term.var position := by
  induction position with
  | head => rfl
  | tail position ih =>
      simp only [vars, get, get_weakenLeft, ih, Term.weakenLeft, Term.rename]

end Args

/-! ## Typed substitution

Substitution is the operation used to inline one graph into another.  A substitution assigns a
well-typed term in `Δ` to every variable in `Γ`; applying it replaces the free variables of a term
without changing the term's result shape.  The let-binding case lifts the assignment across the
new value appended by the binder.
-/

/-- A shape-preserving assignment of terms to every variable in an environment. -/
abbrev Substitution (Γ Δ : List Shape) := {s : Shape} → Var Γ s → Term Δ s

namespace Substitution

/-- Extend a substitution across one value appended to both environments. -/
def liftRight {Γ Δ : List Shape} {t : Shape}
    (σ : Substitution Γ Δ) : Substitution (Γ ++ [t]) (Δ ++ [t]) :=
  match Γ with
  | [] => fun v =>
      match v with
      | .head => .var (Var.last Δ)
      | .tail impossible => nomatch impossible
  | _ :: rest => fun v =>
      match v with
      | .head => Term.weakenRight (σ .head)
      | .tail inherited =>
          liftRight (Γ := rest) (fun v => σ (.tail v)) inherited
termination_by Γ

end Substitution

mutual
  /-- Replace every free variable in a typed operation-argument list. -/
  def Args.substitute {Γ Δ shapes : List Shape} (σ : Substitution Γ Δ) :
      Args Γ shapes → Args Δ shapes
    | .nil => .nil
    | .cons term rest => .cons (Term.substitute σ term) (Args.substitute σ rest)

  /-- Replace every free variable in a term by its assigned term. -/
  def Term.substitute {Γ Δ : List Shape} {s : Shape} (σ : Substitution Γ Δ) :
      Term Γ s → Term Δ s
    | .var v => σ v
    | .cast term equality => .cast (Term.substitute σ term) equality
    | .castEnv term equality =>
        match equality.symm with
        | rfl => Term.substitute σ term
    | .op primitive args => .op primitive (Args.substitute σ args)
    | .let1 value body =>
        .let1 (Term.substitute σ value)
          (Term.substitute (Substitution.liftRight σ) body)
end

namespace Term

/-- Inline a term by supplying one typed argument term for each free variable. -/
def instantiate {Γ Δ : List Shape} {s : Shape}
    (arguments : Args Δ Γ) (term : Term Γ s) : Term Δ s :=
  term.substitute (Args.get arguments)

end Term

/-- A multi-result DAG block with shared A-normal-form bindings.

`Block.let1` computes an intermediate once and makes it available to every eventual result. This
is the representation needed by recurrent steps that return both a new state and an output derived
from that state.
-/
inductive Block : (Γ : List Shape) → List Shape → Type 1 where
  /-- Return one term for each output shape. -/
  | ret {Γ outs} : Args Γ outs → Block Γ outs
  /-- Compute one shared intermediate and append it to the environment. -/
  | let1 {Γ outs σ} : Term Γ σ → Block (Γ ++ [σ]) outs → Block Γ outs

namespace Block

/-- Replace every free variable in a multi-result block. -/
def substitute {Γ Δ outputs : List Shape} (σ : Substitution Γ Δ) :
    Block Γ outputs → Block Δ outputs
  | .ret results => .ret (Args.substitute σ results)
  | .let1 value body =>
      .let1 (Term.substitute σ value)
        (Block.substitute (Substitution.liftRight σ) body)

/-- Inline a multi-result block by supplying all of its free variables. -/
def instantiate {Γ Δ outputs : List Shape}
    (arguments : Args Δ Γ) (block : Block Γ outputs) : Block Δ outputs :=
  block.substitute (Args.get arguments)

/-- Compose two multi-output blocks.

The second block sees the original environment followed by every result of the first block. Any
`let1` bindings inside the first block remain shared. This is the typed DAG analogue of binding a
tuple-valued computation and is the basic operation needed to compose recurrent cells, residual
branches, and encoder-decoder stages.
-/
def andThenAux {Γ Δ middle outputs : List Shape}
    (ρ : {s : Shape} → Var Γ s → Var Δ s)
    (first : Block Δ middle) (second : Block (Γ ++ middle) outputs) : Block Δ outputs :=
  match first with
  | .ret results =>
      second.instantiate (Args.append (Args.rename ρ (Args.vars Γ)) results)
  | .let1 value body =>
      .let1 value <| andThenAux (fun v => Var.weakenRight (ρ v)) body second
termination_by sizeOf first
decreasing_by simp_wf

/-- Feed every output of `first` to `second`, preserving the original environment. -/
def andThen {Γ middle outputs : List Shape}
    (first : Block Γ middle) (second : Block (Γ ++ middle) outputs) : Block Γ outputs :=
  andThenAux (fun v => v) first second

end Block

namespace Env

open Runtime.Autograd.Torch

/-- Typed environment lookup for pure tensors. -/
def tget {α : Type} : {Γ : List Shape} → {s : Shape} → TList α Γ → Var Γ s →
    Spec.Tensor α s
  | _ :: _, _, .cons x _, .head => x
  | _ :: _, _, .cons _ xs, .tail i => tget xs i

/-- Looking up a programmatically selected variable agrees with typed-list lookup. -/
@[simp] theorem tget_ofFin {α : Type} {Γ : List Shape} (env : TList α Γ)
    (i : Fin Γ.length) :
    tget env (Var.ofFin i) = Proofs.Autograd.Algebra.TList.get env i := by
  induction Γ with
  | nil => exact Fin.elim0 i
  | cons shape Γ ih =>
      cases env with
      | cons value rest =>
          refine Fin.cases ?_ ?_ i
          · rfl
          · intro j
            change tget rest (Var.ofFin j) = Proofs.Autograd.Algebra.TList.get rest j
            exact ih rest j

/-- Appending a value does not change the meaning of an existing variable. -/
@[simp] theorem tget_append_weakenRight {α : Type} {Γ : List Shape} {s t : Shape}
    (env : TList α Γ) (value : Spec.Tensor α t) (i : Var Γ s) :
    tget (Proofs.Autograd.Algebra.TList.append env (.cons value .nil))
        (Var.weakenRight i) = tget env i := by
  induction i with
  | head => cases env; rfl
  | tail i ih =>
      cases env with
      | cons _ rest => exact ih rest

/-- Looking up a variable embedded from the left reads the original left environment. -/
@[simp] theorem tget_append_inLeft {α : Type} {Γ Δ : List Shape} {s : Shape}
    (left : TList α Γ) (right : TList α Δ) (v : Var Γ s) :
    tget (Proofs.Autograd.Algebra.TList.append left right) (Var.inLeft Δ v) = tget left v := by
  induction v with
  | head => cases left; rfl
  | tail v ih =>
      cases left with
      | cons _ rest => exact ih rest

/-- Looking up a variable embedded from the right reads the appended right environment. -/
@[simp] theorem tget_append_inRight {α : Type} {Γ Δ : List Shape} {s : Shape}
    (left : TList α Γ) (right : TList α Δ) (v : Var Δ s) :
    tget (Proofs.Autograd.Algebra.TList.append left right) (Var.inRight Γ v) = tget right v := by
  induction Γ with
  | nil => cases left; rfl
  | cons shape Γ ih =>
      cases left with
      | cons _ rest => exact ih rest

/-- Looking up the final variable returns the value most recently appended to an environment. -/
@[simp] theorem tget_append_last {α : Type} {Γ : List Shape} {s : Shape}
    (env : TList α Γ) (value : Spec.Tensor α s) :
    tget (Proofs.Autograd.Algebra.TList.append env (.cons value .nil)) (Var.last Γ) = value := by
  induction Γ with
  | nil => cases env; rfl
  | cons _ Γ ih =>
      cases env with
      | cons _ rest => exact ih rest

/--
Typed environment lookup for backend references.

This is the underlying “variable semantics” for `Term.lower`.
 -/
def rget {Ref : Shape → Type} : {Γ : List Shape} → {s : Shape} →
    Runtime.Autograd.Torch.RefList Ref Γ → Var Γ s → Ref s
  | _ :: _, _, .cons x _, .head => x
  | _ :: _, _, .cons _ xs, .tail i => rget xs i

/-- Reference lookup is unchanged when an existing variable is weakened past an append. -/
@[simp] theorem rget_append_weakenRight {Ref : Shape → Type} {Γ : List Shape} {s t : Shape}
    (env : Runtime.Autograd.Torch.RefList Ref Γ) (value : Ref t) (i : Var Γ s) :
    rget (Runtime.Autograd.Torch.RefList.append env (.cons value .nil))
        (Var.weakenRight i) = rget env i := by
  induction i with
  | head => cases env; rfl
  | tail i ih =>
      cases env with
      | cons _ rest => exact ih rest

/-- Reference lookup of the final variable returns the most recently appended reference. -/
@[simp] theorem rget_append_last {Ref : Shape → Type} {Γ : List Shape} {s : Shape}
    (env : Runtime.Autograd.Torch.RefList Ref Γ) (value : Ref s) :
    rget (Runtime.Autograd.Torch.RefList.append env (.cons value .nil)) (Var.last Γ) = value := by
  induction Γ with
  | nil => cases env; rfl
  | cons _ Γ ih =>
      cases env with
      | cons _ rest => exact ih rest

end Env

namespace Term

open Runtime.Autograd.Torch

/-! ### Spec interpreter -/

mutual
  /-- Evaluate a typed argument list by evaluating each component term under the same environment.
    -/
  def evalArgs
      {Γ : List Shape} {ins : List Shape}
      {α : Type 0} [Context α]
      (env : TList α Γ) :
      Args Γ ins → TList α ins
    | .nil => .nil
    | .cons t ts => .cons (eval (Γ := Γ) (α := α) env t) (evalArgs (Γ := Γ) (α := α) env ts)

  /--
  Pure evaluation of a DAG term.

  This is the “math-first” semantics: we interpret a term as a pure function on tensors.
  No monads, no mutation, no autograd tape — just the Spec definitions of primitives.

  The key runtime discipline is the environment discipline:

  - `var` reads from `env`,
  - `op` evaluates its arguments and feeds them to the primitive’s `specFwd`,
  - `let1` evaluates the bound term once and extends `env` for the body.
  -/
  def eval
      {Γ : List Shape} {τ : Shape}
      {α : Type 0} [Context α]
      (env : TList α Γ) :
      Term Γ τ → Spec.Tensor α τ
    | .var i => Env.tget (α := α) env i
    | .cast t h =>
        match h with
        | rfl => eval (Γ := Γ) (α := α) env t
    | .castEnv t h =>
        -- `h : Γ_src = Γ_tgt` comes from term construction/lowering. For evaluation we want to
        -- *rewrite the term* to the current environment `Γ_tgt`, not rewrite the environment
        -- to the source `Γ_src`, so we match on `h.symm`.
        match h.symm with
        | rfl => eval (Γ := Γ) (α := α) env t
    | .op p args =>
        let xs := evalArgs (Γ := Γ) (ins := _) (α := α) env args
        p.specFwd (α := α) xs
    | .let1 (σ := σ) t body =>
        let v := eval (Γ := Γ) (α := α) env t
        let env' : TList α (Γ ++ [σ]) :=
          Proofs.Autograd.Algebra.TList.append (α := α) (ss₁ := Γ) (ss₂ := [σ]) env (.cons v .nil)
        eval (Γ := Γ ++ [σ]) (α := α) env' body
end

/-- Evaluating an operation node first evaluates its typed arguments, then applies the
primitive's pure semantics. -/
@[simp] theorem eval_op {Γ ins : List Shape} {τ : Shape} {α : Type 0} [Context α]
    (env : TList α Γ) (primitive : PrimOp ins τ) (args : Args Γ ins) :
    eval env (.op primitive args) = primitive.specFwd (evalArgs env args) := by
  rfl

/-- Evaluating an output-shape cast transports the value along the same shape equality. -/
@[simp] theorem eval_cast {Γ : List Shape} {σ τ : Shape} {α : Type 0} [Context α]
    (env : TList α Γ) (term : Term Γ σ) (h : σ = τ) :
    eval env (.cast term h) = h ▸ eval env term := by
  cases h
  rfl

/-! ### Semantics of variable renaming -/

end Term

namespace Env

/-- A variable renaming preserves an environment when every renamed lookup has the same value. -/
def RenamingSound {α : Type} {Γ Δ : List Shape}
    (envΓ : Runtime.Autograd.Torch.TList α Γ)
    (envΔ : Runtime.Autograd.Torch.TList α Δ)
    (ρ : {s : Shape} → Var Γ s → Var Δ s) : Prop :=
  ∀ {s : Shape} (v : Var Γ s), tget envΔ (ρ v) = tget envΓ v

/-- A sound renaming remains sound when the same value is appended to both environments. -/
theorem RenamingSound.liftRight {α : Type} {Γ Δ : List Shape} {t : Shape}
    {envΓ : Runtime.Autograd.Torch.TList α Γ}
    {envΔ : Runtime.Autograd.Torch.TList α Δ}
    {ρ : {s : Shape} → Var Γ s → Var Δ s}
    (hρ : RenamingSound envΓ envΔ ρ) (value : Spec.Tensor α t) :
    RenamingSound
      (Proofs.Autograd.Algebra.TList.append envΓ (.cons value .nil))
      (Proofs.Autograd.Algebra.TList.append envΔ (.cons value .nil))
      (Var.liftRight ρ) := by
  unfold RenamingSound
  intro s v
  induction Γ with
  | nil =>
      cases envΓ
      cases v with
      | head => exact tget_append_last envΔ value
      | tail impossible => nomatch impossible
  | cons shape Γ ih =>
      cases envΓ with
      | cons head tail =>
          cases v with
          | head =>
              change
                tget (Proofs.Autograd.Algebra.TList.append envΔ (.cons value .nil))
                    (Var.weakenRight (ρ .head)) =
                  tget
                    (Proofs.Autograd.Algebra.TList.append (.cons head tail)
                      (.cons value .nil)) .head
              calc
                _ = tget envΔ (ρ .head) := tget_append_weakenRight envΔ value (ρ .head)
                _ = tget (.cons head tail) .head := hρ .head
                _ = _ := (tget_append_weakenRight (.cons head tail) value .head).symm
          | tail v =>
              change
                tget (Proofs.Autograd.Algebra.TList.append envΔ (.cons value .nil))
                    (Var.liftRight (fun w => ρ (.tail w)) v) =
                  tget (Proofs.Autograd.Algebra.TList.append tail (.cons value .nil)) v
              exact ih (envΓ := tail) (ρ := fun w => ρ (.tail w))
                (fun w => hρ (.tail w)) v

end Env

namespace Term

mutual
  private def complexity {Γ : List Shape} {s : Shape} : Term Γ s → Nat
    | .var _ => 1
    | .cast term _ => complexity term + 1
    | .castEnv term _ => complexity term + 1
    | .op _ args => argsComplexity args + 1
    | .let1 value body => complexity value + complexity body + 1

  private def argsComplexity {Γ ins : List Shape} : Args Γ ins → Nat
    | .nil => 1
    | .cons term rest => complexity term + argsComplexity rest + 1
end

mutual
  /-- Pure evaluation commutes with a sound renaming of an operation's arguments. -/
  theorem evalArgs_rename
      {Γ Δ ins : List Shape} {α : Type 0} [Context α]
      (envΓ : Runtime.Autograd.Torch.TList α Γ)
      (envΔ : Runtime.Autograd.Torch.TList α Δ)
      (ρ : {s : Shape} → Var Γ s → Var Δ s)
      (hρ : Env.RenamingSound envΓ envΔ ρ) :
      (args : Args Γ ins) →
        evalArgs envΔ (Args.rename ρ args) = evalArgs envΓ args
    | .nil => rfl
    | .cons term rest => by
        simp only [Args.rename, evalArgs]
        rw [eval_rename envΓ envΔ ρ hρ term,
          evalArgs_rename envΓ envΔ ρ hρ rest]
  termination_by args => argsComplexity args

  decreasing_by all_goals simp [argsComplexity]

  /-- Pure evaluation commutes with any variable renaming that preserves environment lookup. -/
  theorem eval_rename
      {Γ Δ : List Shape} {s : Shape} {α : Type 0} [Context α]
      (envΓ : Runtime.Autograd.Torch.TList α Γ)
      (envΔ : Runtime.Autograd.Torch.TList α Δ)
      (ρ : {t : Shape} → Var Γ t → Var Δ t)
      (hρ : Env.RenamingSound envΓ envΔ ρ) :
      (term : Term Γ s) →
        eval envΔ (Term.rename ρ term) = eval envΓ term
    | .var v => hρ v
    | .cast term rfl => eval_rename envΓ envΔ ρ hρ term
    | .castEnv term rfl => eval_rename envΓ envΔ ρ hρ term
    | .op primitive args => by
        simp only [Term.rename, eval]
        rw [evalArgs_rename envΓ envΔ ρ hρ args]
    | .let1 value body => by
        simp only [Term.rename, eval]
        rw [eval_rename envΓ envΔ ρ hρ value]
        exact eval_rename
          (Proofs.Autograd.Algebra.TList.append envΓ
            (.cons (eval envΓ value) .nil))
          (Proofs.Autograd.Algebra.TList.append envΔ
            (.cons (eval envΓ value) .nil))
          (Var.liftRight ρ) (hρ.liftRight (eval envΓ value)) body
  termination_by term => complexity term
  decreasing_by all_goals simp [complexity]
end

/-- Renaming arguments into the left side of an appended environment preserves their values. -/
@[simp] theorem evalArgs_rename_inLeft
    {Γ Δ ins : List Shape} {α : Type 0} [Context α]
    (left : Runtime.Autograd.Torch.TList α Γ)
    (right : Runtime.Autograd.Torch.TList α Δ) (args : Args Γ ins) :
    evalArgs (Proofs.Autograd.Algebra.TList.append left right)
        (Args.rename (Var.inLeft Δ) args) = evalArgs left args := by
  apply evalArgs_rename left (Proofs.Autograd.Algebra.TList.append left right)
  intro shape position
  exact Env.tget_append_inLeft left right position

/-- Renaming arguments into the right side of an appended environment preserves their values. -/
@[simp] theorem evalArgs_rename_inRight
    {Γ Δ ins : List Shape} {α : Type 0} [Context α]
    (left : Runtime.Autograd.Torch.TList α Γ)
    (right : Runtime.Autograd.Torch.TList α Δ) (args : Args Δ ins) :
    evalArgs (Proofs.Autograd.Algebra.TList.append left right)
        (Args.rename (Var.inRight Γ) args) = evalArgs right args := by
  apply evalArgs_rename right (Proofs.Autograd.Algebra.TList.append left right)
  intro shape position
  exact Env.tget_append_inRight left right position

/-- Renaming a term into the left side of an appended environment preserves its value. -/
@[simp] theorem eval_rename_inLeft
    {Γ Δ : List Shape} {s : Shape} {α : Type 0} [Context α]
    (left : Runtime.Autograd.Torch.TList α Γ)
    (right : Runtime.Autograd.Torch.TList α Δ) (term : Term Γ s) :
    eval (Proofs.Autograd.Algebra.TList.append left right)
        (Term.rename (Var.inLeft Δ) term) = eval left term := by
  apply eval_rename left (Proofs.Autograd.Algebra.TList.append left right)
  intro shape position
  exact Env.tget_append_inLeft left right position

/-- Appending an arbitrary typed environment does not change a weakened term's value. -/
@[simp] theorem eval_weakenAppend
    {Γ Δ : List Shape} {s : Shape} {α : Type 0} [Context α]
    (left : Runtime.Autograd.Torch.TList α Γ)
    (right : Runtime.Autograd.Torch.TList α Δ) (term : Term Γ s) :
    eval (Proofs.Autograd.Algebra.TList.append left right)
        (Term.weakenAppend Δ term) = eval left term := by
  exact eval_rename_inLeft left right term

/-- Renaming a term into the right side of an appended environment preserves its value. -/
@[simp] theorem eval_rename_inRight
    {Γ Δ : List Shape} {s : Shape} {α : Type 0} [Context α]
    (left : Runtime.Autograd.Torch.TList α Γ)
    (right : Runtime.Autograd.Torch.TList α Δ) (term : Term Δ s) :
    eval (Proofs.Autograd.Algebra.TList.append left right)
        (Term.rename (Var.inRight Γ) term) = eval right term := by
  apply eval_rename right (Proofs.Autograd.Algebra.TList.append left right)
  intro shape position
  exact Env.tget_append_inRight left right position

/-- Appending an unrelated value does not change a term's pure meaning. -/
@[simp] theorem eval_weakenRight
    {Γ : List Shape} {s t : Shape} {α : Type 0} [Context α]
    (env : Runtime.Autograd.Torch.TList α Γ) (value : Spec.Tensor α t)
    (term : Term Γ s) :
    eval (Proofs.Autograd.Algebra.TList.append env (.cons value .nil))
      (Term.weakenRight term) = eval env term := by
  apply eval_rename env
    (Proofs.Autograd.Algebra.TList.append env (.cons value .nil)) Var.weakenRight
  unfold Env.RenamingSound
  intro shape v
  exact Env.tget_append_weakenRight env value v

/-- The final variable in an extended environment denotes the value that was just appended. -/
@[simp] theorem eval_var_last_append
    {Γ : List Shape} {t : Shape} {α : Type 0} [Context α]
    (env : Runtime.Autograd.Torch.TList α Γ) (value : Spec.Tensor α t) :
    eval (Proofs.Autograd.Algebra.TList.append env (.cons value .nil))
      (Term.var (Var.last Γ)) = value := by
  simpa only [eval] using Env.tget_append_last env value

end Term

namespace Env

/-- A term substitution represents an environment when every assigned term evaluates to the
value stored at the corresponding source variable. -/
def SubstitutionSound {α : Type} [Context α] {Γ Δ : List Shape}
    (envΓ : Runtime.Autograd.Torch.TList α Γ)
    (envΔ : Runtime.Autograd.Torch.TList α Δ)
    (σ : Substitution Γ Δ) : Prop :=
  ∀ {s : Shape} (v : Var Γ s), Term.eval envΔ (σ v) = Env.tget envΓ v

/-- A sound substitution remains sound across the value introduced by a `let` binding. -/
theorem SubstitutionSound.liftRight {α : Type} [Context α]
    {Γ Δ : List Shape} {t : Shape}
    {envΓ : Runtime.Autograd.Torch.TList α Γ}
    {envΔ : Runtime.Autograd.Torch.TList α Δ}
    {σ : Substitution Γ Δ} (hσ : SubstitutionSound envΓ envΔ σ)
    (value : Spec.Tensor α t) :
    SubstitutionSound
      (Proofs.Autograd.Algebra.TList.append envΓ (.cons value .nil))
      (Proofs.Autograd.Algebra.TList.append envΔ (.cons value .nil))
      (Substitution.liftRight σ) := by
  intro s v
  induction Γ with
  | nil =>
      cases envΓ
      cases v with
      | head =>
          simp only [Substitution.liftRight, Term.eval]
          exact Env.tget_append_last envΔ value
      | tail impossible => nomatch impossible
  | cons shape Γ ih =>
      cases envΓ with
      | cons head tail =>
          cases v with
          | head =>
              simp only [Substitution.liftRight]
              rw [Term.eval_weakenRight]
              exact hσ .head
          | tail v =>
              simp only [Substitution.liftRight]
              exact ih (envΓ := tail) (σ := fun w => σ (.tail w))
                (fun w => hσ (.tail w)) v

end Env

namespace Term

mutual
  /-- Pure evaluation commutes with a sound substitution of operation arguments. -/
  theorem evalArgs_substitute
      {Γ Δ inputs : List Shape} {α : Type 0} [Context α]
      (envΓ : Runtime.Autograd.Torch.TList α Γ)
      (envΔ : Runtime.Autograd.Torch.TList α Δ)
      (σ : Substitution Γ Δ) (hσ : Env.SubstitutionSound envΓ envΔ σ) :
      (args : Args Γ inputs) →
        evalArgs envΔ (Args.substitute σ args) = evalArgs envΓ args
    | .nil => rfl
    | .cons term rest => by
        simp only [Args.substitute, evalArgs]
        rw [eval_substitute envΓ envΔ σ hσ term,
          evalArgs_substitute envΓ envΔ σ hσ rest]
  termination_by args => argsComplexity args
  decreasing_by all_goals simp [argsComplexity]

  /-- Pure evaluation commutes with any term substitution representing the source environment. -/
  theorem eval_substitute
      {Γ Δ : List Shape} {s : Shape} {α : Type 0} [Context α]
      (envΓ : Runtime.Autograd.Torch.TList α Γ)
      (envΔ : Runtime.Autograd.Torch.TList α Δ)
      (σ : Substitution Γ Δ) (hσ : Env.SubstitutionSound envΓ envΔ σ) :
      (term : Term Γ s) →
        eval envΔ (Term.substitute σ term) = eval envΓ term
    | .var v => hσ v
    | .cast term rfl => eval_substitute envΓ envΔ σ hσ term
    | .castEnv term rfl => eval_substitute envΓ envΔ σ hσ term
    | .op primitive args => by
        simp only [Term.substitute, eval]
        rw [evalArgs_substitute envΓ envΔ σ hσ args]
    | .let1 value body => by
        simp only [Term.substitute, eval]
        rw [eval_substitute envΓ envΔ σ hσ value]
        exact eval_substitute
          (Proofs.Autograd.Algebra.TList.append envΓ
            (.cons (eval envΓ value) .nil))
          (Proofs.Autograd.Algebra.TList.append envΔ
            (.cons (eval envΓ value) .nil))
          (Substitution.liftRight σ) (hσ.liftRight (eval envΓ value)) body
  termination_by term => complexity term
  decreasing_by all_goals simp [complexity]
end

/-- Evaluating an argument selected by a typed variable agrees with lookup in the evaluated
argument environment. -/
@[simp] theorem eval_get {Γ Δ : List Shape} {s : Shape} {α : Type 0} [Context α]
    (env : Runtime.Autograd.Torch.TList α Δ) (args : Args Δ Γ) (v : Var Γ s) :
    eval env (Args.get args v) = Env.tget (evalArgs env args) v := by
  induction v with
  | head => cases args; rfl
  | tail v ih =>
      cases args with
      | cons _ rest => exact ih rest

/-- Pure evaluation of an inlined term equals evaluation of the original term under the supplied
argument values. -/
theorem eval_instantiate
    {Γ Δ : List Shape} {s : Shape} {α : Type 0} [Context α]
    (env : Runtime.Autograd.Torch.TList α Δ) (args : Args Δ Γ) (term : Term Γ s) :
    eval env (term.instantiate args) = eval (evalArgs env args) term := by
  apply eval_substitute (evalArgs env args) env (Args.get args)
  intro shape v
  exact eval_get env args v

/-- Evaluating concatenated graph arguments concatenates their tensor values in the same order. -/
theorem evalArgs_append
    {Γ left right : List Shape} {α : Type 0} [Context α]
    (env : Runtime.Autograd.Torch.TList α Γ) :
    (leftArgs : Args Γ left) → (rightArgs : Args Γ right) →
      evalArgs env (Args.append leftArgs rightArgs) =
        Proofs.Autograd.Algebra.TList.append
          (evalArgs env leftArgs) (evalArgs env rightArgs)
  | .nil, _ => rfl
  | .cons term rest, rightArgs => by
      simp only [Args.append, evalArgs, Proofs.Autograd.Algebra.TList.append]
      rw [evalArgs_append env rest rightArgs]

/-- Evaluating a statically split argument list agrees with splitting its evaluated values. -/
theorem evalArgs_splitAppend
    {Γ left right : List Shape} {α : Type 0} [Context α]
    (env : Runtime.Autograd.Torch.TList α Γ) (args : Args Γ (left ++ right)) :
    let argumentParts := Args.splitAppend args
    let valueParts := Proofs.Autograd.Algebra.TList.splitAppend (evalArgs env args)
    (evalArgs env argumentParts.1, evalArgs env argumentParts.2) = valueParts := by
  induction left with
  | nil => rfl
  | cons shape left ih =>
      cases args with
      | cons term rest =>
          simpa only [Args.splitAppend, evalArgs,
            Proofs.Autograd.Algebra.TList.splitAppend] using
            congrArg
              (fun parts =>
                (Proofs.Autograd.Algebra.TList.cons (eval env term) parts.1, parts.2))
              (ih rest)

/-- Evaluating the left part of a typed argument split returns the corresponding value prefix. -/
theorem evalArgs_splitAppend_fst
    {Γ left right : List Shape} {α : Type 0} [Context α]
    (env : Runtime.Autograd.Torch.TList α Γ) (args : Args Γ (left ++ right)) :
    evalArgs env (Args.splitAppend args).1 =
      (Proofs.Autograd.Algebra.TList.splitAppend (evalArgs env args)).1 := by
  exact congrArg Prod.fst (evalArgs_splitAppend env args)

/-- Evaluating the right part of a typed argument split returns the corresponding value suffix. -/
theorem evalArgs_splitAppend_snd
    {Γ left right : List Shape} {α : Type 0} [Context α]
    (env : Runtime.Autograd.Torch.TList α Γ) (args : Args Γ (left ++ right)) :
    evalArgs env (Args.splitAppend args).2 =
      (Proofs.Autograd.Algebra.TList.splitAppend (evalArgs env args)).2 := by
  exact congrArg Prod.snd (evalArgs_splitAppend env args)

/-- Prepending an unrelated value does not change a term's pure meaning. -/
@[simp] theorem eval_weakenLeft
    {Γ : List Shape} {s t : Shape} {α : Type 0} [Context α]
    (env : Runtime.Autograd.Torch.TList α Γ) (value : Spec.Tensor α t)
    (term : Term Γ s) :
    eval (.cons value env) (Term.weakenLeft term) = eval env term := by
  apply eval_rename env (.cons value env) (fun v => .tail v)
  unfold Env.RenamingSound
  intro shape v
  rfl

/-- Prepending an unrelated value does not change a typed argument list's pure meaning. -/
@[simp] theorem evalArgs_weakenLeft
    {Γ ins : List Shape} {t : Shape} {α : Type 0} [Context α]
    (env : Runtime.Autograd.Torch.TList α Γ) (value : Spec.Tensor α t) :
    (args : Args Γ ins) →
      evalArgs (.cons value env) (Args.weakenLeft args) = evalArgs env args
  | .nil => rfl
  | .cons term rest => by
      simp only [Args.weakenLeft, evalArgs, eval_weakenLeft,
        evalArgs_weakenLeft env value rest]
termination_by args => argsComplexity args
decreasing_by simp [argsComplexity]

/-- Evaluating all variables of an environment returns that environment in order. -/
@[simp] theorem evalArgs_vars
    {Γ : List Shape} {α : Type 0} [Context α]
    (env : Runtime.Autograd.Torch.TList α Γ) :
    evalArgs env (Args.vars Γ) = env := by
  induction Γ with
  | nil => cases env; rfl
  | cons shape Γ ih =>
      cases env with
      | cons value rest =>
          simp only [Args.vars, evalArgs, eval, Env.tget, evalArgs_weakenLeft]
          rw [ih rest]

/-! ### Lowering to TorchLean programs -/

/--
`RefT` is the runtime's reference type for tensors of a given shape.

In the executable runtime, primitives operate on *references* (allocated tensors) inside a monad.
This matches how typical deep-learning runtimes model device placement, mutation, and autograd.
 -/
abbrev RefT (m : Type → Type) (α : Type 0) [Context α] [DecidableEq Shape]
    [Runtime.Autograd.Torch.Ops (m := m) (α := α)] (s : Shape) : Type :=
  Runtime.Autograd.Torch.Ops.Ref (m := m) (α := α) s

mutual
  /-- Lower a typed argument list by lowering each component term under the same environment. -/
  def lowerArgs
      {Γ ins : List Shape}
      {α : Type 0} [Context α] [DecidableEq Shape]
      {m : Type → Type} [Monad m] [Runtime.Autograd.Torch.Ops (m := m) (α := α)]
      (env : Runtime.Autograd.Torch.RefList (RefT (m := m) (α := α)) Γ) :
      Args Γ ins → m (Runtime.Autograd.Torch.RefList (RefT (m := m) (α := α)) ins)
    | .nil => pure .nil
    | .cons t ts => do
        let r ← lower (Γ := Γ) (α := α) (m := m) env t
        let rs ← lowerArgs (Γ := Γ) (ins := _) (α := α) (m := m) env ts
        pure (.cons r rs)

  /--
  Lower a typed `Term Γ τ` into the execution monad `m`, producing a reference to a tensor of shape
    `τ`.

  This is the "executable" counterpart of `Term.eval`: instead of returning a pure `Spec.Tensor`, we
  emit runtime operations (`Runtime.Autograd.Torch.Ops`) that allocate tensors and apply primitives.
  -/
  def lower
      {Γ : List Shape} {τ : Shape}
      {α : Type 0} [Context α] [DecidableEq Shape]
      {m : Type → Type} [Monad m] [Runtime.Autograd.Torch.Ops (m := m) (α := α)]
      (env : Runtime.Autograd.Torch.RefList (RefT (m := m) (α := α)) Γ) :
      Term Γ τ → m (RefT (m := m) (α := α) τ)
    | .var i => pure (Env.rget (Ref := RefT (m := m) (α := α)) env i)
    | .cast t h =>
        match h with
        | rfl => lower (Γ := Γ) (α := α) (m := m) env t
    | .castEnv t h =>
        -- Same structure as `eval`: rewrite the term to the current environment.
        match h.symm with
        | rfl => lower (Γ := Γ) (α := α) (m := m) env t
    | .op (ins := ins) p args => do
        let rs ← lowerArgs (Γ := Γ) (ins := ins) (α := α) (m := m) env args
        Runtime.Autograd.Torch.CurriedRef.uncurry (ss := ins)
          (Ref := RefT (m := m) (α := α))
          (p.program (α := α)) rs
    | .let1 (σ := σ) t body => do
        let v ← lower (Γ := Γ) (α := α) (m := m) env t
        let env' :=
          Runtime.Autograd.Torch.RefList.append
            (Ref := RefT (m := m) (α := α))
            (ss₁ := Γ) (ss₂ := [σ]) env (.cons v .nil)
        lower (Γ := Γ ++ [σ]) (α := α) (m := m) env' body
end

end Term

namespace Block

open Runtime.Autograd.Torch

/-- Reinterpret a block's result list along an equality of output shapes. -/
def castOutputs {Γ outputs outputs' : List Shape} (h : outputs = outputs') :
    Block Γ outputs → Block Γ outputs'
  | block => h ▸ block

/-- Evaluate a multi-output block, preserving sharing introduced by `let1`. -/
def eval {Γ outs : List Shape} {α : Type 0} [Context α]
    (env : TList α Γ) : Block Γ outs → TList α outs
  | .ret results => Term.evalArgs (Γ := Γ) (α := α) env results
  | .let1 (σ := σ) value body =>
      let result := Term.eval (Γ := Γ) (α := α) env value
      let env' : TList α (Γ ++ [σ]) :=
        Proofs.Autograd.Algebra.TList.append
          (α := α) (ss₁ := Γ) (ss₂ := [σ]) env (.cons result .nil)
      eval (Γ := Γ ++ [σ]) (α := α) env' body

/-- Casting a block's output shapes casts its evaluated typed result by the same equality. -/
@[simp] theorem eval_castOutputs
    {Γ outputs outputs' : List Shape} {α : Type 0} [Context α]
    (env : TList α Γ) (h : outputs = outputs') (block : Block Γ outputs) :
    eval env (castOutputs h block) = h ▸ eval env block := by
  subst outputs'
  rfl

/-- Substituting terms into a block preserves its pure multi-output semantics whenever the
substitution denotes the original environment. -/
theorem eval_substitute {Γ Δ outs : List Shape} {α : Type 0} [Context α]
    (envΓ : TList α Γ) (envΔ : TList α Δ)
    (σ : Substitution Γ Δ) (hσ : Env.SubstitutionSound envΓ envΔ σ) :
    (block : Block Γ outs) → eval envΔ (block.substitute σ) = eval envΓ block
  | .ret results => by
      simp only [Block.substitute, eval]
      exact Term.evalArgs_substitute envΓ envΔ σ hσ results
  | .let1 value body => by
      simp only [Block.substitute, eval]
      rw [Term.eval_substitute envΓ envΔ σ hσ value]
      exact eval_substitute
        (Proofs.Autograd.Algebra.TList.append envΓ
          (.cons (Term.eval envΓ value) .nil))
        (Proofs.Autograd.Algebra.TList.append envΔ
          (.cons (Term.eval envΓ value) .nil))
        (Substitution.liftRight σ) (hσ.liftRight (Term.eval envΓ value)) body

/-- Evaluating an inlined block is the same as evaluating its original body under the supplied
typed argument values. -/
theorem eval_instantiate
    {Γ Δ outs : List Shape} {α : Type 0} [Context α]
    (env : TList α Δ) (args : Args Δ Γ) (block : Block Γ outs) :
    eval env (block.instantiate args) = eval (Term.evalArgs env args) block := by
  apply eval_substitute (Term.evalArgs env args) env (Args.get args)
  intro shape termVar
  exact Term.eval_get env args termVar

/-- Pure evaluation of block composition is ordinary typed environment extension. -/
theorem eval_andThenAux
    {Γ middle outputs : List Shape} {α : Type 0} [Context α]
    (envΓ : TList α Γ) (second : Block (Γ ++ middle) outputs) :
    ∀ {Δ : List Shape} (first : Block Δ middle) (envΔ : TList α Δ)
      (ρ : {s : Shape} → Var Γ s → Var Δ s),
      Env.RenamingSound envΓ envΔ ρ →
      eval envΔ (andThenAux ρ first second) =
        eval (Proofs.Autograd.Algebra.TList.append envΓ (eval envΔ first)) second
  | _, .ret results, envΔ, ρ, hρ => by
      simp only [andThenAux, eval_instantiate, Term.evalArgs_append]
      rw [Term.evalArgs_rename envΓ envΔ ρ hρ, Term.evalArgs_vars]
      rfl
  | _, .let1 value body, envΔ, ρ, hρ => by
      let result := Term.eval envΔ value
      let envΔ' := Proofs.Autograd.Algebra.TList.append envΔ (.cons result .nil)
      have hρ' : Env.RenamingSound envΓ envΔ' (fun v => Var.weakenRight (ρ v)) := by
        intro shape v
        exact (Env.tget_append_weakenRight envΔ result (ρ v)).trans (hρ v)
      simpa only [andThenAux, eval, result, envΔ'] using
        eval_andThenAux envΓ second body envΔ' (fun v => Var.weakenRight (ρ v)) hρ'

/-- Composing two blocks evaluates the first once and appends its typed outputs for the second. -/
theorem eval_andThen {Γ middle outputs : List Shape} {α : Type 0} [Context α]
    (env : TList α Γ) (first : Block Γ middle) (second : Block (Γ ++ middle) outputs) :
    eval env (first.andThen second) =
      eval (Proofs.Autograd.Algebra.TList.append env (eval env first)) second := by
  unfold andThen
  apply eval_andThenAux env second first env (fun v => v)
  intro shape v
  rfl

/-- Lower a multi-output block for an arbitrary TorchLean execution target. -/
def lower {Γ outs : List Shape} {α : Type 0} [Context α] [DecidableEq Shape]
    {μ : Type → Type} [Monad μ] [Runtime.Autograd.Torch.Ops (m := μ) (α := α)]
    (env : RefList (Term.RefT (m := μ) (α := α)) Γ) :
    Block Γ outs → μ (RefList (Term.RefT (m := μ) (α := α)) outs)
  | .ret results => Term.lowerArgs (Γ := Γ) (α := α) (m := μ) env results
  | .let1 (σ := σ) value body => do
      let result ← Term.lower (Γ := Γ) (α := α) (m := μ) env value
      let env' := RefList.append
        (Ref := Term.RefT (m := μ) (α := α))
        (ss₁ := Γ) (ss₂ := [σ]) env (.cons result .nil)
      lower (Γ := Γ ++ [σ]) (α := α) (μ := μ) env' body

end Block

/-! ## Model wrapper -/

/--
A small “model” wrapper around DAG terms.

This mirrors the sequential `Chain` surface:

- `ps` are parameter tensor shapes (tracked at the type level),
- `ins` are the shapes of *non-parameter inputs* (e.g. data tensors),
- `τ` is the output shape.

The model body is a `Term (ps ++ ins) τ`, i.e. it expects an environment that starts with
parameters and then contains the actual inputs.
 -/
structure Model (ps ins : List Shape) (τ : Shape) where
  /-- init Params. -/
  initParams : Runtime.Autograd.Torch.TList Float ps
  /-- body. -/
  body : Term (ps ++ ins) τ

namespace Model

open Runtime.Autograd.Torch

/-- Inline a model body into a larger graph using explicit parameter and input terms.

The result is still an ordinary DAG term: no primitive boundary is introduced, and subsequent
lowering, differentiation, or numerical analysis can inspect every operation of the model.
-/
def inline {Γ ps ins : List Shape} {τ : Shape} (model : Model ps ins τ)
    (params : Args Γ ps) (inputs : Args Γ ins) : Term Γ τ :=
  model.body.instantiate (Args.append params inputs)

/--
Pure forward semantics of a DAG model.

We build the full environment `Γ = ps ++ ins` by appending the parameter list and the input list,
then evaluate the body using `Term.eval`.
 -/
def specFwd {ps ins : List Shape} {τ : Shape} (m : Model ps ins τ)
    {α : Type 0} [Context α]
    (params : TList α ps) (xs : TList α ins) : Spec.Tensor α τ :=
  let env := Proofs.Autograd.Algebra.TList.append (α := α) (ss₁ := ps) (ss₂ := ins) params xs
  Term.eval (Γ := ps ++ ins) (α := α) env m.body

/-- Inlining a model into a larger DAG preserves the model's pure forward semantics. -/
theorem eval_inline {Γ ps ins : List Shape} {τ : Shape}
    (model : Model ps ins τ) {α : Type 0} [Context α]
    (env : TList α Γ) (params : Args Γ ps) (inputs : Args Γ ins) :
    Term.eval env (model.inline params inputs) =
      model.specFwd (Term.evalArgs env params) (Term.evalArgs env inputs) := by
  rw [inline, Term.eval_instantiate, Term.evalArgs_append]
  rfl

/--
Lower a DAG model to an execution-polymorphic TorchLean program.

The resulting program expects arguments in the order `ps ++ ins` (parameters first, then inputs),
matching the environment discipline used by `specFwd`.
 -/
def toProgram {ps ins : List Shape} {τ : Shape} (m : Model ps ins τ)
    {α : Type 0} [Context α] [DecidableEq Shape] :
    Runtime.Autograd.TorchLean.Program α (ps ++ ins) τ :=
  fun {μ} _ _ =>
    Runtime.Autograd.Torch.CurriedRef.curry
      (Ref := Term.RefT (m := μ) (α := α))
      (ss := ps ++ ins)
      (β := μ (Term.RefT (m := μ) (α := α) τ))
      (fun args => Term.lower (Γ := ps ++ ins) (α := α) (m := μ) args m.body)

end Model

/-! ## Models with several outputs -/

/-- A typed DAG model that returns several tensors.

Recurrent layers commonly return both an updated state and an observable output. Keeping these as
a typed list preserves their individual shapes and avoids flattening unrelated tensors into an
untyped buffer. The block body uses shared `let1` bindings, so an updated state can be computed once
and returned alongside values derived from it.
-/
structure MultiModel (ps ins outs : List Shape) where
  /-- Default parameters, in the same ABI order used by `body`. -/
  initParams : Runtime.Autograd.Torch.TList Float ps
  /-- Shared computation ending in one well-typed term for every output shape. -/
  body : Block (ps ++ ins) outs

namespace MultiModel

open Runtime.Autograd.Torch

/-- Inline a multi-output model into a larger graph.

Shared `let` bindings in the original block remain shared after substitution, so recurrent state
updates are not duplicated when both the state and a derived output are returned.
-/
def inline {Γ ps ins outs : List Shape} (model : MultiModel ps ins outs)
    (params : Args Γ ps) (inputs : Args Γ ins) : Block Γ outs :=
  model.body.instantiate (Args.append params inputs)

/-- Pure reference semantics of a multi-output DAG model. -/
def specFwd {ps ins outs : List Shape} (m : MultiModel ps ins outs)
    {α : Type 0} [Context α] (params : TList α ps) (xs : TList α ins) : TList α outs :=
  let env := Proofs.Autograd.Algebra.TList.append
    (α := α) (ss₁ := ps) (ss₂ := ins) params xs
  Block.eval (Γ := ps ++ ins) (α := α) env m.body

/-- Inlining a multi-output model preserves every output and every shared intermediate in its
pure reference semantics. -/
theorem eval_inline {Γ ps ins outs : List Shape} (model : MultiModel ps ins outs)
    {α : Type 0} [Context α] (env : TList α Γ)
    (params : Args Γ ps) (inputs : Args Γ ins) :
    Block.eval env (model.inline params inputs) =
      model.specFwd (Term.evalArgs env params) (Term.evalArgs env inputs) := by
  rw [inline, Block.eval_instantiate, Term.evalArgs_append]
  rfl

/-- An execution-polymorphic program returning several shape-indexed tensor references. -/
abbrev MultiOutputProgram (α : Type 0) [Context α] [DecidableEq Shape]
    (ins outs : List Shape) : Type 1 :=
  ∀ {μ : Type → Type}, [Monad μ] → [Runtime.Autograd.Torch.Ops (m := μ) (α := α)] →
    CurriedRef (fun s => Term.RefT (m := μ) (α := α) s) ins
      (μ (RefList (Term.RefT (m := μ) (α := α)) outs))

/-- Lower every result of a multi-output model for the selected TorchLean execution target. -/
def toProgram {ps ins outs : List Shape} (m : MultiModel ps ins outs)
    {α : Type 0} [Context α] [DecidableEq Shape] : MultiOutputProgram α (ps ++ ins) outs :=
  fun {μ} _ _ =>
    CurriedRef.curry
      (Ref := Term.RefT (m := μ) (α := α))
      (ss := ps ++ ins)
      (β := μ (RefList (Term.RefT (m := μ) (α := α)) outs))
      (fun args => Block.lower (Γ := ps ++ ins) (α := α) (μ := μ) args m.body)

end MultiModel

end DAG
end GraphSpec
end NN
