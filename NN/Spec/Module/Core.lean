/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor

/-!
# Mathematical modules

`Spec.Module α σ τ` packages a pure tensor map from shape `σ` to shape `τ`. The input and output
shapes occur in the type, so ill-shaped compositions are rejected by Lean.

`Spec.Module.Chain` composes these maps. The `kind` and `pythonExpr` fields are descriptive metadata;
`forward` alone gives the module its mathematical meaning.
-/

@[expose] public section


namespace Spec

open Tensor

/-- A pure, shape-indexed tensor map with descriptive code-generation metadata. -/
structure Module (α : Type) (inShape outShape : Shape) where
  /-- The mathematical meaning of the module. -/
  forward : Tensor α inShape → Tensor α outShape
  /-- A stable operation name used in reports and exported graphs. -/
  kind : String
  /-- A Python expression used by the source exporter. This field is not part of the semantics. -/
  pythonExpr : String

namespace Module

/--
Dependent composition of modules whose adjacent shapes agree definitionally.
-/
inductive Chain (α : Type) : Shape → Shape → Type where
| single {s t} (m : Module α s t) : Chain α s t
| comp {s t u} (a : Chain α s t) (b : Chain α t u) : Chain α s u

namespace Chain

/-- Evaluate a chain from left to right. -/
def forward {α : Type} {s t : Shape} (c : Chain α s t) : Tensor α s → Tensor α t :=
  match c with
  | .single m => m.forward
  | .comp a b => fun x =>
      let y := forward a x
      forward b y

/-- Append one module to the output of a chain. -/
def append {α : Type} {s t u : Shape}
    (a : Chain α s t) (b : Module α t u) : Chain α s u :=
  .comp a (.single b)

/-- Return operation names in evaluation order. -/
def kinds {α : Type} {s t : Shape} : Chain α s t → List String
  | .single m => [m.kind]
  | .comp a b => kinds a ++ kinds b

/-- Return operation names and Python expressions in evaluation order. -/
def layerInfo {α : Type} {s t : Shape} : Chain α s t → List (String × String)
  | .single m => [(m.kind, m.pythonExpr)]
  | .comp a b => layerInfo a ++ layerInfo b

end Chain

/-- Apply the same module independently to every element of a leading dimension. -/
def mapLeading
  {α : Type} [Context α]
    {n : Nat} {elemIn elemOut : Shape} (m : Module α elemIn elemOut) :
    Module α (.dim n elemIn) (.dim n elemOut) where
  forward
    | .dim values => .dim fun i => m.forward (values i)
  kind := m.kind
  pythonExpr := m.pythonExpr

/-- Select one element from the leading dimension. -/
def selectLeading {α : Type} {n : Nat} {elem : Shape} (i : Fin n) :
    Module α (.dim n elem) elem where
  forward
    | .dim values => values i
  kind := "SelectLeading"
  pythonExpr := s!"SelectLeading({i.val})"

end Module
end Spec
