/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved.Correctness.Eval.LinearAlgebra

/-!
# Concat IR Evaluation

Local semantics for IR concat.  The evaluator keeps the generic-axis implementation in the shared
`Graph.evalConcat` helper, which moves the requested axis to the front, folds
`Tensor.concatLeadingAxisSpec`, and moves the result back. `LeadingAxisConcat.Input` packages an
input with its leading dimension, `LeadingAxisConcat.fold` specifies nonempty list concatenation,
and `evalAt_concat_leadingAxis_eq` proves end-to-end graph evaluation correct for every arity of at
least two.
-/

@[expose] public section

namespace NN.Verification.TorchLean.Proved

open _root_.Spec
open _root_.Spec.Tensor
open NN.IR

namespace Correctness

namespace IRStep

/-- Local IR semantics for binary concat, pinned to the shared generic concat interpreter. -/
theorem evalAt_concat_binary_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s₁ s₂ out : Shape} (axis : Nat)
    (lhs : Tensor α s₁) (rhs : Tensor α s₂) :
    Graph.evalAt (α := α)
        (g := binaryGraphOut (.concat axis) s₁ s₂ out)
        (payload := {})
        (input := DVal.mk (α := α) s₁ lhs)
        (vals := #[
          DVal.mk (α := α) s₁ lhs,
          DVal.mk (α := α) s₂ rhs
        ]) (i := 2)
      =
      (Graph.evalConcat (α := α) 2 (binaryNodeOut (.concat axis) out) axis
          [DVal.mk (α := α) s₁ lhs, DVal.mk (α := α) s₂ rhs]).bind
        (Graph.normalizeNodeOutput (α := α) 2 (binaryNodeOut (.concat axis) out)) := by
  simp [Graph.evalAt, binaryGraphOut, binaryNodeOut, Graph.getNode, Graph.getNode?,
    Graph.normalizeNodeOutput, Bind.bind, Except.bind, Pure.pure, Except.pure]

/--
Successful binary concat evaluation, once the shared concat interpreter has produced a value with
the node's declared output shape.
-/
theorem evalAt_concat_binary_ok
    {α : Type} [Context α] [DecidableEq Shape]
    {s₁ s₂ out : Shape} (axis : Nat)
    (lhs : Tensor α s₁) (rhs : Tensor α s₂) (y : Tensor α out)
    (hConcat :
      Graph.evalConcat (α := α) 2 (binaryNodeOut (.concat axis) out) axis
          [DVal.mk (α := α) s₁ lhs, DVal.mk (α := α) s₂ rhs] =
        .ok (DVal.mk (α := α) out y)) :
    Graph.evalAt (α := α)
        (g := binaryGraphOut (.concat axis) s₁ s₂ out)
        (payload := {})
        (input := DVal.mk (α := α) s₁ lhs)
        (vals := #[
          DVal.mk (α := α) s₁ lhs,
          DVal.mk (α := α) s₂ rhs
        ]) (i := 2)
      =
      .ok (DVal.mk (α := α) out y) := by
  rw [evalAt_concat_binary_eq]
  rw [hConcat]
  simp [Graph.normalizeNodeOutput, binaryNodeOut, Except.bind, Pure.pure, Except.pure]

/-- Binary concat evaluation rejects the node whenever the shared concat interpreter rejects it. -/
theorem evalAt_concat_binary_error
    {α : Type} [Context α] [DecidableEq Shape]
    {s₁ s₂ out : Shape} (axis : Nat)
    (lhs : Tensor α s₁) (rhs : Tensor α s₂) (msg : String)
    (hConcat :
      Graph.evalConcat (α := α) 2 (binaryNodeOut (.concat axis) out) axis
          [DVal.mk (α := α) s₁ lhs, DVal.mk (α := α) s₂ rhs] =
        .error msg) :
    Graph.evalAt (α := α)
        (g := binaryGraphOut (.concat axis) s₁ s₂ out)
        (payload := {})
        (input := DVal.mk (α := α) s₁ lhs)
        (vals := #[
          DVal.mk (α := α) s₁ lhs,
          DVal.mk (α := α) s₂ rhs
        ]) (i := 2)
      =
      .error msg := by
  rw [evalAt_concat_binary_eq]
  rw [hConcat]
  rfl

namespace LeadingAxisConcat

/-- Shapes with a common tail and the supplied leading dimensions. -/
def shapes (rest : Shape) (sizes : List Nat) : List Shape :=
  sizes.map (fun size => .dim size rest)

/-- Sum a nonempty sequence of leading dimensions from left to right. -/
def totalSize (head : Nat) (tail : List Nat) : Nat :=
  tail.foldl (· + ·) head

/-- A tensor whose leading dimension is existentially packed while its tail shape stays fixed. -/
abbrev Input (α : Type) [Context α] (rest : Shape) :=
  Sigma fun size => Tensor α (.dim size rest)

/-- Erase a packed leading-axis input to the dynamic value consumed by the IR evaluator. -/
def Input.toDVal {α : Type} [Context α] {rest : Shape} (input : Input α rest) : DVal α :=
  DVal.mk (α := α) (.dim input.1 rest) input.2

/-- Concatenate a nonempty sequence of compatible leading-axis inputs from left to right. -/
def fold {α : Type} [Context α] {rest : Shape}
    (head : Input α rest) (tail : List (Input α rest)) : Input α rest :=
  tail.foldl
    (fun acc next =>
      ⟨acc.1 + next.1,
        Tensor.concatLeadingAxisSpec (α := α) (n := acc.1) (m := next.1) (s := rest)
          acc.2 next.2⟩)
    head

/-- The leading dimension of a concat fold is the left-associated sum of its input dimensions. -/
theorem fold_size {α : Type} [Context α] {rest : Shape}
    (head : Input α rest) (tail : List (Input α rest)) :
    (fold head tail).1 = (tail.map Sigma.fst).foldl (· + ·) head.1 := by
  induction tail generalizing head with
  | nil => rfl
  | cons next tail ih =>
      unfold fold
      rw [List.foldl_cons, List.map_cons, List.foldl_cons]
      simpa only [fold] using
        (ih (head :=
          ⟨head.1 + next.1,
            Tensor.concatLeadingAxisSpec (α := α) (n := head.1) (m := next.1) (s := rest)
              head.2 next.2⟩))

end LeadingAxisConcat

/-- Packaging and then decoding a typed leading-axis input preserves it exactly. -/
@[simp] theorem expectLeadingAxisInput_toDVal
    {α : Type} [Context α] [DecidableEq Shape]
    {rest : Shape} (i : Nat) (input : LeadingAxisConcat.Input α rest) :
    Graph.expectLeadingAxisInput (α := α) i rest input.toDVal = .ok input := by
  rcases input with ⟨size, tensor⟩
  simp [Graph.expectLeadingAxisInput, LeadingAxisConcat.Input.toDVal, DVal.mk,
    Pure.pure, Except.pure]

/-- Decoding a list of typed leading-axis inputs after packaging preserves the whole list. -/
@[simp] theorem mapM_expectLeadingAxisInput_toDVal
    {α : Type} [Context α] [DecidableEq Shape]
    {rest : Shape} (i : Nat) (inputs : List (LeadingAxisConcat.Input α rest)) :
    (inputs.map LeadingAxisConcat.Input.toDVal).mapM
        (Graph.expectLeadingAxisInput (α := α) i rest) =
      .ok inputs := by
  induction inputs with
  | nil => rfl
  | cons input inputs ih =>
      rw [List.map_cons, List.mapM_cons, expectLeadingAxisInput_toDVal, ih]
      rfl

/--
The dynamic leading-axis evaluator agrees with the typed concat fold for every nonempty input
list. This single list-indexed result subsumes the former pair, triple, and quadruple theorems.
-/
theorem evalConcatLeadingAxisFold_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {rest : Shape} (i : Nat)
    (head : LeadingAxisConcat.Input α rest)
    (tail : List (LeadingAxisConcat.Input α rest)) :
    let result := LeadingAxisConcat.fold head tail
    Graph.evalConcatLeadingAxisFold (α := α) i result.1 rest
        ((head :: tail).map LeadingAxisConcat.Input.toDVal)
      = .ok (LeadingAxisConcat.Input.toDVal result) := by
  unfold Graph.evalConcatLeadingAxisFold
  rw [mapM_expectLeadingAxisInput_toDVal (α := α) i (head :: tail)]
  simp [LeadingAxisConcat.fold, LeadingAxisConcat.Input.toDVal, DVal.mk,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Leading-axis shape inference sums every input's leading dimension for any arity of at least two. -/
theorem inferConcatOutShape_leadingAxis_eq
    {rest : Shape} (first second : Nat) (tail : List Nat) :
    OpContracts.inferConcatOutShape 0
        (LeadingAxisConcat.shapes rest (first :: second :: tail)) =
      .ok (.dim (LeadingAxisConcat.totalSize first (second :: tail)) rest) := by
  have hAxis : OpContracts.checkAxisValid 0 (.dim first rest) = .ok () := by
    unfold OpContracts.checkAxisValid
    change (if 0 < 1 + Spec.Shape.rank rest then Except.ok () else
      Except.error s!"invalid axis {0} for rank {Spec.Shape.rank (.dim first rest)}") = Except.ok ()
    simp
  have hTail : (rest != rest) = false := shapeBNe_refl rest
  have hRanks (sizes : List Nat) :
      OpContracts.checkConcatRanks (Spec.Shape.rank (.dim first rest))
          (LeadingAxisConcat.shapes rest sizes) = .ok () := by
    induction sizes with
    | nil => rfl
    | cons size sizes ih =>
        have hRank :
            Spec.Shape.rank (.dim size rest) = Spec.Shape.rank (.dim first rest) := rfl
        change OpContracts.checkConcatRanks (Spec.Shape.rank (.dim first rest))
          (sizes.map (fun size => .dim size rest)) = .ok () at ih
        simpa [LeadingAxisConcat.shapes, OpContracts.checkConcatRanks, hRank] using ih
  have hLeading (sizes : List Nat) (acc : Nat) :
      OpContracts.inferConcatLeadingAxis rest acc
          (LeadingAxisConcat.shapes rest sizes) =
        .ok (.dim (sizes.foldl (· + ·) acc) rest) := by
    induction sizes generalizing acc with
    | nil => rfl
    | cons size sizes ih =>
        change ∀ acc,
          OpContracts.inferConcatLeadingAxis rest acc
              (sizes.map (fun size => .dim size rest)) =
            .ok (.dim (sizes.foldl (· + ·) acc) rest) at ih
        simpa [LeadingAxisConcat.shapes, OpContracts.inferConcatLeadingAxis, hTail] using
          ih (acc + size)
  have hRanksAll := hRanks (first :: second :: tail)
  have hLeadingAll := hLeading (second :: tail) first
  simp only [LeadingAxisConcat.shapes, List.map_cons] at hRanksAll hLeadingAll
  simp [OpContracts.inferConcatOutShape, OpContracts.inferConcatOutShape.go,
    LeadingAxisConcat.shapes, LeadingAxisConcat.totalSize, hAxis, hRanksAll, hLeadingAll,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- The shared concat interpreter agrees with the typed fold for every arity of at least two. -/
theorem evalConcat_leadingAxis_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {rest : Shape} (i : Nat)
    (first second : LeadingAxisConcat.Input α rest)
    (tail : List (LeadingAxisConcat.Input α rest)) :
    let inputs := first :: second :: tail
    let result := LeadingAxisConcat.fold first (second :: tail)
    let parentShapes :=
      (inputs.map fun input => Shape.dim input.1 rest).toArray
    Graph.evalConcat (α := α) i
        (variadicNodeOut (.concat 0) parentShapes (.dim result.1 rest)) 0
        (inputs.map LeadingAxisConcat.Input.toDVal) =
      .ok result.toDVal := by
  dsimp only
  have hInfer :
      OpContracts.inferConcatOutShape 0
          ((first :: second :: tail).map fun input => Shape.dim input.1 rest) =
        .ok (.dim (LeadingAxisConcat.fold first (second :: tail)).1 rest) := by
    have hTailShapes :
        tail.map (fun input => Shape.dim input.1 rest) =
          (tail.map Sigma.fst).map (fun size => Shape.dim size rest) := by
      induction tail with
      | nil => rfl
      | cons input tail ih => simp only [List.map_cons, ih]
    simp only [List.map_cons]
    rw [hTailShapes]
    simpa only [LeadingAxisConcat.shapes, LeadingAxisConcat.totalSize,
      LeadingAxisConcat.fold_size, List.map_cons] using
      (inferConcatOutShape_leadingAxis_eq (rest := rest) first.1 second.1
        (tail.map Sigma.fst))
  have hSame :
      (Shape.dim (LeadingAxisConcat.fold first (second :: tail)).1 rest !=
        Shape.dim (LeadingAxisConcat.fold first (second :: tail)).1 rest) = false :=
    shapeBNe_refl _
  have hShapes :
      ((first :: second :: tail).map LeadingAxisConcat.Input.toDVal).map DVal.shape =
        (first :: second :: tail).map (fun input => Shape.dim input.1 rest) := by
    simp [LeadingAxisConcat.Input.toDVal]
  unfold Graph.evalConcat
  rw [hShapes, hInfer]
  simp only [Bind.bind, Except.bind, Pure.pure, Except.pure, variadicNodeOut]
  rw [hSame]
  simp only [Bool.false_eq_true, ↓reduceIte]
  exact evalConcatLeadingAxisFold_eq (α := α) i first (second :: tail)

/--
End-to-end local IR semantics for leading-axis concat with any number of inputs greater than one.
The graph, parent values, inferred output dimension, and tensor result are all derived from the same
packed input list.
-/
theorem evalAt_concat_leadingAxis_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {rest : Shape}
    (first second : LeadingAxisConcat.Input α rest)
    (tail : List (LeadingAxisConcat.Input α rest)) :
    let inputs := first :: second :: tail
    let result := LeadingAxisConcat.fold first (second :: tail)
    let parentShapes :=
      (inputs.map fun input => Shape.dim input.1 rest).toArray
    let values := (inputs.map LeadingAxisConcat.Input.toDVal).toArray
    Graph.evalAt (α := α)
        (g := variadicGraphOut (.concat 0) parentShapes (.dim result.1 rest))
        (payload := {}) (input := first.toDVal) (vals := values)
        (i := parentShapes.size) =
      .ok result.toDVal := by
  dsimp only
  let inputs := first :: second :: tail
  let parentShapes := (inputs.map fun input => Shape.dim input.1 rest).toArray
  let values := (inputs.map LeadingAxisConcat.Input.toDVal).toArray
  have hSize : values.size = parentShapes.size := by
    simp [values, parentShapes, inputs]
  have hParents :
      (List.range parentShapes.size).map (fun parentId => values[parentId]!) =
        inputs.map LeadingAxisConcat.Input.toDVal := by
    rw [← hSize, range_map_getElem!_eq_toList]
  change Graph.evalAt (α := α)
      (g := variadicGraphOut (.concat 0) parentShapes
        (.dim (LeadingAxisConcat.fold first (second :: tail)).1 rest))
      (payload := {}) (input := first.toDVal) (vals := values)
      (i := parentShapes.size) = _
  unfold Graph.evalAt
  rw [variadicGraphOut_getNode]
  simp only [Bind.bind, Except.bind]
  change (Graph.evalConcat (α := α) parentShapes.size
      (variadicNodeOut (.concat 0) parentShapes
        (.dim (LeadingAxisConcat.fold first (second :: tail)).1 rest)) 0
      ((List.range parentShapes.size).map fun parentId => values[parentId]!)).bind _ = _
  rw [hParents]
  rw [evalConcat_leadingAxis_eq (α := α) parentShapes.size first second tail]
  change Graph.normalizeNodeOutput (α := α) parentShapes.size
      (variadicNodeOut (.concat 0) parentShapes
        (.dim (LeadingAxisConcat.fold first (second :: tail)).1 rest))
      (LeadingAxisConcat.fold first (second :: tail)).toDVal = _
  exact Graph.normalizeNodeOutput_declared _ _ _

end IRStep

end Correctness

end NN.Verification.TorchLean.Proved
