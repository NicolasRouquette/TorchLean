/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Interpreter
import NN.Tactic.Except

/-!
# Compositional properties of typed tensor programs

Theorems here lift local operation rules through the existing typed interpreter.
They use the same program, scalar families, payloads and evaluation functions as execution.

* `run_induction` carries a predicate on values and conversion flags through a successful run.
* `run_rel` carries a relation between two successful runs, allowing different interpreters,
  inputs and parameter payloads.
* `eval_rel` transfers pointwise node relations to the designated results in return order.

These rules remove the traversal argument from proofs about a tensor program. A numerical bound,
permutation property or other relation still needs correct rules for its operations, including
input and constant nodes. Successful execution is an explicit hypothesis; the rules do not prove
that a program executes for every input or that an external backend implements its semantics.
-/

@[expose] public section

namespace NN.IR.Interpreter

open FloatLib.Floats.ExecFloat (ConversionStatus)

private theorem foldlM_rel {α σ τ : Type}
    (nodes : List α) (left : σ → α → Except String σ)
    (right : τ → α → Except String τ) (relation : σ → τ → Prop)
    (step : ∀ node ∈ nodes, ∀ a b c d, relation a b →
      left a node = .ok c → right b node = .ok d → relation c d)
    {a b c d} (initial : relation a b)
    (hl : nodes.foldlM left a = .ok c) (hr : nodes.foldlM right b = .ok d) :
    relation c d := by
  induction nodes generalizing a b with
  | nil =>
      simp only [List.foldlM_nil, pure, Except.pure, Except.ok.injEq] at hl hr
      subst c d
      exact initial
  | cons node nodes ih =>
      simp only [List.foldlM_cons] at hl hr
      cases hleft : left a node with
      | error message =>
          simp only [hleft, Bind.bind, Except.bind] at hl
          cases hl
      | ok nextLeft =>
          cases hright : right b node with
          | error message =>
              simp only [hright, Bind.bind, Except.bind] at hr
              cases hr
          | ok nextRight =>
              simp only [hleft, Bind.bind, Except.bind] at hl
              simp only [hright, Bind.bind, Except.bind] at hr
              apply ih (fun n hn => step n (List.mem_cons_of_mem node hn))
                (step node (List.mem_cons_self ..) a b nextLeft nextRight initial hleft hright)
                hl hr

private theorem run_foldlM (interpreter : Interpreter) (program : Program)
    (payload : Payload interpreter) (arguments : Array (Value interpreter))
    (checked : program.check = .ok ()) (arity : arguments.size = program.inputs.size) :
    interpreter.run program payload arguments =
      program.graph.nodes.foldlM (fun state node => do
        let (value, status) ← interpreter.evalNode program payload arguments state.1 node
        pure (state.1.push value, state.2.merge status)) (#[], {}) := by
  simp only [run, checked, Bind.bind, Except.bind, arity, beq_self_eq_true, ↓reduceIte]
  change (forIn program.graph.nodes (#[], ({} : ConversionStatus))
    (fun node state => do
      let (value, status) ← interpreter.evalNode program payload arguments state.1 node
      pure (.yield (state.1.push value, state.2.merge status))) >>= pure) = _
  rw [bind_pure]
  simpa only [Functor.map, Except.map, Bind.bind, Except.bind, Pure.pure, Except.pure] using
    Array.forIn_yield_eq_foldlM
      (fun node state => interpreter.evalNode program payload arguments state.1 node)
      (fun _ state result => (state.1.push result.1, state.2.merge result.2))
      (#[], ({} : ConversionStatus))

/-- A relation preserved by each operation holds between the completed value tables.

The interpreters may use different scalar representations, and their inputs and payloads may
also differ. The local rule must cover those differences for input, constant and numerical nodes.
The relation can describe the whole prefix and accumulated conversion flags, allowing bounds
and exception conditions to depend on earlier results.
-/
theorem run_rel {left right : Interpreter} (program : Program)
    (leftPayload : Payload left) (rightPayload : Payload right)
    (leftArguments : Array (Value left)) (rightArguments : Array (Value right))
    (relation : (Array (Value left) × ConversionStatus) →
      (Array (Value right) × ConversionStatus) → Prop)
    (initial : relation (#[], {}) (#[], {}))
    (step : ∀ node ∈ program.graph.nodes,
      ∀ leftState rightState leftValue rightValue leftStatus rightStatus,
      relation leftState rightState →
      left.evalNode program leftPayload leftArguments leftState.1 node =
        .ok (leftValue, leftStatus) →
      right.evalNode program rightPayload rightArguments rightState.1 node =
        .ok (rightValue, rightStatus) →
      relation (leftState.1.push leftValue, leftState.2.merge leftStatus)
        (rightState.1.push rightValue, rightState.2.merge rightStatus))
    {leftValues rightValues leftFlags rightFlags}
    (hl : left.run program leftPayload leftArguments = .ok (leftValues, leftFlags))
    (hr : right.run program rightPayload rightArguments = .ok (rightValues, rightFlags)) :
    relation (leftValues, leftFlags) (rightValues, rightFlags) := by
  cases checked : program.check with
  | error message =>
      simp only [run, checked, Bind.bind, Except.bind] at hl
      cases hl
  | ok checkedValue =>
      cases checkedValue
      by_cases leftArity : leftArguments.size = program.inputs.size
      · by_cases rightArity : rightArguments.size = program.inputs.size
        · rw [run_foldlM left program leftPayload leftArguments checked leftArity,
            ← Array.foldlM_toList] at hl
          rw [run_foldlM right program rightPayload rightArguments checked rightArity,
            ← Array.foldlM_toList] at hr
          apply foldlM_rel program.graph.nodes.toList _ _ relation ?_ initial hl hr
          intro node member a b c d related hleft hright
          cases hevalLeft : left.evalNode program leftPayload leftArguments a.1 node with
          | error message =>
              simp only [hevalLeft, Bind.bind, Except.bind] at hleft
              cases hleft
          | ok resultLeft =>
              obtain ⟨valueLeft, statusLeft⟩ := resultLeft
              cases hevalRight : right.evalNode program rightPayload rightArguments b.1 node with
              | error message =>
                  simp only [hevalRight, Bind.bind, Except.bind] at hright
                  cases hright
              | ok resultRight =>
                  obtain ⟨valueRight, statusRight⟩ := resultRight
                  simp only [hevalLeft, Bind.bind, Except.bind, Pure.pure, Except.pure,
                    Except.ok.injEq] at hleft
                  simp only [hevalRight, Bind.bind, Except.bind, Pure.pure, Except.pure,
                    Except.ok.injEq] at hright
                  subst c d
                  exact step node (by simpa using member) a b valueLeft valueRight
                    statusLeft statusRight related hevalLeft hevalRight
        · simp [run, checked, rightArity, Bind.bind, Except.bind] at hr
      · simp [run, checked, leftArity, Bind.bind, Except.bind] at hl

/-- Carry an invariant from the empty value table through every successful operation.

The predicate can express bounds on intermediate tensors or conditions on conversion flags.
It is supplied by the proof, independently of the frontend or model architecture.
-/
theorem run_induction (interpreter : Interpreter) (program : Program)
    (payload : Payload interpreter) (arguments : Array (Value interpreter))
    (property : (Array (Value interpreter) × ConversionStatus) → Prop)
    (initial : property (#[], {}))
    (step : ∀ node ∈ program.graph.nodes, ∀ state value status,
      property state →
      interpreter.evalNode program payload arguments state.1 node = .ok (value, status) →
      property (state.1.push value, state.2.merge status))
    {values flags} (h : interpreter.run program payload arguments = .ok (values, flags)) :
    property (values, flags) := by
  apply run_rel program payload payload arguments arguments (fun state _ => property state)
    initial ?_ h h
  intro node member leftValues _ leftValue _ leftStatus _ invariant heval _
  exact step node member leftValues leftValue leftStatus invariant heval

private theorem lookup_rel {α β : Type} {relation : α → β → Prop}
    {left : Array α} {right : Array β}
    (related : List.Forall₂ relation left.toList right.toList)
    {id : Nat} {a : α} {b : β} (hl : left[id]? = some a) (hr : right[id]? = some b) :
    relation a b := by
  obtain ⟨hiLeft, rfl⟩ := Array.getElem?_eq_some_iff.mp hl
  obtain ⟨hiRight, rfl⟩ := Array.getElem?_eq_some_iff.mp hr
  have h := related.get (i := id) (by simpa using hiLeft) (by simpa using hiRight)
  change relation left.toList[id] right.toList[id] at h
  rw [Array.getElem_toList, Array.getElem_toList] at h
  exact h

/-- Operation rules imply a relation between corresponding returned tensors.

This includes multiple arguments, repeated result nodes and any return order. The rules apply to
the program's actual typed operations and payloads; no architecture specification is required.
-/
theorem eval_rel {left right : Interpreter} (program : Program)
    (leftPayload : Payload left) (rightPayload : Payload right)
    (leftArguments : Array (Value left)) (rightArguments : Array (Value right))
    (relation : Value left → Value right → Prop)
    (step : ∀ node ∈ program.graph.nodes,
      ∀ leftValues rightValues leftValue rightValue leftStatus rightStatus,
      List.Forall₂ relation leftValues.toList rightValues.toList →
      left.evalNode program leftPayload leftArguments leftValues node =
        .ok (leftValue, leftStatus) →
      right.evalNode program rightPayload rightArguments rightValues node =
        .ok (rightValue, rightStatus) → relation leftValue rightValue)
    {leftOutputs rightOutputs leftFlags rightFlags}
    (hl : left.eval program leftPayload leftArguments = .ok (leftOutputs, leftFlags))
    (hr : right.eval program rightPayload rightArguments = .ok (rightOutputs, rightFlags)) :
    List.Forall₂ relation leftOutputs.toList rightOutputs.toList := by
  unfold eval at hl hr
  except_cases hleft : left.run program leftPayload leftArguments using hl with leftResult =>
    obtain ⟨leftValues, leftStatus⟩ := leftResult
    except_cases hright : right.run program rightPayload rightArguments using hr with rightResult =>
      obtain ⟨rightValues, rightStatus⟩ := rightResult
      have related := run_rel program leftPayload rightPayload leftArguments rightArguments
        (fun a b => List.Forall₂ relation a.1.toList b.1.toList) .nil
        (fun node member a b c d fs gs h ha hb => by
          simpa only [Array.toList_push] using
            List.rel_append h (.cons (step node member a.1 b.1 c d fs gs h ha hb) .nil))
        hleft hright
      simp only [hleft, Bind.bind, Except.bind] at hl
      simp only [hright, Bind.bind, Except.bind] at hr
      let fetchLeft (id : Nat) : Except String (Value left) := match leftValues[id]? with
        | some value => pure value
        | none => throw s!"IR interpreter: missing result node {id}"
      let fetchRight (id : Nat) : Except String (Value right) := match rightValues[id]? with
        | some value => pure value
        | none => throw s!"IR interpreter: missing result node {id}"
      change (do let outputs ← program.outputs.mapM fetchLeft
                 pure (outputs, leftStatus)) = .ok (leftOutputs, leftFlags) at hl
      change (do let outputs ← program.outputs.mapM fetchRight
                 pure (outputs, rightStatus)) = .ok (rightOutputs, rightFlags) at hr
      except_cases hselectLeft : program.outputs.mapM fetchLeft using hl with a =>
        except_cases hselectRight : program.outputs.mapM fetchRight using hr with b =>
          simp only [hselectLeft, Bind.bind, Except.bind, Pure.pure, Except.pure,
            Except.ok.injEq, Prod.mk.injEq] at hl
          simp only [hselectRight, Bind.bind, Except.bind, Pure.pure, Except.pure,
            Except.ok.injEq, Prod.mk.injEq] at hr
          obtain ⟨rfl, rfl⟩ := hl
          obtain ⟨rfl, rfl⟩ := hr
          rw [Array.mapM_eq_foldlM, ← Array.foldlM_toList] at hselectLeft hselectRight
          apply foldlM_rel program.outputs.toList _ _
            (fun a b => List.Forall₂ relation a.toList b.toList) ?_ .nil
            hselectLeft hselectRight
          intro id _ a b c d hab hc hd
          dsimp only [fetchLeft] at hc
          dsimp only [fetchRight] at hd
          cases ha : leftValues[id]? with
          | none => simp [ha, Functor.map, Except.map] at hc
          | some valueLeft =>
              cases hb : rightValues[id]? with
              | none => simp [hb, Functor.map, Except.map] at hd
              | some valueRight =>
                  simp only [ha, Pure.pure, Except.pure, Functor.map, Except.map,
                    Except.ok.injEq] at hc
                  simp only [hb, Pure.pure, Except.pure, Functor.map, Except.map,
                    Except.ok.injEq] at hd
                  subst c d
                  simpa only [Array.toList_push] using
                    List.rel_append hab (.cons (lookup_rel related ha hb) .nil)

end NN.IR.Interpreter
