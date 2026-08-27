/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.SelectiveScan
import Mathlib.Algebra.Ring.Basic

/-!
# Proofs for affine selective scan

The Mamba/S4 scan theorem is an algebra theorem about affine maps.  A sequential recurrent update
and a parallel prefix scan are equivalent because affine transition composition is associative:

$$
(a_2,b_2)\circ(a_1,b_1)=(a_2a_1,a_2b_1+b_2).
$$

The tensor/CUDA implementation is allowed to choose an efficient scan schedule, but the mathematical
contract is this file: prefix summaries denote the same state as the left-to-right recurrence.
-/

@[expose] public section

namespace NN
namespace MLTheory
namespace StateSpace

open _root_.Spec
namespace ScalarAffineTransition

variable {α : Type}

/-- Composing scalar affine transitions agrees with function composition. -/
@[simp] theorem compose_apply [Semiring α] (t₂ t₁ : _root_.Spec.ScalarAffineTransition α)
    (h : α) :
    (_root_.Spec.ScalarAffineTransition.compose t₂ t₁).apply h = t₂.apply (t₁.apply h) := by
  cases t₁
  cases t₂
  simp [_root_.Spec.ScalarAffineTransition.compose, _root_.Spec.ScalarAffineTransition.apply,
    mul_add, mul_assoc, add_assoc]

/-- The identity transition is a left identity for composition. -/
@[simp] theorem compose_id_left [Semiring α] (t : _root_.Spec.ScalarAffineTransition α) :
    _root_.Spec.ScalarAffineTransition.compose _root_.Spec.ScalarAffineTransition.id t = t := by
  cases t
  simp [_root_.Spec.ScalarAffineTransition.compose, _root_.Spec.ScalarAffineTransition.id]

/-- The identity transition is a right identity for composition. -/
@[simp] theorem compose_id_right [Semiring α] (t : _root_.Spec.ScalarAffineTransition α) :
    _root_.Spec.ScalarAffineTransition.compose t _root_.Spec.ScalarAffineTransition.id = t := by
  cases t
  simp [_root_.Spec.ScalarAffineTransition.compose, _root_.Spec.ScalarAffineTransition.id]

/-- Scalar affine transition composition is associative. -/
@[simp] theorem compose_assoc [Semiring α]
    (t₃ t₂ t₁ : _root_.Spec.ScalarAffineTransition α) :
    _root_.Spec.ScalarAffineTransition.compose
        (_root_.Spec.ScalarAffineTransition.compose t₃ t₂) t₁ =
      _root_.Spec.ScalarAffineTransition.compose t₃
        (_root_.Spec.ScalarAffineTransition.compose t₂ t₁) := by
  cases t₁
  cases t₂
  cases t₃
  simp [_root_.Spec.ScalarAffineTransition.compose, mul_add, mul_assoc, add_assoc]

/-- Zero is a fixed point of a homogeneous scalar transition. -/
@[simp] theorem homogeneous_zero_fixed [Semiring α] (a : α) :
    (_root_.Spec.ScalarAffineTransition.apply { a := a, b := 0 } 0) = 0 := by
  simp [_root_.Spec.ScalarAffineTransition.apply]

end ScalarAffineTransition

namespace DiagonalTransition

variable {α : Type}

/-- Applying a diagonal transition is exactly the scalar affine update in each channel. -/
@[simp] theorem apply_getScalar [Add α] [Mul α] {stateDim : Nat}
    (tr : _root_.Spec.DiagonalTransition α stateDim)
    (h : _root_.Spec.Tensor α [stateDim])
    (i : Fin stateDim) :
    _root_.Spec.Tensor.getScalar (tr.apply h) i =
      _root_.Spec.Tensor.getScalar tr.a i * _root_.Spec.Tensor.getScalar h i +
        _root_.Spec.Tensor.getScalar tr.b i := by
  cases tr with
  | mk a b =>
      cases a with
      | dim fa =>
          cases b with
          | dim fb =>
              cases h with
              | dim fh =>
                  cases hfa : fa i with
                  | scalar ai =>
                  cases hfh : fh i with
                  | scalar hi =>
                  cases hfb : fb i with
                  | scalar bi =>
                  change
                    _root_.Spec.Tensor.item
                      (_root_.Spec.get
                        (_root_.Spec.Tensor.addSpec
                          (_root_.Spec.Tensor.mulSpec (Tensor.dim fa) (Tensor.dim fh))
                          (Tensor.dim fb)) i) =
                    (fa i).item * (fh i).item + (fb i).item
                  simp [_root_.Spec.get,
                    _root_.Spec.Tensor.addSpec, _root_.Spec.Tensor.mulSpec,
                    _root_.Spec.Tensor.map2Spec, _root_.Spec.Tensor.item,
                    hfa, hfh, hfb]

/--
Composing diagonal transitions agrees channelwise with composing the corresponding scalar affine
maps.  This is the exact algebraic invariant used by the variable-coefficient selective-scan
kernel: each flattened state lane is an independent affine scan.
-/
@[simp] theorem compose_apply_getScalar [Semiring α] {stateDim : Nat}
    (t₂ t₁ : _root_.Spec.DiagonalTransition α stateDim)
    (h : _root_.Spec.Tensor α [stateDim])
    (i : Fin stateDim) :
    _root_.Spec.Tensor.getScalar ((_root_.Spec.DiagonalTransition.compose t₂ t₁).apply h) i =
      _root_.Spec.Tensor.getScalar (t₂.apply (t₁.apply h)) i := by
  cases t₁ with
  | mk a₁ b₁ =>
      cases t₂ with
      | mk a₂ b₂ =>
          cases a₁ with
          | dim fa₁ =>
              cases b₁ with
              | dim fb₁ =>
                  cases a₂ with
                  | dim fa₂ =>
                      cases b₂ with
                      | dim fb₂ =>
                          cases h with
                          | dim fh =>
                              cases hfa₁ : fa₁ i with
                              | scalar a₁ =>
                              cases hfb₁ : fb₁ i with
                              | scalar b₁ =>
                              cases hfa₂ : fa₂ i with
                              | scalar a₂ =>
                              cases hfb₂ : fb₂ i with
                              | scalar b₂ =>
                              cases hfh : fh i with
                              | scalar hi =>
                              change
                                _root_.Spec.Tensor.item
                                  (_root_.Spec.get
                                    (_root_.Spec.Tensor.addSpec
                                      (_root_.Spec.Tensor.mulSpec
                                        (_root_.Spec.Tensor.mulSpec
                                          (Tensor.dim fa₂) (Tensor.dim fa₁))
                                        (Tensor.dim fh))
                                      (_root_.Spec.Tensor.addSpec
                                        (_root_.Spec.Tensor.mulSpec
                                          (Tensor.dim fa₂) (Tensor.dim fb₁))
                                        (Tensor.dim fb₂))) i) =
                                _root_.Spec.Tensor.item
                                  (_root_.Spec.get
                                    (_root_.Spec.Tensor.addSpec
                                      (_root_.Spec.Tensor.mulSpec
                                        (Tensor.dim fa₂)
                                        (_root_.Spec.Tensor.addSpec
                                          (_root_.Spec.Tensor.mulSpec
                                            (Tensor.dim fa₁) (Tensor.dim fh))
                                          (Tensor.dim fb₁)))
                                      (Tensor.dim fb₂)) i)
                              simp [_root_.Spec.get,
                                _root_.Spec.Tensor.addSpec, _root_.Spec.Tensor.mulSpec,
                                _root_.Spec.Tensor.map2Spec,
                                _root_.Spec.Tensor.item,
                                hfa₁, hfb₁, hfa₂, hfb₂, hfh,
                                mul_add, mul_assoc, add_assoc]

end DiagonalTransition

/-- Seeding a scan with existing output preserves its state evolution and prepends that output. -/
theorem scanArrayFrom_eq {State Input Output : Type}
    (step : State → Input → State × Output) (initial : State)
    (initialOutputs : Array Output) (xs : Array Input) :
    _root_.Spec.scanArrayFrom step initial initialOutputs xs =
      let result := _root_.Spec.scanArray step initial xs
      (result.1, initialOutputs ++ result.2) := by
  unfold _root_.Spec.scanArray _root_.Spec.scanArrayFrom
  rw [← Array.foldl_toList, ← Array.foldl_toList]
  generalize xs.toList = items
  induction items generalizing initial initialOutputs with
  | nil => simp
  | cons input rest ih =>
      simp only [List.foldl_cons]
      rcases step initial input with ⟨nextState, output⟩
      rw [ih nextState (initialOutputs.push output)]
      simp only [show (#[] : Array Output).push output = #[output] from rfl]
      rw [ih nextState #[output]]
      simp

/-- The output buffer carried by a scan does not affect its final state. -/
theorem scanArrayFrom_state_eq_foldl {State Input Output : Type}
    (step : State → Input → State × Output) (initial : State)
    (initialOutputs : Array Output) (xs : Array Input) :
    (_root_.Spec.scanArrayFrom step initial initialOutputs xs).1 =
      xs.foldl (fun state input => (step state input).1) initial := by
  unfold _root_.Spec.scanArrayFrom
  rw [← Array.foldl_toList, ← Array.foldl_toList]
  generalize xs.toList = items
  induction items generalizing initial initialOutputs with
  | nil => rfl
  | cons input rest ih =>
      simp only [List.foldl_cons]
      rcases step initial input with ⟨nextState, output⟩
      exact ih nextState (initialOutputs.push output)

/-- The state component of `scanArray` is the ordinary state-only left fold. -/
theorem scanArray_state_eq_foldl {State Input Output : Type}
    (step : State → Input → State × Output) (initial : State) (xs : Array Input) :
    (_root_.Spec.scanArray step initial xs).1 =
      xs.foldl (fun state input => (step state input).1) initial := by
  exact scanArrayFrom_state_eq_foldl step initial #[] xs

/-- A scan over appended inputs is the prefix scan followed by the state-dependent suffix scan. -/
theorem scanArray_append {State Input Output : Type}
    (step : State → Input → State × Output) (initial : State) (xs ys : Array Input) :
    _root_.Spec.scanArray step initial (xs ++ ys) =
      let first := _root_.Spec.scanArray step initial xs
      let second := _root_.Spec.scanArray step first.1 ys
      (second.1, first.2 ++ second.2) := by
  unfold _root_.Spec.scanArray _root_.Spec.scanArrayFrom
  rw [Array.foldl_append]
  generalize hfirst :
    Array.foldl
      (fun stateAndOutputs input =>
        let (state, outputs) := stateAndOutputs
        let (nextState, output) := step state input
        (nextState, outputs.push output))
      (initial, #[]) xs = first
  rcases first with ⟨firstState, firstOutputs⟩
  exact scanArrayFrom_eq step firstState firstOutputs ys

private theorem Array.take_append_left {α : Type} (xs ys : Array α) :
    (xs ++ ys).take xs.size = xs := by
  apply Array.ext
  · simp
  · intro i h₁ h₂
    simp

/-- Appending future inputs cannot change outputs already emitted by a stateful scan. -/
theorem scanArray_append_outputs_take {State Input Output : Type}
    (step : State → Input → State × Output) (initial : State) (xs ys : Array Input) :
    (_root_.Spec.scanArray step initial (xs ++ ys)).2.take xs.size =
      (_root_.Spec.scanArray step initial xs).2 := by
  rw [show (_root_.Spec.scanArray step initial (xs ++ ys)).2 =
      (_root_.Spec.scanArray step initial xs).2 ++
        (_root_.Spec.scanArray step
          (_root_.Spec.scanArray step initial xs).1 ys).2 by
    simpa using congrArg Prod.snd (scanArray_append step initial xs ys)]
  rw [← _root_.Spec.scanArray_outputs_size step initial xs]
  exact Array.take_append_left _ _

/-- A stateful scan emits exactly one value for every input. -/
@[simp] theorem scanArray_outputs_size {State Input Output : Type}
    (step : State → Input → State × Output) (initial : State) (xs : Array Input) :
    (_root_.Spec.scanArray step initial xs).2.size = xs.size :=
  _root_.Spec.scanArray_outputs_size step initial xs

/-- Running appended scalar transitions factors through the state reached after the prefix. -/
theorem runScalarAffine_append {α : Type} [Mul α] [Add α] (h0 : α)
    (xs ys : Array (_root_.Spec.ScalarAffineTransition α)) :
    _root_.Spec.runScalarAffine h0 (xs ++ ys) =
      _root_.Spec.runScalarAffine (_root_.Spec.runScalarAffine h0 xs) ys := by
  simpa [_root_.Spec.runScalarAffine] using
    congrArg Prod.fst (scanArray_append
      (fun state transition =>
        let nextState := transition.apply state
        (nextState, nextState)) h0 xs ys)

/-- Running one scalar transition is the same as applying it. -/
@[simp] theorem runScalarAffine_singleton {α : Type} [Mul α] [Add α] (h0 : α)
    (tr : _root_.Spec.ScalarAffineTransition α) :
    _root_.Spec.runScalarAffine h0 #[tr] = tr.apply h0 := by
  rfl

/-- The affine summary denotes the same state as the sequential recurrence. -/
theorem summarizeScalarAffine_apply_eq_run {α : Type} [Semiring α] (h0 : α)
    (transitions : Array (_root_.Spec.ScalarAffineTransition α)) :
    (_root_.Spec.summarizeScalarAffine transitions).apply h0 =
      _root_.Spec.runScalarAffine h0 transitions := by
  unfold _root_.Spec.runScalarAffine
  rw [scanArray_state_eq_foldl]
  unfold _root_.Spec.summarizeScalarAffine
  rw [← Array.foldr_toList, ← Array.foldl_toList]
  generalize transitions.toList = items
  induction items generalizing h0 with
  | nil =>
      simp [_root_.Spec.ScalarAffineTransition.id,
        _root_.Spec.ScalarAffineTransition.apply]
  | cons transition rest ih =>
      simp only [List.foldr_cons, List.foldl_cons]
      rw [ScalarAffineTransition.compose_apply, ih]

private theorem foldr_compose_summary {α : Type} [Semiring α]
    (initial : _root_.Spec.ScalarAffineTransition α)
    (transitions : Array (_root_.Spec.ScalarAffineTransition α)) :
    transitions.foldr
        (fun transition summary =>
          _root_.Spec.ScalarAffineTransition.compose summary transition) initial =
      _root_.Spec.ScalarAffineTransition.compose initial
        (_root_.Spec.summarizeScalarAffine transitions) := by
  unfold _root_.Spec.summarizeScalarAffine
  rw [← Array.foldr_toList, ← Array.foldr_toList]
  generalize transitions.toList = items
  induction items with
  | nil => simp
  | cons transition rest ih =>
      simp only [List.foldr_cons]
      rw [ih]
      exact ScalarAffineTransition.compose_assoc initial
        (List.foldr
          (fun transition summary =>
            _root_.Spec.ScalarAffineTransition.compose summary transition)
          _root_.Spec.ScalarAffineTransition.id rest) transition

/-- Prefix summaries compose across array append in execution order. -/
theorem summarizeScalarAffine_append {α : Type} [Semiring α]
    (xs ys : Array (_root_.Spec.ScalarAffineTransition α)) :
    _root_.Spec.summarizeScalarAffine (xs ++ ys) =
      _root_.Spec.ScalarAffineTransition.compose
        (_root_.Spec.summarizeScalarAffine ys)
        (_root_.Spec.summarizeScalarAffine xs) := by
  unfold _root_.Spec.summarizeScalarAffine
  rw [Array.foldr_append]
  exact foldr_compose_summary
    (Array.foldr
      (fun transition summary =>
        _root_.Spec.ScalarAffineTransition.compose summary transition)
      _root_.Spec.ScalarAffineTransition.id ys) xs

/-- Prefix summaries composed across append have the expected denotation. -/
theorem summarizeScalarAffine_append_apply {α : Type} [Semiring α] (h0 : α)
    (xs ys : Array (_root_.Spec.ScalarAffineTransition α)) :
    (_root_.Spec.summarizeScalarAffine (xs ++ ys)).apply h0 =
      (_root_.Spec.ScalarAffineTransition.compose
        (_root_.Spec.summarizeScalarAffine ys)
        (_root_.Spec.summarizeScalarAffine xs)).apply h0 := by
  rw [summarizeScalarAffine_append]

/-- The scalar affine scan has one state per transition. -/
@[simp] theorem scalarAffineScan_size {α : Type} [Mul α] [Add α] (h0 : α)
    (transitions : Array (_root_.Spec.ScalarAffineTransition α)) :
    (_root_.Spec.scalarAffineScan h0 transitions).size = transitions.size := by
  exact scanArray_outputs_size _ h0 transitions

/-- Scanning appended scalar transitions is the prefix scan followed by the suffix scan. -/
theorem scalarAffineScan_append {α : Type} [Mul α] [Add α] (h0 : α)
    (xs ys : Array (_root_.Spec.ScalarAffineTransition α)) :
    _root_.Spec.scalarAffineScan h0 (xs ++ ys) =
      _root_.Spec.scalarAffineScan h0 xs ++
        _root_.Spec.scalarAffineScan (_root_.Spec.runScalarAffine h0 xs) ys := by
  simpa [_root_.Spec.scalarAffineScan, _root_.Spec.runScalarAffine] using
    congrArg Prod.snd (scanArray_append
      (fun state transition =>
        let nextState := transition.apply state
        (nextState, nextState)) h0 xs ys)

/-- The diagonal tensor scan has one state per transition. -/
@[simp] theorem diagonalSelectiveScan_size {α : Type} [Add α] [Mul α] {stateDim : Nat}
    (h0 : _root_.Spec.Tensor α [stateDim])
    (transitions : Array (_root_.Spec.DiagonalTransition α stateDim)) :
    (_root_.Spec.diagonalSelectiveScan h0 transitions).size = transitions.size := by
  exact scanArray_outputs_size _ h0 transitions

/-- Running appended diagonal transitions factors through the state after the prefix. -/
theorem runDiagonalTransitions_append {α : Type} [Add α] [Mul α] {stateDim : Nat}
    (h0 : _root_.Spec.Tensor α [stateDim])
    (xs ys : Array (_root_.Spec.DiagonalTransition α stateDim)) :
    _root_.Spec.runDiagonalTransitions h0 (xs ++ ys) =
      _root_.Spec.runDiagonalTransitions (_root_.Spec.runDiagonalTransitions h0 xs) ys := by
  simpa [_root_.Spec.runDiagonalTransitions] using
    congrArg Prod.fst (scanArray_append
      (fun state transition =>
        let nextState := transition.apply state
        (nextState, nextState)) h0 xs ys)

/-- The diagonal scan of an append is the prefix scan followed by the state-dependent suffix scan. -/
theorem diagonalSelectiveScan_append {α : Type} [Add α] [Mul α] {stateDim : Nat}
    (h0 : _root_.Spec.Tensor α [stateDim])
    (xs ys : Array (_root_.Spec.DiagonalTransition α stateDim)) :
    _root_.Spec.diagonalSelectiveScan h0 (xs ++ ys) =
      _root_.Spec.diagonalSelectiveScan h0 xs ++
        _root_.Spec.diagonalSelectiveScan (_root_.Spec.runDiagonalTransitions h0 xs) ys := by
  simpa [_root_.Spec.diagonalSelectiveScan, _root_.Spec.runDiagonalTransitions] using
    congrArg Prod.snd (scanArray_append
      (fun state transition =>
        let nextState := transition.apply state
        (nextState, nextState)) h0 xs ys)

/--
A homogeneous affine transition over $\mathbb{R}$ is Lipschitz with factor $\rho$ whenever
$|a|\leq\rho$.

This is the one-channel stability lemma used to lift diagonal SSMs into contraction proofs.
-/
theorem abs_homogeneous_apply_le (a ρ h : ℝ) (ha : |a| ≤ ρ) :
    |(_root_.Spec.ScalarAffineTransition.apply { a := a, b := 0 } h)| ≤ ρ * |h| := by
  calc
    |(_root_.Spec.ScalarAffineTransition.apply { a := a, b := 0 } h)|
        = |a * h| := by simp [_root_.Spec.ScalarAffineTransition.apply]
    _ = |a| * |h| := by rw [abs_mul]
    _ ≤ ρ * |h| := mul_le_mul_of_nonneg_right ha (abs_nonneg h)

/-- A homogeneous scalar transition with $|a|\leq 1$ is non-expansive. -/
theorem abs_homogeneous_apply_le_self (a h : ℝ) (ha : |a| ≤ 1) :
    |(_root_.Spec.ScalarAffineTransition.apply { a := a, b := 0 } h)| ≤ |h| := by
  calc
    |(_root_.Spec.ScalarAffineTransition.apply { a := a, b := 0 } h)| ≤ 1 * |h| :=
      abs_homogeneous_apply_le a 1 h ha
    _ = |h| := by simp

end StateSpace
end MLTheory
end NN
