/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Optimization.OptimizerLaws

/-!
# First-Order Optimizer Relations

Relations between distinct first-order optimizer updates.

This is the tensor-facing layer: the statements are phrased over `Spec.Tensor` and the executable
operator dictionary `Spec.Context`. When we want ordinary algebraic simplification, such as proving
that a zero weight-decay AdamW update is the same parameter update as Adam, we specialize to `ℝ`,
where mathlib provides the ring laws.
-/

@[expose] public section


namespace Optim
open Spec
open Tensor

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

namespace AdamW

/-!
`Context α` gives us executable operators, but not algebraic laws like `x * 0 = 0`.

Optimizer-relationship theorems (AdamW → Adam, L2 vs weight decay, etc.) therefore live most
naturally over proof backends like `ℝ` where the laws are available from Mathlib.
-/

/--
AdamW reduces to Adam when `weightDecay = 0` (parameter-update equality), over `ℝ`.
-/
theorem update_weight_decay_zero_params_eq_adam_real {s : Shape}
    (state : State ℝ s) (params grads : Tensor ℝ s) (hwd : state.weightDecay = 0) :
    (update state params grads).2 =
      (Adam.update
          ({ lr := state.lr, beta1 := state.beta1, beta2 := state.beta2, epsilon := state.epsilon,
             m := state.m, v := state.v, t := state.t } : Adam.State ℝ s)
          params grads).2 := by
  -- `Context` gives operators; to simplify the decoupled decay term we use `ℝ`'s algebraic laws
  -- together with structural recursion on TorchLean spec tensors.
  --
  -- Helper lemmas: scaling by `0` yields the all-zero tensor, and subtracting that yields identity.
  have scaleSpec_zero : ∀ {s : Shape} (t : Tensor ℝ s), scaleSpec t (0 : ℝ) = fill 0 s := by
    intro s t
    induction t with
    | scalar x =>
        simp [Tensor.scaleSpec, Tensor.mapSpec, Spec.fill]
    | dim g ih =>
        -- `fill` and `mapSpec` are definitional: function ext + IH is enough.
        apply congrArg Tensor.dim
        funext i
        simpa [Tensor.scaleSpec, Tensor.mapSpec, Spec.fill] using ih i
  have subSpec_fill_zero : ∀ {s : Shape} (t : Tensor ℝ s), subSpec t (fill 0 s) = t := by
    intro s t
    induction t with
    | scalar x =>
        simp [Tensor.subSpec, Tensor.map2Spec, Spec.fill]
    | dim g ih =>
        apply congrArg Tensor.dim
        funext i
        simpa [Tensor.subSpec, Tensor.map2Spec, Spec.fill] using ih i
  cases state with
  | mk lr beta1 beta2 epsilon weightDecay m v t =>
      -- Now `weightDecay = 0` makes the decayed-parameter term a no-op.
      -- We rewrite `weightDecay` first, then discharge the tensor equalities via the helpers.
      have hwd' : weightDecay = 0 := by simpa using hwd
      -- `simp` doesn't know tensor algebra, but it will reduce the remaining goal to our helpers.
      simp [AdamW.update, Adam.update, hwd', scaleSpec_zero, subSpec_fill_zero]

end AdamW

end Optim
