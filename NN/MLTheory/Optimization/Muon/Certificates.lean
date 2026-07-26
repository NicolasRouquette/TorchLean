/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Optimization.Muon.NewtonSchulz

/-!
# Muon Step Certificates

This module connects a backend's orthogonalization contract to the direction, state, and parameter
values produced by one executable Muon update. The generic checked-backend theorems are the public
proof interface; concrete QR and Newton-Schulz backends instantiate them in their own modules.
-/

@[expose] public section

namespace Optim

open Spec
open Tensor

namespace Muon

variable {α : Type} [Context α]

/--
Evidence that `direction` is exactly the output of an orthogonalizer on `buffer` and has column
Gram matrix $I$.
-/
structure ExactCertifiedDirection {m n : Nat}
    (orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar)))
    (buffer direction : MatrixTensor α m n) : Prop where
  /-- The certified direction is the backend output. -/
  direction_eq : direction = orthogonalizer.apply buffer
  /-- The certified direction has exact column Gram $I$. -/
  exact_column_gram : HasExactColumnGram direction

/--
Evidence that `direction` is exactly the output of an orthogonalizer on `buffer` and has an
entrywise column-Gram residual bounded by $\varepsilon$.
-/
structure ApproxCertifiedDirection {m n : Nat} (eps : α)
    (orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar)))
    (buffer direction : MatrixTensor α m n) : Prop where
  /-- The certified direction is the backend output. -/
  direction_eq : direction = orthogonalizer.apply buffer
  /-- The certified direction satisfies the requested residual bound. -/
  approx_column_gram : HasApproxColumnGram eps direction

/--
Certificate for one exact Muon update: the fresh momentum buffer is orthogonalized, the new state
stores that buffer, and the parameters move along the certified direction.
-/
structure ExactCertifiedStep {m n : Nat}
    (state : State α (.dim m (.dim n .scalar)))
    (params grads direction : MatrixTensor α m n) : Prop where
  /-- Certificate for the direction computed from the fresh momentum buffer. -/
  direction_cert :
    ExactCertifiedDirection state.orthogonalizer (update state params grads).1.buf direction
  /-- Muon changes only the momentum buffer in its optimizer state. -/
  state_eq :
    (update state params grads).1 =
      { state with buf := OptimizerUtils.updateMomentumBuf state.buf state.momentum grads }
  /-- Parameter equation for the certified update direction. -/
  params_eq :
    (update state params grads).2 = subSpec params (scaleSpec direction state.lr)

/-- The residual-bounded counterpart of `ExactCertifiedStep`. -/
structure ApproxCertifiedStep {m n : Nat} (eps : α)
    (state : State α (.dim m (.dim n .scalar)))
    (params grads direction : MatrixTensor α m n) : Prop where
  /-- Certificate for the direction computed from the fresh momentum buffer. -/
  direction_cert :
    ApproxCertifiedDirection eps state.orthogonalizer (update state params grads).1.buf direction
  /-- Muon changes only the momentum buffer in its optimizer state. -/
  state_eq :
    (update state params grads).1 =
      { state with buf := OptimizerUtils.updateMomentumBuf state.buf state.momentum grads }
  /-- Parameter equation for the certified update direction. -/
  params_eq :
    (update state params grads).2 = subSpec params (scaleSpec direction state.lr)

/-- A local exact backend fact for the fresh buffer produces a certified Muon step. -/
theorem exactCertifiedStep_of_buffer {m n : Nat}
    (state : State α (.dim m (.dim n .scalar)))
    (params grads : MatrixTensor α m n)
    (horth : ExactOrthogonalizesBuffer state.orthogonalizer (update state params grads).1.buf) :
    ∃ direction : MatrixTensor α m n, ExactCertifiedStep state params grads direction := by
  let direction := state.orthogonalizer.apply (update state params grads).1.buf
  refine ⟨direction, ⟨⟨rfl, horth⟩, rfl, ?_⟩⟩
  rfl

/-- A local residual bound for the fresh buffer produces a certified Muon step. -/
theorem approxCertifiedStep_of_buffer {m n : Nat} {eps : α}
    (state : State α (.dim m (.dim n .scalar)))
    (params grads : MatrixTensor α m n)
    (horth :
      ApproxOrthogonalizesBuffer eps state.orthogonalizer (update state params grads).1.buf) :
    ∃ direction : MatrixTensor α m n, ApproxCertifiedStep eps state params grads direction := by
  let direction := state.orthogonalizer.apply (update state params grads).1.buf
  refine ⟨direction, ⟨⟨rfl, horth⟩, rfl, ?_⟩⟩
  rfl

/--
A checked exact backend certifies the concrete direction and equations of one Muon update whenever
its success predicate holds on the fresh momentum buffer.
-/
theorem exactCertifiedStep_of_checkedBackend {m n : Nat}
    (backend : CheckedExactOrthogonalizer α m n)
    (lr momentum : α) (buf params grads : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ExactCertifiedStep
        ({ lr := lr, momentum := momentum, buf := buf,
           orthogonalizer := backend.orthogonalizer } :
          State α (.dim m (.dim n .scalar)))
        params grads direction := by
  let state : State α (.dim m (.dim n .scalar)) :=
    { lr := lr, momentum := momentum, buf := buf, orthogonalizer := backend.orthogonalizer }
  exact exactCertifiedStep_of_buffer state params grads
    (backend.certified (update state params grads).1.buf hsuccess)

/--
A checked approximate backend certifies one Muon update whenever its success predicate establishes
the requested Gram-residual bound on the fresh momentum buffer.
-/
theorem approxCertifiedStep_of_checkedBackend {m n : Nat} {eps : α}
    (backend : CheckedApproxOrthogonalizer α m n eps)
    (lr momentum : α) (buf params grads : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedStep eps
        ({ lr := lr, momentum := momentum, buf := buf,
           orthogonalizer := backend.orthogonalizer } :
          State α (.dim m (.dim n .scalar)))
        params grads direction := by
  let state : State α (.dim m (.dim n .scalar)) :=
    { lr := lr, momentum := momentum, buf := buf, orthogonalizer := backend.orthogonalizer }
  exact approxCertifiedStep_of_buffer state params grads
    (backend.certified (update state params grads).1.buf hsuccess)

/-- A checked exact backend gives $Q^\mathsf{T}Q=I$ for the direction used by an update. -/
theorem checkedBackend_updateDirection_hasExactColumnGram {m n : Nat}
    (backend : CheckedExactOrthogonalizer α m n)
    (lr momentum : α) (buf params grads : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    HasExactColumnGram
      (backend.orthogonalizer.apply
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          params grads).1.buf) := by
  exact backend.certified _ hsuccess

/-- A checked approximate backend gives its residual bound for the direction used by an update. -/
theorem checkedBackend_updateDirection_hasApproxColumnGram {m n : Nat} {eps : α}
    (backend : CheckedApproxOrthogonalizer α m n eps)
    (lr momentum : α) (buf params grads : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    HasApproxColumnGram eps
      (backend.orthogonalizer.apply
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          params grads).1.buf) := by
  exact backend.certified _ hsuccess

/-- Extract exact column orthogonality from a certified step. -/
theorem ExactCertifiedStep.hasExactColumnGram {m n : Nat}
    {state : State α (.dim m (.dim n .scalar))}
    {params grads direction : MatrixTensor α m n}
    (cert : ExactCertifiedStep state params grads direction) :
    HasExactColumnGram direction :=
  cert.direction_cert.exact_column_gram

/-- Extract the Gram-residual bound from an approximate certified step. -/
theorem ApproxCertifiedStep.hasApproxColumnGram {m n : Nat} {eps : α}
    {state : State α (.dim m (.dim n .scalar))}
    {params grads direction : MatrixTensor α m n}
    (cert : ApproxCertifiedStep eps state params grads direction) :
    HasApproxColumnGram eps direction :=
  cert.direction_cert.approx_column_gram

end Muon

end Optim
