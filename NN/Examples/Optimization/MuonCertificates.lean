/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Optimization.Muon

/-!
# Muon Certificate Examples

These examples show how to obtain and consume the exact and approximate certificates attached to
Muon steps. Runtime code configures Muon through `TorchLean.optim.runtimeMuon`; proofs about the
algorithm use the canonical `Optim.Muon` namespace.
-/

@[expose] public section

namespace NN.Examples.Optimization.MuonCertificates

/--
Using the QR checked backend, a positive-pivot proof gives a certified step; from that step we can
recover both the exact Gram certificate for the direction and the parameter-update equation.
-/
theorem qr_update_step_direction_has_exact_gram {m n : Nat}
    (lr momentum : ℝ) (buf params grads : Optim.Muon.MatrixTensor ℝ m n)
    (hpivots :
      Optim.Muon.HasPositiveQRPivots
        (Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := Optim.Muon.qrOrthogonalizer (m := m) (n := n) } :
            Optim.Muon.State ℝ (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    ∃ direction : Optim.Muon.MatrixTensor ℝ m n,
      Optim.Muon.HasExactColumnGram direction ∧
      (Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := Optim.Muon.qrOrthogonalizer (m := m) (n := n) } :
            Optim.Muon.State ℝ (.dim m (.dim n .scalar)))
          params grads).2 =
        _root_.Spec.Tensor.subSpec params (_root_.Spec.Tensor.scaleSpec direction lr) := by
  obtain ⟨direction, hstep⟩ :=
    Optim.Muon.update_has_exact_certified_step_qr lr momentum buf params grads hpivots
  exact ⟨direction,
    hstep.hasExactColumnGram,
    hstep.params_eq⟩

/--
Using the residual-checked Newton-Schulz backend, a residual proof gives a certified approximate
step; from that step we can recover the residual-bound certificate and the parameter equation.
-/
theorem newtonSchulz_update_step_direction_has_approx_gram
    {α : Type} [Context α] {m n : Nat} {eps : α}
    (coeffs : Optim.Muon.NewtonSchulzCoeffs α) (steps : Nat)
    (lr momentum : α) (buf params grads : Optim.Muon.MatrixTensor α m n)
    (hresidual :
      Optim.Muon.ResidualApproxSuccess eps
        (Optim.Muon.newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
        (Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              Optim.Muon.newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            Optim.Muon.State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    ∃ direction : Optim.Muon.MatrixTensor α m n,
      Optim.Muon.HasApproxColumnGram eps direction ∧
      (Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              Optim.Muon.newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            Optim.Muon.State α (.dim m (.dim n .scalar)))
          params grads).2 =
        _root_.Spec.Tensor.subSpec params (_root_.Spec.Tensor.scaleSpec direction lr) := by
  obtain ⟨direction, hstep⟩ :=
    Optim.Muon.approxCertifiedStep_of_checkedBackend
      (backend := Optim.Muon.newtonSchulzResidualCheckedOrthogonalizer
        (α := α) (m := m) (n := n) coeffs steps eps)
      lr momentum buf params grads hresidual
  exact ⟨direction,
    hstep.hasApproxColumnGram,
    hstep.params_eq⟩

/--
If the fresh momentum buffer is already exact-column-orthogonal and fixed by one Newton-Schulz
step, the fixed-point checked backend upgrades Newton-Schulz from an approximate residual-checked
path to an exact certified step.
-/
theorem newtonSchulz_fixed_update_step_direction_has_exact_gram
    {α : Type} [Context α] {m n : Nat}
    (coeffs : Optim.Muon.NewtonSchulzCoeffs α) (steps : Nat)
    (lr momentum : α) (buf params grads : Optim.Muon.MatrixTensor α m n)
    (hsuccess :
      (Optim.Muon.newtonSchulzFixedPointCheckedExactOrthogonalizer
        (α := α) (m := m) (n := n) coeffs steps).Success
        (Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              Optim.Muon.newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            Optim.Muon.State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    ∃ direction : Optim.Muon.MatrixTensor α m n,
      Optim.Muon.HasExactColumnGram direction ∧
      (Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              Optim.Muon.newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            Optim.Muon.State α (.dim m (.dim n .scalar)))
          params grads).2 =
        _root_.Spec.Tensor.subSpec params (_root_.Spec.Tensor.scaleSpec direction lr) := by
  obtain ⟨direction, hstep⟩ :=
    Optim.Muon.exactCertifiedStep_of_checkedBackend
      (backend := Optim.Muon.newtonSchulzFixedPointCheckedExactOrthogonalizer
        (α := α) (m := m) (n := n) coeffs steps)
      lr momentum buf params grads hsuccess
  exact ⟨direction,
    hstep.hasExactColumnGram,
    hstep.params_eq⟩

/--
At an exact-column-orthogonal matrix, one real Newton-Schulz step acts by the scalar $a+b+c$.
This is the algebra behind the fixed-point shortcut below.
-/
theorem newtonSchulz_step_scales_exact_gram_matrix {m n : Nat}
    (coeffs : Optim.Muon.NewtonSchulzCoeffs ℝ)
    (Q : Optim.Muon.MatrixTensor ℝ m n)
    (hgram : Optim.Muon.HasExactColumnGram Q) :
    Optim.Muon.newtonSchulzStep coeffs Q =
      _root_.Spec.Tensor.scaleSpec Q (coeffs.a + coeffs.b + coeffs.c) :=
  Optim.Muon.newtonSchulzStep_eq_scale_sum_of_exact_column_gram coeffs Q hgram

/--
The same algebra shows exact column Gram is preserved when the coefficient sum squares to one,
including the sign-flip case.
-/
theorem newtonSchulz_step_preserves_exact_gram_when_sum_squares_one {m n : Nat}
    (coeffs : Optim.Muon.NewtonSchulzCoeffs ℝ)
    (Q : Optim.Muon.MatrixTensor ℝ m n)
    (hgram : Optim.Muon.HasExactColumnGram Q)
    (hsquare : (coeffs.a + coeffs.b + coeffs.c) *
        (coeffs.a + coeffs.b + coeffs.c) = 1) :
    Optim.Muon.HasExactColumnGram (Optim.Muon.newtonSchulzStep coeffs Q) :=
  Optim.Muon.newtonSchulzStep_hasExactColumnGram_of_exact_column_gram_of_sum_square_one
    coeffs Q hgram hsquare

/--
If the coefficient sum is only approximately square-one, the same one-step algebra gives the
entrywise residual bound used by the approximate Muon certificate.
-/
theorem newtonSchulz_step_has_approx_gram_from_coeff_square_error {m n : Nat}
    (coeffs : Optim.Muon.NewtonSchulzCoeffs ℝ)
    (Q : Optim.Muon.MatrixTensor ℝ m n) (eps : ℝ)
    (hgram : Optim.Muon.HasExactColumnGram Q)
    (herr :
      MathFunctions.abs
        ((coeffs.a + coeffs.b + coeffs.c) * (coeffs.a + coeffs.b + coeffs.c) - 1) ≤ eps)
    (heps : 0 ≤ eps) :
    Optim.Muon.HasApproxColumnGram eps (Optim.Muon.newtonSchulzStep coeffs Q) :=
  Optim.Muon.newtonSchulzStep_hasApproxColumnGram_of_exact_column_gram_of_sum_square_error
    coeffs Q eps hgram herr heps

/--
Over `ℝ`, the common coefficient condition $a+b+c=1$ means an already exact-column-orthogonal
fresh buffer is automatically a fixed point, so the Newton-Schulz Muon update is exactly certified.
-/
theorem newtonSchulz_coeff_sum_update_step_direction_has_exact_gram {m n : Nat}
    (coeffs : Optim.Muon.NewtonSchulzCoeffs ℝ) (steps : Nat)
    (lr momentum : ℝ) (buf params grads : Optim.Muon.MatrixTensor ℝ m n)
    (hgram :
      Optim.Muon.HasExactColumnGram
        (Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              Optim.Muon.newtonSchulzOrthogonalizer (α := ℝ) (m := m) (n := n) coeffs steps } :
            Optim.Muon.State ℝ (.dim m (.dim n .scalar)))
          params grads).1.buf)
    (hsum : coeffs.a + coeffs.b + coeffs.c = 1) :
    ∃ direction : Optim.Muon.MatrixTensor ℝ m n,
      Optim.Muon.HasExactColumnGram direction ∧
      (Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              Optim.Muon.newtonSchulzOrthogonalizer (α := ℝ) (m := m) (n := n) coeffs steps } :
            Optim.Muon.State ℝ (.dim m (.dim n .scalar)))
          params grads).2 =
        _root_.Spec.Tensor.subSpec params (_root_.Spec.Tensor.scaleSpec direction lr) := by
  obtain ⟨direction, hstep⟩ :=
    Optim.Muon.update_has_exact_certified_step_newtonSchulz_exact_gram_checked
      coeffs steps lr momentum buf params grads hgram hsum
  exact ⟨direction,
    hstep.hasExactColumnGram,
    hstep.params_eq⟩

/--
The initialized QR theorem has the same proof shape as the stateful update theorem, but starts from
the state produced by `Optim.Muon.init`.
-/
theorem qr_initialized_step_direction_has_exact_gram {m n : Nat}
    (lr momentum : ℝ) (params grads : Optim.Muon.MatrixTensor ℝ m n)
    (hpivots :
      Optim.Muon.HasPositiveQRPivots
        (Optim.Muon.update
          (Optim.Muon.init lr momentum
            (Optim.Muon.qrOrthogonalizer (m := m) (n := n)) params)
          params grads).1.buf) :
    ∃ direction : Optim.Muon.MatrixTensor ℝ m n,
      Optim.Muon.HasExactColumnGram direction ∧
      (Optim.Muon.update
          (Optim.Muon.init lr momentum
            (Optim.Muon.qrOrthogonalizer (m := m) (n := n)) params)
          params grads).2 =
        _root_.Spec.Tensor.subSpec params (_root_.Spec.Tensor.scaleSpec direction lr) := by
  obtain ⟨direction, hstep⟩ :=
    Optim.Muon.init_has_exact_certified_step_qr lr momentum params grads hpivots
  exact ⟨direction,
    hstep.hasExactColumnGram,
    hstep.params_eq⟩

/--
The same QR certificate also exposes the whole next-state equation, not only the direction
certificate.
-/
theorem qr_initialized_step_state_eq {m n : Nat}
    (lr momentum : ℝ) (params grads : Optim.Muon.MatrixTensor ℝ m n)
    (hpivots :
      Optim.Muon.HasPositiveQRPivots
        (Optim.Muon.update
          (Optim.Muon.init lr momentum
            (Optim.Muon.qrOrthogonalizer (m := m) (n := n)) params)
          params grads).1.buf) :
    (Optim.Muon.update
        (Optim.Muon.init lr momentum
          (Optim.Muon.qrOrthogonalizer (m := m) (n := n)) params)
        params grads).1 =
      { Optim.Muon.init lr momentum
          (Optim.Muon.qrOrthogonalizer (m := m) (n := n)) params with
        buf :=
          _root_.Optim.OptimizerUtils.updateMomentumBuf
            (Optim.Muon.init lr momentum
              (Optim.Muon.qrOrthogonalizer (m := m) (n := n)) params).buf
            (Optim.Muon.init lr momentum
              (Optim.Muon.qrOrthogonalizer (m := m) (n := n)) params).momentum
            grads } := by
  obtain ⟨_direction, hstep⟩ :=
    Optim.Muon.init_has_exact_certified_step_qr lr momentum params grads hpivots
  exact hstep.state_eq

/--
The initialized residual-checked Newton-Schulz theorem is the training-path version: initialize the
Muon state, run one update, then consume the certified step.
-/
theorem newtonSchulz_initialized_step_direction_has_approx_gram
    {α : Type} [Context α] {m n : Nat} {eps : α}
    (coeffs : Optim.Muon.NewtonSchulzCoeffs α) (steps : Nat)
    (lr momentum : α) (params grads : Optim.Muon.MatrixTensor α m n)
    (hresidual :
      Optim.Muon.ResidualApproxSuccess eps
        (Optim.Muon.newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
        (Optim.Muon.update
          (Optim.Muon.init lr momentum
            (Optim.Muon.newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
            params)
          params grads).1.buf) :
    ∃ direction : Optim.Muon.MatrixTensor α m n,
      Optim.Muon.HasApproxColumnGram eps direction ∧
      (Optim.Muon.update
          (Optim.Muon.init lr momentum
            (Optim.Muon.newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
            params)
          params grads).2 =
        _root_.Spec.Tensor.subSpec params (_root_.Spec.Tensor.scaleSpec direction lr) := by
  obtain ⟨direction, hstep⟩ :=
    Optim.Muon.approxCertifiedStep_of_checkedBackend
      (backend := Optim.Muon.newtonSchulzResidualCheckedOrthogonalizer
        (α := α) (m := m) (n := n) coeffs steps eps)
      (lr := lr) (momentum := momentum)
      (buf := _root_.Spec.fill 0 (.dim m (.dim n .scalar)))
      (params := params) (grads := grads) hresidual
  exact ⟨direction,
    hstep.hasApproxColumnGram,
    hstep.params_eq⟩

/--
The initialized residual-checked Newton-Schulz certificate also exposes the whole next-state
equation.
-/
theorem newtonSchulz_initialized_step_state_eq
    {α : Type} [Context α] {m n : Nat} {eps : α}
    (coeffs : Optim.Muon.NewtonSchulzCoeffs α) (steps : Nat)
    (lr momentum : α) (params grads : Optim.Muon.MatrixTensor α m n)
    (hresidual :
      Optim.Muon.ResidualApproxSuccess eps
        (Optim.Muon.newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
        (Optim.Muon.update
          (Optim.Muon.init lr momentum
            (Optim.Muon.newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
            params)
          params grads).1.buf) :
    (Optim.Muon.update
        (Optim.Muon.init lr momentum
          (Optim.Muon.newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
          params)
        params grads).1 =
      { Optim.Muon.init lr momentum
          (Optim.Muon.newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
          params with
        buf :=
          _root_.Optim.OptimizerUtils.updateMomentumBuf
            (Optim.Muon.init lr momentum
              (Optim.Muon.newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
              params).buf
            (Optim.Muon.init lr momentum
              (Optim.Muon.newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
              params).momentum
            grads } := by
  obtain ⟨_direction, hstep⟩ :=
    Optim.Muon.approxCertifiedStep_of_checkedBackend
      (backend := Optim.Muon.newtonSchulzResidualCheckedOrthogonalizer
        (α := α) (m := m) (n := n) coeffs steps eps)
      (lr := lr) (momentum := momentum)
      (buf := _root_.Spec.fill 0 (.dim m (.dim n .scalar)))
      (params := params) (grads := grads) hresidual
  exact hstep.state_eq

end NN.Examples.Optimization.MuonCertificates
