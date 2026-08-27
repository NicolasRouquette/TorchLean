/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec
public import NN.Runtime.Autograd.TorchLean.Functional
public import NN.Tensor

/-!
# TwoStage Core

Shared core for the **TwoStage neural-controller / neural-Lyapunov workflows**.

This directory (`NN/MLTheory/CROWN/Lyapunov/TwoStage/`) is the Lean counterpart of the “three
pipeline” workflow in the TorchLean paper (`arXiv:2602.22631`, Figure 7):

- (i) **Python-only**: PyTorch + α/β-CROWN produce numeric bounds; a Lean result is conditional on
  a proof that the imported certificate is valid for the stated model and region.
- (ii) **Hybrid**: Stage-1 training in PyTorch, exported as *float32 bit patterns*; Stage-2
  refinement + the final IBP/CROWN check run inside TorchLean under exact `IEEE32Exec` semantics.
- (iii) **All-in-Lean**: both stages run inside TorchLean under `IEEE32Exec`; the final IBP/CROWN
  check is also in Lean.

This file is shared by (ii) and (iii). It contains:
- the shapes / parameter pack layout for a small controller and a 1-hidden-layer Lyapunov net, and
- the scalar TorchLean `lossProgram` used for both training and verification lowering.

Key point: the *same* TorchLean program is used in two roles:
- **execution/training** (with `α = IEEE32Exec`), and
- **lowering** to the op-tagged verifier IR used by in-repo IBP/CROWN bound propagation.
-/

@[expose] public section


open _root_.Spec
open _root_.Spec.Tensor
open Runtime
open Runtime.Autograd

namespace NN.MLTheory.CROWN.Lyapunov.TwoStage.Core

/-- State dimension for the 2D Lyapunov example. -/
abbrev xDim : Nat := 2
/-- Control dimension for the 2D Lyapunov example. -/
abbrev uDim : Nat := 1

/-- Tensor shape for the state vector `x`. -/
def xShape : Shape := .dim xDim .scalar
/-- Tensor shape for the control vector `u`. -/
def uShape : Shape := .dim uDim .scalar

/-- Parameter shapes for the two-stage controller and Lyapunov network, as a flat list. -/
def paramShapes (width : Nat) : List Shape :=
  [ .dim uDim (.dim xDim .scalar)      -- Wc
  , .dim uDim .scalar                  -- bc
  , .dim width (.dim xDim .scalar)     -- W1
  , .dim width .scalar                 -- b1
  , .dim 1 (.dim width .scalar)        -- W2
  , .dim 1 .scalar                     -- b2
  ]

/-!
Loss program:

$$
\begin{aligned}
\operatorname{penalty}_{\mathrm{pos}}
  &= \operatorname{ReLU}\!\left(c_V\lVert x\rVert^2-V(x)\right),\\
\operatorname{penalty}_{\mathrm{dec}}
  &= \operatorname{ReLU}\!\left(\dot V(x)+c_DV(x)\right),\\
\operatorname{loss}
  &= \operatorname{penalty}_{\mathrm{pos}}+\operatorname{penalty}_{\mathrm{dec}}.
\end{aligned}
$$

Where:

$$
\begin{aligned}
u(x) &= \mathrm{scale}_U\tanh(W_cx+b_c),\\
s(x) &= w_2\mathbin{\cdot}\tanh(W_1x+b_1)+b_2,\\
V(x) &= s(x)^2,\\
\dot V(x) &= \nabla V(x)\mathbin{\cdot}f(x,u(x)).
\end{aligned}
$$

The last line uses the van der Pol-like dynamics. We compute $\nabla V$ analytically for the
one-hidden-layer tanh network followed by a square, so training needs only first-order AD.
-/

/-- Evaluate the controller and return its scalar control output. -/
@[noinline, nospecialize]
def controllerOutput
    {β : Type} [Context β] [DecidableEq Shape]
    {m : Type → Type} [Monad m]
    [_root_.Runtime.Autograd.Torch.Ops (m := m) (α := β)]
    (wC : TorchLean.RefTy (m := m) (α := β) [uDim, xDim])
    (bC : TorchLean.RefTy (m := m) (α := β) [uDim])
    (x : TorchLean.RefTy (m := m) (α := β) xShape)
    (scaleU : β) : m (TorchLean.RefTy (m := m) (α := β) Shape.scalar) := do
  let uPre ← _root_.Runtime.Autograd.Torch.linear
    (m := m) (α := β) (inDim := xDim) (outDim := uDim) wC bC x
  let uT ← TorchLean.tanh (m := m) (α := β) (s := uShape) uPre
  let uVec ← TorchLean.scale (m := m) (α := β) (s := uShape) uT (c := scaleU)
  TorchLean.select (m := m) (α := β) (s := [uDim]) 0 uVec ⟨0, by decide⟩

/-- Evaluate the Lyapunov network together with its analytic gradient in the state coordinates. -/
@[noinline, nospecialize]
def lyapunovValueAndGradient
    {β : Type} [Context β] [DecidableEq Shape]
    {m : Type → Type} [Monad m]
    [_root_.Runtime.Autograd.Torch.Ops (m := m) (α := β)]
    (width : Nat)
    (w1 : TorchLean.RefTy (m := m) (α := β) [width, xDim])
    (b1 : TorchLean.RefTy (m := m) (α := β) [width])
    (w2 : TorchLean.RefTy (m := m) (α := β) [1, width])
    (b2 : TorchLean.RefTy (m := m) (α := β) [1])
    (x : TorchLean.RefTy (m := m) (α := β) xShape)
    (oneS : TorchLean.RefTy (m := m) (α := β) Shape.scalar)
    (two : β) :
    m (TorchLean.RefTy (m := m) (α := β) Shape.scalar ×
      TorchLean.RefTy (m := m) (α := β) xShape) := do
  let z1 ← _root_.Runtime.Autograd.Torch.linear
    (m := m) (α := β) (inDim := xDim) (outDim := width) w1 b1 x
  let h1 ← TorchLean.tanh (m := m) (α := β) (s := [width]) z1
  let sVec ← _root_.Runtime.Autograd.Torch.linear
    (m := m) (α := β) (inDim := width) (outDim := 1) w2 b2 h1
  let s0 : TorchLean.RefTy (m := m) (α := β) Shape.scalar ←
    TorchLean.reshape (m := m) (α := β) (s₁ := [1]) (s₂ := Shape.scalar) sVec (by
      simp [_root_.Spec.Shape.size])
  let V ← TorchLean.mul (m := m) (α := β) (s := Shape.scalar) s0 s0

  let w2Row : TorchLean.RefTy (m := m) (α := β) [width] ←
    TorchLean.reshape (m := m) (α := β) (s₁ := [1, width]) (s₂ := [width]) w2 (by
      simp [_root_.Spec.Shape.size])
  let h1Sq ← TorchLean.mul (m := m) (α := β) (s := [width]) h1 h1
  let oneW ← TorchLean.broadcastTo (m := m) (α := β) (s₁ := Shape.scalar)
    (s₂ := [width]) (Shape.CanBroadcastTo.scalarTo [width]) oneS
  let dh ← TorchLean.sub (m := m) (α := β) (s := [width]) oneW h1Sq
  let gHidden ← TorchLean.mul (m := m) (α := β) (s := [width]) w2Row dh
  let gHiddenM ← TorchLean.reshape (m := m) (α := β)
    (s₁ := [width]) (s₂ := [width, 1]) gHidden (by
      simp [_root_.Spec.Shape.size])
  let w1T ← TorchLean.swapAdjacentAtDepth (m := m) (α := β)
    (s := [width, xDim]) 0 w1
  let dsM ← TorchLean.matmul (m := m) (α := β)
    (batchA := .scalar) (batchB := .scalar) (batch := .scalar)
    (mDim := xDim) (nDim := width) (pDim := 1) w1T gHiddenM
  let ds ← TorchLean.reshape (m := m) (α := β)
    (s₁ := [xDim, 1]) (s₂ := xShape) dsM (by
      simp [xShape, _root_.Spec.Shape.size])
  let k : TorchLean.RefTy (m := m) (α := β) Shape.scalar ←
    TorchLean.scale (m := m) (α := β) (s := Shape.scalar) s0 (c := two)
  let kV ← TorchLean.broadcastTo (m := m) (α := β) (s₁ := Shape.scalar) (s₂ := xShape)
    (Shape.CanBroadcastTo.scalarTo xShape) k
  let gradV ← TorchLean.mul (m := m) (α := β) (s := xShape) kV ds
  pure (V, gradV)

/-- Evaluate the closed-loop van der Pol-like dynamics at `x`. -/
@[noinline, nospecialize]
def closedLoopDynamics
    {β : Type} [Context β] [DecidableEq Shape]
    {m : Type → Type} [Monad m]
    [_root_.Runtime.Autograd.Torch.Ops (m := m) (α := β)]
    (x : TorchLean.RefTy (m := m) (α := β) xShape)
    (u0 oneS : TorchLean.RefTy (m := m) (α := β) Shape.scalar)
    (mu one : β) : m (TorchLean.RefTy (m := m) (α := β) xShape) := do
  let x1 ← TorchLean.select (m := m) (α := β) (s := [xDim]) 0 x ⟨0, by decide⟩
  let x2 ← TorchLean.select (m := m) (α := β) (s := [xDim]) 0 x ⟨1, by decide⟩
  let x1Sq0 ← TorchLean.mul (m := m) (α := β) (s := Shape.scalar) x1 x1
  let oneMinus ← TorchLean.sub (m := m) (α := β) (s := Shape.scalar) oneS x1Sq0
  let term0 ← TorchLean.mul (m := m) (α := β) (s := Shape.scalar) oneMinus x2
  let term ← TorchLean.scale (m := m) (α := β) (s := Shape.scalar) term0 (c := mu)
  let negx1 ← TorchLean.scale (m := m) (α := β) (s := Shape.scalar) x1 (c := (-one))
  let dx2pre ← TorchLean.add (m := m) (α := β) (s := Shape.scalar) negx1 term
  let dx2 ← TorchLean.add (m := m) (α := β) (s := Shape.scalar) dx2pre u0
  let x2V ← TorchLean.reshape (m := m) (α := β) (s₁ := Shape.scalar)
    (s₂ := .dim 1 .scalar) x2 (by simp [_root_.Spec.Shape.size])
  let dx2V ← TorchLean.reshape (m := m) (α := β) (s₁ := Shape.scalar)
    (s₂ := .dim 1 .scalar) dx2 (by simp [_root_.Spec.Shape.size])
  TorchLean.concatLeadingAxis (m := m) (α := β) (s := .scalar)
    (nDim := 1) (mDim := 1) x2V dx2V

/-- Form the positivity and decrease penalties from `V`, `∇V`, and the closed-loop dynamics. -/
@[noinline, nospecialize]
def lossFromDynamics
    {β : Type} [Context β] [DecidableEq Shape]
    {m : Type → Type} [Monad m]
    [_root_.Runtime.Autograd.Torch.Ops (m := m) (α := β)]
    (x : TorchLean.RefTy (m := m) (α := β) xShape)
    (V : TorchLean.RefTy (m := m) (α := β) Shape.scalar)
    (gradV dynamics : TorchLean.RefTy (m := m) (α := β) xShape)
    (cV cD : β) : m (TorchLean.RefTy (m := m) (α := β) Shape.scalar) := do
  let prod ← TorchLean.mul (m := m) (α := β) (s := xShape) gradV dynamics
  let Vdot ← TorchLean.sum (m := m) (α := β) (s := xShape) prod
  let xSqV ← TorchLean.mul (m := m) (α := β) (s := xShape) x x
  let xSq ← TorchLean.sum (m := m) (α := β) (s := xShape) xSqV
  let posScaled ← TorchLean.scale (m := m) (α := β) (s := Shape.scalar) xSq (c := cV)
  let posExpr ← TorchLean.sub (m := m) (α := β) (s := Shape.scalar) posScaled V
  let posPenalty ← TorchLean.relu (m := m) (α := β) (s := Shape.scalar) posExpr
  let decScaled ← TorchLean.scale (m := m) (α := β) (s := Shape.scalar) V (c := cD)
  let decExpr ← TorchLean.add (m := m) (α := β) (s := Shape.scalar) Vdot decScaled
  let decPenalty ← TorchLean.relu (m := m) (α := β) (s := Shape.scalar) decExpr
  TorchLean.add (m := m) (α := β) (s := Shape.scalar) posPenalty decPenalty

@[noinline, nospecialize]
def lossProgram (width : Nat) :
    ∀ {β : Type}, [Context β] → [DecidableEq Shape] →
      TorchLean.Program β (paramShapes width ++ [xShape]) Shape.scalar :=
  fun {β} _ _ =>
    fun {m} _ _ =>
      fun wC bC w1 b1 w2 b2 x =>
        (do
          let mu : β := ((1 : Nat) : β)
          let scaleU : β := ((1 : Nat) : β)
          let cV : β := ((1 : Nat) : β) / ((10 : Nat) : β)
          let cD : β := ((1 : Nat) : β) / ((10 : Nat) : β)
          let one : β := ((1 : Nat) : β)
          let two : β := ((2 : Nat) : β)

          let oneS ← TorchLean.const (m := m) (α := β) (s := Shape.scalar) (Tensor.scalar one)
          let u0 ← controllerOutput (m := m) wC bC x scaleU
          let (V, gradV) ← lyapunovValueAndGradient (m := m) width w1 b1 w2 b2 x oneS two
          let dynamics ← closedLoopDynamics (m := m) x u0 oneS mu one
          lossFromDynamics (m := m) x V gradV dynamics cV cD
          : m (TorchLean.RefTy (m := m) (α := β) Shape.scalar))

end NN.MLTheory.CROWN.Lyapunov.TwoStage.Core
