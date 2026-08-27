/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.Proofs.Autograd.FDeriv.Elementwise

/-!
# Functional transcendentals + scalar-affine: proofs and runtime regression checks

Positive / negative example for the `nn.functional.{exp, log, scale, shift, affine}`
ops added for scientific forward models — e.g. the soil-moisture retrieval that
combines SMAP (Soil Moisture Active Passive) and NISAR (NASA–ISRO Synthetic
Aperture Radar) observations through the AVS (Attenuation–Volume–Surface) model,
whose surface term is $\exp(-2b\,\mathrm{NDVI})\,c\,|R|^2$.

These operations are differentiated by the **autograd engine**, so a forward model written once
yields its gradient without a hand-coded derivative.
This file has two layers:

* a proof layer handle for the real-valued `exp` op, using
  `Proofs.Autograd.OpSpecFDerivCorrect.exp` and the generic
  `backward_eq_adjoint_fderiv` theorem;
* runtime regression checks that differentiate tiny Float functions and compare the
  autograd gradient to the closed form.

The runtime checks below are not the proof. They make sure the executable tape path used by
scientific examples still follows the expected derivative numerically. The proof layer declarations
state the corresponding operation specifications and derivative theorems.

Each runtime check differentiates a tiny function and compares the autograd gradient to the closed
form:

* positive controls — the gradient matches the analytic value;
* negative controls — a deliberately *wrong* analytic value (notably the
  wrong-sign gradient of $\exp(-2x)$) does **not** match. That is exactly the
  defect class — a sign/factor error in a hand-coded Jacobian — that deriving the
  gradient by autograd eliminates.

`checkAll` runs as a compiled executable — `lake exe transcendentals_check` — and
exits non-zero on any regression. It is deliberately *not* an `#eval` check:
autograd uses the native tape externs, which the interpreter cannot load (see the
`main` entry point below).
-/

@[expose] public section

namespace NN.Examples.Functional.Transcendentals

open Spec.Tensor
open TorchLean.Tensor
open TorchLean

/-! ## Proof objects and runtime checks -/

noncomputable section

/--
The theorem-backed real-valued exp op used by the proof layer.

This is the actual proof layer object: it packages the forward op, its JVP, a Fréchet-derivative
candidate, and the theorem that the JVP is the true derivative. The runtime checks below exercise
the executable Float tape; this declaration points to the corresponding real-valued theorem.
-/
def expProofSurface : Proofs.Autograd.OpSpecFDerivCorrect 1 1 :=
  Proofs.Autograd.OpSpecFDerivCorrect.exp (n := 1)

/--
For scalar exp over `ℝ`, the proved backward rule is the adjoint of the Fréchet derivative.

This is the theorem-level statement that the executable regression check is meant to complement.
-/
theorem expBackward_eq_adjoint_fderiv
    (x δ : Spec.Tensor ℝ [1]) :
    Proofs.Autograd.getScalarE (expProofSurface.correct.op.backward x δ) =
      Proofs.Autograd.vjp expProofSurface.forwardVec (Proofs.Autograd.getScalarE x)
        (Proofs.Autograd.getScalarE δ) :=
  Proofs.Autograd.OpSpecFDerivCorrect.backward_eq_adjoint_fderiv expProofSurface x δ

end

/-! ## Functions under test (written once; gradients come from autograd) -/

/-- $f(x)=e^x$. -/
def expFn : autograd.func.TensorFunction [] [] :=
  fun x => nn.functional.exp x

/-- $f(x)=e^{-2x}$ — the shape of the AVS canopy two-way transmittance as a
function of the attenuation parameter. -/
def expNegativeTwoFn : autograd.func.TensorFunction [] [] :=
  fun x => do
    let u ← nn.functional.scale x (-Numbers.two)
    nn.functional.exp u

/-- $f(x)=3x+1$ via the scalar-affine op. -/
def affineFn : autograd.func.TensorFunction [] [] :=
  fun x => nn.functional.affine x Numbers.three Numbers.one

/-! ## Float checks -/

/-- Absolute-tolerance float compare. -/
def approx (a b : Float) (tol : Float := 1e-6) : Bool := (a - b).abs ≤ tol

/-- Positive control: `name`'s autograd gradient is approximately equal to the expected value;
throws on mismatch. -/
def expectGrad (name : String) (got expected : Float) (tol : Float := 1e-6) : IO Unit :=
  if approx got expected tol then
    IO.println s!"[PASS] {name}: grad = {got} ≈ {expected}"
  else
    throw <| IO.userError s!"[FAIL] {name}: grad = {got}, expected {expected}"

/-- Negative control: the gradient must *not* equal `wrong`; throws if it does. -/
def expectNot (name : String) (got wrong : Float) (tol : Float := 1e-6) : IO Unit :=
  if approx got wrong tol then
    throw <| IO.userError s!"[FAIL-NEG] {name}: grad = {got} wrongly matched {wrong}"
  else
    IO.println s!"[PASS-NEG] {name}: grad = {got} ≠ {wrong} (test discriminates)"

/-- Differentiate a scalar→scalar `Fn` at a Float point, returning the gradient. -/
def gradAt (f : autograd.func.TensorFunction [] []) (x0 : Float) :
    IO Float := do
  let x := Tensor.full [] x0
  let g ← autograd.func.grad (inputShape := []) (α := Float) f x
  pure (Spec.toScalarSpec g)

def checkAll : IO Unit := do
  -- exp:  d/dx eˣ = eˣ
  let ge ← gradAt expFn 0.5
  expectGrad "exp"   ge (Float.exp 0.5)
  expectNot  "exp≠1" ge 1.0                       -- a constant-1 gradient would be caught

  -- affine:  d/dx (3x+1) = 3
  let ga ← gradAt affineFn 0.5
  expectGrad "affine(3x+1)"   ga 3.0
  expectNot  "affine≠1"       ga 1.0              -- the slope is 3, not 1

  -- exp(-2x):  d/dx e^{-2x} = -2·e^{-2x}
  let gn ← gradAt expNegativeTwoFn 0.5
  expectGrad "exp(-2x)"      gn ((-2.0) * Float.exp (-1.0))
  -- THE AVS bug class: the wrong-SIGN analytic (+2·e^{-2x}) must NOT match.
  expectNot  "exp(-2x) sign" gn (( 2.0) * Float.exp (-1.0))

  IO.println "[transcendentals] all positive + negative controls passed ✓"

end NN.Examples.Functional.Transcendentals

/-- Compiled entry point. Autograd uses the native runtime, so this runs as a compiled `lean_exe`
(`lake exe transcendentals_check`), not via `#eval` (the interpreter cannot load native tape
externs). -/
def main : IO Unit := NN.Examples.Functional.Transcendentals.checkAll
