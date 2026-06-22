/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN

/-!
# Functional transcendentals + scalar-affine: autograd correctness

Positive / negative example for the `nn.functional.{exp, log, scale, shift, affine}`
ops added for scientific forward models — e.g. the soil-moisture retrieval that
combines SMAP (Soil Moisture Active Passive) and NISAR (NASA–ISRO Synthetic
Aperture Radar) observations through the AVS (Attenuation–Volume–Surface) model,
whose surface term is `exp(-2·b·NDVI)·c·|R|²`.

The point is that these ops are differentiated by the **autograd engine**, so a
forward model written once yields its gradient with no hand-coded derivative.
Each check differentiates a tiny function and compares the autograd gradient to
the closed form:

* positive controls — the gradient matches the analytic value;
* negative controls — a deliberately *wrong* analytic value (notably the
  wrong-sign gradient of `exp(-2x)`) does **not** match. That is exactly the
  defect class — a sign/factor error in a hand-coded Jacobian — that deriving the
  gradient by autograd eliminates.

`checkAll` runs as a compiled executable — `lake exe transcendentals_check` — and
exits non-zero on any regression. It is deliberately *not* an `#eval` check:
autograd uses the native tape externs, which the interpreter cannot load (see the
`main` entry point below).
-/

@[expose] public section

namespace NN.Examples.Functional.Transcendentals

open Spec
open Tensor
open NN.Tensor
open NN.API

/-! ## Functions under test (written once; gradients come from autograd) -/

/-- `f(x) = eˣ`. -/
def expFn : autograd.fn1.Fn Spec.Shape.scalar Spec.Shape.scalar :=
  fun x => nn.functional.exp x

/-- `f(x) = e^{-2x}` — the shape of the AVS canopy two-way transmittance as a
function of the attenuation parameter. -/
def expNeg2Fn : autograd.fn1.Fn Spec.Shape.scalar Spec.Shape.scalar :=
  fun x => do
    let u ← nn.functional.scale x (-Numbers.two)
    nn.functional.exp u

/-- `f(x) = 3·x + 1` via the scalar-affine op. -/
def affineFn : autograd.fn1.Fn Spec.Shape.scalar Spec.Shape.scalar :=
  fun x => nn.functional.affine x Numbers.three Numbers.one

/-! ## Float checks -/

/-- Absolute-tolerance float compare. -/
def approx (a b : Float) (tol : Float := 1e-6) : Bool := (a - b).abs ≤ tol

/-- Positive control: `name`'s autograd gradient ≈ expected; throws on mismatch. -/
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
def gradAt (f : autograd.fn1.Fn Spec.Shape.scalar Spec.Shape.scalar) (x0 : Float) :
    IO Float := do
  let x : Spec.Tensor Float Spec.Shape.scalar := Spec.fill (x0 : Float) Spec.Shape.scalar
  let g ← autograd.fn1.grad (α := Float) f x
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
  let gn ← gradAt expNeg2Fn 0.5
  expectGrad "exp(-2x)"      gn ((-2.0) * Float.exp (-1.0))
  -- THE AVS bug class: the wrong-SIGN analytic (+2·e^{-2x}) must NOT match.
  expectNot  "exp(-2x) sign" gn (( 2.0) * Float.exp (-1.0))

  IO.println "[transcendentals] all positive + negative controls passed ✓"

end NN.Examples.Functional.Transcendentals

/-- Compiled entry point.  Autograd uses the native runtime, so this runs as a
compiled `lean_exe` (`lake exe transcendentals_check`), not via `#eval` (the
interpreter cannot load the native tape externs). -/
def main : IO Unit := NN.Examples.Functional.Transcendentals.checkAll
