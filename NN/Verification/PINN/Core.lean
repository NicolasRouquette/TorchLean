/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph
public import NN.Tensor
public import NN.Verification.PINN.Architecture
public import NN.Verification.Util.Json

/-!
# PINN Core

PINN helper library: reference graphs, seeding, derivatives, and certificate parsing.

This module is shared by the PINN verification workflows. It provides:
- a dimension-parameterized CROWN graph for a tanh MLP,
- deterministic parameters and typed input-box seeding,
- a few interval/finite-difference residual helpers,
- JSON parsing for the certificate schema used by the surrounding examples.

Run the curated entrypoints instead of importing this file directly:
- `lake exe verify -- pinn-cert [NN/Examples/Verification/PINN/pinn_cert.json]`
- `lake exe verify -- pinn-dataset-check --dataset=PATH.json [--weights=WEIGHTS.json]`

References:
- PINNs (physics-informed neural nets): `https://arxiv.org/abs/1711.10561`
- CROWN (linear bound propagation): `https://arxiv.org/abs/1811.00866`
- IBP (interval bound propagation): `https://arxiv.org/abs/1810.12715`
-/

@[expose] public section


namespace NN.Verification.PINN

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open _root_.Spec
open _root_.Spec.Tensor
open Lean
open Json

/-- Require a JSON object field in an `Except` parser. -/
def expectFieldObjE (ctx key : String) (j : Json) :
    Except String (Std.TreeMap.Raw String Json compare) := do
  TorchLean.Json.expectObjE s!"{ctx}.{key}" (← TorchLean.Json.expectFieldE ctx key j)

/-- Require a JSON array field in an `Except` parser. -/
def expectFieldArrayE (ctx key : String) (j : Json) : Except String (Array Json) := do
  TorchLean.Json.expectArrayE s!"{ctx}.{key}" (← TorchLean.Json.expectFieldE ctx key j)

/-- Require a float-array field. -/
def expectFieldFloatArrayE (ctx key : String) (j : Json) :
    Except String (Array Float) := do
  NN.Verification.Json.expectFiniteFloatArrayE s!"{ctx}.{key}"
    (← TorchLean.Json.expectFieldE ctx key j)

/-- Require an interval object `{ "lo": ..., "hi": ... }`. -/
def expectIntervalPairE (ctx : String) (j : Json) : Except String (Float × Float) := do
  let lo ← NN.Verification.Json.expectFieldFiniteFloatE ctx "lo" j
  let hi ← NN.Verification.Json.expectFieldFiniteFloatE ctx "hi" j
  if lo ≤ hi then
    pure (lo, hi)
  else
    throw s!"{ctx}: lower endpoint {lo} exceeds upper endpoint {hi}"

/-- Require an array of interval pairs with an expected length. -/
def expectIntervalPairArrayE (ctx : String) (j : Json) (expected : Nat) :
    Except String (Array (Float × Float)) := do
  let loA ← expectFieldFloatArrayE ctx "lo" j
  let hiA ← expectFieldFloatArrayE ctx "hi" j
  if h : loA.size = expected ∧ hiA.size = expected then
    pure <| Array.ofFn (fun (i : Fin expected) =>
      have hlo : i.val < loA.size := by
        rw [h.1]
        exact i.isLt
      have hhi : i.val < hiA.size := by
        rw [h.2]
        exact i.isLt
      (loA[i.val]'hlo, hiA[i.val]'hhi))
  else
    throw s!"{ctx}: length mismatch (expected {expected})"

/-- One finite-difference sample from a PINN certificate. -/
abbrev UBoundsEntry :=
  Float × (Float × Float) × (Float × Float) × (Float × Float)

/-- Require one finite-difference `u_bounds` entry. -/
def expectUBoundsEntryE (ctx : String) (j : Json) :
    Except String UBoundsEntry := do
  let x ← NN.Verification.Json.expectFieldFiniteFloatE ctx "x" j
  let uMinus ← expectIntervalPairE s!"{ctx}.u_minus" (← TorchLean.Json.expectFieldE ctx "u_minus" j)
  let u ← expectIntervalPairE s!"{ctx}.u" (← TorchLean.Json.expectFieldE ctx "u" j)
  let uPlus ← expectIntervalPairE s!"{ctx}.u_plus" (← TorchLean.Json.expectFieldE ctx "u_plus" j)
  pure (x, uMinus, u, uPlus)

/-- Configuration parsed from a PINN certificate JSON. -/
structure PinnCfg where
  /-- PDE identifier carried by the certificate. -/
  pde : String
  /-- Grid spacing used by the exported finite-difference residual. -/
  h   : Float
  /-- Input perturbation radius for interval checking. -/
  eps : Float
  /-- Number of sample points encoded in `pts`. -/
  nPts : Nat
  /--
  Sample points as a length-`nPts` 1D tensor.

  PyTorch analogue: this is the `torch.Tensor` you would keep in memory after loading a JSON/CSV
  list of sample coordinates.
  -/
  pts : Spec.Tensor Float [nPts]

/-- Approximate equality for `Float` used by certificate consistency checks. -/
def approxEq (x y : Float) (tol : Float := 1e-5) : Bool :=
  let d := if x > y then x - y else y - x
  decide (d ≤ tol)

/-- The reference scalar-output PINN architecture at an arbitrary input dimension. -/
def referenceArch (inputDim : Nat) : SequentialPINNArch :=
  { inputDim := inputDim
    hiddenDims := #[16, 16]
    outputDim := 1
    activation := .tanh }

/-- Build the reference PINN graph at the requested input dimension. -/
def buildReferenceGraph (inputDim : Nat) : Graph :=
  (referenceArch inputDim).buildGraph

/--
Deterministic reference parameters at an arbitrary input dimension.

These values mirror the bundled exporter. They are demonstration parameters, not trained weights;
production checks should load the exported state through `PINN.PyTorch`.
-/
def referenceParams {α : Type} [Context α] (inputDim : Nat) : ParamStore α :=
  let firstWeight : Tensor α [16, inputDim] :=
    Tensor.dim fun i => Tensor.dim fun j =>
      let base := ((i.val + 1 : Nat) : α) * (Numbers.half * Numbers.oneTenth)
      Tensor.scalar <| if j.val = 0 then base * Numbers.two else base
  let eight : α := Numbers.four * Numbers.two
  let firstBias : Tensor α [16] :=
    Tensor.dim fun i =>
      Tensor.scalar <| Numbers.half * Numbers.oneTenth * ((i.val : Nat) : α) -
        Numbers.half * Numbers.oneTenth * eight
  let middleWeight : Tensor α [16, 16] :=
    Tensor.dim fun i => Tensor.dim fun j =>
      Tensor.scalar <| if i = j then Numbers.one else Numbers.half * Numbers.oneTenth
  let middleBias : Tensor α [16] := Tensor.dim fun _ => Tensor.scalar Numbers.zero
  let hundredth : α := Numbers.oneTenth * Numbers.oneTenth
  let outputWeight : Tensor α [1, 16] :=
    Tensor.dim fun _ => Tensor.dim fun j =>
      Tensor.scalar <| Numbers.oneTenth + hundredth * ((j.val : Nat) : α)
  let outputBias : Tensor α [1] := Tensor.dim fun _ => Tensor.scalar Numbers.zero
  let params : ParamStore α := {}
  let params :=
    { params with
      linearWB := params.linearWB.insert 1
        { m := 16, n := inputDim, w := firstWeight, b := firstBias } }
  let params :=
    { params with
      linearWB := params.linearWB.insert 3
        { m := 16, n := 16, w := middleWeight, b := middleBias } }
  { params with
    linearWB := params.linearWB.insert 5
      { m := 1, n := 16, w := outputWeight, b := outputBias } }

/-- Seed an $\ell_\infty$ input box centered at a typed input tensor. -/
def seedInput {α : Type} [Context α] {inputDim : Nat}
    (ps : ParamStore α) (center : Tensor α [inputDim]) (eps : α) : ParamStore α :=
  ps.seedLInfBall 0 center eps

/-- Enclose one diagonal Hessian entry at the graph's output node. -/
def secondDerivativeBounds (g : Graph) (ps : ParamStore Float)
    (axis : Fin ((ps.inputBoxes[0]?).map (·.dim) |>.getD 0)) : Option (Float × Float) :=
  let ibp := NN.MLTheory.CROWN.Graph.runIBP (α := Float) g ps
  let inDim := (ps.inputBoxes[0]?).map (·.dim) |>.getD 0
  let direction := FlatBox.ofTensor (TorchLean.Tensor.oneHot (α := Float) inDim axis)
  let first := NN.MLTheory.CROWN.Graph.runDirectionalDerivative (α := Float) g ps ibp direction
  let second := NN.MLTheory.CROWN.Graph.runScalarSecondDerivative (α := Float) g ps ibp first
  match second[(SequentialPINNArch.graphOutputId g)]? with
  | some (some bounds) => some (Spec.Tensor.sumSpec bounds.lo, Spec.Tensor.sumSpec bounds.hi)
  | _ => none

/-- Enclose every diagonal Hessian entry at the graph's output node. -/
def hessianDiagonalBounds (g : Graph) (ps : ParamStore Float) : Array (Option (Float × Float)) :=
  let inDim := (ps.inputBoxes[0]?).map (·.dim) |>.getD 0
  Array.ofFn fun axis : Fin inDim => secondDerivativeBounds g ps axis

/-- Enclose the Laplacian by summing all diagonal Hessian enclosures. -/
def laplacianBounds (g : Graph) (ps : ParamStore Float) : Option (Float × Float) := do
  let bounds := hessianDiagonalBounds g ps
  if bounds.isEmpty then none else
    bounds.foldlM (init := (0.0, 0.0)) fun (lo, hi) bound => do
      let (axisLo, axisHi) ← bound
      pure (lo + axisLo, hi + axisHi)

/-- Parse the JSON certificate consumed by the PINN verification CLI. -/
def parseCert (j : Json) : Except String
    (PinnCfg × Array (Float × Float) × Array (Float × Float) × Array UBoundsEntry) := do
  let _ ← TorchLean.Json.expectObjE "PINN certificate" j
  let po ← expectFieldObjE "PINN certificate" "pinn" j
  let pdeStr ←
    match Std.TreeMap.Raw.get? po "pde" with
    | none => pure "u''(x) = 0"
    | some Json.null => pure "u''(x) = 0"
    | some pdeJ =>
        match pdeJ with
        | .str s => pure s
        | _ => throw "PINN certificate.pinn.pde: expected string"
  let h ← NN.Verification.Json.expectFieldFiniteFloatE "PINN certificate.pinn" "h" (.obj po)
  let eps ← NN.Verification.Json.expectFieldFiniteFloatE "PINN certificate.pinn" "eps" (.obj po)
  unless h > 0.0 do
    throw s!"PINN certificate.pinn.h: expected a positive spacing, got {h}"
  unless eps ≥ 0.0 do
    throw s!"PINN certificate.pinn.eps: expected a nonnegative radius, got {eps}"
  let ptsA ← expectFieldFloatArrayE "PINN certificate.pinn" "points" (.obj po)
  let nPts := ptsA.size
  unless nPts > 0 do
    throw "PINN certificate.pinn.points: expected at least one sample point"
  let pts : Spec.Tensor Float [nPts] :=
    Spec.Tensor.dim (fun i => Spec.Tensor.scalar ptsA[i])
  let rb ← TorchLean.Json.expectFieldE "PINN certificate" "residual_bounds" j
  let resPairs ← expectIntervalPairArrayE "PINN certificate.residual_bounds" rb nPts
  let derivJ ← TorchLean.Json.expectFieldE "PINN certificate" "residual_bounds_deriv" j
  let resPairsDeriv ←
    expectIntervalPairArrayE "PINN certificate.residual_bounds_deriv" derivJ nPts
  let ubA ← expectFieldArrayE "PINN certificate" "u_bounds" j
  let uTriples ← ubA.mapIdxM fun i entry =>
    expectUBoundsEntryE s!"PINN certificate.u_bounds[{i}]" entry
  unless uTriples.size = nPts do
    throw s!"PINN certificate.u_bounds: expected {nPts} entries, got {uTriples.size}"
  pure ({ pde := pdeStr, h := h, eps := eps, nPts := nPts, pts := pts }
    , resPairs, resPairsDeriv, uTriples)

/-- Finite-difference residual bounds for 1D second derivative. -/
def fdResidualBounds (u_minus : Float × Float) (u0 : Float × Float) (u_plus : Float × Float) (h :
  Float) : Float × Float :=
  let (lminus, hminus) := u_minus
  let (l0, h0) := u0
  let (lplus, hplus) := u_plus
  let num_lo := lplus - 2.0 * h0 + lminus
  let num_hi := hplus - 2.0 * l0 + hminus
  let s := Numbers.one / (h * h)
  (num_lo * s, num_hi * s)

end NN.Verification.PINN
