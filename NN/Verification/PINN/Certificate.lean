/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.PINN.Core
public import NN.Verification.PINN.PdeAst
public import NN.Verification.PINN.PdeParse
public import NN.MLTheory.CROWN.Graph
public import NN.Verification.Util.Json

/-!
# PINN Certificate

PINN certificate checker (recompute-and-compare).

This module is the executable checker for the PINN certificate workflow:
- parse a JSON certificate produced by Python,
- rebuild the same CROWN graph and seed the same input boxes,
- recompute IBP + derivative bounds in Lean, and
- compare the resulting residual intervals against the exported values.

It is conservative by design: it validates the export/import path and interval computations, rather
than trying to be a fully featured PDE verifier.

References / context:
- PINNs: Raissi et al. (2019), "Physics-informed neural networks" (JCP)
- CROWN/LiRPA background (for the bound propagation machinery): `https://arxiv.org/abs/1811.00866`

Export (Python):
`python3.12 scripts/verification/pinn/export_pinn_cert.py`

Run (Lean):
`lake exe verify -- pinn-cert [NN/Examples/Verification/PINN/pinn_cert.json]`
-/

@[expose] public section


namespace NN.Verification.PINN.Certificate

open NN.Verification.PINN
open NN.Verification.PINN.PdeAst
open NN.Verification.PINN.PdeParse
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open _root_.Spec
open _root_.Spec.Tensor
open Lean
open Json

/-- Bundled PINN certificate sample used by `lake exe verify -- pinn-cert`. -/
def defaultCertPath : String :=
  "NN/Examples/Verification/PINN/pinn_cert.json"

/-- Reject a certificate interval that differs from Lean's recomputed interval. -/
def requireApproxPair (ctx : String) (leanPair artifactPair : Float × Float) : IO Unit :=
  if approxEq leanPair.1 artifactPair.1 && approxEq leanPair.2 artifactPair.2 then
    pure ()
  else
    throw <| IO.userError
      s!"{ctx}: Lean {leanPair} differs from certificate {artifactPair}"

/-- IO entry that reads the cert, recomputes bounds, and prints comparisons. -/
def verifyCert (path : String) : IO Unit := do
  let j ← NN.Verification.Json.readJsonFile path
  match parseCert j with
  | .error msg => throw <| IO.userError s!"Bad Cert JSON: {msg}"
  | .ok (cfg, residPairs, residPairsDeriv, uTriples) => do
    let g := buildGraph
    let outId := g.nodes.size - 1
    let basePs := seedParamsFloat
    let residA := residPairs.toArray
    let residDerivA := residPairsDeriv.toArray
    let uTriplesA := uTriples.toArray
    for i in List.finRange cfg.nPts do
      let x := Tensor.vecGet cfg.pts i
      let xs := #[x - cfg.h, x, x + cfg.h]
      let mut uTrip : List (Float × Float) := []
      let mut duTrip : List (Float × Float) := []
      let mut d2uTrip : List (Float × Float) := []
      for xi in xs do
        let ps := seedInputFloat basePs xi cfg.eps
        let boxes := NN.MLTheory.CROWN.Graph.runIBP (α:=Float) g ps
        let outB ←
          match NN.MLTheory.CROWN.Graph.outputBox? boxes outId with
          | .ok outB => pure outB
          | .error msg => throw <| IO.userError s!"PINN IBP failed: {msg}"
        let loVal := Spec.Tensor.sumSpec outB.lo
        let hiVal := Spec.Tensor.sumSpec outB.hi
        uTrip := uTrip ++ [(loVal, hiVal)]
        let dboxes := NN.MLTheory.CROWN.Graph.runFirstDerivative1D (α:=Float) g ps boxes
        let dB ←
          match NN.MLTheory.CROWN.Graph.outputBox? dboxes outId with
          | .ok dB => pure dB
          | .error msg => throw <| IO.userError s!"PINN first-derivative propagation failed: {msg}"
        let dlo := Spec.Tensor.sumSpec dB.lo
        let dhi := Spec.Tensor.sumSpec dB.hi
        duTrip := duTrip ++ [(dlo, dhi)]
        let d2boxes := NN.MLTheory.CROWN.Graph.runSecondDerivative1D (α:=Float) g ps boxes dboxes
        let d2B ←
          match NN.MLTheory.CROWN.Graph.outputBox? d2boxes outId with
          | .ok d2B => pure d2B
          | .error msg => throw <| IO.userError s!"PINN second-derivative propagation failed: {msg}"
        let d2lo := Spec.Tensor.sumSpec d2B.lo
        let d2hi := Spec.Tensor.sumSpec d2B.hi
        d2uTrip := d2uTrip ++ [(d2lo, d2hi)]
      match uTrip with
      | [(lm, hm), (l0, h0), (lp, hp)] =>
        let (xArtifact, umArtifact, u0Artifact, upArtifact) ←
          match uTriplesA[i.1]? with
          | some entry => pure entry
          | none =>
              throw <| IO.userError
                s!"PINN certificate u_bounds list missing index {i.1} (size={uTriplesA.size})"
        if !approxEq x xArtifact then
          throw <| IO.userError
            s!"PINN certificate point mismatch at index {i.1}: Lean {x}, certificate {xArtifact}"
        requireApproxPair s!"u(x-h) mismatch at x={x}" (lm, hm) umArtifact
        requireApproxPair s!"u(x) mismatch at x={x}" (l0, h0) u0Artifact
        requireApproxPair s!"u(x+h) mismatch at x={x}" (lp, hp) upArtifact
        let (rlo, rhi) := fdResidualBounds (lm,hm) (l0,h0) (lp,hp) cfg.h
        let (eradLo, eradHi) ←
          match residA[i.1]? with
          | some pair => pure pair
          | none =>
              throw <| IO.userError
                s!"PINN certificate residual list missing index {i.1} (size={residA.size})"
        requireApproxPair s!"finite-difference residual mismatch at x={x}"
          (rlo, rhi) (eradLo, eradHi)
        let (dLoPy, dHiPy) ←
          match residDerivA[i.1]? with
          | some pair => pure pair
          | none =>
              throw <| IO.userError
                s!"PINN certificate derivative-residual list missing index {i.1} (size={residDerivA.size})"
        -- Compute and print residual bounds from the PDE specification via the parser/AST.
        -- We support a small DSL: u, ux, uxx, uy, uyy, +, -, *, scaling constants, parentheses, and
        -- powers by ^n.
        let env : String → Option Float := fun _ => none
        -- identifiers map, can be extended to constants
        let pdeParsed ←
          match parseExpr env cfg.pde with
          | .ok e => pure e
          | .error msg => throw <| IO.userError s!"PINN PDE parse failed: {msg}"
        -- Build primitive bounds at the central point x using computed intervals
        let prims ←
          match uTrip, duTrip, d2uTrip with
          | [_, (u0l,u0h), _], [_, (d1l,d1h), _], [_, (d2l,d2h), _] =>
            pure
              { u := some (u0l, u0h)
                duX := some (d1l, d1h)
                duY := none
                d2uX := some (d2l, d2h)
                d2uY := none }
          | _, _, _ => throw <| IO.userError "PINN derivative triplets are incomplete"
        let pdeResidual ←
          match eval prims pdeParsed with
          | some residual => pure residual
          | none =>
              throw <| IO.userError
                s!"PINN PDE '{cfg.pde}' evaluation failed because required primitives are missing"
        requireApproxPair s!"derivative residual mismatch at x={x}"
          pdeResidual (dLoPy, dHiPy)
        IO.println
          s!"Residual R(x) from PDE '{cfg.pde}': [{pdeResidual.1},{pdeResidual.2}]"
        match duTrip with
        | [d1, d2, d3] =>
          let (d1l, d1h) := d1; let (d2l, d2h) := d2; let (d3l, d3h) := d3
          IO.println s!"u'(x-h)∈[{d1l},{d1h}], u'(x)∈[{d2l},{d2h}], u'(x+h)∈[{d3l},{d3h}]"
        | _ => pure ()
        match d2uTrip with
        | [dd1, dd2, dd3] =>
          let (dd1l, dd1h) := dd1; let (dd2l, dd2h) := dd2; let (dd3l, dd3h) := dd3
          IO.println s!"u''(x-h)∈[{dd1l},{dd1h}], u''(x)∈[{dd2l},{dd2h}], u''(x+h)∈[{dd3l},{dd3h}]"
        | _ => pure ()
      | _ => throw <| IO.userError "unexpected uTrip structure"
    IO.println "PINN artifact replay matched Lean's recomputed residual bounds."

end NN.Verification.PINN.Certificate
