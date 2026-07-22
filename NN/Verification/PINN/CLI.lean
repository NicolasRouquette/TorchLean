/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.CLI
public import NN.Verification.PINN.Core
public import NN.Verification.PINN.PdeParse
public import NN.Verification.PINN.PyTorch
public import NN.Verification.Util.Json

/-!
# PINN CLI

PINN residual-bounding CLI.

This file implements a small interactive tool for bounding PDE residuals of a trained
physics-informed neural network (PINN) over a box input region. Concretely, it:
- parses a compact PDE mini-language (see `PdeAst` / `PdeParse`),
- evaluates that expression using interval bounds for `u`, `du`, and `d2u`,
- computes those primitive bounds via IBP/CROWN-style propagation on a CROWN `Graph`,
- optionally tightens bounds by recursively splitting the input box (1D/2D).

This CLI is the interactive PINN residual-bound tool:
- use it when you want to inspect residual bounds interactively;
- use it when you are iterating on a PDE expression or input box;
- use the certificate checker when you want a stable artifact for docs, papers, or CI.

The stable artifact checker is:
`lake exe verify -- pinn-cert [NN/Examples/Verification/PINN/pinn_cert.json]`

Run this CLI via the unified verification dispatcher:
`lake exe verify -- pinn-cli -- [flags] "<PDE>" x eps`
or for 2D:
`lake exe verify -- pinn-cli -- [flags] "<PDE>" x y eps`

Examples:
- `lake exe verify -- pinn-cli -- "u_xx + u" 0.0 0.1`
- `lake exe verify -- pinn-cli -- --backend=float "u_t - u_xx" 0.0 0.05`

References:
- PINNs: `https://arxiv.org/abs/1711.10561`
- IBP: `https://arxiv.org/abs/1810.12715`
- CROWN: `https://arxiv.org/abs/1811.00866`
-/

@[expose] public section


namespace NN.Verification.PINN.CLI

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open NN.Verification.PINN
open NN.Verification.PINN.PdeAst
open NN.Verification.PINN.PdeParse
open NN.Verification.PINN.ResidualAffine
open _root_.Spec
open _root_.Spec.Tensor
open TorchLean.Floats

/-- Backend selection for the PINN CLI. -/
inductive Backend where
  | float

/-- Parse the backend flag used by the PINN residual CLI. -/
def parseBackendVal (s : String) : Option Backend :=
  match s with
  | "float" => some .float
  | _ => none

/-- Parse a decimal Float literal used by CLI flags. -/
def parseFloat : String → Option Float :=
  TorchLean.CLI.parseFloatLit

/-- Parse a non-scientific decimal string into a generic numeric α using Numbers and Context.
  Supports optional leading '-' and a single '.'. -/
def parseAlphaDecimal {α : Type} [Context α] (s : String) : Option α :=
  let rec pow10 (k : Nat) : α :=
    match k with
    | 0 => Numbers.one
    | Nat.succ k' => Numbers.ten * pow10 k'
  let mk (neg : Bool) (intPart fracPart : String) : Option α :=
    let intVal : α := (
      (intPart.toList.foldl (fun (acc : α × Bool) ch =>
        let (accv, seen) := acc
        if seen then (accv, true) else
          if ch = '0' then (accv * Numbers.ten, false)
          else if ch ≥ '0' ∧ ch ≤ '9' then
            (accv * Numbers.ten + ((ch.toNat - '0'.toNat) : Nat), false)
          else (accv, true)
      ) (Numbers.zero, false)).fst)
    let fracVal? : Option α :=
      if fracPart.isEmpty then some Numbers.zero else
      let digits := fracPart.toList
      let numDen := digits.foldl (fun (acc : α × Nat × Bool) ch =>
        let (n, d, bad) := acc
        if bad then (n, d, bad) else
          if ch ≥ '0' ∧ ch ≤ '9' then
            (n * Numbers.ten + ((ch.toNat - '0'.toNat) : Nat), d + 1, false)
          else (n, d, true)
      ) (Numbers.zero, 0, false)
      let (num, denK, bad) := numDen
      if bad then none else
        some (num / (pow10 denK))
    match fracVal? with
    | some fracVal =>
      let v := intVal + fracVal
      some (if neg then (-v) else v)
    | none => none
  let s := s.trimAscii.toString
  if s.isEmpty then none else
  let neg := s.front = '-'
  let body : String := if neg then (s.drop 1).toString else s
  match body.splitOn "." with
  | [intPart] => mk neg intPart ""
  | [intPart, fracPart] => mk neg intPart fracPart
  | _ => none



/-- Select interval-only, forward CROWN, or backward CROWN bounds for the PINN output value. -/
inductive UBoundsMethod
  | ibp
  | crownFwd
  | crownBwd

/-- Compute primitives u, duX, duY, d2uX, d2uY at the unique output node (id=5) for 1D/2D models.
    Optionally, replace the `u`-interval with a tighter CROWN/DeepPoly bound. -/
def computePrimsAt (g : Graph) (ps : ParamStore Float) (uMethod : UBoundsMethod := .ibp) (backend :
  Backend := .float) : IO Prims := do
  let outId := NN.Verification.PINN.SequentialPINNArch.graphOutputId g
  let _ := backend
  let ibp := runIBP (α:=Float) g ps
  let outB ←
    match NN.MLTheory.CROWN.Graph.outputBox? ibp outId with
    | .ok outB => pure outB
    | .error msg => throw <| IO.userError s!"IBP failed at output: {msg}"
  let uLo := Spec.Tensor.sumSpec outB.lo
  let uHi := Spec.Tensor.sumSpec outB.hi
  -- Determine input dimension from graph's input node shape
  let inDim : Nat :=
    match g.nodes[0]? with
    | some n0 => (match n0.outShape with | .dim n .scalar => n | _ => 1)
    | none => 1
  let hasY : Bool :=
    match inDim with
    | 0 => false
    | 1 => false
    | _ => true
  -- First/second derivative along X (dir 0)
  let seedX := FlatBox.ofTensor (NN.Tensor.oneHotNat (α := Float) inDim 0)
  let d1x := runDerivDirectional (α:=Float) g ps ibp seedX
  let d2x := runDeriv2D (α:=Float) g ps ibp d1x
  let d1xOpt := (NN.MLTheory.CROWN.Graph.outputBox? d1x outId).toOption
  let d2xOpt := (NN.MLTheory.CROWN.Graph.outputBox? d2x outId).toOption
  let (duX, d2uX) :=
    match d1xOpt, d2xOpt with
    | some dxB, some d2xB =>
      (some (Spec.Tensor.sumSpec dxB.lo, Spec.Tensor.sumSpec dxB.hi),
       some (Spec.Tensor.sumSpec d2xB.lo, Spec.Tensor.sumSpec d2xB.hi))
    | _, _ => (none, none)
  -- First/second derivative along Y (dir 1) if available
  let duY : Option (Float × Float) :=
    if hasY then
      let seedY := FlatBox.ofTensor (NN.Tensor.oneHotNat (α := Float) inDim 1)
      let d1y := runDerivDirectional (α:=Float) g ps ibp seedY
      match NN.MLTheory.CROWN.Graph.outputBox? d1y outId with
      | .ok dyB => some (Spec.Tensor.sumSpec dyB.lo, Spec.Tensor.sumSpec dyB.hi)
      | .error _ => none
    else none
  let d2uY : Option (Float × Float) :=
    if hasY then
      let seedY := FlatBox.ofTensor (NN.Tensor.oneHotNat (α := Float) inDim 1)
      let d1y := runDerivDirectional (α:=Float) g ps ibp seedY
      let d2y := runDeriv2D (α:=Float) g ps ibp d1y
      match NN.MLTheory.CROWN.Graph.outputBox? d2y outId with
      | .ok d2yB => some (Spec.Tensor.sumSpec d2yB.lo, Spec.Tensor.sumSpec d2yB.hi)
      | .error _ => none
    else none
  let base : Prims := { u := some (uLo, uHi), duX := duX, duY := duY, d2uX := d2uX, d2uY := d2uY }
  match uMethod with
  | .ibp => pure base
  | .crownFwd =>
    match crownUBoundsForward g ps ibp with
    | some (al, ah) => pure { base with u := some (al, ah) }
    | none => pure base
  | .crownBwd =>
    match crownUBoundsBackward g ps ibp with
    | some (al, ah) => pure { base with u := some (al, ah) }
    | none => pure base

/--
Load optional PyTorch-exported PINN weights for the exploratory CLI.

Malformed files or mismatched input dimensions fall back to the provided built-in graph and
parameters. That permissive behavior is intentional here: `pinn-cli` is an interactive inspection
tool. Stable certificate checkers should reject bad artifacts instead.
-/
def loadWeightsOrDefault
    (weights? : Option String)
    (expectedInputDim : Nat)
    (defaultGraph : Graph)
    (defaultParams : ParamStore Float) :
    IO (Graph × ParamStore Float) := do
  match weights? with
  | none => pure (defaultGraph, defaultParams)
  | some path => do
      try
        let j ← NN.Verification.Json.readJsonFile path
        match Import.PINNPyTorch.loadPinnState j with
        | some sd =>
            if sd.arch.inputDim ≠ expectedInputDim then
              IO.eprintln <|
                (s!"[PINN] Loaded weights expect input dim={sd.arch.inputDim}; using " ++
                  s!"built-in {expectedInputDim}D weights.")
              pure (defaultGraph, defaultParams)
            else
              pure (Import.PINNPyTorch.buildGraph sd, Import.PINNPyTorch.toParamStore sd)
        | none =>
            IO.eprintln <|
              ("[PINN] Weights JSON did not match expected shapes; falling back to " ++
                "built-in weights")
            pure (defaultGraph, defaultParams)
      catch e =>
        IO.eprintln s!"[PINN] Failed to parse weights JSON: {e}; falling back to built-in weights"
        pure (defaultGraph, defaultParams)

/-- User-facing bound method selected by `--method`. -/
inductive Method
  | ibp
  | crownFwd
  | crownBwd

instance : ToString Method :=
  ⟨fun
    | .ibp => "ibp"
    | .crownFwd => "crown-fwd"
    | .crownBwd => "crown-bwd"⟩

/-- Parse the `--method` value accepted by the PINN residual checker. -/
def parseMethodVal (s : String) : Option Method :=
  match s.toLower with
  | "ibp" => some .ibp
  | "crown" => some .crownBwd
  | "crown-fwd" => some .crownFwd
  | "crown-bwd" => some .crownBwd
  | _ => none

/-- Parsed options for the PINN residual CLI. -/
structure Opts where
  /-- Bound propagation method used for the output interval. -/
  method : Method := .ibp
  /-- Runtime backend used for interval evaluation. -/
  backend : Backend := .float
  /-- Optional PyTorch-exported PINN weights JSON. -/
  weights? : Option String := none
  /-- Recursive interval split depth for one-dimensional checks. -/
  splitDepth : Nat := 0

/-- Parse recognized flags and return the remaining positional arguments. -/
def parseFlags (args : List String) : Except String (Opts × List String) := do
  let args := TorchLean.CLI.dropDashDash args
  let (weights?, args) ← TorchLean.CLI.takeFlagValueOnce args "weights"
  let (splitDepth, args) ← TorchLean.CLI.takeNatFlagDefault args "split-depth" 0
  let (method, args) ←
    TorchLean.CLI.takeParsedFlagDefault args "method" "ibp" fun s =>
      match parseMethodVal s with
      | some method => pure method
      | none => throw s!"--method: expected ibp, crown, crown-fwd, or crown-bwd; got `{s}`"
  let (backend, args) ←
    TorchLean.CLI.takeParsedFlagDefault args "backend" "float" fun s =>
      match parseBackendVal s with
      | some backend => pure backend
      | none => throw s!"--backend: expected float; got `{s}`"
  pure ({ method := method, backend := backend, weights? := weights?, splitDepth := splitDepth }, args)

/--
Entry point for the PINN residual-bounding CLI.

This is an interactive tool registered as:
`lake exe verify -- pinn-cli -- ...`

For certificate checking, use:
`lake exe verify -- pinn-cert [NN/Examples/Verification/PINN/pinn_cert.json]`

Run:
`lake exe verify -- pinn-cli -- [--method=ibp|crown-fwd|crown-bwd] [--split-depth=N]
  [--backend=float] [--weights=PATH.json] "<PDE>" x eps`
or (2D):
`lake exe verify -- pinn-cli -- [flags] "<PDE>" x y eps`
-/
def main (args : List String) : IO Unit := do
  let (opts, rest) ←
    match parseFlags args with
    | .ok parsed => pure parsed
    | .error e => throw <| IO.userError e
  let method := opts.method
  let backend := opts.backend
  let weights? := opts.weights?
  let splitDepth := opts.splitDepth
  let uMethod : UBoundsMethod :=
    match method with
    | .ibp => .ibp
    | .crownFwd => .crownFwd
    | .crownBwd => .crownBwd

  let rec split1D (x : Float) (eps : Float) (d : Nat)
    (evalAt : Float → Float → IO (Float × Float)) : IO (Float × Float) := do
    match d with
    | 0 => evalAt x eps
    | Nat.succ d' =>
      let eps' := eps * 0.5
      let (l1, h1) ← split1D (x - eps') eps' d' evalAt
      let (l2, h2) ← split1D (x + eps') eps' d' evalAt
      pure (if l1 < l2 then l1 else l2, if h1 > h2 then h1 else h2)

  let rec split2D (x y eps : Float) (d : Nat)
    (evalAt : Float → Float → Float → IO (Float × Float)) : IO (Float × Float) := do
    match d with
    | 0 => evalAt x y eps
    | Nat.succ d' =>
      let eps' := eps * 0.5
      let (l1, h1) ← split2D (x - eps') (y - eps') eps' d' evalAt
      let (l2, h2) ← split2D (x - eps') (y + eps') eps' d' evalAt
      let (l3, h3) ← split2D (x + eps') (y - eps') eps' d' evalAt
      let (l4, h4) ← split2D (x + eps') (y + eps') eps' d' evalAt
      let lo12 := if l1 < l2 then l1 else l2
      let lo34 := if l3 < l4 then l3 else l4
      let hi12 := if h1 > h2 then h1 else h2
      let hi34 := if h3 > h4 then h3 else h4
      pure (if lo12 < lo34 then lo12 else lo34, if hi12 > hi34 then hi12 else hi34)
  match rest.toArray with
  | #[pdeStr, xStr, epsStr] =>
    match backend with
    | .float =>
      let x? := parseFloat xStr
      let eps? := parseFloat epsStr
      match x?, eps? with
      | some x, some eps => do
        let expr ←
          match parseExpr (fun _ => none) pdeStr with
          | .ok e => pure e
          | .error msg => throw <| IO.userError s!"Parse error: {msg}"
        let (g, baseParams) ← loadWeightsOrDefault weights? 1 buildGraph seedParamsFloat
        let evalAt : Float → Float → IO (Float × Float) :=
          fun xc epsc => do
            let ps := seedInputFloat baseParams xc epsc
            let prims ← computePrimsAt g ps uMethod backend
            match eval prims expr with
            | some (lo, hi) => pure (lo, hi)
            | none => throw <| IO.userError "PDE evaluation failed (insufficient primitives)"
        let (lo0, hi0) ← evalAt x eps
        let (loS, hiS) ←
          if splitDepth = 0 then
            pure (lo0, hi0)
          else
            split1D x eps splitDepth evalAt
        -- Never return a worse interval when splitting: intersect when consistent, otherwise fall
        -- back to hull.
        let loI := if lo0 > loS then lo0 else loS
        let hiI := if hi0 < hiS then hi0 else hiS
        let (lo, hi) :=
          if loI ≤ hiI then
            (loI, hiI)
          else
            (if lo0 < loS then lo0 else loS, if hi0 > hiS then hi0 else hiS)
        IO.println <|
          (s!"PDE='{pdeStr}' at x={x}, eps={eps}, method={method}, " ++
            s!"splitDepth={splitDepth}: residual ∈ [{lo},{hi}]")
      | _, _ =>
        IO.eprintln s!"invalid float input(s): x={xStr}, eps={epsStr}"
  | #[pdeStr, xStr, yStr, epsStr] =>
    match backend with
    | .float =>
      let x? := parseFloat xStr
      let y? := parseFloat yStr
      let eps? := parseFloat epsStr
      match x?, y?, eps? with
      | some x, some y, some eps => do
        let expr ←
          match parseExpr (fun _ => none) pdeStr with
          | .ok e => pure e
          | .error msg => throw <| IO.userError s!"Parse error: {msg}"
        let (g, baseParams) ← loadWeightsOrDefault weights? 2 buildGraph2D seedParamsFloat2D
        let evalAt : Float → Float → Float → IO (Float × Float) :=
          fun xc yc epsc => do
            let ps := seedInputFloat2D baseParams xc yc epsc
            let prims ← computePrimsAt g ps uMethod backend
            match eval prims expr with
            | some (lo, hi) => pure (lo, hi)
            | none => throw <| IO.userError "PDE evaluation failed (insufficient primitives)"
        let (lo0, hi0) ← evalAt x y eps
        let (loS, hiS) ←
          if splitDepth = 0 then
            pure (lo0, hi0)
          else
            split2D x y eps splitDepth evalAt
        let loI := if lo0 > loS then lo0 else loS
        let hiI := if hi0 < hiS then hi0 else hiS
        let (lo, hi) :=
          if loI ≤ hiI then
            (loI, hiI)
          else
            (if lo0 < loS then lo0 else loS, if hi0 > hiS then hi0 else hiS)
        IO.println <|
          (s!"PDE='{pdeStr}' at (x,y)=({x},{y}), eps={eps}, method={method}, " ++
            s!"splitDepth={splitDepth}: residual ∈ [{lo},{hi}]")
      | _, _, _ =>
        IO.eprintln s!"invalid float input(s): x={xStr}, y={yStr}, eps={epsStr}"
  | _ =>
    throw <| IO.userError <|
      ("Usage:\n  lake exe verify -- pinn-cli -- " ++
        "[--method=ibp|crown-fwd|crown-bwd] [--split-depth=N] " ++
        "[--backend=float] [--weights=path.json] \"<PDE>\" x eps   " ++
        " # 1D\n  lake exe verify -- pinn-cli -- " ++
        "[--method=ibp|crown-fwd|crown-bwd] [--split-depth=N] " ++
        "[--backend=float] [--weights=path.json] \"<PDE>\" x y eps " ++
        " # 2D")

end NN.Verification.PINN.CLI
