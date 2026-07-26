/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.CLI
public import NN.Verification.Robustness.TopLabel
public import NN.Verification.Util.Array
public import NN.Verification.Util.Json

/-!
# Margin Certificate Checker

Reusable checker for per-example logit-margin certificates (`robust_margin_cert_v0_1`).

The checker reads exported output bounds and recomputes the strict top-label margin:

$\mathrm{logits}_{\mathrm{hi}}[j]<\mathrm{logits}_{\mathrm{lo}}[\mathrm{label}]$ for every
$j\neq\mathrm{label}$.
-/

@[expose] public section

namespace NN
namespace Verification
namespace Robustness
namespace MarginCert

open Lean
open Data
open NN.API
open NN.Verification.Json

/-- Certificate format tag expected at the top level of margin certificate JSON files. -/
def formatTag : String := "robust_margin_cert_v0_1"

/-- Running counters for nominal and certified accuracy reports. -/
structure Counters where
  /-- Number of examples checked. -/
  total : Nat := 0
  /-- Number of examples whose optional nominal prediction equals the label. -/
  nominalOk : Nat := 0
  /-- Number of examples certified by the margin predicate. -/
  certifiedOk : Nat := 0
deriving Repr

namespace Counters

/-- Add one `(nominalOk, certifiedOk)` outcome to the report counters. -/
def push (counts : Counters) (nominalOk cert : Bool) : Counters :=
  { counts with
    total := counts.total + 1
    nominalOk := counts.nominalOk + (if nominalOk then 1 else 0)
    certifiedOk := counts.certifiedOk + (if cert then 1 else 0) }

end Counters

/-- Check one JSON example object and return `(nominalOk, certifiedOk)`. -/
def checkOneExample (numClasses : Nat) (ex : Json) : IO (Bool × Bool) := do
  let exObj ← expectObj ex "example"
  let label ← expectFieldNat exObj "label" "example"
  let lo ← expectFieldFiniteFloatArray exObj "logits_lo" "example"
  let hi ← expectFieldFiniteFloatArray exObj "logits_hi" "example"
  if lo.size ≠ numClasses || hi.size ≠ numClasses then
    throw <| IO.userError s!"example logits length mismatch (expected {numClasses})"
  if !allPairwise lo hi NN.Verification.Util.Array.floatLe then
    throw <| IO.userError "example has invalid bounds (lo ≤ hi violated)"
  let cert := TopLabel.certifiesLabelFromArrayBounds lo hi label

  match ← optionalFieldBool? exObj "certified" "example" with
  | some b =>
      if b != cert then
        throw <| IO.userError "example.certified does not match margin predicate"
  | none => pure ()

  let nominalOk :=
    match ← optionalFieldNat? exObj "pred" "example" with
    | some p => decide (p = label)
    | none => false

  pure (nominalOk, cert)

/--
Check a `robust_margin_cert_v0_1` JSON certificate file.

If `timing = true`, prints per-example timings every `timingEvery` examples.
-/
def checkWithTiming (path : String) (timing : Bool) (timingEvery : Nat) : IO Unit := do
  let topObj ← readJsonObjectFile path
  expectFormat topObj formatTag
  let numClasses ← expectFieldNat topObj "num_classes" "top-level"
  let examples ← expectFieldArray topObj "examples" "top-level"

  let timeMs {α : Type} (act : IO α) : IO (α × Float) := do
    let t0 ← IO.monoNanosNow
    let a ← act
    let t1 ← IO.monoNanosNow
    let ms := (t1 - t0).toFloat / 1_000_000.0
    pure (a, ms)

  let mut counts : Counters := {}
  let mut totalMs : Float := 0.0
  let mut maxMs : Float := 0.0
  for ex in examples do
    if timing then
      let ((nominalOk, cert), ms) ← timeMs (checkOneExample numClasses ex)
      counts := counts.push nominalOk cert
      totalMs := totalMs + ms
      if ms > maxMs then
        maxMs := ms
      if timingEvery > 0 && counts.total % timingEvery == 0 then
        IO.println s!"[margin cert] example {counts.total}: {ms} ms"
    else
      let (nominalOk, cert) ← checkOneExample numClasses ex
      counts := counts.push nominalOk cert

  IO.println s!"[margin cert] examples={counts.total}"
  IO.println s!"[margin cert] nominal_ok={counts.nominalOk} (requires 'pred' in examples)"
  IO.println s!"[margin cert] certified_ok={counts.certifiedOk}"
  if timing then
    let avgMs := if counts.total == 0 then 0.0 else totalMs / counts.total.toFloat
    IO.println s!"[margin cert] timing avg_ms={avgMs} max_ms={maxMs}"

  match ← optionalField? topObj "summary" "top-level" with
  | none => pure ()
  | some summaryJ =>
      let summaryObj ← expectObj summaryJ "summary"
      let checkNatField (k : String) (v : Nat) : IO Unit := do
        match ← optionalFieldNat? summaryObj k "summary" with
        | none => pure ()
        | some n =>
            if n != v then
              throw <| IO.userError s!"summary.{k} mismatch (expected {v}, got {n})"
      checkNatField "examples" counts.total
      checkNatField "nominal_ok" counts.nominalOk
      checkNatField "certified_ok" counts.certifiedOk

/-- Check a margin certificate file with timing disabled. -/
def check (path : String) : IO Unit :=
  checkWithTiming path false 0

/-- Parsed CLI flags for a margin-certificate run. -/
structure RunArgs where
  /-- Certificate JSON path. -/
  path : String
  /-- Print per-example checker timings. -/
  timing : Bool := false
  /-- Print every `timingEvery` examples when timing is enabled; `0` disables periodic lines. -/
  timingEvery : Nat := 0

/-- Parse shared margin-certificate CLI flags. -/
def parseRunArgs (defaultPath : String) (args : List String) : Except String RunArgs := do
  let args := TorchLean.CLI.dropDashDash args
  let (timing, args) ← TorchLean.CLI.takeBoolFlagOnce args "timing"
  let (timingEvery, args) ← TorchLean.CLI.takeNatFlagDefault args "timing-every" 0
  let (path, args) ← TorchLean.CLI.takePositionalDefault args defaultPath
  TorchLean.CLI.checkNoArgs args
  pure { path := path, timing := timing, timingEvery := timingEvery }

/-- Run the checker with a caller-provided default certificate path. -/
def runWithDefault (defaultPath : String) (args : List String) : IO Unit := do
  let parsed ←
    match parseRunArgs defaultPath args with
    | .ok parsed => pure parsed
    | .error err => throw <| IO.userError err
  checkWithTiming parsed.path parsed.timing parsed.timingEvery

end MarginCert
end Robustness
end Verification
end NN
