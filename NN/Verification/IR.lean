/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.IR
public import NN.Runtime.PyTorch.Export.TorchExport

/-!
# Checking captured tensor computations

This entry point checks a typed IR artifact and optionally executes supplied
arguments or compares supplied reference outputs. The result states the scope of
each check. Passing these checks is not a theorem about all inputs, a proof of the
frontend translation, or a validation of external kernels.

Unknown requests are reported as unsupported. They must not disappear from a
successful report. Existing certificate checkers and mathematical proof APIs
remain separate tools with their own contracts.
-/

@[expose] public section

namespace NN.Verification.IR

open Lean NN.IR

namespace Internal

/-- One named check with its outcome and an explanation of its scope. -/
def finding (name status message : String) : Json :=
  Json.mkObj [("name", toJson name), ("status", toJson status), ("message", toJson message)]

/-- Read a supported tensor from complete element words, with no host-float conversion. -/
def value (json : Json) : Except String (Interpreter.Value Interpreter.reference) := do
  let dtype ← DType.fromJson (← json.getObjVal? "dtype")
  let shape ← Import.PyTorch.TorchExport.Internal.shape "tensor.shape"
    (← json.getObjVal? "shape")
  let words ← Import.PyTorch.TorchExport.Internal.nats "tensor.bits"
    (← json.getObjVal? "bits")
  Reference.load dtype shape words

/-- Element type, shape, and complete words all participate in a sample comparison. -/
def compare (actual reference : Array (Interpreter.Value Interpreter.reference)) :
    Except String Unit := do
  unless actual.size == reference.size do
    throw "IR comparison: output count differs"
  for i in [:actual.size] do
    let some output := actual[i]? | throw "IR comparison: missing output"
    let some expected := reference[i]? | throw "IR comparison: missing reference"
    unless output.dtype == expected.dtype && output.shape == expected.shape do
      throw s!"IR comparison: result {i} has a different element type or shape"
    unless (← Reference.words output) == (← Reference.words expected) do
      throw s!"IR comparison: result {i} differs in its interchange words"

/-- Serialize computed results for inspection without a lossy numerical display conversion. -/
def result (value : Interpreter.Value Interpreter.reference) : Except String Json := do
  pure <| Json.mkObj [
    ("element_type", toJson value.dtype.describe), ("dtype", toJson value.dtype),
    ("shape", toJson value.shape), ("bits", toJson (← Reference.words value))]

end Internal

/-- Run requested checks, returning explicit failures and unsupported requests as report entries.

`execute` and `compare` concern only `arguments` stored in the artifact. A comparison
uses complete binary words and can expose legitimate differences in reduction or
elementary-function implementations; a mismatch alone does not identify a bug.
-/
def check (artifact : Json) (requests : Array String := #["structure", "types"]) : Json := Id.run do
  let mut findings : Array Json := #[]
  let mut ok := true
  let mut outputs : Json := .null
  let mut conversionFlags : Json := .null
  let parsed := Import.IR.parseProgram artifact
  let execution : Except String _ := do
    unless requests.contains "execute" || requests.contains "compare" do
      throw "Execution was not requested."
    let program ← parsed
    let payload ← Import.IR.parsePayload Interpreter.reference Reference.load artifact
    let arguments ← (← (← artifact.getObjVal? "arguments").getArr?).mapM Internal.value
    Interpreter.reference.eval program payload arguments
  if let .ok (values, flags) := execution then
    match values.mapM Internal.result with
    | .ok results => outputs := toJson results
    | .error _ => pure ()
    conversionFlags := Json.mkObj [
      ("inexact", toJson flags.inexact), ("overflow", toJson flags.overflow),
      ("underflow", toJson flags.underflow), ("invalid", toJson flags.invalid)]
  for request in requests do
    let (passed, status, message) :=
      match request with
      | "structure" | "types" =>
          match parsed with
          | .ok _ =>
              (true, "checked", "Declared dependencies, shapes, types and casts are consistent.")
          | .error message => (false, "failed", message)
      | "execute" | "compare" =>
          match execution with
          | .error message => (false, "failed", message)
          | .ok (values, _) =>
              if request == "compare" then
                let comparison : Except String Unit := do
                  let reference ← (← (← artifact.getObjVal? "reference_outputs").getArr?).mapM
                    Internal.value
                  Internal.compare values reference
                match comparison with
                | .ok _ =>
                    (true, "compared", "Interchange words agree for the supplied arguments only.")
                | .error message => (false, "failed", message)
              else
                (true, "executed", "CPU reference execution completed for the supplied arguments.")
      | _ =>
          (false, "unsupported", s!"No checker for `{request}`; no proof was produced.")
    ok := ok && passed
    findings := findings.push (Internal.finding request status message)
  if requests.isEmpty then
    ok := false
    findings := #[Internal.finding "request" "failed" "At least one check must be requested."]
  return Json.mkObj [
    ("format", toJson "torchlean.verification.v1"), ("ok", toJson ok),
    ("checks", toJson findings), ("outputs", outputs), ("conversion_flags", conversionFlags),
    ("frontend_equivalence", toJson "not_checked"),
    ("frontend_device_layout_constraints", toJson "not_checked"),
    ("external_kernels", toJson "not_checked")]

/-- CLI: check an artifact as JSON, or emit the existing PyTorch capture adapter. -/
def run (arguments : List String) : IO Unit := do
  match arguments with
  | ["--export-pytorch", path] =>
      IO.FS.writeFile path Export.PyTorch.TorchExport.script
  | path :: requests =>
      let artifact ← IO.ofExcept (Json.parse (← IO.FS.readFile path))
      let requests := if requests.isEmpty then #["structure", "types"] else requests.toArray
      let report := check artifact requests
      IO.println report.compress
      unless (← IO.ofExcept ((← IO.ofExcept (report.getObjVal? "ok")).getBool?)) do
        throw (IO.userError "one or more IR checks failed; see the JSON report")
  | [] => throw (IO.userError "usage: verify ir <artifact.json> [structure|types|execute|compare]")

end NN.Verification.IR
