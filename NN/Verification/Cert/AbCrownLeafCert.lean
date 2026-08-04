/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.CLI
public import NN.Verification.Util.Array
public import NN.Verification.Util.FloatApprox
public import NN.Verification.Util.Json

/-!
# AbCrown Leaf Artifact

Alpha-beta-CROWN (AbCrown) leaf-artifact checker.

This module checks a small TorchLean JSON schema (`abcrown_leaf_artifact_v0_1`). Vanilla
alpha-beta-CROWN does not emit this TorchLean schema directly; use
`scripts/verification/abcrown/export_leaf_artifact.py` to convert terminal leaf/domain data from an
external verifier into the checked schema.

The checker does **not** run bound propagation itself. It validates only the finite claims present
in the artifact:
- each leaf input box is nested inside the declared root input box, and
- each leaf contains a witness that refutes the unsafe threshold
  ($\mathrm{lb}[i]>\mathrm{threshold}[i]$ for some $i$).

This is useful for:
- regression testing JSON export/import paths, and
- reviewer-friendly validation of the leaf data that TorchLean actually checks.

References:
- beta-CROWN paper (NeurIPS 2021): `https://arxiv.org/abs/2103.06624`
- alpha-beta-CROWN implementation: `https://github.com/Verified-Intelligence/alpha-beta-CROWN`

Run:
`lake exe verify -- abcrown-leaf [path/to/artifact.json]`
-/

@[expose] public section


namespace NN.Verification.Cert.AbCrownLeafCert

open Lean
open Data
open NN.Verification.Json

/-- Bundled sample alpha-beta-CROWN-style leaf artifact. -/
def defaultArtifactPath : String :=
  "NN/Examples/Verification/AbCrown/sample_abcrown_leaf_artifact_v0_1.json"

/--
Parse and validate a `abcrown_leaf_artifact_v0_1` JSON artifact.

On failure this throws `IO.userError` with a brief message.
-/
def checkAbCrownLeafArtifact (path : String) : IO Unit := do
  let topObj ← readJsonObjectFile path
  expectFormat topObj "abcrown_leaf_artifact_v0_1"
  let inputDim ← expectFieldNat topObj "input_dim" "top-level"

  let rootObj ← expectFieldObj topObj "root" "top-level"
  let root ← fromExcept <| expectEndpointBoxRegionE "root" rootObj
  if root.dim ≠ inputDim then
    throw <| IO.userError
      s!"root dimension mismatch: input_dim={inputDim}, endpoints={root.dim}"

  let leaves ← expectFieldArray topObj "leaves" "top-level"
  if leaves.isEmpty then
    throw <| IO.userError "invalid leaf artifact: leaves must be nonempty"

  let mut okCount := 0
  let mut badCount := 0
  for leaf in leaves do
    let leafObj ← expectObj leaf "leaf"
    let region ← fromExcept <| expectEndpointBoxRegionE "leaf" leafObj
    let lb ← expectFieldFiniteFloatArray leafObj "lb" "leaf"
    let thr ← expectFieldFiniteFloatArray leafObj "threshold" "leaf"
    if region.dim ≠ inputDim then
      throw <| IO.userError
        s!"leaf dimension mismatch: input_dim={inputDim}, endpoints={region.dim}"
    if lb.size ≠ thr.size then
      throw <| IO.userError
        s!"leaf lower-bound/threshold length mismatch: lb={lb.size}, threshold={thr.size}"

    let within :=
      NN.Verification.Util.Array.boxWithin root.lo root.hi region.lo region.hi
    let witnessIdx? ← optionalFieldNat? leafObj "witness_idx" "leaf"
    let witnessMargin? ← optionalFieldFiniteFloat? leafObj "witness_margin" "leaf"
    let verified :=
      match witnessIdx? with
      | some wi => NN.Verification.Util.Array.refutesThresholdAt lb thr wi
      | none => NN.Verification.Util.Array.refutesThreshold lb thr
    let marginMatches :=
      match witnessIdx?, witnessMargin? with
      | some wi, some claimedMargin =>
          if hLb : wi < lb.size then
            if hThr : wi < thr.size then
              let actualMargin := lb[wi]'hLb - thr[wi]'hThr
              NN.Verification.Util.approxEq actualMargin claimedMargin (tol := 1e-6)
            else false
          else false
      | none, some _ => false
      | _, none => true
    if within && verified && marginMatches then
      okCount := okCount + 1
    else
      badCount := badCount + 1

  IO.println s!"[artifact] Checked {leaves.size} leaves: ok={okCount}, bad={badCount}"
  if badCount > 0 then
    throw <| IO.userError s!"Artifact failed checks for {badCount} leaves"

/--
CLI entry point: `lake exe verify -- abcrown-leaf [artifact.json]`.

If no path is provided, checks a small bundled sample artifact under
`NN/Examples/Verification/AbCrown/`.
-/
def run (args : List String) : IO Unit := do
  let usage :=
    String.intercalate "\n" [
      "Usage:",
      "  lake exe verify -- abcrown-leaf [<path/to/artifact.json>]",
      "",
      "If no path is provided, runs a small bundled sample artifact:",
      s!"  {defaultArtifactPath}"
    ]

  if TorchLean.CLI.hasHelp args then
    IO.println usage
    return

  let args := TorchLean.CLI.dropDashDash args
  let (path, rest) ←
    match TorchLean.CLI.takePositionalDefault args defaultArtifactPath with
    | .ok result => pure result
    | .error e => throw <| IO.userError s!"{e}\n\n{usage}"
  match TorchLean.CLI.checkNoArgs rest with
  | .ok () => pure ()
  | .error e => throw <| IO.userError s!"{e}\n\n{usage}"
  checkAbCrownLeafArtifact path

end NN.Verification.Cert.AbCrownLeafCert
