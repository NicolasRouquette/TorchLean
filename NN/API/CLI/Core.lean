/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Lean.Data.Json.Parser

/-!
# TorchLean CLI Parsers

Pure command-line parsers shared by examples, verification tools, and application API helpers. The
definitions live directly under `TorchLean.CLI`; `NN.API.CLI` is the lightweight import path.

This module stays independent of tensors and runtime modules so lightweight artifact checkers can
reuse the CLI surface without importing the full public API.
-/

@[expose] public section

namespace TorchLean
namespace CLI

/-- Lift a shared CLI parser result into `IO.userError`. -/
def orThrowIO {α : Type} (x : Except String α) : IO α :=
  match x with
  | .ok value => pure value
  | .error e => throw <| IO.userError e

/--
Strip at most one occurrence of a `--key` flag from an argument list.

Accepted forms:
- `--key value`
- `--key=value`
-/
def takeFlagValueOnce (args : List String) (key : String) :
    Except String (Option String × List String) :=
  let eqPrefix := s!"--{key}="
  let keyTok := s!"--{key}"
  let rec go :
      List String → Option String → List String → Except String (Option String × List String)
    | [], found, acc => .ok (found, acc.reverse)
    | a :: rest, found, acc =>
        if a == keyTok then
          match rest with
          | [] => .error s!"{keyTok}: expected a value"
          | v :: rest' =>
              if found.isSome then
                .error s!"{keyTok}: duplicate flag"
              else
                go rest' (some v) acc
        else if a.startsWith eqPrefix then
          let v := (a.drop eqPrefix.length).toString
          if found.isSome then
            .error s!"{keyTok}: duplicate flag"
          else
            go rest (some v) acc
        else
          go rest found (a :: acc)
  go args none []

/--
Parse an optional string-valued flag and fall back to a provided default.

Use this when a command parser wants a concrete string immediately rather than an optional override.
-/
def takeFlagValueDefault
    (args : List String)
    (key : String)
    (default : String) :
    Except String (String × List String) := do
  let (value?, rest) ← takeFlagValueOnce args key
  pure (value?.getD default, rest)

/-- Parse a required string-valued flag and return the remaining arguments. -/
def takeRequiredFlagValue
    (args : List String)
    (key : String)
    (missing? : Option String := none) :
    Except String (String × List String) := do
  let (value?, rest) ← takeFlagValueOnce args key
  match value? with
  | some value => pure (value, rest)
  | none => throw <| missing?.getD s!"missing --{key}=<value>"

/--
Look up a string-valued flag without returning the remaining arguments.

This is useful for command shapes that support optional overrides but do not otherwise need a
left-to-right consuming parser. Accepted forms are the same as `takeFlagValueOnce`:
`--key value` and `--key=value`.
-/
def flagValue? (args : List String) (key : String) : Except String (Option String) := do
  let (value?, _) ← takeFlagValueOnce args key
  pure value?

/-- Require a string-valued flag, accepting both `--key value` and `--key=value`. -/
def requireFlagValue (args : List String) (key : String) : Except String String := do
  match ← flagValue? args key with
  | some value => pure value
  | none => throw s!"missing --{key}=<value>"

/--
Parse an optional string-valued flag, fall back to a provided default spelling when absent, and
decode the selected spelling with a caller-supplied parser.

This is useful for enum-like CLI flags whose valid strings remain command-specific.
-/
def takeParsedFlagDefault
    {α : Type}
    (args : List String)
    (key : String)
    (default : String)
    (parse : String → Except String α) :
    Except String (α × List String) := do
  let (raw, rest) ← takeFlagValueDefault args key default
  let value ← parse raw
  pure (value, rest)

/-- Return true when `args` contains `--key value` or `--key=value`. -/
def hasFlagValue (args : List String) (key : String) : Bool :=
  let eqPrefix := s!"--{key}="
  let keyTok := s!"--{key}"
  args.any (fun a => a == keyTok || a.startsWith eqPrefix)

/--
Remove every occurrence of a string-valued flag, accepting both `--key value` and `--key=value`.

This is for wrapper commands that own a flag locally and forward the remaining arguments to another
tool. It deliberately does not reject duplicates; the wrapper's local parser decides whether a
duplicated flag is an error.
-/
def stripFlagValues (args : List String) (keys : List String) : List String :=
  let rec go : List String → List String
    | [] => []
    | a :: rest =>
        let matchesEq := keys.any (fun key => a.startsWith s!"--{key}=")
        if matchesEq then
          go rest
        else if keys.any (fun key => a == s!"--{key}") then
          match rest with
          | [] => []
          | _value :: rest' => go rest'
        else
          a :: go rest
  go args

/-- Remove a no-value boolean flag once, returning whether it appeared. -/
partial def takeBoolFlagOnce (args : List String) (key : String) :
    Except String (Bool × List String) := do
  let keyTok := s!"--{key}"
  let rec go : List String → Bool → List String → Except String (Bool × List String)
    | [], seen, acc => pure (seen, acc.reverse)
    | a :: rest, seen, acc =>
        if a == keyTok then
          if seen then
            throw s!"{keyTok}: duplicate flag"
          else
            go rest true acc
        else
          go rest seen (a :: acc)
  go args false []

/-- Drop the leading `--` separator commonly used with `lean --run`. -/
def dropDashDash (args : List String) : List String :=
  match args with
  | "--" :: rest => rest
  | xs => xs

/-- Return true when the argument list requests command help. -/
def hasHelp (args : List String) : Bool :=
  args.contains "--help" || args.contains "-h"

/-- Fail if there are any unconsumed CLI arguments. -/
def checkNoArgs (args : List String) : Except String Unit :=
  if args.isEmpty then
    .ok ()
  else
    .error s!"unexpected arguments: {args}"

/--
Take at most one positional argument, leaving flags untouched.

This is useful for commands with a single optional artifact path plus named flags. A second
positional argument is reported as an error instead of being silently ignored.
-/
def takePositionalOnce (args : List String) :
    Except String (Option String × List String) :=
  let rec go :
      List String → Option String → List String → Except String (Option String × List String)
    | [], found, acc => .ok (found, acc.reverse)
    | a :: rest, found, acc =>
        if a.startsWith "--" then
          go rest found (a :: acc)
        else if found.isSome then
          .error s!"unexpected positional argument: {a}"
        else
          go rest (some a) acc
  go args none []

/-- Take one optional positional argument and fall back to `default` when it is absent. -/
def takePositionalDefault (args : List String) (default : String) :
    Except String (String × List String) := do
  let (value?, rest) ← takePositionalOnce args
  pure (value?.getD default, rest)

/--
Normalize commands that accept either a positional path or a named path flag.

If `--key` / `--key=...` is already present, the argument list is returned unchanged. Otherwise the
first positional argument is rewritten to `--key=<path>`. If there is no positional path, the
provided default path is inserted.
-/
def defaultPathFlagFromPositional
    (args : List String)
    (key default : String) :
    List String :=
  let args := dropDashDash args
  if hasFlagValue args key then
    args
  else
    match args with
    | [] => [s!"--{key}={default}"]
    | a :: rest =>
        if a.startsWith "--" then
          s!"--{key}={default}" :: a :: rest
        else
          s!"--{key}={a}" :: rest

/-- Like `takeFlagValueOnce`, but parse the value as a `Nat`. -/
def takeNatFlagOnce (args : List String) (key : String) :
    Except String (Option Nat × List String) := do
  let (value?, rest) ← takeFlagValueOnce args key
  match value? with
  | none => pure (none, rest)
  | some value =>
      match value.toNat? with
      | some n => pure (some n, rest)
      | none => throw s!"--{key}: expected a natural number, got `{value}`"

/--
Parse an optional natural-number flag and fall back to the provided default.
-/
def takeNatFlagDefault
    (args : List String)
    (key : String)
    (default : Nat) :
    Except String (Nat × List String) := do
  let (value?, rest) ← takeNatFlagOnce args key
  pure (value?.getD default, rest)

/--
Parse an optional natural-number flag, fall back to a default, and require that the selected value
is strictly positive.
-/
def takePositiveNatFlagDefault
    (args : List String)
    (exeName : String)
    (key : String)
    (default : Nat) :
    Except String (Nat × List String) := do
  let (value, rest) ← takeNatFlagDefault args key default
  if value = 0 then
    throw s!"{exeName}: --{key} must be > 0"
  pure (value, rest)

/-- Parse one ASCII decimal digit. -/
def parseDecimalDigit? (c : Char) : Option Nat :=
  if '0' ≤ c ∧ c ≤ '9' then
    some (c.toNat - '0'.toNat)
  else
    none

/-- Parse a nonempty string of ASCII decimal digits. -/
def parseNatDigits? (s : String) : Option Nat :=
  if s.isEmpty then
    none
  else
    s.toList.foldl
      (fun acc? c =>
        match acc?, parseDecimalDigit? c with
        | some acc, some d => some (acc * 10 + d)
        | _, _ => none)
      (some 0)

/--
Parse a signed decimal float literal.

The primary path accepts the same numeric syntax as `Lean.Json`, including scientific notation. The
fallback accepts the CLI-friendly decimal form `1.`.
-/
def parseFloatLit (s : String) : Option Float :=
  match Lean.Json.parse s with
  | Except.ok (.num n) => some n.toFloat
  | _ =>
      let (neg, body) :=
        if s.startsWith "-" then
          (true, (s.drop 1).toString)
        else
          (false, s)
      match body.splitOn "." with
      | [intTxt, fracTxt] =>
          match parseNatDigits? intTxt with
          | none => none
          | some intVal =>
              let fracVal? :=
                if fracTxt.isEmpty then
                  some 0.0
                else
                  match parseNatDigits? fracTxt with
                  | some fracVal =>
                      some (Float.ofNat fracVal / Float.ofNat (Nat.pow 10 fracTxt.length))
                  | none => none
              match fracVal? with
              | some fracVal =>
                  let v := Float.ofNat intVal + fracVal
                  some (if neg then -v else v)
              | none => none
      | _ => none

/-- Like `takeFlagValueOnce`, but parse the value as a `Float`. -/
def takeFloatFlagOnce (args : List String) (key : String) :
    Except String (Option Float × List String) := do
  let (value?, rest) ← takeFlagValueOnce args key
  match value? with
  | none => pure (none, rest)
  | some value =>
      match parseFloatLit value with
      | some x => pure (some x, rest)
      | none => throw s!"--{key}: expected a float literal, got `{value}`"

/--
Parse an optional floating-point flag and fall back to the provided default.
-/
def takeFloatFlagDefault
    (args : List String)
    (key : String)
    (default : Float) :
    Except String (Float × List String) := do
  let (value?, rest) ← takeFloatFlagOnce args key
  pure (value?.getD default, rest)

/-- Parse a required floating-point flag and return the remaining arguments. -/
def takeRequiredFloatFlag
    (args : List String)
    (key : String)
    (missing? : Option String := none) :
    Except String (Float × List String) := do
  let (value?, rest) ← takeFloatFlagOnce args key
  match value? with
  | some value => pure (value, rest)
  | none => throw <| missing?.getD s!"missing --{key}=<float>"

/-- Parse a CLI boolean value. Accepted spellings are `true`, `false`, `1`, and `0`. -/
def parseBoolLit (s : String) : Option Bool :=
  match s.toLower with
  | "true" => some true
  | "1" => some true
  | "false" => some false
  | "0" => some false
  | _ => none

/-- Like `takeFlagValueOnce`, but parse the value as a boolean. -/
def takeBoolValueFlagOnce (args : List String) (key : String) :
    Except String (Option Bool × List String) := do
  let (value?, rest) ← takeFlagValueOnce args key
  match value? with
  | none => pure (none, rest)
  | some value =>
      match parseBoolLit value with
      | some b => pure (some b, rest)
      | none => throw s!"--{key}: expected true, false, 1, or 0; got `{value}`"

/-- Parse an optional boolean-valued flag and fall back to the provided default. -/
def takeBoolValueFlagDefault (args : List String) (key : String) (default : Bool) :
    Except String (Bool × List String) := do
  let (value?, rest) ← takeBoolValueFlagOnce args key
  pure (value?.getD default, rest)

/--
Remove a boolean flag that may be written either as a bare switch or with an explicit value.

Accepted forms:
- `--key`
- `--key=true`
- `--key=false`
- `--key true`
- `--key false`

When `--key` is followed by a non-boolean token, the flag is treated as a bare switch and the next
token is left for the caller. Duplicate occurrences are rejected.
-/
partial def takeBoolFlagOptionalValueOnce (args : List String) (key : String) :
    Except String (Option Bool × List String) := do
  let keyTok := s!"--{key}"
  let eqPrefix := s!"--{key}="
  let rec go (args : List String) (seen : Option Bool) (acc : List String) :
      Except String (Option Bool × List String) := do
    match args with
    | [] => pure (seen, acc.reverse)
    | a :: rest =>
        if a == keyTok then
          if seen.isSome then
            throw s!"{keyTok}: duplicate flag"
          else
            match rest with
            | v :: rest' =>
                match parseBoolLit v with
                | some b => go rest' (some b) acc
                | none => go rest (some true) acc
            | [] => go rest (some true) acc
        else if a.startsWith eqPrefix then
          if seen.isSome then
            throw s!"{keyTok}: duplicate flag"
          else
            let raw := (a.drop eqPrefix.length).toString
            match parseBoolLit raw with
            | some b => go rest (some b) acc
            | none => throw s!"{keyTok}: expected true, false, 1, or 0; got `{raw}`"
        else
          go rest seen (a :: acc)
  go args none []

/-- Parse a bare-or-valued boolean flag and fall back to the provided default. -/
def takeBoolFlagOptionalValueDefault (args : List String) (key : String) (default : Bool) :
    Except String (Bool × List String) := do
  let (value?, rest) ← takeBoolFlagOptionalValueOnce args key
  pure (value?.getD default, rest)

/--
Parse an optional floating-point flag, fall back to the provided default, and require that the
selected value is strictly positive.
-/
def takePositiveFloatFlagDefault
    (args : List String)
    (exeName : String)
    (key : String)
    (default : Float) :
    Except String (Float × List String) := do
  let (value, rest) ← takeFloatFlagDefault args key default
  if value <= 0.0 then
    throw s!"{exeName}: --{key} must be > 0"
  pure (value, rest)

/--
Parse an optional floating-point flag, fall back to the provided default, and require that the
selected value is nonnegative.
-/
def takeNonnegativeFloatFlagDefault
    (args : List String)
    (exeName : String)
    (key : String)
    (default : Float) :
    Except String (Float × List String) := do
  let (value, rest) ← takeFloatFlagDefault args key default
  if value < 0.0 then
    throw s!"{exeName}: --{key} must be >= 0"
  pure (value, rest)

/-- Like `takeFlagValueOnce`, but return the value as a `System.FilePath`. -/
def takePathFlagOnce (args : List String) (key : String) :
    Except String (Option System.FilePath × List String) := do
  let (value?, rest) ← takeFlagValueOnce args key
  pure (value?.map (fun s => (s : System.FilePath)), rest)

/--
Parse an optional path flag and fall back to the provided default path.

Use this when an example parser wants a concrete path immediately instead of an optional override.
-/
def takePathFlagDefault
    (args : List String)
    (key : String)
    (default : System.FilePath) :
    Except String (System.FilePath × List String) := do
  let (path?, rest) ← takePathFlagOnce args key
  pure (path?.getD default, rest)

/--
Parse a required path flag such as `--data-file corpus.txt`.

The error message includes `exeName` when provided.
-/
def takeRequiredPathFlag
    (args : List String)
    (key : String)
    (exeName : String := "") :
    Except String (System.FilePath × List String) := do
  let (path?, rest) ← takePathFlagOnce args key
  match path? with
  | some path => pure (path, rest)
  | none => do
      let pfx := if exeName.isEmpty then "" else s!"{exeName}: "
      throw s!"{pfx}missing required --{key} <path>"

/--
Parse two optional path flags that must appear together if either one is present.

This is useful for paired artifacts such as tokenizer vocab/merge files, where a single path is not
meaningful on its own.
-/
def takePairedPathFlags
    (args : List String)
    (firstKey secondKey : String) :
    Except String ((Option System.FilePath × Option System.FilePath) × List String) := do
  let (firstPath?, args) ← takePathFlagOnce args firstKey
  let (secondPath?, args) ← takePathFlagOnce args secondKey
  match firstPath?, secondPath? with
  | some _, none => throw s!"--{firstKey} requires --{secondKey}"
  | none, some _ => throw s!"--{secondKey} requires --{firstKey}"
  | _, _ => pure ((firstPath?, secondPath?), args)

/--
Common training flags for epoch-oriented loader/tutorial commands: `--epochs` and `--batch`.
-/
structure EpochBatch where
  /-- Number of epochs to train for. -/
  epochs : Nat
  /-- Batch size. -/
  batch : Nat

/--
Parse optional `--epochs` and `--batch` flags for the epoch-oriented tutorial helpers.

Returns the parsed values (falling back to the provided defaults) and the remaining args.
-/
def takeEpochBatch (args : List String) (defaultEpochs defaultBatch : Nat) :
    Except String (EpochBatch × List String) := do
  let (epochs?, args) ← takeNatFlagOnce args "epochs"
  let (batch?, args) ← takeNatFlagOnce args "batch"
  pure ({ epochs := epochs?.getD defaultEpochs, batch := batch?.getD defaultBatch }, args)

/--
Parse optional `--epochs` and `--batch` flags for the epoch-oriented tutorial helpers, fall back to
the provided defaults, and require that both selected values are strictly positive.
-/
def takePositiveEpochBatch
    (args : List String)
    (exeName : String)
    (defaultEpochs defaultBatch : Nat) :
    Except String (EpochBatch × List String) := do
  let (eb, args) ← takeEpochBatch args defaultEpochs defaultBatch
  if eb.epochs = 0 then
    throw s!"{exeName}: --epochs must be > 0"
  if eb.batch = 0 then
    throw s!"{exeName}: --batch must be > 0"
  pure (eb, args)

/--
Parse an optional `--steps` flag and fall back to the provided default.
-/
def takeStepsFlagDefault (args : List String) (default : Nat) : Except String (Nat × List String) := do
  let (steps?, args) ← takeNatFlagOnce args "steps"
  pure (steps?.getD default, args)

/-- Parse an optional `--seed` flag (defaults to the provided value). -/
def takeSeed (args : List String) (default : Nat := 0) :
    Except String (Nat × List String) := do
  let (seed?, rest) ← takeNatFlagOnce args "seed"
  pure (seed?.getD default, rest)

end CLI
end TorchLean
