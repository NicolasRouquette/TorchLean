/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.CLI.Parser

/-!
# Command Process Boundaries

Helpers for reporting pure parser failures through an executable's `IO` boundary.
-/

@[expose] public section

namespace TorchLean.CLI

/-- Lift a shared CLI parser result into `IO.userError`. -/
def orThrowIO {α : Type} (x : Except String α) : IO α :=
  match x with
  | .ok value => pure value
  | .error message => throw <| IO.userError message

/-- Lift a parser result into `IO`, prefixing failures with the executable name. -/
def orThrow {α : Type} (exeName : String) : Except String α → IO α
  | .ok value => pure value
  | .error message => throw <| IO.userError s!"{exeName}: {message}"

/-- Parse `--seed N`, returning the selected seed and remaining arguments. -/
def seed (exeName : String) (args : List String) (default : Nat := 0) :
    IO (Nat × List String) :=
  orThrow exeName <| takeSeed args default

/-- Parse a positive natural-number flag, using `default` when it is absent. -/
def positiveNatFlag
    (exeName : String)
    (args : List String)
    (name : String)
    (default : Nat) :
    IO (Nat × List String) :=
  orThrow exeName <| takePositiveNatFlag args exeName name default

/-- Fail when command-specific arguments remain after parsing. -/
def requireNoArgs (exeName : String) (args : List String) : IO Unit :=
  orThrow exeName <| checkNoArgs args

end TorchLean.CLI
