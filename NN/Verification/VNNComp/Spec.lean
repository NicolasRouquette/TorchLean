/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.BoundOps
public import NN.Verification.Util.Array
public import NN.Verification.Util.Json

/-!
# VNNLIB-style output specifications

This module contains the benchmark-independent part of the VNN-COMP artifact boundary:

- an input box plus a disjunction-of-conjunctions output spec,
- a JSON loader for the compact `vnnlib_suite_v0_1` export format,
- the outward-rounded interval-box refutation check used by the executable VNN-COMP runner.

Model-specific checkers, such as MNIST-FC, build a graph and output bounds. The arithmetic for
interpreting the VNNLIB rows lives here.
-/

@[expose] public section

namespace NN.Verification.VNNComp.VNNLib

open Lean
open Json
open NN.Verification.Json
open NN.MLTheory.CROWN

/-- One conjunction term $\mathrm{mat}\,y\leq\mathrm{rhs}$ in a VNNLIB disjunction. -/
abbrev Term := Array (Array Float) × Array Float

/-- A VNNLIB-style unsafe-region spec: a disjunction of conjunction terms. -/
abbrev Spec := Array Term

/--
One exported VNN-COMP instance.

`spec` is a disjunction-of-conjunctions: each term is a conjunction
$\mathrm{mat}\,y\leq\mathrm{rhs}$ over the
network output vector `y`.
-/
structure Instance where
  /-- Instance id copied from the exported suite JSON. -/
  id : Nat
  /-- Lower bound for the input box. -/
  inputLo : Array Float
  /-- Upper bound for the input box. -/
  inputHi : Array Float
  /-- Unsafe output-region specification. -/
  spec : Spec

/-- Whether two endpoint arrays define a finite-dimensional, componentwise ordered input box. -/
def inputBoxValid (lo hi : Array Float) : Bool :=
  lo.size == hi.size && allPairwise lo hi NN.Verification.Util.Array.floatLe

/--
Load the compact VNNLIB suite JSON format used by TorchLean checkers.

Expected top-level format: `vnnlib_suite_v0_1`.
-/
def loadSuite (path : String) : IO (Array Instance) := do
  let top ← readJsonObjectFile path
  expectFormat top "vnnlib_suite_v0_1"
  let instArr ← expectFieldArray top "instances" "top-level"
  let mut out : Array Instance := #[]
  for ex in instArr do
    let exo ← expectObj ex "instance"
    let id ← expectFieldNat exo "id" "instance"
    let lo ← expectFieldFiniteFloatArray exo "input_lo" "instance"
    let hi ← expectFieldFiniteFloatArray exo "input_hi" "instance"
    if !inputBoxValid lo hi then
      throw <| IO.userError s!"instance {id}: input box endpoints are mismatched or reversed"
    let specArr ← expectFieldArray exo "spec" "instance"
    let mut specOut : Spec := #[]
    for t in specArr do
      let termObj ← expectObj t "spec term"
      let matJ ← expectField termObj "mat" "spec term"
      let mat ← expectFiniteFloatMatrix matJ "spec term.mat"
      let rhs ← expectFieldFiniteFloatArray termObj "rhs" "spec term"
      specOut := specOut.push (mat, rhs)
    out := out.push { id := id, inputLo := lo, inputHi := hi, spec := specOut }
  pure out

/--
Lower-bound one linear row over an output interval box.

For each coefficient $a_j$, the minimum of $a_jy_j$ over
$y_j\in[\mathrm{lo}_j,\mathrm{hi}_j]$ is the smaller of the endpoint products.
-/
def rowLowerBoundOnBox? (row yLo yHi : Array Float) : Option Float :=
  if hLo : yLo.size = row.size then
    if hHi : yHi.size = row.size then
      some <| (List.finRange row.size).foldl (fun (acc : Float) (j : Fin row.size) =>
        let a := row[j.1]'j.2
        let lo :=
          have h : j.1 < yLo.size := by
            simp [hLo, j.2]
          yLo[j.1]'h
        let hi :=
          have h : j.1 < yHi.size := by
            simp [hHi, j.2]
          yHi[j.1]'h
        BoundOps.addDown acc
          (min (BoundOps.mulDown a lo) (BoundOps.mulDown a hi))) 0.0
    else
      none
  else
    none

/-- Check whether a conjunction term is refuted by the output interval box. -/
def termRefutedByOutputBox (yLo yHi : Array Float) (term : Term) : Bool :=
  let outDim := yLo.size
  let mat := term.fst
  let rhs := term.snd
  if hHi : yHi.size = outDim then
  if hRhs : rhs.size = mat.size then
    (List.finRange mat.size).any (fun (i : Fin mat.size) =>
      let row := mat[i.1]'i.2
      match rowLowerBoundOnBox? row yLo yHi with
      | some lb =>
          let rhsI :=
            have h : i.1 < rhs.size := by
              simp [hRhs, i.2]
            rhs[i.1]'h
          lb > rhsI
      | none => false)
  else
    false
  else
    false

/--
Check whether an unsafe VNNLIB spec is refuted by an output interval box.

The spec is a disjunction of conjunctions. To prove the unsafe region is empty, every disjunct must
be refuted. For a conjunction, it is enough for one row lower bound to exceed its right-hand side.
This executable predicate uses the explicit host-`Float` `BoundOps` boundary; its Boolean result is
not itself a Lean theorem about real-valued graph semantics.
-/
def refutedByOutputBox (yLo yHi : Array Float) (spec : Spec) : Bool :=
  spec.all (termRefutedByOutputBox yLo yHi)

end NN.Verification.VNNComp.VNNLib
