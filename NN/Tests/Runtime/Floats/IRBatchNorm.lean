/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Check
public import NN.IR.Semantics
public import NN.MLTheory.CROWN.Graph.Engine
public import NN.Runtime.Autograd.Compiled.IRExec
public import NN.Runtime.PyTorch.Export.IRPyTorch
public import NN.Tests.Runtime.Floats.Utils

/-!
# IR BatchNorm Checks

Regression checks for the first-class eval-mode BatchNorm2d IR node.
-/

@[expose] public section

namespace Tests
namespace Floats
namespace IRBatchNorm

open Spec
open Tensor
open NN.IR
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open Tests.Floats.Utils

abbrev n : Nat := 2
abbrev c : Nat := 2
abbrev h : Nat := 2
abbrev w : Nat := 2
abbrev sNCHW : Shape := .dim n (.dim c (.dim h (.dim w .scalar)))

def input : Tensor Float sNCHW :=
  Tensor.dim (fun ni =>
    Tensor.dim (fun ci =>
      Tensor.dim (fun hi =>
        Tensor.dim (fun wi =>
          let base := Float.ofNat (ni.val * 8 + ci.val * 4 + hi.val * 2 + wi.val + 1)
          Tensor.scalar (if ci.val = 0 then base else -base)))))

def gamma : Tensor Float (.dim c .scalar) := tensor! [1.0, 0.5]
def beta : Tensor Float (.dim c .scalar) := tensor! [0.0, 0.1]
def mean : Tensor Float (.dim c .scalar) := tensor! [2.0, -3.0]
def var : Tensor Float (.dim c .scalar) := tensor! [4.0, 9.0]
def bnEps : Float := 1e-5

def graph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := [], kind := .input, outShape := sNCHW },
      { id := 1, parents := [0], kind := .batchNorm2dNchwEval c, outShape := sNCHW }
    ] }

def payload : Payload Float :=
  { batchNorm2dNchwEval? := fun id =>
      if id = 1 then
        some { c := c, gamma := gamma, beta := beta, mean := mean, var := var, eps := bnEps }
      else
        none }

def expected (x gamma beta mean var eps : Float) : Float :=
  ((x - mean) / Float.sqrt (max var 0.0 + eps)) * gamma + beta

def flatVal {n : Nat} (t : Tensor Float (.dim n .scalar)) (i : Fin n) : Float :=
  match getAtSpec t i with
  | .scalar v => v

def expectSome {α : Type} (label : String) : Option α → IO α
  | some x => pure x
  | none => throw (IO.userError s!"ir_batchnorm: expected {label}")

def verifierBox : FlatBox Float :=
  { dim := Shape.size sNCHW
    lo := Tensor.flattenSpec input
    hi := Tensor.flattenSpec input }

def verifierParams : ParamStore Float :=
  { inputBoxes := (Std.HashMap.emptyWithCapacity.insert 0 verifierBox)
    batchNorm2dNchwEval :=
      (Std.HashMap.emptyWithCapacity.insert 1
        { c := c, gamma := gamma, beta := beta, mean := mean, var := var, eps := bnEps }) }

def unitObjective : FlatVec Float :=
  { n := Shape.size sNCHW
    v := Tensor.dim (fun i => Tensor.scalar (if decide (i.val = 0) then 1.0 else 0.0)) }

def expectNCHW (label : String) (v : DVal Float) : IO (Tensor Float sNCHW) := do
  match v with
  | ⟨s, t⟩ =>
      if hs : s = sNCHW then
        pure (hs ▸ t)
      else
        throw (IO.userError s!"{label}: expected {repr sNCHW}, got {repr s}")

def checkTensor (label : String) (y : Tensor Float sNCHW) : IO Unit := do
  for ni in List.finRange n do
    for ci in List.finRange c do
      for hi in List.finRange h do
        for wi in List.finRange w do
          let want :=
            expected (nchwVal input ni ci hi wi) (vecVal gamma ci) (vecVal beta ci)
              (vecVal mean ci) (vecVal var ci) bnEps
          assertApprox s!"{label}[{ni.val},{ci.val},{hi.val},{wi.val}]"
            (nchwVal y ni ci hi wi) want 1e-5

def run : IO Unit := do
  IO.println "ir_batchnorm: begin"
  match Graph.checkShapes graph with
  | Except.ok () => pure ()
  | Except.error e => throw (IO.userError s!"ir_batchnorm: graph shape check failed: {e}")

  let yDenote ←
    match Graph.denote (α := Float) (g := graph) (payload := payload)
        (input := DVal.mk (α := Float) sNCHW input) (outputId := 1) with
    | .ok v => expectNCHW "ir_batchnorm denote" v
    | .error e => throw (IO.userError s!"ir_batchnorm: denote failed: {e}")
  checkTensor "ir_batchnorm denote" yDenote

  let exec ←
    match Runtime.Autograd.Compiled.execGraphOfIR (α := Float) graph payload with
    | Except.ok e => pure e
    | Except.error e => throw (IO.userError s!"ir_batchnorm: compile failed: {e}")
  let inputExec : Tensor Float exec.inShape ←
    if hIn : exec.inShape = sNCHW then
      pure (hIn.symm ▸ input)
    else
      throw (IO.userError s!"ir_batchnorm: compiled input shape mismatch: {repr exec.inShape}")
  let vals := Runtime.Autograd.Compiled.ExecGraphData.denoteAll (α := Float) exec inputExec
  match vals[1]? with
  | some v =>
      let y ← expectNCHW "ir_batchnorm compiled" v
      checkTensor "ir_batchnorm compiled" y
  | none =>
      throw (IO.userError "ir_batchnorm: compiled output node missing")

  let ibp := runIBP (α := Float) graph verifierParams
  let yB ← expectSome "IBP BatchNorm box" ibp[1]!
  let want0 := expected (nchwVal input 0 0 0 0) (vecVal gamma 0) (vecVal beta 0)
    (vecVal mean 0) (vecVal var 0) bnEps
  if hdim : yB.dim = Shape.size sNCHW then
    let lo : Tensor Float (.dim (Shape.size sNCHW) .scalar) :=
      NN.MLTheory.CROWN.Graph.castDimScalar (α := Float) hdim yB.lo
    let hi : Tensor Float (.dim (Shape.size sNCHW) .scalar) :=
      NN.MLTheory.CROWN.Graph.castDimScalar (α := Float) hdim yB.hi
    assertApprox "ir_batchnorm ibp lo[0]" (flatVal lo ⟨0, by decide⟩) want0 1e-5
    assertApprox "ir_batchnorm ibp hi[0]" (flatVal hi ⟨0, by decide⟩) want0 1e-5
  else
    throw (IO.userError s!"ir_batchnorm: IBP output dimension mismatch: {yB.dim}")

  let ctx : AffineCtx := { inputId := 0, inputDim := Shape.size sNCHW }
  let aff := runAffine (α := Float) graph verifierParams ctx ibp
  let _ ← expectSome "affine BatchNorm transfer" aff[1]!
  let crown := runCROWN (α := Float) graph verifierParams ctx ibp
  let _ ← expectSome "CROWN BatchNorm transfer" crown[1]!
  let _ ← expectSome "backward CROWN BatchNorm transfer"
    (runCROWNBackwardObjective (α := Float) graph verifierParams ctx ibp 1 unitObjective)

  let pyCode ←
    match Export.IRPyTorch.emit graph verifierParams 0 1 with
    | .ok code => pure code
    | .error e => throw (IO.userError s!"ir_batchnorm: IR->PyTorch export failed: {e}")
  unless pyCode.contains "eps=0.000010" do
    throw (IO.userError "ir_batchnorm: IR->PyTorch export did not preserve BatchNorm eps")

  IO.println "ir_batchnorm: ok"

end IRBatchNorm
end Floats
end Tests
