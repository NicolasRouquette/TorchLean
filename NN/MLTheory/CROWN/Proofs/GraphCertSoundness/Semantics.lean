/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph
public import NN.MLTheory.CROWN.Models.Mlp

public import Mathlib.Analysis.SpecialFunctions.Sigmoid
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

/-!
# Graph Certificate Semantics

Value semantics for the verifier graph dialect over `ℝ`, together with the local semantic
consistency predicate used by certificate soundness proofs.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN

namespace CertSoundness

noncomputable section

/-!
## Basic types and predicates

We work over `ℝ` because it has the right order structure for “true” soundness theorems.
The runtime checkers operate over `Float` (fast, executable), and can be used to connect a
Python-produced floating certificate to the *same* computations in Lean.
-/

abbrev Val := FlatVec ℝ

/-- Componentwise enclosure predicate for a tensor point inside a `FlatBox`. -/
abbrev encloses (B : FlatBox ℝ) (x : Tensor ℝ (.dim B.dim .scalar)) : Prop :=
  NN.MLTheory.CROWN.Graph.Theorems.Semantics.encloses (α := ℝ) B x

/-- `EnclosesBox B v` means the value vector `v` lies inside the interval box `B`.

We phrase enclosure using the existing `Sem.encloses` predicate, but our semantic values are
`FlatVec`s (carrying their dimension as a `Nat`), so we also carry a dimension equality witness.
-/
def EnclosesBox (B : FlatBox ℝ) (v : Val) : Prop :=
  ∃ h : B.dim = v.n, encloses B (castDimScalar (α := ℝ) h.symm v.v)

/-!
## Denotational (value) semantics for the verifier graph dialect

The semantics is defined as a *safe* `Option` evaluator:

* If required parameters are missing, it returns `none`.
* If parents are missing (not yet evaluated) or dimensions mismatch, it returns `none`.
-/

/-- Safe lookup of a previously computed parent value. -/
def getVal? (vals : Array (Option Val)) (pid : Nat) : Option Val :=
  if _h : pid < vals.size then vals[pid]! else none

/-- Value semantics for a single node in the supported dialect (over `ℝ`). -/
def evalNode? (nodes : Array Node) (ps : ParamStore ℝ) (inputs : Std.HashMap Nat Val)
    (vals : Array (Option Val)) (id : Nat) : Option Val :=
  let node := nodes[id]!
  match node.kind with
  | .input =>
      inputs[id]?
  | .const _ =>
      ps.constVals[id]?
  | .detach =>
      match node.parents with
      | p1 :: _ => getVal? vals p1
      | _ => none
    | .add =>
        match node.parents with
        | p1 :: p2 :: _ =>
            match getVal? vals p1, getVal? vals p2 with
            | some x, some y =>
                if h : x.n = y.n then
                  -- Use an explicit cast rather than `by simpa [h]` to keep later proofs stable.
                  let yv : Tensor ℝ (.dim x.n .scalar) :=
                    castDimScalar (α := ℝ) (Eq.symm h) y.v
                  some { n := x.n, v := Tensor.addSpec (α := ℝ) x.v yv }
                else
                  none
            | _, _ => none
        | _ => none
  | .sub =>
      match node.parents with
      | p1 :: p2 :: _ =>
          match getVal? vals p1, getVal? vals p2 with
          | some x, some y =>
              if h : x.n = y.n then
                let yv : Tensor ℝ (.dim x.n .scalar) :=
                  castDimScalar (α := ℝ) (Eq.symm h) y.v
                some { n := x.n, v := Tensor.subSpec (α := ℝ) x.v yv }
              else
                none
          | _, _ => none
      | _ => none
  | .mul_elem =>
      match node.parents with
      | p1 :: p2 :: _ =>
          match getVal? vals p1, getVal? vals p2 with
          | some x, some y =>
              if h : x.n = y.n then
                let yv : Tensor ℝ (.dim x.n .scalar) :=
                  castDimScalar (α := ℝ) (Eq.symm h) y.v
                some { n := x.n, v := Tensor.mulSpec (α := ℝ) x.v yv }
              else
                none
          | _, _ => none
      | _ => none
  | .maxPool2d kH kW stride =>
      match node.parents with
      | p1 :: _ =>
          match getVal? vals p1 with
          | some x =>
              match nodes[p1]!.outShape with
              | .dim inC (.dim inH (.dim inW .scalar)) =>
                  let expectedInDim := inC * inH * inW
                  if hIn : x.n = expectedInDim then
                    if hkH : kH = 0 then
                      none
                    else if hkW : kW = 0 then
                      none
                    else if hStride : stride = 0 then
                      none
                    else
                      let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
                      let sFlat : Shape := .dim x.n .scalar
                      have hsize : sFlat.size = sIn.size := by
                        simp [Spec.Shape.size, sFlat, sIn, hIn, expectedInDim, Nat.mul_assoc]
                      let xCHW : Tensor ℝ sIn := Tensor.reshapeSpec (α := ℝ) (s₁ := sFlat) (s₂ :=
                        sIn) x.v hsize
                      let outShape : Shape := Spec.pool2dMultiOutShape inC inH inW kH kW stride
                      let layer : Spec.MaxPool2dSpec kH kW stride hkH hkW hStride := {}
                      let y : Tensor ℝ outShape :=
                        Spec.maxPool2dMultiSpec (α := ℝ) (kH := kH) (kW := kW)
                          (inH := inH) (inW := inW) (inC := inC) (stride := stride)
                          (layer := layer) (input := xCHW)
                      let flat := Tensor.flattenSpec (α := ℝ) y
                      some { n := outShape.size, v := flat }
                  else none
              | _ => none
          | none => none
      | _ => none
  | .maxPool2dPad kH kW stride padding =>
      match node.parents with
      | p1 :: _ =>
          match getVal? vals p1 with
          | some x =>
              match nodes[p1]!.outShape with
              | .dim inC (.dim inH (.dim inW .scalar)) =>
                  let expectedInDim := inC * inH * inW
                  if hIn : x.n = expectedInDim then
                    if hkH : kH = 0 then
                      none
                    else if hkW : kW = 0 then
                      none
                    else if hStride : stride = 0 then
                      none
                    else
                      let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
                      let sFlat : Shape := .dim x.n .scalar
                      have hsize : sFlat.size = sIn.size := by
                        simp [Spec.Shape.size, sFlat, sIn, hIn, expectedInDim, Nat.mul_assoc]
                      let xCHW : Tensor ℝ sIn := Tensor.reshapeSpec (α := ℝ) (s₁ := sFlat) (s₂ :=
                        sIn) x.v hsize
                      let outShape : Shape := Spec.pool2dMultiOutShapePad inC inH inW kH kW
                        stride padding
                      let layer : Spec.MaxPool2dSpec kH kW stride hkH hkW hStride := {}
                      let y : Tensor ℝ outShape :=
                        Spec.maxPool2dMultiSpecPad (α := ℝ) (kH := kH) (kW := kW)
                          (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
                            padding)
                          (layer := layer) (input := xCHW)
                      let flat := Tensor.flattenSpec (α := ℝ) y
                      some { n := outShape.size, v := flat }
                  else none
              | _ => none
          | none => none
      | _ => none
  | .avgPool2d kH kW stride =>
      match node.parents with
      | p1 :: _ =>
          match getVal? vals p1 with
          | some x =>
              match nodes[p1]!.outShape with
              | .dim inC (.dim inH (.dim inW .scalar)) =>
                  let expectedInDim := inC * inH * inW
                  if hIn : x.n = expectedInDim then
                    if hkH : kH = 0 then
                      none
                    else if hkW : kW = 0 then
                      none
                    else if hStride : stride = 0 then
                      none
                    else
                      let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
                      let sFlat : Shape := .dim x.n .scalar
                      have hsize : sFlat.size = sIn.size := by
                        simp [Spec.Shape.size, sFlat, sIn, hIn, expectedInDim, Nat.mul_assoc]
                      let xCHW : Tensor ℝ sIn := Tensor.reshapeSpec (α := ℝ) (s₁ := sFlat) (s₂ :=
                        sIn) x.v hsize
                      let outShape : Shape := Spec.pool2dMultiOutShape inC inH inW kH kW stride
                      let layer : Spec.AvgPool2dSpec kH kW stride hkH hkW hStride := {}
                      let y : Tensor ℝ outShape :=
                        Spec.avgPool2dMultiSpec (α := ℝ) (kH := kH) (kW := kW)
                          (inH := inH) (inW := inW) (inC := inC) (stride := stride)
                          (h1 := hkH) (h2 := hkW) (layer := layer) (input := xCHW)
                      let flat := Tensor.flattenSpec (α := ℝ) y
                      some { n := outShape.size, v := flat }
                  else none
              | _ => none
          | none => none
      | _ => none
  | .avgPool2dPad kH kW stride padding =>
      match node.parents with
      | p1 :: _ =>
          match getVal? vals p1 with
          | some x =>
              match nodes[p1]!.outShape with
              | .dim inC (.dim inH (.dim inW .scalar)) =>
                  let expectedInDim := inC * inH * inW
                  if hIn : x.n = expectedInDim then
                    if hkH : kH = 0 then
                      none
                    else if hkW : kW = 0 then
                      none
                    else if hStride : stride = 0 then
                      none
                    else
                      let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
                      let sFlat : Shape := .dim x.n .scalar
                      have hsize : sFlat.size = sIn.size := by
                        simp [Spec.Shape.size, sFlat, sIn, hIn, expectedInDim, Nat.mul_assoc]
                      let xCHW : Tensor ℝ sIn := Tensor.reshapeSpec (α := ℝ) (s₁ := sFlat) (s₂ :=
                        sIn) x.v hsize
                      let outShape : Shape := Spec.pool2dMultiOutShapePad inC inH inW kH kW
                        stride padding
                      let layer : Spec.AvgPool2dSpec kH kW stride hkH hkW hStride := {}
                      let y : Tensor ℝ outShape :=
                        Spec.avgPool2dMultiSpecPad (α := ℝ) (kH := kH) (kW := kW)
                          (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
                            padding)
                          (h1 := hkH) (h2 := hkW) (layer := layer) (input := xCHW)
                      let flat := Tensor.flattenSpec (α := ℝ) y
                      some { n := outShape.size, v := flat }
                  else none
              | _ => none
          | none => none
      | _ => none
  | .relu =>
      match node.parents with
      | p1 :: _ =>
          match getVal? vals p1 with
          | some x => some { n := x.n, v := Activation.reluSpec (α := ℝ) x.v }
          | none => none
      | _ => none
  | .tanh =>
      match node.parents with
      | p1 :: _ =>
          match getVal? vals p1 with
          | some x => some { n := x.n, v := Activation.tanhSpec (α := ℝ) x.v }
          | none => none
      | _ => none
  | .sigmoid =>
      match node.parents with
      | p1 :: _ =>
          match getVal? vals p1 with
          | some x => some { n := x.n, v := Activation.sigmoidSpec (α := ℝ) x.v }
          | none => none
      | _ => none
  | .sin =>
      match node.parents with
      | p1 :: _ =>
          match getVal? vals p1 with
          | some x =>
              some
                { n := x.n
                  v := Tensor.mapSpec (α := ℝ) (s := .dim x.n .scalar) (fun z => Real.sin z) x.v }
          | none => none
      | _ => none
  | .cos =>
      match node.parents with
      | p1 :: _ =>
          match getVal? vals p1 with
          | some x =>
              some
                { n := x.n
                  v := Tensor.mapSpec (α := ℝ) (s := .dim x.n .scalar) (fun z => Real.cos z) x.v }
          | none => none
      | _ => none
  | .linear =>
        match node.parents with
        | p1 :: _ =>
            match getVal? vals p1, ps.linearWB[id]? with
            | some x, some p =>
                if h : x.n = p.n then
                  let xv : Tensor ℝ (.dim p.n .scalar) := castDimScalar (α := ℝ) h x.v
                  let yv : Tensor ℝ (.dim p.m .scalar) :=
                    Spec.linearSpec (α := ℝ) { weights := p.w, bias := p.b } xv
                  some { n := p.m, v := yv }
                else
                  none
            | _, _ => none
        | _ => none
  | .matmul =>
      match node.parents with
      | p1 :: _ =>
          match getVal? vals p1, ps.matmulW[id]? with
          | some x, some p =>
              if h : x.n = p.n then
                let xv : Tensor ℝ (.dim p.n .scalar) := castDimScalar (α := ℝ) h x.v
                let z : Tensor ℝ (.dim p.m .scalar) := Spec.fill (α := ℝ) 0 (.dim p.m .scalar)
                let yv : Tensor ℝ (.dim p.m .scalar) :=
                  Spec.linearSpec (α := ℝ) { weights := p.w, bias := z } xv
                some { n := p.m, v := yv }
              else
                none
          | _, _ => none
      | _ => none
  | .sum =>
      match node.parents with
      | p1 :: _ =>
          match getVal? vals p1 with
          | some x =>
              let onesRow : Tensor ℝ (.dim 1 (.dim x.n .scalar)) :=
                Spec.fill (α := ℝ) 1 (.dim 1 (.dim x.n .scalar))
              let y : Tensor ℝ (.dim 1 .scalar) := Spec.matVecMulSpec (α := ℝ) onesRow x.v
              some { n := 1, v := y }
          | none => none
      | _ => none
  | .reshape _ _ | .flatten _ =>
      match node.parents with
      | p1 :: _ =>
          match getVal? vals p1 with
          | some x =>
              if h : x.n = node.outShape.size then
                let xv : Tensor ℝ (.dim node.outShape.size .scalar) :=
                  castDimScalar (α := ℝ) h x.v
                some { n := node.outShape.size, v := xv }
              else
                none
          | none => none
      | _ => none
  | .concat _ =>
      match node.parents with
      | p1 :: p2 :: _ =>
          match getVal? vals p1, getVal? vals p2 with
          | some x, some y =>
              match x.v, y.v with
              | .dim fx, .dim fy =>
                  let outDim := x.n + y.n
                  let z : Tensor ℝ (.dim outDim .scalar) :=
                    Tensor.dim (fun i =>
                      Fin.addCases (fun i1 => fx i1) (fun i2 => fy i2) i)
                  some { n := outDim, v := z }
          | _, _ => none
      | _ => none
  | _ =>
      none

/-- Evaluate an entire graph in node-id order using `evalNode?`. -/
def evalGraph? (g : Graph) (ps : ParamStore ℝ) (inputs : Std.HashMap Nat Val) :
    Array (Option Val) :=
  let init := Array.replicate g.nodes.size none
  (List.finRange g.nodes.size).foldl
    (fun acc i => acc.set! i (evalNode? g.nodes ps inputs acc i))
    init

/-!
Even though we provided an executable `evalGraph?`, the **main soundness theorem** below does not
depend on it.

Reason: proving properties about the `foldl` evaluator would introduce a lot of “bookkeeping”
lemmas about `Array.set!` and list folds.

Instead, we state soundness for *any* array `vals` that is a **local model** of the semantics step:
each node’s value must equal `evalNode?` computed from its parents’ values.

This is a standard technique in proof engineering: separate “semantic consistency” from
“the particular implementation of the evaluator”.
-/

/-!
## Local semantic consistency (`SemLocalOK`)

`SemLocalOK g ps inputs vals` means:

* `vals` has the correct length, and
* each entry `vals[id]` equals `evalNode?` computed from the full array `vals`.

For a DAG (and only for a DAG), this is exactly the property that `vals` is a valid interpretation
of the graph semantics.

Existence and uniqueness of `vals` are evaluator-correctness facts. This file proves the certificate
theorem in the reusable form: for any semantic interpretation `vals`, a locally-correct certificate
encloses it.
-/

def SemLocalOK (g : Graph) (ps : ParamStore ℝ) (inputs : Std.HashMap Nat Val)
    (vals : Array (Option Val)) : Prop :=
  vals.size = g.nodes.size ∧
  ∀ id : Nat, id < g.nodes.size → vals[id]! = evalNode? g.nodes ps inputs vals id

end

end CertSoundness

end NN.MLTheory.CROWN.Graph
