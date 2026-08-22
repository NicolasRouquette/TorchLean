/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.Semantics

/-!
# IBP Certificate Step

Safe, option-returning certificate-step semantics for the graph IBP checker.  This is the proof layer
counterpart of the executable bound propagation step.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN

namespace CertSoundness

noncomputable section

/-!
## The IBP “certificate step” (safe, total)

This is a safe version of `propagateIBPNode` tailored to the subset of ops we prove soundness for.
It defines what it means for a per-node certificate to be “locally well-formed”.

Important: the runtime implementation `propagateIBPNode` uses `get!` on parent boxes; it assumes
topological order and that earlier boxes exist. Here we avoid partiality by returning `none`
whenever parents are missing.
-/

def getBox? (cert : Array (Option (FlatBox ℝ))) (pid : Nat) : Option (FlatBox ℝ) :=
  if _h : pid < cert.size then cert[pid]! else none

/--
Safe per-node IBP step for the checker semantics.

This is the total (option-returning) analogue of the runtime `propagateIBPNode`, restricted to the
ops handled in the soundness development.
-/
def certStepNode? (nodes : Array Node) (ps : ParamStore ℝ) (cert : Array (Option (FlatBox ℝ))) (id :
  Nat) :
    Option (FlatBox ℝ) :=
  let node := nodes[id]!
  match node.kind with
  | .input =>
      ps.inputBoxes[id]?
  | .const _ =>
      match ps.constVals[id]? with
      | none => none
      | some v => some { dim := v.n, lo := v.v, hi := v.v }
  | .detach =>
      match node.parents with
      | p1 :: _ => getBox? cert p1
      | _ => none
  | .add =>
      match node.parents with
      | p1 :: p2 :: _ =>
          match getBox? cert p1, getBox? cert p2 with
          | some B1, some B2 =>
              if _h : B1.dim = B2.dim then
                some (boxAdd (α := ℝ) B1 B2)
              else
                none
          | _, _ => none
      | _ => none
  | .sub =>
      match node.parents with
      | p1 :: p2 :: _ =>
          match getBox? cert p1, getBox? cert p2 with
          | some B1, some B2 =>
              if _h : B1.dim = B2.dim then
                some (boxSub (α := ℝ) B1 B2)
              else
                none
          | _, _ => none
      | _ => none
  | .mul_elem =>
      match node.parents with
      | p1 :: p2 :: _ =>
          match getBox? cert p1, getBox? cert p2 with
          | some B1, some B2 => boxMulElem (α := ℝ) B1 B2
          | _, _ => none
      | _ => none
  | .relu =>
      match node.parents with
      | p1 :: _ =>
          match getBox? cert p1 with
          | some B => some (boxRelu (α := ℝ) B)
          | none => none
      | _ => none
  | .tanh =>
      match node.parents with
      | p1 :: _ =>
          match getBox? cert p1 with
          | some Xin =>
              let yB := NN.MLTheory.CROWN.Runtime.Ops.IBP.tanh (α := ℝ) (n := Xin.dim) (ofFlatBox (α
                := ℝ) Xin)
              some (toFlatBox (α := ℝ) Xin.dim yB)
          | none => none
      | _ => none
  | .sigmoid =>
      match node.parents with
      | p1 :: _ =>
          match getBox? cert p1 with
          | some Xin =>
              let yB := NN.MLTheory.CROWN.Runtime.Ops.IBP.sigmoid (α := ℝ) (n := Xin.dim) (ofFlatBox
                (α := ℝ) Xin)
              some (toFlatBox (α := ℝ) Xin.dim yB)
          | none => none
      | _ => none
  | .sin =>
      match node.parents with
      | p1 :: _ =>
          match getBox? cert p1 with
          | some Xin =>
              let yB := NN.MLTheory.CROWN.Runtime.Ops.IBP.sin (α := ℝ) (n := Xin.dim) (ofFlatBox (α
                := ℝ) Xin)
              some (toFlatBox (α := ℝ) Xin.dim yB)
          | none => none
      | _ => none
  | .cos =>
      match node.parents with
      | p1 :: _ =>
          match getBox? cert p1 with
          | some Xin =>
              let yB := NN.MLTheory.CROWN.Runtime.Ops.IBP.cos (α := ℝ) (n := Xin.dim) (ofFlatBox (α
                := ℝ) Xin)
              some (toFlatBox (α := ℝ) Xin.dim yB)
          | none => none
      | _ => none
  | .linear =>
      match node.parents with
      | p1 :: _ =>
          match getBox? cert p1 with
          | some Xin => ibpLinear (α := ℝ) id ps Xin
          | none => none
      | _ => none
  | .matmul =>
      match node.parents with
      | p1 :: _ =>
          match getBox? cert p1 with
          | some Xin => ibpMatmul (α := ℝ) id ps Xin
          | none => none
      | _ => none
  | _ =>
      none

/-- A certificate is *locally consistent* if every node equals `certStepNode?` at that node. -/
def CertLocalOK (g : Graph) (ps : ParamStore ℝ) (cert : Array (Option (FlatBox ℝ))) : Prop :=
  cert.size = g.nodes.size ∧
  ∀ id : Nat, id < g.nodes.size → cert[id]! = certStepNode? g.nodes ps cert id

end

end CertSoundness

end NN.MLTheory.CROWN.Graph
