/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Compile.Builder

/-!
# Compiled Verification API

Public entrypoints for compiling TorchLean programs and querying IBP and CROWN bounds. Graph
construction lives in `Compile.Builder`; this module contains the stable result type and operations
that users apply after compilation.
-/

@[expose] public section

namespace NN.Verification.TorchLean

open _root_.Spec
open _root_.Spec.Tensor
open NN.IR

/-! ## Public compile entrypoints -/

/--
Result of compiling a TorchLean forward model to verifier IR.

This bundles:
- the produced IR graph (`NN.IR.Graph`),
- a CROWN/LiRPA-style `ParamStore` containing constants and layer parameters, and
- the distinguished input/output node ids (used by bound propagation and certificate checkers).
-/
structure CompiledIR (α : Type) [Context α] where
  /-- Compiled IR graph. -/
  graph    : Graph
  /-- Parameters/constants for verifier algorithms (IBP, CROWN, etc.). -/
  ps       : NN.MLTheory.CROWN.Graph.ParamStore α
  /-- Distinguished input node id (kept stable as `0`). -/
  inputId  : Nat
  /-- Output node id. -/
  outputId : Nat

/-- Seed the distinguished verifier input with an explicit flat input box. -/
def CompiledIR.seedInputBox {α : Type} [Context α]
    (compiled : CompiledIR α) (xB : NN.MLTheory.CROWN.FlatBox α) :
    NN.MLTheory.CROWN.Graph.ParamStore α :=
  compiled.ps.seedInputBox compiled.inputId xB

/-- Flatten a shaped center/radius pair into the verifier input-box representation. -/
def lInfBox {α : Type} [Context α] {s : Shape}
    (center radius : Tensor α s) : NN.MLTheory.CROWN.FlatBox α :=
  NN.MLTheory.CROWN.FlatBox.lInfBox (α := α) center radius

/-- Uniform $\ell^\infty$ box around a shaped TorchLean input tensor. -/
def lInfBall {α : Type} [Context α] {s : Shape}
    (center : Tensor α s) (eps : α) : NN.MLTheory.CROWN.FlatBox α :=
  NN.MLTheory.CROWN.FlatBox.lInfBall (α := α) center eps

/-- Seed the distinguished verifier input with a uniform $\ell^\infty$ ball. -/
def CompiledIR.seedLInfBall {α : Type} [Context α] {s : Shape}
    (compiled : CompiledIR α) (center : Tensor α s) (eps : α) :
    NN.MLTheory.CROWN.Graph.ParamStore α :=
  compiled.ps.seedLInfBall compiled.inputId center eps

/-- Shape of the distinguished verifier input node. -/
def CompiledIR.inputShape? {α : Type} [Context α] (compiled : CompiledIR α) :
    Except String Shape := do
  match compiled.graph.nodes[compiled.inputId]? with
  | some node => pure node.outShape
  | none =>
      throw s!"compiled verifier input node {compiled.inputId} is out of bounds for {compiled.graph.nodes.size} graph nodes"

/-- Flattened dimension of the distinguished verifier input node. -/
def CompiledIR.inputDim? {α : Type} [Context α] (compiled : CompiledIR α) :
    Except String Nat := do
  pure (Spec.Shape.size (← compiled.inputShape?))

/-- Affine/CROWN context for the distinguished verifier input. -/
def CompiledIR.affineCtx? {α : Type} [Context α] (compiled : CompiledIR α) :
    Except String NN.MLTheory.CROWN.Graph.AffineCtx := do
  pure { inputId := compiled.inputId, inputDim := ← compiled.inputDim? }

/-- Run IBP on a compiled verifier graph. -/
def CompiledIR.runIBP {α : Type} [Context α] [NN.MLTheory.CROWN.BoundOps α]
    (compiled : CompiledIR α)
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) :
    Array (Option (NN.MLTheory.CROWN.FlatBox α)) :=
  NN.MLTheory.CROWN.Graph.runIBP (α := α) compiled.graph ps

/-- Read the verifier output box from an IBP result array. -/
def CompiledIR.outputBox? {α : Type} [Context α]
    (compiled : CompiledIR α)
    (boxes : Array (Option (NN.MLTheory.CROWN.FlatBox α))) :
    Except String (NN.MLTheory.CROWN.FlatBox α) := do
  NN.MLTheory.CROWN.Graph.outputBox? boxes compiled.outputId

/-- Read the compiled verifier output box, throwing an `IO.userError` if it is missing. -/
def CompiledIR.outputBoxOrThrow {α : Type} [Context α]
    (compiled : CompiledIR α)
    (boxes : Array (Option (NN.MLTheory.CROWN.FlatBox α))) :
    IO (NN.MLTheory.CROWN.FlatBox α) := do
  match compiled.outputBox? boxes with
  | .ok outB => pure outB
  | .error msg => throw <| IO.userError msg

/-- Read the verifier output affine form from a forward affine result array. -/
def CompiledIR.outputAffine? {α : Type} [Context α]
    (compiled : CompiledIR α)
    (affs : Array (Option (NN.MLTheory.CROWN.Graph.FlatAffine α))) :
    Except String (NN.MLTheory.CROWN.Graph.FlatAffine α) := do
  match affs[compiled.outputId]? with
  | some (some outAff) => pure outAff
  | some none => throw s!"verification output affine missing at node {compiled.outputId}"
  | none =>
      throw s!"verification output node {compiled.outputId} is out of bounds for {affs.size} affine entries"

/-- Read the verifier output CROWN bounds from a forward CROWN result array. -/
def CompiledIR.outputCROWN? {α : Type} [Context α]
    (compiled : CompiledIR α)
    (bounds : Array (Option (NN.MLTheory.CROWN.Graph.FlatAffineBounds α))) :
    Except String (NN.MLTheory.CROWN.Graph.FlatAffineBounds α) := do
  match bounds[compiled.outputId]? with
  | some (some outB) => pure outB
  | some none => throw s!"verification CROWN output missing at node {compiled.outputId}"
  | none =>
      throw s!"verification output node {compiled.outputId} is out of bounds for {bounds.size} CROWN entries"

/-- Run forward CROWN and evaluate the compiled verifier output on a selected input box. -/
def CompiledIR.outputBoxCROWN? {α : Type} [Context α] [NN.MLTheory.CROWN.BoundOps α]
    (compiled : CompiledIR α) (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (xB : NN.MLTheory.CROWN.FlatBox α) :
    Except String (NN.MLTheory.CROWN.FlatBox α) := do
  let inputDim ← compiled.inputDim?
  NN.MLTheory.CROWN.Graph.outputBoxCROWN? (α := α) compiled.graph ps xB
    compiled.inputId compiled.outputId inputDim

/-- Run forward CROWN for a compiled verifier graph, throwing an `IO.userError` on failure. -/
def CompiledIR.outputBoxCROWNOrThrow {α : Type} [Context α] [NN.MLTheory.CROWN.BoundOps α]
    (compiled : CompiledIR α) (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (xB : NN.MLTheory.CROWN.FlatBox α) :
    IO (NN.MLTheory.CROWN.FlatBox α) := do
  match compiled.outputBoxCROWN? ps xB with
  | .ok outB => pure outB
  | .error msg => throw <| IO.userError msg

/-- Run objective-dependent backward CROWN and evaluate the scalar objective on the input box. -/
def CompiledIR.backwardObjectiveBox? {α : Type} [Context α] [NN.MLTheory.CROWN.BoundOps α]
    (compiled : CompiledIR α) (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (ibp : Array (Option (NN.MLTheory.CROWN.FlatBox α)))
    (xB : NN.MLTheory.CROWN.FlatBox α) (obj : NN.MLTheory.CROWN.Graph.FlatVec α) :
    Except String (NN.MLTheory.CROWN.FlatBox α) := do
  let ctx ← compiled.affineCtx?
  NN.MLTheory.CROWN.Graph.backwardObjectiveBox? (α := α) compiled.graph ps ctx
    ibp xB compiled.outputId obj

/-- `IO` wrapper around `CompiledIR.backwardObjectiveBox?`. -/
def CompiledIR.backwardObjectiveBoxOrThrow {α : Type} [Context α] [NN.MLTheory.CROWN.BoundOps α]
    (compiled : CompiledIR α) (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (ibp : Array (Option (NN.MLTheory.CROWN.FlatBox α)))
    (xB : NN.MLTheory.CROWN.FlatBox α) (obj : NN.MLTheory.CROWN.Graph.FlatVec α) :
    IO (NN.MLTheory.CROWN.FlatBox α) := do
  match compiled.backwardObjectiveBox? ps ibp xB obj with
  | .ok outB => pure outB
  | .error msg => throw <| IO.userError msg

/-- Convert a parameter `TList` into a `RefList` of compile-time constants. -/
def refListConstOfTList {α : Type} [Context α] :
    {ss : List Shape} → Runtime.Autograd.Torch.TList α ss → Runtime.Autograd.Torch.RefList (Ref α)
      ss
  | [], .nil => .nil
  | _s :: ss, .cons t ts => .cons (.const t) (refListConstOfTList (ss := ss) ts)

/-- Compile a TorchLean forward model with a single distinguished input (the last argument). -/
def compileForward
    {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes : List Shape} {inShape outShape : Shape}
    (model : Runtime.Autograd.TorchLean.Program α (paramShapes ++ [inShape]) outShape)
    (params : Runtime.Autograd.Torch.TList α paramShapes) :
    Except String (CompiledIR α) :=
  let build : BuildM α Nat := do
    let x : Ref α inShape ← emitInput (α := α)
    let psRefs : Runtime.Autograd.Torch.RefList (Ref α) paramShapes :=
      refListConstOfTList (α := α) (ss := paramShapes) params
    let allRefs : Runtime.Autograd.Torch.RefList (Ref α) (paramShapes ++ [inShape]) :=
      Runtime.Autograd.Torch.RefList.append (ss₁ := paramShapes) (ss₂ := [inShape]) psRefs (.cons x
        .nil)
    let outRef ← Runtime.Autograd.Torch.CurriedRef.uncurry
      (Ref := fun s => Ref α s) (ss := paramShapes ++ [inShape]) (model (m := BuildM α)) allRefs
    ensureNode (α := α) outRef
  match StateT.run build { nodes := #[], ps := {} } with
  | Except.error e => Except.error e
  | Except.ok (outId, st) =>
      let g : Graph := { nodes := st.nodes }
      match (g.checkWellFormed *> g.checkShapes) with
      | Except.error e => Except.error s!"TorchLean→IR: produced an ill-formed graph: {e}"
      | Except.ok _ =>
          Except.ok { graph := g, ps := st.ps, inputId := 0, outputId := outId }

end NN.Verification.TorchLean
