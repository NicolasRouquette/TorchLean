/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.MLTheory.CROWN.Core
public import NN.MLTheory.CROWN.Graph
public import NN.Verification.TorchLean.Compile

/-!
# TorchLean Robustness Workflow

End-to-end robustness certification for a TorchLean model.

We build a compact 2-class classifier in TorchLean, compile it to the verifier IR, and certify a
margin condition on an $\ell^\infty$ input box using:

- IBP (`runIBP`)
- a simple CROWN/affine pass (`runAffine` + `AffineVec.eval_on_box`)

Spec we certify (binary logits):

$$
\forall x\in[x_0-\varepsilon,x_0+\varepsilon],\quad
\operatorname{logit}_0(x)>\operatorname{logit}_1(x).
$$

Run:
  `lake exe verify -- torchlean-robustness`
  `lake exe verify -- torchlean-robustness --float32-mode ieee754exec`
-/

@[expose] public section


namespace NN.Verification.Robustness.TorchLean

open _root_.Spec
open _root_.Spec.Tensor
open _root_.TorchLean

open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN

/-- Input dimension for the TorchLean robustness example. -/
def inDim : Nat := 2
/-- Hidden width for the TorchLean robustness example. -/
def hidDim : Nat := 1
/-- Number of output logits/classes. -/
def outDim : Nat := 2

/-- First-layer weight shape. -/
def hiddenWeightShape : Spec.Shape := .dim hidDim (.dim inDim .scalar)
/-- First-layer bias shape. -/
def hiddenBiasShape : Spec.Shape := .dim hidDim .scalar
/-- Second-layer weight shape. -/
def outputWeightShape : Spec.Shape := .dim outDim (.dim hidDim .scalar)
/-- Second-layer bias shape. -/
def outputBiasShape : Spec.Shape := .dim outDim .scalar
/-- Spec.Shape of one input vector supplied to the certified two-layer network. -/
def xShape : Spec.Shape := .dim inDim .scalar
/-- Output/logit shape. -/
def yShape : Spec.Shape := .dim outDim .scalar

/-- Parameter shapes list used by the compiled TorchLean program (`[hiddenWeight,hiddenBias,outputWeight,outputBias]`). -/
def paramShapes : List Spec.Shape := [hiddenWeightShape, hiddenBiasShape, outputWeightShape, outputBiasShape]

/-- Compute a conservative margin lower bound
$\mathrm{lo}_0-\mathrm{hi}_1$ from logit bounds. -/
def margin {α : Type} [Context α]
    (lo hi : Tensor α yShape) : α :=
  let lo0 := _root_.Spec.Tensor.vecGet lo fin0!
  let hi1 := _root_.Spec.Tensor.vecGet hi fin1!
  lo0 - hi1

/-- Decide if the output bounds certify
$\operatorname{logit}_0>\operatorname{logit}_1$. -/
def certifiedMargin {α : Type} [Context α]
    (lo hi : Tensor α yShape) : Bool :=
  let lo0 := _root_.Spec.Tensor.vecGet lo fin0!
  let hi1 := _root_.Spec.Tensor.vecGet hi fin1!
  Context.gtBool lo0 hi1

/-- TorchLean program for a 2-layer ReLU MLP producing two logits. -/
def classifier {α : Type} [Context α] [DecidableEq Spec.Shape] :
    _root_.Runtime.Autograd.TorchLean.Program α (paramShapes ++ [xShape]) yShape :=
  fun {m} _ _ =>
    fun w1 hiddenBias w2 outputBias x =>
      (do
        let z1 ← Ops.linear (m := m) (α := α) (inDim := inDim) (outDim := hidDim) w1 hiddenBias x
        let h ← Ops.relu (m := m) (α := α) (s := hiddenBiasShape) z1
        Ops.linear (m := m) (α := α) (inDim := hidDim) (outDim := outDim) w2 outputBias h
        : m (Ops.RefTy (m := m) (α := α) yShape))

/--
Run the robustness check once under a chosen scalar backend `α`.

This compiles the TorchLean program to the verifier IR, then computes output bounds with IBP and an
affine/CROWN-style pass.
-/
def runOnce {α : Type} [Runtime.SemanticScalar α] [DecidableEq Spec.Shape] [ToString α]
    [Runtime.Scalar α] [BoundOps α] : IO Unit := do
  let cast : Float → α := Runtime.ofFloat
  -- These in-source constants make the TorchLean-native verifier path fully inspectable.
  -- Data-backed robustness uses
  -- `NN.Verification.Robustness.Digits`, which loads weights and examples from JSON assets.
  --
  -- The chosen weights keep the hidden pre-activation positive over the whole ε-box, so the ReLU
  -- stays linear and the expected certified margin is easy to inspect by hand.
  let hiddenWeight : Tensor α hiddenWeightShape :=
    NN.Tensor.ofListOfLength (α := α) [1, 2] [cast 1.0, cast 1.0] (by rfl)
  let hiddenBias : Tensor α hiddenBiasShape :=
    NN.Tensor.ofListOfLength (α := α) [1] [cast 0.0] (by rfl)
  let outputWeight : Tensor α outputWeightShape :=
    NN.Tensor.ofListOfLength (α := α) [2, 1] [cast 1.0, cast (-1.0)] (by rfl)
  let outputBias : Tensor α outputBiasShape :=
    NN.Tensor.ofListOfLength (α := α) [2] [cast 0.0, cast 0.0] (by rfl)

  let params : nn.ParamTensors α paramShapes :=
    nn.ParamTensors.quad hiddenWeight hiddenBias outputWeight outputBias

  let compiled ←
    match Verification.compileProgram
          (α := α) (paramShapes := paramShapes) (σ := xShape) (τ := yShape)
          (classifier (α := α)) params with
    | .ok c => pure c
    | .error e => throw <| IO.userError e

  let x0 : Tensor α xShape :=
    NN.Tensor.ofListOfLength (α := α) [2] [cast 1.0, cast 1.0] (by rfl)
  let eps : α := Runtime.ofFloat 0.1
  let xB : FlatBox α := Verification.lInfBall (α := α) x0 eps
  let ps : ParamStore α := compiled.seedInputBox xB

  IO.println s!"compiled IR nodes: {compiled.graph.nodes.size}"
  IO.println s!"x0 = {pretty x0}, eps = {eps}"

  -- IBP
  let ibp := compiled.runIBP ps
  let outB ← compiled.outputBoxOrThrow ibp
  IO.println s!"[IBP] logits lo = {pretty outB.lo}"
  IO.println s!"[IBP] logits hi = {pretty outB.hi}"
  if h : outB.dim = outDim then
    let loY : Tensor α yShape := by
      simpa [yShape] using outB.loAsDim h
    let hiY : Tensor α yShape := by
      simpa [yShape] using outB.hiAsDim h
    IO.println s!"[IBP] margin(lo0 - hi1) = {margin (α := α) loY hiY}"
    IO.println s!"[IBP] certified? {certifiedMargin (α := α) loY hiY}"
  else
    IO.println s!"[IBP] unexpected output dim {outB.dim} (expected {outDim}); skipping margin check"

  -- CROWN / affine (w.r.t. the input node)
  let ctx : AffineCtx := { inputId := compiled.inputId, inputDim := inDim }
  let affs := runAffine (α := α) compiled.graph ps ctx ibp
  match compiled.outputAffine? affs with
  | .error msg =>
      IO.println s!"[CROWN] {msg}"
  | .ok outAff =>
      if hIn : outAff.inDim = inDim then
        if hOut : outAff.outDim = outDim then
          let outC := outAff.evalOnFlatBoxAsDim xB hIn.symm hOut
          let loY : Tensor α yShape := by
            simpa [yShape] using outC.lo
          let hiY : Tensor α yShape := by
            simpa [yShape] using outC.hi
          IO.println s!"[CROWN] logits lo = {pretty loY}"
          IO.println s!"[CROWN] logits hi = {pretty hiY}"
          IO.println s!"[CROWN] margin(lo0 - hi1) = {margin (α := α) loY hiY}"
          IO.println s!"[CROWN] certified? {certifiedMargin (α := α) loY hiY}"
        else
          IO.println <|
            (s!"[CROWN] unexpected output dim {outAff.outDim} (expected {outDim}); " ++
              s!"skipping margin check")
      else
        IO.println
          s!"[CROWN] unexpected input dim {outAff.inDim} (expected {inDim}); skipping affine eval"

  -- Backward/dual CROWN (objective-dependent) for the margin: logit0 - logit1.
  let objV : Tensor α (.dim outDim .scalar) :=
    NN.Tensor.ofListOfLength (α := α) [2] [cast 1.0, cast (-1.0)] (by rfl)
  let obj : FlatVec α := { n := outDim, v := objV }
  match compiled.backwardObjectiveBox? ps ibp xB obj with
  | .ok outC =>
      let loM : α := getAtOrZero outC.lo [0]
      let hiM : α := getAtOrZero outC.hi [0]
      IO.println s!"[CROWN-backward] margin lo = {loM}"
      IO.println s!"[CROWN-backward] margin hi = {hiM}"
      IO.println s!"[CROWN-backward] certified? {Context.gtBool loM (0 : α)}"
  | .error msg =>
      IO.println s!"[CROWN-backward] {msg}"

/--
CLI entry point for the TorchLean robustness workflow.

This is wired into `lake exe verify -- torchlean-robustness`.
-/
def main (args : List String) : IO Unit :=
  NN.Verification.TorchLean.runWithBoundDType "TorchLean → IR → IBP/CROWN robustness" args
    (@runOnce)

end NN.Verification.Robustness.TorchLean
