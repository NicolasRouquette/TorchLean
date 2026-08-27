/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.MLTheory.CROWN.Core
public import NN.MLTheory.CROWN.Graph
public import NN.Verification.TorchLean.Lowering

public import NN.API.Verification

/-!
# TorchLean Robustness Workflow

End-to-end robustness certification for a TorchLean model.

We build a compact 2-class classifier in TorchLean, lower it to the verifier IR, and certify a
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
  `lake exe verify -- torchlean-robustness --scalar ieee32-exec`
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

/-- Parameter shapes list used by the lowered TorchLean program (`[hiddenWeight,hiddenBias,outputWeight,outputBias]`). -/
def paramShapes : List Spec.Shape := [hiddenWeightShape, hiddenBiasShape, outputWeightShape, outputBiasShape]

/-- Compute a conservative margin lower bound
$\mathrm{lo}_0-\mathrm{hi}_1$ from logit bounds. -/
def margin {α : Type} [Context α]
    (lo hi : Spec.Tensor α yShape) : α :=
  let lo0 := _root_.Spec.Tensor.getScalar lo ⟨0, by decide⟩
  let hi1 := _root_.Spec.Tensor.getScalar hi ⟨1, by decide⟩
  lo0 - hi1

/-- Decide if the output bounds certify
$\operatorname{logit}_0>\operatorname{logit}_1$. -/
def certifiedMargin {α : Type} [Context α]
    (lo hi : Spec.Tensor α yShape) : Bool :=
  let lo0 := _root_.Spec.Tensor.getScalar lo ⟨0, by decide⟩
  let hi1 := _root_.Spec.Tensor.getScalar hi ⟨1, by decide⟩
  Context.gtBool lo0 hi1

/-- TorchLean program for a 2-layer ReLU MLP producing two logits. -/
def classifier {α : Type} [Context α] [DecidableEq Spec.Shape] :
    _root_.Runtime.Autograd.TorchLean.Program α (paramShapes ++ [xShape]) yShape :=
  fun {m} _ _ =>
    fun w1 hiddenBias w2 outputBias x =>
      (do
        let z1 ← Runtime.linear
          (m := m) (α := α)
          (leading := []) (inDim := inDim) (outDim := hidDim) w1 hiddenBias x
        let h ← Runtime.relu (m := m) (α := α) (s := hiddenBiasShape) z1
        Runtime.linear
          (m := m) (α := α)
          (leading := []) (inDim := hidDim) (outDim := outDim) w2 outputBias h
        : m (Runtime.ValueRef (m := m) (α := α) yShape))

/--
Run the robustness check once under a chosen scalar backend `α`.

This lowers the TorchLean program to the verifier IR, then computes output bounds with IBP and an
affine/CROWN-style pass.
-/
def runOnce {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α]
    [Runtime.FromFloat α] [BoundOps α] : IO Unit := do
  let cast : Float → α := Runtime.ofFloat
  -- These in-source constants make the TorchLean-native verifier path fully inspectable.
  -- Data-backed robustness uses
  -- `NN.Verification.Robustness.Digits`, which loads weights and examples from JSON assets.
  --
  -- The chosen weights keep the hidden pre-activation positive over the whole ε-box, so the ReLU
  -- stays linear and the expected certified margin is easy to inspect by hand.
  let hiddenWeight : Spec.Tensor α hiddenWeightShape :=
    Spec.Tensor.map cast (tensorOfArray! (ty := Float) [1, 2] #[1.0, 1.0])
  let hiddenBias : Spec.Tensor α hiddenBiasShape :=
    Spec.Tensor.map cast (tensorOfArray! (ty := Float) [1] #[0.0])
  let outputWeight : Spec.Tensor α outputWeightShape :=
    Spec.Tensor.map cast (tensorOfArray! (ty := Float) [2, 1] #[1.0, -1.0])
  let outputBias : Spec.Tensor α outputBiasShape :=
    Spec.Tensor.map cast (tensorOfArray! (ty := Float) [2] #[0.0, 0.0])

  let params : _root_.TorchLean.TensorPack α paramShapes :=
    _root_.TorchLean.TensorPack! hiddenWeight, hiddenBias, outputWeight, outputBias

  let lowered ←
    match Verification.lowerProgramToIR
          (α := α) (paramShapes := paramShapes) (σ := xShape) (τ := yShape)
          (classifier (α := α)) params with
    | .ok c => pure c
    | .error e => throw <| IO.userError e

  let x0 : Spec.Tensor α xShape :=
    Spec.Tensor.map cast (tensorOfArray! (ty := Float) [2] #[1.0, 1.0])
  let eps : α := Runtime.ofFloat 0.1
  let xB : FlatBox α := NN.Verification.TorchLean.lInfBall (α := α) x0 eps
  let ps : ParamStore α := lowered.seedInputBox xB

  IO.println s!"lowered IR nodes: {lowered.graph.nodes.size}"
  IO.println s!"x0 = {pretty x0}, eps = {eps}"

  -- IBP
  let ibp := lowered.runIBP ps
  let outB ← lowered.outputBoxOrThrow ibp
  IO.println s!"[IBP] logits lo = {pretty outB.lo}"
  IO.println s!"[IBP] logits hi = {pretty outB.hi}"
  if h : outB.dim = outDim then
    let loY : Spec.Tensor α yShape := by
      simpa [yShape] using outB.loAsDim h
    let hiY : Spec.Tensor α yShape := by
      simpa [yShape] using outB.hiAsDim h
    IO.println s!"[IBP] margin(lo0 - hi1) = {margin (α := α) loY hiY}"
    IO.println s!"[IBP] certified? {certifiedMargin (α := α) loY hiY}"
  else
    IO.println s!"[IBP] unexpected output dim {outB.dim} (expected {outDim}); skipping margin check"

  -- CROWN / affine (w.r.t. the input node)
  let ctx : AffineCtx := { inputId := lowered.inputId, inputDim := inDim }
  let affs := runAffine (α := α) lowered.graph ps ctx ibp
  match lowered.outputAffine? affs with
  | .error msg =>
      IO.println s!"[CROWN] {msg}"
  | .ok outAff =>
      if hIn : outAff.inDim = inDim then
        if hOut : outAff.outDim = outDim then
          have hBoxDim : xB.dim = inDim := by
            change (NN.Verification.TorchLean.lInfBall x0 eps).dim = inDim
            rw [NN.Verification.TorchLean.lInfBall_dim]
            rfl
          let outC := outAff.evalOnFlatBoxAsDim xB (hBoxDim.trans hIn.symm) hOut
          let loY : Spec.Tensor α yShape := by
            simpa [yShape] using outC.lo
          let hiY : Spec.Tensor α yShape := by
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
  let objV : Spec.Tensor α [outDim] :=
    Spec.Tensor.map cast (tensorOfArray! (ty := Float) [2] #[1.0, -1.0])
  let obj : FlatTensor α := { n := outDim, v := objV }
  match lowered.backwardObjectiveBox? ps ibp xB obj with
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
  NN.Verification.TorchLean.runWithBoundScalar "TorchLean → IR → IBP/CROWN robustness" args
    (@runOnce)

end NN.Verification.Robustness.TorchLean
