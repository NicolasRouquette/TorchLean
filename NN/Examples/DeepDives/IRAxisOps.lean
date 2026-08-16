/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.IR.Check
public import NN.Runtime.Autograd.IRExec

/-!
# IR axis operations

IR axis-ops runtime tutorial.

This example is a small regression guard for three ops where TorchLean’s IR uses an explicit `axis`:

- `softmax axis` (PyTorch: `torch.softmax(x, dim=axis)`)
- `concat axis` (PyTorch: `torch.cat(xs, dim=axis)`)
- `layernorm axis`
  PyTorch: `F.layer_norm(x, normalized_shape=x.shape[axis:])`

Why this tutorial exists:

* These three ops are easy to accidentally restrict to “last axis only” (because the spec primitives
  we reuse are last-axis).
* The denotational IR semantics supports the PyTorch meaning on *any* valid axis:
  it implements non-last axes by reshaping/permuting into a form the spec primitive already
  supports.
* Forward-graph IR execution is more conservative today. This tutorial runs it only for
  supported cases and prints an explicit skip for known backend gaps, instead of treating the
  backend covers more than it does.

Run:

`lake exe torchlean ir_axis_ops --scalar float32 --execution eager`
-/

@[expose] public section

namespace NN.Examples.DeepDives.IRAxisOps

open TorchLean
open NN.IR

open Runtime.Autograd.TypedGraph

/-- Command-line help for the IR axis-ops tutorial. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean IR axis-ops tutorial"
    , ""
    , "Usage:"
    , "  lake exe torchlean ir_axis_ops [options]"
    , ""
    , "Options:"
    , "  --scalar float32|ieee32-exec|complex64"
    , "  --execution eager|typed-graph"
    , "  --device auto|cpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external"
    , "  --show-backend                    print backend capsules as they execute"
    ]

/-!
## Test Shapes

We keep shapes compact while still exercising the “axis is not last / not 0” code paths.
-/

abbrev baseRankThreeShape : Shape := NN.Tensor.shapeOfDims [2, 3, 4]
abbrev widerMiddleAxisShape : Shape := NN.Tensor.shapeOfDims [2, 5, 4]
abbrev concatenatedMiddleAxisShape : Shape := NN.Tensor.shapeOfDims [2, 8, 4]

/-!
## Small IR Graphs
-/

def softmaxMiddleAxisGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := [], kind := .input, outShape := baseRankThreeShape }
    , { id := 1, parents := [0], kind := .softmax (axis := 1), outShape := baseRankThreeShape }
    ] }

def layerNormMiddleAxisGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := [], kind := .input, outShape := baseRankThreeShape }
    , { id := 1, parents := [0], kind := .layernorm (axis := 1), outShape := baseRankThreeShape }
    ] }

def concatMiddleAxisGraph : NN.IR.Graph :=
  -- The concat example uses `rand_uniform` sources, so the input node only fixes the graph's
  -- single-input interface for the shared runner.
  { nodes := #[
      { id := 0, parents := [], kind := .input, outShape := Spec.Shape.scalar }
    , { id := 1, parents := [], kind := .randUniform (seed := 0), outShape := baseRankThreeShape }
    , { id := 2, parents := [], kind := .randUniform (seed := 1), outShape := widerMiddleAxisShape }
    , { id := 3, parents := [1, 2], kind := .concat (axis := 1), outShape := concatenatedMiddleAxisShape }
    ] }

/-!
## Runner Helpers
-/

def checkIR (tag : String) (g : NN.IR.Graph) : IO Unit := do
  CLI.orThrow s!"{tag}:checkWellFormed" <| g.checkWellFormed
  CLI.orThrow s!"{tag}:checkShapes" <| NN.IR.Graph.checkShapes g

def firstScalars {α : Type} [ToString α] : List α → String
  | [] => "[]"
  | xs =>
      let ys := xs.take 8
      "[" ++ ", ".intercalate (ys.map toString) ++ (if xs.length > ys.length then ", ..." else "")
        ++ "]"

def runOne
    {α : Type} [_root_.Context α] [DecidableEq Shape] [ToString α] [Runtime.FromFloat α]
    (tag : String)
    (g : NN.IR.Graph)
    (payload : NN.IR.Payload α)
    (inputShape : Shape)
    (x : Spec.Tensor α inputShape)
    (outputId : Fin g.nodes.size)
    (runForward : Bool := true) : IO Unit := do
  checkIR tag g

  -- Spec semantics (denotational model).
  let input : NN.IR.DVal α := NN.IR.DVal.mk (α := α) inputShape x
  let outSpec ← CLI.orThrow s!"{tag}:spec" <|
    NN.IR.Graph.denote (α := α) (g := g) (payload := payload) (input := input) (outputId :=
      outputId)
  let ⟨sSpec, tSpec⟩ := outSpec
  IO.println s!"[{tag}] spec outShape: {repr sSpec}"
  IO.println s!"[{tag}] spec first scalars: {firstScalars (Spec.toList tSpec)}"

  if !runForward then
    IO.println s!"[{tag}] IR execution skipped: current IRExec path supports fewer axis cases than the spec semantics."
    return ()

  -- Lower the IR to its forward-only executable graph, then evaluate it.
  let eg ← CLI.orThrow s!"{tag}:forward-graph" <|
    Runtime.Autograd.IRExec.lowerToForwardGraph (α := α) (g := g) (payload := payload)
  let xExec : Spec.Tensor α eg.inShape ←
    if hIn : inputShape = eg.inShape then
      pure <| Spec.Tensor.castShape (t := x) hIn
    else
      -- Well-formed tutorials keep these shapes aligned; this branch keeps the error readable if a
      -- graph edit and the `inputShape` argument drift apart.
      throw <| IO.userError
        s!"{tag}: inputShape mismatch: arg={repr inputShape}, graph={repr eg.inShape}"
  let valsExec := Runtime.Autograd.IRExec.ForwardGraph.denoteAll (α := α) eg xExec
  let outExec ←
    if hOut : outputId.1 < valsExec.size then
      pure <| valsExec[outputId.1]'hOut
    else
      throw <| IO.userError
        s!"{tag}: forward graph output index out of bounds: index={outputId.1}, size={valsExec.size}"
  let ⟨sExec, tExec⟩ := outExec
  IO.println s!"[{tag}] forward graph outShape: {repr sExec}"
  IO.println s!"[{tag}] forward graph first scalars: {firstScalars (Spec.toList tExec)}"

  if sExec = sSpec then
    pure ()
  else
    throw <| IO.userError
      s!"{tag}: spec/forward-graph outShape mismatch: spec={repr sSpec}, forwardGraph={repr sExec}"

def runOnce
    {α : Type} [_root_.Context α] [DecidableEq Shape] [ToString α] [Runtime.FromFloat α]
    : IO Unit := do
  let payload : NN.IR.Payload α := {}

  -- A small but nontrivial 2×3×4 input tensor.
  let x234 : Spec.Tensor α baseRankThreeShape :=
    Tensor.generateFromFloat (α := α) Runtime.ofFloat [2, 3, 4] (fun i =>
      (Float.ofNat i) / 10.0 - 1.0)

  IO.println ""
  IO.println "== IR axis ops tutorial =="
  IO.println "This checks shape validation and compares the spec semantics with forward-graph execution."

  IO.println ""
  IO.println "-- softmax axis=1 on shape [2,3,4]"
  runOne (α := α) (tag := "softmax_middle_axis") (g := softmaxMiddleAxisGraph) (payload := payload)
    (inputShape := baseRankThreeShape) (x := x234) (outputId := ⟨1, by decide⟩) (runForward := false)

  IO.println ""
  IO.println "-- layernorm axis=1 on rank-3 shape [2,3,4]"
  IO.println "PyTorch meaning: normalized_shape = x.shape[axis:] = [3,4]"
  runOne (α := α) (tag := "layernorm_rank3_middle_axis") (g := layerNormMiddleAxisGraph) (payload := payload)
    (inputShape := baseRankThreeShape) (x := x234) (outputId := ⟨1, by decide⟩) (runForward := false)

  IO.println ""
  IO.println "-- concat axis=1: [2,3,4] ++ [2,5,4] -> [2,8,4]"
  let x0 : Spec.Tensor α Spec.Shape.scalar :=
    Spec.Tensor.scalar (Runtime.ofFloat (α := α) 0.0)
  runOne (α := α) (tag := "concat_middle_axis") (g := concatMiddleAxisGraph) (payload := payload)
    (inputShape := Spec.Shape.scalar) (x := x0) (outputId := ⟨3, by decide⟩)

/-- Runtime-selected entrypoint body for the axis-ops tutorial. -/
def runSelected
    {α : Type} [_root_.Context α] [DecidableEq Shape] [ToString α] [Runtime.FromFloat α]
    (rest : List String) : IO Unit := do
  CLI.requireNoArgs "ir_axis_ops" rest
  runOnce (α := α)

def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return
  Module.withRuntime args
    (fun {α} _ _ _ _ _cast _opts rest => runSelected (α := α) rest)

end NN.Examples.DeepDives.IRAxisOps
