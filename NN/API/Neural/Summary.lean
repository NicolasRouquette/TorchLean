/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded
public import NN.API.Tensor

/-!
# Model Summaries

Structured model-summary rendering for checked sequential models.
-/

@[expose] public section

namespace TorchLean

namespace nn

def paramCount (shapes : List Shape) : Nat :=
  shapes.foldl (fun acc s => acc + Spec.Shape.size s) 0

/-- Dimensions of a tensor shape, outermost first. Scalars have no dimensions. -/
def shapeDims : Shape → List Nat
  | .scalar => []
  | .dim n rest => n :: shapeDims rest

/-- User-facing tensor shape display for one model-summary shape. -/
def shapeDisplay (s : Shape) : String :=
  match shapeDims s with
  | [] => "scalar"
  | dims => "[" ++ String.intercalate ", " (dims.map toString) ++ "]"

/-- User-facing display for a list of parameter tensor shapes. -/
def shapeListString (shapes : List Shape) : String :=
  match shapes with
  | [] => "[]"
  | _ => "[" ++ String.intercalate ", " (shapes.map shapeDisplay) ++ "]"

/-- Structured per-layer summary derived from a checked sequential model. -/
structure LayerSummary where
  /-- Zero-based position in the sequential layer list. -/
  index : Nat
  /-- User-facing layer kind string. -/
  kind : String
  /-- Checked input shape for this layer. -/
  inputShape : Shape
  /-- Checked output shape for this layer. -/
  outputShape : Shape
  /-- Parameter tensor shapes owned by this layer. -/
  paramShapes : List Shape
  /-- Total scalar parameter count for this layer. -/
  paramCount : Nat

namespace LayerSummary

/-- One-line rendering of one layer summary. -/
def render (s : LayerSummary) : String :=
  s!"  [{s.index}] {s.kind}: {shapeDisplay s.inputShape} -> {shapeDisplay s.outputShape} " ++
    s!"params={s.paramCount} {shapeListString s.paramShapes}"

instance : ToString LayerSummary where
  toString := render

end LayerSummary

/-- Structured whole-model summary derived from a checked sequential model. -/
structure ModelSummary where
  /-- Checked input shape for the full model. -/
  inputShape : Shape
  /-- Checked output shape for the full model. -/
  outputShape : Shape
  /-- Per-layer summaries in order. -/
  layers : List LayerSummary
  /-- Total number of layers in the sequential model. -/
  layerCount : Nat
  /-- Total scalar parameter count across all layers. -/
  totalParams : Nat

namespace ModelSummary

/-- Header line for the model summary. -/
def header (s : ModelSummary) : String :=
  s!"Sequential: {shapeDisplay s.inputShape} -> {shapeDisplay s.outputShape}, " ++
    s!"layers={s.layerCount}, params={s.totalParams}"

/-- Multi-line rendering of the structured model summary. -/
def render (s : ModelSummary) : String :=
  String.intercalate "\n" (header s :: s.layers.map LayerSummary.render)

instance : ToString ModelSummary where
  toString := render

end ModelSummary

/-- Recursive worker that records one summary row for each layer in a sequential model. -/
def layerSummariesAux :
    {σ τ : Shape} → Nat → Sequential σ τ → List LayerSummary
  | _, _, _, .id _ => []
  | σ, _, i, .cons (τ := τ') layer rest =>
      { index := i
        kind := layer.kind
        inputShape := σ
        outputShape := τ'
        paramShapes := layer.paramShapes
        paramCount := paramCount layer.paramShapes } ::
      layerSummariesAux (i + 1) rest

/-- Structured checked model summary for a sequential model. -/
def summary {σ τ : Shape} (model : Sequential σ τ) : ModelSummary :=
  let layers := layerSummariesAux 0 model
  { inputShape := σ
    outputShape := τ
    layers := layers
    layerCount := layers.length
    totalParams := paramCount (paramShapes model) }

/--
Model description derived from a sequential model value.

This walks the checked `Seq` itself, so the printed layer list stays attached to the model that will
actually run.
-/
def info {σ τ : Shape} (model : Sequential σ τ) : String :=
  (summary model).render

/-- Print the checked model description for a sequential model. -/
def printInfo {σ τ : Shape} (model : Sequential σ τ) : IO Unit :=
  IO.println (info model)

end nn

end TorchLean
