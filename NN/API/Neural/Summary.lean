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

namespace Internal

/-- Number of scalar elements represented by a list of tensor shapes. -/
def elementCount (shapes : List Shape) : Nat :=
  shapes.foldl (fun acc s => acc + Spec.Shape.size s) 0

/-- Number of trainable scalar elements in a model-state shape list. -/
def trainableCount : List Shape → List Bool → Nat
  | shape :: shapes, true :: flags => Spec.Shape.size shape + trainableCount shapes flags
  | _ :: shapes, false :: flags => trainableCount shapes flags
  | _, _ => 0

/-- User-facing tensor shape display for one model-summary shape. -/
def shapeDisplay (s : Shape) : String :=
  match Spec.Shape.toList s with
  | [] => "scalar"
  | dims => "[" ++ String.intercalate ", " (dims.map toString) ++ "]"

/-- User-facing display for a list of tensor shapes. -/
def shapeListString (shapes : List Shape) : String :=
  match shapes with
  | [] => "[]"
  | _ => "[" ++ String.intercalate ", " (shapes.map shapeDisplay) ++ "]"

end Internal

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
  /-- Trainable parameter and persistent-buffer shapes owned by this layer. -/
  stateShapes : List Shape
  /-- Number of trainable scalar parameters in this layer. -/
  paramCount : Nat
  /-- Number of scalar elements in this layer's complete state. -/
  stateCount : Nat

namespace LayerSummary

/-- One-line rendering of one layer summary. -/
def render (s : LayerSummary) : String :=
  s!"  [{s.index}] {s.kind}: {Internal.shapeDisplay s.inputShape} -> " ++
    s!"{Internal.shapeDisplay s.outputShape} params={s.paramCount}, state={s.stateCount} " ++
    Internal.shapeListString s.stateShapes

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
  /-- Total scalar element count across trainable parameters and persistent buffers. -/
  totalState : Nat

namespace ModelSummary

/-- Header line for the model summary. -/
def header (s : ModelSummary) : String :=
  s!"Sequential: {Internal.shapeDisplay s.inputShape} -> " ++
    s!"{Internal.shapeDisplay s.outputShape}, " ++
    s!"layers={s.layerCount}, params={s.totalParams}, state={s.totalState}"

/-- Multi-line rendering of the structured model summary. -/
def render (s : ModelSummary) : String :=
  String.intercalate "\n" (header s :: s.layers.map LayerSummary.render)

instance : ToString ModelSummary where
  toString := render

end ModelSummary

namespace Internal

/-- Recursive worker that records one summary row for each layer in a sequential model. -/
def layerSummaries :
    {σ τ : Shape} → Nat → Sequential σ τ → List LayerSummary
  | _, _, _, .id _ => []
  | σ, _, i, .cons (τ := τ') layer rest =>
      { index := i
        kind := layer.kind
        inputShape := σ
        outputShape := τ'
        stateShapes := layer.stateShapes
        paramCount := trainableCount layer.stateShapes layer.requiresGrad
        stateCount := elementCount layer.stateShapes } ::
      layerSummaries (i + 1) rest

end Internal

/-- Structured checked model summary for a sequential model. -/
def summary {σ τ : Shape} (model : Sequential σ τ) : ModelSummary :=
  let layers := Internal.layerSummaries 0 model
  { inputShape := σ
    outputShape := τ
    layers := layers
    layerCount := layers.length
    totalParams := Internal.trainableCount (stateShapes model) (requiresGrad model)
    totalState := Internal.elementCount (stateShapes model) }

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
