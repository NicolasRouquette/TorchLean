/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded
public import NN.Tensor

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
  shapes.foldl (fun acc s => acc + Shape.size s) 0

/-- Number of trainable scalar elements in a model-state shape list. -/
def trainableCount (shapes : List Shape) (flags : Array Bool) : Nat :=
  let rec go : List Shape → Nat → Nat
    | [], _ => 0
    | shape :: rest, index =>
        (if flags[index]?.getD false then Shape.size shape else 0) + go rest (index + 1)
  go shapes 0

/-- User-facing tensor shape display for one model-summary shape. -/
def shapeDisplay (s : Shape) : String :=
  match Shape.toList s with
  | [] => "scalar"
  | dims => "[" ++ String.intercalate ", " (dims.map toString) ++ "]"

/-- User-facing display for an array of tensor shapes. -/
def shapeArrayString (shapes : Array Shape) : String :=
  match shapes with
  | #[] => "[]"
  | _ => "[" ++ String.intercalate ", " (shapes.map shapeDisplay).toList ++ "]"

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
  stateShapes : Array Shape
  /-- Number of trainable scalar parameters in this layer. -/
  paramCount : Nat
  /-- Number of scalar elements in this layer's complete state. -/
  stateCount : Nat

namespace LayerSummary

/-- One-line rendering of one layer summary. -/
def render (s : LayerSummary) : String :=
  s!"  [{s.index}] {s.kind}: {Internal.shapeDisplay s.inputShape} -> " ++
    s!"{Internal.shapeDisplay s.outputShape} params={s.paramCount}, state={s.stateCount} " ++
    Internal.shapeArrayString s.stateShapes

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
  layers : Array LayerSummary
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
  String.intercalate "\n" (#[header s] ++ s.layers.map LayerSummary.render).toList

instance : ToString ModelSummary where
  toString := render

end ModelSummary

namespace Internal

/-- Recursive worker that records one summary row for each layer in a sequential model. -/
def layerSummaries :
    {σ τ : Shape} → Nat → Sequential σ τ → Array LayerSummary
  | _, _, _, .id _ => #[]
  | σ, _, i, .cons (τ := τ') layer rest =>
      let row : LayerSummary :=
        { index := i
          kind := layer.kind
          inputShape := σ
          outputShape := τ'
          stateShapes := layer.stateShapes.toArray
          paramCount := trainableCount layer.stateShapes layer.requiresGrad
          stateCount := elementCount layer.stateShapes }
      #[row] ++
      layerSummaries (i + 1) rest

end Internal

/-- Structured checked model summary for a sequential model. -/
def summary {σ τ : Shape} (model : Sequential σ τ) : ModelSummary :=
  let layers := Internal.layerSummaries 0 model
  { inputShape := σ
    outputShape := τ
    layers := layers
    layerCount := layers.size
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
