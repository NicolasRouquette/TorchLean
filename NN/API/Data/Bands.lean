/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Data.Synthetic
public import NN.API.Data.Training

/-!
# Synthetic Band Dataset

Several TorchLean examples use a compact 4×4 image classification task:

- class `0`: a vertical band
- class `1`: a horizontal band

This domain module owns both the renderer and the canonical 4×4 dataset. Keeping these definitions
out of `TorchLean.Data.Synthetic` prevents a particular image layout from becoming part of TorchLean's
general tensor and sample abstractions.
-/

@[expose] public section

namespace TorchLean.Data

open Spec

namespace Bands

/-! ## Renderer -/

/-- Spatial axis along which a synthetic band varies. -/
inductive Axis
  | row
  | column
  deriving Repr, DecidableEq

/-- Human-readable class name associated with an axis. -/
def Axis.name : Axis → String
  | .row => "horizontal"
  | .column => "vertical"

/--
Render a channel-first rank-three tensor from an in-bounds scalar function.

This shape is part of this particular dataset, not a restriction on TorchLean tensors or models.
-/
def render (channels height width : Nat)
    (value : Fin channels → Fin height → Fin width → Float) :
    Spec.Tensor Float (.dim channels (.dim height (.dim width .scalar))) :=
  Spec.Tensor.dim (fun channel =>
    Spec.Tensor.dim (fun row =>
      Spec.Tensor.dim (fun column =>
        Spec.Tensor.scalar (value channel row column))))

/-- Render a binary-valued channel-first tensor from a finite predicate. -/
def renderBinary (channels height width : Nat)
    (selected : Fin channels → Fin height → Fin width → Bool)
    (onValue : Float := 1.0) (offValue : Float := 0.0) :
    Spec.Tensor Float (.dim channels (.dim height (.dim width .scalar))) :=
  render channels height width fun channel row column =>
    if selected channel row column then onValue else offValue

/-- Render a single-channel horizontal or vertical band. -/
def renderBand (height width : Nat) (axis : Axis) (offset : Nat) (thickness : Nat := 2)
    (onValue : Float := 1.0) (offValue : Float := 0.0) :
    Spec.Tensor Float (.dim 1 (.dim height (.dim width .scalar))) :=
  renderBinary 1 height width
    (match axis with
    | .row => fun _ row _ => offset ≤ row.1 ∧ row.1 < offset + thickness
    | .column => fun _ _ column => offset ≤ column.1 ∧ column.1 < offset + thickness)
    onValue offValue

/-! ## Classes and datasets -/

/-- Label metadata for one family of synthetic bands. -/
structure Class where
  /-- Axis occupied by the band. -/
  axis : Axis
  /-- Class label for the two-class band task. -/
  label : Fin 2
  /-- Display name used by reports. -/
  name : String

/-- Construct the vertical-band class. -/
def vertical (name : String := "vertical") : Class :=
  { axis := .column, label := 0, name }

/-- Construct the horizontal-band class. -/
def horizontal (name : String := "horizontal") : Class :=
  { axis := .row, label := 1, name }

/-- Generate `(tensor, label)` samples for every class/offset pair. -/
def samples (height width : Nat) (classes : List Class) (offsets : List Nat)
    (thickness : Nat := 2) :
    List (Spec.Tensor Float (.dim 1 (.dim height (.dim width .scalar))) × Fin 2) :=
  classes.foldr
    (fun cls acc =>
      offsets.map (fun offset => (renderBand height width cls.axis offset thickness, cls.label)) ++
        acc)
    []

/-- Generate named samples for reports and prediction probes. -/
def namedSamples (height width : Nat) (specs : List (Class × Nat)) (thickness : Nat := 2) :
    List (String × Spec.Tensor Float (.dim 1 (.dim height (.dim width .scalar))) × Nat) :=
  specs.map fun (cls, offset) =>
    (s!"{cls.name}-{offset}", renderBand height width cls.axis offset thickness, cls.label.val)

/-- Canonical label set for the band dataset: vertical ↦ `0`, horizontal ↦ `1`. -/
def classes : List Class :=
  [ vertical
  , horizontal
  ]

/-! ### Typed Tensors (Tensor-First) -/

/-- Canonical image shape for the band dataset (single-channel 4×4). -/
abbrev shape : Spec.Shape := .dim 1 (.dim 4 (.dim 4 .scalar))

/-- Training set samples: a small list of `(x, label)` pairs. -/
def trainFloat : List (Spec.Tensor Float shape × Fin 2) :=
  samples 4 4 classes [0, 1, 2]

/-- Probe set for reporting: `(name, x, expectedLabel)` triples. -/
def probesFloat : List (String × Spec.Tensor Float shape × Nat) :=
  namedSamples 4 4
    [ (vertical, 1)
    , (vertical, 2)
    , (horizontal, 1)
    , (horizontal, 2)
    ]

/-- Small vertical-versus-horizontal dataset with one-hot class targets. -/
def dataset : Trainer.DataSource shape (.dim 2 .scalar) :=
  { build := fun {α} _ =>
      pure <| labeled (α := α) (σ := shape) 2 trainFloat }

/-- Named vertical and horizontal inputs used to inspect a trained classifier. -/
def probes : List (Trainer.ClassProbe shape) :=
  probesFloat.map (fun (name, xF, expected) =>
    { name := name
      input := fun {α} _ _ =>
        TorchLean.Tensor.castFloat (_root_.TorchLean.Runtime.ofFloat (α := α)) xF
      expected := expected })

/-- Concrete `Float` probe inputs for prediction examples. -/
def probeSamples : List (String × Tensor Float shape × Nat) :=
  probesFloat

end Bands
end TorchLean.Data
