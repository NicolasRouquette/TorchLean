/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Neural.Leading

/-!
# Normalization

Feature and arbitrary-rank channel normalization layer constructors.
-/

@[expose] public section

namespace TorchLean
namespace nn
namespace Internal

/--
LayerNorm parameter initialization.

PyTorch analogue: `torch.nn.LayerNorm`.
See `https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html`.
-/
structure LayerNormConfig where
  /-- Seed for deterministic initialization of `gamma` (scale). -/
  seedGamma : Nat := 0
  /-- Seed for deterministic initialization of `beta` (shift). -/
  seedBeta : Nat := 0

/--
Layer normalization over the final axis of a tensor.

Every index in `leading` selects one width-`width` vector. Empty leading axes are allowed; only the
normalized axis must be nonempty.
-/
def layerNorm (leading : List Nat := []) {width : Nat} (cfg : LayerNormConfig := {})
    [NeZero width] : Sequential (leading ++ [width]) (leading ++ [width]) := by
  simpa only [Spec.Shape.ofList_append, Spec.Shape.appendDim_eq_concat] using
    (of <| _root_.Runtime.Autograd.TorchLean.NN.layerNorm (Spec.Shape.ofList leading) width
      (hWidth := Nat.pos_of_ne_zero (NeZero.ne (n := width)))
      (seedGamma := cfg.seedGamma) (seedBeta := cfg.seedBeta))

/-- RMSNorm parameter initialization. -/
structure RmsNormConfig where
  /-- Seed for deterministic initialization of `gamma` (scale). -/
  seedGamma : Nat := 0

/-- RMS normalization over the final axis of a tensor. -/
def rmsNorm (leading : List Nat := []) {width : Nat} (cfg : RmsNormConfig := {})
    [NeZero width] : Sequential (leading ++ [width]) (leading ++ [width]) := by
  simpa only [Spec.Shape.ofList_append, Spec.Shape.appendDim_eq_concat] using
    (of <| _root_.Runtime.Autograd.TorchLean.NN.rmsNorm (Spec.Shape.ofList leading) width
      (hWidth := Nat.pos_of_ne_zero (NeZero.ne (n := width)))
      (seedGamma := cfg.seedGamma))

/-- Parameter initialization for affine channel normalization. -/
structure ChannelNormConfig where
  seedGamma : Nat := 0
  seedBeta : Nat := 0

/-- Batch normalization over `(leading..., channels, spatial...)` for any spatial rank. -/
def batchNorm (leading : List Nat := []) {d channels : Nat}
    (spatial : Tensor Nat [d]) (cfg : ChannelNormConfig := {})
    [NeZero leading.prod] [NeZero channels] [NeZero spatial.prod] :
    Sequential
      (leading ++ channels :: spatial.toList)
      (leading ++ channels :: spatial.toList) := by
  let leadingShape := Spec.Shape.ofList leading
  let spatialShape := Spec.Shape.ofList spatial.toList
  let batch := Spec.Shape.size leadingShape
  letI : NeZero batch := by simpa [batch, leadingShape, Spec.Shape.size_ofList]
  letI : NeZero (Spec.Shape.size spatialShape) := by
    constructor
    simpa [spatialShape, Spec.Shape.size_ofList, ← Spec.Tensor.prod_eq_toList_prod] using
      NeZero.ne spatial.prod
  let hInput :
      ((spatialShape.prependDim channels).prependDim batch).wellFormed :=
    ⟨Nat.pos_of_ne_zero (NeZero.ne batch),
      Nat.pos_of_ne_zero (NeZero.ne channels),
      Spec.Shape.wellFormed_of_size_pos
        (Nat.pos_of_ne_zero (NeZero.ne (Spec.Shape.size spatialShape)))⟩
  simpa only [Spec.Shape.ofList_append] using
    (of <| adaptLeadingShape leadingShape <|
      _root_.Runtime.Autograd.TorchLean.NN.batchNorm batch channels spatialShape hInput
        cfg.seedGamma cfg.seedBeta)

/-- Instance normalization over `(leading..., channels, spatial...)` for any spatial rank. -/
def instanceNorm (leading : List Nat := []) {d channels : Nat}
    (spatial : Tensor Nat [d]) (cfg : ChannelNormConfig := {})
    [NeZero leading.prod] [NeZero channels] [NeZero spatial.prod] :
    Sequential
      (leading ++ channels :: spatial.toList)
      (leading ++ channels :: spatial.toList) := by
  let leadingShape := Spec.Shape.ofList leading
  let spatialShape := Spec.Shape.ofList spatial.toList
  let batch := Spec.Shape.size leadingShape
  letI : NeZero batch := by simpa [batch, leadingShape, Spec.Shape.size_ofList]
  letI : NeZero (Spec.Shape.size spatialShape) := by
    constructor
    simpa [spatialShape, Spec.Shape.size_ofList, ← Spec.Tensor.prod_eq_toList_prod] using
      NeZero.ne spatial.prod
  let hInput :
      ((spatialShape.prependDim channels).prependDim batch).wellFormed :=
    ⟨Nat.pos_of_ne_zero (NeZero.ne batch),
      Nat.pos_of_ne_zero (NeZero.ne channels),
      Spec.Shape.wellFormed_of_size_pos
        (Nat.pos_of_ne_zero (NeZero.ne (Spec.Shape.size spatialShape)))⟩
  simpa only [Spec.Shape.ofList_append] using
    (of <| adaptLeadingShape leadingShape <|
      _root_.Runtime.Autograd.TorchLean.NN.instanceNorm batch channels spatialShape hInput
        cfg.seedGamma cfg.seedBeta)

/-- Group normalization over `(leading..., channels, spatial...)` for any spatial rank. -/
def groupNorm (leading : List Nat := []) {d channels : Nat}
    (spatial : Tensor Nat [d]) (groups : Nat) (hGroups : groups > 0)
    (hGroupsLe : channels ≥ groups) (hDiv : channels % groups = 0)
    (cfg : ChannelNormConfig := {}) [NeZero leading.prod] [NeZero channels]
    [NeZero spatial.prod] :
    Sequential
      (leading ++ channels :: spatial.toList)
      (leading ++ channels :: spatial.toList) := by
  let leadingShape := Spec.Shape.ofList leading
  let spatialShape := Spec.Shape.ofList spatial.toList
  let batch := Spec.Shape.size leadingShape
  letI : NeZero batch := by simpa [batch, leadingShape, Spec.Shape.size_ofList]
  letI : NeZero (Spec.Shape.size spatialShape) := by
    constructor
    simpa [spatialShape, Spec.Shape.size_ofList, ← Spec.Tensor.prod_eq_toList_prod] using
      NeZero.ne spatial.prod
  let hInput :
      ((spatialShape.prependDim channels).prependDim batch).wellFormed :=
    ⟨Nat.pos_of_ne_zero (NeZero.ne batch),
      Nat.pos_of_ne_zero (NeZero.ne channels),
      Spec.Shape.wellFormed_of_size_pos
        (Nat.pos_of_ne_zero (NeZero.ne (Spec.Shape.size spatialShape)))⟩
  simpa only [Spec.Shape.ofList_append] using
    (of <| adaptLeadingShape leadingShape <|
      _root_.Runtime.Autograd.TorchLean.NN.groupNorm batch channels groups spatialShape
        hInput hGroups hGroupsLe hDiv cfg.seedGamma cfg.seedBeta)
