/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.Fno1d

/-!
# Fourier Neural Operators over Arbitrary Spatial Rank

This module implements a dense, correctness-oriented Fourier layer over any finite collection of
spatial axes. The transform phase is the sum of the per-axis phases, so this is the tensor-product
multidimensional DFT rather than a one-dimensional DFT of flattened storage. Real and imaginary
parts are represented by separate tensors, allowing the model to run over ordinary real scalar
backends. The spectral linear map is applied independently at every retained frequency.
-/

@[expose] public section

namespace Runtime.Autograd.TorchLean.NN.FNO

open Spec Tensor

/-- Tensor shape `spatial... × channels`. -/
abbrev fieldShape {d : Nat} (spatial : Vector Nat d) (channels : Nat) : Shape :=
  Shape.ofList (spatial.toList ++ [channels])

/-- Tensor shape of a scalar field over the spatial grid. -/
abbrev scalarFieldShape {d : Nat} (spatial : Vector Nat d) : Shape :=
  Shape.ofList spatial.toList

/-- Number of spatial grid points. -/
def gridSize {d : Nat} (spatial : Vector Nat d) : Nat :=
  spatial.toList.prod

/-- Matrix view that flattens the spatial axes and preserves the channel axis. -/
abbrev flatFieldShape {d : Nat} (spatial : Vector Nat d) (channels : Nat) : Shape :=
  .dim (gridSize spatial) (.dim channels .scalar)

/-- Learned frequency-wise channel maps, one `channels × channels` matrix per grid frequency. -/
abbrev spectralWeightShape {d : Nat} (spatial : Vector Nat d) (channels : Nat) : Shape :=
  .dim (gridSize spatial) (.dim channels (.dim channels .scalar))

namespace Internal

/-- Decode a row-major flat index into coordinates for the given axis extents. -/
def coordinates : List Nat → Nat → List Nat
  | [], _ => []
  | _ :: extents, flat =>
      let stride := extents.prod
      let coordinate := if stride = 0 then 0 else flat / stride
      let remainder := if stride = 0 then 0 else flat % stride
      coordinate :: coordinates extents remainder

/-- Sum `inputᵢ * frequencyᵢ / extentᵢ` over matching spatial coordinates. -/
def phaseFraction {α : Type} [Context α] : List Nat → List Nat → List Nat → α
  | extent :: extents, input :: inputs, frequency :: frequencies =>
      (input : α) * (frequency : α) / (extent : α) +
        phaseFraction extents inputs frequencies
  | _, _, _ => 0

/-- Tensor-product DFT phase for two flattened spatial coordinates. -/
def phase {α : Type} [Context α] (extents : List Nat) (input frequency : Nat) : α :=
  phaseFraction extents (coordinates extents input) (coordinates extents frequency)

/-- Cosine part of the dense multidimensional DFT matrix. -/
def dftCosMatrix {α : Type} [Context α] {d : Nat} (spatial : Vector Nat d) :
    Tensor α (FNO1D.mat (gridSize spatial) (gridSize spatial)) :=
  Tensor.dim (fun frequency =>
    Tensor.dim (fun input =>
      let angle := Numbers.two * MathFunctions.pi *
        phase (α := α) spatial.toList input.val frequency.val
      Tensor.scalar (MathFunctions.cos angle)))

/-- Negative-sine part of the dense multidimensional DFT matrix. -/
def dftNegSinMatrix {α : Type} [Context α] {d : Nat} (spatial : Vector Nat d) :
    Tensor α (FNO1D.mat (gridSize spatial) (gridSize spatial)) :=
  Tensor.dim (fun frequency =>
    Tensor.dim (fun input =>
      let angle := Numbers.two * MathFunctions.pi *
        phase (α := α) spatial.toList input.val frequency.val
      Tensor.scalar (0 - MathFunctions.sin angle)))

/-- Normalized cosine part of the dense multidimensional inverse DFT matrix. -/
def idftCosMatrix {α : Type} [Context α] {d : Nat} (spatial : Vector Nat d) :
    Tensor α (FNO1D.mat (gridSize spatial) (gridSize spatial)) :=
  Tensor.dim (fun input =>
    Tensor.dim (fun frequency =>
      let angle := Numbers.two * MathFunctions.pi *
        phase (α := α) spatial.toList input.val frequency.val
      Tensor.scalar (MathFunctions.cos angle / (gridSize spatial : α))))

/-- Normalized sine part of the dense multidimensional inverse DFT matrix. -/
def idftSinMatrix {α : Type} [Context α] {d : Nat} (spatial : Vector Nat d) :
    Tensor α (FNO1D.mat (gridSize spatial) (gridSize spatial)) :=
  Tensor.dim (fun input =>
    Tensor.dim (fun frequency =>
      let angle := Numbers.two * MathFunctions.pi *
        phase (α := α) spatial.toList input.val frequency.val
      Tensor.scalar (MathFunctions.sin angle / (gridSize spatial : α))))

/-- Whether one coordinate lies in the retained low- or high-frequency bands. -/
def keepCoordinate (extent modes coordinate : Nat) : Bool :=
  coordinate < modes || extent - modes ≤ coordinate

/--
Test membership in the rectangular frequency set selected by `modes`.

The flat index is decoded in the same outermost-first row-major order used by `Shape.ofList`.
-/
def keepFrequency : List Nat → List Nat → Nat → Bool
  | [], [], _ => true
  | extent :: extents, modes :: remainingModes, flat =>
      let stride := extents.prod
      let coordinate := if stride = 0 then 0 else flat / stride
      let remainder := if stride = 0 then 0 else flat % stride
      keepCoordinate extent modes coordinate && keepFrequency extents remainingModes remainder
  | _, _, _ => false

/-- Pointwise mask for the retained multidimensional Fourier modes. -/
def frequencyMask {α : Type} [Context α] {d channels : Nat}
    (spatial modes : Vector Nat d) : Tensor α (flatFieldShape spatial channels) :=
  Tensor.dim (fun frequency =>
    Tensor.dim (fun _ =>
      Tensor.scalar (if keepFrequency spatial.toList modes.toList frequency.val then 1 else 0)))

/-- Reshape a spatial field to its matrix view. -/
def flattenSpatial {d channels : Nat} (spatial : Vector Nat d) :
    Layer (fieldShape spatial channels) (flatFieldShape spatial channels) :=
  let source : Shape := fieldShape spatial channels
  let target : Shape := flatFieldShape spatial channels
  have sameSize : Shape.size source = Shape.size target := by
    simp [source, target, fieldShape, gridSize, Shape.size, List.prod_append]
  { kind := "ReshapeSpatial"
    stateShapes := []
    initState := .nil
    requiresGrad := []
    forward := fun _ {α} _ _ => fun {m} _ _ => fun x =>
      TorchLean.reshape (m := m) (α := α) (s₁ := source) (s₂ := target) x sameSize }

/-- Restore a matrix view to its spatial axes. -/
def restoreSpatial {d channels : Nat} (spatial : Vector Nat d) :
    Layer (flatFieldShape spatial channels) (fieldShape spatial channels) :=
  let source : Shape := flatFieldShape spatial channels
  let target : Shape := fieldShape spatial channels
  have sameSize : Shape.size source = Shape.size target := by
    simp [source, target, fieldShape, gridSize, Shape.size, List.prod_append]
  { kind := "RestoreSpatial"
    stateShapes := []
    initState := .nil
    requiresGrad := []
    forward := fun _ {α} _ _ => fun {m} _ _ => fun x =>
      TorchLean.reshape (m := m) (α := α) (s₁ := source) (s₂ := target) x sameSize }

/-- Add the singleton channel axis used inside an FNO model. -/
def addScalarChannel {d : Nat} (spatial : Vector Nat d) :
    Layer (scalarFieldShape spatial) (fieldShape spatial 1) :=
  let source : Shape := scalarFieldShape spatial
  let target : Shape := fieldShape spatial 1
  have sameSize : Shape.size source = Shape.size target := by
    simp [source, target, scalarFieldShape, fieldShape, List.prod_append]
  { kind := "AddScalarChannel"
    stateShapes := []
    initState := .nil
    requiresGrad := []
    forward := fun _ {α} _ _ => fun {m} _ _ => fun x =>
      TorchLean.reshape (m := m) (α := α) (s₁ := source) (s₂ := target) x sameSize }

/-- Remove the singleton channel axis after the output projection. -/
def removeScalarChannel {d : Nat} (spatial : Vector Nat d) :
    Layer (fieldShape spatial 1) (scalarFieldShape spatial) :=
  let source : Shape := fieldShape spatial 1
  let target : Shape := scalarFieldShape spatial
  have sameSize : Shape.size source = Shape.size target := by
    simp [source, target, scalarFieldShape, fieldShape, List.prod_append]
  { kind := "RemoveScalarChannel"
    stateShapes := []
    initState := .nil
    requiresGrad := []
    forward := fun _ {α} _ _ => fun {m} _ _ => fun x =>
      TorchLean.reshape (m := m) (α := α) (s₁ := source) (s₂ := target) x sameSize }

end Internal

/--
One multidimensional FNO block.

The first `d` axes are transformed. A learned channel map is applied at every retained frequency,
the discarded frequency rectangle is set to zero, and a pointwise affine skip is added after the
inverse transform.
-/
def block {d : Nat} (spatial modes : Vector Nat d) (width : Nat)
    (activation : FNO1D.Activation := .tanh) (seed : Nat := 0) :
    Layer (fieldShape spatial width) (fieldShape spatial width) :=
  let grid := gridSize spatial
  let field : Shape := fieldShape spatial width
  let flat : Shape := flatFieldShape spatial width
  let spectralShape : Shape := spectralWeightShape spatial width
  let skipShape : Shape := FNO1D.mat width width
  let biasShape : Shape := FNO1D.vec width
  let spectralReal0 : Tensor Float spectralShape :=
    Torch.Init.tensor (s := spectralShape) (sch := .uniform (-0.05) 0.05) (seed := seed)
  let spectralImag0 : Tensor Float spectralShape :=
    Torch.Init.tensor (s := spectralShape) (sch := .uniform (-0.05) 0.05) (seed := seed + 1)
  let skip0 : Tensor Float skipShape :=
    Torch.Init.tensor (s := skipShape) (sch := .uniform (-0.05) 0.05) (seed := seed + 2)
  let bias0 : Tensor Float biasShape :=
    Torch.Init.tensor (s := biasShape) (sch := .zeros) (seed := seed + 3)
  { kind := "FNOBlock"
    stateShapes := [spectralShape, spectralShape, skipShape, biasShape]
    initState := .cons spectralReal0 (.cons spectralImag0 (.cons skip0 (.cons bias0 .nil)))
    runtimeInit := some <| .cons (.uniform (-0.05) 0.05 seed) <|
      .cons (.uniform (-0.05) 0.05 (seed + 1)) <|
      .cons (.uniform (-0.05) 0.05 (seed + 2)) <| .cons .zeros .nil
    requiresGrad := [true, true, true, true]
    forward := fun mode {α} _ _ => fun {m} _ _ => fun spectralReal spectralImag skip bias x =>
      (show m (RefTy (m := m) (α := α) field) from do
        let xMatrix ← (Internal.flattenSpatial spatial).forward mode (α := α) (m := m) x
        let transformShape : Shape := FNO1D.mat grid grid
        let cosRef ← TorchLean.const (m := m) (α := α) (s := transformShape)
          (Internal.dftCosMatrix (α := α) spatial)
        let negSinRef ← TorchLean.const (m := m) (α := α) (s := transformShape)
          (Internal.dftNegSinMatrix (α := α) spatial)
        let inverseCosRef ← TorchLean.const (m := m) (α := α) (s := transformShape)
          (Internal.idftCosMatrix (α := α) spatial)
        let inverseSinRef ← TorchLean.const (m := m) (α := α) (s := transformShape)
          (Internal.idftSinMatrix (α := α) spatial)
        let xReal ← TorchLean.mm (m := m) (α := α)
          (mDim := grid) (nDim := grid) (pDim := width) cosRef xMatrix
        let xImag ← TorchLean.mm (m := m) (α := α)
          (mDim := grid) (nDim := grid) (pDim := width) negSinRef xMatrix
        let xRealBmm ← (FNO1D.reshapeModesMatToBmmIn grid width).forward mode
          (α := α) (m := m) xReal
        let xImagBmm ← (FNO1D.reshapeModesMatToBmmIn grid width).forward mode
          (α := α) (m := m) xImag
        let realReal ← TorchLean.bmm (m := m) (α := α)
          (batch := grid) (mDim := 1) (nDim := width) (pDim := width) xRealBmm spectralReal
        let imagImag ← TorchLean.bmm (m := m) (α := α)
          (batch := grid) (mDim := 1) (nDim := width) (pDim := width) xImagBmm spectralImag
        let realImag ← TorchLean.bmm (m := m) (α := α)
          (batch := grid) (mDim := 1) (nDim := width) (pDim := width) xRealBmm spectralImag
        let imagReal ← TorchLean.bmm (m := m) (α := α)
          (batch := grid) (mDim := 1) (nDim := width) (pDim := width) xImagBmm spectralReal
        let transformedRealBmm ← TorchLean.sub (m := m) (α := α)
          (s := .dim grid (.dim 1 (.dim width .scalar))) realReal imagImag
        let transformedImagBmm ← TorchLean.add (m := m) (α := α)
          (s := .dim grid (.dim 1 (.dim width .scalar))) realImag imagReal
        let transformedReal ← (FNO1D.reshapeBmmOutToModesMat grid width).forward mode
          (α := α) (m := m) transformedRealBmm
        let transformedImag ← (FNO1D.reshapeBmmOutToModesMat grid width).forward mode
          (α := α) (m := m) transformedImagBmm
        let mask : Tensor α flat := Internal.frequencyMask (channels := width) spatial modes
        let maskRef ← TorchLean.const (m := m) (α := α) (s := flat) mask
        let retainedReal ← TorchLean.mul (m := m) (α := α) (s := flat) transformedReal maskRef
        let retainedImag ← TorchLean.mul (m := m) (α := α) (s := flat) transformedImag maskRef
        let inverseReal ← TorchLean.mm (m := m) (α := α)
          (mDim := grid) (nDim := grid) (pDim := width) inverseCosRef retainedReal
        let inverseImag ← TorchLean.mm (m := m) (α := α)
          (mDim := grid) (nDim := grid) (pDim := width) inverseSinRef retainedImag
        let spectralMatrix ← TorchLean.sub (m := m) (α := α) (s := flat) inverseReal inverseImag
        let spectralResult ← (Internal.restoreSpatial spatial).forward mode
          (α := α) (m := m) spectralMatrix
        let skipMatrix ← TorchLean.mm (m := m) (α := α)
          (mDim := grid) (nDim := width) (pDim := width) xMatrix skip
        let biasBroadcast ← TorchLean.broadcastTo (m := m) (α := α)
          (s₁ := biasShape) (s₂ := flat) Shape.BroadcastTo.proof bias
        let skipBiased ← TorchLean.add (m := m) (α := α) (s := flat) skipMatrix biasBroadcast
        let skipField ← (Internal.restoreSpatial spatial).forward mode
          (α := α) (m := m) skipBiased
        let y ← TorchLean.add (m := m) (α := α) (s := field) spectralResult skipField
        match activation with
        | .tanh => TorchLean.tanh (m := m) (α := α) (s := field) y
        | .relu => TorchLean.relu (m := m) (α := α) (s := field) y) }

/-- Repeated multidimensional FNO blocks with deterministic, disjoint parameter seeds. -/
def blocks {d : Nat} (spatial modes : Vector Nat d) (width blockCount : Nat)
    (activation : FNO1D.Activation := .tanh) (seed : Nat := 0) :
    Seq (fieldShape spatial width) (fieldShape spatial width) :=
  match blockCount with
  | 0 => Seq.id _
  | count + 1 =>
      Seq.cons (block spatial modes width activation (seed + 10 * count))
        (blocks spatial modes width count activation seed)

/--
FNO mapping one scalar field on `spatial` to another scalar field on the same grid.

The dense reference carries real and imaginary Fourier components explicitly, so it runs over real
scalar backends. Fused implementations may be selected through backend-specific kernel capsules.
-/
def model {d : Nat} (spatial modes : Vector Nat d) (width blockCount : Nat)
    (activation : FNO1D.Activation := .tanh) (seed : Nat := 0) :
    Seq (scalarFieldShape spatial) (scalarFieldShape spatial) :=
  let grid := gridSize spatial
  let lift := FNO1D.matAffine grid 1 width seed (seed + 1)
  let project := FNO1D.matAffine grid width 1 (seed + 100) (seed + 101)
  Seq.cons (Internal.addScalarChannel spatial) <|
    Seq.cons (Internal.flattenSpatial spatial) <|
      Seq.cons lift <|
        Seq.cons (Internal.restoreSpatial spatial) <|
          Seq.comp (blocks spatial modes width blockCount activation seed) <|
            Seq.cons (Internal.flattenSpatial spatial) <|
              Seq.cons project <|
                Seq.cons (Internal.restoreSpatial spatial) <|
                  Seq.cons (Internal.removeScalarChannel spatial) (Seq.id _)

end Runtime.Autograd.TorchLean.NN.FNO
