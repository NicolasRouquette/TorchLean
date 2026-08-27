/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine

/-!
# CROWN Soundness Guardrails

Regression checks for IR configurations whose semantics are intentionally outside the current
flattened CROWN engine. Unsupported convolution geometry, non-leading-axis concat, and
payload-bearing LayerNorm must fail closed, while the supported dense convolution fragment remains
available.
-/

@[expose] public section

namespace NN.Tests.MLTheory.CROWNSoundnessGuardrails

open _root_.Spec
open NN.IR
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph

def expectNoBoundAt {β : Type} (label : String) (values : Array (Option β)) (id : Nat) : IO Unit :=
  match values[id]? with
  | some none => pure ()
  | _ => throw <| IO.userError s!"{label}: expected node {id} to have no bound"

def expectBoundAt {β : Type} (label : String) (values : Array (Option β)) (id : Nat) : IO Unit :=
  match values[id]? with
  | some (some _) => pure ()
  | _ => throw <| IO.userError s!"{label}: expected node {id} to have a bound"

def pointBox {s : Shape} (value : Tensor Float s) : FlatBox Float :=
  FlatBox.ofTensor (Tensor.flattenSpec value)

def checkConvolutionGuards : IO Unit := do
  let inputSpatial : Tensor Nat [1] := Tensor.ofArrayExact #[4] (by simp)
  let kernel : Tensor Nat [1] := Tensor.ofArrayExact #[2] (by simp)
  let stride : Tensor Nat [1] := Tensor.ofArrayExact #[1] (by simp)
  let padding : Tensor Nat [1] := Tensor.ofArrayExact #[1] (by simp)
  let dilationOne : Tensor Nat [1] := Tensor.ofArrayExact #[1] (by simp)
  let dilationTwo : Tensor Nat [1] := Tensor.ofArrayExact #[2] (by simp)
  let paddingAfterZero : Tensor Nat [1] := Tensor.ofArrayExact #[0] (by simp)
  let weights : Tensor Float [2, 2, 2] := Spec.fill (α := Float) 1 [2, 2, 2]
  let bias : Tensor Float [2] := Spec.fill (α := Float) 0 [2]
  have hKernel : ∀ i : Fin 1, kernel.getScalar i ≠ 0 := by
    intro i
    fin_cases i
    simp [kernel]
  have hStride : ∀ i : Fin 1, stride.getScalar i ≠ 0 := by
    intro i
    fin_cases i
    simp [stride]
  let base : ConvParams Float :=
    { spatialRank := 1
      inChannels := 2
      outChannels := 2
      kernel := kernel
      stride := stride
      padding := padding
      dilation := dilationOne
      paddingAfter := padding
      groups := 1
      inputSpatial := inputSpatial
      inChannelsNonzero := by decide
      kernelNonzero := hKernel
      strideNonzero := hStride
      spec := { kernel := by simpa [kernel] using weights, bias := bias } }
  let baseConfig : ConvConfig :=
    { spatialRank := 1
      kernel := kernel
      stride := stride
      padding := padding
      dilation := dilationOne
      paddingAfter := padding
      groups := 1
      channelAxis := 0
      inChannels := 2
      outChannels := 2 }
  let input : Tensor Float [2, 4] := Spec.fill (α := Float) 1 [2, 4]
  let inputBox := pointBox input
  let graphFor (config : ConvConfig) (outShape : Shape) : NN.IR.Graph :=
    { nodes :=
        #[ { id := 0, parents := #[], kind := .input, outShape := [2, 4] }
         , { id := 1, parents := #[0], kind := .conv config, outShape := outShape } ] }
  let storeFor (params : ConvParams Float) : ParamStore Float :=
    { inputBoxes := Std.HashMap.emptyWithCapacity.insert 0 inputBox
      convCfg := Std.HashMap.emptyWithCapacity.insert 1 params }
  let supportedGraph := graphFor baseConfig [2, 5]
  let supportedStore := storeFor base
  unless crownGraphSemanticsSupported supportedGraph supportedStore do
    throw <| IO.userError "supported dense convolution was rejected"
  expectBoundAt "supported dense convolution"
    (runIBP supportedGraph supportedStore) 1

  let groupedParams := { base with groups := 2 }
  let groupedConfig := { baseConfig with groups := 2 }
  let groupedGraph := graphFor groupedConfig [2, 5]
  let groupedStore := storeFor groupedParams
  unless !crownGraphSemanticsSupported groupedGraph groupedStore do
    throw <| IO.userError "grouped convolution was accepted"
  expectNoBoundAt "grouped convolution" (runIBP groupedGraph groupedStore) 1

  let dilatedParams := { base with dilation := dilationTwo }
  let dilatedConfig := { baseConfig with dilation := dilationTwo }
  let dilatedGraph := graphFor dilatedConfig [2, 4]
  let dilatedStore := storeFor dilatedParams
  unless !crownGraphSemanticsSupported dilatedGraph dilatedStore do
    throw <| IO.userError "dilated convolution was accepted"
  expectNoBoundAt "dilated convolution" (runIBP dilatedGraph dilatedStore) 1

  let asymmetricParams := { base with paddingAfter := paddingAfterZero }
  let asymmetricConfig := { baseConfig with paddingAfter := paddingAfterZero }
  let asymmetricGraph := graphFor asymmetricConfig [2, 4]
  let asymmetricStore := storeFor asymmetricParams
  unless !crownGraphSemanticsSupported asymmetricGraph asymmetricStore do
    throw <| IO.userError "asymmetrically padded convolution was accepted"
  let ibp := runIBP asymmetricGraph asymmetricStore
  expectNoBoundAt "asymmetrically padded convolution" ibp 1
  let ctx : AffineCtx := { inputId := 0, inputDim := 8 }
  expectNoBoundAt "asymmetric convolution CROWN"
    (runCROWN asymmetricGraph asymmetricStore ctx ibp) 1

def checkConcatGuard : IO Unit := do
  let input : Tensor Float [2, 2] :=
    Tensor.dim fun i => Tensor.dim fun j => Tensor.scalar (Float.ofNat (2 * i.val + j.val + 1))
  let inputBox := pointBox input
  let graph : NN.IR.Graph :=
    { nodes :=
        #[ { id := 0, parents := #[], kind := .input, outShape := [2, 2] }
         , { id := 1, parents := #[], kind := .input, outShape := [2, 2] }
         , { id := 2, parents := #[0, 1], kind := .concat 1, outShape := [2, 4] } ] }
  let store : ParamStore Float :=
    { inputBoxes := (Std.HashMap.emptyWithCapacity.insert 0 inputBox).insert 1 inputBox }
  unless !crownGraphSemanticsSupported graph store do
    throw <| IO.userError "non-leading-axis concat was accepted"
  let ibp := runIBP graph store
  expectNoBoundAt "non-leading-axis concat IBP" ibp 2
  let ctx : AffineCtx := { inputId := 0, inputDim := 4 }
  expectNoBoundAt "non-leading-axis concat affine" (runAffine graph store ctx ibp) 2
  expectNoBoundAt "non-leading-axis concat CROWN" (runCROWN graph store ctx ibp) 2
  let forgedOutput : FlatBox Float :=
    FlatBox.ofTensor (Spec.fill (α := Float) 0 [8])
  let forgedIbp := #[some inputBox, some inputBox, some forgedOutput]
  let objective : FlatTensor Float := { n := 8, v := Spec.fill (α := Float) 1 [8] }
  match runCROWNBackwardObjective graph store ctx forgedIbp 2 objective with
  | none => pure ()
  | some _ => throw <| IO.userError "non-leading-axis concat backward CROWN was accepted"

def checkLayerNormPayloadGuard : IO Unit := do
  let input : Tensor Float [2, 2] := Spec.fill (α := Float) 1 [2, 2]
  let params : LayerNormParams Float :=
    { normalizedShape := [2]
      gamma := Spec.fill (α := Float) 2 [2]
      beta := Spec.fill (α := Float) 3 [2]
      eps := 0.25 }
  let graph : NN.IR.Graph :=
    { nodes :=
        #[ { id := 0, parents := #[], kind := .input, outShape := [2, 2] }
         , { id := 1, parents := #[0], kind := .layernorm 1, outShape := [2, 2] } ] }
  let store : ParamStore Float :=
    { inputBoxes := Std.HashMap.emptyWithCapacity.insert 0 (pointBox input)
      layerNorm := Std.HashMap.emptyWithCapacity.insert 1 params }
  unless !crownGraphSemanticsSupported graph store do
    throw <| IO.userError "payload-bearing LayerNorm was accepted"
  let ibp := runIBP graph store
  expectNoBoundAt "payload-bearing LayerNorm IBP" ibp 1
  let ctx : AffineCtx := { inputId := 0, inputDim := 4 }
  expectNoBoundAt "payload-bearing LayerNorm affine" (runAffine graph store ctx ibp) 1
  expectNoBoundAt "payload-bearing LayerNorm CROWN" (runCROWN graph store ctx ibp) 1
  let forgedIbp :=
    #[some (pointBox input), some (FlatBox.ofTensor (Spec.fill (α := Float) 0 [4]))]
  let objective : FlatTensor Float := { n := 4, v := Spec.fill (α := Float) 1 [4] }
  match runCROWNBackwardObjective graph store ctx forgedIbp 1 objective with
  | none => pure ()
  | some _ => throw <| IO.userError "payload-bearing LayerNorm backward CROWN was accepted"

def run : IO Unit := do
  checkConvolutionGuards
  checkConcatGuard
  checkLayerNormPayloadGuard

end NN.Tests.MLTheory.CROWNSoundnessGuardrails
