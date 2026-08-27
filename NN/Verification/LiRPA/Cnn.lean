/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Operators.Conv
public import NN.Verification.LiRPA.ExampleInputs

/-!
# LiRPA convolutional certificate checker

LiRPA/IBP certificate checker for a convolution followed by a linear head.

This workflow:
- encodes a convolution as an affine form (so the graph stays in the flat-vector LiRPA engine),
- adds a linear head, and
- checks a JSON certificate (produced by Python) using `NN.Verification.Cert.IBPCert`.

References:
- IBP: arXiv:1810.12715 `https://arxiv.org/abs/1810.12715`
- auto_LiRPA (reference implementation / cert exporter inspiration):
  `https://github.com/Verified-Intelligence/auto_LiRPA`

Export (Python):
`python3.12 scripts/verification/lirpa/export_cnn_cert.py`

Run (Lean):
`lake exe verify -- lirpa-cnn [NN/Examples/Verification/LiRPA/cnn_cert.json]`
-/

@[expose] public section


namespace NN.Verification.LiRPA.Cnn

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open _root_.Spec
open _root_.Spec.Tensor

/--
Small fixed graph:
`input(flattened) -> linear(convolution) -> ReLU -> linear(head)`.

We keep it flat so the certificate checker works over `FlatBox` inputs.
-/
def buildGraph : Graph :=
  let inC := 1; let inH := 4; let inW := 4
  let inShape := Shape.dim inC (Shape.dim inH (Shape.dim inW Shape.scalar))
  let nIn := inShape.size
  let outC := 1; let kH := 3; let kW := 3; let stride := 1; let padding := 0
  let outH := Spec.Shape.slidingWindowOutDim inH kH stride padding
  let outW := Spec.Shape.slidingWindowOutDim inW kW stride padding
  let outShape := Shape.dim outC (Shape.dim outH (Shape.dim outW Shape.scalar))
  let nConv := outShape.size
  let nOut := 2
  let inputNode : Node := { id := 0, parents := #[], kind := .input, outShape := .dim nIn .scalar }
  let convAffineNode : Node := { id := 1, parents := #[0], kind := .linear, outShape := .dim nConv .scalar }
  let reluNode : Node := { id := 2, parents := #[1], kind := .relu, outShape := .dim nConv .scalar }
  let classifierNode : Node := { id := 3, parents := #[2], kind := .linear, outShape := .dim nOut .scalar }
  { nodes := #[inputNode, convAffineNode, reluNode, classifierNode] }

/--
Seed deterministic parameters and the input box.

The convolution is materialized as its exact flattened matrix and bias. ReLU remains an explicit
graph node, so IBP applies its interval rule instead of disguising one affine relaxation as an exact
linear layer.
-/
def seedParamsFloat : ParamStore Float :=
  let inC := 1; let outC := 1; let kH := 3; let kW := 3; let stride := 1; let padding := 0
  let inH := 4; let inW := 4
  let kernelShape : Spec.Tensor Nat [2] :=
    Spec.Tensor.ofArrayExact #[kH, kW] (by simp)
  let strides : Spec.Tensor Nat [2] :=
    Spec.Tensor.ofArrayExact #[stride, stride] (by simp)
  let paddings : Spec.Tensor Nat [2] :=
    Spec.Tensor.ofArrayExact #[padding, padding] (by simp)
  let inputSpatial : Spec.Tensor Nat [2] :=
    Spec.Tensor.ofArrayExact #[inH, inW] (by simp)
  let inShape := Shape.ofList (inC :: inputSpatial.toList)
  let outSpatial := Spec.convOutSpatial inputSpatial kernelShape strides paddings
  let outShape := Shape.ofList (outC :: outSpatial.toList)
  let nIn := inShape.size
  let nConv := outShape.size
  let kernelValues : Tensor Float [outC, inC, kH, kW] :=
    Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.dim (fun i => Tensor.dim (fun j =>
      Tensor.scalar (Float.ofNat (1 + (i.val + j.val)))))))
  let kernel : Tensor Float (Shape.ofList (outC :: inC :: kernelShape.toList)) := by
    simpa [kernelShape] using kernelValues
  let bias : Tensor Float [outC] := Tensor.dim (fun _ => Tensor.scalar (0.0))
  let conv : Spec.ConvSpec 2 inC outC kernelShape strides paddings Float :=
    { kernel := kernel, bias := bias }
  -- Seed input box (center ones, eps)
  let inputCenter : Tensor Float inShape := Spec.fill 1.0 inShape
  let eps : Float := 0.1
  let rad := Spec.fill (α := Float) eps inShape
  let xB : Box Float inShape :=
    { lo := Tensor.subSpec inputCenter rad, hi := Tensor.addSpec inputCenter rad }
  let convWeight : Tensor Float [nConv, nIn] :=
    NN.MLTheory.CROWN.convLinearMatrix (α := Float)
    (inSpatial := inputSpatial) conv
  let convBias : Tensor Float [nConv] :=
    NN.MLTheory.CROWN.convBiasBroadcast (α := Float) (outSpatial := outSpatial) conv.bias
  -- Linear head 4→2
  let headWeight : Tensor Float [2, nConv] := Tensor.dim (fun i => Tensor.dim (fun j
    => Tensor.scalar (Float.ofNat (2 + (i.val + j.val)))))
  let headBias : Tensor Float [2] := Tensor.dim (fun i => Tensor.scalar (Float.ofNat
    (i.val)))
  let emptyStore : ParamStore Float := {}
  -- set input box
  let inFlat : FlatBox Float :=
    { dim := nIn, lo := Tensor.flattenSpec xB.lo, hi := Tensor.flattenSpec xB.hi }
  let withInputBox := emptyStore.seedInputBox 0 inFlat
  -- Store the exact flattened convolution as a linear node.
  let withConvAffine :=
    { withInputBox with
      linearWB :=
        withInputBox.linearWB.insert 1
          { m := nConv
            n := nIn
            w := convWeight
            b := convBias } }
  -- set head linear
  let withClassifier :=
    { withConvAffine with
      linearWB := withConvAffine.linearWB.insert 3
        ({ m := 2, n := nConv, w := headWeight, b := headBias }) }
  withClassifier

/--
Check an IBP certificate JSON against this CNN graph.

This is wired into `lake exe verify -- lirpa-cnn [path]`.
-/
def verifyCert (path : String) : IO Unit := do
  let g := buildGraph
  let ps := seedParamsFloat
  NN.Verification.IBPCert.checkOrThrow g ps (outId := 3) path

end NN.Verification.LiRPA.Cnn
