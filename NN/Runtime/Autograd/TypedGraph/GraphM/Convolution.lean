/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TypedGraph.GraphM.Neural

/-!
# GraphM Convolution Ops

N-dimensional and two-dimensional convolution and transposed-convolution builders.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace TypedGraph
namespace GraphM

open Spec
open Tensor
open Proofs.Autograd.Algebra
open Runtime.Autograd.TorchLean

/--
N-dimensional convolution (channels-first) on a single sample tensor.

The input shape is `(inC, spatial...)`, the kernel shape is `(outC, inC, kernelSpatial...)`, and the
bias shape is `(outC)`. The output spatial sizes use the PyTorch-style floor-division formula.

The JVP follows bilinearity:
`d(conv(k,b,x)) = conv(k,0,dx) + conv(dk,db,x)`.
-/
def conv {α : Type} {Δ : Type} [Context α] [DecidableEq Shape]
  {Γ : List Shape} {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (w : Var (Shape.ofList (outC :: inC :: kernel.toList)))
  (b : Var (.dim outC .scalar))
  (x : Var (Shape.ofList (inC :: inSpatial.toList))) :
  MWith α Δ Γ (Var (Shape.ofList (outC :: (Spec.convOutSpatial inSpatial kernel stride padding).toList))) := do
  have _ := hInC
  have _ := hKernel
  let ⟨ss, g⟩ ← get
  let iw ← liftM (mkIdx (_α := α) (Γ := Γ) ss w)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss b)
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)

  let outSpatial : Vector Nat d :=
    Spec.convOutSpatial inSpatial kernel stride padding
  let outS : Shape := Shape.ofList (outC :: outSpatial.toList)
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        let wv := getIdx (α := α) (xs := ctx) iw
        let bv := getIdx (α := α) (xs := ctx) ib
        let xv := getIdx (α := α) (xs := ctx) ix
        let layer : Spec.ConvSpec d inC outC kernel stride padding α :=
          { kernel := wv, bias := bv }
        Spec.convSpec (layer := layer) xv
      jvp := fun ctx dctx _d =>
        let wv := getIdx (α := α) (xs := ctx) iw
        let xv := getIdx (α := α) (xs := ctx) ix
        let dW := getIdx (α := α) (xs := dctx) iw
        let dB := getIdx (α := α) (xs := dctx) ib
        let dX := getIdx (α := α) (xs := dctx) ix
        let zeroBias : Tensor α (.dim outC .scalar) := fill (0 : α) (.dim outC .scalar)
        let layerX : Spec.ConvSpec d inC outC kernel stride padding α :=
          { kernel := wv, bias := zeroBias }
        let layerParams : Spec.ConvSpec d inC outC kernel stride padding α :=
          { kernel := dW, bias := dB }
        addSpec (Spec.convSpec (layer := layerX) dX) (Spec.convSpec (layer := layerParams) xv)
      vjp := fun ctx _d dLdy =>
        let wv := getIdx (α := α) (xs := ctx) iw
        let bv := getIdx (α := α) (xs := ctx) ib
        let xv := getIdx (α := α) (xs := ctx) ix
        let layer : Spec.ConvSpec d inC outC kernel stride padding α :=
          { kernel := wv, bias := bv }
        let (dW, dB, dX) := Spec.convBackwardSpec (layer := layer) xv dLdy
        let z0 :=
          TList.add (α := α) (ss := Γ ++ ss)
            (TList.single (α := α) (Γ := Γ ++ ss)
              (s := Shape.ofList (outC :: inC :: kernel.toList)) iw dW)
            (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim outC .scalar) ib dB)
        TList.add (α := α) (ss := Γ ++ ss) z0
          (TList.single (α := α) (Γ := Γ ++ ss)
            (s := Shape.ofList (inC :: inSpatial.toList)) ix dX) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
N-D transpose convolution (channels-first) on a single sample tensor (no batch axis).

Conventions:
- input shape is `(inC, spatial...)`,
- kernel shape is `(inC, outC, kernelSpatial...)` (PyTorch layout),
- bias shape is `(outC)`,
- output spatial sizes use:
  `out[a] = (in[a] - 1) * stride[a] - 2*padding[a] + kernel[a]` (with `output_padding = 0`).

PyTorch comparison: `torch.nn.functional.conv_transpose{d}d`, specialized to a single sample.

Forward-mode JVP uses bilinearity:
`d(convTranspose(k,b,x)) = convTranspose(k,0,dx) + convTranspose(dk,db,x)`.
-/
def convTranspose {α : Type} {Δ : Type} [Context α] [DecidableEq Shape]
  {Γ : List Shape} {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (w : Var (Shape.ofList (inC :: outC :: kernel.toList)))
  (b : Var (.dim outC .scalar))
  (x : Var (Shape.ofList (inC :: inSpatial.toList))) :
  MWith α Δ Γ (Var (Shape.ofList (outC :: (Spec.convTransposeOutSpatial inSpatial kernel stride padding).toList))) := do
  have _ := hInC
  have _ := hKernel
  let ⟨ss, g⟩ ← get
  let iw ← liftM (mkIdx (_α := α) (Γ := Γ) ss w)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss b)
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)

  let outSpatial : Vector Nat d :=
    Spec.convTransposeOutSpatial inSpatial kernel stride padding
  let outS : Shape := Shape.ofList (outC :: outSpatial.toList)
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        let wv := getIdx (α := α) (xs := ctx) iw
        let bv := getIdx (α := α) (xs := ctx) ib
        let xv := getIdx (α := α) (xs := ctx) ix
        let layer : Spec.ConvTransposeSpec d inC outC kernel stride padding α :=
          { kernel := wv, bias := bv }
        Spec.convTransposeSpec (layer := layer) xv
      jvp := fun ctx dctx _d =>
        let wv := getIdx (α := α) (xs := ctx) iw
        let xv := getIdx (α := α) (xs := ctx) ix
        let dW := getIdx (α := α) (xs := dctx) iw
        let dB := getIdx (α := α) (xs := dctx) ib
        let dX := getIdx (α := α) (xs := dctx) ix
        let zeroBias : Tensor α (.dim outC .scalar) := fill (0 : α) (.dim outC .scalar)
        let layerX : Spec.ConvTransposeSpec d inC outC kernel stride padding α :=
          { kernel := wv, bias := zeroBias }
        let layerParams : Spec.ConvTransposeSpec d inC outC kernel stride padding α :=
          { kernel := dW, bias := dB }
        addSpec (Spec.convTransposeSpec (layer := layerX) dX)
          (Spec.convTransposeSpec (layer := layerParams) xv)
      vjp := fun ctx _d dLdy =>
        let wv := getIdx (α := α) (xs := ctx) iw
        let bv := getIdx (α := α) (xs := ctx) ib
        let xv := getIdx (α := α) (xs := ctx) ix
        let layer : Spec.ConvTransposeSpec d inC outC kernel stride padding α :=
          { kernel := wv, bias := bv }
        let (dW, dB, dX) := Spec.convTransposeBackwardSpec (layer := layer) xv dLdy
        let z0 :=
          TList.add (α := α) (ss := Γ ++ ss)
            (TList.single (α := α) (Γ := Γ ++ ss)
              (s := Shape.ofList (inC :: outC :: kernel.toList)) iw dW)
            (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim outC .scalar) ib dB)
        TList.add (α := α) (ss := Γ ++ ss) z0
          (TList.single (α := α) (Γ := Γ ++ ss)
            (s := Shape.ofList (inC :: inSpatial.toList)) ix dX) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
2D convolution (channel-first) on a single image tensor.

PyTorch comparison: `torch.nn.functional.conv2d` (without a batch dimension).

Forward-mode JVP uses bilinearity:
`d(conv2d(k,b,x)) = conv2d(k,0,dx) + conv2d(dk,db,x)`.
-/
def conv2d {α : Type} {Δ : Type} [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {Γ : List Shape} {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (kernel : Var (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))))
  (bias : Var (.dim outC .scalar))
  (input : Var (.dim inC (.dim inH (.dim inW .scalar)))) :
  MWith α Δ Γ (Var (.dim outC (.dim (Spec.Shape.slidingWindowOutDim inH kH stride padding) (.dim (Spec.Shape.slidingWindowOutDim inW kW stride padding) .scalar)))) := do
  let ⟨ss, g⟩ ← get
  let ik ← liftM (mkIdx (_α := α) (Γ := Γ) ss kernel)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss bias)
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss input)
  let outH : Nat := Spec.Shape.slidingWindowOutDim inH kH stride padding
  let outW : Nat := Spec.Shape.slidingWindowOutDim inW kW stride padding
  let outS : Shape := .dim outC (.dim outH (.dim outW .scalar))
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        let kern := getIdx (α := α) (xs := ctx) ik
        let bv := getIdx (α := α) (xs := ctx) ib
        let inp := getIdx (α := α) (xs := ctx) ix
        let layer :
            Spec.Conv2dSpec inC outC kH kW stride padding α h1 h2 h3 :=
          { kernel := kern
            bias := bv }
        Spec.conv2dSpec (layer := layer) inp
      jvp := fun ctx dctx _d =>
        let kern := getIdx (α := α) (xs := ctx) ik
        let inp := getIdx (α := α) (xs := ctx) ix
        let dKernel := getIdx (α := α) (xs := dctx) ik
        let dBias := getIdx (α := α) (xs := dctx) ib
        let dInput := getIdx (α := α) (xs := dctx) ix
        let zeroBias : Tensor α (.dim outC .scalar) := fill (0 : α) (.dim outC .scalar)
        let layerX :
            Spec.Conv2dSpec inC outC kH kW stride padding α h1 h2 h3 :=
          { kernel := kern
            bias := zeroBias }
        let layerParams :
            Spec.Conv2dSpec inC outC kH kW stride padding α h1 h2 h3 :=
          { kernel := dKernel
            bias := dBias }
        addSpec (Spec.conv2dSpec (layer := layerX) dInput)
          (Spec.conv2dSpec (layer := layerParams) inp)
      vjp := fun ctx _d dLdy =>
        let kern := getIdx (α := α) (xs := ctx) ik
        let bv := getIdx (α := α) (xs := ctx) ib
        let inp := getIdx (α := α) (xs := ctx) ix
        let layer :
            Spec.Conv2dSpec inC outC kH kW stride padding α h1 h2 h3 :=
          { kernel := kern
            bias := bv }
        let (dKernel, dBias, dInput) := Spec.conv2dBackwardSpec (layer := layer) inp dLdy
        let z0 :=
          TList.add (α := α) (ss := Γ ++ ss)
            (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim outC (.dim inC (.dim kH (.dim kW
              .scalar)))) ik dKernel)
            (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim outC .scalar) ib dBias)
        TList.add (α := α) (ss := Γ ++ ss) z0
          (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim inC (.dim inH (.dim inW .scalar))) ix
            dInput) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
2D transpose convolution (channel-first) on a single image tensor.

PyTorch comparison: `torch.nn.functional.conv_transpose2d` (without a batch dimension).

Forward-mode JVP uses bilinearity:
`d(convTranspose2d(k,b,x)) = convTranspose2d(k,0,dx) + convTranspose2d(dk,db,x)`.
-/
def convTranspose2d {α : Type} {Δ : Type} [Context α]
  [DecidableEq Shape]
  {Γ : List Shape} {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (kernel : Var (.dim inC (.dim outC (.dim kH (.dim kW .scalar)))))
  (bias : Var (.dim outC .scalar))
  (input : Var (.dim inC (.dim inH (.dim inW .scalar)))) :
  MWith α Δ Γ (Var (.dim outC (.dim (Spec.convTransposeOutDim inH kH stride padding)
    (.dim (Spec.convTransposeOutDim inW kW stride padding) .scalar)))) := do
  have h1' : inC > 0 := Nat.pos_of_ne_zero h1
  let ⟨ss, g⟩ ← get
  let ik ← liftM (mkIdx (_α := α) (Γ := Γ) ss kernel)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss bias)
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss input)
  let outH : Nat := Spec.convTransposeOutDim inH kH stride padding
  let outW : Nat := Spec.convTransposeOutDim inW kW stride padding
  let outS : Shape := .dim outC (.dim outH (.dim outW .scalar))
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        let kern := getIdx (α := α) (xs := ctx) ik
        let bv := getIdx (α := α) (xs := ctx) ib
        let inp := getIdx (α := α) (xs := ctx) ix
        let layer :
            Spec.ConvTranspose2dSpec inC outC kH kW stride padding α h1' h2 h3 :=
          { kernel := kern
            bias := bv }
        Spec.convTranspose2dSpec (layer := layer) inp
      jvp := fun ctx dctx _d =>
        let kern := getIdx (α := α) (xs := ctx) ik
        let inp := getIdx (α := α) (xs := ctx) ix
        let dKernel := getIdx (α := α) (xs := dctx) ik
        let dBias := getIdx (α := α) (xs := dctx) ib
        let dInput := getIdx (α := α) (xs := dctx) ix
        let zeroBias : Tensor α (.dim outC .scalar) := fill (0 : α) (.dim outC .scalar)
        let layerX :
            Spec.ConvTranspose2dSpec inC outC kH kW stride padding α h1' h2 h3 :=
          { kernel := kern
            bias := zeroBias }
        let layerParams :
            Spec.ConvTranspose2dSpec inC outC kH kW stride padding α h1' h2 h3 :=
          { kernel := dKernel
            bias := dBias }
        addSpec (Spec.convTranspose2dSpec (layer := layerX) dInput)
          (Spec.convTranspose2dSpec (layer := layerParams) inp)
      vjp := fun ctx _d dLdy =>
        let kern := getIdx (α := α) (xs := ctx) ik
        let bv := getIdx (α := α) (xs := ctx) ib
        let inp := getIdx (α := α) (xs := ctx) ix
        let layer :
            Spec.ConvTranspose2dSpec inC outC kH kW stride padding α h1' h2 h3 :=
          { kernel := kern
            bias := bv }
        let (dKernel, dBias, dInput) := Spec.convTranspose2dBackwardSpec (layer := layer) inp dLdy
        let z0 :=
          TList.add (α := α) (ss := Γ ++ ss)
            (TList.single (α := α) (Γ := Γ ++ ss)
              (s := .dim inC (.dim outC (.dim kH (.dim kW .scalar)))) ik dKernel)
            (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim outC .scalar) ib dBias)
        TList.add (α := α) (ss := Γ ++ ss) z0
          (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim inC (.dim inH (.dim inW .scalar))) ix
            dInput) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node


end GraphM
end TypedGraph
end Autograd
end Runtime
