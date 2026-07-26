/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Core
public import NN.Runtime.Autograd.Engine.Cuda.Ops
public import NN.Tensor
public import NN.Tests.Runtime.Cuda.Utils

/-!
# CUDA Kernel Coverage: Transposed Convolution

Compares CPU eager tape vs CUDA eager tape for `conv_transpose`:
- a 2D case (`d = 2`), and
- a non-2D case (`d = 3`).

Both cases check forward output and gradients (including `dInput`) via `backwardDenseAll`.
Inputs are small so stub-mode remains lightweight and float64/float32 roundoff differences stay
limited.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace ConvTranspose

open Spec
open Tensor
open Runtime.Autograd

-- Regression for output-size arithmetic with padding and a one-cell input.
example : Spec.convTransposeOutDim 1 3 1 1 = 1 := by decide

/-!
## 2D case ($d=2$)
-/

/-- Spatial rank for the 2D transposed-convolution coverage case. -/
abbrev d2 : Nat := 2

abbrev inC2 : Nat := 1
abbrev outC2 : Nat := 1
abbrev kH : Nat := 2
abbrev kW : Nat := 2
abbrev stride2 : Nat := 1
abbrev padding2 : Nat := 0
abbrev inH2 : Nat := 3
abbrev inW2 : Nat := 3

theorem hInC2 : inC2 ≠ 0 := by decide

def kernel2Dims : Vector Nat d2 :=
  #v[kH, kW]

def stride2Dims : Vector Nat d2 :=
  #v[stride2, stride2]

def padding2Dims : Vector Nat d2 :=
  #v[padding2, padding2]

def inSpatial2Dims : Vector Nat d2 :=
  #v[inH2, inW2]

theorem hKernel2 : ∀ i : Fin d2, kernel2Dims.get i ≠ 0 := by
  intro i
  fin_cases i <;> simp [kernel2Dims, Vector.get]

def outSpatial2Dims : Vector Nat d2 :=
  Spec.convTransposeOutSpatial inSpatial2Dims kernel2Dims stride2Dims padding2Dims

def outShape2 : Shape :=
  Shape.ofList (outC2 :: outSpatial2Dims.toList)

def kernelShape2 : Shape :=
  Shape.ofList (inC2 :: outC2 :: kernel2Dims.toList)

def inputShape2 : Shape :=
  Shape.ofList (inC2 :: inSpatial2Dims.toList)

def kernel2 : Tensor Float kernelShape2 :=
  tensorOfList! [inC2, outC2, kH, kW] [
    0.2, -0.1,
    0.3, 0.4
  ]

def bias2 : Tensor Float (shape![outC2]) :=
  tensorOfList! [outC2] [0.05]

def input2 : Tensor Float inputShape2 :=
  tensorOfList! [inC2, inH2, inW2] [
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0
  ]

def runConvTranspose2 : IO Unit := do
  IO.println "== conv_transpose (d=2) =="

  -- CPU tape
  let t0 : Tape Float := Tape.empty
  let (t1, kId) := Tape.leaf (t := t0) kernel2 (name := some "kernel")
  let (t2, bId) := Tape.leaf (t := t1) bias2 (name := some "bias")
  let (t3, xId) := Tape.leaf (t := t2) input2 (name := some "input")
  let (t4, yId) ← Utils.okOrThrow
    (Tape.convTranspose (α := Float) (t := t3)
      (d := d2) (inC := inC2) (outC := outC2)
      (kernel := kernel2Dims) (stride := stride2Dims) (padding := padding2Dims)
      (inSpatial := inSpatial2Dims)
      kId bId xId (name := "conv_transpose[d=2]"))
  let yCpu ← Utils.cpuValue (s := outShape2) t4 yId
  let seedCpu : Runtime.AnyTensor Float := AnyTensor.mk (fill (1.0 : Float) outShape2)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t4) yId seedCpu)
  let dKCpu ← Utils.cpuGrad (s := kernelShape2) gradsCpu kId
  let dBCpu ← Utils.cpuGrad (s := shape![outC2]) gradsCpu bId
  let dXCpu ← Utils.cpuGrad (s := inputShape2) gradsCpu xId

  -- CUDA tape
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, kIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer kernel2)
    (name := some "kernel")
  let (t2c, bIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t1c) (Utils.tensorToAnyBuffer bias2)
    (name := some "bias")
  let (t3c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t2c) (Utils.tensorToAnyBuffer input2)
    (name := some "input")
  let (t4c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.convTranspose (t := t3c)
      (d := d2) (inC := inC2) (outC := outC2)
      (kernel := kernel2Dims) (stride := stride2Dims) (padding := padding2Dims)
      (inSpatial := inSpatial2Dims)
      kIdc bIdc xIdc (hInC := hInC2) (hKernel := hKernel2))
  let yCuda ← Utils.cudaValue (s := outShape2) t4c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := outShape2
      buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Spec.Shape.size outShape2)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t4c) yIdc seedCuda)
  let dKCuda ← Utils.cudaGrad (s := kernelShape2) gradsCuda kIdc
  let dBCuda ← Utils.cudaGrad (s := shape![outC2]) gradsCuda bIdc
  let dXCuda ← Utils.cudaGrad (s := inputShape2) gradsCuda xIdc

  Utils.assertTensorApprox (s := outShape2) "conv_transpose[d=2] forward" yCuda yCpu (tol := 5e-3)
  Utils.assertTensorApprox (s := kernelShape2) "conv_transpose[d=2] dKernel" dKCuda dKCpu (tol := 5e-3)
  Utils.assertTensorApprox (s := shape![outC2]) "conv_transpose[d=2] dBias" dBCuda dBCpu (tol := 5e-3)
  Utils.assertTensorApprox (s := inputShape2) "conv_transpose[d=2] dInput" dXCuda dXCpu (tol := 5e-3)

/-!
## 3D case ($d=3$)
-/

abbrev d3 : Nat := 3

abbrev inC3 : Nat := 1
abbrev outC3 : Nat := 1

abbrev inD0 : Nat := 2
abbrev inD1 : Nat := 2
abbrev inD2 : Nat := 2

abbrev k0 : Nat := 2
abbrev k1 : Nat := 2
abbrev k2 : Nat := 2

theorem hInC3 : inC3 ≠ 0 := by decide

def kernel3Dims : Vector Nat d3 :=
  #v[k0, k1, k2]

def stride3Dims : Vector Nat d3 :=
  #v[1, 1, 1]

def padding3Dims : Vector Nat d3 :=
  #v[0, 0, 0]

def inSpatial3Dims : Vector Nat d3 :=
  #v[inD0, inD1, inD2]

theorem hKernel3 : ∀ i : Fin d3, kernel3Dims.get i ≠ 0 := by
  intro i
  fin_cases i <;> simp [kernel3Dims, Vector.get]

def outSpatial3Dims : Vector Nat d3 :=
  Spec.convTransposeOutSpatial inSpatial3Dims kernel3Dims stride3Dims padding3Dims

def outShape3 : Shape :=
  Shape.ofList (outC3 :: outSpatial3Dims.toList)

def kernelShape3 : Shape :=
  Shape.ofList (inC3 :: outC3 :: kernel3Dims.toList)

def inputShape3 : Shape :=
  Shape.ofList (inC3 :: inSpatial3Dims.toList)

def kernel3 : Tensor Float kernelShape3 :=
  tensorOfList! [inC3, outC3, k0, k1, k2] [
    0.2, -0.1,
    0.3, 0.4,
    -0.25, 0.15,
    0.05, -0.35
  ]

def bias3 : Tensor Float (shape![outC3]) :=
  tensorOfList! [outC3] [0.01]

def input3 : Tensor Float inputShape3 :=
  tensorOfList! [inC3, inD0, inD1, inD2] [
    1.0, 2.0,
    3.0, 4.0,

    5.0, 6.0,
    7.0, 8.0
  ]

def runConvTranspose3 : IO Unit := do
  IO.println "== conv_transpose (d=3) =="

  -- CPU tape
  let t0 : Tape Float := Tape.empty
  let (t1, kId) := Tape.leaf (t := t0) kernel3 (name := some "kernel")
  let (t2, bId) := Tape.leaf (t := t1) bias3 (name := some "bias")
  let (t3, xId) := Tape.leaf (t := t2) input3 (name := some "input")
  let (t4, yId) ← Utils.okOrThrow
    (Tape.convTranspose (α := Float) (t := t3)
      (d := d3) (inC := inC3) (outC := outC3)
      (kernel := kernel3Dims) (stride := stride3Dims) (padding := padding3Dims)
      (inSpatial := inSpatial3Dims)
      kId bId xId (name := "conv_transpose[d=3]"))
  let yCpu ← Utils.cpuValue (s := outShape3) t4 yId
  let seedCpu : Runtime.AnyTensor Float := AnyTensor.mk (fill (1.0 : Float) outShape3)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t4) yId seedCpu)
  let dKCpu ← Utils.cpuGrad (s := kernelShape3) gradsCpu kId
  let dBCpu ← Utils.cpuGrad (s := shape![outC3]) gradsCpu bId
  let dXCpu ← Utils.cpuGrad (s := inputShape3) gradsCpu xId

  -- CUDA tape
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, kIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer kernel3)
    (name := some "kernel")
  let (t2c, bIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t1c) (Utils.tensorToAnyBuffer bias3)
    (name := some "bias")
  let (t3c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t2c) (Utils.tensorToAnyBuffer input3)
    (name := some "input")
  let (t4c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.convTranspose (t := t3c)
      (d := d3) (inC := inC3) (outC := outC3)
      (kernel := kernel3Dims) (stride := stride3Dims) (padding := padding3Dims)
      (inSpatial := inSpatial3Dims)
      kIdc bIdc xIdc (hInC := hInC3) (hKernel := hKernel3))
  let yCuda ← Utils.cudaValue (s := outShape3) t4c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := outShape3
      buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Spec.Shape.size outShape3)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t4c) yIdc seedCuda)
  let dKCuda ← Utils.cudaGrad (s := kernelShape3) gradsCuda kIdc
  let dBCuda ← Utils.cudaGrad (s := shape![outC3]) gradsCuda bIdc
  let dXCuda ← Utils.cudaGrad (s := inputShape3) gradsCuda xIdc

  Utils.assertTensorApprox (s := outShape3) "conv_transpose[d=3] forward" yCuda yCpu (tol := 1e-2)
  Utils.assertTensorApprox (s := kernelShape3) "conv_transpose[d=3] dKernel" dKCuda dKCpu (tol := 1e-2)
  Utils.assertTensorApprox (s := shape![outC3]) "conv_transpose[d=3] dBias" dBCuda dBCpu (tol := 1e-2)
  Utils.assertTensorApprox (s := inputShape3) "conv_transpose[d=3] dInput" dXCuda dXCpu (tol := 1e-2)

/-- Excessive padding saturates transpose-convolution output size only after adding the kernel. -/
def runSaturatedOutputGeometry : IO Unit := do
  IO.println "== conv_transpose saturated output geometry =="
  let input : Tensor Float (shape![1, 1, 1]) := tensorOfList! [1, 1, 1] [2.0]
  let kernel : Tensor Float (shape![1, 1, 3, 3]) :=
    tensorOfList! [1, 1, 3, 3] [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  let bias : Tensor Float (shape![1]) := tensorOfList! [1] [0.0]
  let (t1, kernelId) := Runtime.Autograd.Cuda.Tape.empty.leaf (Utils.tensorToAnyBuffer kernel)
  let (t2, biasId) := t1.leaf (Utils.tensorToAnyBuffer bias)
  let (t3, inputId) := t2.leaf (Utils.tensorToAnyBuffer input)
  let (t4, outputId) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.convTranspose2d (t := t3)
      (inC := 1) (outC := 1) (kH := 3) (kW := 3) (stride := 1) (padding := 2)
      (inH := 1) (inW := 1) (h1 := by decide) (h2 := by decide) (h3 := by decide)
      kernelId biasId inputId)
  let empty2dShape : Shape := shape![1, 0, 0]
  let output ← Utils.okOrThrow <|
    Runtime.Autograd.Cuda.Tape.requireValue t4 outputId empty2dShape
  unless Runtime.Autograd.Cuda.Buffer.size output = 0 do
    throw <| IO.userError "conv_transpose2d excessive padding produced a nonempty buffer"

  let inSpatial : Vector Nat 1 := #v[1]
  let kernelDims : Vector Nat 1 := #v[3]
  let strideDims : Vector Nat 1 := #v[1]
  let paddingDims : Vector Nat 1 := #v[2]
  let hKernel : ∀ i : Fin 1, kernelDims.get i ≠ 0 := by
    intro i
    fin_cases i
    simp [kernelDims, Vector.get]
  let inputNd : Tensor Float (Shape.ofList [1, 1]) := tensorOfList! [1, 1] [2.0]
  let kernelNd : Tensor Float (Shape.ofList [1, 1, 3]) := tensorOfList! [1, 1, 3] [1.0, 1.0, 1.0]
  let (tn1, kernelNdId) :=
    Runtime.Autograd.Cuda.Tape.empty.leaf (Utils.tensorToAnyBuffer kernelNd)
  let (tn2, biasNdId) := tn1.leaf (Utils.tensorToAnyBuffer bias)
  let (tn3, inputNdId) := tn2.leaf (Utils.tensorToAnyBuffer inputNd)
  let (tn4, outputNdId) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.convTranspose (t := tn3)
      (d := 1) (inC := 1) (outC := 1) (inSpatial := inSpatial) (kernel := kernelDims)
      (stride := strideDims) (padding := paddingDims) kernelNdId biasNdId inputNdId
      (hInC := by decide) (hKernel := hKernel))
  let emptyNdShape : Shape := Shape.ofList [1, 0]
  let outputNd ← Utils.okOrThrow <|
    Runtime.Autograd.Cuda.Tape.requireValue tn4 outputNdId emptyNdShape
  unless Runtime.Autograd.Cuda.Buffer.size outputNd = 0 do
    throw <| IO.userError "N-D conv_transpose excessive padding produced a nonempty buffer"

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: conv_transpose ==="
  runConvTranspose2
  runConvTranspose3
  runSaturatedOutputGeometry

end ConvTranspose
end Cuda
end Tests
