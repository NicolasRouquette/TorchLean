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
import Batteries.Data.Vector.Lemmas

/-!
# CUDA Kernel Coverage: Conv2d + Pooling

Compares CPU eager tape vs CUDA eager tape for:
- `conv2d`
- `max_pool2d`
- `smooth_max_pool2d`
- `avg_pool2d`

All cases are single-image, channels-first, small shapes.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace ConvPool

open Spec
open Tensor
open Runtime.Autograd

/-- Input channel count used by the Conv2d/pooling CUDA coverage cases. -/
abbrev inC : Nat := 1
abbrev outC : Nat := 1
abbrev kH : Nat := 2
abbrev kW : Nat := 2
abbrev stride : Nat := 1
abbrev padding : Nat := 0
abbrev inH : Nat := 3
abbrev inW : Nat := 3

theorem hInC : inC ≠ 0 := by decide
theorem hKH : kH ≠ 0 := by decide
theorem hKW : kW ≠ 0 := by decide

def outH : Nat := Spec.Shape.slidingWindowOutDim inH kH stride padding
def outW : Nat := Spec.Shape.slidingWindowOutDim inW kW stride padding

def kernel : Tensor Float (shape![outC, inC, kH, kW]) :=
  tensorOfList! [outC, inC, kH, kW] [0.2, -0.1, 0.3, 0.4]

def bias : Tensor Float (shape![outC]) :=
  tensorOfList! [outC] [0.05]

def input : Tensor Float (shape![inC, inH, inW]) :=
  tensorOfList! [inC, inH, inW] [
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0
  ]

/-!
## N-D runtime cases ($d=3$)

These exercise the new "ND" ConvPool CUDA entrypoints (`conv`/`max_pool`/`avg_pool`/`smooth_max_pool`)
which accept per-axis parameters.
-/

abbrev d3 : Nat := 3
abbrev inD0 : Nat := 3
abbrev inD1 : Nat := 3
abbrev inD2 : Nat := 3

abbrev k0 : Nat := 2
abbrev k1 : Nat := 2
abbrev k2 : Nat := 2

def inSpatial3 : Vector Nat d3 :=
  #v[inD0, inD1, inD2]

def kernel3V : Vector Nat d3 :=
  #v[k0, k1, k2]

def stride3V : Vector Nat d3 :=
  #v[1, 1, 1]

def padding3V : Vector Nat d3 :=
  #v[0, 0, 0]

theorem hKernel3V : ∀ i : Fin d3, kernel3V.get i ≠ 0 := by
  intro i
  fin_cases i <;> simp [kernel3V, Vector.get]

def outSpatial3 : Vector Nat d3 :=
  Spec.convOutSpatial inSpatial3 kernel3V stride3V padding3V

def outShape3 : Shape :=
  Shape.ofList (outC :: outSpatial3.toList)

def kernel3 : Tensor Float (Shape.ofList (outC :: inC :: kernel3V.toList)) :=
  tensorOfList! [outC, inC, k0, k1, k2] [
    0.2, -0.1,
    0.3, 0.4,
    -0.25, 0.15,
    0.05, -0.35
  ]

def input3 : Tensor Float (Shape.ofList (inC :: inSpatial3.toList)) :=
  tensorOfList! [inC, inD0, inD1, inD2] [
    1.0,  2.0,  3.0,
    4.0,  5.0,  6.0,
    7.0,  8.0,  9.0,

    10.0, 11.0, 12.0,
    13.0, 14.0, 15.0,
    16.0, 17.0, 18.0,

    19.0, 20.0, 21.0,
    22.0, 23.0, 24.0,
    25.0, 26.0, 27.0
  ]

def runConv3 : IO Unit := do
  IO.println "== conv (d=3) =="

  -- CPU tape
  let t0 : Tape Float := Tape.empty
  let (t1, kId) := Tape.leaf (t := t0) kernel3 (name := some "kernel")
  let (t2, bId) := Tape.leaf (t := t1) bias (name := some "bias")
  let (t3, xId) := Tape.leaf (t := t2) input3 (name := some "input")
  let (t4, yId) ← Utils.okOrThrow
    (Tape.conv (α := Float) (t := t3)
      (d := d3) (inC := inC) (outC := outC)
      (kernel := kernel3V) (stride := stride3V) (padding := padding3V)
      (inSpatial := inSpatial3)
      kId bId xId (name := "conv[d=3]"))
  let yCpu ← Utils.cpuValue (s := outShape3) t4 yId
  let seedCpu : Spec.PackedTensor Float := Spec.PackedTensor.ofTensor (fill (1.0 : Float) outShape3)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t4) yId seedCpu)
  let dKCpu ← Utils.cpuGrad (s := Shape.ofList (outC :: inC :: kernel3V.toList)) gradsCpu kId
  let dBCpu ← Utils.cpuGrad (s := shape![outC]) gradsCpu bId
  let dXCpu ← Utils.cpuGrad (s := Shape.ofList (inC :: inSpatial3.toList)) gradsCpu xId

  -- CUDA tape
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, kIdc) :=
    Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer kernel3) (name := some "kernel")
  let (t2c, bIdc) :=
    Runtime.Autograd.Cuda.Tape.leaf (t := t1c) (Utils.tensorToAnyBuffer bias) (name := some "bias")
  let (t3c, xIdc) :=
    Runtime.Autograd.Cuda.Tape.leaf (t := t2c) (Utils.tensorToAnyBuffer input3) (name := some "input")
  let (t4c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.conv (t := t3c)
      (d := d3) (inC := inC) (outC := outC)
      (kernel := kernel3V) (stride := stride3V) (padding := padding3V)
      (inSpatial := inSpatial3)
      kIdc bIdc xIdc (hInC := hInC) (hKernel := hKernel3V))
  let yCuda ← Utils.cudaValue (s := outShape3) t4c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := outShape3
      buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Spec.Shape.size outShape3)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t4c) yIdc seedCuda)
  let dKCuda ← Utils.cudaGrad (s := Shape.ofList (outC :: inC :: kernel3V.toList)) gradsCuda kIdc
  let dBCuda ← Utils.cudaGrad (s := shape![outC]) gradsCuda bIdc
  let dXCuda ← Utils.cudaGrad (s := Shape.ofList (inC :: inSpatial3.toList)) gradsCuda xIdc

  Utils.assertTensorApprox (s := outShape3) "conv[d=3] forward" yCuda yCpu (tol := 1e-2)
  Utils.assertTensorApprox (s := Shape.ofList (outC :: inC :: kernel3V.toList))
    "conv[d=3] dKernel" dKCuda dKCpu (tol := 1e-2)
  Utils.assertTensorApprox (s := shape![outC]) "conv[d=3] dBias" dBCuda dBCpu (tol := 1e-2)
  Utils.assertTensorApprox (s := Shape.ofList (inC :: inSpatial3.toList))
    "conv[d=3] dInput" dXCuda dXCpu (tol := 1e-2)

def runMaxPool3 : IO Unit := do
  IO.println "== max_pool (d=3) =="

  let outSpatial3 := Spec.poolOutSpatialPad inSpatial3 kernel3V stride3V padding3V
  let yShape : Shape := Shape.ofList (inC :: outSpatial3.toList)

  -- CPU
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) input3 (name := some "input")
  let (t2, yId) ← Utils.okOrThrow
    (Tape.maxPool (α := Float) (t := t1)
      (d := d3) (C := inC)
      (inSpatial := inSpatial3) (kernel := kernel3V) (stride := stride3V) (padding := padding3V)
      (hKernel := hKernel3V) xId)
  let yCpu ← Utils.cpuValue (s := yShape) t2 yId
  let seedCpu : Spec.PackedTensor Float := Spec.PackedTensor.ofTensor (fill (1.0 : Float) yShape)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := Shape.ofList (inC :: inSpatial3.toList)) gradsCpu xId

  -- CUDA
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer input3)
    (name := some "input")
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.maxPool (t := t1c)
      (d := d3) (C := inC)
      (inSpatial := inSpatial3) (kernel := kernel3V) (stride := stride3V) (padding := padding3V)
      (hKernel := hKernel3V) xIdc)
  let yCuda ← Utils.cudaValue (s := yShape) t2c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := yShape
      buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Spec.Shape.size yShape)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := Shape.ofList (inC :: inSpatial3.toList)) gradsCuda xIdc

  Utils.assertTensorApprox (s := yShape) "max_pool[d=3] forward" yCuda yCpu (tol := 1e-6)
  Utils.assertTensorApprox (s := Shape.ofList (inC :: inSpatial3.toList))
    "max_pool[d=3] dx" dxCuda dxCpu (tol := 1e-6)

def runSmoothMaxPool3 : IO Unit := do
  IO.println "== smooth_max_pool (d=3) =="

  let outSpatial3 := Spec.poolOutSpatialPad inSpatial3 kernel3V stride3V padding3V
  let yShape : Shape := Shape.ofList (inC :: outSpatial3.toList)
  let beta : Float := 0.5

  -- CPU
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) input3 (name := some "input")
  let (t2, yId) ← Utils.okOrThrow
    (Tape.smoothMaxPool (α := Float) (t := t1)
      (d := d3) (C := inC)
      (inSpatial := inSpatial3) (kernel := kernel3V) (stride := stride3V) (padding := padding3V)
      (hKernel := hKernel3V) xId beta)
  let yCpu ← Utils.cpuValue (s := yShape) t2 yId
  let seedCpu : Spec.PackedTensor Float := Spec.PackedTensor.ofTensor (fill (1.0 : Float) yShape)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := Shape.ofList (inC :: inSpatial3.toList)) gradsCpu xId

  -- CUDA
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer input3)
    (name := some "input")
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.smoothMaxPool (t := t1c)
      (d := d3) (C := inC)
      (inSpatial := inSpatial3) (kernel := kernel3V) (stride := stride3V) (padding := padding3V)
      (hKernel := hKernel3V) xIdc beta)
  let yCuda ← Utils.cudaValue (s := yShape) t2c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := yShape
      buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Spec.Shape.size yShape)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := Shape.ofList (inC :: inSpatial3.toList)) gradsCuda xIdc

  Utils.assertTensorApprox (s := yShape) "smooth_max_pool[d=3] forward" yCuda yCpu (tol := 1e-2)
  Utils.assertTensorApprox (s := Shape.ofList (inC :: inSpatial3.toList))
    "smooth_max_pool[d=3] dx" dxCuda dxCpu (tol := 1e-2)

def runAvgPool3 : IO Unit := do
  IO.println "== avg_pool (d=3) =="

  let outSpatial3 := Spec.poolOutSpatialPad inSpatial3 kernel3V stride3V padding3V
  let yShape : Shape := Shape.ofList (inC :: outSpatial3.toList)

  -- CPU
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) input3 (name := some "input")
  let (t2, yId) ← Utils.okOrThrow
    (Tape.avgPool (α := Float) (t := t1)
      (d := d3) (C := inC)
      (inSpatial := inSpatial3) (kernel := kernel3V) (stride := stride3V) (padding := padding3V)
      (hKernel := hKernel3V) xId)
  let yCpu ← Utils.cpuValue (s := yShape) t2 yId
  let seedCpu : Spec.PackedTensor Float := Spec.PackedTensor.ofTensor (fill (1.0 : Float) yShape)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := Shape.ofList (inC :: inSpatial3.toList)) gradsCpu xId

  -- CUDA
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer input3)
    (name := some "input")
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.avgPool (t := t1c)
      (d := d3) (C := inC)
      (inSpatial := inSpatial3) (kernel := kernel3V) (stride := stride3V) (padding := padding3V)
      (hKernel := hKernel3V) xIdc)
  let yCuda ← Utils.cudaValue (s := yShape) t2c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := yShape
      buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Spec.Shape.size yShape)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := Shape.ofList (inC :: inSpatial3.toList)) gradsCuda xIdc

  Utils.assertTensorApprox (s := yShape) "avg_pool[d=3] forward" yCuda yCpu (tol := 1e-2)
  Utils.assertTensorApprox (s := Shape.ofList (inC :: inSpatial3.toList))
    "avg_pool[d=3] dx" dxCuda dxCpu (tol := 1e-2)

def runConv2d : IO Unit := do
  IO.println "== conv2d =="

  let yShape : Shape := shape![outC, outH, outW]

  -- CPU tape
  let t0 : Tape Float := Tape.empty
  let (t1, kId) := Tape.leaf (t := t0) kernel (name := some "kernel")
  let (t2, bId) := Tape.leaf (t := t1) bias (name := some "bias")
  let (t3, xId) := Tape.leaf (t := t2) input (name := some "input")
  let (t4, yId) ← Utils.okOrThrow
    (Tape.conv2d (α := Float) (t := t3)
      (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
      (inH := inH) (inW := inW) (h1 := hInC) (h2 := hKH) (h3 := hKW)
      kId bId xId)
  let yCpu ← Utils.cpuValue (s := yShape) t4 yId
  let seedCpu : Spec.PackedTensor Float := Spec.PackedTensor.ofTensor (fill (1.0 : Float) yShape)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t4) yId seedCpu)
  let dKCpu ← Utils.cpuGrad (s := shape![outC, inC, kH, kW]) gradsCpu kId
  let dBCpu ← Utils.cpuGrad (s := shape![outC]) gradsCpu bId
  let dXCpu ← Utils.cpuGrad (s := shape![inC, inH, inW]) gradsCpu xId

  -- CUDA tape
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, kIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer kernel)
    (name := some "kernel")
  let (t2c, bIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t1c) (Utils.tensorToAnyBuffer bias)
    (name := some "bias")
  let (t3c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t2c) (Utils.tensorToAnyBuffer input)
    (name := some "input")
  let (t4c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.conv2d (t := t3c)
      (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
      (inH := inH) (inW := inW) (h1 := hInC) (h2 := hKH) (h3 := hKW)
      kIdc bIdc xIdc)
  let yCuda ← Utils.cudaValue (s := yShape) t4c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := yShape, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Spec.Shape.size yShape)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t4c) yIdc seedCuda)
  let dKCuda ← Utils.cudaGrad (s := shape![outC, inC, kH, kW]) gradsCuda kIdc
  let dBCuda ← Utils.cudaGrad (s := shape![outC]) gradsCuda bIdc
  let dXCuda ← Utils.cudaGrad (s := shape![inC, inH, inW]) gradsCuda xIdc

  Utils.assertTensorApprox (s := yShape) "conv2d forward" yCuda yCpu (tol := 5e-3)
  Utils.assertTensorApprox (s := shape![outC, inC, kH, kW])
    "conv2d dKernel" dKCuda dKCpu (tol := 5e-3)
  Utils.assertTensorApprox (s := shape![outC]) "conv2d dBias" dBCuda dBCpu (tol := 5e-3)
  Utils.assertTensorApprox (s := shape![inC, inH, inW])
    "conv2d dInput" dXCuda dXCpu (tol := 5e-3)

def runMaxPool : IO Unit := do
  IO.println "== max_pool2d =="
  let yShape : Shape := shape![inC, outH, outW]

  -- CPU
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) input (name := some "input")
  let (t2, yId) ← Utils.okOrThrow
    (Tape.maxPool2d (α := Float) (t := t1)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := hKH) (h2 := hKW) xId)
  let yCpu ← Utils.cpuValue (s := yShape) t2 yId
  let seedCpu : Spec.PackedTensor Float := Spec.PackedTensor.ofTensor (fill (1.0 : Float) yShape)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := shape![inC, inH, inW]) gradsCpu xId

  -- CUDA
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer input)
    (name := some "input")
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.maxPool2d (t := t1c)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := hKH) (h2 := hKW) xIdc)
  let yCuda ← Utils.cudaValue (s := yShape) t2c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := yShape, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Spec.Shape.size yShape)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := shape![inC, inH, inW]) gradsCuda xIdc

  Utils.assertTensorApprox (s := yShape) "max_pool2d forward" yCuda yCpu (tol := 1e-6)
  Utils.assertTensorApprox (s := shape![inC, inH, inW]) "max_pool2d dx" dxCuda dxCpu (tol := 1e-6)

def runMaxPoolPadNegative : IO Unit := do
  IO.println "== max_pool2d padding negative inputs =="

  let x : Tensor Float (shape![1, 1, 1]) :=
    tensorOfList! [1, 1, 1] [-3.0]
  let yShape : Shape := shape![1, 2, 2]
  let expectedY : Tensor Float (shape![1, 2, 2]) :=
    tensorOfList! [1, 2, 2] [-3.0, -3.0, -3.0, -3.0]
  let expectedDx : Tensor Float (shape![1, 1, 1]) :=
    tensorOfList! [1, 1, 1] [4.0]

  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) x (name := some "input")
  let (t2, yId) ← Utils.okOrThrow
    (Tape.maxPool2dPad (α := Float) (t := t1)
      (kH := 2) (kW := 2) (inH := 1) (inW := 1) (inC := 1) (stride := 1) (padding := 1)
      (h1 := by decide) (h2 := by decide) xId)
  let yCpu ← Utils.cpuValue (s := yShape) t2 yId
  let seedCpu : Spec.PackedTensor Float := Spec.PackedTensor.ofTensor (fill (1.0 : Float) yShape)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := shape![1, 1, 1]) gradsCpu xId

  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer x)
    (name := some "input")
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.maxPool2dPad (t := t1c)
      (kH := 2) (kW := 2) (inH := 1) (inW := 1) (inC := 1) (stride := 1) (padding := 1)
      (h1 := by decide) (h2 := by decide) xIdc)
  let yCuda ← Utils.cudaValue (s := yShape) t2c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := yShape, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Spec.Shape.size yShape)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := shape![1, 1, 1]) gradsCuda xIdc

  Utils.assertTensorApprox (s := yShape) "max_pool2d_pad negative CPU expected" yCpu expectedY (tol := 1e-6)
  Utils.assertTensorApprox (s := yShape) "max_pool2d_pad negative CUDA expected" yCuda expectedY (tol := 1e-6)
  Utils.assertTensorApprox (s := shape![1, 1, 1])
    "max_pool2d_pad negative CPU dx" dxCpu expectedDx (tol := 1e-6)
  Utils.assertTensorApprox (s := shape![1, 1, 1])
    "max_pool2d_pad negative CUDA dx" dxCuda expectedDx (tol := 1e-6)

def runMaxPool3PadNegative : IO Unit := do
  IO.println "== max_pool (d=3) padding negative inputs =="

  let inSpatial : Vector Nat 3 := #v[1, 1, 1]
  let kernel : Vector Nat 3 := #v[2, 2, 2]
  let stride : Vector Nat 3 := #v[1, 1, 1]
  let padding : Vector Nat 3 := #v[1, 1, 1]
  let hKernel : ∀ i : Fin 3, kernel.get i ≠ 0 := by
    intro i
    fin_cases i <;> simp [kernel, Vector.get]
  let yShape : Shape := Shape.ofList [1, 2, 2, 2]
  let x : Tensor Float (Shape.ofList [1, 1, 1, 1]) :=
    tensorOfList! [1, 1, 1, 1] [-3.0]
  let expectedY : Tensor Float (Shape.ofList [1, 2, 2, 2]) :=
    tensorOfList! [1, 2, 2, 2] [-3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0]
  let expectedDx : Tensor Float (Shape.ofList [1, 1, 1, 1]) :=
    tensorOfList! [1, 1, 1, 1] [8.0]

  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) x (name := some "input")
  let (t2, yId) ← Utils.okOrThrow
    (Tape.maxPool (α := Float) (t := t1)
      (d := 3) (C := 1)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel) xId)
  let yCpu ← Utils.cpuValue (s := yShape) t2 yId
  let seedCpu : Spec.PackedTensor Float := Spec.PackedTensor.ofTensor (fill (1.0 : Float) yShape)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := Shape.ofList [1, 1, 1, 1]) gradsCpu xId

  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer x)
    (name := some "input")
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.maxPool (t := t1c)
      (d := 3) (C := 1)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel) xIdc)
  let yCuda ← Utils.cudaValue (s := yShape) t2c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := yShape, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Spec.Shape.size yShape)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := Shape.ofList [1, 1, 1, 1]) gradsCuda xIdc

  Utils.assertTensorApprox (s := yShape) "max_pool[d=3] pad negative CPU expected" yCpu expectedY
    (tol := 1e-6)
  Utils.assertTensorApprox (s := yShape) "max_pool[d=3] pad negative CUDA expected" yCuda expectedY
    (tol := 1e-6)
  Utils.assertTensorApprox (s := Shape.ofList [1, 1, 1, 1])
    "max_pool[d=3] pad negative CPU dx" dxCpu expectedDx (tol := 1e-6)
  Utils.assertTensorApprox (s := Shape.ofList [1, 1, 1, 1])
    "max_pool[d=3] pad negative CUDA dx" dxCuda expectedDx (tol := 1e-6)

def runSmoothMaxPool : IO Unit := do
  IO.println "== smooth_max_pool2d =="
  let yShape : Shape := shape![inC, outH, outW]
  let beta : Float := 0.5

  -- CPU
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) input (name := some "input")
  let (t2, yId) ← Utils.okOrThrow
    (Tape.smoothMaxPool2d (α := Float) (t := t1)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := hKH) (h2 := hKW) xId beta)
  let yCpu ← Utils.cpuValue (s := yShape) t2 yId
  let seedCpu : Spec.PackedTensor Float := Spec.PackedTensor.ofTensor (fill (1.0 : Float) yShape)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := shape![inC, inH, inW]) gradsCpu xId

  -- CUDA
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer input)
    (name := some "input")
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.smoothMaxPool2d (t := t1c)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := hKH) (h2 := hKW) xIdc beta)
  let yCuda ← Utils.cudaValue (s := yShape) t2c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := yShape, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Spec.Shape.size yShape)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := shape![inC, inH, inW]) gradsCuda xIdc

  Utils.assertTensorApprox (s := yShape) "smooth_max_pool2d forward" yCuda yCpu (tol := 5e-3)
  Utils.assertTensorApprox (s := shape![inC, inH, inW])
    "smooth_max_pool2d dx" dxCuda dxCpu (tol := 5e-3)

/-- Require a CUDA/runtime boundary operation to reject invalid parameters. -/
def expectCudaResultError {α : Type} (label : String) : Except String α → IO Unit
  | .error _ => pure ()
  | .ok _ => throw <| IO.userError s!"{label}: expected rejection"

/-- Require a pooling operation with invalid geometry to produce the specified empty shape. -/
def expectCudaEmptyOutput (label : String) (expectedShape : Shape)
    (result : Except String (Runtime.Autograd.Cuda.Tape × Nat)) : IO Unit := do
  let (tape, id) ← Utils.okOrThrow result
  let output ← Utils.okOrThrow <|
    Runtime.Autograd.Cuda.Tape.requireValue tape id expectedShape
  unless Runtime.Autograd.Cuda.Buffer.size output = 0 do
    throw <| IO.userError s!"{label}: expected an empty native buffer"

/-- Check the stable two-dimensional smooth-max formula at scales where $\beta x$ overflows FP32. -/
def runSmoothMaxPool2dStabilityCase (beta expectedSign : Float)
    (expectedDx : Tensor Float (shape![1, 1, 2])) : IO Unit := do
  let x : Tensor Float (shape![1, 1, 2]) :=
    tensorOfList! [1, 1, 2] [1e20, -1e20]
  let yShape : Shape := shape![1, 1, 1]

  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) x
  let (t2, yId) ← Utils.okOrThrow
    (Tape.smoothMaxPool2d (α := Float) (t := t1)
      (kH := 1) (kW := 2) (inH := 1) (inW := 2) (inC := 1) (stride := 1)
      (h1 := by decide) (h2 := by decide) xId beta)
  let yCpu ← Utils.cpuValue (s := yShape) t2 yId
  let gradsCpu ← Utils.okOrThrow
    (Tape.backwardDenseAll (α := Float) (t := t2) yId (Spec.PackedTensor.ofTensor (fill 1.0 yShape)))
  let dxCpu ← Utils.cpuGrad (s := shape![1, 1, 2]) gradsCpu xId

  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer x)
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.smoothMaxPool2d (t := t1c)
      (kH := 1) (kW := 2) (inH := 1) (inW := 2) (inC := 1) (stride := 1)
      (h1 := by decide) (h2 := by decide) xIdc beta)
  let yCuda ← Utils.cudaValue (s := yShape) t2c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := yShape, buf := Runtime.Autograd.Cuda.Buffer.full 1 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := shape![1, 1, 2]) gradsCuda xIdc

  let yCpuScalar := (Runtime.Autograd.Cuda.Convert.flattenFloat yCpu).get! 0
  let yCudaScalar := (Runtime.Autograd.Cuda.Convert.flattenFloat yCuda).get! 0
  Utils.assertApprox "smooth_max_pool2d large CPU" (yCpuScalar / 1e20) expectedSign 1e-5
  Utils.assertApprox "smooth_max_pool2d large CUDA" (yCudaScalar / 1e20) expectedSign 1e-5
  Utils.assertTensorApprox "smooth_max_pool2d large CPU gradient" dxCpu expectedDx 1e-5
  Utils.assertTensorApprox "smooth_max_pool2d large CUDA gradient" dxCuda expectedDx 1e-5

/-- Check the generic N-D smooth-max kernel and reference path under the same overflow pressure. -/
def runSmoothMaxPoolNdStabilityCase (beta expectedSign : Float)
    (expectedDx : Tensor Float (Shape.ofList [1, 2])) : IO Unit := do
  let inSpatial : Vector Nat 1 := #v[2]
  let kernel : Vector Nat 1 := #v[2]
  let stride : Vector Nat 1 := #v[1]
  let padding : Vector Nat 1 := #v[0]
  let hKernel : ∀ i : Fin 1, kernel.get i ≠ 0 := by
    intro i
    fin_cases i
    simp [kernel, Vector.get]
  let x : Tensor Float (Shape.ofList [1, 2]) :=
    tensorOfList! [1, 2] [1e20, -1e20]
  let yShape : Shape := Shape.ofList [1, 1]

  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) x
  let (t2, yId) ← Utils.okOrThrow
    (Tape.smoothMaxPool (α := Float) (t := t1)
      (d := 1) (C := 1) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding) (hKernel := hKernel) xId beta)
  let yCpu ← Utils.cpuValue (s := yShape) t2 yId
  let gradsCpu ← Utils.okOrThrow
    (Tape.backwardDenseAll (α := Float) (t := t2) yId (Spec.PackedTensor.ofTensor (fill 1.0 yShape)))
  let dxCpu ← Utils.cpuGrad (s := Shape.ofList [1, 2]) gradsCpu xId

  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer x)
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.smoothMaxPool (t := t1c)
      (d := 1) (C := 1) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding) (hKernel := hKernel) xIdc beta)
  let yCuda ← Utils.cudaValue (s := yShape) t2c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := yShape, buf := Runtime.Autograd.Cuda.Buffer.full 1 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := Shape.ofList [1, 2]) gradsCuda xIdc

  let yCpuScalar := (Runtime.Autograd.Cuda.Convert.flattenFloat yCpu).get! 0
  let yCudaScalar := (Runtime.Autograd.Cuda.Convert.flattenFloat yCuda).get! 0
  Utils.assertApprox "smooth_max_pool N-D large CPU" (yCpuScalar / 1e20) expectedSign 1e-5
  Utils.assertApprox "smooth_max_pool N-D large CUDA" (yCudaScalar / 1e20) expectedSign 1e-5
  Utils.assertTensorApprox "smooth_max_pool N-D large CPU gradient" dxCpu expectedDx 1e-5
  Utils.assertTensorApprox "smooth_max_pool N-D large CUDA gradient" dxCuda expectedDx 1e-5

/-- Stable large-magnitude behavior for positive and negative inverse temperatures. -/
def runSmoothMaxPoolStability : IO Unit := do
  IO.println "== smooth max pooling stability =="
  let maxDx2d : Tensor Float (shape![1, 1, 2]) := tensorOfList! [1, 1, 2] [1.0, 0.0]
  let minDx2d : Tensor Float (shape![1, 1, 2]) := tensorOfList! [1, 1, 2] [0.0, 1.0]
  let maxDxNd : Tensor Float (Shape.ofList [1, 2]) := tensorOfList! [1, 2] [1.0, 0.0]
  let minDxNd : Tensor Float (Shape.ofList [1, 2]) := tensorOfList! [1, 2] [0.0, 1.0]
  runSmoothMaxPool2dStabilityCase 1e20 1.0 maxDx2d
  runSmoothMaxPool2dStabilityCase (-1e20) (-1.0) minDx2d
  runSmoothMaxPoolNdStabilityCase 1e20 1.0 maxDxNd
  runSmoothMaxPoolNdStabilityCase (-1e20) (-1.0) minDxNd

/-- Invalid inverse temperatures and zero-rank N-D pooling fail before reaching native code. -/
def runSmoothMaxPoolDomainChecks : IO Unit := do
  IO.println "== smooth max pooling domain checks =="
  let x2d : Tensor Float (shape![1, 1, 2]) := tensorOfList! [1, 1, 2] [1.0, 2.0]
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) x2d
  expectCudaResultError "CPU smooth-max zero beta"
    (Tape.smoothMaxPool2d (α := Float) (t := t1)
      (kH := 1) (kW := 2) (inH := 1) (inW := 2) (inC := 1) (stride := 1)
      (h1 := by decide) (h2 := by decide) xId 0.0)
  expectCudaResultError "CPU smooth-max infinite beta"
    (Tape.smoothMaxPool2d (α := Float) (t := t1)
      (kH := 1) (kW := 2) (inH := 1) (inW := 2) (inC := 1) (stride := 1)
      (h1 := by decide) (h2 := by decide) xId (1.0 / 0.0))

  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer x2d)
  for (label, invalidBeta) in
      [("zero", 0.0), ("binary32 overflow", 1e300), ("binary32 underflow", 1e-300)] do
    expectCudaResultError s!"CUDA smooth-max {label} beta"
      (Runtime.Autograd.Cuda.Tape.smoothMaxPool2d (t := t1c)
        (kH := 1) (kW := 2) (inH := 1) (inW := 2) (inC := 1) (stride := 1)
        (h1 := by decide) (h2 := by decide) xIdc invalidBeta)

  let empty : Vector Nat 0 := #v[]
  let scalarInput : Tensor Float (Shape.ofList [1]) := tensorOfList! [1] [2.0]
  let (scalarCpu, scalarCpuId) := Tape.leaf (t := Tape.empty) scalarInput
  expectCudaResultError "CPU smooth-max zero spatial rank"
    (Tape.smoothMaxPool (α := Float) (t := scalarCpu)
      (d := 0) (C := 1) (inSpatial := empty) (kernel := empty) (stride := empty)
      (padding := empty) (hKernel := fun i => Fin.elim0 i) scalarCpuId 1.0)
  let (scalarCuda, scalarCudaId) :=
    Runtime.Autograd.Cuda.Tape.leaf (t := Runtime.Autograd.Cuda.Tape.empty)
      (Utils.tensorToAnyBuffer scalarInput)
  expectCudaResultError "CUDA smooth-max zero spatial rank"
    (Runtime.Autograd.Cuda.Tape.smoothMaxPool (t := scalarCuda)
      (d := 0) (C := 1) (inSpatial := empty) (kernel := empty) (stride := empty)
      (padding := empty) (hKernel := fun i => Fin.elim0 i) scalarCudaId 1.0)

def runAvgPool : IO Unit := do
  IO.println "== avg_pool2d =="
  let yShape : Shape := shape![inC, outH, outW]

  -- CPU
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) input (name := some "input")
  let (t2, yId) ← Utils.okOrThrow
    (Tape.avgPool2d (α := Float) (t := t1)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := hKH) (h2 := hKW) xId)
  let yCpu ← Utils.cpuValue (s := yShape) t2 yId
  let seedCpu : Spec.PackedTensor Float := Spec.PackedTensor.ofTensor (fill (1.0 : Float) yShape)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := shape![inC, inH, inW]) gradsCpu xId

  -- CUDA
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer input)
    (name := some "input")
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.avgPool2d (t := t1c)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := hKH) (h2 := hKW) xIdc)
  let yCuda ← Utils.cudaValue (s := yShape) t2c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := yShape, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Spec.Shape.size yShape)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := shape![inC, inH, inW]) gradsCuda xIdc

  Utils.assertTensorApprox (s := yShape) "avg_pool2d forward" yCuda yCpu (tol := 5e-3)
  Utils.assertTensorApprox (s := shape![inC, inH, inW]) "avg_pool2d dx" dxCuda dxCpu (tol := 5e-3)

/-- Every two-dimensional CUDA convolution/pooling wrapper rejects zero stride before its FFI. -/
def runZeroStrideChecks : IO Unit := do
  IO.println "== conv/pool zero-stride validation =="
  let unitInput : Tensor Float (shape![1, 1, 1]) := tensorOfList! [1, 1, 1] [2.0]
  let unitKernel : Tensor Float (shape![1, 1, 1, 1]) := tensorOfList! [1, 1, 1, 1] [1.0]
  let unitBias : Tensor Float (shape![1]) := tensorOfList! [1] [0.0]
  let (t1, kernelId) := Runtime.Autograd.Cuda.Tape.empty.leaf (Utils.tensorToAnyBuffer unitKernel)
  let (t2, biasId) := t1.leaf (Utils.tensorToAnyBuffer unitBias)
  let (t3, inputId) := t2.leaf (Utils.tensorToAnyBuffer unitInput)

  expectCudaResultError "conv2d zero stride"
    (Runtime.Autograd.Cuda.Tape.conv2d (t := t3)
      (inC := 1) (outC := 1) (kH := 1) (kW := 1) (stride := 0) (padding := 0)
      (inH := 1) (inW := 1) (h1 := by decide) (h2 := by decide) (h3 := by decide)
      kernelId biasId inputId)
  expectCudaResultError "conv_transpose2d zero stride"
    (Runtime.Autograd.Cuda.Tape.convTranspose2d (t := t3)
      (inC := 1) (outC := 1) (kH := 1) (kW := 1) (stride := 0) (padding := 0)
      (inH := 1) (inW := 1) (h1 := by decide) (h2 := by decide) (h3 := by decide)
      kernelId biasId inputId)
  expectCudaResultError "max_pool2d zero stride"
    (Runtime.Autograd.Cuda.Tape.maxPool2d (t := t3)
      (kH := 1) (kW := 1) (inH := 1) (inW := 1) (inC := 1) (stride := 0)
      (h1 := by decide) (h2 := by decide) inputId)
  expectCudaResultError "max_pool2d_pad zero stride"
    (Runtime.Autograd.Cuda.Tape.maxPool2dPad (t := t3)
      (kH := 1) (kW := 1) (inH := 1) (inW := 1) (inC := 1) (stride := 0)
      (padding := 0) (h1 := by decide) (h2 := by decide) inputId)
  expectCudaResultError "smooth_max_pool2d zero stride"
    (Runtime.Autograd.Cuda.Tape.smoothMaxPool2d (t := t3)
      (kH := 1) (kW := 1) (inH := 1) (inW := 1) (inC := 1) (stride := 0)
      (h1 := by decide) (h2 := by decide) inputId 1.0)
  expectCudaResultError "smooth_max_pool2d_pad zero stride"
    (Runtime.Autograd.Cuda.Tape.smoothMaxPool2dPad (t := t3)
      (kH := 1) (kW := 1) (inH := 1) (inW := 1) (inC := 1) (stride := 0)
      (padding := 0) (h1 := by decide) (h2 := by decide) inputId 1.0)
  expectCudaResultError "avg_pool2d zero stride"
    (Runtime.Autograd.Cuda.Tape.avgPool2d (t := t3)
      (kH := 1) (kW := 1) (inH := 1) (inW := 1) (inC := 1) (stride := 0)
      (by decide) (by decide) inputId)
  expectCudaResultError "avg_pool2d_pad zero stride"
    (Runtime.Autograd.Cuda.Tape.avgPool2dPad (t := t3)
      (kH := 1) (kW := 1) (inH := 1) (inW := 1) (inC := 1) (stride := 0)
      (padding := 0) (by decide) (by decide) inputId)

/-- Native output-size arithmetic agrees with the spec on empty and heavily padded geometries. -/
def runBoundaryGeometryChecks : IO Unit := do
  IO.println "== conv/pool boundary geometry =="
  let tinyInput : Tensor Float (shape![1, 1, 1]) := tensorOfList! [1, 1, 1] [2.0]
  let (tinyTape, tinyId) :=
    Runtime.Autograd.Cuda.Tape.empty.leaf (Utils.tensorToAnyBuffer tinyInput)

  -- Pooling padding beyond half the kernel is outside the valid domain and totalizes to empty.
  let hugePadding : Nat := 32768
  expectCudaEmptyOutput "max_pool2d_pad excessive padding" (shape![1, 0, 0])
    (Runtime.Autograd.Cuda.Tape.maxPool2dPad (t := tinyTape)
      (kH := 1) (kW := 1) (inH := 1) (inW := 1) (inC := 1) (stride := 1)
      (padding := hugePadding) (h1 := by decide) (h2 := by decide) tinyId)

  let unitKernel : Tensor Float (shape![1, 1, 1, 1]) := tensorOfList! [1, 1, 1, 1] [1.0]
  let unitBias : Tensor Float (shape![1]) := tensorOfList! [1] [0.0]
  let (hugeConvT1, unitKernelId) :=
    Runtime.Autograd.Cuda.Tape.empty.leaf (Utils.tensorToAnyBuffer unitKernel)
  let (hugeConvT2, unitBiasId) := hugeConvT1.leaf (Utils.tensorToAnyBuffer unitBias)
  let (hugeConvT3, tinyInputId) := hugeConvT2.leaf (Utils.tensorToAnyBuffer tinyInput)
  expectCudaResultError "conv2d oversized output"
    (Runtime.Autograd.Cuda.Tape.conv2d (t := hugeConvT3)
      (inC := 1) (outC := 1) (kH := 1) (kW := 1) (stride := 1)
      (padding := hugePadding) (inH := 1) (inW := 1)
      (h1 := by decide) (h2 := by decide) (h3 := by decide)
      unitKernelId unitBiasId tinyInputId)

  let emptyChannelInput : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := shape![0, 1, 1], buf := Runtime.Autograd.Cuda.Buffer.zeros 0 }
  let (emptyChannelTape, emptyChannelId) :=
    Runtime.Autograd.Cuda.Tape.empty.leaf emptyChannelInput
  expectCudaEmptyOutput "max_pool2d_pad excessive padding with zero channels" (shape![0, 0, 0])
    (Runtime.Autograd.Cuda.Tape.maxPool2dPad (t := emptyChannelTape)
      (kH := 1) (kW := 1) (inH := 1) (inW := 1) (inC := 0) (stride := 1)
      (padding := hugePadding) (h1 := by decide) (h2 := by decide) emptyChannelId)

  let wideSpatial : Vector Nat 2 := #v[65536, 65536]
  let unitSpatial : Vector Nat 2 := #v[1, 1]
  let noPadding : Vector Nat 2 := #v[0, 0]
  let hUnitSpatial : ∀ i : Fin 2, unitSpatial.get i ≠ 0 := by
    intro i
    fin_cases i <;> simp [unitSpatial, Vector.get]
  let hWideSpatial : ∀ i : Fin 2, wideSpatial.get i ≠ 0 := by
    intro i
    fin_cases i <;> simp [wideSpatial, Vector.get]
  let hiddenLargeInput : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := Shape.ofList [0, 65536, 65536], buf := Runtime.Autograd.Cuda.Buffer.zeros 0 }
  let (hiddenLargeInputTape, hiddenLargeInputId) :=
    Runtime.Autograd.Cuda.Tape.empty.leaf hiddenLargeInput
  expectCudaResultError "max_pool oversized input spatial product with zero channels"
    (Runtime.Autograd.Cuda.Tape.maxPool (t := hiddenLargeInputTape)
      (d := 2) (C := 0) (inSpatial := wideSpatial) (kernel := unitSpatial)
      (stride := unitSpatial) (padding := noPadding) (hKernel := hUnitSpatial)
      hiddenLargeInputId)
  expectCudaResultError "max_pool oversized kernel spatial product with zero channels"
    (Runtime.Autograd.Cuda.Tape.maxPool (t := emptyChannelTape)
      (d := 2) (C := 0) (inSpatial := unitSpatial) (kernel := wideSpatial)
      (stride := unitSpatial) (padding := noPadding) (hKernel := hWideSpatial)
      emptyChannelId)

  let hugeNdPadding : Vector Nat 2 := #v[hugePadding, hugePadding]
  let emptyKernel : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := Shape.ofList [0, 1, 1, 1], buf := Runtime.Autograd.Cuda.Buffer.zeros 0 }
  let emptyBias : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := shape![0], buf := Runtime.Autograd.Cuda.Buffer.zeros 0 }
  let (hiddenOutputT1, emptyKernelId) := Runtime.Autograd.Cuda.Tape.empty.leaf emptyKernel
  let (hiddenOutputT2, emptyBiasId) := hiddenOutputT1.leaf emptyBias
  let (hiddenOutputT3, hiddenOutputInputId) :=
    hiddenOutputT2.leaf (Utils.tensorToAnyBuffer tinyInput)
  expectCudaResultError "conv oversized output spatial product with zero output channels"
    (Runtime.Autograd.Cuda.Tape.conv (t := hiddenOutputT3)
      (d := 2) (inC := 1) (outC := 0) (inSpatial := unitSpatial) (kernel := unitSpatial)
      (stride := unitSpatial) (padding := hugeNdPadding)
      emptyKernelId emptyBiasId hiddenOutputInputId
      (hInC := by decide) (hKernel := hUnitSpatial))

  let (poolTape, poolId) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.maxPool2d (t := tinyTape)
      (kH := 3) (kW := 3) (inH := 1) (inW := 1) (inC := 1) (stride := 1)
      (h1 := by decide) (h2 := by decide) tinyId)
  let empty2dShape : Shape := shape![1, 0, 0]
  let emptyPool ← Utils.okOrThrow <|
    Runtime.Autograd.Cuda.Tape.requireValue poolTape poolId empty2dShape
  unless Runtime.Autograd.Cuda.Buffer.size emptyPool = 0 do
    throw <| IO.userError "max_pool2d invalid geometry produced a nonempty native buffer"

  let wideKernel : Tensor Float (shape![1, 1, 3, 3]) :=
    tensorOfList! [1, 1, 3, 3] [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  let bias : Tensor Float (shape![1]) := tensorOfList! [1] [0.0]
  let (convT1, kernelId) := Runtime.Autograd.Cuda.Tape.empty.leaf (Utils.tensorToAnyBuffer wideKernel)
  let (convT2, biasId) := convT1.leaf (Utils.tensorToAnyBuffer bias)
  let (convT3, inputId) := convT2.leaf (Utils.tensorToAnyBuffer tinyInput)
  let (convT4, convId) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.conv2d (t := convT3)
      (inC := 1) (outC := 1) (kH := 3) (kW := 3) (stride := 1) (padding := 0)
      (inH := 1) (inW := 1) (h1 := by decide) (h2 := by decide) (h3 := by decide)
      kernelId biasId inputId)
  let emptyConv ← Utils.okOrThrow <|
    Runtime.Autograd.Cuda.Tape.requireValue convT4 convId empty2dShape
  unless Runtime.Autograd.Cuda.Buffer.size emptyConv = 0 do
    throw <| IO.userError "conv2d invalid geometry produced a nonempty native buffer"

  let inSpatial : Vector Nat 1 := #v[1]
  let poolKernel : Vector Nat 1 := #v[2]
  let poolStride : Vector Nat 1 := #v[1]
  let excessivePadding : Vector Nat 1 := #v[2]
  let hPoolKernel : ∀ i : Fin 1, poolKernel.get i ≠ 0 := by
    intro i
    fin_cases i
    simp [poolKernel, Vector.get]
  let ndInput : Tensor Float (Shape.ofList [1, 1]) := tensorOfList! [1, 1] [2.0]
  let (ndTape, ndInputId) :=
    Runtime.Autograd.Cuda.Tape.empty.leaf (Utils.tensorToAnyBuffer ndInput)
  let emptyNdShape : Shape := Shape.ofList [1, 0]
  let (maxTape, maxId) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.maxPool (t := ndTape)
      (d := 1) (C := 1) (inSpatial := inSpatial) (kernel := poolKernel)
      (stride := poolStride) (padding := excessivePadding) (hKernel := hPoolKernel) ndInputId)
  let (avgTape, avgId) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.avgPool (t := ndTape)
      (d := 1) (C := 1) (inSpatial := inSpatial) (kernel := poolKernel)
      (stride := poolStride) (padding := excessivePadding) hPoolKernel ndInputId)
  let (smoothTape, smoothId) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.smoothMaxPool (t := ndTape)
      (d := 1) (C := 1) (inSpatial := inSpatial) (kernel := poolKernel)
      (stride := poolStride) (padding := excessivePadding) (hKernel := hPoolKernel) ndInputId 1.0)
  for (label, tape, id) in
      [("max_pool", maxTape, maxId), ("avg_pool", avgTape, avgId),
       ("smooth_max_pool", smoothTape, smoothId)] do
    let output ← Utils.okOrThrow <|
      Runtime.Autograd.Cuda.Tape.requireValue tape id emptyNdShape
    unless Runtime.Autograd.Cuda.Buffer.size output = 0 do
      throw <| IO.userError s!"{label} invalid N-D geometry produced a nonempty native buffer"

  let convKernelDims : Vector Nat 1 := #v[1]
  let convPadding : Vector Nat 1 := #v[2]
  let hConvKernel : ∀ i : Fin 1, convKernelDims.get i ≠ 0 := by
    intro i
    fin_cases i
    simp [convKernelDims, Vector.get]
  let ndKernel : Tensor Float (Shape.ofList [1, 1, 1]) := tensorOfList! [1, 1, 1] [1.0]
  let (ndConvT1, ndKernelId) :=
    Runtime.Autograd.Cuda.Tape.empty.leaf (Utils.tensorToAnyBuffer ndKernel)
  let (ndConvT2, ndBiasId) := ndConvT1.leaf (Utils.tensorToAnyBuffer bias)
  let (ndConvT3, ndConvInputId) := ndConvT2.leaf (Utils.tensorToAnyBuffer ndInput)
  let (ndConvT4, ndConvId) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.conv (t := ndConvT3)
      (d := 1) (inC := 1) (outC := 1) (inSpatial := inSpatial) (kernel := convKernelDims)
      (stride := poolStride) (padding := convPadding) ndKernelId ndBiasId ndConvInputId
      (hInC := by decide) (hKernel := hConvKernel))
  let paddedConvShape : Shape := Shape.ofList [1, 5]
  let paddedConv ← Utils.okOrThrow <|
    Runtime.Autograd.Cuda.Tape.requireValue ndConvT4 ndConvId paddedConvShape
  unless Runtime.Autograd.Cuda.Buffer.size paddedConv = 5 do
    throw <| IO.userError "N-D convolution incorrectly applied pooling padding restrictions"

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: conv2d + pooling ==="
  runConv2d
  runConv3
  runMaxPool
  runMaxPoolPadNegative
  runMaxPool3
  runMaxPool3PadNegative
  runSmoothMaxPool
  runSmoothMaxPool3
  runSmoothMaxPoolStability
  runSmoothMaxPoolDomainChecks
  runAvgPool
  runAvgPool3
  runZeroStrideChecks
  runBoundaryGeometryChecks

end ConvPool
end Cuda
end Tests
