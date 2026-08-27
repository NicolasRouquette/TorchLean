/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.Tests.Runtime.Floats.Utils
public import Std

/-!
# TorchLeanIndexShapeCheck

Runtime checks for TorchLean indexing and shape helpers over floats.

These checks focus on shape-manipulating ops and indexing helpers that are easy to break when
refactoring tensor APIs.
-/

@[expose] public section

open Spec
open Tensor
open Tests.Floats.Utils

namespace Tests
namespace Floats
namespace TorchLeanIndexShapeCheck

/-- A task head preserves every leading axis rather than treating only axis zero as a batch. -/
def multiLeadingClassifier :
    TorchLean.nn.Sequential [2, 3, 4, 5] [2, 3, 7] :=
  by
    simpa [Spec.Shape.ofList, Spec.Shape.concat, Spec.Shape.appendDim] using
      (TorchLean.nn.heads.classifier (leading := [2, 3]) (shape := [4, 5]) 7)

/-- Check that prefix flattening preserves every leading slice and row-major suffix order. -/
def checkFlattenThenTake : IO Unit := do
  let x : Tensor Float [2, 2, 2, 2] :=
    Tensor.generate [2, 2, 2, 2] fun coordinate =>
      Float.ofNat (100 * coordinate.getD 0 0 + 10 * coordinate.getD 1 0 +
        2 * coordinate.getD 2 0 + coordinate.getD 3 0)
  let y : Tensor Float [2, 2, 3] :=
    TorchLean.Tensor.flattenThenTake [2, 2] 3 (by decide) x
  for i in List.finRange 2 do
    for j in List.finRange 2 do
      let row := Spec.get (Spec.get y i) j
      for k in List.finRange 3 do
        assertApprox s!"flattenThenTake[{i.val},{j.val},{k.val}]"
          (vecVal row k) (Float.ofNat (100 * i.val + 10 * j.val + k.val))

def gradSelect (execution : _root_.Runtime.Autograd.Torch.ExecutionMode) :
    IO (Tensor Float [3]) := do
  let sess ← _root_.Runtime.Autograd.TorchLean.Session.new (α := Float)
    (opts := { execution := execution })
  let xVal : Tensor Float [3] := tensor! (ty := Float) [1.0, 2.0, 3.0]
  let x : _root_.Runtime.Autograd.Torch.TensorRef Float [3] ←
    _root_.Runtime.Autograd.TorchLean.Session.input sess xVal
      (name := some "x") (requiresGrad := true)
  let y : _root_.Runtime.Autograd.Torch.TensorRef Float Shape.scalar ←
    _root_.Runtime.Autograd.TorchLean.Session.select
      (α := Float) (shape := Shape.ofList [3]) sess 0 x ⟨1, by decide⟩
  let grads ← _root_.Runtime.Autograd.TorchLean.Session.backwardScalarDenseAll sess y
  _root_.Runtime.Autograd.TorchLean.Session.grad sess grads x

def gradIndexSelectVector (execution : _root_.Runtime.Autograd.Torch.ExecutionMode) :
    IO (Tensor Float [3]) := do
  let sess ← _root_.Runtime.Autograd.TorchLean.Session.new (α := Float)
    (opts := { execution := execution })
  let xVal : Tensor Float [3] := tensor! (ty := Float) [1.0, 2.0, 3.0]
  let indices : Fin 3 → Fin 3 :=
    ![⟨2, by decide⟩, ⟨0, by decide⟩, ⟨2, by decide⟩]
  let idx : Tensor (Fin 3) [3] := Tensor.ofFn indices
  let x : _root_.Runtime.Autograd.Torch.TensorRef Float [3] ←
    _root_.Runtime.Autograd.TorchLean.Session.input sess xVal
      (name := some "x") (requiresGrad := true)
  let y ← _root_.Runtime.Autograd.TorchLean.Session.indexSelect
    (α := Float) (shape := Shape.ofList [3]) sess 0 3 x idx
  let total ← _root_.Runtime.Autograd.TorchLean.Session.sum sess (sh := [3]) y
  let grads ← _root_.Runtime.Autograd.TorchLean.Session.backwardScalarDenseAll sess total
  _root_.Runtime.Autograd.TorchLean.Session.grad sess grads x

def gradIndexSelectRows (execution : _root_.Runtime.Autograd.Torch.ExecutionMode) :
    IO (Tensor Float [3, 2]) :=
  do
  let sess ← _root_.Runtime.Autograd.TorchLean.Session.new (α := Float)
    (opts := { execution := execution })
  let xVal : Tensor Float [3, 2] :=
    Tensor.generate [3, 2] fun coordinate =>
      Float.ofNat (coordinate.getD 0 0 * 10 + coordinate.getD 1 0 + 1)
  let idx : Tensor (Fin 3) [2] :=
    Tensor.ofFn (fun _ => ⟨2, by decide⟩)
  let x : _root_.Runtime.Autograd.Torch.TensorRef Float [3, 2] ←
    _root_.Runtime.Autograd.TorchLean.Session.input sess xVal
      (name := some "x") (requiresGrad := true)
  let y ← _root_.Runtime.Autograd.TorchLean.Session.indexSelect
    (α := Float) (shape := Shape.ofList [3, 2]) sess 0 2 x idx
  let total ← _root_.Runtime.Autograd.TorchLean.Session.sum sess (sh := [2, 2]) y
  let grads ← _root_.Runtime.Autograd.TorchLean.Session.backwardScalarDenseAll sess total
  _root_.Runtime.Autograd.TorchLean.Session.grad sess grads x

def gradBroadcastScalar (execution : _root_.Runtime.Autograd.Torch.ExecutionMode) : IO Float := do
  let sess ← _root_.Runtime.Autograd.TorchLean.Session.new (α := Float)
    (opts := { execution := execution })
  let sVal : Tensor Float Shape.scalar := Tensor.scalar 2.0
  let sRef ← _root_.Runtime.Autograd.TorchLean.Session.input sess sVal (name := some "s") (requiresGrad := true)
  let cb : Shape.CanBroadcastTo Shape.scalar [4] :=
    Shape.CanBroadcastTo.scalarTo [4]
  let v ← _root_.Runtime.Autograd.TorchLean.Session.broadcastTo sess
    (sh1 := Shape.scalar) (sh2 := [4]) cb sRef
  let total ← _root_.Runtime.Autograd.TorchLean.Session.sum sess (sh := [4]) v
  let grads ← _root_.Runtime.Autograd.TorchLean.Session.backwardScalarDenseAll sess total
  let dsT : Tensor Float Shape.scalar ← _root_.Runtime.Autograd.TorchLean.Session.grad
    sess (sh := Shape.scalar) grads sRef
  pure (scalarVal dsT)

def gradReshapeMat (execution : _root_.Runtime.Autograd.Torch.ExecutionMode) :
    IO (Tensor Float [2, 3]) := do
  let sess ← _root_.Runtime.Autograd.TorchLean.Session.new (α := Float)
    (opts := { execution := execution })
  let xVal : Tensor Float [2, 3] :=
    Tensor.generate [2, 3] fun coordinate =>
      Float.ofNat (coordinate.getD 0 0 * 10 + coordinate.getD 1 0)
  let x : _root_.Runtime.Autograd.Torch.TensorRef Float [2, 3] ←
    _root_.Runtime.Autograd.TorchLean.Session.input sess xVal
      (name := some "x") (requiresGrad := true)
  let h : Spec.Shape.size [2, 3] = Spec.Shape.size [6] := by decide
  let y ← _root_.Runtime.Autograd.TorchLean.Session.reshape sess
    (sh1 := [2, 3]) (sh2 := [6]) x
    h
  let total ← _root_.Runtime.Autograd.TorchLean.Session.sum sess (sh := [6]) y
  let grads ← _root_.Runtime.Autograd.TorchLean.Session.backwardScalarDenseAll sess total
  _root_.Runtime.Autograd.TorchLean.Session.grad sess grads x

def gradTransposeMat (execution : _root_.Runtime.Autograd.Torch.ExecutionMode) :
    IO (Tensor Float [2, 3]) :=
  do
  let sess ← _root_.Runtime.Autograd.TorchLean.Session.new (α := Float)
    (opts := { execution := execution })
  let xVal : Tensor Float [2, 3] :=
    Tensor.generate [2, 3] fun coordinate =>
      Float.ofNat (coordinate.getD 0 0 + coordinate.getD 1 0 + 1)
  let x : _root_.Runtime.Autograd.Torch.TensorRef Float [2, 3] ←
    _root_.Runtime.Autograd.TorchLean.Session.input sess xVal
      (name := some "x") (requiresGrad := true)
  let xt ← _root_.Runtime.Autograd.TorchLean.Session.swapAdjacentAtDepth sess 0 x
  let total ← _root_.Runtime.Autograd.TorchLean.Session.sum sess (sh := [3, 2]) xt
  let grads ← _root_.Runtime.Autograd.TorchLean.Session.backwardScalarDenseAll sess total
  _root_.Runtime.Autograd.TorchLean.Session.grad sess grads x

def gradReduceMeanVec (execution : _root_.Runtime.Autograd.Torch.ExecutionMode) :
    IO (Tensor Float [3]) := do
  let sess ← _root_.Runtime.Autograd.TorchLean.Session.new (α := Float)
    (opts := { execution := execution })
  let xVal : Tensor Float [3] := tensor! (ty := Float) [1.0, 2.0, 3.0]
  let x : _root_.Runtime.Autograd.Torch.TensorRef Float [3] ←
    _root_.Runtime.Autograd.TorchLean.Session.input sess xVal
      (name := some "x") (requiresGrad := true)
  let m : _root_.Runtime.Autograd.Torch.TensorRef Float Shape.scalar ←
    _root_.Runtime.Autograd.TorchLean.Session.reduceMean
      (α := Float) (sh := Shape.ofList [3]) sess 0 x
  let grads ← _root_.Runtime.Autograd.TorchLean.Session.backwardScalarDenseAll sess m
  _root_.Runtime.Autograd.TorchLean.Session.grad sess grads x

def gradScatterAddVec (execution : _root_.Runtime.Autograd.Torch.ExecutionMode) :
    IO (Tensor Float [3] × Float) :=
  do
  let sess ← _root_.Runtime.Autograd.TorchLean.Session.new (α := Float)
    (opts := { execution := execution })
  let xVal : Tensor Float [3] := tensor! (ty := Float) [1.0, 2.0, 3.0]
  let vVal : Tensor Float [1] := tensor! (ty := Float) [5.0]
  let idx : Tensor (Fin 3) [1] := Tensor.ofFn (fun _ => ⟨2, by decide⟩)
  let x : _root_.Runtime.Autograd.Torch.TensorRef Float [3] ←
    _root_.Runtime.Autograd.TorchLean.Session.input sess xVal
      (name := some "x") (requiresGrad := true)
  let v : _root_.Runtime.Autograd.Torch.TensorRef Float [1] ←
    _root_.Runtime.Autograd.TorchLean.Session.input sess vVal
      (name := some "v") (requiresGrad := true)
  let y ← _root_.Runtime.Autograd.TorchLean.Session.scatterAdd
    (α := Float) (shape := Shape.ofList [3]) sess 0 1 x v idx
  let total ← _root_.Runtime.Autograd.TorchLean.Session.sum sess (sh := [3]) y
  let grads ← _root_.Runtime.Autograd.TorchLean.Session.backwardScalarDenseAll sess total
  let dx ← _root_.Runtime.Autograd.TorchLean.Session.grad sess grads x
  let dvT : Tensor Float [1] ← _root_.Runtime.Autograd.TorchLean.Session.grad
    sess (sh := [1]) grads v
  pure (dx, vecVal dvT ⟨0, by decide⟩)

/-- A scalar objective whose labels are bounded by the class count in their element type. -/
def boundedLabelObjective :
    _root_.Runtime.Autograd.TorchLean.Module.ObjectiveDef (Fin 3) [[2, 3]] [] [[2]] where
  initState := .cons (Spec.fill 0.0 [2, 3]) .nil
  loss := fun {α} => by
    intro _ _
    exact fun {m} _ _ =>
      fun logits => fun labels =>
        let logitsIndexed : Runtime.Autograd.TorchLean.RefTy m α
            (([2] : Shape).concat [3]) := by simpa using logits
        let labelsIndexed : Runtime.Autograd.Torch.DataRef (m := m) (α := α) (Fin 3)
            (([2] : Shape).concat Shape.scalar) := by simpa using labels
        TorchLean.Loss.crossEntropy (m := m) (α := α)
          1 rfl logitsIndexed labelsIndexed

/-- Check that bounded labels pass through eager and typed-graph scalar objectives unchanged. -/
def boundedLabelLoss (execution : _root_.Runtime.Autograd.Torch.ExecutionMode) : IO Float := do
  let module ← _root_.Runtime.Autograd.TorchLean.Module.ObjectiveDef.instantiateFloat64
    boundedLabelObjective { execution := execution }
  let logits : Tensor Float [2, 3] :=
    Tensor.generate [2, 3] fun coordinate =>
      if coordinate.getD 0 0 = coordinate.getD 1 0 then 2.0 else 0.0
  module.loadState (.cons logits .nil)
  let labels : Tensor (Fin 3) [2] :=
    Tensor.ofFn fun row => if row = 0 then ⟨0, by decide⟩ else ⟨2, by decide⟩
  let loss ← module.loss .nil (.cons labels .nil)
  pure loss.item

/-- Check generic non-differentiable inputs under both execution modes. -/
def checkBoundedLabelObjective : IO Unit := do
  let eagerLoss ← boundedLabelLoss .eager
  let graphLoss ← boundedLabelLoss .typedGraph
  assertApprox "bounded-label loss eager/typed-graph" eagerLoss graphLoss

def run : IO Unit := do
  IO.println "torchlean_index_shape_check: begin"

  let gE ← gradSelect .eager
  let gC ← gradSelect .typedGraph
  for i in List.finRange 3 do
    assertApprox s!"gather grad[{i.val}] eager/typed-graph" (vecVal gE i) (vecVal gC i)
  for i in List.finRange 3 do
    assertApprox s!"gather grad[{i.val}] expected" (vecVal gE i) (if i.val = 1 then 1.0 else 0.0)

  let gvE ← gradIndexSelectVector .eager
  let gvC ← gradIndexSelectVector .typedGraph
  for i in List.finRange 3 do
    assertApprox s!"indexSelect vector dx[{i.val}] eager/typed-graph"
      (vecVal gvE i) (vecVal gvC i)
  assertApprox "indexSelect vector dx[0] expected" (vecVal gvE ⟨0, by decide⟩) 1.0
  assertApprox "indexSelect vector dx[1] expected" (vecVal gvE ⟨1, by decide⟩) 0.0
  assertApprox "indexSelect vector dx[2] expected" (vecVal gvE ⟨2, by decide⟩) 2.0

  let grE ← gradIndexSelectRows .eager
  let grC ← gradIndexSelectRows .typedGraph
  for i in List.finRange 3 do
    for j in List.finRange 2 do
      assertApprox s!"indexSelect rows dx[{i.val},{j.val}] eager/typed-graph"
        (matVal grE i j) (matVal grC i j)
  for j in List.finRange 2 do
    assertApprox s!"indexSelect rows dx[0,{j.val}] expected" (matVal grE ⟨0, by decide⟩ j) 0.0
    assertApprox s!"indexSelect rows dx[1,{j.val}] expected" (matVal grE ⟨1, by decide⟩ j) 0.0
    assertApprox s!"indexSelect rows dx[2,{j.val}] expected" (matVal grE ⟨2, by decide⟩ j) 2.0

  let bsE ← gradBroadcastScalar .eager
  let bsC ← gradBroadcastScalar .typedGraph
  assertApprox "broadcast scalar grad eager/typed-graph" bsE bsC
  assertApprox "broadcast scalar grad expected" bsE 4.0

  let rE ← gradReshapeMat .eager
  let rC ← gradReshapeMat .typedGraph
  for i in List.finRange 2 do
    for j in List.finRange 3 do
      assertApprox s!"reshape grad[{i.val},{j.val}] eager/typed-graph" (matVal rE i j) (matVal rC i j)
      assertApprox s!"reshape grad[{i.val},{j.val}] expected" (matVal rE i j) 1.0

  let tE ← gradTransposeMat .eager
  let tC ← gradTransposeMat .typedGraph
  for i in List.finRange 2 do
    for j in List.finRange 3 do
      assertApprox s!"transpose grad[{i.val},{j.val}] eager/typed-graph" (matVal tE i j) (matVal tC i
        j)
      assertApprox s!"transpose grad[{i.val},{j.val}] expected" (matVal tE i j) 1.0

  let mE ← gradReduceMeanVec .eager
  let mC ← gradReduceMeanVec .typedGraph
  for i in List.finRange 3 do
    assertApprox s!"reduce_mean grad[{i.val}] eager/typed-graph" (vecVal mE i) (vecVal mC i)
    assertApprox s!"reduce_mean grad[{i.val}] expected" (vecVal mE i) (1.0 / 3.0) 1e-6

  let (sxE, svE) ← gradScatterAddVec .eager
  let (sxC, svC) ← gradScatterAddVec .typedGraph
  for i in List.finRange 3 do
    assertApprox s!"scatter_add_vec dx[{i.val}] eager/typed-graph" (vecVal sxE i) (vecVal sxC i)
    assertApprox s!"scatter_add_vec dx[{i.val}] expected" (vecVal sxE i) 1.0
  assertApprox "scatter_add_vec dv eager/typed-graph" svE svC
  assertApprox "scatter_add_vec dv expected" svE 1.0

  checkFlattenThenTake
  checkBoundedLabelObjective

  IO.println "torchlean_index_shape_check: ok"

end TorchLeanIndexShapeCheck
end Floats
end Tests
