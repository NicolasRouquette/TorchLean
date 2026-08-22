/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved.Correctness.Eval.BatchNorm
public import NN.Verification.TorchLean.Proved.Correctness.Eval.PayloadOps

/-!
# ParamStore to IR Payload Bridge

The verifier pipeline stores constants and layer parameters in `ParamStore`; the executable IR
semantics reads them through `Payload`.  These lemmas make that boundary explicit for every
payload-backed evaluator path.
-/

@[expose] public section

namespace NN.Verification.TorchLean.Proved

open _root_.Spec
open NN.IR

namespace Correctness

namespace IRStep

open NN.Verification.TorchLean

/-- Convert a verifier flat vector into the IR constant payload format. -/
def irConstOfFlatVec {α : Type} [Context α]
    (c : NN.MLTheory.CROWN.Graph.FlatVec α) : ConstFlat α :=
  { n := c.n, v := c.v }

/-- Convert verifier linear parameters into the IR linear payload format. -/
def irLinearOfLinParams {α : Type} [Context α]
    (p : NN.MLTheory.CROWN.Graph.LinParams α) : LinearWB α :=
  { outDim := p.m, inDim := p.n, W := p.w, b := p.b }

/-- Constants are forwarded from `ParamStore.constVals` to `Payload.const?` without changing data. -/
theorem payloadOfParamStore_const?_eq
    {α : Type} [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) (id : Nat) :
    (payloadOfParamStore (α := α) ps).const? id =
      (ps.constVals.get? id).map (irConstOfFlatVec (α := α)) := by
  rfl

/-- A present constant entry becomes the matching IR constant payload. -/
theorem payloadOfParamStore_const?_some
    {α : Type} [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) (id : Nat)
    (c : NN.MLTheory.CROWN.Graph.FlatVec α)
    (h : ps.constVals.get? id = some c) :
    (payloadOfParamStore (α := α) ps).const? id =
      some (irConstOfFlatVec (α := α) c) := by
  rw [payloadOfParamStore_const?_eq, h]
  rfl

/-- Missing constant entries remain missing after converting to an IR payload. -/
theorem payloadOfParamStore_const?_none
    {α : Type} [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) (id : Nat)
    (h : ps.constVals.get? id = none) :
    (payloadOfParamStore (α := α) ps).const? id = none := by
  rw [payloadOfParamStore_const?_eq, h]
  rfl

/-- Linear weights are forwarded from `ParamStore.linearWB` to `Payload.linear?`. -/
theorem payloadOfParamStore_linear?_eq
    {α : Type} [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) (id : Nat) :
    (payloadOfParamStore (α := α) ps).linear? id =
      (ps.linearWB.get? id).map (irLinearOfLinParams (α := α)) := by
  rfl

/-- A present linear entry becomes the matching IR linear payload. -/
theorem payloadOfParamStore_linear?_some
    {α : Type} [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) (id : Nat)
    (p : NN.MLTheory.CROWN.Graph.LinParams α)
    (h : ps.linearWB.get? id = some p) :
    (payloadOfParamStore (α := α) ps).linear? id =
      some (irLinearOfLinParams (α := α) p) := by
  rw [payloadOfParamStore_linear?_eq, h]
  rfl

/-- Missing linear entries remain missing after converting to an IR payload. -/
theorem payloadOfParamStore_linear?_none
    {α : Type} [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) (id : Nat)
    (h : ps.linearWB.get? id = none) :
    (payloadOfParamStore (α := α) ps).linear? id = none := by
  rw [payloadOfParamStore_linear?_eq, h]
  rfl

/-- Convolution parameters are forwarded from `ParamStore.conv2dCfg` to `Payload.conv2d?`. -/
theorem payloadOfParamStore_conv2d?_eq
    {α : Type} [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) (id : Nat) :
    (payloadOfParamStore (α := α) ps).conv2d? id = ps.conv2dCfg.get? id := by
  rfl

/-- A present convolution entry becomes the matching IR convolution payload. -/
theorem payloadOfParamStore_conv2d?_some
    {α : Type} [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) (id : Nat)
    (cfg : NN.IR.Conv2dParams α)
    (h : ps.conv2dCfg.get? id = some cfg) :
    (payloadOfParamStore (α := α) ps).conv2d? id = some cfg := by
  simpa [payloadOfParamStore_conv2d?_eq] using h

/-- Missing convolution entries remain missing after converting to an IR payload. -/
theorem payloadOfParamStore_conv2d?_none
    {α : Type} [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) (id : Nat)
    (h : ps.conv2dCfg.get? id = none) :
    (payloadOfParamStore (α := α) ps).conv2d? id = none := by
  rw [payloadOfParamStore_conv2d?_eq, h]

/-- BatchNorm parameters are forwarded from `ParamStore.batchNorm2dNchwEval` to the IR payload. -/
theorem payloadOfParamStore_batchNorm2dNchwEval?_eq
    {α : Type} [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) (id : Nat) :
    (payloadOfParamStore (α := α) ps).batchNorm2dNchwEval? id =
      ps.batchNorm2dNchwEval.get? id := by
  rfl

/-- A present BatchNorm entry becomes the matching IR BatchNorm payload. -/
theorem payloadOfParamStore_batchNorm2dNchwEval?_some
    {α : Type} [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) (id : Nat)
    (p : NN.IR.BatchNorm2dNchwEvalParams α)
    (h : ps.batchNorm2dNchwEval.get? id = some p) :
    (payloadOfParamStore (α := α) ps).batchNorm2dNchwEval? id = some p := by
  simpa [payloadOfParamStore_batchNorm2dNchwEval?_eq] using h

/-- Missing BatchNorm entries remain missing after converting to an IR payload. -/
theorem payloadOfParamStore_batchNorm2dNchwEval?_none
    {α : Type} [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) (id : Nat)
    (h : ps.batchNorm2dNchwEval.get? id = none) :
    (payloadOfParamStore (α := α) ps).batchNorm2dNchwEval? id = none := by
  rw [payloadOfParamStore_batchNorm2dNchwEval?_eq, h]

/-! ## Evaluator facts for ParamStore-backed payloads -/

/-- `Graph.evalConst` reads flat constants through the `ParamStore` bridge at any node id. -/
theorem evalConst_from_paramStore
    {α : Type} [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (id : Nat) (s : Shape)
    (v : Tensor α (.dim (Spec.Shape.size s) .scalar))
    (hStore :
      ps.constVals.get? id =
        some ({ n := Spec.Shape.size s, v := v } : NN.MLTheory.CROWN.Graph.FlatVec α)) :
    Graph.evalConst (α := α) (payload := payloadOfParamStore (α := α) ps) (id := id)
        (s := s)
      =
      Except.ok (Tensor.unflattenSpec (α := α) (s := s) v) := by
  simp [Graph.evalConst,
    payloadOfParamStore_const?_some (ps := ps) (id := id)
      ({ n := Spec.Shape.size s, v := v } : NN.MLTheory.CROWN.Graph.FlatVec α) hStore,
    irConstOfFlatVec, Graph.castDimScalar, Pure.pure, Except.pure]

/-- Missing `ParamStore.constVals` entries are rejected by `Graph.evalConst` at any node id. -/
theorem evalConst_missing_from_paramStore
    {α : Type} [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (id : Nat) (s : Shape)
    (hMissing : ps.constVals.get? id = none) :
    Graph.evalConst (α := α) (payload := payloadOfParamStore (α := α) ps) (id := id)
        (s := s)
      =
      Except.error s!"IR eval: missing const payload for node {id}" := by
  simp [Graph.evalConst, payloadOfParamStore_const?_none (ps := ps) (id := id) hMissing]
  rfl

/-- `Graph.evalLinear` reads affine parameters through the `ParamStore` bridge at any node id. -/
theorem evalLinear_from_paramStore
    {α : Type} [Context α] [DecidableEq Shape]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (id outDim inDim : Nat)
    (W : Tensor α (.dim outDim (.dim inDim .scalar)))
    (b : Tensor α (.dim outDim .scalar))
    (x : Tensor α (.dim inDim .scalar))
    (hStore :
      ps.linearWB.get? id =
        some ({ m := outDim, n := inDim, w := W, b := b } :
          NN.MLTheory.CROWN.Graph.LinParams α)) :
    Graph.evalLinear (α := α) (payload := payloadOfParamStore (α := α) ps) (id := id)
        (x := Spec.PackedTensor.mk (α := α) (.dim inDim .scalar) x)
        (outShape := .dim outDim .scalar)
      =
      Except.ok
        (Spec.PackedTensor.mk (α := α) (.dim outDim .scalar)
          (Tensor.addSpec (α := α)
            (Spec.matVecMulSpec (α := α) (m := outDim) (n := inDim) W x) b)) := by
  simp [Graph.evalLinear,
    payloadOfParamStore_linear?_some (ps := ps) (id := id)
      ({ m := outDim, n := inDim, w := W, b := b } :
        NN.MLTheory.CROWN.Graph.LinParams α) hStore,
    irLinearOfLinParams, Graph.expectShape, Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Missing `ParamStore.linearWB` entries are rejected by `Graph.evalLinear` at any node id. -/
theorem evalLinear_missing_from_paramStore
    {α : Type} [Context α] [DecidableEq Shape]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (id outDim inDim : Nat)
    (x : Tensor α (.dim inDim .scalar))
    (hMissing : ps.linearWB.get? id = none) :
    Graph.evalLinear (α := α) (payload := payloadOfParamStore (α := α) ps) (id := id)
        (x := Spec.PackedTensor.mk (α := α) (.dim inDim .scalar) x)
        (outShape := .dim outDim .scalar)
      =
      Except.error s!"IR eval: missing linear payload for node {id}" := by
  simp [Graph.evalLinear,
    payloadOfParamStore_linear?_none (ps := ps) (id := id) hMissing]
  rfl

/-- `Graph.evalConv2d` reads convolution parameters through the `ParamStore` bridge at any node id. -/
theorem evalConv2d_from_paramStore
    {α : Type} [Context α] [DecidableEq Shape]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (id : Nat)
    (cfg : NN.IR.Conv2dParams α)
    (x : Tensor α (.dim cfg.inC (.dim cfg.inH (.dim cfg.inW .scalar))))
    (hStore : ps.conv2dCfg.get? id = some cfg)
    (hHeight : OpContracts.checkWindowFits "conv2d" "height" cfg.inH cfg.kH cfg.padding = .ok ())
    (hWidth : OpContracts.checkWindowFits "conv2d" "width" cfg.inW cfg.kW cfg.padding = .ok ()) :
    let outShape : Shape :=
      .dim cfg.outC
        (.dim (Spec.Shape.slidingWindowOutDim cfg.inH cfg.kH cfg.stride cfg.padding)
          (.dim (Spec.Shape.slidingWindowOutDim cfg.inW cfg.kW cfg.stride cfg.padding) .scalar))
    Graph.evalConv2d (α := α) (payload := payloadOfParamStore (α := α) ps) (id := id)
        (x := Spec.PackedTensor.mk (α := α) (.dim cfg.inC (.dim cfg.inH (.dim cfg.inW .scalar))) x)
      =
      Except.ok
        (Spec.PackedTensor.mk (α := α) outShape
          (Spec.conv2dSpec (α := α) (layer := cfg.spec) (input := x))) := by
  have hInfer :
      OpContracts.inferConv2dOutShape cfg.inC cfg.outC cfg.kH cfg.kW cfg.stride
          cfg.padding (.dim cfg.inC (.dim cfg.inH (.dim cfg.inW .scalar))) =
        Except.ok
          (.dim cfg.outC
            (.dim (Spec.Shape.slidingWindowOutDim cfg.inH cfg.kH cfg.stride cfg.padding)
              (.dim (Spec.Shape.slidingWindowOutDim cfg.inW cfg.kW cfg.stride cfg.padding) .scalar))) := by
    simp [OpContracts.inferConv2dOutShape, OpContracts.checkPositive,
      OpContracts.inferConvOutShape, Shape.toList, Shape.ofList, OpContracts.inferSlidingWindowDims, OpContracts.inferSlidingWindowDims.go, OpContracts.slideOutPad, cfg.hIn, cfg.hKH,
      cfg.hKW, cfg.hStride, hHeight, hWidth, Bind.bind, Except.bind, Pure.pure,
      Except.pure]
  simp [Graph.evalConv2d,
    payloadOfParamStore_conv2d?_some (ps := ps) (id := id) cfg hStore,
    Graph.expectShape, hInfer,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Missing `ParamStore.conv2dCfg` entries are rejected by `Graph.evalConv2d` at any node id. -/
theorem evalConv2d_missing_from_paramStore
    {α : Type} [Context α] [DecidableEq Shape]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (id : Nat)
    (cfg : NN.IR.Conv2dParams α)
    (x : Tensor α (.dim cfg.inC (.dim cfg.inH (.dim cfg.inW .scalar))))
    (hMissing : ps.conv2dCfg.get? id = none) :
    Graph.evalConv2d (α := α) (payload := payloadOfParamStore (α := α) ps) (id := id)
        (x := Spec.PackedTensor.mk (α := α) (.dim cfg.inC (.dim cfg.inH (.dim cfg.inW .scalar))) x)
      =
      Except.error s!"IR eval: missing conv2d payload for node {id}" := by
  simp [Graph.evalConv2d,
    payloadOfParamStore_conv2d?_none (ps := ps) (id := id) hMissing]
  rfl

/--
`Graph.evalBatchNorm2dNchwEval` reads eval-mode BatchNorm parameters through the `ParamStore`
bridge at any node id.
-/
theorem evalBatchNorm2dNchwEval_from_paramStore
    {α : Type} [Context α] [DecidableEq Shape]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (id n c h w : Nat)
    (gamma beta mean var : Tensor α (.dim c .scalar))
    (eps : α)
    (x : Tensor α (.dim n (.dim c (.dim h (.dim w .scalar)))))
    (hStore :
      ps.batchNorm2dNchwEval.get? id =
        some ({ c := c, gamma := gamma, beta := beta, mean := mean, var := var, eps := eps } :
          NN.IR.BatchNorm2dNchwEvalParams α)) :
    Graph.evalBatchNorm2dNchwEval (α := α) (payload := payloadOfParamStore (α := α) ps)
        (id := id)
        (x := Spec.PackedTensor.mk (α := α) (.dim n (.dim c (.dim h (.dim w .scalar)))) x)
      =
      Except.ok
        (Spec.PackedTensor.mk (α := α) (.dim n (.dim c (.dim h (.dim w .scalar))))
          (batchNorm2dNchwEvalTensor (α := α) gamma beta mean var eps x)) := by
  simp [Graph.evalBatchNorm2dNchwEval,
    payloadOfParamStore_batchNorm2dNchwEval?_some (ps := ps) (id := id)
      ({ c := c, gamma := gamma, beta := beta, mean := mean, var := var, eps := eps } :
        NN.IR.BatchNorm2dNchwEvalParams α) hStore,
    Graph.expectShape,
    batchNorm2dNchwEvalTensor, Bind.bind, Except.bind,
    Pure.pure, Except.pure]

/-- Missing `ParamStore.batchNorm2dNchwEval` entries are rejected by BatchNorm evaluation. -/
theorem evalBatchNorm2dNchwEval_missing_from_paramStore
    {α : Type} [Context α] [DecidableEq Shape]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (id n c h w : Nat)
    (x : Tensor α (.dim n (.dim c (.dim h (.dim w .scalar)))))
    (hMissing : ps.batchNorm2dNchwEval.get? id = none) :
    Graph.evalBatchNorm2dNchwEval (α := α) (payload := payloadOfParamStore (α := α) ps)
        (id := id)
        (x := Spec.PackedTensor.mk (α := α) (.dim n (.dim c (.dim h (.dim w .scalar)))) x)
      =
      Except.error s!"IR eval: missing batch_norm2d_nchw_eval payload for node {id}" := by
  simp [Graph.evalBatchNorm2dNchwEval,
    payloadOfParamStore_batchNorm2dNchwEval?_none (ps := ps) (id := id) hMissing]
  rfl

/-! ## One-step graph facts for ParamStore-backed payload nodes -/

/-- A `const` node in any graph reads its value from the matching `ParamStore.constVals` entry. -/
theorem evalAt_const_from_paramStore_of_getNode
    {α : Type} [Context α] [DecidableEq Shape]
    (g : Graph) (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (i id : Nat) (s inputShape : Shape)
    (input : Tensor α inputShape)
    (vals : Array (Spec.PackedTensor α))
    (v : Tensor α (.dim (Spec.Shape.size s) .scalar))
    (hNode :
      Graph.getNode (g := g) i =
        pure (NN.IR.Node.mk id [] (NN.IR.OpKind.const s) s))
    (hStore :
      ps.constVals.get? id =
        some ({ n := Spec.Shape.size s, v := v } : NN.MLTheory.CROWN.Graph.FlatVec α)) :
    Graph.evalAt (α := α) (g := g) (payload := payloadOfParamStore (α := α) ps)
        (input := Spec.PackedTensor.mk (α := α) inputShape input)
        (vals := vals) (i := i)
      =
      Except.ok (Spec.PackedTensor.mk (α := α) s (Tensor.unflattenSpec (α := α) (s := s) v)) := by
  simp [Graph.evalAt, Graph.evalNode, hNode,
    evalConst_from_paramStore (ps := ps) (id := id) (s := s) (v := v) hStore,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Missing `ParamStore.constVals` entries are rejected at any `const` node id. -/
theorem evalAt_const_missing_from_paramStore_of_getNode
    {α : Type} [Context α] [DecidableEq Shape]
    (g : Graph) (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (i id : Nat) (s inputShape : Shape)
    (input : Tensor α inputShape)
    (vals : Array (Spec.PackedTensor α))
    (hNode :
      Graph.getNode (g := g) i =
        pure (NN.IR.Node.mk id [] (NN.IR.OpKind.const s) s))
    (hMissing : ps.constVals.get? id = none) :
    Graph.evalAt (α := α) (g := g) (payload := payloadOfParamStore (α := α) ps)
        (input := Spec.PackedTensor.mk (α := α) inputShape input)
        (vals := vals) (i := i)
      =
      Except.error s!"IR eval: missing const payload for node {id}" := by
  simp [Graph.evalAt, Graph.evalNode, hNode,
    evalConst_missing_from_paramStore (ps := ps) (id := id) (s := s) hMissing,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- A `linear` node in any graph reads weights and bias from its `ParamStore.linearWB` entry. -/
theorem evalAt_linear_from_paramStore_of_getNode
    {α : Type} [Context α] [DecidableEq Shape]
    (g : Graph) (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (i id pId outDim inDim : Nat)
    (inputShape : Shape)
    (input : Tensor α inputShape)
    (vals : Array (Spec.PackedTensor α))
    (W : Tensor α (.dim outDim (.dim inDim .scalar)))
    (b : Tensor α (.dim outDim .scalar))
    (x : Tensor α (.dim inDim .scalar))
    (parent : Spec.PackedTensor α)
    (hNode :
      Graph.getNode (g := g) i =
        pure (NN.IR.Node.mk id [pId] NN.IR.OpKind.linear
          (Shape.dim outDim Shape.scalar)))
    (hParentValue : vals[pId]? = some parent)
    (hParent :
      Graph.expectShape (α := α) (expected := Shape.dim inDim Shape.scalar) parent =
        Except.ok x)
    (hStore :
      ps.linearWB.get? id =
        some ({ m := outDim, n := inDim, w := W, b := b } :
          NN.MLTheory.CROWN.Graph.LinParams α)) :
    Graph.evalAt (α := α) (g := g) (payload := payloadOfParamStore (α := α) ps)
        (input := Spec.PackedTensor.mk (α := α) inputShape input)
        (vals := vals) (i := i)
      =
      Except.ok
        (Spec.PackedTensor.mk (α := α) (.dim outDim .scalar)
          (Tensor.addSpec (α := α)
            (Spec.matVecMulSpec (α := α) (m := outDim) (n := inDim) W x) b)) := by
  simp [Graph.evalAt, Graph.evalNode, Graph.normalizeNodeOutput, hNode, Graph.evalLinear,
    payloadOfParamStore_linear?_some (ps := ps) (id := id)
      ({ m := outDim, n := inDim, w := W, b := b } :
        NN.MLTheory.CROWN.Graph.LinParams α) hStore,
    irLinearOfLinParams, hParentValue, hParent,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Missing `ParamStore.linearWB` entries are rejected at any `linear` node id. -/
theorem evalAt_linear_missing_from_paramStore_of_getNode
    {α : Type} [Context α] [DecidableEq Shape]
    (g : Graph) (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (i id pId outDim : Nat)
    (inputShape : Shape)
    (input : Tensor α inputShape)
    (vals : Array (Spec.PackedTensor α))
    (parent : Spec.PackedTensor α)
    (hNode :
      Graph.getNode (g := g) i =
        pure (NN.IR.Node.mk id [pId] NN.IR.OpKind.linear
          (Shape.dim outDim Shape.scalar)))
    (hParentValue : vals[pId]? = some parent)
    (hMissing : ps.linearWB.get? id = none) :
    Graph.evalAt (α := α) (g := g) (payload := payloadOfParamStore (α := α) ps)
        (input := Spec.PackedTensor.mk (α := α) inputShape input)
        (vals := vals) (i := i)
      =
      Except.error s!"IR eval: missing linear payload for node {id}" := by
  simp [Graph.evalAt, Graph.evalNode, Graph.normalizeNodeOutput, hNode, Graph.evalLinear,
    payloadOfParamStore_linear?_none (ps := ps) (id := id) hMissing, hParentValue,
    Bind.bind, Except.bind, Pure.pure, Except.pure]
  rfl

/-- A `conv2d` node in any graph reads its convolution payload from `ParamStore.conv2dCfg`. -/
theorem evalAt_conv2d_from_paramStore_of_getNode
    {α : Type} [Context α] [DecidableEq Shape]
    (g : Graph) (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (i id pId : Nat)
    (cfg : NN.IR.Conv2dParams α)
    (inputShape : Shape)
    (input : Tensor α inputShape)
    (vals : Array (Spec.PackedTensor α))
    (x : Tensor α (.dim cfg.inC (.dim cfg.inH (.dim cfg.inW .scalar))))
    (parent : Spec.PackedTensor α)
    (hNode :
      let outShape : Shape :=
        .dim cfg.outC
          (.dim (Spec.Shape.slidingWindowOutDim cfg.inH cfg.kH cfg.stride cfg.padding)
            (.dim (Spec.Shape.slidingWindowOutDim cfg.inW cfg.kW cfg.stride cfg.padding) .scalar))
      Graph.getNode (g := g) i =
        pure (NN.IR.Node.mk id [pId]
          (NN.IR.OpKind.conv2d cfg.inC cfg.outC cfg.kH cfg.kW cfg.stride cfg.padding)
          outShape))
    (hParentValue : vals[pId]? = some parent)
    (hParent :
      Graph.expectShape (α := α)
          (expected := .dim cfg.inC (.dim cfg.inH (.dim cfg.inW .scalar))) parent =
        Except.ok x)
    (hStore : ps.conv2dCfg.get? id = some cfg)
    (hHeight : OpContracts.checkWindowFits "conv2d" "height" cfg.inH cfg.kH cfg.padding = .ok ())
    (hWidth : OpContracts.checkWindowFits "conv2d" "width" cfg.inW cfg.kW cfg.padding = .ok ()) :
    let outShape : Shape :=
      .dim cfg.outC
        (.dim (Spec.Shape.slidingWindowOutDim cfg.inH cfg.kH cfg.stride cfg.padding)
          (.dim (Spec.Shape.slidingWindowOutDim cfg.inW cfg.kW cfg.stride cfg.padding) .scalar))
    Graph.evalAt (α := α) (g := g) (payload := payloadOfParamStore (α := α) ps)
        (input := Spec.PackedTensor.mk (α := α) inputShape input)
        (vals := vals) (i := i)
      =
      Except.ok
        (Spec.PackedTensor.mk (α := α) outShape
          (Spec.conv2dSpec (α := α) (layer := cfg.spec) (input := x))) := by
  have hParentShape :
      parent.shape = .dim cfg.inC (.dim cfg.inH (.dim cfg.inW .scalar)) :=
    shape_eq_of_expectShape_eq_ok hParent
  have hInfer :
      OpContracts.inferConv2dOutShape cfg.inC cfg.outC cfg.kH cfg.kW cfg.stride
          cfg.padding parent.shape =
        Except.ok
          (.dim cfg.outC
            (.dim (Spec.Shape.slidingWindowOutDim cfg.inH cfg.kH cfg.stride cfg.padding)
              (.dim (Spec.Shape.slidingWindowOutDim cfg.inW cfg.kW cfg.stride cfg.padding) .scalar))) := by
    rw [hParentShape]
    simp [OpContracts.inferConv2dOutShape, OpContracts.checkPositive,
      OpContracts.inferConvOutShape, Shape.toList, Shape.ofList, OpContracts.inferSlidingWindowDims, OpContracts.inferSlidingWindowDims.go, OpContracts.slideOutPad, cfg.hIn, cfg.hKH,
      cfg.hKW, cfg.hStride, hHeight, hWidth, Bind.bind, Except.bind, Pure.pure,
      Except.pure]
  simp [Graph.evalAt, Graph.evalNode, Graph.normalizeNodeOutput, hNode, Graph.evalConv2d,
    payloadOfParamStore_conv2d?_some (ps := ps) (id := id) cfg hStore,
    hParentValue, hParent, hInfer, shapeBNe_refl,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Missing `ParamStore.conv2dCfg` entries are rejected at any `conv2d` node id. -/
theorem evalAt_conv2d_missing_from_paramStore_of_getNode
    {α : Type} [Context α] [DecidableEq Shape]
    (g : Graph) (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (i id pId : Nat)
    (cfg : NN.IR.Conv2dParams α)
    (inputShape : Shape)
    (input : Tensor α inputShape)
    (vals : Array (Spec.PackedTensor α))
    (parent : Spec.PackedTensor α)
    (hNode :
      let outShape : Shape :=
        .dim cfg.outC
          (.dim (Spec.Shape.slidingWindowOutDim cfg.inH cfg.kH cfg.stride cfg.padding)
            (.dim (Spec.Shape.slidingWindowOutDim cfg.inW cfg.kW cfg.stride cfg.padding) .scalar))
      Graph.getNode (g := g) i =
        pure (NN.IR.Node.mk id [pId]
          (NN.IR.OpKind.conv2d cfg.inC cfg.outC cfg.kH cfg.kW cfg.stride cfg.padding)
          outShape))
    (hParentValue : vals[pId]? = some parent)
    (hMissing : ps.conv2dCfg.get? id = none) :
    Graph.evalAt (α := α) (g := g) (payload := payloadOfParamStore (α := α) ps)
        (input := Spec.PackedTensor.mk (α := α) inputShape input)
        (vals := vals) (i := i)
      =
      Except.error s!"IR eval: missing conv2d payload for node {id}" := by
  simp [Graph.evalAt, Graph.evalNode, Graph.normalizeNodeOutput, hNode, Graph.evalConv2d,
    payloadOfParamStore_conv2d?_none (ps := ps) (id := id) hMissing, hParentValue,
    Bind.bind, Except.bind, Pure.pure, Except.pure]
  rfl

/-- A BatchNorm node in any graph reads eval-mode NCHW parameters from its ParamStore entry. -/
theorem evalAt_batchNorm2dNchwEval_from_paramStore_of_getNode
    {α : Type} [Context α] [DecidableEq Shape]
    (g : Graph) (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (i id pId n c h w : Nat)
    (inputShape : Shape)
    (input : Tensor α inputShape)
    (vals : Array (Spec.PackedTensor α))
    (gamma beta mean var : Tensor α (.dim c .scalar))
    (eps : α)
    (x : Tensor α (.dim n (.dim c (.dim h (.dim w .scalar)))))
    (parent : Spec.PackedTensor α)
    (hNode :
      Graph.getNode (g := g) i =
        pure (NN.IR.Node.mk id [pId] (NN.IR.OpKind.batchNorm2dNchwEval c)
          (Shape.dim n (Shape.dim c (Shape.dim h (Shape.dim w Shape.scalar))))))
    (hParentValue : vals[pId]? = some parent)
    (hParent :
      Graph.expectShape (α := α)
          (expected := Shape.dim n (Shape.dim c (Shape.dim h (Shape.dim w Shape.scalar))))
          parent =
        Except.ok x)
    (hStore :
      ps.batchNorm2dNchwEval.get? id =
        some ({ c := c, gamma := gamma, beta := beta, mean := mean, var := var, eps := eps } :
          NN.IR.BatchNorm2dNchwEvalParams α)) :
    Graph.evalAt (α := α) (g := g) (payload := payloadOfParamStore (α := α) ps)
        (input := Spec.PackedTensor.mk (α := α) inputShape input)
        (vals := vals) (i := i)
      =
      Except.ok
        (Spec.PackedTensor.mk (α := α) (.dim n (.dim c (.dim h (.dim w .scalar))))
          (batchNorm2dNchwEvalTensor (α := α) gamma beta mean var eps x)) := by
  have hParentShape :
      parent.shape = Shape.dim n (Shape.dim c (Shape.dim h (Shape.dim w Shape.scalar))) :=
    shape_eq_of_expectShape_eq_ok hParent
  simp [Graph.evalAt, Graph.evalNode, Graph.normalizeNodeOutput, hNode, Graph.evalBatchNorm2dNchwEval,
    payloadOfParamStore_batchNorm2dNchwEval?_some (ps := ps) (id := id)
      ({ c := c, gamma := gamma, beta := beta, mean := mean, var := var, eps := eps } :
        NN.IR.BatchNorm2dNchwEvalParams α) hStore,
    hParentValue, hParentShape, hParent,
    batchNorm2dNchwEvalTensor, shapeBNe_refl,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Missing BatchNorm ParamStore entries are rejected at any eval-mode NCHW BatchNorm node id. -/
theorem evalAt_batchNorm2dNchwEval_missing_from_paramStore_of_getNode
    {α : Type} [Context α] [DecidableEq Shape]
    (g : Graph) (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (i id pId n c h w : Nat)
    (inputShape : Shape)
    (input : Tensor α inputShape)
    (vals : Array (Spec.PackedTensor α))
    (parent : Spec.PackedTensor α)
    (hNode :
      Graph.getNode (g := g) i =
        pure (NN.IR.Node.mk id [pId] (NN.IR.OpKind.batchNorm2dNchwEval c)
          (Shape.dim n (Shape.dim c (Shape.dim h (Shape.dim w Shape.scalar))))))
    (hParentValue : vals[pId]? = some parent)
    (hMissing : ps.batchNorm2dNchwEval.get? id = none) :
    Graph.evalAt (α := α) (g := g) (payload := payloadOfParamStore (α := α) ps)
        (input := Spec.PackedTensor.mk (α := α) inputShape input)
        (vals := vals) (i := i)
      =
      Except.error s!"IR eval: missing batch_norm2d_nchw_eval payload for node {id}" := by
  simp [Graph.evalAt, Graph.evalNode, Graph.normalizeNodeOutput, hNode, Graph.evalBatchNorm2dNchwEval,
    payloadOfParamStore_batchNorm2dNchwEval?_none (ps := ps) (id := id) hMissing, hParentValue,
    Bind.bind, Except.bind, Pure.pure, Except.pure]
  rfl

end IRStep

end Correctness

end NN.Verification.TorchLean.Proved
