/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved.Correctness.Eval.PayloadBridge

/-!
# Compiler Payload Insertion

The forward-fragment compiler emits an IR node and, when the node needs external data, records that
data in the verifier `ParamStore` at the same fresh node id.  These lemmas pin down that insertion
step for the payload-backed constructors in the proved forward fragment.
-/

@[expose] public section

namespace NN.Verification.TorchLean.Proved

open _root_.Spec
open _root_.Spec.Tensor
open NN.IR

namespace Correctness

open NN.Verification.TorchLean

namespace IRStep

/-- Compiling a literal constant stores its flattened tensor at the fresh IR node id. -/
theorem compileNode_const_payload
    {α : Type} [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {s : Shape}
    (id : Nat)
    (wf : Shape.WellFormed s)
    (t : Tensor α s)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) :
    (compileNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
        (out := s) id (.const wf t) params ps).2.constVals.get? id =
      some (flatOfTensor (α := α) (s := s) wf t) := by
  simp [compileNode]

/-- Compiling a parameter constant stores the selected parameter tensor at the fresh IR node id. -/
theorem compileNode_paramConst_payload
    {α : Type} [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {s : Shape}
    (id : Nat)
    (wf : Shape.WellFormed s)
    (p : Idx paramShapes s)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) :
    (compileNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
        (out := s) id (.paramConst wf p) params ps).2.constVals.get? id =
      some (flatOfTensor (α := α) (s := s) wf
        (getParam (α := α) (paramShapes := paramShapes) params p)) := by
  simp [compileNode]

/-- Compiling a linear node stores exactly the selected weight and bias tensors. -/
theorem compileNode_linear_payload
    {α : Type} [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape}
    (id inDim outDim : Nat)
    (w : Idx paramShapes (.dim outDim (.dim inDim .scalar)))
    (b : Idx paramShapes (.dim outDim .scalar))
    (x : Idx (Ctx inShape ss) (.dim inDim .scalar))
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) :
    (compileNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
        (out := .dim outDim .scalar) id (.linear inDim outDim w b x) params ps).2.linearWB.get? id =
      some
        ({ m := outDim
           n := inDim
           w := getParam (α := α) (paramShapes := paramShapes) params w
           b := getParam (α := α) (paramShapes := paramShapes) params b } :
          NN.MLTheory.CROWN.Graph.LinParams α) := by
  simp [compileNode]

/-- The compiled IR node for a literal constant is the corresponding payload-backed `const` node. -/
theorem compileNode_const_node
    {α : Type} [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {s : Shape}
    (id : Nat)
    (wf : Shape.WellFormed s)
    (t : Tensor α s)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) :
    (compileNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
        (out := s) id (.const wf t) params ps).1 =
      { id := id, parents := [], kind := .const s, outShape := s } := by
  rfl

/-- The compiled IR node for a parameter constant is the corresponding payload-backed `const` node. -/
theorem compileNode_paramConst_node
    {α : Type} [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {s : Shape}
    (id : Nat)
    (wf : Shape.WellFormed s)
    (p : Idx paramShapes s)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) :
    (compileNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
        (out := s) id (.paramConst wf p) params ps).1 =
      { id := id, parents := [], kind := .const s, outShape := s } := by
  rfl

/-- The compiled IR node for a linear source node has one activation parent and external payload. -/
theorem compileNode_linear_node
    {α : Type} [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape}
    (id inDim outDim : Nat)
    (w : Idx paramShapes (.dim outDim (.dim inDim .scalar)))
    (b : Idx paramShapes (.dim outDim .scalar))
    (x : Idx (Ctx inShape ss) (.dim inDim .scalar))
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) :
    (compileNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
        (out := .dim outDim .scalar) id (.linear inDim outDim w b x) params ps).1 =
      { id := id, parents := [x.id], kind := .linear, outShape := .dim outDim .scalar } := by
  rfl

/-- Compiling a suffix preserves already-existing constant payload lookups seen by IR evaluation. -/
theorem compileFGraph_payloadOfParamStore_const?_lt
    {α : Type} [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (g : FGraph α paramShapes inShape ss out)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (c : NN.Verification.TorchLean.CompiledIR α)
    {k : Nat} (hk : k < c.graph.nodes.size) :
    (payloadOfParamStore (α := α)
        (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
          (out := out) g params c).ps).const? k =
      (payloadOfParamStore (α := α) c.ps).const? k := by
  rw [payloadOfParamStore_const?_eq, payloadOfParamStore_const?_eq,
    compileFGraph_ps_constVals_get?_lt (α := α) (paramShapes := paramShapes)
      (inShape := inShape) (ss := ss) (out := out) g params c hk]

/-- Compiling a suffix preserves already-existing linear payload lookups seen by IR evaluation. -/
theorem compileFGraph_payloadOfParamStore_linear?_lt
    {α : Type} [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (g : FGraph α paramShapes inShape ss out)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (c : NN.Verification.TorchLean.CompiledIR α)
    {k : Nat} (hk : k < c.graph.nodes.size) :
    (payloadOfParamStore (α := α)
        (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
          (out := out) g params c).ps).linear? k =
      (payloadOfParamStore (α := α) c.ps).linear? k := by
  rw [payloadOfParamStore_linear?_eq, payloadOfParamStore_linear?_eq,
    compileFGraph_ps_linearWB_get?_lt (α := α) (paramShapes := paramShapes)
      (inShape := inShape) (ss := ss) (out := out) g params c hk]

/-- Compiling a suffix preserves already-existing convolution payload lookups seen by IR evaluation. -/
theorem compileFGraph_payloadOfParamStore_conv2d?_lt
    {α : Type} [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (g : FGraph α paramShapes inShape ss out)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (c : NN.Verification.TorchLean.CompiledIR α)
    {k : Nat} (hk : k < c.graph.nodes.size) :
    (payloadOfParamStore (α := α)
        (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
          (out := out) g params c).ps).conv2d? k =
      (payloadOfParamStore (α := α) c.ps).conv2d? k := by
  rw [payloadOfParamStore_conv2d?_eq, payloadOfParamStore_conv2d?_eq,
    compileFGraph_ps_conv2dCfg_get?_lt (α := α) (paramShapes := paramShapes)
      (inShape := inShape) (ss := ss) (out := out) g params c hk]

/-- Compiling a suffix preserves already-existing BatchNorm payload lookups seen by IR evaluation. -/
theorem compileFGraph_payloadOfParamStore_batchNorm2dNchwEval?_lt
    {α : Type} [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (g : FGraph α paramShapes inShape ss out)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (c : NN.Verification.TorchLean.CompiledIR α)
    {k : Nat} (hk : k < c.graph.nodes.size) :
    (payloadOfParamStore (α := α)
        (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
          (out := out) g params c).ps).batchNorm2dNchwEval? k =
      (payloadOfParamStore (α := α) c.ps).batchNorm2dNchwEval? k := by
  rw [payloadOfParamStore_batchNorm2dNchwEval?_eq, payloadOfParamStore_batchNorm2dNchwEval?_eq,
    compileFGraph_ps_batchNorm2dNchwEval_get?_lt (α := α) (paramShapes := paramShapes)
      (inShape := inShape) (ss := ss) (out := out) g params c hk]

end IRStep

end Correctness

end NN.Verification.TorchLean.Proved
