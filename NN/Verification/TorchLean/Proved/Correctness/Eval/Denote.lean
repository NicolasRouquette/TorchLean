/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved.Correctness.Eval.NodeShape
public import NN.Verification.TorchLean.Proved.Correctness.Eval.PayloadBridge
public import NN.Verification.TorchLean.Proved.Correctness.Eval.Elementwise

/-!
# Lowered Forward Evaluation: SSA Denotation Agreement
-/

@[expose] public section

namespace NN.Verification.TorchLean.Proved

open _root_.Spec
open _root_.Spec.Tensor
open NN.IR

namespace Correctness

open NN.Verification.TorchLean


/--
`denoteAllFrom` for the lowered IR agrees with the forward-fragment evaluator that returns all
intermediate values. Lowering preserves the full SSA value vector up to the current
lowering point, not only the final output.
-/
theorem denoteAllFrom_lowerForwardLetChain_eq_evalForwardLetChainVals
    {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (g : ForwardLetChain α paramShapes inShape ss out)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (c : NN.Verification.TorchLean.LoweredIR α)
    (x : Tensor α inShape)
    (vals : Array (Spec.PackedTensor α))
    (hSize : vals.size = c.graph.nodes.size)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss) :
    (NN.IR.Graph.denoteAllFrom (α := α)
      (g := (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape)
        (ss := ss) (out := out) g params c).graph)
      (payload := payloadOfParamStore (α := α)
        (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape)
          (ss := ss) (out := out) g params c).ps)
      (input := Spec.PackedTensor.mk (α := α) inShape x)
      (i := c.graph.nodes.size)
      (vals := vals)
        =
    evalForwardLetChainVals (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
      (out := out) g params vals) := by
    classical
    induction g generalizing c vals with
    | ret y =>
        -- No more nodes: the lowered graph doesn't add nodes, so `denoteAllFrom` returns `vals`.
        simp [lowerForwardLetChain, evalForwardLetChainVals, NN.IR.Graph.denoteAllFrom]
    | @let1 ss₀ mid₀ out₀ node gNext ih =>
      let id := c.graph.nodes.size
      let res :=
        lowerNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀) (out :=
          mid₀)
          id node params c.ps
      let n : NN.IR.Node := res.1
      let ps' : NN.MLTheory.CROWN.Graph.ParamStore α := res.2
      let c' : NN.Verification.TorchLean.LoweredIR α :=
        { c with graph := { nodes := c.graph.nodes.push n }, ps := ps', outputId := id }
      let cOut :=
        lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀ ++
          [mid₀]) (out := out₀)
          gNext params c'
      have hLt : id < cOut.graph.nodes.size := by
        have hmono :=
          lowerForwardLetChain_nodesSize_le (α := α) (paramShapes := paramShapes) (inShape := inShape)
            (ss := ss₀ ++ [mid₀]) (out := out₀) (g := gNext) (params := params) (c := c')
        have : id + 1 ≤ cOut.graph.nodes.size := by
          simpa [cOut, c', id, Array.size_push] using hmono
        exact Nat.lt_of_lt_of_le (Nat.lt_succ_self id) this
      -- Rewrite the goal to the one-step expansion and apply IH to the suffix.
      -- The only nontrivial work is showing `evalAt` matches `evalNode` at the fresh id.
      have hConst :
          cOut.ps.constVals.get? id = c'.ps.constVals.get? id :=
        lowerForwardLetChain_ps_constVals_get?_lt (α := α) (paramShapes := paramShapes) (inShape :=
          inShape)
          (ss := ss₀ ++ [mid₀]) (out := out₀) (g := gNext) (params := params) (c := c') (hk := by
            simp [c', id, Array.size_push])
      have hLin :
          cOut.ps.linearWB.get? id = c'.ps.linearWB.get? id :=
        lowerForwardLetChain_ps_linearWB_get?_lt (α := α) (paramShapes := paramShapes) (inShape := inShape)
          (ss := ss₀ ++ [mid₀]) (out := out₀) (g := gNext) (params := params) (c := c') (hk := by
            simp [c', id, Array.size_push])
      have hConv :
          cOut.ps.conv2dCfg.get? id = c'.ps.conv2dCfg.get? id :=
        lowerForwardLetChain_ps_conv2dCfg_get?_lt (α := α) (paramShapes := paramShapes)
          (inShape := inShape) (ss := ss₀ ++ [mid₀]) (out := out₀) (g := gNext)
          (params := params) (c := c') (hk := by
            simp [c', id, Array.size_push])
      -- `getNode` at the fresh index is exactly the freshly pushed node.
      have hnId : n.id = id := by
        cases node <;> simp [n, res, lowerNode, id]
      have hGetNode : NN.IR.Graph.getNode (g := cOut.graph) id = pure n := by
        have hPres :=
          lowerForwardLetChain_getNode_lt (α := α) (paramShapes := paramShapes) (inShape := inShape)
            (ss := ss₀ ++ [mid₀]) (out := out₀) (g := gNext) (params := params) (c := c')
            (i := id) (hi := by simp [c', id, Array.size_push])
        have hAtPush : NN.IR.Graph.getNode (g := c'.graph) id = pure n := by
          simp [NN.IR.Graph.getNode, NN.IR.Graph.getNode?, c', id, hnId]
        simpa [cOut] using Eq.trans hPres hAtPush
      -- One-step correctness: IR `evalAt` at the fresh id matches `evalNode`.
      have hStep :
          NN.IR.Graph.evalAt (α := α)
              (g := cOut.graph)
              (payload := payloadOfParamStore (α := α) cOut.ps)
              (input := Spec.PackedTensor.mk (α := α) inShape x)
              (vals := vals) (i := id)
            =
          evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀) (out :=
            mid₀)
              node params vals := by
        classical
        -- reduce to the freshly pushed IR node, then finish by cases on the source node
        -- (doing this case-by-case avoids `simp` timeouts on the full IR evaluator).
        cases node with
        | const wf t =>
            let : Shape.WellFormed mid₀ := wf
            -- Show the const payload is present and evaluates back to `t`.
            let flat : NN.MLTheory.CROWN.Graph.FlatVec α :=
              flatOfTensor (α := α) (s := mid₀) wf t
            have hnKind : n.kind = .const mid₀ := by
              simp [lowerNode, res, n]
            have hnParents : n.parents = [] := by
              simp [lowerNode, res, n]
            have hnOut : n.outShape = mid₀ := by
              simp [lowerNode, res, n]
            have hGet' : c'.ps.constVals.get? id = some flat := by
              have hIns : (c.ps.constVals.insert id flat).get? id = some flat := by
                -- Use the `m[k]?` lemma; it is definitionaly `m.get? k`.
                simp
              -- `c'.ps.constVals = c.ps.constVals.insert id flat`.
              simp [c', res, lowerNode, ps', flat]
            have hGet : cOut.ps.constVals.get? id = some flat :=
              hConst.trans hGet'
            have hStoreConst :
                cOut.ps.constVals.get? id =
                  some ({ n := Spec.Shape.size mid₀, v := t.flattenSpec } :
                    NN.MLTheory.CROWN.Graph.FlatVec α) := by
              simpa [flat, flatOfTensor] using hGet
            have hUF : unflattenSpec mid₀ t.flattenSpec = t := by
              simpa using (Spec.Tensor.flatten_unflatten_inverse_wf (α := α) (s := mid₀) (t := t))
            have hn :
                n = { id := id, parents := [], kind := .const mid₀, outShape := mid₀ } := by
              simp [n, res, lowerNode]
            have hGetNodeConst :
                NN.IR.Graph.getNode (g := cOut.graph) id =
                  pure ({ id := id, parents := [], kind := .const mid₀, outShape := mid₀ } :
                    NN.IR.Node) := by
              simp [hGetNode, hn]
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := Spec.PackedTensor.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok (Spec.PackedTensor.mk (α := α) mid₀ t) := by
              simpa [hUF] using
                IRStep.evalAt_const_from_paramStore_of_getNode
                  (α := α) (g := cOut.graph) (ps := cOut.ps) (i := id) (id := id)
                  (s := mid₀) (inputShape := inShape) (input := x) (vals := vals)
                  (v := t.flattenSpec)
                  hGetNodeConst hStoreConst
            have hEvalNode :
                evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀) (out
                  := mid₀)
                    (Node.const wf t) params vals
                  =
                Except.ok (Spec.PackedTensor.mk (α := α) mid₀ t) := by
              rfl
            simp [hEvalNode]
            exact hEvalAt
        | paramConst wf p =>
            let : Shape.WellFormed mid₀ := wf
            -- Same as `const`, but the stored constant comes from `params`.
            let tp : Tensor α mid₀ := getParam (α := α) (paramShapes := paramShapes) params p
            let flat : NN.MLTheory.CROWN.Graph.FlatVec α :=
              flatOfTensor (α := α) (s := mid₀) wf tp
            have hGet' : c'.ps.constVals.get? id = some flat := by
              simp [c', res, lowerNode, ps', flat, tp]
            have hGet : cOut.ps.constVals.get? id = some flat :=
              hConst.trans hGet'
            have hStoreConst :
                cOut.ps.constVals.get? id =
                  some ({ n := Spec.Shape.size mid₀, v := tp.flattenSpec } :
                    NN.MLTheory.CROWN.Graph.FlatVec α) := by
              simpa [flat, flatOfTensor] using hGet
            have hUF : unflattenSpec mid₀ tp.flattenSpec = tp := by
              simpa using (Spec.Tensor.flatten_unflatten_inverse_wf (α := α) (s := mid₀) (t :=
                tp))
            have hnKind : n.kind = .const mid₀ := by
              simp [lowerNode, res, n]
            have hnParents : n.parents = [] := by
              simp [lowerNode, res, n]
            have hnOut : n.outShape = mid₀ := by
              simp [lowerNode, res, n]
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := Spec.PackedTensor.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok (Spec.PackedTensor.mk (α := α) mid₀ tp) := by
              have hn :
                  n = { id := id, parents := [], kind := .const mid₀, outShape := mid₀ } := by
                simp [n, res, lowerNode]
              have hGetNodeConst :
                  NN.IR.Graph.getNode (g := cOut.graph) id =
                    pure ({ id := id, parents := [], kind := .const mid₀, outShape := mid₀ } :
                      NN.IR.Node) := by
                simp [hGetNode, hn]
              simpa [hUF] using
                IRStep.evalAt_const_from_paramStore_of_getNode
                  (α := α) (g := cOut.graph) (ps := cOut.ps) (i := id) (id := id)
                  (s := mid₀) (inputShape := inShape) (input := x) (vals := vals)
                  (v := tp.flattenSpec)
                  hGetNodeConst hStoreConst
            have hEvalNode :
                evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀) (out
                  := mid₀)
                    (Node.paramConst wf p) params vals
                  =
                Except.ok (Spec.PackedTensor.mk (α := α) mid₀ tp) := by
              rfl
            simpa [hEvalNode] using hEvalAt
        | add a b =>
            simpa [evalNode, IRStep.BinaryElementwiseOp.denote] using
              IRStep.evalAt_binaryElementwise_of_getNode
                (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) .add a b cOut.graph
              (payloadOfParamStore (α := α) cOut.ps) (Spec.PackedTensor.mk (α := α) inShape x) vals id n
              hShapes hGetNode
                (by simp [lowerNode, res, n, IRStep.BinaryElementwiseOp.toOpKind])
                (by simp [lowerNode, res, n])
                (by simp [lowerNode, res, n])
        | sub a b =>
            simpa [evalNode, IRStep.BinaryElementwiseOp.denote] using
              IRStep.evalAt_binaryElementwise_of_getNode
                (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) .sub a b cOut.graph
              (payloadOfParamStore (α := α) cOut.ps) (Spec.PackedTensor.mk (α := α) inShape x) vals id n
              hShapes hGetNode
                (by simp [lowerNode, res, n, IRStep.BinaryElementwiseOp.toOpKind])
                (by simp [lowerNode, res, n])
                (by simp [lowerNode, res, n])
        | mulElem a b =>
            simpa [evalNode, IRStep.BinaryElementwiseOp.denote] using
              IRStep.evalAt_binaryElementwise_of_getNode
                (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) .mul a b cOut.graph
              (payloadOfParamStore (α := α) cOut.ps) (Spec.PackedTensor.mk (α := α) inShape x) vals id n
              hShapes hGetNode
                (by simp [lowerNode, res, n, IRStep.BinaryElementwiseOp.toOpKind])
                (by simp [lowerNode, res, n])
                (by simp [lowerNode, res, n])
        | relu xIdx =>
            simpa [evalNode, IRStep.UnaryElementwiseOp.denote] using
              IRStep.evalAt_unaryElementwise_of_getNode
                (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) .relu xIdx cOut.graph
                (payloadOfParamStore (α := α) cOut.ps) (Spec.PackedTensor.mk (α := α) inShape x) vals id n
                hShapes hGetNode
                (by simp [lowerNode, res, n, IRStep.UnaryElementwiseOp.toOpKind])
                (by simp [lowerNode, res, n]) (by simp [lowerNode, res, n])
        | exp xIdx =>
            simpa [evalNode, IRStep.UnaryElementwiseOp.denote] using
              IRStep.evalAt_unaryElementwise_of_getNode
                (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) .exp xIdx cOut.graph
                (payloadOfParamStore (α := α) cOut.ps) (Spec.PackedTensor.mk (α := α) inShape x) vals id n
                hShapes hGetNode
                (by simp [lowerNode, res, n, IRStep.UnaryElementwiseOp.toOpKind])
                (by simp [lowerNode, res, n]) (by simp [lowerNode, res, n])
        | log xIdx =>
            have hnKind : n.kind = .log := by
              simp [lowerNode, res, n]
            have hnParents : n.parents = [xIdx.id] := by
              simp [lowerNode, res, n]
            have hnOut : n.outShape = mid₀ := by
              simp [lowerNode, res, n]
            have hn :
                n =
                  ({ id := id
                     parents := [xIdx.id]
                     kind := .log
                     outShape := mid₀ } : NN.IR.Node) := by
              simp [n, res, lowerNode]
            let tx : Tensor α mid₀ := tensorAt vals xIdx hShapes
            have hExpectX :
                NN.IR.Graph.expectShape (α := α) (expected := mid₀) (packedAt vals xIdx hShapes) = Except.ok tx
                  := by
              simpa [tx] using expectShape_packedAt_eq_ok vals xIdx hShapes
            have hGetValX :
                getVal (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) vals xIdx = Except.ok
                  tx := by
              simpa [tx] using getVal_eq_ok_of_shapesOfVals_eq vals xIdx hShapes
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := Spec.PackedTensor.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                if Tensor.allSpec (α := α) (s := mid₀) (fun v => decide (0 < v)) tx then
                  Except.ok (Spec.PackedTensor.mk (α := α) mid₀ (Tensor.logSpec (α := α) tx))
                else
                  throw
                    "IR eval: log: input contains values <= 0 (or NaN); use `safe_log` if you want epsilon protection" := by
              simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                NN.IR.Graph.normalizeNodeOutput, hGetNode, hn,
                getElem?_eq_some_packedAt vals xIdx hShapes, hExpectX,
                throw, throwThe, MonadExceptOf.throw, Bind.bind, Except.bind, Pure.pure, Except.pure]
            simpa [evalNode, hGetValX, Spec.PackedTensor.mk, tx, Except.bind, Except.pure, bind, pure] using hEvalAt
        | inv xIdx =>
            simpa [evalNode, IRStep.UnaryElementwiseOp.denote] using
              IRStep.evalAt_unaryElementwise_of_getNode
                (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) .inv xIdx cOut.graph
                (payloadOfParamStore (α := α) cOut.ps) (Spec.PackedTensor.mk (α := α) inShape x) vals id n
                hShapes hGetNode
                (by simp [lowerNode, res, n, IRStep.UnaryElementwiseOp.toOpKind])
                (by simp [lowerNode, res, n]) (by simp [lowerNode, res, n])
        | matmul2d m nDim p a b =>
            have hnKind : n.kind = .matmul := by
              simp [lowerNode, res, n]
            have hnParents : n.parents = [a.id, b.id] := by
              simp [lowerNode, res, n]
            have hnOut : n.outShape = .dim m (.dim p .scalar) := by
              simp [lowerNode, res, n]
            have hn :
                n =
                  ({ id := id
                     parents := [a.id, b.id]
                     kind := .matmul
                     outShape := .dim m (.dim p .scalar) } : NN.IR.Node) := by
              simp [n, res, lowerNode]
            let ta : Tensor α (.dim m (.dim nDim .scalar)) := tensorAt vals a hShapes
            let tb : Tensor α (.dim nDim (.dim p .scalar)) := tensorAt vals b hShapes
            have hExpectA :
                NN.IR.Graph.expectShape (α := α)
                    (expected := .dim m (.dim nDim .scalar)) (packedAt vals a hShapes) =
                  Except.ok ta := by
              simpa [ta] using expectShape_packedAt_eq_ok vals a hShapes
            have hExpectB :
                NN.IR.Graph.expectShape (α := α)
                    (expected := .dim nDim (.dim p .scalar)) (packedAt vals b hShapes) =
                  Except.ok tb := by
              simpa [tb] using expectShape_packedAt_eq_ok vals b hShapes
            have hGetValA :
                getVal (α := α) (inShape := inShape) (ss := ss₀)
                    (s := .dim m (.dim nDim .scalar)) vals a =
                  Except.ok ta := by
              simpa [ta] using getVal_eq_ok_of_shapesOfVals_eq vals a hShapes
            have hGetValB :
                getVal (α := α) (inShape := inShape) (ss := ss₀)
                    (s := .dim nDim (.dim p .scalar)) vals b =
                  Except.ok tb := by
              simpa [tb] using getVal_eq_ok_of_shapesOfVals_eq vals b hShapes
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := Spec.PackedTensor.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (Spec.PackedTensor.mk (α := α) (.dim m (.dim p .scalar))
                    (Tensor.matMulSpec (α := α) (m := m) (n := nDim) (p := p) ta tb)) := by
              simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                NN.IR.Graph.normalizeNodeOutput, hGetNode, hn,
                getElem?_eq_some_packedAt vals a hShapes,
                getElem?_eq_some_packedAt vals b hShapes, hExpectA, hExpectB,
                Bind.bind, Except.bind, Pure.pure, Except.pure]
            simpa [evalNode, hGetValA, hGetValB, Spec.PackedTensor.mk, ta, tb, Except.bind, Except.pure, bind, pure] using hEvalAt
        | bmm batch m nDim p a b =>
            have hnKind : n.kind = .matmul := by
              simp [lowerNode, res, n]
            have hnParents : n.parents = [a.id, b.id] := by
              simp [lowerNode, res, n]
            have hnOut : n.outShape = .dim batch (.dim m (.dim p .scalar)) := by
              simp [lowerNode, res, n]
            have hn :
                n =
                  ({ id := id
                     parents := [a.id, b.id]
                     kind := .matmul
                     outShape := .dim batch (.dim m (.dim p .scalar)) } : NN.IR.Node) := by
              simp [n, res, lowerNode]
            let ta : Tensor α (.dim batch (.dim m (.dim nDim .scalar))) :=
              tensorAt vals a hShapes
            let tb : Tensor α (.dim batch (.dim nDim (.dim p .scalar))) :=
              tensorAt vals b hShapes
            have hExpectA :
                NN.IR.Graph.expectShape (α := α)
                    (expected := .dim batch (.dim m (.dim nDim .scalar))) (packedAt vals a hShapes) =
                  Except.ok ta := by
              simpa [ta] using expectShape_packedAt_eq_ok vals a hShapes
            have hExpectB :
                NN.IR.Graph.expectShape (α := α)
                    (expected := .dim batch (.dim nDim (.dim p .scalar))) (packedAt vals b hShapes) =
                  Except.ok tb := by
              simpa [tb] using expectShape_packedAt_eq_ok vals b hShapes
            have hGetValA :
                getVal (α := α) (inShape := inShape) (ss := ss₀)
                    (s := .dim batch (.dim m (.dim nDim .scalar))) vals a =
                  Except.ok ta := by
              simpa [ta] using getVal_eq_ok_of_shapesOfVals_eq vals a hShapes
            have hGetValB :
                getVal (α := α) (inShape := inShape) (ss := ss₀)
                    (s := .dim batch (.dim nDim (.dim p .scalar))) vals b =
                  Except.ok tb := by
              simpa [tb] using getVal_eq_ok_of_shapesOfVals_eq vals b hShapes
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := Spec.PackedTensor.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (Spec.PackedTensor.mk (α := α) (.dim batch (.dim m (.dim p .scalar)))
                    (Tensor.bmmSpec (α := α) (batch := batch) (m := m) (n := nDim) (p := p) ta tb))
                      := by
              simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                NN.IR.Graph.normalizeNodeOutput, hGetNode, hn,
                getElem?_eq_some_packedAt vals a hShapes,
                getElem?_eq_some_packedAt vals b hShapes, hExpectA, hExpectB,
                Bind.bind, Except.bind, Pure.pure, Except.pure]
            simpa [evalNode, hGetValA, hGetValB, Spec.PackedTensor.mk, ta, tb, Except.bind, Except.pure, bind, pure] using hEvalAt
        | reshape inS mid₀ h xIdx =>
            have hnKind : n.kind = .reshape inS mid₀ := by
              simp [lowerNode, res, n]
            have hnParents : n.parents = [xIdx.id] := by
              simp [lowerNode, res, n]
            have hnOut : n.outShape = mid₀ := by
              simp [lowerNode, res, n]
            have hn :
                n =
                  ({ id := id
                     parents := [xIdx.id]
                     kind := .reshape inS mid₀
                     outShape := mid₀ } : NN.IR.Node) := by
              simp [n, res, lowerNode]
            let tx : Tensor α inS := tensorAt vals xIdx hShapes
            have hExpectX :
                NN.IR.Graph.expectShape (α := α) (expected := inS) (packedAt vals xIdx hShapes) = Except.ok tx
                  := by
              simpa [tx] using expectShape_packedAt_eq_ok vals xIdx hShapes
            have hGetValX :
                getVal (α := α) (inShape := inShape) (ss := ss₀) (s := inS) vals xIdx = Except.ok tx
                  := by
              simpa [tx] using getVal_eq_ok_of_shapesOfVals_eq vals xIdx hShapes
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := Spec.PackedTensor.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (Spec.PackedTensor.mk (α := α) mid₀
                    (Tensor.reshapeSpec (α := α) (s₁ := inS) (s₂ := mid₀) tx h)) := by
              simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                NN.IR.Graph.normalizeNodeOutput, hGetNode, hn,
                getElem?_eq_some_packedAt vals xIdx hShapes, hExpectX, h,
                Bind.bind, Except.bind, Pure.pure, Except.pure]
            simpa [evalNode, hGetValX, Spec.PackedTensor.mk, h, tx, Except.bind, Except.pure, bind, pure] using hEvalAt
        | swap_first_two m nDim rest xIdx =>
            have hnKind : n.kind = .swap_first_two := by
              simp [lowerNode, res, n]
            have hnParents : n.parents = [xIdx.id] := by
              simp [lowerNode, res, n]
            have hnOut : n.outShape = .dim nDim (.dim m rest) := by
              simp [lowerNode, res, n]
            have hn :
                n =
                  ({ id := id
                     parents := [xIdx.id]
                     kind := .swap_first_two
                     outShape := .dim nDim (.dim m rest) } : NN.IR.Node) := by
              simp [n, res, lowerNode]
            let tx : Tensor α (.dim m (.dim nDim rest)) := tensorAt vals xIdx hShapes
            have hExpectX :
                NN.IR.Graph.expectShape (α := α) (expected := .dim m (.dim nDim rest))
                  (packedAt vals xIdx hShapes) =
                  Except.ok tx := by
              simpa [tx] using expectShape_packedAt_eq_ok vals xIdx hShapes
            have hGetValX :
                getVal (α := α) (inShape := inShape) (ss := ss₀) (s := .dim m (.dim nDim rest)) vals
                  xIdx =
                  Except.ok tx := by
              simpa [tx] using getVal_eq_ok_of_shapesOfVals_eq vals xIdx hShapes
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := Spec.PackedTensor.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (Spec.PackedTensor.mk (α := α) (.dim nDim (.dim m rest))
                    (Tensor.swapFirstTwoSpec (α := α) (m := m) (n := nDim) (s := rest) tx)) := by
              simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                NN.IR.Graph.normalizeNodeOutput, hGetNode, hn,
                getElem?_eq_some_packedAt vals xIdx hShapes, hExpectX,
                Bind.bind, Except.bind, Pure.pure, Except.pure]
            simpa [evalNode, hGetValX, Spec.PackedTensor.mk, tx, hnOut, Except.bind, Except.pure, bind, pure] using hEvalAt
        | transpose3dLastTwo a b c xIdx =>
            have hnKind : n.kind = .transpose3dLastTwo := by
              simp [lowerNode, res, n]
            have hnParents : n.parents = [xIdx.id] := by
              simp [lowerNode, res, n]
            have hnOut : n.outShape = .dim a (.dim c (.dim b .scalar)) := by
              simp [lowerNode, res, n]
            have hn :
                n =
                  ({ id := id
                     parents := [xIdx.id]
                     kind := .transpose3dLastTwo
                     outShape := .dim a (.dim c (.dim b .scalar)) } : NN.IR.Node) := by
              simp [n, res, lowerNode]
            let tx : Tensor α (.dim a (.dim b (.dim c .scalar))) :=
              tensorAt vals xIdx hShapes
            have hExpectX :
                NN.IR.Graph.expectShape (α := α)
                    (expected := .dim a (.dim b (.dim c .scalar))) (packedAt vals xIdx hShapes) =
                  Except.ok tx := by
              simpa [tx] using expectShape_packedAt_eq_ok vals xIdx hShapes
            have hGetValX :
                getVal (α := α) (inShape := inShape) (ss := ss₀)
                    (s := .dim a (.dim b (.dim c .scalar))) vals xIdx =
                  Except.ok tx := by
              simpa [tx] using getVal_eq_ok_of_shapesOfVals_eq vals xIdx hShapes
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := Spec.PackedTensor.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (Spec.PackedTensor.mk (α := α) (.dim a (.dim c (.dim b .scalar)))
                    (Tensor.transpose3DLastTwoSpec (α := α) (a := a) (b := b) (c := c) tx)) :=
                      by
              simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                NN.IR.Graph.normalizeNodeOutput, hGetNode, hn,
                getElem?_eq_some_packedAt vals xIdx hShapes, hExpectX,
                Bind.bind, Except.bind, Pure.pure, Except.pure]
            simpa [evalNode, hGetValX, Spec.PackedTensor.mk, tx, hnOut, Except.bind, Except.pure, bind, pure] using hEvalAt
        | softmaxLast hRank xIdx =>
            have hnKind : n.kind = .softmax (Spec.Shape.rank mid₀ - 1) := by
              simp [lowerNode, res, n]
            have hnParents : n.parents = [xIdx.id] := by
              simp [lowerNode, res, n]
            have hnOut : n.outShape = mid₀ := by
              simp [lowerNode, res, n]
            have hn :
                n =
                  ({ id := id
                     parents := [xIdx.id]
                     kind := .softmax (Spec.Shape.rank mid₀ - 1)
                     outShape := mid₀ } : NN.IR.Node) := by
              simp [n, res, lowerNode]
            let tx : Tensor α mid₀ := tensorAt vals xIdx hShapes
            have hExpectX :
                NN.IR.Graph.expectShape (α := α) (expected := mid₀) (packedAt vals xIdx hShapes) = Except.ok tx
                  := by
              simpa [tx] using expectShape_packedAt_eq_ok vals xIdx hShapes
            have hGetValX :
                getVal (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) vals xIdx = Except.ok
                  tx := by
              simpa [tx] using getVal_eq_ok_of_shapesOfVals_eq vals xIdx hShapes
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := Spec.PackedTensor.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (Spec.PackedTensor.mk (α := α) mid₀
                    (Activation.softmaxLastSpec (α := α) tx)) := by
              cases hAxis : Shape.axisInBounds? (Shape.rank mid₀ - 1) mid₀ with
              | none =>
                  have hSome := Shape.axisInBounds?_isSome
                    (axis := Shape.rank mid₀ - 1) (s := mid₀)
                    (h := Shape.axisInBoundsLast hRank)
                  simp [hAxis] at hSome
              | some h =>
                  simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                    NN.IR.Graph.normalizeNodeOutput, hGetNode, hn,
                    getElem?_eq_some_packedAt vals xIdx hShapes, hAxis,
                    hExpectX, Bind.bind, Except.bind, Pure.pure, Except.pure]
                  exact Activation.softmaxSpec_last (α := α) hRank tx
            simpa [evalNode, hGetValX, Spec.PackedTensor.mk, tx, Except.bind, Except.pure, bind, pure] using hEvalAt
        | layernorm2d seqLen embedDim hSeq hEmb xIdx =>
            have hParams :
                OpContracts.layerNormMatrixDims 1
                    (Shape.dim seqLen (Shape.dim embedDim Shape.scalar)) =
                  .ok (seqLen, embedDim) := by
              have hSeqNe : seqLen ≠ 0 := Nat.ne_of_gt hSeq
              have hEmbNe : embedDim ≠ 0 := Nat.ne_of_gt hEmb
              -- For a `(seqLen × embedDim)` tensor, `axis=1` normalizes the last dimension.
              simp [OpContracts.layerNormMatrixDims, OpContracts.checkAxisValid,
                OpContracts.checkPositive, Shape.toList, hSeqNe, hEmbNe]
              rfl
            have hxF : (packedAt vals xIdx hShapes).shape =
                .dim seqLen (.dim embedDim .scalar) :=
              packedAt_shape vals xIdx hShapes
            have hnKind : n.kind = .layernorm 1 := by
              simp [lowerNode, res, n]
            have hnParents : n.parents = [xIdx.id] := by
              simp [lowerNode, res, n]
            have hnOut : n.outShape = .dim seqLen (.dim embedDim .scalar) := by
              simp [lowerNode, res, n]
            have hn :
                n =
                  ({ id := id
                     parents := [xIdx.id]
                     kind := .layernorm 1
                     outShape := .dim seqLen (.dim embedDim .scalar) } : NN.IR.Node) := by
              simp [n, res, lowerNode]
            have hGetNodeLN :
                NN.IR.Graph.getNode (g := cOut.graph) id =
                  pure
                    ({ id := id
                       parents := [xIdx.id]
                       kind := .layernorm 1
                       outShape := .dim seqLen (.dim embedDim .scalar) } : NN.IR.Node) := by
              simp [hGetNode, hn]
            cases hnOut
            let tx : Tensor α (.dim seqLen (.dim embedDim .scalar)) :=
              tensorAt vals xIdx hShapes
            have hExpect :
                NN.IR.Graph.expectShape (α := α)
                    (expected := .dim seqLen (.dim embedDim .scalar)) (packedAt vals xIdx hShapes) =
                  Except.ok tx := by
              simpa [tx] using expectShape_packedAt_eq_ok vals xIdx hShapes
            have hGetVal :
                getVal (α := α) (inShape := inShape) (ss := ss₀)
                    (s := .dim seqLen (.dim embedDim .scalar)) vals xIdx =
                  Except.ok tx := by
              simpa [tx] using getVal_eq_ok_of_shapesOfVals_eq vals xIdx hShapes
            have hLN :
                NN.IR.Graph.layerNormWithoutAffine (α := α) (seqLen := seqLen) (embedDim := embedDim)
                    (Tensor.reshapeSpec (α := α)
                      (s₁ := .dim seqLen (.dim embedDim .scalar))
                      (s₂ := .dim seqLen (.dim embedDim .scalar))
                      tx rfl)
                  =
                Except.ok
                  (Spec.layerNorm (α := α) (seqLen := seqLen) (embedDim := embedDim)
                    (x := Tensor.reshapeSpec (α := α)
                      (s₁ := .dim seqLen (.dim embedDim .scalar))
                      (s₂ := .dim seqLen (.dim embedDim .scalar))
                      tx rfl)
                    (gamma := Spec.fill (α := α) 1 (.dim embedDim .scalar))
                    (beta := Spec.fill (α := α) 0 (.dim embedDim .scalar))
                    (h_seq_pos := hSeq) (h_embed_pos := hEmb)) := by
              simp [NN.IR.Graph.layerNormWithoutAffine, hSeq, hEmb]
              rfl
            have hNumel :
                Spec.Shape.size (.dim seqLen (.dim embedDim .scalar)) =
                  Spec.Shape.size (.dim seqLen (.dim embedDim .scalar)) := rfl
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := Spec.PackedTensor.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (Spec.PackedTensor.mk (α := α) (.dim seqLen (.dim embedDim .scalar))
                    (Tensor.reshapeSpec (α := α)
                      (s₁ := .dim seqLen (.dim embedDim .scalar))
                      (s₂ := .dim seqLen (.dim embedDim .scalar))
                      (Spec.layerNorm (α := α) (seqLen := seqLen) (embedDim := embedDim)
                        (x := Tensor.reshapeSpec (α := α)
                          (s₁ := .dim seqLen (.dim embedDim .scalar))
                          (s₂ := .dim seqLen (.dim embedDim .scalar))
                          tx rfl)
                        (gamma := Spec.fill (α := α) 1 (.dim embedDim .scalar))
                        (beta := Spec.fill (α := α) 0 (.dim embedDim .scalar))
                        (h_seq_pos := hSeq) (h_embed_pos := hEmb))
                      rfl)) := by
                simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                  NN.IR.Graph.normalizeNodeOutput, hGetNodeLN,
                  getElem?_eq_some_packedAt vals xIdx hShapes, hExpect, hParams,
                  throw, throwThe, MonadExceptOf.throw]
                simpa [hExpect, hParams, hNumel, Spec.PackedTensor.mk, Except.bind, Except.pure, bind, pure] using
                  congrArg
                    (fun e =>
                      (fun a : Tensor α (.dim seqLen (.dim embedDim .scalar)) =>
                        Spec.PackedTensor.mk (α := α) (.dim seqLen (.dim embedDim .scalar))
                          (Tensor.reshapeSpec (α := α)
                            (s₁ := .dim seqLen (.dim embedDim .scalar))
                            (s₂ := .dim seqLen (.dim embedDim .scalar))
                            a rfl)) <$> e)
                    hLN
            simpa [evalNode, hGetVal, Spec.PackedTensor.shape, Spec.PackedTensor.tensor, hxF, Spec.PackedTensor.mk,
              tx, Tensor.reshapeSpec, Tensor.flatten_unflatten_inverse] using hEvalAt
        | linear inDim outDim w b xIdx =>
            let wT : Tensor α (.dim outDim (.dim inDim .scalar)) :=
              getParam (α := α) (paramShapes := paramShapes) params w
            let bT : Tensor α (.dim outDim .scalar) :=
              getParam (α := α) (paramShapes := paramShapes) params b
            have hxF : (packedAt vals xIdx hShapes).shape = .dim inDim .scalar :=
              packedAt_shape vals xIdx hShapes
            have hLin' : cOut.ps.linearWB[n.id]? = c'.ps.linearWB[n.id]? := by
              simpa [hnId] using hLin
            have hn :
                n =
                  ({ id := id
                     parents := [xIdx.id]
                     kind := .linear
                     outShape := .dim outDim .scalar } : NN.IR.Node) := by
              simp [n, res, lowerNode]
            have hGetNodeLinear :
                NN.IR.Graph.getNode (g := cOut.graph) id =
                  pure
                    ({ id := id
                       parents := [xIdx.id]
                       kind := .linear
                       outShape := .dim outDim .scalar } : NN.IR.Node) := by
              simp [hGetNode, hn]
            let xT : Tensor α (.dim inDim .scalar) :=
              tensorAt vals xIdx hShapes
            have hExpectIn :
                NN.IR.Graph.expectShape (α := α)
                    (expected := .dim inDim .scalar) (packedAt vals xIdx hShapes) =
                  Except.ok xT := by
              simpa [xT] using expectShape_packedAt_eq_ok vals xIdx hShapes
            have hGetVal :
                getVal (α := α) (inShape := inShape) (ss := ss₀)
                    (s := .dim inDim .scalar) vals xIdx =
                  Except.ok xT := by
              simpa [xT] using getVal_eq_ok_of_shapesOfVals_eq vals xIdx hShapes
            have hLinearStore :
                cOut.ps.linearWB.get? id =
                  some ({ m := outDim, n := inDim, w := wT, b := bT } :
                    NN.MLTheory.CROWN.Graph.LinParams α) := by
              rw [hLin]
              simp [c', res, lowerNode, ps', wT, bT]
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := Spec.PackedTensor.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (Spec.PackedTensor.mk (α := α) (.dim outDim .scalar)
                      (Tensor.addSpec (α := α)
                        (Tensor.matVecMulSpec (α := α) (m := outDim) (n := inDim) wT xT) bT)) := by
              simpa [Spec.matVecMulSpec] using
                IRStep.evalAt_linear_from_paramStore_of_getNode
                  (α := α) (g := cOut.graph) (ps := cOut.ps) (i := id) (id := id)
                  (pId := xIdx.id) (outDim := outDim) (inDim := inDim)
                  (inputShape := inShape) (input := x) (vals := vals)
                  (W := wT) (b := bT) (parent := packedAt vals xIdx hShapes) (x := xT)
                  hGetNodeLinear (getElem?_eq_some_packedAt vals xIdx hShapes)
                  hExpectIn hLinearStore
            simpa [evalNode, hGetVal, Spec.PackedTensor.shape, Spec.PackedTensor.tensor, Spec.PackedTensor.mk, hxF, xT, wT, bT] using
              hEvalAt
        | conv2d inC outC kH kW stride padding inH inW hIn hKH hKW hStride hHeight hWidth kernel bias xIdx =>
            let kT : Tensor α (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))) :=
              getParam (α := α) (paramShapes := paramShapes) params kernel
            let bT : Tensor α (.dim outC .scalar) :=
              getParam (α := α) (paramShapes := paramShapes) params bias
            let outShape : Shape :=
              .dim outC
                (.dim (Spec.Shape.slidingWindowOutDim inH kH stride padding)
                  (.dim (Spec.Shape.slidingWindowOutDim inW kW stride padding) .scalar))
            have hxF : (packedAt vals xIdx hShapes).shape =
                .dim inC (.dim inH (.dim inW .scalar)) :=
              packedAt_shape vals xIdx hShapes
            have hn :
                n =
                  ({ id := id
                     parents := [xIdx.id]
                     kind := .conv2d inC outC kH kW stride padding
                     outShape := outShape } : NN.IR.Node) := by
              simp [n, res, lowerNode, outShape]
            have hGetNodeConv :
                NN.IR.Graph.getNode (g := cOut.graph) id =
                  pure
                    ({ id := id
                       parents := [xIdx.id]
                       kind := .conv2d inC outC kH kW stride padding
                       outShape := outShape } : NN.IR.Node) := by
              simp [hGetNode, hn]
            let xT : Tensor α (.dim inC (.dim inH (.dim inW .scalar))) :=
              tensorAt vals xIdx hShapes
            have hExpectIn :
                NN.IR.Graph.expectShape (α := α)
                    (expected := .dim inC (.dim inH (.dim inW .scalar))) (packedAt vals xIdx hShapes) =
                  Except.ok xT := by
              simpa [xT] using expectShape_packedAt_eq_ok vals xIdx hShapes
            have hGetVal :
                getVal (α := α) (inShape := inShape) (ss := ss₀)
                    (s := .dim inC (.dim inH (.dim inW .scalar))) vals xIdx =
                  Except.ok xT := by
              simpa [xT] using getVal_eq_ok_of_shapesOfVals_eq vals xIdx hShapes
            let spec : Spec.Conv2dSpec inC outC kH kW stride padding α hIn hKH hKW :=
              { kernel := kT, bias := bT }
            let cfg : NN.IR.Conv2dParams α :=
              { inC := inC, outC := outC, kH := kH, kW := kW
                stride := stride, padding := padding
                inH := inH, inW := inW
                hIn := hIn, hKH := hKH, hKW := hKW, hStride := hStride,
                spec := spec }
            have hConvStore :
                cOut.ps.conv2dCfg.get? id = some cfg := by
              rw [hConv]
              simp [c', res, lowerNode, ps', cfg, spec, kT, bT]
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := Spec.PackedTensor.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (Spec.PackedTensor.mk (α := α) outShape
                    (Spec.conv2dSpec (α := α) (layer := spec) (input := xT))) := by
              simpa [cfg, spec, outShape] using
                IRStep.evalAt_conv2d_from_paramStore_of_getNode
                  (α := α) (g := cOut.graph) (ps := cOut.ps) (i := id) (id := id)
                  (pId := xIdx.id) (cfg := cfg) (inputShape := inShape) (input := x)
                  (vals := vals) (parent := packedAt vals xIdx hShapes) (x := xT)
                  hGetNodeConv (getElem?_eq_some_packedAt vals xIdx hShapes)
                  hExpectIn hConvStore hHeight hWidth
            simpa [evalNode, hGetVal, Spec.PackedTensor.shape, Spec.PackedTensor.tensor, Spec.PackedTensor.mk, hxF, xT, kT, bT,
              spec, outShape] using hEvalAt
        | mseLoss yhat target =>
            rename_i s
            let yV := packedAt vals yhat hShapes
            let tV := packedAt vals target hShapes
            have hSomeY : vals[yhat.id]? = some yV := by
              simpa [yV] using getElem?_eq_some_packedAt vals yhat hShapes
            have hSomeT : vals[target.id]? = some tV := by
              simpa [tV] using getElem?_eq_some_packedAt vals target hShapes
            have hy : (packedAt vals yhat hShapes).shape = s :=
              packedAt_shape vals yhat hShapes
            have ht : (packedAt vals target hShapes).shape = s :=
              packedAt_shape vals target hShapes
            have hnKind : n.kind = .mseLoss := by
              simp [lowerNode, res, n]
            have hnParents : n.parents = [yhat.id, target.id] := by
              simp [lowerNode, res, n]
            have hnOut : n.outShape = .scalar := by
              simp [lowerNode, res, n]
            have hn :
                n =
                  ({ id := id
                     parents := [yhat.id, target.id]
                     kind := .mseLoss
                     outShape := .scalar } : NN.IR.Node) := by
              simp [n, res, lowerNode]
            have hSameShape : yV.shape = tV.shape := by
              exact hy.trans ht.symm
            let yT : Tensor α yV.shape := yV.tensor
            let tT : Tensor α yV.shape := hSameShape.symm ▸ tV.tensor
            let diff : Tensor α yV.shape := Tensor.subSpec (α := α) yT tT
            let mean : α :=
              (Tensor.mulSpec (α := α) diff diff).sumSpec /
                (↑(NN.IR.Graph.meanDenom yV.shape) : α)
            let result : Spec.PackedTensor α := Spec.PackedTensor.mk (α := α) .scalar (Tensor.scalar mean)
            have hIREval :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := Spec.PackedTensor.mk (α := α) inShape x)
                    (vals := vals) (i := id) =
                  Except.ok result := by
              simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                NN.IR.Graph.normalizeNodeOutput, NN.IR.Graph.mseLossPackedTensor,
                hGetNode, hn, hSomeY, hSomeT, result, mean, diff, yT, tT, yV, tV,
                Bind.bind, Except.bind, Pure.pure, Except.pure]
            have hTypedEval :
                evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape)
                    (ss := ss₀) (out := .scalar) (.mseLoss yhat target) params vals =
                  Except.ok result := by
              simp only [evalNode, getValue?, hSomeY, hSomeT, Bind.bind, Except.bind]
              rw [dif_pos hSameShape]
              rfl
            exact hIREval.trans hTypedEval.symm
      -- Unfold both evaluators one step, then dispatch by cases on the shared `evalNode`.
      have hStart :
          NN.IR.Graph.denoteAllFrom (α := α) (g := cOut.graph)
              (payload := payloadOfParamStore (α := α) cOut.ps)
              (input := Spec.PackedTensor.mk (α := α) inShape x)
              (i := id) (vals := vals)
            =
          (do
            let v ← NN.IR.Graph.evalAt (α := α) (g := cOut.graph)
              (payload := payloadOfParamStore (α := α) cOut.ps)
              (input := Spec.PackedTensor.mk (α := α) inShape x)
              (vals := vals) (i := id)
            NN.IR.Graph.denoteAllFrom (α := α) (g := cOut.graph)
              (payload := payloadOfParamStore (α := α) cOut.ps)
              (input := Spec.PackedTensor.mk (α := α) inShape x)
              (i := id + 1) (vals := vals.push v)) := by
        -- Unfold `denoteAllFrom` once at the top-level; don't simp-recursively unfold the recursive
        -- call.
        rw [NN.IR.Graph.denoteAllFrom.eq_1]
        simp [hLt]
      -- Rewrite the goal using `hStart` and the one-step lemma `hStep`.
      have hStart' :
          NN.IR.Graph.denoteAllFrom (α := α) (g := cOut.graph)
              (payload := payloadOfParamStore (α := α) cOut.ps)
              (input := Spec.PackedTensor.mk (α := α) inShape x)
              (i := id) (vals := vals)
            =
          (do
            let vOut ← evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss :=
              ss₀) (out := mid₀)
              node params vals
            NN.IR.Graph.denoteAllFrom (α := α) (g := cOut.graph)
              (payload := payloadOfParamStore (α := α) cOut.ps)
              (input := Spec.PackedTensor.mk (α := α) inShape x)
              (i := id + 1) (vals := vals.push vOut)) := by
        -- Rewrite `denoteAllFrom` once, then replace `evalAt` with the already-verified `hStep`.
        -- Doing this explicitly avoids `simp` unfolding `Spec.PackedTensor.mk` too early, which can prevent
        -- matching on the `hStart` rewrite.
        rw [hStart]
        have hStep' :
            NN.IR.Graph.evalAt (α := α) (g := cOut.graph)
                (payload := payloadOfParamStore (α := α) cOut.ps)
                (input := ⟨inShape, x⟩)
                (vals := vals) (i := id)
              =
            evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀) (out := mid₀)
              node params vals := by
          simpa [Spec.PackedTensor.mk] using hStep
        simp [hStep']
      -- Now split on the result of `evalNode`.
      cases hEval : evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀)
        (out := mid₀)
          node params vals with
        | error e =>
            -- If the next DSL node fails, both evaluators stop immediately with the same error.
            -- First unfold lowering/evaluation one step so the goal is stated in terms of
            -- `cOut`.
            simp [lowerForwardLetChain, evalForwardLetChainVals]
            -- The simp step above rewrites the lowered graph to `cOut.graph`, but may unfold
            -- `Spec.PackedTensor.mk` to `⟨_,_⟩`. Normalize before rewriting with `hStart'`.
            have hStart'' :
                cOut.graph.denoteAllFrom
                    (payloadOfParamStore (α := α) cOut.ps) (⟨inShape, x⟩) id vals
                  =
                (do
                  let vOut ← evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape)
                    (ss := ss₀) (out := mid₀) node params vals
                  cOut.graph.denoteAllFrom
                    (payloadOfParamStore (α := α) cOut.ps) (⟨inShape, x⟩) (id + 1) (vals.push vOut)) := by
              simpa [Spec.PackedTensor.mk] using hStart'
            rw [hStart'']
            simp [hEval]
            rfl
        | ok vOut =>
          have hSize' : (vals.push vOut).size = c'.graph.nodes.size := by
            simp [c', hSize, Array.size_push]
          have hvOutShape : vOut.1 = mid₀ :=
            evalNode_ok_shape_of_hShapes (α := α) (paramShapes := paramShapes) (inShape := inShape)
              (ss := ss₀) (out := mid₀) node params vals hShapes (v := vOut) (by simp [hEval])
          have hShapes' : shapesOfVals (α := α) (vals.push vOut) = Ctx inShape (ss₀ ++ [mid₀]) := by
            calc
              shapesOfVals (α := α) (vals.push vOut)
                  = shapesOfVals (α := α) vals ++ [vOut.1] := shapesOfVals_push (α := α) (vals :=
                    vals) (v := vOut)
              _ = Ctx inShape ss₀ ++ [vOut.1] := by simp [hShapes]
              _ = (inShape :: ss₀) ++ [mid₀] := by simp [Ctx, hvOutShape]
              _ = Ctx inShape (ss₀ ++ [mid₀]) := by simp [Ctx, List.cons_append]
          have hIH :=
            ih (c := c') (vals := vals.push vOut) (hSize := hSize') (hShapes := hShapes')
          -- Rewrite the overall goal to the suffix goal (start at `id+1`), then discharge with IH.
          simp [lowerForwardLetChain, evalForwardLetChainVals]
          have hStart'' :
              cOut.graph.denoteAllFrom
                  (payloadOfParamStore (α := α) cOut.ps) (⟨inShape, x⟩) id vals
                =
              (do
                let vOut ← evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape)
                  (ss := ss₀) (out := mid₀) node params vals
                cOut.graph.denoteAllFrom
                  (payloadOfParamStore (α := α) cOut.ps) (⟨inShape, x⟩) (id + 1) (vals.push vOut)) := by
            simpa [Spec.PackedTensor.mk] using hStart'
          rw [hStart'']
          simp [hEval]
          -- now the goal is exactly the suffix IH (start index is `id+1 = c'.graph.nodes.size`).
          -- `Spec.PackedTensor.mk` is definitional `⟨_,_⟩`, but the pretty-printer may choose either form; normalize
          -- before applying the IH.
          simpa [c', id, Spec.PackedTensor.mk, Except.bind, Except.pure, bind, pure] using hIH
end Correctness

end NN.Verification.TorchLean.Proved
