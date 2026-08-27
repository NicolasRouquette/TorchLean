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
    (params : TorchLean.TensorPack α paramShapes)
    (c : NN.Verification.TorchLean.LoweredIR α)
    (x : Tensor α inShape)
    (vals : Array (Spec.SomeTensor α))
    (hSize : vals.size = c.graph.nodes.size)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss) :
    (NN.IR.Graph.denoteAllFrom (α := α)
      (g := (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape)
        (ss := ss) (out := out) g params c).graph)
      (payload := payloadOfParamStore (α := α)
        (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape)
          (ss := ss) (out := out) g params c).ps)
      (input := Spec.SomeTensor.mk (α := α) inShape x)
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
          cOut.ps.convCfg.get? id = c'.ps.convCfg.get? id :=
        lowerForwardLetChain_ps_convCfg_get?_lt (α := α) (paramShapes := paramShapes)
          (inShape := inShape) (ss := ss₀ ++ [mid₀]) (out := out₀) (g := gNext)
          (params := params) (c := c') (hk := by
            simp [c', id, Array.size_push])
      have hLayerNorm :
          cOut.ps.layerNorm.get? id = c'.ps.layerNorm.get? id :=
        lowerForwardLetChain_ps_layerNorm_get?_lt (α := α) (paramShapes := paramShapes)
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
              (input := Spec.SomeTensor.mk (α := α) inShape x)
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
            let flat : NN.MLTheory.CROWN.Graph.FlatTensor α :=
              flatOfTensor (α := α) (s := mid₀) wf t
            have hnKind : n.kind = .const mid₀ := by
              simp [lowerNode, res, n]
            have hnParents : n.parents = #[] := by
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
                    NN.MLTheory.CROWN.Graph.FlatTensor α) := by
              simpa [flat, flatOfTensor] using hGet
            have hUF : unflattenSpec mid₀ t.flattenSpec = t := by
              simpa using (Spec.Tensor.flatten_unflatten_inverse_wf (α := α) (s := mid₀) (t := t))
            have hn :
                n = { id := id, parents := #[], kind := .const mid₀, outShape := mid₀ } := by
              simp [n, res, lowerNode]
            have hGetNodeConst :
                NN.IR.Graph.getNode (g := cOut.graph) id =
                  pure ({ id := id, parents := #[], kind := .const mid₀, outShape := mid₀ } :
                    NN.IR.Node) := by
              simp [hGetNode, hn]
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := Spec.SomeTensor.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok (Spec.SomeTensor.mk (α := α) mid₀ t) := by
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
                Except.ok (Spec.SomeTensor.mk (α := α) mid₀ t) := by
              rfl
            simp [hEvalNode]
            exact hEvalAt
        | paramConst wf p =>
            let : Shape.WellFormed mid₀ := wf
            -- Same as `const`, but the stored constant comes from `params`.
            let tp : Tensor α mid₀ := getParam (α := α) (paramShapes := paramShapes) params p
            let flat : NN.MLTheory.CROWN.Graph.FlatTensor α :=
              flatOfTensor (α := α) (s := mid₀) wf tp
            have hGet' : c'.ps.constVals.get? id = some flat := by
              simp [c', res, lowerNode, ps', flat, tp]
            have hGet : cOut.ps.constVals.get? id = some flat :=
              hConst.trans hGet'
            have hStoreConst :
                cOut.ps.constVals.get? id =
                  some ({ n := Spec.Shape.size mid₀, v := tp.flattenSpec } :
                    NN.MLTheory.CROWN.Graph.FlatTensor α) := by
              simpa [flat, flatOfTensor] using hGet
            have hUF : unflattenSpec mid₀ tp.flattenSpec = tp := by
              simpa using (Spec.Tensor.flatten_unflatten_inverse_wf (α := α) (s := mid₀) (t :=
                tp))
            have hnKind : n.kind = .const mid₀ := by
              simp [lowerNode, res, n]
            have hnParents : n.parents = #[] := by
              simp [lowerNode, res, n]
            have hnOut : n.outShape = mid₀ := by
              simp [lowerNode, res, n]
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := Spec.SomeTensor.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok (Spec.SomeTensor.mk (α := α) mid₀ tp) := by
              have hn :
                  n = { id := id, parents := #[], kind := .const mid₀, outShape := mid₀ } := by
                simp [n, res, lowerNode]
              have hGetNodeConst :
                  NN.IR.Graph.getNode (g := cOut.graph) id =
                    pure ({ id := id, parents := #[], kind := .const mid₀, outShape := mid₀ } :
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
                Except.ok (Spec.SomeTensor.mk (α := α) mid₀ tp) := by
              rfl
            simpa [hEvalNode] using hEvalAt
        | add a b =>
            simpa [evalNode, IRStep.BinaryElementwiseOp.denote] using
              IRStep.evalAt_binaryElementwise_of_getNode
                (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) .add a b cOut.graph
              (payloadOfParamStore (α := α) cOut.ps) (Spec.SomeTensor.mk (α := α) inShape x) vals id n
              hShapes hGetNode
                (by simp [lowerNode, res, n, IRStep.BinaryElementwiseOp.toOpKind])
                (by simp [lowerNode, res, n])
                (by simp [lowerNode, res, n])
        | sub a b =>
            simpa [evalNode, IRStep.BinaryElementwiseOp.denote] using
              IRStep.evalAt_binaryElementwise_of_getNode
                (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) .sub a b cOut.graph
              (payloadOfParamStore (α := α) cOut.ps) (Spec.SomeTensor.mk (α := α) inShape x) vals id n
              hShapes hGetNode
                (by simp [lowerNode, res, n, IRStep.BinaryElementwiseOp.toOpKind])
                (by simp [lowerNode, res, n])
                (by simp [lowerNode, res, n])
        | mulElem a b =>
            simpa [evalNode, IRStep.BinaryElementwiseOp.denote] using
              IRStep.evalAt_binaryElementwise_of_getNode
                (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) .mul a b cOut.graph
              (payloadOfParamStore (α := α) cOut.ps) (Spec.SomeTensor.mk (α := α) inShape x) vals id n
              hShapes hGetNode
                (by simp [lowerNode, res, n, IRStep.BinaryElementwiseOp.toOpKind])
                (by simp [lowerNode, res, n])
                (by simp [lowerNode, res, n])
        | relu xIdx =>
            simpa [evalNode, IRStep.UnaryElementwiseOp.denote] using
              IRStep.evalAt_unaryElementwise_of_getNode
                (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) .relu xIdx cOut.graph
                (payloadOfParamStore (α := α) cOut.ps) (Spec.SomeTensor.mk (α := α) inShape x) vals id n
                hShapes hGetNode
                (by simp [lowerNode, res, n, IRStep.UnaryElementwiseOp.toOpKind])
                (by simp [lowerNode, res, n]) (by simp [lowerNode, res, n])
        | exp xIdx =>
            simpa [evalNode, IRStep.UnaryElementwiseOp.denote] using
              IRStep.evalAt_unaryElementwise_of_getNode
                (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) .exp xIdx cOut.graph
                (payloadOfParamStore (α := α) cOut.ps) (Spec.SomeTensor.mk (α := α) inShape x) vals id n
                hShapes hGetNode
                (by simp [lowerNode, res, n, IRStep.UnaryElementwiseOp.toOpKind])
                (by simp [lowerNode, res, n]) (by simp [lowerNode, res, n])
        | log xIdx =>
            have hnKind : n.kind = .log := by
              simp [lowerNode, res, n]
            have hnParents : n.parents = #[xIdx.id] := by
              simp [lowerNode, res, n]
            have hnOut : n.outShape = mid₀ := by
              simp [lowerNode, res, n]
            have hn :
                n =
                  ({ id := id
                     parents := #[xIdx.id]
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
                    (input := Spec.SomeTensor.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                if Tensor.allSpec (α := α) (s := mid₀) (fun v => decide (0 < v)) tx then
                  Except.ok (Spec.SomeTensor.mk (α := α) mid₀ (Tensor.logSpec (α := α) tx))
                else
                  throw
                    "IR eval: log: input contains values <= 0 (or NaN); use `safe_log` if you want epsilon protection" := by
              simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                NN.IR.Graph.unaryParentId, NN.IR.unaryParent?,
                NN.IR.Graph.normalizeNodeOutput, hGetNode, hn,
                getElem?_eq_some_packedAt vals xIdx hShapes, hExpectX,
                throw, throwThe, MonadExceptOf.throw, Bind.bind, Except.bind, Pure.pure, Except.pure]
            simpa [evalNode, hGetValX, Spec.SomeTensor.mk, tx, Except.bind, Except.pure, bind, pure] using hEvalAt
        | inv xIdx =>
            simpa [evalNode, IRStep.UnaryElementwiseOp.denote] using
              IRStep.evalAt_unaryElementwise_of_getNode
                (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) .inv xIdx cOut.graph
                (payloadOfParamStore (α := α) cOut.ps) (Spec.SomeTensor.mk (α := α) inShape x) vals id n
                hShapes hGetNode
                (by simp [lowerNode, res, n, IRStep.UnaryElementwiseOp.toOpKind])
                (by simp [lowerNode, res, n]) (by simp [lowerNode, res, n])
        | @matmul leftShape rightShape _ op a b =>
            let ta : Tensor α leftShape := tensorAt vals a hShapes
            let tb : Tensor α rightShape := tensorAt vals b hShapes
            have hGetValA :
                getVal (α := α) (inShape := inShape) (ss := ss₀)
                    (s := leftShape) vals a =
                  Except.ok ta := by
              simpa [ta] using getVal_eq_ok_of_shapesOfVals_eq vals a hShapes
            have hGetValB :
                getVal (α := α) (inShape := inShape) (ss := ss₀)
                    (s := rightShape) vals b =
                  Except.ok tb := by
              simpa [tb] using getVal_eq_ok_of_shapesOfVals_eq vals b hShapes
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := Spec.SomeTensor.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (Spec.SomeTensor.mk (α := α) mid₀ (op.denote ta tb)) := by
              simpa [hGetValA, hGetValB, Bind.bind, Except.bind, Pure.pure, Except.pure] using
                (IRStep.evalAt_matmul_of_getNode
                  (α := α) (inShape := inShape) (ss := ss₀) op a b cOut.graph
                  (payloadOfParamStore (α := α) cOut.ps)
                  (Spec.SomeTensor.mk (α := α) inShape x) vals id n hShapes hGetNode
                  (by simp [lowerNode, res, n]) (by simp [lowerNode, res, n])
                  (by simp [lowerNode, res, n]))
            simpa [evalNode, hGetValA, hGetValB, Spec.SomeTensor.mk, ta, tb,
              Except.bind, Except.pure, bind, pure] using hEvalAt
        | reshape inS mid₀ h xIdx =>
            have hnKind : n.kind = .reshape inS mid₀ := by
              simp [lowerNode, res, n]
            have hnParents : n.parents = #[xIdx.id] := by
              simp [lowerNode, res, n]
            have hnOut : n.outShape = mid₀ := by
              simp [lowerNode, res, n]
            have hn :
                n =
                  ({ id := id
                     parents := #[xIdx.id]
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
                    (input := Spec.SomeTensor.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (Spec.SomeTensor.mk (α := α) mid₀
                    (Tensor.reshapeSpec (α := α) (s₁ := inS) (s₂ := mid₀) tx h)) := by
              simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                NN.IR.Graph.unaryParentId, NN.IR.unaryParent?,
                NN.IR.Graph.normalizeNodeOutput, hGetNode, hn,
                getElem?_eq_some_packedAt vals xIdx hShapes, hExpectX, h,
                Bind.bind, Except.bind, Pure.pure, Except.pure]
            simpa [evalNode, hGetValX, Spec.SomeTensor.mk, h, tx, Except.bind, Except.pure, bind, pure] using hEvalAt
        | transpose axis₁ axis₂ hOut xIdx =>
            rename_i s
            have hn :
                n =
                  ({ id := id
                     parents := #[xIdx.id]
                     kind := .transpose axis₁ axis₂
                     outShape := mid₀ } : NN.IR.Node) := by
              simp [n, res, lowerNode]
            let tx : Tensor α s := tensorAt vals xIdx hShapes
            have hGetValX :
                getVal (α := α) (inShape := inShape) (ss := ss₀) (s := s) vals xIdx =
                  Except.ok tx := by
              simpa [tx] using getVal_eq_ok_of_shapesOfVals_eq vals xIdx hShapes
            have hPacked :
                Spec.SomeTensor.ofTensor tx = packedAt vals xIdx hShapes := by
              simpa [tx] using ofTensor_tensorAt_eq_packedAt vals xIdx hShapes
            cases hPerm : OpContracts.transposePerm s.rank axis₁ axis₂ with
            | error e =>
                simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                  NN.IR.Graph.unaryParentId, NN.IR.unaryParent?,
                  hGetNode, hn,
                  getElem?_eq_some_packedAt vals xIdx hShapes, ← hPacked,
                  evalNode, hGetValX, hPerm,
                  Bind.bind, Except.bind, Pure.pure, Except.pure]
            | ok perm =>
                cases hEval : NN.IR.Graph.permuteSomeTensor (α := α)
                    (Spec.SomeTensor.mk (α := α) s tx) perm with
                | error e =>
                    simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                      NN.IR.Graph.unaryParentId, NN.IR.unaryParent?,
                      hGetNode, hn,
                      getElem?_eq_some_packedAt vals xIdx hShapes, ← hPacked,
                      evalNode, hGetValX, hPerm, hEval,
                      Bind.bind, Except.bind, Pure.pure, Except.pure]
                | ok y =>
                    cases hShape : NN.IR.Graph.expectShape (α := α) (expected := mid₀) y with
                    | error e =>
                        simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                          NN.IR.Graph.unaryParentId, NN.IR.unaryParent?,
                          hGetNode, hn,
                          getElem?_eq_some_packedAt vals xIdx hShapes, ← hPacked,
                          evalNode, hGetValX, hPerm, hEval, hShape,
                          Bind.bind, Except.bind, Pure.pure, Except.pure]
                    | ok ty =>
                        simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                          NN.IR.Graph.unaryParentId, NN.IR.unaryParent?,
                          NN.IR.Graph.normalizeNodeOutput, hGetNode, hn,
                          getElem?_eq_some_packedAt vals xIdx hShapes, ← hPacked,
                          evalNode, hGetValX, hPerm, hEval, hShape,
                          Bind.bind, Except.bind, Pure.pure, Except.pure]
        | softmax axis hAxis xIdx =>
            have hn :
                n =
                  ({ id := id
                     parents := #[xIdx.id]
                     kind := .softmax axis
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
                    (input := Spec.SomeTensor.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (Spec.SomeTensor.mk (α := α) mid₀
                    (@Activation.softmaxSpec α _ mid₀ axis hAxis tx)) := by
              cases hDynamic : Shape.axisInBounds? axis mid₀ with
              | none =>
                  have hSome := Shape.axisInBounds?_isSome
                    (axis := axis) (s := mid₀) (h := hAxis)
                  simp [hDynamic] at hSome
              | some h =>
                  simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                    NN.IR.Graph.unaryParentId, NN.IR.unaryParent?,
                    NN.IR.Graph.normalizeNodeOutput, hGetNode, hn,
                    getElem?_eq_some_packedAt vals xIdx hShapes, hDynamic,
                    hExpectX, Bind.bind, Except.bind, Pure.pure, Except.pure]
            simpa [evalNode, hGetValX, Spec.SomeTensor.mk, tx, Except.bind, Except.pure, bind, pure] using hEvalAt
        | layerNorm op xIdx =>
            let matrixShape : Shape := .dim op.rows (.dim op.width .scalar)
            have hn :
                n =
                  ({ id := id
                     parents := #[xIdx.id]
                     kind := .layernorm op.axis
                     outShape := mid₀ } : NN.IR.Node) := by
              simp [n, res, lowerNode]
            let tx : Tensor α mid₀ := tensorAt vals xIdx hShapes
            have hExpect :
                NN.IR.Graph.expectShape (α := α) (expected := mid₀)
                    (packedAt vals xIdx hShapes) =
                  Except.ok tx := by
              simpa [tx] using expectShape_packedAt_eq_ok vals xIdx hShapes
            have hGetVal :
                getVal (α := α) (inShape := inShape) (ss := ss₀)
                    (s := mid₀) vals xIdx =
                  Except.ok tx := by
              simpa [tx] using getVal_eq_ok_of_shapesOfVals_eq vals xIdx hShapes
            let xMatrix : Tensor α matrixShape :=
              Tensor.reshapeSpec (α := α) (s₁ := mid₀) (s₂ := matrixShape) tx op.size_eq
            let yMatrix : Tensor α matrixShape :=
              Spec.layerNorm (α := α) (seqLen := op.rows) (embedDim := op.width)
                (x := xMatrix)
                (gamma := Spec.fill (α := α) 1 (.dim op.width .scalar))
                (beta := Spec.fill (α := α) 0 (.dim op.width .scalar))
                (h_seq_pos := op.rows_pos) (h_embed_pos := op.width_pos)
            have hLN :
                NN.IR.Graph.layerNormWithoutAffine (α := α) op.rows op.width xMatrix =
                  Except.ok yMatrix := by
              simp [NN.IR.Graph.layerNormWithoutAffine, op.rows_pos, op.width_pos, yMatrix]
              rfl
            have hNoLayerNorm :
                (payloadOfParamStore (α := α) cOut.ps).layerNorm? id = none := by
              rw [IRStep.payloadOfParamStore_layerNorm?_eq, hLayerNorm]
              simp [c', res, lowerNode, ps']
            have hLayerNormMatrix :
                NN.IR.Graph.layerNormMatrix op.rows op.width xMatrix
                    (Spec.fill (α := α) 1 [op.width]) (Spec.fill (α := α) 0 [op.width])
                    Numbers.normalizationEpsilon = .ok yMatrix := by
              simpa [NN.IR.Graph.layerNormWithoutAffine, NN.IR.Graph.layerNormMatrix] using hLN
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := Spec.SomeTensor.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (Spec.SomeTensor.mk (α := α) mid₀
                    (Tensor.reshapeSpec (α := α) (s₁ := matrixShape) (s₂ := mid₀)
                      yMatrix op.size_eq.symm)) := by
              simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                NN.IR.Graph.unaryParentId, NN.IR.unaryParent?,
                NN.IR.Graph.normalizeNodeOutput, hGetNode, hn,
                    getElem?_eq_some_packedAt vals xIdx hShapes, hExpect, op.matrixDims,
                op.size_eq, matrixShape, xMatrix, hNoLayerNorm,
                NN.IR.Graph.resolveLayerNormAffine, hLayerNormMatrix,
                Bind.bind, Except.bind, Pure.pure, Except.pure]
            simpa [evalNode, hGetVal, matrixShape, xMatrix, yMatrix,
              Spec.SomeTensor.mk, Except.bind, Except.pure, bind, pure] using hEvalAt
        | linear inDim outDim w b xIdx =>
            let wT : Tensor α [outDim, inDim] :=
              getParam (α := α) (paramShapes := paramShapes) params w
            let bT : Tensor α [outDim] :=
              getParam (α := α) (paramShapes := paramShapes) params b
            have hxF : (packedAt vals xIdx hShapes).shape = .dim inDim .scalar :=
              packedAt_shape vals xIdx hShapes
            have hLin' : cOut.ps.linearWB[n.id]? = c'.ps.linearWB[n.id]? := by
              simpa [hnId] using hLin
            have hn :
                n =
                  ({ id := id
                     parents := #[xIdx.id]
                     kind := .linear
                     outShape := .dim outDim .scalar } : NN.IR.Node) := by
              simp [n, res, lowerNode]
            have hGetNodeLinear :
                NN.IR.Graph.getNode (g := cOut.graph) id =
                  pure
                    ({ id := id
                       parents := #[xIdx.id]
                       kind := .linear
                       outShape := .dim outDim .scalar } : NN.IR.Node) := by
              simp [hGetNode, hn]
            let xT : Tensor α [inDim] :=
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
                    (input := Spec.SomeTensor.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (Spec.SomeTensor.mk (α := α) (.dim outDim .scalar)
                      (Tensor.addSpec (α := α)
                        (Tensor.matVecMulSpec (α := α) (m := outDim) (n := inDim) wT xT) bT)) := by
              simpa [Spec.matVecMulSpec] using
                IRStep.evalAt_linear_from_paramStore_of_getNode
                  (α := α) (g := cOut.graph) (ps := cOut.ps) (i := id) (id := id)
                  (parentId := xIdx.id) (outDim := outDim) (inDim := inDim)
                  (inputShape := inShape) (input := x) (vals := vals)
                  (weight := wT) (bias := bT) (parent := packedAt vals xIdx hShapes) (x := xT)
                  hGetNodeLinear (getElem?_eq_some_packedAt vals xIdx hShapes)
                  hExpectIn hLinearStore
            simpa [evalNode, hGetVal, Spec.SomeTensor.shape, Spec.SomeTensor.tensor, Spec.SomeTensor.mk, hxF, xT, wT, bT] using
              hEvalAt
        | conv inC outC kernelShape stride padding inSpatial hIn hKernel hStride hInfer
            kernel bias xIdx =>
            rename_i d
            let inputShape' : Shape := Shape.ofList (inC :: inSpatial.toList)
            let outputShape' : Shape := Shape.ofList
              (outC :: (Spec.convOutSpatial inSpatial kernelShape stride padding).toList)
            let config : ConvConfig :=
              { spatialRank := d
                kernel := kernelShape
                stride := stride
                padding := padding
                dilation := Spec.fill 1 [d]
                paddingAfter := padding
                groups := 1
                channelAxis := 0
                inChannels := inC
                outChannels := outC }
            let kT : Tensor α (Shape.ofList (outC :: inC :: kernelShape.toList)) :=
              getParam (α := α) (paramShapes := paramShapes) params kernel
            let bT : Tensor α [outC] :=
              getParam (α := α) (paramShapes := paramShapes) params bias
            let spec : Spec.ConvSpec d inC outC kernelShape stride padding α :=
              { kernel := kT, bias := bT }
            let cfg : NN.IR.ConvParams α :=
              { spatialRank := d
                inChannels := inC
                outChannels := outC
                kernel := kernelShape
                stride := stride
                padding := padding
                dilation := Spec.fill 1 [d]
                paddingAfter := padding
                groups := 1
                inputSpatial := inSpatial
                inChannelsNonzero := hIn
                kernelNonzero := hKernel
                strideNonzero := hStride
                spec := spec }
            have hxF : (packedAt vals xIdx hShapes).shape = inputShape' := by
              simp [inputShape']
            have hn :
                n =
                  ({ id := id
                     parents := #[xIdx.id]
                     kind := .conv config
                     outShape := outputShape' } : NN.IR.Node) := by
              simp [n, res, lowerNode, config, outputShape', Spec.fill]
            have hGetNodeConv :
                NN.IR.Graph.getNode (g := cOut.graph) id =
                  pure
                    ({ id := id
                       parents := #[xIdx.id]
                       kind := .conv config
                       outShape := cfg.outputShape .scalar } : NN.IR.Node) := by
              have hOutput : cfg.outputShape .scalar = outputShape' := by
                simp only [cfg, ConvParams.outputShape, Shape.concat, Shape.ofList]
                rw [Spec.convOutSpatialDilated_one_symmetric]
              simpa [hOutput] using hGetNode.trans (congrArg pure hn)
            let xT : Tensor α inputShape' := tensorAt vals xIdx hShapes
            have hExpectIn :
                NN.IR.Graph.expectShape (α := α) (expected := cfg.inputShape .scalar)
                    (packedAt vals xIdx hShapes) = Except.ok xT := by
              simpa [cfg, ConvParams.inputShape, Shape.concat, inputShape', xT] using
                expectShape_packedAt_eq_ok vals xIdx hShapes
            have hGetVal :
                getVal (α := α) (inShape := inShape) (ss := ss₀)
                    (s := inputShape') vals xIdx = Except.ok xT := by
              simpa [xT] using getVal_eq_ok_of_shapesOfVals_eq vals xIdx hShapes
            have hConvStore : cOut.ps.convCfg.get? id = some cfg := by
              rw [hConv]
              simp [c', res, lowerNode, ps', cfg, spec, kT, bT]
            have hConfig : cfg.matchesConfig config = true := by
              simp [cfg, config, ConvParams.matchesConfig]
            have hInfer' :
                OpContracts.inferConvConfigOutShape "conv" config
                    (packedAt vals xIdx hShapes).shape =
                  .ok (cfg.outputShape .scalar) := by
              have hOutput : cfg.outputShape .scalar = outputShape' := by
                simp only [cfg, ConvParams.outputShape, Shape.concat, Shape.ofList]
                rw [Spec.convOutSpatialDilated_one_symmetric]
              simpa [OpContracts.inferConvOutShape, config, hxF, hOutput] using hInfer
            have hLeading :
                Shape.ofList
                    ((packedAt vals xIdx hShapes).shape.toList.take config.channelAxis) =
                  .scalar := by
              rfl
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := Spec.SomeTensor.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                .ok (Spec.SomeTensor.ofTensor
                  (Tensor.mapEach .scalar
                    (Spec.groupedConvSpec (α := α) (stride := cfg.stride)
                      (dilation := cfg.dilation) (paddingBefore := cfg.padding)
                      (paddingAfter := cfg.paddingAfter) cfg.groups cfg.spec.kernel cfg.spec.bias)
                    xT)) := by
              simpa using
                IRStep.evalAt_conv_from_paramStore_of_getNode
                  (α := α) (g := cOut.graph) (ps := cOut.ps) (i := id) (id := id)
                  (parentId := xIdx.id) (params := cfg) (config := config)
                  (leading := .scalar) (inputShape := inShape) (input := x)
                  (vals := vals) (parent := packedAt vals xIdx hShapes) (x := xT)
                  hGetNodeConv (getElem?_eq_some_packedAt vals xIdx hShapes)
                  hExpectIn hConvStore hConfig hInfer' hLeading
            have hPacked :
                Spec.SomeTensor.ofTensor
                    (Tensor.mapEach .scalar
                      (Spec.groupedConvSpec (α := α) (stride := cfg.stride)
                        (dilation := cfg.dilation) (paddingBefore := cfg.padding)
                        (paddingAfter := cfg.paddingAfter) cfg.groups cfg.spec.kernel cfg.spec.bias)
                      xT) =
              Spec.SomeTensor.mk (α := α) outputShape'
                    (Spec.convSpec (α := α) (layer := spec) (input := xT)) := by
              let generalConv :=
                Spec.groupedConvSpec (α := α) (stride := stride)
                  (dilation := Spec.fill 1 [d]) (paddingBefore := padding)
                  (paddingAfter := padding) 1 kT bT xT
              let denseConv := Spec.convSpec (α := α) (layer := spec) (input := xT)
              let shapeEq :=
                congrArg (fun spatial => Shape.ofList (outC :: spatial.toList))
                  (Spec.convOutSpatialDilated_one_symmetric inSpatial kernelShape stride padding)
              have hCast : Tensor.castShape generalConv shapeEq = denseConv := by
                simpa [generalConv, denseConv, shapeEq, spec] using
                  Spec.castShape_groupedConvSpec_one_symmetric
                    (α := α) (weights := kT) (bias := bT) (input := xT)
              have hErased :
                  Spec.SomeTensor.ofTensor generalConv = Spec.SomeTensor.ofTensor denseConv := by
                calc
                  Spec.SomeTensor.ofTensor generalConv =
                      Spec.SomeTensor.ofTensor (Tensor.castShape generalConv shapeEq) :=
                    (Spec.SomeTensor.ofTensor_castShape generalConv shapeEq).symm
                  _ = Spec.SomeTensor.ofTensor denseConv :=
                    congrArg Spec.SomeTensor.ofTensor hCast
              simpa [generalConv, denseConv, cfg, spec, outputShape', Tensor.mapEach,
                Shape.concat] using hErased
            have hEvalAt' := hEvalAt.trans (congrArg Except.ok hPacked)
            simpa [evalNode, hGetVal, Spec.SomeTensor.shape, Spec.SomeTensor.tensor,
              Spec.SomeTensor.mk, hxF, xT, kT, bT, spec, inputShape', outputShape'] using hEvalAt'
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
            have hnParents : n.parents = #[yhat.id, target.id] := by
              simp [lowerNode, res, n]
            have hnOut : n.outShape = .scalar := by
              simp [lowerNode, res, n]
            have hn :
                n =
                  ({ id := id
                     parents := #[yhat.id, target.id]
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
            let result : Spec.SomeTensor α := Spec.SomeTensor.mk (α := α) .scalar (Tensor.scalar mean)
            have hIREval :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := Spec.SomeTensor.mk (α := α) inShape x)
                    (vals := vals) (i := id) =
                  Except.ok result := by
              simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                NN.IR.Graph.binaryParentIds, NN.IR.binaryParents?,
                NN.IR.Graph.normalizeNodeOutput, NN.IR.Graph.mseLossSomeTensor,
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
              (input := Spec.SomeTensor.mk (α := α) inShape x)
              (i := id) (vals := vals)
            =
          (do
            let v ← NN.IR.Graph.evalAt (α := α) (g := cOut.graph)
              (payload := payloadOfParamStore (α := α) cOut.ps)
              (input := Spec.SomeTensor.mk (α := α) inShape x)
              (vals := vals) (i := id)
            NN.IR.Graph.denoteAllFrom (α := α) (g := cOut.graph)
              (payload := payloadOfParamStore (α := α) cOut.ps)
              (input := Spec.SomeTensor.mk (α := α) inShape x)
              (i := id + 1) (vals := vals.push v)) := by
        -- Unfold `denoteAllFrom` once at the top-level; don't simp-recursively unfold the recursive
        -- call.
        rw [NN.IR.Graph.denoteAllFrom.eq_1]
        simp [hLt]
      -- Rewrite the goal using `hStart` and the one-step lemma `hStep`.
      have hStart' :
          NN.IR.Graph.denoteAllFrom (α := α) (g := cOut.graph)
              (payload := payloadOfParamStore (α := α) cOut.ps)
              (input := Spec.SomeTensor.mk (α := α) inShape x)
              (i := id) (vals := vals)
            =
          (do
            let vOut ← evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss :=
              ss₀) (out := mid₀)
              node params vals
            NN.IR.Graph.denoteAllFrom (α := α) (g := cOut.graph)
              (payload := payloadOfParamStore (α := α) cOut.ps)
              (input := Spec.SomeTensor.mk (α := α) inShape x)
              (i := id + 1) (vals := vals.push vOut)) := by
        -- Rewrite `denoteAllFrom` once, then replace `evalAt` with the already-verified `hStep`.
        -- Doing this explicitly avoids `simp` unfolding `Spec.SomeTensor.mk` too early, which can prevent
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
          simpa [Spec.SomeTensor.mk] using hStep
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
            -- `Spec.SomeTensor.mk` to `⟨_,_⟩`. Normalize before rewriting with `hStart'`.
            have hStart'' :
                cOut.graph.denoteAllFrom
                    (payloadOfParamStore (α := α) cOut.ps) (⟨inShape, x⟩) id vals
                  =
                (do
                  let vOut ← evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape)
                    (ss := ss₀) (out := mid₀) node params vals
                  cOut.graph.denoteAllFrom
                    (payloadOfParamStore (α := α) cOut.ps) (⟨inShape, x⟩) (id + 1) (vals.push vOut)) := by
              simpa [Spec.SomeTensor.mk] using hStart'
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
            simpa [Spec.SomeTensor.mk] using hStart'
          rw [hStart'']
          simp [hEval]
          -- now the goal is exactly the suffix IH (start index is `id+1 = c'.graph.nodes.size`).
          -- `Spec.SomeTensor.mk` is definitional `⟨_,_⟩`, but the pretty-printer may choose either form; normalize
          -- before applying the IH.
          simpa [c', id, Spec.SomeTensor.mk, Except.bind, Except.pure, bind, pure] using hIH
end Correctness

end NN.Verification.TorchLean.Proved
