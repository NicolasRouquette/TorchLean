/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved.Correctness.Eval.Return

/-!
# Compiled Forward Evaluation: End-to-End Correctness
-/

@[expose] public section

namespace NN.Verification.TorchLean.Proved

open _root_.Spec
open _root_.Spec.Tensor
open NN.IR

namespace Correctness

open NN.Verification.TorchLean

  /--
  **Main compiler correctness theorem (verified forward fragment).**

  In words: compiling a first-order forward program `p` into the verifier IR and then
  evaluating the compiled graph yields the same output as directly evaluating `p` with
  `evalForward`.
  -/
  theorem runForwardIR_compileVerifiedForward_eq_evalForward
    {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes : List Shape} {inShape outShape : Shape}
    (p : Program α paramShapes inShape outShape)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (x : Tensor α inShape) :
    runForwardIR (α := α) (inShape := inShape) (outShape := outShape)
        (c := compileVerifiedForward (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape
          := outShape) p params)
        x
      =
    evalForward (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape := outShape) p
      params x := by
    classical
    -- Evaluate `compileVerifiedForward` via the IR semantics, and rewrite it to the DSL evaluator.
    let inputVal : DVal α := DVal.mk (α := α) inShape x
    let inputNode : NN.IR.Node := { id := 0, parents := [], kind := .input, outShape := inShape }
    let c0 : NN.Verification.TorchLean.CompiledIR α :=
      { graph := { nodes := #[inputNode] }, ps := {}, inputId := 0, outputId := 0 }
    have hWF :
        (compileVerifiedForward (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape :=
          outShape) p params).graph.wellFormed = true :=
      compileVerifiedForward_wellFormed (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape
        := outShape) p params
    -- Unfold the compiled evaluator down to the IR `denoteAllFrom` suffix, then apply the correctness
    -- lemma.
    -- The input node is always id=0, so we start the suffix evaluation at `i=1` with
    -- `vals=[inputVal]`.
    have hDenote :
        (NN.IR.Graph.denoteAllFrom (α := α)
            (g := (compileVerifiedForward (α := α) (paramShapes := paramShapes) (inShape := inShape)
              (outShape := outShape) p params).graph)
            (payload := payloadOfParamStore (α := α)
              (compileVerifiedForward (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape :=
                outShape) p params).ps)
            (input := inputVal)
            (i := 1) (vals := #[inputVal]))
          =
        evalFGraphVals (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := []) (out :=
          outShape) p params #[inputVal] := by
      -- `compileVerifiedForward` is `compileFGraph` starting from `c0`, so apply the lemma at `c=c0` and
      -- `vals=[inputVal]`.
      have hSize0 : (#[inputVal] : Array (DVal α)).size = c0.graph.nodes.size := by
        simp [c0]
      have hShapes0 : shapesOfVals (α := α) (#[inputVal] : Array (DVal α)) = Ctx inShape [] := by
        simp [shapesOfVals, Ctx, inputVal, DVal.mk]
      simpa [compileVerifiedForward, c0, inputNode, inputVal, DVal.mk] using
        denoteAllFrom_compileFGraph_eq_evalFGraphVals (α := α) (paramShapes := paramShapes) (inShape
          := inShape) (ss := []) (out := outShape)
        (g := p) (params := params) (c := c0) (x := x) (vals := #[inputVal]) hSize0 hShapes0
    -- Unfold both front-end evaluators, then rewrite both sides to a shared `evalFGraphVals`
    -- computation.
    have hOutId :
        (compileVerifiedForward (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape :=
          outShape) p params).outputId
          =
        (outIdx (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := []) (out :=
          outShape) p).id := by
      simpa [compileVerifiedForward] using
        compileFGraph_outputId_eq_outIdx_id (α := α) (paramShapes := paramShapes) (inShape := inShape)
          (ss := []) (out := outShape) p params c0

    -- Rewrite `Graph.denoteAll` to start at `i=1` with the already-evaluated input.
    have hDenoteAll0 :
        NN.IR.Graph.denoteAll (α := α)
            (g := (compileVerifiedForward (α := α) (paramShapes := paramShapes) (inShape := inShape)
              (outShape := outShape) p params).graph)
            (payload := payloadOfParamStore (α := α)
              (compileVerifiedForward (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape :=
                outShape) p params).ps)
            (input := inputVal)
          =
        NN.IR.Graph.denoteAllFrom (α := α)
            (g := (compileVerifiedForward (α := α) (paramShapes := paramShapes) (inShape := inShape)
              (outShape := outShape) p params).graph)
            (payload := payloadOfParamStore (α := α)
              (compileVerifiedForward (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape :=
                outShape) p params).ps)
            (input := inputVal) (i := 1) (vals := #[inputVal]) := by
      -- `denoteAll` runs `denoteAllFrom` from `i=0` with `vals=[]`.
      simp (config := { zeta := false }) [NN.IR.Graph.denoteAll, hWF]
      -- Unfold one step at `i=0`: the `input` node deterministically yields `inputVal`.
      have h0 :
          (0 : Nat) <
            (compileVerifiedForward (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape :=
              outShape)
                p params).graph.nodes.size := by
        have h0c : (0 : Nat) < c0.graph.nodes.size := by
          simp [c0]
        have hLe :
            c0.graph.nodes.size ≤
              (compileVerifiedForward (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape :=
                outShape)
                  p params).graph.nodes.size := by
          simpa [compileVerifiedForward] using
            compileFGraph_nodesSize_le (α := α) (paramShapes := paramShapes) (inShape := inShape)
              (ss := []) (out := outShape) (g := p) (params := params) (c := c0)
        exact Nat.lt_of_lt_of_le h0c hLe
      have hGet0 :
          NN.IR.Graph.getNode
              (g := (compileVerifiedForward (α := α) (paramShapes := paramShapes) (inShape := inShape)
                (outShape := outShape)
                  p params).graph)
              0
            =
          pure inputNode := by
        have hi : (0 : Nat) < c0.graph.nodes.size := by
          simp [c0]
        -- `compileVerifiedForward` is `compileFGraph` starting from `c0`; indices < `c0`'s size are
        -- preserved.
        simpa [compileVerifiedForward, c0, inputNode, NN.IR.Graph.getNode, NN.IR.Graph.getNode?,
          BEq.beq] using
          compileFGraph_getNode_lt (α := α) (paramShapes := paramShapes) (inShape := inShape)
            (ss := []) (out := outShape) (g := p) (params := params) (c := c0) (i := 0) (hi := hi)
      have hEval0 :
          NN.IR.Graph.evalAt (α := α)
            (g := (compileVerifiedForward (α := α) (paramShapes := paramShapes) (inShape := inShape)
              (outShape := outShape)
                p params).graph)
            (payload := payloadOfParamStore (α := α)
              (compileVerifiedForward (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape :=
                outShape) p params).ps)
            (input := inputVal) (vals := #[]) (i := 0)
          =
          Except.ok inputVal := by
        -- `getNode 0` returns the input node, and the `.input` branch deterministically returns
        -- `inputVal`.
        simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput, hGet0, inputNode, inputVal, NN.IR.Graph.expectShape,
          DVal.shape, DVal.mk, DVal.tensor,
          Bind.bind, Pure.pure, Except.pure, Except.bind]
      rw [NN.IR.Graph.denoteAllFrom.eq_1, dif_pos h0]
      rw [hEval0]
      simp
      have hPushEq : (#[].push inputVal : Array (DVal α)) = #[inputVal] := by
        rfl
      exact congrArg
        (fun vals =>
          NN.IR.Graph.denoteAllFrom (α := α)
            (g := (compileVerifiedForward (α := α) (paramShapes := paramShapes) (inShape := inShape)
              (outShape := outShape) p params).graph)
            (payload := payloadOfParamStore (α := α)
              (compileVerifiedForward (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape :=
                outShape) p params).ps)
            (input := inputVal) (i := 1) (vals := vals))
        hPushEq

    -- Now both `runForwardIR` and `evalForward` can be expressed via `evalFGraphVals`.
    -- We finish by case-splitting on `evalFGraphVals` and simplifying the shared `if`/lookup logic.
    -- (The "out of bounds" / "shape mismatch" branches are unreachable under the well-typedness
    -- invariants we establish.)
    have hEvalForward :
        evalForward (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape := outShape)
          p params x
          =
        (do
          let vals' ←
            evalFGraphVals (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := []) (out
              := outShape) p params #[inputVal]
          let v : DVal α ← getDVal? vals' ((outIdx (α := α) (paramShapes := paramShapes)
            (inShape := inShape) (ss := []) (out := outShape) p).id)
          if h : v.shape = outShape then
            pure (h ▸ v.tensor)
          else
            throw s!"TorchLeanVerified: expected shape {repr outShape}, got {repr v.shape}") := by
      simpa [evalForward, inputVal] using
        (evalFGraph_eq_evalFGraphVals_outIdx (α := α) (paramShapes := paramShapes) (inShape :=
          inShape)
          (ss := []) (out := outShape) (g := p) (params := params) (vals := #[inputVal]))

    -- Rewrite the RHS, and unfold the compiled evaluator down to the same `evalFGraphVals`.
    rw [hEvalForward]
    rw [NN.Verification.TorchLean.runForwardIR, NN.IR.Graph.denote, hOutId]

    -- The remaining LHS still mentions IR `denoteAll`; rewrite it using the established equalities.
    have hDenoteAll0' :
        NN.IR.Graph.denoteAll (α := α)
            (g := (compileVerifiedForward (α := α) (paramShapes := paramShapes) (inShape := inShape)
              (outShape := outShape) p params).graph)
            (payload := payloadOfParamStore (α := α)
              (compileVerifiedForward (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape :=
                outShape) p params).ps)
            (input := DVal.mk (α := α) inShape x)
          =
        NN.IR.Graph.denoteAllFrom (α := α)
            (g := (compileVerifiedForward (α := α) (paramShapes := paramShapes) (inShape := inShape)
              (outShape := outShape) p params).graph)
            (payload := payloadOfParamStore (α := α)
              (compileVerifiedForward (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape :=
                outShape) p params).ps)
            (input := DVal.mk (α := α) inShape x) (i := 1) (vals := #[DVal.mk (α := α) inShape x]) :=
              by
      simpa [inputVal] using hDenoteAll0
    have hDenote' :
        NN.IR.Graph.denoteAllFrom (α := α)
            (g := (compileVerifiedForward (α := α) (paramShapes := paramShapes) (inShape := inShape)
              (outShape := outShape) p params).graph)
            (payload := payloadOfParamStore (α := α)
              (compileVerifiedForward (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape :=
                outShape) p params).ps)
            (input := DVal.mk (α := α) inShape x) (i := 1) (vals := #[DVal.mk (α := α) inShape x])
          =
        evalFGraphVals (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := []) (out :=
          outShape) p params
          #[DVal.mk (α := α) inShape x] := by
      simpa [inputVal, Except.bind, Except.pure, Pure.pure] using hDenote
    -- Rewrite `Graph.denoteAll` → `denoteAllFrom` → `evalFGraphVals` on the compiled side.
    rw [hDenoteAll0', hDenote']

    -- Now both sides bind the same `evalFGraphVals`; split on that result.
    cases hVals :
        evalFGraphVals (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := []) (out :=
          outShape) p params #[inputVal] with
    | error e =>
        -- Both sides are `Except.error e` because the first monadic bind fails.
        simp [Bind.bind, Except.bind]
    | ok vals' =>
        -- Establish that the output index is in-bounds and has the expected shape.
        have hShapes' :
            shapesOfVals (α := α) vals' = Ctx inShape (finalSs (α := α) (paramShapes := paramShapes)
              (inShape := inShape)
              (ss := []) (out := outShape) p) := by
          exact evalFGraphVals_shapes_of_hShapes (α := α) (paramShapes := paramShapes) (inShape :=
            inShape)
            (ss := []) (out := outShape) (g := p) (params := params) (vals := #[inputVal]) (vals' :=
              vals')
            (hShapes := by simp [shapesOfVals, Ctx, inputVal, DVal.mk]) (hOk := by simp [hVals])
        have hOutShape' :
            (vals'[(outIdx (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := []) (out
              := outShape) p).id]!).shape =
              outShape := by
          simpa [DVal.shape] using
            shape_of_vals_of_hShapes (α := α) (vals := vals')
              (idx := outIdx (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := [])
                (out := outShape) p)
              (hShapes := hShapes')
        have hOutLt :
            (outIdx (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := []) (out :=
              outShape) p).id < vals'.size := by
          have hLen : vals'.size = (Ctx inShape (finalSs (α := α) (paramShapes := paramShapes)
            (inShape := inShape)
                (ss := []) (out := outShape) p)).length := by
            have := congrArg List.length hShapes'
            simpa [shapesOfVals_length] using this
          have hIdx :
              (outIdx (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := []) (out :=
                outShape) p).id <
                (Ctx inShape (finalSs (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss
                  := []) (out := outShape) p)).length :=
            idx_id_lt_length (x := outIdx (α := α) (paramShapes := paramShapes) (inShape := inShape)
              (ss := []) (out := outShape) p)
          simpa [hLen] using hIdx
        -- In-bounds array lookup: `get?` returns `some (get!)`.
        have hOutSome :
            vals'[(outIdx (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := []) (out
              := outShape) p).id]? =
              some (vals'[(outIdx (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss :=
                []) (out := outShape) p).id]!) := by
          simp [getElem?_pos, hOutLt]
        have hCond :
            (vals'[(outIdx (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := []) (out
              := outShape) p).id]!).fst =
              outShape := by
          simpa [DVal.shape] using hOutShape'
        -- Rewrite the output lookup to the known in-bounds value and eliminate the dead
        -- shape-mismatch branch.
        cases hGet :
            vals'[(outIdx (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := []) (out
              := outShape) p).id]? with
        | none =>
            have hEq :
                (none : Option (DVal α)) =
                  some (vals'[(outIdx (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss :=
                    []) (out := outShape) p).id]!) := by
              simp [hGet] at hOutSome
            cases hEq
        | some out =>
            have hOutEq :
                out =
                  vals'[(outIdx (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := [])
                    (out := outShape) p).id]! := by
              simp [hGet] at hOutSome
              exact hOutSome
            subst out
            -- Both sides take the successful `get?` branch and the successful shape check under
            -- `hCond`.
            simp [getDVal?, hGet, hCond, DVal.shape, DVal.tensor, Bind.bind, Pure.pure, Except.pure,
              Except.bind]

end Correctness

end NN.Verification.TorchLean.Proved
