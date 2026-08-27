/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Lowering.Common

/-!
# Shape and Concatenation IR Lowering

Checked lowering for permutations, reshaping, flattening, concatenation, and transpose.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace IRExec

open Spec
open Tensor
open Proofs.Autograd.Algebra
open NN.IR

namespace Internal

/-- Checked lowering for permutations, reshaping, flattening, concatenation, and transpose. -/
@[simp] def lowerShape {α : Type} [Context α] [shapeDecidable : DecidableEq Shape]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (kind : OpKind) :
    NodeLoweringResult ctx := do
  let g := ctx.graph
  let payload := ctx.payload
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : _root_.TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match kind with
  | .permute perm =>
      match unaryParent? n.parents with
      | some pId =>
          let pNode ← g.getNode pId
          let sIn := pNode.outShape
          let ip ← parentIdx pId sIn
          match Spec.Shape.permute? sIn perm.toList with
          | none =>
              throw s!"IRExec: node {i}: invalid permutation {repr perm} for shape {repr sIn}"
          | some expected =>
              let swaps ← NN.IR.Graph.swapDepthsForPerm perm (Spec.Shape.rank sIn)
              let sFinal : Shape := swapShapeBySwaps sIn swaps
              if hFinal : sFinal = expected then
                if hOut : expected = τ then
                  let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
                    let x := getIdx (α := α) (xs := ctx) ip
                    let y : Tensor α sFinal := applySwapsTensor (α := α) (s := sIn) (swaps :=
                      swaps) x
                    let yExpected : Tensor α expected := Tensor.castShape y hFinal
                    Tensor.castShape yExpected hOut
                  pure <| fwd forward
                else
                  throw <|
                    s!"IRExec: node {i}: permute outShape mismatch: " ++
                      s!"expected={repr expected}, declared={repr τ}"
              else
                throw <|
                  s!"IRExec: node {i}: permute shape mismatch: computed={repr sFinal}, " ++
                    s!"expected={repr expected} ({n.summary})"
      | _ => throw s!"IRExec: node {i}: permute expects 1 parent ({n.summary})"
  | .reshape inS outS =>
      match unaryParent? n.parents with
      | some pId =>
          let ip ← parentIdx pId inS
          if hNumel : Spec.Shape.size inS = Spec.Shape.size outS then
            if hOut : outS = τ then
              let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
                let x := getIdx (α := α) (xs := ctx) ip
                hOut ▸ Tensor.reshapeSpec (α := α) (s₁ := inS) (s₂ := outS) x hNumel
              pure <| fwd forward
            else
              throw <|
                s!"IRExec: node {i}: reshape outShape mismatch: kind={repr outS}, " ++
                  s!"declared={repr τ}"
          else
            throw <|
              s!"IRExec: node {i}: reshape numel mismatch: {Spec.Shape.size inS} vs " ++
                s!"{Spec.Shape.size outS}"
      | _ => throw s!"IRExec: node {i}: reshape expects 1 parent ({n.summary})"
  | .flatten s =>
      match unaryParent? n.parents with
      | some pId =>
          let ip ← parentIdx pId s
          let expected : Shape := .dim (Spec.Shape.size s) .scalar
          if hOut : expected = τ then
            let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
              let x := getIdx (α := α) (xs := ctx) ip
              let y : Tensor α expected := Tensor.flattenSpec (α := α) (s := s) x
              hOut ▸ y
            pure <| fwd forward
          else
            throw <|
              s!"IRExec: node {i}: flatten outShape mismatch: " ++
                s!"expected={repr expected}, declared={repr τ} ({n.summary})"
      | _ => throw s!"IRExec: node {i}: flatten expects 1 parent ({n.summary})"
  | .concat axis =>
      let parents := n.parents
      if parents.size < 2 then
        throw s!"IRExec: node {i}: concat expects at least 2 parents"

      let parentShapes : Array Shape ← parents.mapM (fun pid => do
        let pNode ← g.getNode pid
        pure pNode.outShape)
      let expected ←
        match OpContracts.inferConcatOutShape axis parentShapes with
        | .ok s => pure s
        | .error msg => throw s!"IRExec: node {i}: {msg} ({n.summary})"
      if expected != τ then
        throw <|
          s!"IRExec: node {i}: concat outShape mismatch: expected={repr expected}, " ++
            s!"declared={repr τ} ({n.summary})"

      if axis = 0 then
        match hτ : τ with
        | .dim nOut rest =>
            -- Precompute typed indices for each parent and check that tails match.
            let infos : Array (Sigma fun nP => Idx Γ (.dim nP rest)) ←
              parents.mapM (fun pid => do
                let pNode ← g.getNode pid
                match pNode.outShape with
                | .dim nP restP =>
                    if hRest : restP = rest then
                      let ip ← parentIdx pid (.dim nP rest)
                      pure ⟨nP, ip⟩
                    else
                      throw <|
                        s!"IRExec: node {i}: concat axis=0 tail mismatch: {repr restP} vs " ++
                          s!"{repr rest}"
                | _ =>
                    throw <|
                      s!"IRExec: node {i}: concat axis=0 expects rank≥1 parents, got " ++
                        s!"{repr pNode.outShape}")
            let nSum : Nat := infos.foldl (fun acc info => acc + info.1) 0
            if hSum : nSum = nOut then
              let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
                let outSigma : Sigma fun n => Tensor α (.dim n rest) :=
                  concatLeadingAxisFromInfos (α := α) (Γ := Γ) (rest := rest) ctx infos
                have houtSigma :
                    outSigma =
                      concatLeadingAxisFromInfos (α := α) (Γ := Γ) (rest := rest) ctx
                        infos := rfl
                let nSum' : Nat := outSigma.1
                let tSum : Tensor α (.dim nSum' rest) := outSigma.2
                have hn : nSum' = nSum := by
                  -- `nSum'` is the first component of the same fold used to compute `nSum`.
                  change outSigma.1 = nSum
                  rw [houtSigma]
                  simpa [nSum] using
                    (concatLeadingAxisFromInfos_size_eq_sum (α := α) (Γ := Γ) (rest :=
                      rest) ctx infos)
                let tSum' : Tensor α (.dim nSum rest) :=
                  Tensor.castShape tSum (by simp [hn])
                have hOutShape : Shape.dim nSum rest = τ := by
                  have hDim : Shape.dim nSum rest = Shape.dim nOut rest := by
                    simpa using congrArg (fun k => Shape.dim k rest) hSum
                  exact hDim.trans hτ.symm
                Tensor.castShape tSum' hOutShape
              pure <| fwd forward
            else
              throw <|
                s!"IRExec: node {i}: concat out dim mismatch: declared {nOut}, computed " ++
                  s!"{nSum} ({n.summary})"
        | _ =>
            throw s!"IRExec: node {i}: concat axis=0 expects rank≥1 outShape, got {repr τ}"
      else
        -- General axis concat: permute `axis` to the front (axis 0), concatenate along axis 0,
        -- then permute back.
        let permFront ←
          match OpContracts.permMoveAxisToFront axis τ with
          | .ok perm => pure perm
          | .error msg => throw s!"IRExec: node {i}: concat: {msg}"
        let permBack ←
          match OpContracts.inversePerm permFront with
          | .ok perm => pure perm
          | .error msg => throw s!"IRExec: node {i}: concat: {msg}"
        match Spec.Shape.permute? τ permFront.toList with
        | none =>
            throw <|
              s!"IRExec: node {i}: concat: invalid permutation {repr permFront} for " ++
                s!"shape {repr τ}"
        | some outFrontExpected =>
            match hOutFrontExpected : outFrontExpected with
            | .dim nOutFront restFront =>
                let swapsFront ← NN.IR.Graph.swapDepthsForPerm permFront (Spec.Shape.rank τ)
                let τFrontFinal : Shape := swapShapeBySwaps τ swapsFront
                if hOutFrontFinal : τFrontFinal = outFrontExpected then
                  let swapsBack ← NN.IR.Graph.swapDepthsForPerm permBack (Spec.Shape.rank
                    outFrontExpected)
                  let τBackFinal : Shape := swapShapeBySwaps outFrontExpected swapsBack
                  if hOutBackFinal : τBackFinal = τ then
                    let getters :
                        Array (Sigma fun nP => _root_.TorchLean.TensorPack α Γ → Tensor α (.dim nP
                          restFront)) ←
                      parents.mapM (fun pid => do
                        let pNode ← g.getNode pid
                        let sIn := pNode.outShape
                        let ip ← parentIdx pid sIn
                        match Spec.Shape.permute? sIn permFront.toList with
                        | none =>
                            throw <|
                              s!"IRExec: node {i}: concat: invalid permutation " ++
                                s!"{repr permFront} for parent shape {repr sIn}"
                        | some (.dim nP restP) =>
                            let sFrontExpected : Shape := .dim nP restP
                            if hRest : restP = restFront then
                              let sFrontFinal : Shape := swapShapeBySwaps sIn swapsFront
                              if hFinal : sFrontFinal = sFrontExpected then
                                let getT := fun ctx : _root_.TorchLean.TensorPack α Γ =>
                                  let x : Tensor α sIn := getIdx (α := α) (xs := ctx) ip
                                  let yFinal : Tensor α sFrontFinal :=
                                    applySwapsTensor (α := α) (s := sIn) (swaps := swapsFront) x
                                  let yExpected : Tensor α sFrontExpected :=
                                    Tensor.castShape yFinal hFinal
                                  -- `yExpected` has tail `restP`; cast to the shared
                                  -- `restFront`.
                                  let yExpected' : Tensor α (.dim nP restP) := by
                                    simpa [sFrontExpected] using yExpected
                                  (by
                                    simpa [hRest] using yExpected' : Tensor α (.dim nP
                                      restFront))
                                pure ⟨nP, getT⟩
                              else
                                throw <|
                                  s!"IRExec: node {i}: concat permute shape mismatch: " ++
                                  s!"computed={repr sFrontFinal}, " ++
                                    s!"expected={repr sFrontExpected} ({n.summary})"
                            else
                              throw <|
                                s!"IRExec: node {i}: concat: permuted tail mismatch: " ++
                                  s!"{repr restP} vs {repr restFront} ({n.summary})"
                        | some _ =>
                            throw <|
                              s!"IRExec: node {i}: concat expects rank≥1 parents, got " ++
                                s!"{repr sIn}"
                      )

                    let nSum : Nat := getters.foldl (fun acc info => acc + info.1) 0
                    if hSum : nSum = nOutFront then
                      let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
                        let empty : Tensor α (.dim 0 restFront) :=
                          Spec.fill (α := α) 0 (.dim 0 restFront)
                        let outSigma :
                            Sigma fun n => Tensor α (.dim n restFront) :=
                          getters.foldl
                            (fun acc nxt =>
                              match acc, nxt with
                              | ⟨n1, t1⟩, ⟨n2, get2⟩ =>
                                  let t2 : Tensor α (.dim n2 restFront) := get2 ctx
                                  ⟨n1 + n2, Tensor.concatAxisSpec .scalar (α := α) (n := n1) (m := n2)
                                    (suffix := restFront) t1 t2⟩)
                            ⟨0, empty⟩
                        let tSum : Tensor α (.dim nSum restFront) :=
                          Tensor.castShape outSigma.2 (by
                            -- The fold's nat component is the sum of the input sizes.
                            have hFold :
                                outSigma.1 =
                                  getters.foldl (fun acc info => acc + info.1) 0 := by
                              -- General lemma: the `.1` component of this fold is just a nat
                              -- fold.
                              have hGen :
                                  ∀ (xs : List (Sigma fun nP => _root_.TorchLean.TensorPack α Γ →
                                    Tensor α (.dim nP restFront)))
                                    (n0 : Nat) (t0 : Tensor α (.dim n0 restFront)),
                                    (xs.foldl
                                        (fun acc nxt =>
                                          match acc, nxt with
                                          | ⟨n1, t1⟩, ⟨n2, get2⟩ =>
                                              let _t2 : Tensor α (.dim n2 restFront) := get2 ctx
                                              ⟨n1 + n2, Tensor.concatAxisSpec .scalar (α := α) (n :=
                                                n1) (m := n2) (suffix := restFront) t1 _t2⟩)
                                        (⟨n0, t0⟩ : Sigma fun n => Tensor α (.dim n
                                          restFront))).1 =
                                      xs.foldl (fun acc info => acc + info.1) n0 := by
                                intro xs n0 t0
                                induction xs generalizing n0 t0 with
                                | nil =>
                                    simp
                                | cons x xs ih =>
                                    -- Unfold both folds one step and apply the IH to the
                                    -- updated accumulator.
                                    simp [List.foldl] at *
                                    -- After unfolding, the goal is exactly the IH instantiated
                                    -- at `n0 + x.1`.
                                    simpa using
                                      (ih (n0 := n0 + x.1)
                                        (t0 := Tensor.concatAxisSpec .scalar (α := α) (n := n0) (m :=
                                          x.1)
                                          (suffix := restFront) t0 (x.2 ctx)))
                              simp only [outSigma]
                              rw [← Array.foldl_toList, ← Array.foldl_toList]
                              simpa [outSigma] using (hGen getters.toList 0 empty)
                            have hn : outSigma.1 = nSum := by
                              simpa [nSum] using hFold
                            simp [hn])
                        have hOutFront : Shape.dim nSum restFront = outFrontExpected := by
                          have hDim : Shape.dim nSum restFront = Shape.dim nOutFront restFront
                            := by
                            simpa using congrArg (fun k => Shape.dim k restFront) hSum
                          simpa [hOutFrontExpected] using hDim
                        let tFront : Tensor α outFrontExpected := Tensor.castShape tSum
                          hOutFront
                        let tBack : Tensor α τBackFinal :=
                          applySwapsTensor (α := α) (s := outFrontExpected) (swaps := swapsBack)
                            tFront
                        Tensor.castShape tBack hOutBackFinal
                      pure <| fwd forward
                    else
                      throw <|
                        s!"IRExec: node {i}: concat out dim mismatch: declared {nOutFront}, " ++
                          s!"computed {nSum} ({n.summary})"
                  else
                    throw <|
                      s!"IRExec: node {i}: concat permute-back shape mismatch: " ++
                        s!"computed={repr τBackFinal}, expected={repr τ} ({n.summary})"
                else
                  throw <|
                    s!"IRExec: node {i}: concat permute-to-front shape mismatch: " ++
                    s!"computed={repr τFrontFinal}, expected={repr outFrontExpected} " ++
                      s!"({n.summary})"
            | _ =>
                throw s!"IRExec: node {i}: concat expects rank≥1 outShape, got {repr τ}"
  | .transpose axis₁ axis₂ =>
      match unaryParent? n.parents with
      | some pId =>
          let pNode ← g.getNode pId
          let sIn := pNode.outShape
          let ip ← parentIdx pId sIn
          let perm ← OpContracts.transposePerm sIn.rank axis₁ axis₂
          let expected ←
            match Shape.permute? sIn perm.toList with
            | some expected => pure expected
            | none => throw s!"IRExec: node {i}: invalid transpose axes ({n.summary})"
          let swaps ← NN.IR.Graph.swapDepthsForPerm perm sIn.rank
          let computed : Shape := swapShapeBySwaps sIn swaps
          if hComputed : computed = expected then
            if hOut : expected = τ then
              let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
                let x := getIdx (α := α) (xs := ctx) ip
                let y : Tensor α computed := applySwapsTensor (α := α) (s := sIn)
                  (swaps := swaps) x
                Tensor.castShape (Tensor.castShape y hComputed) hOut
              pure <| fwd forward
            else
              throw s!"IRExec: node {i}: transpose outShape mismatch ({n.summary})"
          else
            throw s!"IRExec: node {i}: transpose lowering mismatch ({n.summary})"
      | _ => throw s!"IRExec: node {i}: transpose expects 1 parent ({n.summary})"
  | _ => throw s!"IRExec: internal error: operation routed to lowerShape"


end Internal
end IRExec
end Autograd
end Runtime
