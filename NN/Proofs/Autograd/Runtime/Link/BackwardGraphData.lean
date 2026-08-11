/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Runtime.Link.BackwardGraph

/-!
# GraphData Backward Pass Link

This file lifts the dense-backward correctness theorem from plain graphs to `GraphData`, where the
forward/backward closures carry an additional payload such as parameters or configuration data.
-/

@[expose] public section

namespace Proofs
namespace Autograd
namespace Algebra

open Spec
open Tensor

namespace Graph

open Runtime
open Runtime.Autograd

/--
Variant of `backwardDenseFrom_compileAux_eq_backpropAllCtx` for the `GraphData` interface.

This is useful when a graph carries extra payload `Δ` (e.g. parameters/config) through forward and
backward closures.
-/
theorem backwardDenseFrom_compileAuxData_eq_backpropAllCtx {α : Type} {Δ : Type} [DecidableEq Shape]
  [CommSemiring α]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TList α Γ) (d0 : Δ)
    (seed : TList α (Γ ++ ss)) :
    Runtime.Autograd.Tape.backwardDenseFrom (t := (compileAuxData (α := α) (Δ := Δ) (Γ := Γ) (ss :=
      ss) g x d0).1)
        (grads0 := TList.toAnyArray (α := α) (ss := Γ ++ ss) seed) =
      .ok
        (TList.toAnyArray (α := α) (ss := Γ ++ ss)
          (_root_.Proofs.Autograd.Algebra.GraphData.backpropAllCtx (α := α) (Δ := Δ) (Γ := Γ) (ss :=
            ss) g x d0 seed)) := by
  induction g with
  | nil =>
      -- Only leaf nodes; `backwardDenseFromLoop` does nothing because every leaf's `backward` is
      -- `[]`.
      have hnsize :
          Γ.length =
            (addLeaves (α := α) (t := Runtime.Autograd.Tape.empty (α := α)) (Γ := Γ) x).nodes.size
              := by
        simp [size_addLeaves, Runtime.Autograd.Tape.empty]

      have hloop :
          Runtime.Autograd.Tape.backwardDenseFromLoop
              (t := addLeaves (α := α) (t := Runtime.Autograd.Tape.empty (α := α)) (Γ := Γ) x)
              (addLeaves (α := α) (t := Runtime.Autograd.Tape.empty (α := α)) (Γ := Γ) x).nodes.size
              (TList.toAnyArray (α := α) (ss := Γ) (TList.cast (α := α) (h := (List.append_nil Γ))
                seed)) =
            Except.ok
              (TList.toAnyArray (α := α) (ss := Γ)
                (TList.cast (α := α) (h := (List.append_nil Γ)) seed)) := by
        let t :=
          addLeaves (α := α) (t := Runtime.Autograd.Tape.empty (α := α)) (Γ := Γ) x
        have hnodes :
            t.nodes =
              (TList.toAnyArray (α := α) (ss := Γ) x).map (leafNodeOfAny (α := α)) := by
          simp [t, nodes_addLeaves, Runtime.Autograd.Tape.empty]

        let seedArr :=
          TList.toAnyArray (α := α) (ss := Γ) (TList.cast (α := α) (h := List.append_nil Γ) seed)
        have htlen : t.nodes.size = Γ.length := by
          simpa [t] using hnsize.symm

        have loop_id :
            ∀ n, n ≤ t.nodes.size →
              Runtime.Autograd.Tape.backwardDenseFromLoop (t := t) n seedArr = Except.ok seedArr :=
                by
          intro n hnle
          induction n with
          | zero =>
              rfl
          | succ n ihn =>
              have hnlt : n < t.nodes.size :=
                Nat.lt_of_lt_of_le (Nat.lt_succ_self n) hnle
              have hnle' : n ≤ t.nodes.size :=
                Nat.le_trans (Nat.le_succ n) hnle
              have hidSeed : n < seedArr.size := by
                have : n < Γ.length := by simpa [htlen] using hnlt
                simpa [seedArr, TList.size_toAnyArray] using this
              have hidX : n < (TList.toAnyArray (α := α) (ss := Γ) x).size := by
                have : n < Γ.length := by simpa [htlen] using hnlt
                simpa [TList.size_toAnyArray] using this

              have hnode :
                  t.getNode? n =
                    some
                      (leafNodeOfAny (α := α)
                        ((TList.toAnyArray (α := α) (ss := Γ) x)[n]'hidX)) := by
                simp [Runtime.Autograd.Tape.getNode?, hnodes, Array.getElem?_map, leafNodeOfAny,
                  Array.getElem?_eq_getElem (xs := TList.toAnyArray (α := α) (ss := Γ) x) (i := n)
                    hidX]

              have hshape :
                  (seedArr[n]'hidSeed).s =
                    ((TList.toAnyArray (α := α) (ss := Γ) x)[n]'hidX).s := by
                let i : Fin Γ.length := ⟨n, by
                  have : n < Γ.length := by simpa [htlen] using hnlt
                  exact this⟩
                have hx_s :
                    ((TList.toAnyArray (α := α) (ss := Γ) x)[n]'hidX).s = Γ.get i := by
                  simpa [i, Runtime.Autograd.AnyTensor.mk] using
                    congrArg Runtime.AnyTensor.s (TList.get_toAnyArray (α := α) (ss := Γ) x i)
                have hseed_s :
                    (seedArr[n]'hidSeed).s = Γ.get i := by
                  simpa [seedArr, i, Runtime.Autograd.AnyTensor.mk] using congrArg Runtime.AnyTensor.s
                    (TList.get_toAnyArray (α := α) (ss := Γ)
                      (TList.cast (α := α) (h := List.append_nil Γ) seed) i)
                exact hseed_s.trans hx_s.symm

              have hstepn :
                  Runtime.Autograd.Tape.backwardDenseFromStep (t := t) seedArr n = Except.ok seedArr
                    := by
                have hidSeed0 : n < (TList.toAnyArray (α := α) (ss := Γ ++ []) seed).size := by
                  have : n < Γ.length := by simpa [htlen] using hnlt
                  simpa [TList.size_toAnyArray] using this
                simp [Runtime.Autograd.Tape.backwardDenseFromStep, hnode, leafNodeOfAny, seedArr,
                  Array.getElem?_eq_getElem (xs := (TList.toAnyArray (α := α) (ss := Γ ++ []) seed))
                    (i := n) hidSeed0]
                have hcond : seed.toAnyArray[n].s = x.toAnyArray[n].s := by
                  simpa [seedArr] using hshape
                simp [hcond]
                rfl

              simp [Runtime.Autograd.Tape.backwardDenseFromLoop, hstepn]
              change Runtime.Autograd.Tape.backwardDenseFromLoop (t := t) n seedArr =
                Except.ok seedArr
              exact ihn hnle'

        simpa [t, seedArr] using loop_id t.nodes.size (le_rfl)

      simpa [compileAuxData, _root_.Proofs.Autograd.Algebra.GraphData.backpropAllCtx,
        Runtime.Autograd.Tape.backwardDenseFrom, hnsize, TList.toAnyArray_cast] using hloop
  | snoc g node ih =>
      rename_i ssPrev τ
      rcases hprev : compileAuxData (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0 with ⟨tPrev,
        ctxPrev⟩
      have hctxPrev :
          ctxPrev = _root_.Proofs.Autograd.Algebra.GraphData.eval (α := α) (Δ := Δ) (Γ := Γ) (ss :=
            ssPrev) g x d0 := by
        simpa [hprev] using
          (compileAuxData_ctx_eq_eval (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0)
      have htPrevSize : tPrev.nodes.size = Γ.length + ssPrev.length := by
        simpa [hprev] using
          (compileAuxData_nodes_size (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0)

      let assoc : (Γ ++ ssPrev) ++ [τ] = Γ ++ (ssPrev ++ [τ]) := List.append_assoc Γ ssPrev [τ]
      let seed' : TList α ((Γ ++ ssPrev) ++ [τ]) := TList.cast (α := α) (h := assoc.symm) seed
      let seedPrev : TList α (Γ ++ ssPrev) :=
        (TList.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seed').1
      let seedOut : Tensor α τ :=
        (TList.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seed').2
      have hseed' :
          TList.snoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seedPrev seedOut = seed' := by
        simpa [seedPrev, seedOut] using
          (TList.snoc_unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) (xs := seed'))

      let outAny : Runtime.AnyTensor α := Runtime.Autograd.AnyTensor.mk seedOut

      let y := node.forward ctxPrev d0
      let runtimeNode : Runtime.Autograd.Node α :=
        { name := some "proof-compiled"
          value := Runtime.Autograd.AnyTensor.mk y
          requires_grad := true
          parents := []
          backward := fun dLdyAny => by
            if h : dLdyAny.s = τ then
              let dLdy : Tensor α τ := Tensor.castShape dLdyAny.t h
              let contribs := node.vjp ctxPrev d0 dLdy
              exact .ok (TList.toIndexedAnyList (α := α) (ss := Γ ++ ssPrev) contribs 0)
            else
              exact .error "autograd: upstream gradient shape mismatch"
        }
      let tNext : Runtime.Autograd.Tape α := (Runtime.Autograd.Tape.addNode (t := tPrev)
        runtimeNode).1
      have htNextNodes :
          tNext.nodes = tPrev.nodes.push runtimeNode := by
        simp [tNext, Runtime.Autograd.Tape.addNode]
      have htNextSize :
          tNext.nodes.size = tPrev.nodes.size + 1 := by
        simp [htNextNodes]

      have hseedArr :
          TList.toAnyArray (α := α) (ss := Γ ++ (ssPrev ++ [τ])) seed =
            (TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) seedPrev).push outAny := by
        have hcast :
            TList.toAnyArray (α := α) (ss := (Γ ++ ssPrev) ++ [τ]) seed' =
              TList.toAnyArray (α := α) (ss := Γ ++ (ssPrev ++ [τ])) seed := by
          simp [seed']
        rw [← hcast]
        have : seed' = TList.snoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seedPrev seedOut := by
          simpa using hseed'.symm
        simp [this, outAny, TList.toAnyArray_snoc]

      have hsizeCheck :
          (TList.toAnyArray (α := α) (ss := Γ ++ (ssPrev ++ [τ])) seed).size = tNext.nodes.size :=
            by
        simp [hseedArr, htNextSize, htPrevSize, TList.size_toAnyArray, Nat.add_assoc]

      let ctx := _root_.Proofs.Autograd.Algebra.GraphData.eval (α := α) (Δ := Δ) (Γ := Γ) (ss :=
        ssPrev) g x d0
      let contrib := node.vjp ctx d0 seedOut
      let seedPrev' := TList.add (α := α) (ss := Γ ++ ssPrev) seedPrev contrib
      let gradsPrev :=
        _root_.Proofs.Autograd.Algebra.GraphData.backpropAllCtx (α := α) (Δ := Δ) (Γ := Γ) (ss :=
          ssPrev) g x d0 seedPrev'

      have hTape :
          (compileAuxData (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev ++ [τ]) (.snoc (ss := ssPrev) (τ
            := τ) g node) x d0).1 =
            tNext := by
        simp [compileAuxData, hprev, tNext, y, runtimeNode]

      have hBackpropArr :
          TList.toAnyArray (α := α) (ss := Γ ++ (ssPrev ++ [τ]))
              (_root_.Proofs.Autograd.Algebra.GraphData.backpropAllCtx (α := α) (Δ := Δ) (Γ := Γ)
                (ss := ssPrev ++ [τ])
                (.snoc (ss := ssPrev) (τ := τ) g node) x d0 seed) =
            (TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) gradsPrev).push outAny := by
        simp [_root_.Proofs.Autograd.Algebra.GraphData.backpropAllCtx, seed', seedPrev, seedOut,
          ctx, contrib,
          seedPrev', gradsPrev, outAny, TList.toAnyArray_cast, TList.toAnyArray_snoc]

      have hmain :
          Runtime.Autograd.Tape.backwardDenseFrom (t := tNext)
              (grads0 := TList.toAnyArray (α := α) (ss := Γ ++ (ssPrev ++ [τ])) seed) =
            .ok ((TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) gradsPrev).push outAny) := by
        simp [Runtime.Autograd.Tape.backwardDenseFrom, hseedArr, htNextSize, htPrevSize]
        let n : Nat := tPrev.nodes.size
        let seedPrevArr : Array (Runtime.AnyTensor α) :=
          TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) seedPrev
        let seedPrevArr' : Array (Runtime.AnyTensor α) :=
          TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) seedPrev'
        let gradsPrevArr : Array (Runtime.AnyTensor α) :=
          TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) gradsPrev

        have hsizeSeedPrevArr : seedPrevArr.size = n := by
          simp [seedPrevArr, n, TList.size_toAnyArray, htPrevSize, List.length_append]

        have hnodeLast :
            Runtime.Autograd.Tape.getNode? (t := tNext) n = some runtimeNode := by
          simp [Runtime.Autograd.Tape.getNode?, tNext, htNextNodes, n]

        have hreqLast : runtimeNode.requires_grad = true := by rfl

        have hshapeLast : outAny.s = runtimeNode.value.s := by
          simp [outAny, runtimeNode]
          rfl

        have hpids :
            ∀ {pid : Nat} {pg : Runtime.AnyTensor α},
              (pid, pg) ∈ (TList.toIndexedAnyList (α := α) (ss := Γ ++ ssPrev) contrib 0) → pid < n
                := by
          intro pid pg hmem
          have hback :
              runtimeNode.backward outAny =
                .ok (TList.toIndexedAnyList (α := α) (ss := Γ ++ ssPrev) contrib 0) := by
            have hτ : outAny.s = τ := by rfl
            have hcastAny : ∀ h : outAny.s = τ, Tensor.castShape outAny.t h = seedOut := by
              intro h
              cases h
              rfl
            simp [runtimeNode, outAny, hτ, hcastAny, ctx, contrib, hctxPrev]
          have hpidlt :=
            compileAuxData_backward_pids_lt_id (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev ++ [τ])
              (.snoc (ss := ssPrev) (τ := τ) g node) x d0
              n runtimeNode (by
                simpa [compileAuxData, hprev, tNext, runtimeNode, Runtime.Autograd.Tape.getNode?,
                  htNextNodes, n])
              outAny _ hback hmem
          simpa [n] using hpidlt

        have hnodes0 :
            ∀ i (hi : i < (Γ ++ ssPrev).length),
              let id := (0 : Nat) + i
              ∃ nodeAt : Runtime.Autograd.Node α,
                tNext.getNode? id = some nodeAt ∧ nodeAt.requires_grad = true ∧
                  nodeAt.value.s =
                    ((TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) seedPrev)[i]'(by
                        simpa [TList.size_toAnyArray] using hi)).s := by
          intro i hi
          have hiT : i < tPrev.nodes.size := by
            -- `tPrev.nodes.size = (Γ ++ ssPrev).length`
            simpa [htPrevSize, List.length_append, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]
              using hi
          let nodeAt : Runtime.Autograd.Node α := tPrev.nodes[i]'hiT
          have hgetNext : tNext.getNode? i = some nodeAt := by
            -- index `< tPrev.nodes.size`, so `push` doesn't change it
            have : (tPrev.nodes.push runtimeNode)[i]? = some (tPrev.nodes[i]'hiT) := by
              simpa using (Array.getElem?_push_lt (xs := tPrev.nodes) (x := runtimeNode) hiT)
            simpa [Runtime.Autograd.Tape.getNode?, tNext, htNextNodes, nodeAt] using this
          have hreq : nodeAt.requires_grad = true := by
            have hreq' :=
              (compileAuxData_requires_grad_true (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0) i
                (by
                simpa [hprev] using hiT)
            simpa [hprev, nodeAt] using hreq'
          -- Shapes: both are the `i`th shape in `Γ ++ ssPrev`.
          have hseedShape :
              nodeAt.value.s =
                ((TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) seedPrev)[i]'(by
                    simpa [TList.size_toAnyArray] using hi)).s := by
            let fi : Fin (Γ ++ ssPrev).length := ⟨i, hi⟩
            -- `tPrev.nodes.map value = ctxPrev.toAnyArray`
            have hvals :
                tPrev.nodes.map (fun nd => nd.value) =
                  TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) ctxPrev := by
              simpa [hprev] using
                (compileAuxData_values_eq (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0)
            have hvalOpt := congrArg (fun a => a[i]?) hvals
            -- Evaluate both sides at `i`.
            have hnodeVal :
                nodeAt.value = (TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) ctxPrev)[i]'(by
                  -- `i < ctxPrev.toAnyArray.size` because it matches `tPrev.nodes.size`
                  simpa [TList.size_toAnyArray, htPrevSize, List.length_append, Nat.add_assoc] using
                    hiT) := by
              -- Left: map+index gives `some nodeAt.value`
              have hleft :
                  (tPrev.nodes.map (fun nd => nd.value))[i]? = some nodeAt.value := by
                have : tPrev.nodes[i]? = some nodeAt := by
                  simp [nodeAt,
                    Array.getElem?_eq_getElem (xs := tPrev.nodes) (i := i) hiT]
                simp [Array.getElem?_map, this, nodeAt]
              -- Right: in-bounds `getElem?` is `some _`
              have hright :
                  (TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) ctxPrev)[i]? =
                    some ((TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) ctxPrev)[i]'(by
                      simpa [TList.size_toAnyArray, htPrevSize, List.length_append, Nat.add_assoc]
                        using hiT)) := by
                have hiCtx :
                    i < (TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) ctxPrev).size := by
                  simpa [TList.size_toAnyArray, htPrevSize, List.length_append, Nat.add_assoc] using
                    hiT
                simp [Array.getElem?_eq_getElem (xs := (TList.toAnyArray (α := α) (ss := Γ ++
                  ssPrev) ctxPrev)) (i := i) hiCtx]
              -- Combine and extract the value equality.
              have : some nodeAt.value =
                  some ((TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) ctxPrev)[i]'(by
                    simpa [TList.size_toAnyArray, htPrevSize, List.length_append, Nat.add_assoc]
                      using hiT)) := by
                -- rewrite both sides of `hvalOpt` using `hleft`/`hright`
                simpa [hleft, hright] using hvalOpt
              simpa using congrArg (fun o => o.getD nodeAt.value) this
            have hnode_s :
                nodeAt.value.s = (Γ ++ ssPrev).get fi := by
              have hiCtx :
                  i < (TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) ctxPrev).size := by
                simpa [TList.size_toAnyArray, htPrevSize, List.length_append, Nat.add_assoc] using
                  hiT
              have hctx_s :
                  ((TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) ctxPrev)[i]'hiCtx).s =
                    (Γ ++ ssPrev).get fi := by
                -- `ctxPrev.get fi : Tensor α ((Γ ++ ssPrev).get fi)`, so the RHS shape is
                -- definitional.
                simpa [fi, Runtime.Autograd.AnyTensor.mk] using
                  congrArg Runtime.AnyTensor.s
                    (TList.get_toAnyArray (α := α) (ss := Γ ++ ssPrev) ctxPrev fi)
              -- rewrite the LHS using `hnodeVal`
              simpa [hnodeVal] using hctx_s
            have hseed_s :
                ((TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) seedPrev)[i]'(by
                    simpa [TList.size_toAnyArray] using hi)).s = (Γ ++ ssPrev).get fi := by
              simpa [fi, Runtime.Autograd.AnyTensor.mk] using congrArg Runtime.AnyTensor.s
                (TList.get_toAnyArray (α := α) (ss := Γ ++ ssPrev) seedPrev fi)
            exact hnode_s.trans hseed_s.symm

          -- discharge the `let id := 0 + i`
          refine ⟨nodeAt, ?_, hreq, hseedShape⟩
          simpa [Nat.zero_add] using hgetNext

        have hstepLast :
            Runtime.Autograd.Tape.backwardDenseFromStep (t := tNext) (seedPrevArr.push outAny) n =
              .ok (seedPrevArr'.push outAny) := by
          have haccLast : (seedPrevArr.push outAny)[n]? = some outAny := by
            have : (seedPrevArr.push outAny)[seedPrevArr.size]? = some outAny := by
              simp
            simpa [hsizeSeedPrevArr] using this

          -- Show the `addGradAll` fold for the last node matches `TList.add` on the prefix, leaving
          -- `[outAny]` untouched.
          have hfoldLast :
              (TList.toIndexedAnyList (α := α) (ss := Γ ++ ssPrev) contrib 0).foldlM
                  (fun acc2 (pid, pg) => Runtime.Autograd.Tape.addGradAll (t := tNext) acc2 pid pg)
                  (seedPrevArr.push outAny) =
                .ok (seedPrevArr'.push outAny) := by
            have hfold :=
              foldlM_addGradAll_toIndexedAnyList_eq_add (α := α) (t := tNext)
                (ss := Γ ++ ssPrev) (pref := #[]) (seed := seedPrev) (contrib := contrib) (suffix :=
                  #[outAny])
                (by
                  intro i hi
                  have := hnodes0 i hi
                  simpa using this)
            -- Simplify the array concatenations and rewrite `seedPrev'`.
            simpa [seedPrevArr, seedPrevArr', seedPrev', Array.append_assoc, Array.append_empty,
              Array.empty_append,
              Array.append_singleton, TList.toAnyArray_cast] using hfold

          -- Unfold the step and rewrite the `backward` call using `hfoldLast`.
          cases hshapeLast
          have hreqLast : runtimeNode.requires_grad = true := by rfl
          have hout : outAny.s = τ := by rfl
          have hshapeNode : outAny.s = runtimeNode.value.s := by rfl
          have hbackLast :
              runtimeNode.backward outAny =
                .ok (TList.toIndexedAnyList (α := α) (ss := Γ ++ ssPrev) contrib 0) := by
            simp [runtimeNode, outAny, hctxPrev, ctx, contrib,
              Runtime.Autograd.AnyTensor.mk]

          have hbackLast2 :
              runtimeNode.backward
                  { s := runtimeNode.value.s
                    t := Tensor.castShape outAny.t hshapeNode } =
                .ok (TList.toIndexedAnyList (α := α) (ss := Γ ++ ssPrev) contrib 0) := by
            -- Keep `Tensor.cast_shape` folded so this rewrite matches the `backwardDenseFromStep`
            -- unfolding.
            cases hshapeNode
            change runtimeNode.backward outAny =
              .ok (TList.toIndexedAnyList (α := α) (ss := Γ ++ ssPrev) contrib 0)
            exact hbackLast
          -- Unfold the step and reduce the control flow (`getNode?`, `requires_grad`, `acc[id]?`,
          -- shape check),
          -- then rewrite the `backward` call and finish with the pre-proved fold lemma.
          simp [Runtime.Autograd.Tape.backwardDenseFromStep, hnodeLast, hreqLast, haccLast]
          simp [hshapeNode]
          rw [hbackLast2]
          change
            List.foldlM (fun acc2 x => tNext.addGradAll acc2 x.1 x.2)
                (seedPrev.toAnyArray.push (Runtime.Autograd.AnyTensor.mk seedOut))
                (contrib.toIndexedAnyList 0) =
              Except.ok (seedPrev'.toAnyArray.push (Runtime.Autograd.AnyTensor.mk seedOut))
          simpa [seedPrevArr, seedPrevArr', outAny] using hfoldLast

        have ihPrevLoop :
            Runtime.Autograd.Tape.backwardDenseFromLoop (t := tPrev) n seedPrevArr' =
              .ok gradsPrevArr := by
          have ihPrev :
              Runtime.Autograd.Tape.backwardDenseFrom (t := tPrev) (grads0 := seedPrevArr') =
                .ok gradsPrevArr := by
            have h := ih (seed := seedPrev')
            simpa [hprev, seedPrevArr', gradsPrevArr, gradsPrev] using h
          have hsizeSeedPrevArr' : seedPrevArr'.size = n := by
            simp [seedPrevArr', n, TList.size_toAnyArray, htPrevSize, List.length_append]
          have hsize : seedPrevArr'.size = tPrev.nodes.size := by
            simpa [n] using hsizeSeedPrevArr'
          simpa [Runtime.Autograd.Tape.backwardDenseFrom, hsize, n] using ihPrev

        -- Helper: `addGradAll` commutes with pushing an unused last slot.
        have haddGradAllPush :
            ∀ (acc : Array (Runtime.AnyTensor α)) (hacc : acc.size = n)
              (pid : Nat) (pg : Runtime.AnyTensor α),
              pid < n →
              Runtime.Autograd.Tape.addGradAll (t := tNext) (grads := acc.push outAny) pid pg =
                Except.map (fun a => a.push outAny)
                  (Runtime.Autograd.Tape.addGradAll (t := tPrev) (grads := acc) pid pg) := by
          intro acc hacc pid pg hpid
          have hpidPrev : pid < tPrev.nodes.size := by
            simpa [n] using hpid
          let nodeAt : Runtime.Autograd.Node α := tPrev.nodes[pid]'hpidPrev
          have hnodePrev : Runtime.Autograd.Tape.getNode? (t := tPrev) pid = some nodeAt := by
            simp [Runtime.Autograd.Tape.getNode?, nodeAt,
              Array.getElem?_eq_getElem (xs := tPrev.nodes) (i := pid) hpidPrev]
          have hnodeNext : Runtime.Autograd.Tape.getNode? (t := tNext) pid = some nodeAt := by
            have : (tPrev.nodes.push runtimeNode)[pid]? = some (tPrev.nodes[pid]'hpidPrev) := by
              simpa using (Array.getElem?_push_lt (xs := tPrev.nodes) (x := runtimeNode) hpidPrev)
            simpa [Runtime.Autograd.Tape.getNode?, tNext, htNextNodes, nodeAt] using this

          have hpidAcc : pid < acc.size := by
            simpa [hacc] using hpid
          have hgetPrev : acc[pid]? = some (acc[pid]'hpidAcc) := by
            simp
          have hgetNext : (acc.push outAny)[pid]? = some (acc[pid]'hpidAcc) := by
            simpa using (Array.getElem?_push_lt (xs := acc) (x := outAny) hpidAcc)

          cases hreq : nodeAt.requires_grad with
          | false =>
              simp [Runtime.Autograd.Tape.addGradAll, hnodePrev, hnodeNext, hreq, Except.map]
              rfl
          | true =>
              by_cases hshape : pg.s = nodeAt.value.s
              · by_cases hex : (acc[pid]'hpidAcc).s = nodeAt.value.s
                ·
                  let pg' : Runtime.AnyTensor α :=
                    { s := nodeAt.value.s, t := Tensor.castShape pg.t hshape }
                  let existing' : Runtime.AnyTensor α :=
                    { s := nodeAt.value.s, t := Tensor.castShape (acc[pid]'hpidAcc).t hex }
                  cases hadd : Runtime.Autograd.AnyTensor.add existing' pg' with
                  | error e =>
                      have hprev :
                          Runtime.Autograd.Tape.addGradAll (t := tPrev) (grads := acc) pid pg =
                            .error e := by
                        simp [Runtime.Autograd.Tape.addGradAll, hnodePrev, hreq, hshape, hgetPrev,
                          hex, pg', existing', hadd,
                          throw, throwThe, MonadExceptOf.throw]
                        simp [Bind.bind, Except.bind]
                      have hnext :
                          Runtime.Autograd.Tape.addGradAll (t := tNext) (grads := acc.push outAny)
                            pid pg = .error e := by
                        simp [Runtime.Autograd.Tape.addGradAll, hnodeNext, hreq, hshape, hgetNext,
                          hex, pg', existing', hadd,
                          throw, throwThe, MonadExceptOf.throw]
                        simp [Bind.bind, Except.bind]
                      simp [hprev, hnext, Except.map]
                  | ok summed =>
                      have hpidAccPush : pid < (acc.push outAny).size := by
                        simpa [Array.size_push] using Nat.lt_trans hpidAcc (Nat.lt_succ_self
                          acc.size)
                      have hprev :
                          Runtime.Autograd.Tape.addGradAll (t := tPrev) (grads := acc) pid pg =
                            .ok (acc.set pid summed (h := hpidAcc)) := by
                        simp [Runtime.Autograd.Tape.addGradAll, hnodePrev, hreq, hshape, hex, pg',
                          existing', hadd,
                          hpidAcc, throw, throwThe, MonadExceptOf.throw]
                      have hnext :
                          Runtime.Autograd.Tape.addGradAll (t := tNext) (grads := acc.push outAny)
                            pid pg =
                            .ok ((acc.set pid summed (h := hpidAcc)).push outAny) := by
                        have hpid_le : pid ≤ acc.size := Nat.le_of_lt hpidAcc
                        have hget : (acc.push outAny)[pid] = acc[pid] := by
                          simpa using
                            (Array.getElem_push_lt (xs := acc) (x := outAny) (i := pid) hpidAcc)
                        simp [Runtime.Autograd.Tape.addGradAll, hnodeNext, hreq, hshape, hex, pg',
                          existing', hadd,
                          hpidAcc, hpid_le, hget, Array.set_push, throw, throwThe,
                            MonadExceptOf.throw]
                      simp [hprev, hnext, Except.map]
                ·
                  simp [Runtime.Autograd.Tape.addGradAll, hnodePrev, hnodeNext, hreq, hshape,
                    hgetPrev, hgetNext, hex,
                    Except.map, throw, throwThe, MonadExceptOf.throw]
              ·
                simp [Runtime.Autograd.Tape.addGradAll, hnodePrev, hnodeNext, hreq, hshape,
                  Except.map,
                  throw, throwThe, MonadExceptOf.throw]

        -- Helper: `backwardDenseFromStep` commutes with pushing an unused last slot for ids `< n`.
        have hstepPush :
            ∀ (id : Nat) (hid : id < n) (acc : Array (Runtime.AnyTensor α)),
              acc.size = n →
              Runtime.Autograd.Tape.backwardDenseFromStep (t := tNext) (acc.push outAny) id =
                Except.map (fun a => a.push outAny)
                  (Runtime.Autograd.Tape.backwardDenseFromStep (t := tPrev) acc id) := by
          intro id hid acc hacc
          have hidPrev : id < tPrev.nodes.size := by
            simpa [n] using hid
          let nodeAt : Runtime.Autograd.Node α := tPrev.nodes[id]'hidPrev
          have hnodePrev : Runtime.Autograd.Tape.getNode? (t := tPrev) id = some nodeAt := by
            simp [Runtime.Autograd.Tape.getNode?, nodeAt,
              Array.getElem?_eq_getElem (xs := tPrev.nodes) (i := id) hidPrev]
          have hnodeNext : Runtime.Autograd.Tape.getNode? (t := tNext) id = some nodeAt := by
            have : (tPrev.nodes.push runtimeNode)[id]? = some (tPrev.nodes[id]'hidPrev) := by
              simpa using (Array.getElem?_push_lt (xs := tPrev.nodes) (x := runtimeNode) hidPrev)
            simpa [Runtime.Autograd.Tape.getNode?, tNext, htNextNodes, nodeAt] using this
          have hidAcc : id < acc.size := by
            simpa [hacc] using hid
          have hgetAcc : acc[id]? = some (acc[id]'hidAcc) := by
            simp
          have hgetAccPush : (acc.push outAny)[id]? = some (acc[id]'hidAcc) := by
            simpa using (Array.getElem?_push_lt (xs := acc) (x := outAny) hidAcc)
          cases hreq : nodeAt.requires_grad with
          | false =>
              simp [Runtime.Autograd.Tape.backwardDenseFromStep, hnodePrev, hnodeNext, hreq,
                hgetAcc, hgetAccPush,
                Except.map]
              rfl
          | true =>
              by_cases hshape : (acc[id]'hidAcc).s = nodeAt.value.s
              · -- shape ok, split on `backward` result
                let dLdy : Runtime.AnyTensor α :=
                  { s := nodeAt.value.s, t := Tensor.castShape (acc[id]'hidAcc).t hshape }
                cases hback : nodeAt.backward dLdy with
                | error e =>
                    simp [Runtime.Autograd.Tape.backwardDenseFromStep, hnodePrev, hnodeNext, hreq,
                      hgetAcc, hgetAccPush,
                      hshape, dLdy, hback, Except.map]
                    rfl
                | ok contribs =>
                    have hpids :
                          ∀ {pid : Nat} {pg : Runtime.AnyTensor α}, (pid, pg) ∈ contribs → pid < id
                            := by
                        intro pid pg hmem
                        have hgetComp :
                            Runtime.Autograd.Tape.getNode?
                                (t := (compileAuxData (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x
                                  d0).1)
                                id =
                              some nodeAt := by
                          simpa [hprev] using hnodePrev
                        exact
                          compileAuxData_backward_pids_lt_id (α := α) (Δ := Δ) (Γ := Γ) (ss :=
                            ssPrev) g x d0 id nodeAt
                              hgetComp dLdy contribs hback hmem

                    have hfoldAux :
                        ∀ (cs : List (Nat × Runtime.AnyTensor α)) (acc0 accOut : Array
                          (Runtime.AnyTensor α)),
                          acc0.size = n →
                          (∀ {pid : Nat} {pg : Runtime.AnyTensor α}, (pid, pg) ∈ cs → pid < n) →
                          cs.foldlM (fun acc2 (pid, pg) => Runtime.Autograd.Tape.addGradAll (t :=
                            tNext) acc2 pid pg)
                              (acc0.push outAny) =
                            Except.map (fun a => a.push outAny)
                              (cs.foldlM
                                (fun acc2 (pid, pg) => Runtime.Autograd.Tape.addGradAll (t := tPrev)
                                  acc2 pid pg) acc0) := by
                      intro cs
                      induction cs with
                      | nil =>
                          intro acc0 accOut _hsize _hpids
                          simp [List.foldlM, Except.map]
                          rfl
                      | cons hd tl ih =>
                          intro acc0 accOut hsize hpids
                          rcases hd with ⟨pid, pg⟩
                          have hpid : pid < n := by
                            exact hpids (pid := pid) (pg := pg) (by simp)
                          have hadd :=
                            haddGradAllPush (acc := acc0) (hacc := hsize) (pid := pid) (pg := pg)
                              hpid
                          cases hret : Runtime.Autograd.Tape.addGradAll (t := tPrev) (grads := acc0)
                            pid pg with
                          | error e =>
                              -- both folds error at the first step
                              simp [List.foldlM, hret, hadd, Except.map]
                              rfl
                          | ok acc1 =>
                              have hret' :
                                  Runtime.Autograd.Tape.addGradAll (t := tNext) (grads := acc0.push
                                    outAny) pid pg =
                                    .ok (acc1.push outAny) := by
                                -- unfold `Except.map` in `hadd`
                                simpa [Except.map, hret] using hadd
                              have hsize1 : acc1.size = n := by
                                have := addGradAll_ok_size (t := tPrev) (grads := acc0) (id := pid)
                                  (g := pg)
                                  (grads' := acc1) (by simpa using hret)
                                simpa [hsize] using this
                              have hpids_tl :
                                  ∀ {pid : Nat} {pg : Runtime.AnyTensor α}, (pid, pg) ∈ tl → pid < n
                                    := by
                                intro pid pg hmem
                                exact hpids (pid := pid) (pg := pg) (by simp [hmem])
                              have ih' :=
                                ih (acc0 := acc1) (accOut := accOut) hsize1 hpids_tl
                              -- unfold the `foldlM` for the cons case on both sides
                              simp [List.foldlM, hret, hret']
                              cases htl :
                                  List.foldlM (fun acc2 x => tPrev.addGradAll acc2 x.1 x.2)
                                    acc1 tl <;>
                                simpa [Bind.bind, Except.bind, htl] using ih'

                    have hpids_n :
                        ∀ {pid : Nat} {pg : Runtime.AnyTensor α}, (pid, pg) ∈ contribs → pid < n :=
                          by
                      intro pid pg hmem
                      exact Nat.lt_trans (hpids (pid := pid) (pg := pg) hmem) hid

                    -- Apply the fold lemma.
                    have hfold :=
                      hfoldAux contribs acc (accOut := by
                        exact acc) hacc hpids_n
                    -- Unfold the step definitions, then discharge the remaining fold goal via
                    -- `hfold`.
                    simp [Runtime.Autograd.Tape.backwardDenseFromStep, hnodePrev, hnodeNext, hreq,
                      hgetAcc, hgetAccPush,
                      hshape, dLdy, hback]
                    cases hcs :
                        List.foldlM (fun acc2 x => tPrev.addGradAll acc2 x.1 x.2)
                          acc contribs <;>
                      simpa [Bind.bind, Except.bind, hcs] using hfold
              · -- shape mismatch
                simp [Runtime.Autograd.Tape.backwardDenseFromStep, hnodePrev, hnodeNext, hreq,
                  hgetAcc, hgetAccPush, hshape,
                  Except.map, throw, throwThe, MonadExceptOf.throw]

        -- The loop itself commutes with pushing an unused last slot.
        have hloopPush :
            ∀ m (hm : m ≤ n) (acc : Array (Runtime.AnyTensor α)),
              acc.size = n →
              Runtime.Autograd.Tape.backwardDenseFromLoop (t := tNext) m (acc.push outAny) =
                Except.map (fun a => a.push outAny)
                  (Runtime.Autograd.Tape.backwardDenseFromLoop (t := tPrev) m acc) := by
            intro m hm acc hacc
            induction m generalizing acc with
            | zero =>
                simp [Runtime.Autograd.Tape.backwardDenseFromLoop, Except.map]
                rfl
            | succ m ihm =>
                have hm' : m ≤ n := Nat.le_trans (Nat.le_succ m) hm
                have hmid : m < n := Nat.lt_of_lt_of_le (Nat.lt_succ_self m) hm
                have hstep := hstepPush (id := m) (hid := hmid) (acc := acc) hacc
                cases hret : Runtime.Autograd.Tape.backwardDenseFromStep (t := tPrev) acc m with
                | error e =>
                    -- both loops error on this step
                    simp [Runtime.Autograd.Tape.backwardDenseFromLoop, hret, hstep, Except.map]
                    rfl
                | ok acc1 =>
                    have hstep' :
                        Runtime.Autograd.Tape.backwardDenseFromStep (t := tNext) (acc.push outAny) m
                          =
                          .ok (acc1.push outAny) := by
                      simpa [Except.map, hret] using hstep
                    have hsize1 : acc1.size = n := by
                      have := backwardDenseFromStep_ok_size (t := tPrev) (acc := acc) (id := m)
                        (acc' := acc1)
                        (by simpa using hret)
                      simpa [hacc] using this
                    have ih' := ihm (acc := acc1) hm' hsize1
                    simp [Runtime.Autograd.Tape.backwardDenseFromLoop, hret, hstep']
                    cases hloop :
                        Runtime.Autograd.Tape.backwardDenseFromLoop (t := tPrev) m acc1 <;>
                      simpa [Bind.bind, Except.bind, hloop] using ih'

        have hloopFinal :
            Runtime.Autograd.Tape.backwardDenseFromLoop (t := tNext) n (seedPrevArr'.push outAny) =
              .ok (gradsPrevArr.push outAny) := by
          have h := hloopPush n (le_rfl) seedPrevArr' (by
            simp [seedPrevArr', n, TList.size_toAnyArray, htPrevSize, List.length_append])
          simpa [ihPrevLoop, Except.map] using h

        have hloopAll :
            Runtime.Autograd.Tape.backwardDenseFromLoop (t := tNext) (n + 1) (seedPrevArr.push
              outAny) =
              .ok (gradsPrevArr.push outAny) := by
          -- Unfold the loop one step, rewrite via `hstepLast`, then discharge with `hloopFinal`.
          simp [Runtime.Autograd.Tape.backwardDenseFromLoop, hstepLast]
          change Runtime.Autograd.Tape.backwardDenseFromLoop (t := tNext) n
              (seedPrevArr'.push outAny) =
            Except.ok (gradsPrevArr.push outAny)
          exact hloopFinal
        simpa [n, htPrevSize, Nat.add_assoc] using hloopAll

      simpa [hTape, hBackpropArr] using hmain
  end Graph

  end Algebra
  end Autograd
  end Proofs
