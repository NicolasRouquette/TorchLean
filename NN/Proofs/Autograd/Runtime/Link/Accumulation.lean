/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Runtime.Link.Invariants

/-!
# Runtime Gradient Accumulation Link

This file connects the executable dense-gradient array used by the runtime tape to the typed context
addition used in the proved autograd algebra. The main lemmas show that folding runtime gradient
updates over indexed tensors agrees with the proof-level `TList` accumulation operation.
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
Key accumulation lemma for the runtime dense gradient array:

Folding `Tape.addGradAll` over the contributions corresponding to a `TList` (via `toIndexedAnyList`)
is equivalent to pointwise addition of the typed contexts (`TList.add`), embedded back into the
array layout `pref ++ seed ++ suffix`.

This is the “runtime accumulation matches proved addition” bridge.
-/
theorem foldlM_addGradAll_toIndexedAnyList_eq_add {α : Type} [Add α] [DecidableEq Shape]
    (t : Runtime.Autograd.Tape α) :
    ∀ {ss : List Shape} (pref : Array (Runtime.AnyTensor α)) (seed contrib : TList α ss)
      (suffix : Array (Runtime.AnyTensor α)),
      (∀ i (hi : i < ss.length),
        let id := pref.size + i
        ∃ node : Runtime.Autograd.Node α,
          t.getNode? id = some node ∧
            node.requires_grad = true ∧
            node.value.s =
              ((TList.toAnyArray (α := α) (ss := ss) seed)[i]'(by
                  simpa [TList.size_toAnyArray] using hi)).s) →
      (TList.toIndexedAnyList (α := α) (ss := ss) contrib pref.size).foldlM
          (fun acc2 (pid, pg) => Runtime.Autograd.Tape.addGradAll (t := t) acc2 pid pg)
          (pref ++ TList.toAnyArray (α := α) (ss := ss) seed ++ suffix) =
        .ok (pref ++
              TList.toAnyArray (α := α) (ss := ss) (TList.add (α := α) (ss := ss) seed contrib) ++
              suffix) := by
  intro ss pref seed contrib suffix hnodes
  induction ss generalizing pref with
  | nil =>
      cases seed; cases contrib
      simp [TList.toIndexedAnyList, TList.toAnyArray, TList.toAnyList, TList.add]
      rfl
  | cons s ss ih =>
      cases seed with
      | cons seedHead seedTail =>
        cases contrib with
        | cons contribHead contribTail =>
          let seedHeadAny : Runtime.AnyTensor α := Runtime.Autograd.AnyTensor.mk seedHead
          let contribHeadAny : Runtime.AnyTensor α := Runtime.Autograd.AnyTensor.mk contribHead
          let newHeadAny : Runtime.AnyTensor α := Runtime.Autograd.AnyTensor.mk (addSpec seedHead
            contribHead)

          have hseedArr :
              TList.toAnyArray (α := α) (ss := s :: ss) (TList.cons seedHead seedTail) =
                #[seedHeadAny] ++ TList.toAnyArray (α := α) (ss := ss) seedTail := by
            simpa [seedHeadAny] using
              (TList.toAnyArray_cons (α := α) (s := s) (ss := ss) seedHead seedTail)

          have hacc0 :
              pref ++ TList.toAnyArray (α := α) (ss := s :: ss) (TList.cons seedHead seedTail) ++
                suffix =
                (pref.push seedHeadAny) ++ TList.toAnyArray (α := α) (ss := ss) seedTail ++ suffix
                  := by
            -- Avoid `simp` loops between `push` and `++ #[x]`.
            -- Expand the seed array, reassociate, then rewrite `pref ++ #[x]` as `pref.push x`.
            rw [hseedArr]
            simp [Array.append_assoc]

          have h0 := hnodes 0 (by simp [List.length_cons])
          rcases h0 with ⟨node0, hnode0, hreq0, hshape0'⟩

          have hshape0 : node0.value.s = seedHeadAny.s := by
            have : ((TList.toAnyArray (α := α) (ss := s :: ss) (TList.cons seedHead
              seedTail))[0]'(by
              simp [TList.size_toAnyArray, List.length_cons])).s = seedHeadAny.s := by
              simp [hseedArr, seedHeadAny]
            exact hshape0'.trans this

          have hgetExisting :
              ((pref.push seedHeadAny) ++ TList.toAnyArray (α := α) (ss := ss) seedTail ++
                suffix)[pref.size]? =
                some seedHeadAny := by
            have hlt : pref.size < (pref.push seedHeadAny).size := by
              simp
            simp [Array.getElem?_append]

          have hsummed : Runtime.Autograd.AnyTensor.add seedHeadAny contribHeadAny = .ok newHeadAny
            := by
            -- Reduce the shape-cast using definitional equality of shapes.
            have hs :
                (Runtime.Autograd.AnyTensor.mk seedHead).s =
                  (Runtime.Autograd.AnyTensor.mk contribHead).s := by
              rfl
            cases hs
            simp [Runtime.Autograd.AnyTensor.add, Runtime.Autograd.AnyTensor.mk,
              seedHeadAny, contribHeadAny, newHeadAny]

          have hset :
              ((pref.push seedHeadAny) ++ TList.toAnyArray (α := α) (ss := ss) seedTail ++
                suffix).set
                  pref.size newHeadAny
                  (by
                    simp [Array.size_append, Nat.add_assoc]) =
                (pref.push newHeadAny) ++ TList.toAnyArray (α := α) (ss := ss) seedTail ++ suffix :=
                  by
            have hlt : pref.size < (pref.push seedHeadAny).size := by
              simp
            simp [Array.set_append_left (xs := pref.push seedHeadAny)
                  (ys := TList.toAnyArray (α := α) (ss := ss) seedTail ++ suffix)
                  (i := pref.size) (x := newHeadAny) hlt,
              Array.set_push]

          have hadd0 :
              Runtime.Autograd.Tape.addGradAll (t := t)
                  (pref ++ TList.toAnyArray (α := α) (ss := s :: ss) (TList.cons seedHead seedTail)
                    ++ suffix)
                  pref.size contribHeadAny =
                .ok ((pref.push newHeadAny) ++ TList.toAnyArray (α := α) (ss := ss) seedTail ++
                  suffix) := by
              have hidAcc :
                  pref.size <
                  ((pref.push seedHeadAny) ++ TList.toAnyArray (α := α) (ss := ss) seedTail ++
                    suffix).size := by
                simp [Array.size_append, Nat.add_assoc]
              have hshapeG : contribHeadAny.s = node0.value.s := by
                calc
                  contribHeadAny.s = seedHeadAny.s := by rfl
                  _ = node0.value.s := hshape0.symm
              have hshapeExisting : seedHeadAny.s = node0.value.s := by
                simpa using hshape0.symm
              have hnode0' : t.getNode? pref.size = some node0 := by
                simpa [Nat.add_zero] using hnode0
              have hgetExisting' :
                  ((pref.push seedHeadAny) ++ (TList.toAnyArray (α := α) (ss := ss) seedTail ++
                    suffix))[pref.size]? =
                    some seedHeadAny := by
                simpa [Array.append_assoc] using hgetExisting
              have hidAcc' :
                  pref.size <
                    ((pref.push seedHeadAny) ++ (TList.toAnyArray (α := α) (ss := ss) seedTail ++
                      suffix)).size := by
                simpa [Array.append_assoc] using hidAcc

              have : Runtime.Autograd.Tape.addGradAll (t := t)
                  ((pref.push seedHeadAny) ++ (TList.toAnyArray (α := α) (ss := ss) seedTail ++
                    suffix))
                  pref.size contribHeadAny =
                    .ok ((pref.push newHeadAny) ++
                      (TList.toAnyArray (α := α) (ss := ss) seedTail ++ suffix)) := by
                -- After rewriting `node0.value.s = seedHeadAny.s`, all shape casts become
                -- definitional.
                cases hshape0
                -- Reduce the dependent shape-casts by eliminating the equality proofs.
                cases hshapeExisting
                cases hshapeG

                have hid :
                    pref.size < pref.size + 1 + (ss.length + suffix.size) := by
                  -- `pref.size < pref.size + 1` and adding to the RHS preserves `<`.
                  exact Nat.lt_of_lt_of_le (Nat.lt_succ_self pref.size)
                    (Nat.le_add_right (pref.size + 1) (ss.length + suffix.size))

                have hseedShape : (Runtime.Autograd.AnyTensor.mk seedHead).s = node0.value.s := by
                  rfl
                have hcontribShape :
                    (Runtime.Autograd.AnyTensor.mk contribHead).s = node0.value.s := by
                  rfl

                -- Now `addGradAll` is a straight-line computation: fetch node, check flags/shapes,
                -- add, and overwrite the `pref.size` slot.
                -- Keep `Tensor.cast_shape` opaque here: the following tensor equalities are about
                -- the accumulated value, not the proof terms used to align shapes.
                simp [Runtime.Autograd.Tape.addGradAll, hnode0', hreq0, hseedShape, hcontribShape,
                  hid, seedHeadAny, contribHeadAny, newHeadAny, Array.set_push]

                have hsummed' :
                    Runtime.Autograd.AnyTensor.add
                        { s := node0.value.s
                          t := Tensor.castShape (Runtime.Autograd.AnyTensor.mk seedHead).t
                            hseedShape }
                        { s := node0.value.s
                          t := Tensor.castShape (Runtime.Autograd.AnyTensor.mk contribHead).t
                            hcontribShape } =
                      .ok newHeadAny := by
                  cases hseedShape
                  cases hcontribShape
                  simp [Runtime.Autograd.AnyTensor.add, Runtime.Autograd.AnyTensor.mk,
                    seedHeadAny, contribHeadAny, newHeadAny] at hsummed ⊢

                rw [hsummed']
                simp [newHeadAny]

              -- Rewrite back to the original associative form for the outer goal.
              rw [hacc0]
              rw [Array.append_assoc]
              conv_rhs => rw [Array.append_assoc]
              exact this

          have hnodesTail :
              ∀ i (hi : i < ss.length),
                let id := (pref.push newHeadAny).size + i
                ∃ node : Runtime.Autograd.Node α,
                  t.getNode? id = some node ∧ node.requires_grad = true ∧
                    node.value.s =
                      ((TList.toAnyArray (α := α) (ss := ss) seedTail)[i]'(by
                          simpa [TList.size_toAnyArray] using hi)).s := by
            intro i hi
            have h' :=
              hnodes (i + 1) (by
                simpa [List.length_cons] using Nat.succ_lt_succ hi)
            rcases h' with ⟨node, hnode, hreq, hshape⟩
            refine ⟨node, ?_, hreq, ?_⟩
            · simpa [Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using hnode
            ·
              have hiFull :
                  i + 1 < (TList.toAnyArray (α := α) (ss := s :: ss) (TList.cons seedHead
                    seedTail)).size := by
                have : i + 1 < (s :: ss).length := by
                  simpa [List.length_cons] using Nat.succ_lt_succ hi
                simpa [TList.size_toAnyArray] using this
              have hiTail :
                  i < (TList.toAnyArray (α := α) (ss := ss) seedTail).size := by
                simpa [TList.size_toAnyArray] using hi
              have hidx :
                  (TList.toAnyArray (α := α) (ss := s :: ss) (TList.cons seedHead seedTail))[i +
                    1]'hiFull =
                    (TList.toAnyArray (α := α) (ss := ss) seedTail)[i]'hiTail := by
                have hcons :=
                  (TList.toAnyArray_cons (α := α) (s := s) (ss := ss) seedHead seedTail)
                have hxs : (#[(Runtime.Autograd.AnyTensor.mk seedHead)] : Array (Runtime.AnyTensor
                  α)).size ≤ i + 1 := by
                  simp
                have : (i + 1) - (#[(Runtime.Autograd.AnyTensor.mk seedHead)] : Array
                  (Runtime.AnyTensor α)).size = i := by
                  simp
                simp [hcons]
              have hshape' :
                  ((TList.toAnyArray (α := α) (ss := s :: ss) (TList.cons seedHead seedTail))[i +
                    1]'hiFull).s =
                    ((TList.toAnyArray (α := α) (ss := ss) seedTail)[i]'hiTail).s := by
                simpa using congrArg Runtime.AnyTensor.s hidx
              simpa [hshape'] using hshape

          have htail :=
            ih (pref := pref.push newHeadAny) (seed := seedTail) (contrib := contribTail) hnodesTail

          -- The `foldlM` over the cons list runs one `addGradAll` step, then continues with the
          -- tail.
          -- We need the "push-form" of `hadd0` to rewrite that first step.
          have hadd0Push :
              Runtime.Autograd.Tape.addGradAll (t := t)
                  ((pref.push seedHeadAny) ++ (TList.toAnyArray (α := α) (ss := ss) seedTail ++
                    suffix))
                  pref.size contribHeadAny =
                .ok ((pref.push newHeadAny) ++ (TList.toAnyArray (α := α) (ss := ss) seedTail ++
                  suffix)) := by
            have hadd0' :
                Runtime.Autograd.Tape.addGradAll (t := t)
                    ((pref.push seedHeadAny) ++ TList.toAnyArray (α := α) (ss := ss) seedTail ++
                      suffix)
                    pref.size contribHeadAny =
                  .ok ((pref.push newHeadAny) ++ TList.toAnyArray (α := α) (ss := ss) seedTail ++
                    suffix) := by
              simpa [hacc0] using hadd0
            simpa [Array.append_assoc] using hadd0'

          simpa [TList.toIndexedAnyList, List.foldlM, Bind.bind, Except.bind, Pure.pure,
            Except.pure, hadd0Push, TList.add, TList.toAnyArray_cons, Array.append_assoc,
            seedHeadAny, contribHeadAny, newHeadAny]
            using htail

/--
`Tape.addGradAll` never changes the size of the dense gradient array in the `.ok` case.

This is a structural property needed to show the runtime reverse loop preserves array sizes.
-/
theorem addGradAll_size_preserved {α : Type} [Add α] [DecidableEq Shape]
    (t : Runtime.Autograd.Tape α) (grads : Array (Runtime.AnyTensor α)) (id : Nat) (g :
      Runtime.AnyTensor α) :
    match Runtime.Autograd.Tape.addGradAll (t := t) grads id g with
    | .ok grads' => grads'.size = grads.size
    | .error _ => True := by
  cases hnode : Runtime.Autograd.Tape.getNode? (t := t) id with
  | none =>
      -- Reduce `addGradAll` to its initial `throw`, then simplify the match.
      have hadd :
          Runtime.Autograd.Tape.addGradAll (t := t) grads id g =
            .error "autograd: invalid parent id during backward" := by
        simp [Runtime.Autograd.Tape.addGradAll, hnode, throw, throwThe, MonadExceptOf.throw]
        rfl
      simp [hadd]
  | some node =>
      by_cases hreq : node.requires_grad = false
      · -- No-op case: `pure grads`.
        have hadd : Runtime.Autograd.Tape.addGradAll (t := t) grads id g = .ok grads := by
          simp [Runtime.Autograd.Tape.addGradAll, hnode, hreq]
          rfl
        simp [hadd]
      · by_cases hshape : g.s = node.value.s
        ·
          let g' : Runtime.AnyTensor α := { s := node.value.s, t := Tensor.castShape g.t hshape }
          cases hgrad : grads[id]? with
          | none =>
              simp [Runtime.Autograd.Tape.addGradAll, hnode, hreq, hshape, hgrad,
                throw, throwThe, MonadExceptOf.throw]
          | some existing =>
              by_cases hex : existing.s = node.value.s
              ·
                let existing' : Runtime.AnyTensor α :=
                  { s := node.value.s, t := Tensor.castShape existing.t hex }
                cases hadd : Runtime.Autograd.AnyTensor.add existing' g' with
                | error e =>
                    -- In the `AnyTensor.add = .error e` case, `addGradAll` errors too, so the size
                    -- statement is `True`.
                    have haddAll :
                        Runtime.Autograd.Tape.addGradAll (t := t) grads id g = .error e := by
                      simp [Runtime.Autograd.Tape.addGradAll, hnode, hreq, hshape, hgrad, hex, g',
                        existing', hadd,
                        throw, throwThe, MonadExceptOf.throw]
                      simp [Bind.bind, Except.bind]
                    simp [haddAll]
                | ok summed =>
                    -- `grads[id]? = some existing` implies `id < grads.size`, so `set` is
                    -- in-bounds.
                    rcases
                      (Array.getElem_of_getElem? (xs := grads) (i := id) (a := existing) hgrad) with
                      ⟨hid, hget⟩

                    -- Help `simp` pick the correct shape-check branch:
                    -- `grads[id].s = existing.s = node.value.s`.
                    have hs : grads[id].s = node.value.s := by
                      simpa [hget] using hex

                    -- Now the `do`-block in `addGradAll` reduces all the way to an in-bounds `set`.
                    have haddAll :
                        Runtime.Autograd.Tape.addGradAll (t := t) grads id g =
                          .ok (grads.set id summed (h := hid)) := by
                      -- First reduce the control flow of `addGradAll` down to the final `map/set`
                      -- line.
                      simp [Runtime.Autograd.Tape.addGradAll, hnode, hreq, hshape,
                        hid, hs, throw, throwThe, MonadExceptOf.throw]

                      -- Identify the `AnyTensor.add` call produced by the reduced code with our
                      -- `hadd`.
                      have hadd2 :
                          Runtime.Autograd.AnyTensor.add
                              { s := node.value.s, t := Tensor.castShape grads[id].t hs }
                              { s := node.value.s, t := Tensor.castShape g.t hshape } =
                            Except.ok summed := by
                        -- The first argument is `existing'` up to proof-irrelevant `cast_shape`.
                        have harg1 :
                            ({ s := node.value.s, t := Tensor.castShape grads[id].t hs } :
                              Runtime.AnyTensor α) =
                              existing' := by
                          -- Rewrite `grads[id] = existing`, then use proof irrelevance on the
                          -- shape-cast proof.
                          cases hget
                          simp [existing']
                        simpa [g', harg1] using hadd

                      simp [hadd2]

                    simp [haddAll, Array.size_set]
              ·
                simp [Runtime.Autograd.Tape.addGradAll, hnode, hreq, hshape, hgrad, hex,
                  throw, throwThe, MonadExceptOf.throw]
        ·
          simp [Runtime.Autograd.Tape.addGradAll, hnode, hreq, hshape, throw, throwThe,
            MonadExceptOf.throw]

/-- If `addGradAll` returns `.ok grads'`, then `grads'.size = grads.size`. -/
theorem addGradAll_ok_size {α : Type} [Add α] [DecidableEq Shape]
    (t : Runtime.Autograd.Tape α) :
    ∀ {grads : Array (Runtime.AnyTensor α)} {id : Nat} {g : Runtime.AnyTensor α}
      {grads' : Array (Runtime.AnyTensor α)},
      Runtime.Autograd.Tape.addGradAll (t := t) grads id g = .ok grads' →
        grads'.size = grads.size := by
  intro grads id g grads' h
  simpa [h] using addGradAll_size_preserved (t := t) grads id g

/--
If one step of the runtime dense backward loop succeeds, it preserves the accumulator array size.

This is proved by showing the internal `foldlM addGradAll` preserves size, then splitting on the
control flow of `backwardDenseFromStep`.
-/
theorem backwardDenseFromStep_ok_size {α : Type} [Add α] [DecidableEq Shape]
    (t : Runtime.Autograd.Tape α) :
    ∀ {acc : Array (Runtime.AnyTensor α)} {id : Nat} {acc' : Array (Runtime.AnyTensor α)},
      Runtime.Autograd.Tape.backwardDenseFromStep (t := t) acc id = .ok acc' →
        acc'.size = acc.size := by
  intro acc id acc' h
  -- `foldlM` over `addGradAll` preserves `size` in the `.ok` case.
  have fold_ok_size :
      ∀ (contribs : List (Nat × Runtime.AnyTensor α)) (acc0 accOut : Array (Runtime.AnyTensor α)),
        (contribs.foldlM (fun acc2 (pid, pg) => Runtime.Autograd.Tape.addGradAll (t := t) acc2 pid
          pg) acc0 =
            .ok accOut) →
          accOut.size = acc0.size := by
    intro contribs acc0 accOut hfold
    induction contribs generalizing acc0 accOut with
    | nil =>
        simp [List.foldlM] at hfold
        cases hfold
        rfl
    | cons head tail ih =>
        cases head with
        | mk pid pg =>
            cases h1 : Runtime.Autograd.Tape.addGradAll (t := t) acc0 pid pg with
            | error e =>
                simp [List.foldlM, h1] at hfold
                cases hfold
            | ok acc1 =>
                have htail :
                    tail.foldlM
                        (fun acc2 (pid, pg) => Runtime.Autograd.Tape.addGradAll (t := t) acc2 pid
                          pg) acc1 =
                      .ok accOut := by
                  simpa [List.foldlM, Bind.bind, Except.bind, Pure.pure, Except.pure, h1] using hfold
                have hs1 : acc1.size = acc0.size :=
                  addGradAll_ok_size (t := t) (grads := acc0) (id := pid) (g := pg) (grads' := acc1)
                    (by
                    simpa using h1)
                have := ih (acc0 := acc1) (accOut := accOut) htail
                simpa [hs1] using this

  cases hnode : Runtime.Autograd.Tape.getNode? (t := t) id with
  | none =>
      simp [Runtime.Autograd.Tape.backwardDenseFromStep, hnode] at h
      cases h
  | some node =>
      by_cases hreq : node.requires_grad = false
      · simp [Runtime.Autograd.Tape.backwardDenseFromStep, hnode, hreq] at h
        cases h
        simp
      · cases hgrad : acc[id]? with
        | none =>
            simp [Runtime.Autograd.Tape.backwardDenseFromStep, hnode, hreq, hgrad] at h
            cases h
          | some dLdyAny =>
              by_cases hshape : dLdyAny.s = node.value.s
              ·
                let dLdy : Runtime.AnyTensor α :=
                  { s := node.value.s, t := Tensor.castShape dLdyAny.t hshape }
                cases hback : node.backward dLdy with
                | error e =>
                    simp [Runtime.Autograd.Tape.backwardDenseFromStep, hnode, hreq, hgrad, hshape,
                      dLdy, hback] at h
                    cases h
                | ok contribs =>
                    have hfold :
                        contribs.foldlM
                            (fun acc2 (pid, pg) =>
                              Runtime.Autograd.Tape.addGradAll (t := t) acc2 pid pg)
                            acc =
                          .ok acc' := by
                      simpa
                        [Runtime.Autograd.Tape.backwardDenseFromStep, hnode, hreq, hgrad, hshape,
                          dLdy, hback, Bind.bind, Except.bind, Pure.pure, Except.pure]
                        using h
                    simpa using fold_ok_size contribs acc acc' hfold
              ·
                have : False := by
                  simp
                    [Runtime.Autograd.Tape.backwardDenseFromStep, hnode, hreq, hgrad, hshape]
                    at h
                cases this


end Graph

end Algebra
end Autograd
end Proofs
